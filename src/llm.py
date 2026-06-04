"""
LLM answering layer.

Builds a RAG prompt tuned to the active model profile (Qwen small vs reasoning),
calls an OpenAI-compatible endpoint, and parses a single A/B/C/D letter.

Small exam models (Qwen3/3.5 4B–8B):
  * Disable thinking via chat_template_kwargs (saves ~20s and avoids empty content).
  * Short XML context + one-line answer format.
  * Low max_tokens; temperature 0.

Robustness:
  * Parses content, reasoning, and reasoning_content.
  * JSON {"answer":"B"} and Vietnamese "Đáp án: B" patterns.
  * Embedding fallback when LLM fails or times out.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

import numpy as np

import config
from embedder import Embedder
from textutils import MCQuestion, extract_answer_letter

# --------------------------------------------------------------------------- #
# Prompt templates (profile-specific)                                          #
# --------------------------------------------------------------------------- #
_PROMPTS: Dict[str, tuple[str, str]] = {
    # Qwen3/3.5 small: minimal instructions, strict final line (no chain-of-thought).
    "qwen_small": (
        "Bạn trả lời trắc nghiệm tiếng Việt. CHỈ dựa vào <context>. "
        "Không suy luận dài. Kết thúc bằng đúng một dòng: Đáp án: X "
        "(X là A, B, C hoặc D).",
        "",  # user template built in _build_user_prompt
    ),
    "reasoning": (
        "Bạn là trợ lý trả lời câu hỏi trắc nghiệm dựa HOÀN TOÀN vào ngữ cảnh được cung cấp. "
        "Chỉ dựa vào ngữ cảnh để chọn đáp án đúng nhất. "
        "Sau khi suy luận, BẮT BUỘC kết thúc bằng một dòng cuối cùng có dạng "
        "'Đáp án: X' với X là một trong các ký tự A, B, C hoặc D.",
        "",
    ),
    "default": (
        "Bạn trả lời trắc nghiệm dựa vào <context>. "
        "Kết thúc bằng đúng một dòng: Đáp án: X (X là A, B, C hoặc D).",
        "",
    ),
}


def trim_contexts(contexts: List[str]) -> List[str]:
    """Fit retrieved chunks into the proxy context budget."""
    max_total = config.CONTEXT_MAX_CHARS
    max_chunk = config.CONTEXT_MAX_CHUNK_CHARS
    out: List[str] = []
    used = 0
    for raw in contexts:
        c = (raw or "").strip()
        if not c:
            continue
        if len(c) > max_chunk:
            c = c[: max_chunk - 3].rstrip() + "..."
        if used + len(c) > max_total:
            remain = max_total - used
            if remain < 80:
                break
            c = c[: remain - 3].rstrip() + "..."
        out.append(c)
        used += len(c) + 2
        if used >= max_total:
            break
    return out


def _build_user_prompt(
    profile: str,
    question: MCQuestion,
    contexts: List[str],
) -> str:
    ctx = trim_contexts(contexts)
    ctx_block = "\n".join(f"- {c}" for c in ctx) if ctx else "(không có ngữ cảnh)"

    if question.has_options:
        opts = "\n".join(
            f"{k}. {v}" for k, v in sorted(question.options.items())
        )
        letters = ", ".join(question.option_letters())
        if profile == "qwen_small":
            return (
                f"<context>\n{ctx_block}\n</context>\n\n"
                f"Câu hỏi: {question.stem}\n"
                f"Lựa chọn:\n{opts}\n\n"
                f"Chọn đáp án đúng nhất ({letters}). "
                f"Trả lời một dòng: Đáp án: <chữ cái>"
            )
        return (
            f"NGỮ CẢNH:\n{ctx_block}\n\n"
            f"CÂU HỎI:\n{question.stem}\n\n"
            f"CÁC LỰA CHỌN:\n{opts}\n\n"
            f"Dựa vào ngữ cảnh, chọn đáp án đúng nhất ({letters}). "
            f"Kết thúc bằng: Đáp án: <chữ cái>"
        )

    if profile == "qwen_small":
        return (
            f"<context>\n{ctx_block}\n</context>\n\n"
            f"Câu hỏi: {question.raw}\n"
            f"Trả lời một dòng: Đáp án: A, B, C hoặc D"
        )
    return (
        f"NGỮ CẢNH:\n{ctx_block}\n\n"
        f"CÂU HỎI:\n{question.raw}\n\n"
        f"Kết thúc bằng: Đáp án: <chữ cái A/B/C/D>"
    )


def _extract_json_answer(text: str, valid: List[str]) -> str | None:
    """Parse {"answer":"B"} style outputs (recommended for small Qwen MCQ)."""
    if not text:
        return None
    m = re.search(r'\{\s*"answer"\s*:\s*"([ABCD])"\s*\}', text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            ans = str(obj.get("answer", "")).strip().upper()
            if ans in valid:
                return ans
    except Exception:
        pass
    return None


def _extract_from_message(msg, valid: List[str]) -> str | None:
    """Pull a letter from an OpenAI-style message."""
    content = (getattr(msg, "content", None) or "").strip()
    letter = _extract_json_answer(content, valid) or extract_answer_letter(content, valid)
    if letter:
        return letter

    for field in ("reasoning_content", "reasoning"):
        reasoning = (getattr(msg, field, None) or "").strip()
        letter = _extract_json_answer(reasoning, valid) or extract_answer_letter(
            reasoning, valid
        )
        if letter:
            return letter
    return None


def _completion_kwargs(profile: str, model: str) -> Dict[str, Any]:
    """Extra API kwargs (Qwen thinking toggle, token budget)."""
    kwargs: Dict[str, Any] = {}
    thinking = config.resolve_enable_thinking(model)
    if profile == "qwen_small" or "qwen" in model.lower():
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if not thinking:
            kwargs["max_tokens"] = min(config.LLM_MAX_TOKENS, 96)
    elif profile == "reasoning":
        kwargs["max_tokens"] = config.LLM_MAX_TOKENS
    return kwargs


class LLMClient:
    def __init__(self) -> None:
        from openai import OpenAI

        base_url, api_key, model = config.resolve_llm()
        self.model = model
        self.profile = config.resolve_llm_profile(model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-none",
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )
        sys_prompt = _PROMPTS.get(self.profile, _PROMPTS["default"])[0]
        print(
            f"[llm] ready -> {base_url} model={model} "
            f"profile={self.profile} thinking={config.resolve_enable_thinking(model)}"
        )
        self._system_prompt = sys_prompt

    def answer(
        self,
        question: MCQuestion,
        contexts: List[str],
        deadline: float | None = None,
    ) -> str | None:
        valid = question.option_letters() if question.has_options else ["A", "B", "C", "D"]
        user_prompt = _build_user_prompt(self.profile, question, contexts)

        extra = _completion_kwargs(self.profile, self.model)
        max_tokens = extra.pop("max_tokens", config.LLM_MAX_TOKENS)

        last_err: Exception | None = None
        for _attempt in range(config.LLM_MAX_RETRIES + 1):
            if deadline is not None and time.monotonic() >= deadline:
                break
            timeout = config.LLM_TIMEOUT
            if deadline is not None:
                timeout = max(2.0, min(timeout, deadline - time.monotonic()))
            try:
                resp = self.client.with_options(timeout=timeout).chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=max_tokens,
                    **extra,
                )
                letter = _extract_from_message(resp.choices[0].message, valid)
                if letter:
                    return letter
            except Exception as exc:
                last_err = exc
        if last_err:
            print(f"[llm] call failed: {last_err!r}")
        return None


def fallback_answer(question: MCQuestion, contexts: List[str]) -> str:
    """
    Embedding-similarity fallback when the LLM is unavailable.

    Scores each option against retrieved context (centroid + max chunk sim).
    """
    valid = question.option_letters() if question.has_options else ["A", "B", "C", "D"]

    if not question.has_options or not contexts:
        return valid[0]

    emb = Embedder.get()
    ctx_vecs = emb.encode(contexts)
    ctx_centroid = ctx_vecs.mean(axis=0)
    ctx_centroid /= np.linalg.norm(ctx_centroid) + 1e-9

    letters = question.option_letters()
    candidate_texts = [f"{question.stem} {question.options[l]}" for l in letters]
    cand_vecs = emb.encode(candidate_texts)

    centroid_sim = cand_vecs @ ctx_centroid
    max_chunk_sim = (cand_vecs @ ctx_vecs.T).max(axis=1)
    scores = 0.55 * centroid_sim + 0.45 * max_chunk_sim

    best = int(np.argmax(scores))
    return letters[best]


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
