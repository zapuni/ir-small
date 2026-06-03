"""
LLM answering layer.

Builds a tight RAG prompt, calls an OpenAI-compatible endpoint, and parses out
a single A/B/C/D letter.

The endpoint + key + model come from config.resolve_llm(), which selects either
the temporary generic endpoint (LLM_BASE_URL in .env) or the Teacher proxy
(STUDENT_ID as the key) depending on configuration. No code change is needed to
switch between them on competition day — just edit .env.

Robustness features:
  * Low temperature -> deterministic answers.
  * Reasoning-model aware: if `content` is empty, parse `reasoning_content`.
  * Retries on transient errors, deadline-aware so /ask stays within budget.
  * Offline fallback (in fallback_answer) so we ALWAYS return a valid letter.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

import config
from embedder import Embedder
from textutils import MCQuestion, extract_answer_letter

_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi trắc nghiệm dựa HOÀN TOÀN vào ngữ cảnh được cung cấp. "
    "Chỉ dựa vào ngữ cảnh để chọn đáp án đúng nhất. "
    "Sau khi suy luận, BẮT BUỘC kết thúc bằng một dòng cuối cùng có dạng "
    "'Đáp án: X' với X là một trong các ký tự A, B, C hoặc D."
)


def _build_user_prompt(question: MCQuestion, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"[Đoạn {i + 1}] {c}" for i, c in enumerate(contexts))
    if question.has_options:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(question.options.items()))
        return (
            f"NGỮ CẢNH:\n{ctx}\n\n"
            f"CÂU HỎI:\n{question.stem}\n\n"
            f"CÁC LỰA CHỌN:\n{opts}\n\n"
            f"Dựa vào ngữ cảnh, hãy chọn đáp án đúng nhất. "
            f"Chỉ trả lời bằng một ký tự: {', '.join(question.option_letters())}."
        )
    return (
        f"NGỮ CẢNH:\n{ctx}\n\n"
        f"CÂU HỎI:\n{question.raw}\n\n"
        f"Chỉ trả lời bằng một ký tự A, B, C hoặc D."
    )


def _extract_from_message(msg, valid: List[str]) -> str | None:
    """Pull a letter from an OpenAI-style message, preferring final content."""
    content = (getattr(msg, "content", None) or "").strip()
    letter = extract_answer_letter(content, valid)
    if letter:
        return letter
    # Reasoning models (e.g. mimo) may leave content empty and put the chain of
    # thought — which usually ends with the choice — in reasoning_content.
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    return extract_answer_letter(reasoning, valid)


class LLMClient:
    def __init__(self) -> None:
        from openai import OpenAI

        base_url, api_key, model = config.resolve_llm()
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-none",
            timeout=config.LLM_TIMEOUT,
            max_retries=0,  # we handle retries ourselves
        )
        print(f"[llm] client ready -> base_url={base_url} model={model}")

    def answer(
        self,
        question: MCQuestion,
        contexts: List[str],
        deadline: float | None = None,
    ) -> str | None:
        """
        Ask the LLM. Returns a single letter, or None on failure.

        `deadline` is an absolute time.monotonic() value; we stop retrying once
        it passes so the caller stays within its per-question time budget.
        """
        valid = question.option_letters() if question.has_options else ["A", "B", "C", "D"]
        user_prompt = _build_user_prompt(question, contexts)

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
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
                letter = _extract_from_message(resp.choices[0].message, valid)
                if letter:
                    return letter
            except Exception as exc:  # network/proxy/timeout
                last_err = exc
        if last_err:
            print(f"[llm] call failed: {last_err!r}")
        return None


def fallback_answer(question: MCQuestion, contexts: List[str]) -> str:
    """
    Embedding-similarity fallback used when the LLM is unavailable.

    For each option we score how well "<stem> <option>" matches the retrieved
    context, and return the best letter. Guarantees a valid A/B/C/D.
    """
    valid = question.option_letters() if question.has_options else ["A", "B", "C", "D"]

    if not question.has_options or not contexts:
        return valid[0]

    emb = Embedder.get()
    ctx_vecs = emb.encode(contexts)                       # (C, d)
    ctx_centroid = ctx_vecs.mean(axis=0)
    ctx_centroid /= (np.linalg.norm(ctx_centroid) + 1e-9)

    letters = question.option_letters()
    candidate_texts = [f"{question.stem} {question.options[l]}" for l in letters]
    cand_vecs = emb.encode(candidate_texts)               # (N, d)

    # combine: similarity to context centroid + max similarity to any chunk
    centroid_sim = cand_vecs @ ctx_centroid
    max_chunk_sim = (cand_vecs @ ctx_vecs.T).max(axis=1)
    scores = 0.5 * centroid_sim + 0.5 * max_chunk_sim

    best = int(np.argmax(scores))
    return letters[best]


# Lazy singleton
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
