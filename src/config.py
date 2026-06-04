"""
Central configuration for the Offline RAG Competition system.

Everything is driven by environment variables, loaded from the project-root
`.env` file (the folder ABOVE `src/`). This lets you swap the LLM endpoint,
student ID, and tuning knobs without touching code — ideal for switching from
the temporary LLM to the real Teacher proxy on competition day.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles default to cp1252, which cannot encode Vietnamese text and
# would crash on print(). Force UTF-8 so logging Vietnamese chunks/questions is
# always safe, regardless of how the process was launched (python / uv run).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

# src/ -> project root is one level up. The .env lives at the project root.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load .env from the project root (…/ir/.env). override=False keeps any real
# environment variables you export at the shell taking precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)
# Optional local overrides (e.g. Qwen test endpoint) — not committed.
load_dotenv(PROJECT_ROOT / ".env.local", override=False)


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


# --------------------------------------------------------------------------- #
# Identity                                                                     #
# --------------------------------------------------------------------------- #
# Your student ID (UPPERCASE). Used as the X-Student-ID header when talking to
# the Teacher Server, and as the proxy API key when PROXY mode is active.
STUDENT_ID: str = _get("STUDENT_ID", "B22DCAT102").upper()

# --------------------------------------------------------------------------- #
# LLM configuration                                                            #
# --------------------------------------------------------------------------- #
# Two ways to point at an LLM:
#
#   1) Temporary / generic OpenAI-compatible endpoint (current .env):
#        LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
#
#   2) Competition Teacher proxy (no separate key — the API key IS your MSSV):
#        PROXY_BASE_URL, LLM_MODEL  (+ STUDENT_ID as the key)
#
# Precedence: if LLM_BASE_URL is set we use mode (1); otherwise we use the
# Teacher proxy (2). So on competition day you either clear LLM_BASE_URL or set
# LLM_USE_PROXY=true to switch over.
LLM_USE_PROXY: bool = _get("LLM_USE_PROXY", "false").lower() in ("1", "true", "yes")

# Mode (1): generic endpoint
LLM_BASE_URL: str = _get("LLM_BASE_URL", "")
LLM_API_KEY: str = _get("LLM_API_KEY", "")
LLM_MODEL_NAME: str = _get("LLM_MODEL_NAME", "gpt-4o-mini")

# Mode (2): Teacher proxy (from the competition spec)
PROXY_BASE_URL: str = _get("PROXY_BASE_URL", "http://192.168.50.218:8000/api/v1/proxy")
PROXY_MODEL: str = _get("LLM_MODEL", "gpt-4o-mini")


def resolve_llm() -> tuple[str, str, str]:
    """
    Return (base_url, api_key, model) for the active LLM, based on the rules
    above. Centralised so app + client always agree.
    """
    use_proxy = LLM_USE_PROXY or not LLM_BASE_URL
    if use_proxy:
        return PROXY_BASE_URL, STUDENT_ID, PROXY_MODEL
    return LLM_BASE_URL, (LLM_API_KEY or STUDENT_ID), LLM_MODEL_NAME


LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(_get("LLM_MAX_TOKENS", "512"))
LLM_TIMEOUT: float = float(_get("LLM_TIMEOUT", "30"))      # seconds per attempt
LLM_MAX_RETRIES: int = int(_get("LLM_MAX_RETRIES", "1"))

# Prompt / generation profile for small exam LLMs (Qwen3/3.5 4B–8B).
#   auto  -> detect from model name (qwen -> qwen_small)
#   qwen_small | reasoning | default
LLM_PROFILE: str = _get("LLM_PROFILE", "auto").lower()
# Qwen3 "thinking" burns most of the 60s budget; disable for MCQ (vLLM/SGLang).
# Empty = auto (off for qwen_small, on for reasoning profile).
_think_env = _get("LLM_ENABLE_THINKING", "").lower()
if _think_env in ("0", "false", "no", "off"):
    LLM_ENABLE_THINKING: bool | None = False
elif _think_env in ("1", "true", "yes", "on"):
    LLM_ENABLE_THINKING = True
else:
    LLM_ENABLE_THINKING = None

# Cap retrieved context so the prompt fits proxy seq len (2048–4096 tokens).
CONTEXT_MAX_CHARS: int = int(_get("CONTEXT_MAX_CHARS", "6000"))
CONTEXT_MAX_CHUNK_CHARS: int = int(_get("CONTEXT_MAX_CHUNK_CHARS", "900"))

# --------------------------------------------------------------------------- #
# Teacher server (competition control plane)                                   #
# --------------------------------------------------------------------------- #
TEACHER_BASE_URL: str = _get("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1")

# --------------------------------------------------------------------------- #
# Embedding model (downloaded once, then loaded locally on CPU)                #
# --------------------------------------------------------------------------- #
EMBED_MODEL_NAME: str = _get("EMBED_MODEL_NAME", "keepitreal/vietnamese-sbert")
MODEL_DIR: Path = Path(_get("MODEL_DIR", str(SRC_DIR / "models" / "vietnamese-sbert")))
EMBED_MAX_SEQ_LEN: int = int(_get("EMBED_MAX_SEQ_LEN", "256"))
EMBED_BATCH_SIZE: int = int(_get("EMBED_BATCH_SIZE", "32"))

# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #
CHUNK_SIZE_WORDS: int = int(_get("CHUNK_SIZE_WORDS", "180"))
CHUNK_OVERLAP_WORDS: int = int(_get("CHUNK_OVERLAP_WORDS", "40"))

# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #
TOP_K_DENSE: int = int(_get("TOP_K_DENSE", "12"))
TOP_K_BM25: int = int(_get("TOP_K_BM25", "12"))
TOP_K_CONTEXT: int = int(_get("TOP_K_CONTEXT", "5"))
RRF_K: int = int(_get("RRF_K", "60"))
MMR_LAMBDA: float = float(_get("MMR_LAMBDA", "0.3"))

# --------------------------------------------------------------------------- #
# Server                                                                       #
# --------------------------------------------------------------------------- #
HOST: str = _get("HOST", "0.0.0.0")
PORT: int = int(_get("PORT", "5000"))

# Hard wall-clock budget (seconds) for the whole /ask handler. We MUST return
# before the competition's 60s/question limit; the fallback fires if the LLM
# hasn't answered by this deadline. Keep a safety margin below 60s.
ASK_DEADLINE: float = float(_get("ASK_DEADLINE", "45"))


def resolve_llm_profile(model: str) -> str:
    """Return active prompt profile: qwen_small, reasoning, or default."""
    if LLM_PROFILE not in ("", "auto"):
        return LLM_PROFILE
    m = model.lower()
    if "qwen" in m:
        return "qwen_small"
    if any(x in m for x in ("mimo", "o1", "reason", "r1")):
        return "reasoning"
    return "default"


def resolve_enable_thinking(model: str) -> bool:
    """Whether to allow Qwen-style thinking blocks (slow; off for exam MCQ)."""
    if LLM_ENABLE_THINKING is not None:
        return bool(LLM_ENABLE_THINKING)
    return resolve_llm_profile(model) == "reasoning"


def describe() -> str:
    base, key, model = resolve_llm()
    masked = (key[:4] + "…") if key else "(none)"
    mode = "PROXY (Teacher)" if (LLM_USE_PROXY or not LLM_BASE_URL) else "GENERIC (.env)"
    prof = resolve_llm_profile(model)
    think = resolve_enable_thinking(model)
    return (
        f"LLM mode={mode} | base_url={base} | model={model} | key={masked}\n"
        f"profile={prof} | thinking={think} | max_tokens={LLM_MAX_TOKENS}\n"
        f"embed={EMBED_MODEL_NAME} | chunk={CHUNK_SIZE_WORDS}/{CHUNK_OVERLAP_WORDS} "
        f"| top_k_ctx={TOP_K_CONTEXT} | ctx_max_chars={CONTEXT_MAX_CHARS} "
        f"| ask_deadline={ASK_DEADLINE}s"
    )
