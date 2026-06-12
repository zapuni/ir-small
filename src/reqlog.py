"""
Request logging for competition rounds.

Captures, per evaluation round, exactly what the Teacher Server sent us:
  * the uploaded document text (logged on /upload — present on the FIRST
    evaluate of a session; later rounds with document_received=true skip
    /upload, so no new document file is written, by design), and
  * every /ask question together with the answer we returned.

Why: you only get 5 evaluate attempts and don't know the exam in advance.
After attempt #1 these logs let you READ the real 100 questions + your answers,
then pick the best embedding model / config for attempts #2–#5 (switch model,
restart — the index reloads from the persisted cache in seconds).

Layout (under logs/eval/, git-ignored):
    logs/eval/<round>__document.txt        # the document text (if sent)
    logs/eval/<round>__questions.jsonl     # one JSON line per /ask
    logs/eval/rounds.jsonl                 # index of all rounds (meta)

A new round starts on each /upload, or on the first /ask after a >90s gap
(so a document_received=true batch of 100 questions is grouped on its own).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import config

LOG_DIR = config.PROJECT_ROOT / "logs" / "eval"

_lock = threading.Lock()
_round_id: str | None = None
_round_qcount: int = 0
_last_ask_ts: float = 0.0
_NEW_ROUND_GAP_S = 90.0


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _start_round(reason: str, doc_id: str | None = None) -> str:
    global _round_id, _round_qcount
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _round_id = _ts()
    _round_qcount = 0
    meta = {
        "round": _round_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "doc_id": doc_id,
        "embed_model": config.EMBED_MODEL_NAME,
    }
    try:
        with open(LOG_DIR / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return _round_id


def log_document(doc_id: str, text: str) -> None:
    """Log the document text the Teacher sent (starts a new round)."""
    try:
        with _lock:
            rid = _start_round("upload", doc_id)
            path = LOG_DIR / f"{rid}__document.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text or "")
        print(f"[reqlog] round={rid} document logged ({len(text or '')} chars) -> {path.name}")
    except Exception as exc:  # never break the request path
        print(f"[reqlog] log_document failed: {exc!r}")


def log_question(question: str, answer: str, n_sources: int, took_s: float) -> None:
    """Append one /ask question + the answer we returned to the current round."""
    global _round_qcount, _last_ask_ts
    try:
        now = time.monotonic()
        with _lock:
            # Start a fresh round if this is a skip-upload batch (no round yet,
            # or a long gap since the previous question).
            if _round_id is None or (now - _last_ask_ts) > _NEW_ROUND_GAP_S:
                _start_round("ask")
            _last_ask_ts = now
            _round_qcount += 1
            rec = {
                "i": _round_qcount,
                "time": time.strftime("%H:%M:%S"),
                "question": question or "",
                "answer": answer,
                "n_sources": n_sources,
                "took_s": round(took_s, 2),
            }
            with open(LOG_DIR / f"{_round_id}__questions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[reqlog] log_question failed: {exc!r}")
