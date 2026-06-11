"""
Persistent document store.

Saves the LAST document uploaded via /upload to disk (text + doc_id) so that:

  1. The server can rebuild its in-RAM index on restart WITHOUT the Teacher
     Server having to re-send the document (no extra /evaluate, no re-upload).
  2. You can re-embed the SAME text with OTHER embedding models offline
     (scripts/embed_all_models.py) to pre-fill the vector cache for later
     /evaluate runs with a different model.

Why this matters for the competition (Modified 2 rules):
  * Embedding 100+ chunks on CPU can exceed the Teacher's 120s /upload timeout.
  * On the FIRST /evaluate you send {"document_received": false}; even if the
    Teacher reports a timeout, the text is saved here and embedding finishes in
    the background. After that you call /evaluate with {"document_received":
    true} so the Teacher SKIPS /upload and goes straight to the questions.
  * The saved vector cache + this document store let you restart the server
    (to tweak prompt/search/model) and be ready again in seconds.

Storage: a single JSON file at  <project_root>/cache/last_document.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import config

DOC_STORE_PATH: Path = config.PROJECT_ROOT / "cache" / "last_document.json"


def save_document(doc_id: str, text: str) -> Path:
    """Persist the most recently uploaded document to disk."""
    DOC_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": doc_id or "none",
        "text": text or "",
        "chars": len(text or ""),
        "timestamp": time.time(),
    }
    # Write atomically (temp file -> replace) so a crash mid-write can't corrupt it.
    tmp = DOC_STORE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(DOC_STORE_PATH)
    print(f"[docstore] Saved document doc_id={payload['doc_id']} chars={payload['chars']} -> {DOC_STORE_PATH}")
    return DOC_STORE_PATH


def load_document() -> Optional[Tuple[str, str]]:
    """
    Load the last saved document.

    Returns (doc_id, text) if present and non-empty, else None.
    """
    if not DOC_STORE_PATH.exists():
        return None
    try:
        with open(DOC_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("text") or ""
        if not text.strip():
            return None
        doc_id = data.get("doc_id") or "none"
        ts = data.get("timestamp", 0)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "unknown"
        print(f"[docstore] Loaded saved document doc_id={doc_id} chars={len(text)} (saved {when})")
        return doc_id, text
    except Exception as exc:  # pragma: no cover
        print(f"[docstore] Failed to load {DOC_STORE_PATH.name}: {exc!r}")
        return None


def has_document() -> bool:
    """True if a non-empty document has been saved."""
    return load_document() is not None


def clear_document() -> bool:
    """Delete the saved document. Returns True if a file was removed."""
    if DOC_STORE_PATH.exists():
        DOC_STORE_PATH.unlink()
        print(f"[docstore] Cleared {DOC_STORE_PATH}")
        return True
    return False
