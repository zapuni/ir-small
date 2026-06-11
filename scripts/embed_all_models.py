#!/usr/bin/env python3
"""
Pre-embed the SAVED document with several embedding models — OFFLINE.

Use this AFTER the first /evaluate (when the Teacher has sent the document via
/upload and your server saved it to disk via docstore). It re-embeds that exact
text with every model you list and writes each result to the persistent vector
cache. Then, for later /evaluate runs, you just:

    1. switch the model in .env   (scripts/switch_model.py --set <model>)
    2. restart the server         (it loads the index from cache in seconds)
    3. compete.py evaluate --document-received

No Internet is required to RUN this (embedding is local CPU) — but the models
themselves must already be downloaded (do that during the setup window with
scripts/download_all_models.py).

Usage:
    python scripts/embed_all_models.py                 # use tmp/model-embedding.md
    python scripts/embed_all_models.py --models a b c   # explicit list
    python scripts/embed_all_models.py --force          # re-embed even if cached
    python scripts/embed_all_models.py --list-file path/to/list.md
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_LIST_FILE = PROJECT_ROOT / "tmp" / "model-embedding.md"

# Fallback list if the list file is missing.
FALLBACK_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert",
    "AITeamVN/Vietnamese_Embedding",
    "thanhtantran/Vietnamese_Embedding_v2",
    "dangvantuan/vietnamese-embedding",
    "intfloat/multilingual-e5-base",
]


def read_model_list(list_file: Path) -> list[str]:
    """Read one model name per line; ignore blanks and comment lines (#)."""
    if not list_file.exists():
        print(f"[embed-all] List file not found: {list_file} — using fallback list.")
        return list(FALLBACK_MODELS)
    models: list[str] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        models.append(s)
    if not models:
        print(f"[embed-all] {list_file} had no models — using fallback list.")
        return list(FALLBACK_MODELS)
    return models


def embed_with_model(model_name: str, doc_id: str, text: str, force: bool) -> dict:
    """Embed the saved document with one model and persist it to the cache."""
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    os.environ["EMBED_MODEL_NAME"] = model_name

    # Reload modules so config picks up the new EMBED_MODEL_NAME and the
    # embedder/retriever rebind to it.
    import config
    import textutils
    import embedder
    import vector_cache
    import retriever

    importlib.reload(config)
    importlib.reload(textutils)
    importlib.reload(embedder)
    importlib.reload(vector_cache)
    importlib.reload(retriever)

    # Reset the embedder singleton so it reloads the new model.
    embedder.Embedder._instance = None
    embedder.Embedder._current_model_name = None

    result = {"model": model_name, "status": "success"}

    # Skip if already cached (unless --force).
    if not force:
        cached = vector_cache.load_cache(model_name=model_name, doc_id=doc_id, text=text)
        if cached is not None:
            chunks, _emb, _meta = cached
            print(f"[embed-all] ✓ Already cached ({len(chunks)} chunks) — skipping. Use --force to re-embed.")
            result["status"] = "cached"
            result["chunks"] = len(chunks)
            return result

    try:
        t0 = time.time()
        n = retriever.INDEX.build(text, doc_id=doc_id, use_cache=True)
        dt = time.time() - t0
        result["chunks"] = n
        result["seconds"] = dt
        print(f"[embed-all] ✓ Done: {n} chunks in {dt:.1f}s (cached).")
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"[embed-all] ✗ FAILED: {e!r}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-embed the saved document with multiple models")
    ap.add_argument("--models", nargs="*", help="Explicit model list (overrides --list-file)")
    ap.add_argument("--list-file", default=str(DEFAULT_LIST_FILE), help="File with one model per line")
    ap.add_argument("--force", action="store_true", help="Re-embed even if a cache already exists")
    args = ap.parse_args()

    # Load the saved document ONCE (before any module reload).
    import config  # noqa: F401  (ensures src on path + .env loaded)
    import docstore

    saved = docstore.load_document()
    if saved is None:
        print("\n[embed-all] No saved document found.")
        print("            Run /upload first (start server + compete.py evaluate),")
        print(f"            or check {docstore.DOC_STORE_PATH}")
        sys.exit(1)
    doc_id, text = saved
    print(f"[embed-all] Document: doc_id={doc_id} chars={len(text)}")

    models = args.models if args.models else read_model_list(Path(args.list_file))
    print(f"[embed-all] Will embed with {len(models)} model(s):")
    for m in models:
        print(f"    - {m}")

    results = [embed_with_model(m, doc_id, text, args.force) for m in models]

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<55} {'Status':<10} {'Chunks':<8}")
    print("-" * 73)
    for r in results:
        status = {"success": "✓ OK", "cached": "• cached", "failed": "✗ FAIL"}.get(r["status"], r["status"])
        print(f"{r['model']:<55} {status:<10} {str(r.get('chunks', '-')):<8}")
    print("=" * 70)
    print("\nNext: switch_model.py --set <model>  ->  restart server  ->  evaluate --document-received\n")


if __name__ == "__main__":
    main()
