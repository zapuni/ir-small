#!/usr/bin/env python3
"""
Download ALL embedding models into the local cache — run during the SETUP
window while Internet is still available (before the LAN-only phase).

After this finishes, every model can be loaded fully offline. Models are stored
in the Hugging Face cache layout under src/models/ (EMBED_CACHE_DIR), so the
embedder finds them without a network connection.

Usage:
    python scripts/download_all_models.py                # use tmp/model-embedding.md
    python scripts/download_all_models.py --models a b c   # explicit list
    python scripts/download_all_models.py --list-file path/to/list.md

Tip: verify offline afterwards with
    set HF_HUB_OFFLINE=1   (cmd)   /   $env:HF_HUB_OFFLINE=1   (PowerShell)
then start the server — it must load without touching the network.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_LIST_FILE = PROJECT_ROOT / "tmp" / "model-embedding.md"

FALLBACK_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert",
    "AITeamVN/Vietnamese_Embedding",
    "thanhtantran/Vietnamese_Embedding_v2",
    "dangvantuan/vietnamese-embedding",
    "intfloat/multilingual-e5-base",
]


def read_model_list(list_file: Path) -> list[str]:
    if not list_file.exists():
        print(f"[download-all] List file not found: {list_file} — using fallback list.")
        return list(FALLBACK_MODELS)
    models: list[str] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        models.append(s)
    return models or list(FALLBACK_MODELS)


def download_one(model_name: str, cache_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print(f"Downloading: {model_name}")
    print("=" * 70)
    result = {"model": model_name, "status": "success"}
    try:
        from sentence_transformers import SentenceTransformer

        # Jina v5 (and a few others) ship custom code -> needs trust_remote_code.
        trust = "jina" in model_name.lower()
        t0 = time.time()
        model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=str(cache_dir),
            trust_remote_code=trust,
        )
        # Tiny encode to force full weight materialisation + sanity check.
        vec = model.encode(["Xin chào, kiểm tra mô hình."])
        dim = vec.shape[1]
        result["dim"] = dim
        result["seconds"] = time.time() - t0
        print(f"[download-all] ✓ OK  dim={dim}  ({result['seconds']:.1f}s)")
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"[download-all] ✗ FAILED: {e!r}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Download all embedding models for offline use")
    ap.add_argument("--models", nargs="*", help="Explicit model list (overrides --list-file)")
    ap.add_argument("--list-file", default=str(DEFAULT_LIST_FILE), help="File with one model per line")
    args = ap.parse_args()

    import config

    cache_dir = config.EMBED_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download-all] Cache directory: {cache_dir}")

    models = args.models if args.models else read_model_list(Path(args.list_file))
    print(f"[download-all] Will download {len(models)} model(s):")
    for m in models:
        print(f"    - {m}")

    results = [download_one(m, cache_dir) for m in models]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<55} {'Status':<10} {'Dim':<8}")
    print("-" * 73)
    failed = 0
    for r in results:
        status = "✓ OK" if r["status"] == "success" else "✗ FAIL"
        if r["status"] != "success":
            failed += 1
        print(f"{r['model']:<55} {status:<10} {str(r.get('dim', '-')):<8}")
    print("=" * 70)
    if failed:
        print(f"\n⚠ {failed} model(s) failed to download. Re-run while Internet is available.\n")
        sys.exit(1)
    print("\nAll models downloaded. You can now go offline.\n")


if __name__ == "__main__":
    main()
