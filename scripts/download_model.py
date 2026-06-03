"""
Download the Vietnamese SBERT embedding model ONCE while you still have
Internet, and save it into the local folder (src/models/vietnamese-sbert).
On competition day (no Internet) the server loads the model from disk.

Run:  python scripts/download_model.py
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from config import EMBED_MODEL_NAME, MODEL_DIR


def main() -> None:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"[download] Model already present at {MODEL_DIR} — nothing to do.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Loading '{EMBED_MODEL_NAME}' from the Hugging Face Hub...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[download] sentence-transformers is not installed yet.")
        print("           Run:  pip install -r requirements.txt")
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    print(f"[download] Saving model to: {MODEL_DIR}")
    model.save(str(MODEL_DIR))

    vec = model.encode(["Xin chào, đây là bài kiểm tra mô hình."])
    print(f"[download] OK. Embedding dimension = {vec.shape[1]}")
    print("[download] Done. The model is now available offline.")


if __name__ == "__main__":
    main()
