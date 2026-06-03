"""
Local embedding wrapper around keepitreal/vietnamese-sbert.

- Loads from the local MODEL_DIR if present (offline), otherwise falls back to
  the model name on the Hub (online, first run).
- Configures torch to use all CPU cores for max throughput on a laptop.
- Produces L2-normalised float32 vectors so cosine similarity == dot product.
"""
from __future__ import annotations

import os
import threading
from typing import List

import numpy as np

import config

# Make PyTorch use every physical core available on the laptop.
try:
    import torch

    _n = os.cpu_count() or 4
    torch.set_num_threads(_n)
except Exception:  # pragma: no cover - torch always present in practice
    torch = None  # type: ignore


class Embedder:
    """Thread-safe singleton-ish wrapper over a SentenceTransformer."""

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_path = (
            str(config.MODEL_DIR)
            if config.MODEL_DIR.exists()
            else config.EMBED_MODEL_NAME
        )
        source = "local folder" if config.MODEL_DIR.exists() else "Hugging Face Hub"
        print(f"[embedder] Loading model from {source}: {model_path}")

        self.model = SentenceTransformer(model_path, device="cpu")
        self.model.max_seq_length = config.EMBED_MAX_SEQ_LEN
        # API renamed across sentence-transformers versions; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] Ready. dim={self.dim}, max_seq_len={config.EMBED_MAX_SEQ_LEN}")

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Embedder()
        return cls._instance

    def encode(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        """Return an (N, dim) float32 matrix of L2-normalised embeddings."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size or config.EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string -> (dim,) float32 vector."""
        return self.encode([text])[0]


def warm_up() -> Embedder:
    """Load the model and run a tiny encode so the first real request is fast."""
    emb = Embedder.get()
    emb.encode_one("khởi động mô hình")
    return emb
