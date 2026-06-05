"""
Multi-model embedding wrapper for Vietnamese and multilingual models.

Supports:
  - Vietnamese-specific models (keepitreal, AITeamVN, dangvantuan, etc.)
  - Multilingual models (Jina AI, E5, etc.)
  - Auto-detection of model-specific configurations
  - Per-model caching in separate folders
  - Automatic prefix handling for models that require it (E5, Jina v5)

Model-specific configurations:
  - E5 models: require "query: " and "passage: " prefixes
  - Jina embeddings v5: support task-specific adapters
  - BGE-M3 based models: standard sentence-transformers usage
  - PhoBERT models: Vietnamese-optimized

- Loads from local cache if present (offline), otherwise downloads from Hub.
- Configures torch to use all CPU cores for max throughput.
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


# Model-specific configuration registry
MODEL_CONFIGS = {
    # E5 models require prefixes
    "intfloat/multilingual-e5": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
    },
    # Jina v5 models support task adapters (LoRA-based)
    "jinaai/jina-embeddings-v5": {
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 8192,  # v5 supports very long context
        "trust_remote_code": True,
        "use_task_type": True,  # Use task_type parameter instead of prefix
        "task_type": "retrieval",  # Main task type for Jina v5
    },
    # BGE-M3 based Vietnamese models (AITeamVN, thanhtantran)
    "vietnamese_embedding": {
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 512,
    },
    # PhoBERT-based models (dangvantuan)
    "vietnamese-embedding": {
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 256,
    },
    # Default keepitreal/vietnamese-sbert
    "vietnamese-sbert": {
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 256,
    },
}


def _detect_model_config(model_name: str) -> dict:
    """Auto-detect model-specific configuration based on model name."""
    model_lower = model_name.lower()
    
    # Check each registered pattern
    for pattern, cfg in MODEL_CONFIGS.items():
        if pattern in model_lower:
            return cfg.copy()
    
    # Default config for unknown models
    return {
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 512,
        "trust_remote_code": False,
        "use_task_type": False,
    }


class Embedder:
    """Thread-safe singleton wrapper over a SentenceTransformer with multi-model support."""

    _instance: "Embedder | None" = None
    _lock = threading.Lock()
    _current_model_name: str | None = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        # Ensure cache directory exists
        config.EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Detect model-specific configuration
        self.model_config = _detect_model_config(config.EMBED_MODEL_NAME)
        
        # Determine model path (local cache or HF Hub)
        model_path = (
            str(config.MODEL_DIR)
            if config.MODEL_DIR.exists()
            else config.EMBED_MODEL_NAME
        )
        source = "local cache" if config.MODEL_DIR.exists() else "Hugging Face Hub"
        print(f"[embedder] Loading model from {source}: {model_path}")
        print(f"[embedder] Cache directory: {config.MODEL_DIR}")

        # Load model with auto-detection of special parameters
        load_kwargs = {
            "device": "cpu",
            "cache_folder": str(config.EMBED_CACHE_DIR),
        }
        if self.model_config.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(model_path, **load_kwargs)
        
        # Set max sequence length (override config if model-specific)
        max_seq_len = min(
            config.EMBED_MAX_SEQ_LEN,
            self.model_config.get("max_seq_length", 512)
        )
        self.model.max_seq_length = max_seq_len
        
        # Get embedding dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()
        
        print(f"[embedder] Ready. model={config.EMBED_MODEL_NAME}")
        print(f"[embedder] dim={self.dim}, max_seq_len={max_seq_len}")
        if self.model_config.get("use_task_type"):
            print(f"[embedder] Using task type: '{self.model_config.get('task_type', 'retrieval')}'")
        elif self.model_config["query_prefix"]:
            print(f"[embedder] Using prefixes: query='{self.model_config['query_prefix']}' "
                  f"passage='{self.model_config['passage_prefix']}'")

    @classmethod
    def get(cls) -> "Embedder":
        """Get or create the singleton Embedder instance."""
        # Reinitialize if model name changed
        if cls._current_model_name != config.EMBED_MODEL_NAME:
            with cls._lock:
                if cls._current_model_name != config.EMBED_MODEL_NAME:
                    cls._instance = None
                    cls._current_model_name = config.EMBED_MODEL_NAME
        
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Embedder()
        return cls._instance

    def _add_prefix(self, texts: List[str], prefix: str) -> List[str]:
        """Add prefix to texts if model requires it."""
        if not prefix:
            return texts
        return [prefix + text for text in texts]

    def encode(
        self,
        texts: List[str],
        batch_size: int | None = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Return an (N, dim) float32 matrix of L2-normalised embeddings.
        
        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding (default from config)
            is_query: If True, use query settings; if False, use passage settings
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        
        # Prepare encode kwargs
        encode_kwargs = {
            "batch_size": batch_size or config.EMBED_BATCH_SIZE,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        
        # Handle model-specific encoding
        if self.model_config.get("use_task_type"):
            # Jina v5: Use task parameter (same for both query and passage)
            task = self.model_config.get("task_type", "retrieval")
            encode_kwargs["task"] = task
            vecs = self.model.encode(texts, **encode_kwargs)
        else:
            # Standard models: Use prefix
            prefix = (
                self.model_config["query_prefix"] if is_query
                else self.model_config["passage_prefix"]
            )
            texts_with_prefix = self._add_prefix(texts, prefix)
            vecs = self.model.encode(texts_with_prefix, **encode_kwargs)
        
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode a single string -> (dim,) float32 vector."""
        return self.encode([text], is_query=is_query)[0]


def warm_up() -> Embedder:
    """Load the model and run a tiny encode so the first real request is fast."""
    emb = Embedder.get()
    emb.encode_one("khởi động mô hình", is_query=True)
    return emb
