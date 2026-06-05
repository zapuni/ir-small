"""
Persistent vector cache for embeddings.

Caches embeddings per (model, doc_id) so switching models doesn't require re-embedding.
Cache structure:
    cache/
    ├── keepitreal_vietnamese-sbert/
    │   └── doc_123.npz  (chunks, embeddings, metadata)
    ├── AITeamVN_Vietnamese_Embedding/
    │   └── doc_123.npz
    └── ...

Each .npz file contains:
    - chunks: array of text chunks
    - embeddings: numpy array (N, dim)
    - metadata: dict with doc_id, timestamp, config
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


def _get_cache_dir() -> Path:
    """Get base cache directory for all models."""
    cache_dir = config.PROJECT_ROOT / "cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_model_cache_dir(model_name: str) -> Path:
    """Get cache directory for a specific model."""
    # Normalize model name to safe directory name
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    model_dir = _get_cache_dir() / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_cache_key(doc_id: str, text: str) -> str:
    """
    Generate cache key from doc_id and text hash.
    Uses text hash to detect if document content changed.
    """
    if doc_id and doc_id != "none":
        # Use doc_id + text hash for verification
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{doc_id}_{text_hash}"
    else:
        # No doc_id, use full text hash
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_cache_path(model_name: str, doc_id: str, text: str) -> Path:
    """Get cache file path for a document."""
    model_dir = _get_model_cache_dir(model_name)
    cache_key = _get_cache_key(doc_id, text)
    return model_dir / f"{cache_key}.npz"


def save_cache(
    model_name: str,
    doc_id: str,
    text: str,
    chunks: List[str],
    embeddings: np.ndarray,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save chunks and embeddings to cache.
    
    Args:
        model_name: Embedding model name
        doc_id: Document ID
        text: Original text (for hash verification)
        chunks: List of text chunks
        embeddings: Numpy array of embeddings (N, dim)
        metadata: Optional metadata dict
    
    Returns:
        Path to saved cache file
    """
    cache_path = _get_cache_path(model_name, doc_id, text)
    
    # Prepare metadata
    meta = metadata or {}
    meta.update({
        "model_name": model_name,
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0],
        "timestamp": time.time(),
        "chunk_size": config.CHUNK_SIZE_WORDS,
        "chunk_overlap": config.CHUNK_OVERLAP_WORDS,
        "adaptive": config.CHUNK_ADAPTIVE,
    })
    
    # Save to .npz (compressed numpy format)
    np.savez_compressed(
        cache_path,
        chunks=np.array(chunks, dtype=object),
        embeddings=embeddings,
        metadata=np.array([json.dumps(meta)], dtype=object),
    )
    
    print(f"[cache] Saved to {cache_path.name} ({embeddings.shape[0]} chunks, {embeddings.shape[1]} dim)")
    return cache_path


def load_cache(
    model_name: str,
    doc_id: str,
    text: str,
) -> Optional[Tuple[List[str], np.ndarray, Dict]]:
    """
    Load chunks and embeddings from cache.
    
    Args:
        model_name: Embedding model name
        doc_id: Document ID
        text: Original text (for hash verification)
    
    Returns:
        Tuple of (chunks, embeddings, metadata) if cache exists and valid, else None
    """
    cache_path = _get_cache_path(model_name, doc_id, text)
    
    if not cache_path.exists():
        return None
    
    try:
        # Load from .npz
        data = np.load(cache_path, allow_pickle=True)
        
        chunks = data["chunks"].tolist()
        embeddings = data["embeddings"]
        metadata = json.loads(data["metadata"][0])
        
        # Validate
        if len(chunks) != embeddings.shape[0]:
            print(f"[cache] Invalid cache (chunks mismatch): {cache_path.name}")
            return None
        
        print(f"[cache] Loaded from {cache_path.name} ({len(chunks)} chunks, {embeddings.shape[1]} dim)")
        print(f"[cache] Original timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        
        return chunks, embeddings, metadata
        
    except Exception as e:
        print(f"[cache] Failed to load {cache_path.name}: {e!r}")
        return None


def clear_cache(model_name: Optional[str] = None, doc_id: Optional[str] = None) -> int:
    """
    Clear cache files.
    
    Args:
        model_name: If specified, only clear this model's cache
        doc_id: If specified, only clear caches for this doc_id
    
    Returns:
        Number of files deleted
    """
    if model_name:
        model_dir = _get_model_cache_dir(model_name)
        if doc_id:
            # Clear specific doc for specific model
            pattern = f"{doc_id}_*.npz"
            files = list(model_dir.glob(pattern))
        else:
            # Clear all for specific model
            files = list(model_dir.glob("*.npz"))
    else:
        # Clear all models
        cache_dir = _get_cache_dir()
        if doc_id:
            # Clear specific doc across all models
            files = list(cache_dir.glob(f"*/{doc_id}_*.npz"))
        else:
            # Clear everything
            files = list(cache_dir.glob("*/*.npz"))
    
    count = 0
    for f in files:
        try:
            f.unlink()
            count += 1
        except Exception as e:
            print(f"[cache] Failed to delete {f.name}: {e!r}")
    
    print(f"[cache] Cleared {count} cache file(s)")
    return count


def list_cache(model_name: Optional[str] = None) -> List[Dict]:
    """
    List all cache files.
    
    Args:
        model_name: If specified, only list this model's cache
    
    Returns:
        List of dicts with cache info
    """
    if model_name:
        model_dir = _get_model_cache_dir(model_name)
        files = list(model_dir.glob("*.npz"))
    else:
        cache_dir = _get_cache_dir()
        files = list(cache_dir.glob("*/*.npz"))
    
    cache_list = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            metadata = json.loads(data["metadata"][0])
            
            cache_list.append({
                "file": f.name,
                "model": f.parent.name,
                "doc_id": metadata.get("doc_id", "unknown"),
                "chunks": metadata.get("num_chunks", 0),
                "dim": metadata.get("embedding_dim", 0),
                "timestamp": metadata.get("timestamp", 0),
                "size_mb": f.stat().st_size / (1024 * 1024),
            })
        except Exception as e:
            print(f"[cache] Failed to read {f.name}: {e!r}")
    
    return sorted(cache_list, key=lambda x: x["timestamp"], reverse=True)


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache_dir = _get_cache_dir()
    
    if not cache_dir.exists():
        return {"total_files": 0, "total_size_mb": 0, "models": {}}
    
    total_size = 0
    models = {}
    
    for model_dir in cache_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        files = list(model_dir.glob("*.npz"))
        model_size = sum(f.stat().st_size for f in files)
        total_size += model_size
        
        models[model_dir.name] = {
            "files": len(files),
            "size_mb": model_size / (1024 * 1024),
        }
    
    return {
        "total_files": sum(m["files"] for m in models.values()),
        "total_size_mb": total_size / (1024 * 1024),
        "models": models,
    }
