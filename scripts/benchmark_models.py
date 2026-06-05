#!/usr/bin/env python3
"""
Benchmark script to compare different embedding models.

Usage:
    python scripts/benchmark_models.py

Tests multiple embedding models and compares:
    - Loading time
    - Encoding speed
    - Memory usage
    - Embedding dimension
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test texts in Vietnamese
TEST_TEXTS = [
    "Trí tuệ nhân tạo là gì?",
    "Machine learning và deep learning có gì khác nhau?",
    "Python là ngôn ngữ lập trình phổ biến cho data science.",
    "Hệ thống tìm kiếm thông tin sử dụng embedding vectors.",
    "RAG (Retrieval Augmented Generation) kết hợp retrieval và generation.",
]

# Models to test
MODELS_TO_TEST = [
    "keepitreal/vietnamese-sbert",
    "AITeamVN/Vietnamese_Embedding",
    "thanhtantran/Vietnamese_Embedding_v2",
    "dangvantuan/vietnamese-embedding",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v5-text-nano",
    "jinaai/jina-embeddings-v5-text-small",
    "intfloat/multilingual-e5-base",
]


def benchmark_model(model_name: str) -> dict:
    """Benchmark a single model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    # Set environment variable
    os.environ["EMBED_MODEL_NAME"] = model_name
    
    # Force reimport to use new model
    import importlib
    import config
    import embedder
    importlib.reload(config)
    importlib.reload(embedder)
    
    # Clear singleton
    embedder.Embedder._instance = None
    embedder.Embedder._current_model_name = None
    
    results = {
        "model": model_name,
        "status": "success",
    }
    
    try:
        # Measure loading time
        start = time.time()
        emb = embedder.Embedder.get()
        load_time = time.time() - start
        results["load_time"] = load_time
        results["dimension"] = emb.dim
        
        print(f"✓ Loaded in {load_time:.2f}s")
        print(f"  Dimension: {emb.dim}")
        
        # Measure encoding time (warmup)
        start = time.time()
        _ = emb.encode_one(TEST_TEXTS[0], is_query=True)
        warmup_time = time.time() - start
        results["warmup_time"] = warmup_time
        print(f"  Warmup: {warmup_time:.2f}s")
        
        # Measure batch encoding
        start = time.time()
        vectors = emb.encode(TEST_TEXTS, is_query=True)
        batch_time = time.time() - start
        results["batch_time"] = batch_time
        results["texts_per_second"] = len(TEST_TEXTS) / batch_time
        
        print(f"✓ Encoded {len(TEST_TEXTS)} texts in {batch_time:.3f}s")
        print(f"  Speed: {results['texts_per_second']:.1f} texts/sec")
        print(f"  Output shape: {vectors.shape}")
        
        # Check model size on disk
        cache_dir = config.EMBED_CACHE_DIR / model_name.replace("/", "_").replace("\\", "_")
        if cache_dir.exists():
            size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            results["size_mb"] = size_mb
            print(f"  Cache size: {size_mb:.1f} MB")
        
        print(f"✓ SUCCESS")
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"✗ FAILED: {e}")
    
    return results


def main():
    """Run benchmark on all models."""
    print("\n" + "="*70)
    print("EMBEDDING MODELS BENCHMARK")
    print("="*70)
    print(f"Testing {len(MODELS_TO_TEST)} models with {len(TEST_TEXTS)} sample texts")
    
    all_results = []
    
    for model_name in MODELS_TO_TEST:
        result = benchmark_model(model_name)
        all_results.append(result)
        time.sleep(1)  # Cool down
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<45} {'Status':<10} {'Load(s)':<10} {'Speed(t/s)':<12} {'Dim':<8}")
    print("-" * 95)
    
    for r in all_results:
        status = "✓ OK" if r["status"] == "success" else "✗ FAIL"
        load_time = f"{r.get('load_time', 0):.1f}" if r["status"] == "success" else "-"
        speed = f"{r.get('texts_per_second', 0):.1f}" if r["status"] == "success" else "-"
        dim = str(r.get('dimension', '-'))
        
        print(f"{r['model']:<45} {status:<10} {load_time:<10} {speed:<12} {dim:<8}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    successful = [r for r in all_results if r["status"] == "success"]
    if successful:
        # Find fastest
        fastest = min(successful, key=lambda x: x.get("batch_time", float("inf")))
        print(f"⚡ Fastest: {fastest['model']} ({fastest['texts_per_second']:.1f} texts/sec)")
        
        # Find smallest
        with_size = [r for r in successful if "size_mb" in r]
        if with_size:
            smallest = min(with_size, key=lambda x: x["size_mb"])
            print(f"💾 Smallest: {smallest['model']} ({smallest['size_mb']:.0f} MB)")
        
        # Recommend Vietnamese
        vietnamese_models = [r for r in successful if any(
            keyword in r["model"].lower() 
            for keyword in ["vietnamese", "vietnam", "aiteamvn", "dangvantuan"]
        )]
        if vietnamese_models:
            print(f"\n🇻🇳 Vietnamese-specific models:")
            for vm in vietnamese_models:
                print(f"   - {vm['model']}")
    
    print("\n" + "="*70)
    print("Note: For production RAG with Vietnamese, AITeamVN/Vietnamese_Embedding")
    print("      is recommended for best quality despite slower speed.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
