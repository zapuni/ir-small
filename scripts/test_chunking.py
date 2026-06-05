#!/usr/bin/env python3
"""
Test and visualize chunking strategies.

Usage:
    python scripts/test_chunking.py                      # Test current config
    python scripts/test_chunking.py --compare            # Compare strategies
    python scripts/test_chunking.py --model MODEL_NAME   # Test specific model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from textutils import chunk_text, get_optimal_chunk_config
import config

# Sample Vietnamese text for testing
SAMPLE_TEXT = """
Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người. AI đã có những bước tiến đáng kể trong những năm gần đây, đặc biệt là trong các lĩnh vực như xử lý ngôn ngữ tự nhiên, thị giác máy tính và học máy.

Machine Learning là một nhánh quan trọng của AI, cho phép máy tính học từ dữ liệu mà không cần được lập trình cụ thể. Deep Learning, một phương pháp học máy sử dụng mạng neural nhân tạo, đã đạt được những thành tựu đột phá trong nhiều ứng dụng thực tế.

Retrieval Augmented Generation (RAG) là một kỹ thuật kết hợp giữa tìm kiếm thông tin và sinh văn bản. RAG hoạt động bằng cách truy xuất thông tin liên quan từ một kho dữ liệu lớn, sau đó sử dụng thông tin này làm ngữ cảnh để sinh ra câu trả lời chính xác và phù hợp. Kỹ thuật này đặc biệt hữu ích trong các hệ thống hỏi đáp và chatbot.

Embedding là quá trình chuyển đổi văn bản thành vector số trong không gian nhiều chiều. Các vector này nắm bắt được ý nghĩa ngữ nghĩa của văn bản, cho phép so sánh và tìm kiếm dựa trên độ tương đồng. Các mô hình embedding hiện đại như BERT, GPT và các biến thể của chúng đã cải thiện đáng kể chất lượng của việc biểu diễn văn bản.

Chunking là một bước quan trọng trong xử lý văn bản, đặc biệt khi làm việc với các tài liệu dài. Việc chia văn bản thành các đoạn nhỏ hơn (chunks) giúp cải thiện hiệu quả tìm kiếm và giảm thiểu việc mất mát thông tin. Overlap giữa các chunks đảm bảo rằng ngữ cảnh quan trọng không bị cắt đứt ở ranh giới giữa các đoạn.
"""


def analyze_chunks(chunks: List[str], config_name: str = "Current") -> dict:
    """Phân tích các chunks được tạo ra."""
    if not chunks:
        return {"num_chunks": 0}
    
    word_counts = [len(chunk.split()) for chunk in chunks]
    char_counts = [len(chunk) for chunk in chunks]
    
    # Tính overlap thực tế
    overlaps = []
    for i in range(len(chunks) - 1):
        words1 = set(chunks[i].split())
        words2 = set(chunks[i + 1].split())
        overlap = len(words1 & words2)
        overlaps.append(overlap)
    
    analysis = {
        "config_name": config_name,
        "num_chunks": len(chunks),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_chars": sum(char_counts) / len(char_counts),
        "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
        "overlap_ratio": (sum(overlaps) / len(overlaps) / (sum(word_counts) / len(word_counts))) if overlaps else 0,
    }
    
    return analysis


def print_analysis(analysis: dict):
    """In ra phân tích chunk."""
    print(f"\n{'='*70}")
    print(f"Config: {analysis['config_name']}")
    print(f"{'='*70}")
    print(f"Number of chunks:     {analysis['num_chunks']}")
    print(f"Avg words per chunk:  {analysis['avg_words']:.1f}")
    print(f"Min/Max words:        {analysis['min_words']}-{analysis['max_words']}")
    print(f"Avg chars per chunk:  {analysis['avg_chars']:.0f}")
    print(f"Avg overlap words:    {analysis['avg_overlap']:.1f}")
    print(f"Overlap ratio:        {analysis['overlap_ratio']:.1%}")
    print(f"{'='*70}")


def visualize_chunks(chunks: List[str], max_display: int = 3):
    """Hiển thị một vài chunks mẫu."""
    print(f"\n{'='*70}")
    print(f"Sample Chunks (showing first {min(max_display, len(chunks))})")
    print(f"{'='*70}")
    
    for i, chunk in enumerate(chunks[:max_display]):
        words = chunk.split()
        preview = " ".join(words[:30])
        if len(words) > 30:
            preview += "..."
        
        print(f"\nChunk {i+1} ({len(words)} words):")
        print(f"  {preview}")
        
        # Highlight overlap với chunk trước
        if i > 0:
            prev_words = set(chunks[i-1].split())
            curr_words = set(words)
            overlap = prev_words & curr_words
            print(f"  → Overlap với chunk {i}: {len(overlap)} words")


def test_current_config(text: str = SAMPLE_TEXT):
    """Test với config hiện tại."""
    print("\n" + "="*70)
    print("TESTING CURRENT CONFIGURATION")
    print("="*70)
    
    # Get optimal config for current model
    optimal = get_optimal_chunk_config()
    print(f"\nCurrent model: {config.EMBED_MODEL_NAME}")
    print(f"Optimal config: {optimal['chunk_size']} words, {optimal['overlap']} overlap")
    print(f"Reasoning: {optimal['reasoning']}")
    
    # Chunk with adaptive config
    chunks = chunk_text(text, use_adaptive=True)
    
    # Analyze
    analysis = analyze_chunks(chunks, "Adaptive (Model-specific)")
    print_analysis(analysis)
    visualize_chunks(chunks)
    
    return chunks, analysis


def compare_strategies(text: str = SAMPLE_TEXT):
    """So sánh các chunking strategies khác nhau."""
    print("\n" + "="*70)
    print("COMPARING CHUNKING STRATEGIES")
    print("="*70)
    
    strategies = {
        "Current (.env)": {
            "chunk_size": config.CHUNK_SIZE_WORDS,
            "overlap": config.CHUNK_OVERLAP_WORDS,
        },
        "Optimal (Adaptive)": {
            "chunk_size": None,  # Auto-detect
            "overlap": None,
        },
        "Conservative (Small)": {
            "chunk_size": 120,
            "overlap": 30,
        },
        "Aggressive (Large)": {
            "chunk_size": 300,
            "overlap": 80,
        },
        "High Overlap": {
            "chunk_size": 180,
            "overlap": 60,
        },
        "Low Overlap": {
            "chunk_size": 180,
            "overlap": 25,
        },
    }
    
    results = []
    
    for name, cfg in strategies.items():
        print(f"\nTesting: {name}")
        print(f"  Config: {cfg['chunk_size']} words, {cfg['overlap']} overlap")
        
        chunks = chunk_text(
            text, 
            chunk_size=cfg["chunk_size"], 
            overlap=cfg["overlap"],
            use_adaptive=(cfg["chunk_size"] is None),
        )
        
        analysis = analyze_chunks(chunks, name)
        results.append(analysis)
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Strategy':<25} {'Chunks':<8} {'Avg Words':<12} {'Overlap %':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['config_name']:<25} {r['num_chunks']:<8} {r['avg_words']:<12.1f} {r['overlap_ratio']:<12.1%}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find best balance
    best_balance = min(results, key=lambda x: abs(x['overlap_ratio'] - 0.27))
    print(f"\n✓ Best overlap ratio (~27%): {best_balance['config_name']}")
    
    # Find most chunks (best granularity)
    most_chunks = max(results, key=lambda x: x['num_chunks'])
    print(f"✓ Most granular: {most_chunks['config_name']} ({most_chunks['num_chunks']} chunks)")
    
    # Find least chunks (best overview)
    least_chunks = min(results, key=lambda x: x['num_chunks'])
    print(f"✓ Best overview: {least_chunks['config_name']} ({least_chunks['num_chunks']} chunks)")


def test_model_specific(model_name: str, text: str = SAMPLE_TEXT):
    """Test chunking cho một model cụ thể."""
    print("\n" + "="*70)
    print(f"TESTING FOR MODEL: {model_name}")
    print("="*70)
    
    # Temporarily override config
    import os
    original = os.environ.get("EMBED_MODEL_NAME")
    os.environ["EMBED_MODEL_NAME"] = model_name
    
    # Reload config
    import importlib
    importlib.reload(config)
    
    # Get optimal config
    optimal = get_optimal_chunk_config(model_name)
    print(f"\nOptimal config for {model_name}:")
    print(f"  Chunk size: {optimal['chunk_size']} words")
    print(f"  Overlap: {optimal['overlap']} words ({optimal['overlap_ratio']:.1%})")
    print(f"  Reasoning: {optimal['reasoning']}")
    
    # Chunk
    chunks = chunk_text(text, use_adaptive=True)
    
    # Analyze
    analysis = analyze_chunks(chunks, f"{model_name} (Adaptive)")
    print_analysis(analysis)
    visualize_chunks(chunks, max_display=2)
    
    # Restore original
    if original:
        os.environ["EMBED_MODEL_NAME"] = original
    
    return chunks, analysis


def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize chunking strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple chunking strategies",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_NAME",
        help="Test chunking for a specific embedding model",
    )
    parser.add_argument(
        "--text",
        metavar="FILE",
        help="Path to text file to chunk (default: sample text)",
    )
    
    args = parser.parse_args()
    
    # Load custom text if provided
    text = SAMPLE_TEXT
    if args.text:
        text_path = Path(args.text)
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            print(f"Loaded text from: {args.text} ({len(text)} chars)")
        else:
            print(f"Warning: File not found: {args.text}")
            print("Using sample text instead.")
    
    if args.model:
        test_model_specific(args.model, text)
    elif args.compare:
        compare_strategies(text)
    else:
        test_current_config(text)


if __name__ == "__main__":
    from typing import List
    main()
