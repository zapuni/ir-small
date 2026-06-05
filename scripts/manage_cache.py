#!/usr/bin/env python3
"""
Manage vector embeddings cache.

Usage:
    python scripts/manage_cache.py stats              # Show cache statistics
    python scripts/manage_cache.py list               # List all cached files
    python scripts/manage_cache.py clear              # Clear all cache
    python scripts/manage_cache.py clear --model MODEL_NAME
    python scripts/manage_cache.py clear --doc DOC_ID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vector_cache


def show_stats():
    """Show cache statistics."""
    stats = vector_cache.get_cache_stats()
    
    print("\n" + "="*70)
    print("VECTOR CACHE STATISTICS")
    print("="*70)
    print(f"\nTotal files:  {stats['total_files']}")
    print(f"Total size:   {stats['total_size_mb']:.2f} MB")
    
    if stats['models']:
        print(f"\nPer-model breakdown:")
        print(f"{'Model':<50} {'Files':<10} {'Size (MB)':<15}")
        print("-" * 70)
        for model, info in sorted(stats['models'].items()):
            print(f"{model:<50} {info['files']:<10} {info['size_mb']:<15.2f}")
    else:
        print("\n(No cache files)")
    
    print("="*70 + "\n")


def list_cache(model_name: str | None = None):
    """List cached files."""
    caches = vector_cache.list_cache(model_name)
    
    print("\n" + "="*70)
    print("CACHED EMBEDDINGS")
    print("="*70)
    
    if not caches:
        print("\n(No cache files)")
        print("="*70 + "\n")
        return
    
    print(f"\nFound {len(caches)} cache file(s):\n")
    print(f"{'Model':<40} {'Doc ID':<20} {'Chunks':<10} {'Size (MB)':<12}")
    print("-" * 90)
    
    for c in caches:
        model = c['model'][:38] + ".." if len(c['model']) > 40 else c['model']
        doc_id = c['doc_id'][:18] + ".." if len(c['doc_id']) > 20 else c['doc_id']
        print(f"{model:<40} {doc_id:<20} {c['chunks']:<10} {c['size_mb']:<12.2f}")
    
    total_size = sum(c['size_mb'] for c in caches)
    print("-" * 90)
    print(f"{'TOTAL':<40} {'':<20} {'':<10} {total_size:<12.2f}")
    
    print("="*70 + "\n")


def clear_cache(model_name: str | None = None, doc_id: str | None = None, confirm: bool = True):
    """Clear cache files."""
    # Show what will be deleted
    if model_name and doc_id:
        target = f"doc '{doc_id}' for model '{model_name}'"
    elif model_name:
        target = f"all docs for model '{model_name}'"
    elif doc_id:
        target = f"doc '{doc_id}' across all models"
    else:
        target = "ALL cached embeddings"
    
    print("\n" + "="*70)
    print("CLEAR CACHE")
    print("="*70)
    print(f"\nTarget: {target}")
    
    if confirm:
        response = input("\nAre you sure? This cannot be undone. [y/N]: ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return
    
    count = vector_cache.clear_cache(model_name=model_name, doc_id=doc_id)
    
    print(f"\n✓ Deleted {count} cache file(s)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manage vector embeddings cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all cached files")
    list_parser.add_argument("--model", help="Filter by model name")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--model", help="Only clear this model's cache")
    clear_parser.add_argument("--doc", dest="doc_id", help="Only clear this doc_id")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "stats":
        show_stats()
    elif args.command == "list":
        list_cache(model_name=args.model)
    elif args.command == "clear":
        clear_cache(model_name=args.model, doc_id=args.doc_id, confirm=not args.yes)


if __name__ == "__main__":
    main()
