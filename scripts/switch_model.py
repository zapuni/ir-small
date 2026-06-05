#!/usr/bin/env python3
"""
Quick script to switch embedding models in .env file.

Usage:
    python scripts/switch_model.py                    # Show current model
    python scripts/switch_model.py --list             # List available models
    python scripts/switch_model.py --set MODEL_NAME   # Switch to a model
    
Examples:
    python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding
    python scripts/switch_model.py --set jinaai/jina-embeddings-v5-text-nano
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Available models with descriptions
AVAILABLE_MODELS = {
    "Vietnamese-Specific Models": [
        ("keepitreal/vietnamese-sbert", "Default, small, fast (256 dim)"),
        ("AITeamVN/Vietnamese_Embedding", "⭐ Best for Vietnamese RAG (1024 dim, BGE-M3)"),
        ("thanhtantran/Vietnamese_Embedding_v2", "Improved BGE-M3 (1024 dim)"),
        ("dangvantuan/vietnamese-embedding", "PhoBERT-based, balanced (768 dim)"),
    ],
    "Multilingual Models": [
        ("jinaai/jina-embeddings-v5-text-nano", "Compact, modern (512 dim, 239M params)"),
        ("jinaai/jina-embeddings-v5-text-small", "Better quality (512 dim, 677M params)"),
        ("intfloat/multilingual-e5-base", "⭐ Proven multilingual (768 dim, 278M params)"),
    ],
}


def get_env_path() -> Path:
    """Get path to .env file."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root / ".env"


def read_env() -> dict[str, str]:
    """Read .env file into a dict."""
    env_path = get_env_path()
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
    
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    return env_vars


def write_env(env_vars: dict[str, str]) -> None:
    """Write env vars back to .env file, preserving comments."""
    env_path = get_env_path()
    
    # Read original file to preserve structure and comments
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Update lines with new values
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env_vars:
                # Update this line
                new_lines.append(f"{key}={env_vars[key]}\n")
            else:
                new_lines.append(line)
        else:
            # Keep comments and empty lines
            new_lines.append(line)
    
    # Write back
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def get_current_model() -> str:
    """Get current embedding model from .env."""
    env_vars = read_env()
    return env_vars.get("EMBED_MODEL_NAME", "not set")


def list_models() -> None:
    """List all available models."""
    print("\n" + "="*70)
    print("AVAILABLE EMBEDDING MODELS")
    print("="*70 + "\n")
    
    for category, models in AVAILABLE_MODELS.items():
        print(f"{category}:")
        print("-" * 70)
        for model_name, description in models:
            print(f"  {model_name}")
            print(f"    → {description}")
        print()
    
    print("="*70)
    print("\nTo switch model, run:")
    print("  python scripts/switch_model.py --set MODEL_NAME")
    print("\nExample:")
    print("  python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding")
    print("="*70 + "\n")


def set_model(model_name: str) -> None:
    """Set embedding model in .env."""
    # Validate model name
    all_models = []
    for models in AVAILABLE_MODELS.values():
        all_models.extend([m[0] for m in models])
    
    if model_name not in all_models:
        print(f"\n⚠️  Warning: '{model_name}' is not in the recommended list.")
        print("   You can still use it, but it may not work correctly.")
        response = input("   Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("   Cancelled.")
            return
    
    # Update .env
    env_vars = read_env()
    old_model = env_vars.get("EMBED_MODEL_NAME", "not set")
    env_vars["EMBED_MODEL_NAME"] = model_name
    write_env(env_vars)
    
    print("\n" + "="*70)
    print("MODEL SWITCHED")
    print("="*70)
    print(f"  Old: {old_model}")
    print(f"  New: {model_name}")
    print("="*70)
    print("\n✓ Updated .env file")
    print("\nNext steps:")
    print("  1. Restart your server to load the new model")
    print("  2. The model will be auto-downloaded on first use")
    print("  3. Check cache in: src/models/{}/".format(
        model_name.replace("/", "_").replace("\\", "_")
    ))
    print("\n" + "="*70 + "\n")


def show_current() -> None:
    """Show current model."""
    current = get_current_model()
    print("\n" + "="*70)
    print("CURRENT EMBEDDING MODEL")
    print("="*70)
    print(f"  {current}")
    print("="*70)
    
    # Find description
    for category, models in AVAILABLE_MODELS.items():
        for model_name, description in models:
            if model_name == current:
                print(f"\n  Category: {category}")
                print(f"  Description: {description}")
                break
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Switch embedding models for the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models",
    )
    parser.add_argument(
        "--set",
        metavar="MODEL_NAME",
        help="Set the embedding model (e.g., AITeamVN/Vietnamese_Embedding)",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.set:
        set_model(args.set)
    else:
        show_current()


if __name__ == "__main__":
    main()
