"""
Make the project's ``src/`` directory importable.

Scripts in this folder (tests, setup, competition CLI) need to import the core
modules that live in ``src/`` (``config``, ``embedder``, ``retriever``,
``llm``, ``textutils``). Importing this module first puts ``src/`` on sys.path.

Usage at the top of a script:

    import _bootstrap  # noqa: F401  (adds src/ to sys.path)
    import config
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
