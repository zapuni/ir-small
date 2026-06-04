"""Quick LLM probe: python scripts/_probe_llm.py"""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401
import config
from llm import get_client
from textutils import parse_mc_question

Q = (
    "RAG là viết tắt của khái niệm nào? "
    "A. Random Access Generation "
    "B. Retrieval-Augmented Generation "
    "C. Recursive Aggregation Graph "
    "D. Rapid Answer Generator"
)
CTX = [
    "Retrieval-Augmented Generation (RAG) kết hợp truy xuất thông tin và mô hình ngôn ngữ.",
]


def main() -> None:
    print(config.describe())
    mcq = parse_mc_question(Q)
    t0 = time.time()
    ans = get_client().answer(mcq, CTX, deadline=time.monotonic() + 30)
    print(f"answer={ans!r} elapsed={time.time() - t0:.2f}s (gold=B)")


if __name__ == "__main__":
    main()
