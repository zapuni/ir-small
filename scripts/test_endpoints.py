"""
HTTP test against a running Student Server (src/app.py).

Run (with the server already running):  python scripts/test_endpoints.py
"""
from __future__ import annotations

import time

import requests

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import config

BASE = f"http://127.0.0.1:{config.PORT}"

DOC = """
Retrieval-Augmented Generation (RAG) là kỹ thuật kết hợp truy xuất thông tin với
mô hình ngôn ngữ lớn. Hệ thống truy xuất các đoạn tài liệu liên quan từ vector
database rồi đưa vào ngữ cảnh để sinh câu trả lời chính xác.

BM25 là thuật toán xếp hạng theo tần suất từ khóa cho tìm kiếm từ vựng. Kết hợp BM25
với dense retrieval tạo thành hybrid search giúp tăng độ chính xác.

Mô hình embedding tiếng Việt vietnamese-sbert dựa trên PhoBERT, nhẹ và chạy tốt trên CPU.
"""

QUESTIONS = [
    "RAG là viết tắt của gì? A. Random Access Generation B. Retrieval-Augmented Generation C. Recursive Aggregation Graph D. Rapid Answer Generator",
    "Thuật toán nào xếp hạng theo tần suất từ khóa? A. BM25 B. PhoBERT C. Cosine D. MMR",
    "Mô hình embedding tiếng Việt nào chạy tốt trên CPU? A. GPT-4 B. vietnamese-sbert C. Llama D. BERT-base",
]


def main() -> None:
    print("[health]", requests.get(f"{BASE}/health", timeout=10).json())

    t0 = time.time()
    r = requests.post(f"{BASE}/upload", json={"doc_id": "none", "text": DOC}, timeout=120)
    print(f"[upload] {r.json()}  ({time.time() - t0:.2f}s)")

    for q in QUESTIONS:
        t0 = time.time()
        r = requests.post(f"{BASE}/ask", json={"question": q}, timeout=60)
        data = r.json()
        ans = data.get("answer")
        print(f"[ask] answer={ans!r}  ({time.time() - t0:.2f}s)  sources={len(data.get('sources', []))}")
        assert ans in ("A", "B", "C", "D"), f"INVALID answer: {ans!r}"

    print("All endpoint checks passed (answers are valid single letters).")


if __name__ == "__main__":
    main()
