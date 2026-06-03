"""
End-to-end self-test:  python scripts/selftest.py

Builds the index from a sample doc, then for each question runs the FULL
pipeline: parse -> hybrid retrieve -> LLM (from .env) -> letter, with the
embedding fallback as a safety net. Reports accuracy + timing.

Works whether or not the LLM endpoint is reachable: if the LLM fails, the
fallback answers and the test still completes.
"""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import config
from llm import fallback_answer, get_client
from retriever import INDEX
from textutils import parse_mc_question

SAMPLE_DOC = """
Retrieval-Augmented Generation (RAG) là một kỹ thuật kết hợp giữa truy xuất thông tin
và mô hình ngôn ngữ lớn. Thay vì chỉ dựa vào kiến thức đã học, mô hình sẽ truy xuất
các đoạn tài liệu liên quan từ một kho dữ liệu (vector database) rồi đưa vào ngữ cảnh
để sinh câu trả lời chính xác hơn.

Quy trình RAG gồm ba bước chính. Bước thứ nhất là chunking, tức là chia tài liệu lớn
thành các đoạn nhỏ. Bước thứ hai là embedding, biến mỗi đoạn thành một vector số học
bằng mô hình embedding. Bước thứ ba là truy xuất, tìm các đoạn có vector gần nhất với
câu hỏi rồi đưa cho mô hình ngôn ngữ.

Vector database là nơi lưu trữ các vector embedding. Nó cho phép tìm kiếm theo độ tương
đồng cosine một cách nhanh chóng. Các hệ thống phổ biến gồm FAISS, Milvus và Qdrant.

BM25 là một thuật toán xếp hạng dựa trên tần suất từ khóa, thường được dùng cho tìm kiếm
từ vựng (lexical search). Khi kết hợp BM25 với tìm kiếm ngữ nghĩa (dense retrieval), ta
được hybrid search, giúp tăng độ chính xác của hệ thống truy vấn.

Embedding model tiếng Việt phổ biến là vietnamese-sbert, dựa trên kiến trúc PhoBERT.
Mô hình này nhẹ và có thể chạy tốt trên CPU mà không cần GPU.
"""

QUESTIONS = [
    ("RAG là viết tắt của khái niệm nào? A. Random Access Generation "
     "B. Retrieval-Augmented Generation C. Recursive Aggregation Graph "
     "D. Rapid Answer Generator", "B"),
    ("Thuật toán nào dựa trên tần suất từ khóa cho tìm kiếm từ vựng? "
     "A. BM25 B. PhoBERT C. Cosine D. MMR", "A"),
    ("Mô hình embedding tiếng Việt nào được nhắc tới có thể chạy trên CPU? "
     "A. BERT-base B. GPT-4 C. vietnamese-sbert D. Llama", "C"),
    ("Bước nào chia tài liệu lớn thành các đoạn nhỏ? "
     "A. Embedding B. Chunking C. Truy xuất D. Sinh văn bản", "B"),
    ("Hệ thống nào KHÔNG phải là vector database được nhắc tới? "
     "A. FAISS B. Milvus C. Qdrant D. PostgreSQL", "D"),
]


def main() -> None:
    print(config.describe())
    print("=" * 64)
    n = INDEX.build(SAMPLE_DOC)
    print(f"Indexed {n} chunks.\n")

    try:
        client = get_client()
        llm_ok = True
    except Exception as exc:
        print(f"[selftest] LLM unavailable ({exc!r}); using fallback only.")
        client = None
        llm_ok = False

    correct = 0
    for q, gold in QUESTIONS:
        mcq = parse_mc_question(q)
        queries = [mcq.stem] + [f"{mcq.stem} {mcq.options[l]}" for l in mcq.option_letters()]
        contexts = [r.chunk for r in INDEX.search(queries, top_k=config.TOP_K_CONTEXT)]

        t0 = time.time()
        ans = None
        if llm_ok:
            try:
                ans = client.answer(mcq, contexts, deadline=time.monotonic() + 30)
            except Exception as exc:
                print(f"   (llm error: {exc!r})")
        if ans not in ("A", "B", "C", "D"):
            ans = fallback_answer(mcq, contexts)
            src = "fallback"
        else:
            src = "llm"
        dt = time.time() - t0

        ok = "OK " if ans == gold else "XX "
        correct += int(ans == gold)
        print(f"{ok} [{src:8s}] pred={ans} gold={gold}  ({dt:.2f}s)  {mcq.stem[:50]}")

    print("=" * 64)
    print(f"Accuracy: {correct}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
