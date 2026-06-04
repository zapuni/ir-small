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

# SAMPLE_DOC = """
# Retrieval-Augmented Generation (RAG) là một kỹ thuật kết hợp giữa truy xuất thông tin
# và mô hình ngôn ngữ lớn. Thay vì chỉ dựa vào kiến thức đã học, mô hình sẽ truy xuất
# các đoạn tài liệu liên quan từ một kho dữ liệu (vector database) rồi đưa vào ngữ cảnh
# để sinh câu trả lời chính xác hơn.

# Quy trình RAG gồm ba bước chính. Bước thứ nhất là chunking, tức là chia tài liệu lớn
# thành các đoạn nhỏ. Bước thứ hai là embedding, biến mỗi đoạn thành một vector số học
# bằng mô hình embedding. Bước thứ ba là truy xuất, tìm các đoạn có vector gần nhất với
# câu hỏi rồi đưa cho mô hình ngôn ngữ.

# Vector database là nơi lưu trữ các vector embedding. Nó cho phép tìm kiếm theo độ tương
# đồng cosine một cách nhanh chóng. Các hệ thống phổ biến gồm FAISS, Milvus và Qdrant.

# BM25 là một thuật toán xếp hạng dựa trên tần suất từ khóa, thường được dùng cho tìm kiếm
# từ vựng (lexical search). Khi kết hợp BM25 với tìm kiếm ngữ nghĩa (dense retrieval), ta
# được hybrid search, giúp tăng độ chính xác của hệ thống truy vấn.

# Embedding model tiếng Việt phổ biến là vietnamese-sbert, dựa trên kiến trúc PhoBERT.
# Mô hình này nhẹ và có thể chạy tốt trên CPU mà không cần GPU.
# """

# QUESTIONS = [
#     ("RAG là viết tắt của khái niệm nào? A. Random Access Generation "
#      "B. Retrieval-Augmented Generation C. Recursive Aggregation Graph "
#      "D. Rapid Answer Generator", "B"),
#     ("Thuật toán nào dựa trên tần suất từ khóa cho tìm kiếm từ vựng? "
#      "A. BM25 B. PhoBERT C. Cosine D. MMR", "A"),
#     ("Mô hình embedding tiếng Việt nào được nhắc tới có thể chạy trên CPU? "
#      "A. BERT-base B. GPT-4 C. vietnamese-sbert D. Llama", "C"),
#     ("Bước nào chia tài liệu lớn thành các đoạn nhỏ? "
#      "A. Embedding B. Chunking C. Truy xuất D. Sinh văn bản", "B"),
#     ("Hệ thống nào KHÔNG phải là vector database được nhắc tới? "
#      "A. FAISS B. Milvus C. Qdrant D. PostgreSQL", "D"),
# ]

SAMPLE_DOC = """Dự án Phục hồi Sinh thái Lưu vực sông Trà Giang (Dự án TG-20) được phê duyệt vào năm 2018 với mục tiêu ban đầu là tái thiết lập 40% diện tích rừng ngập mặn trước năm 2026. Tuy nhiên, theo Nghiên cứu sửa đổi năm 2021 của Viện Hải học, chỉ số sinh học đa dạng (H') chỉ có thể đạt mức tối ưu (trên 3.5) nếu tỷ lệ bao phủ của loài Đước đôi (Rhizophora apiculata) không vượt quá 60% tổng diện tích tái trồng, đồng thời phải duy trì sự hiện diện của loài bần trắng (Avicennia alba) ở mức tối thiểu 15%.

Đáng chú ý, Nghị định 84 ban hành năm 2022 quy định rằng mọi dự án lâm nghiệp ven biển nhận ngân sách từ Quỹ Tài khóa Xanh phải đảm bảo các loài cây bản địa chiếm ít nhất 75% mật độ diện tích. Trong khi đó, tài liệu phân loại thực vật của địa phương xếp loài Đước đôi vào nhóm cây bán bản địa (loài nhập cư từ thế kỷ 18 nhưng đã tự nhiên hóa), còn bần trắng và vẹt dù (Bruguiera gymnorhiza) là các loài bản địa thuần túy. Để đảm bảo đa dạng sinh học, dự án TG-20 quyết định cơ cấu cây trồng sẽ chỉ gồm ba loài này. Tuy nhiên, các khảo sát sinh thái cho thấy loài vẹt dù chỉ có thể sinh trưởng tốt nếu tổng tỷ lệ diện tích của Đước đôi và bần trắng không vượt quá 70%.

Nếu TG-20 áp dụng nguồn vốn từ Quỹ Tài khóa Xanh từ quý II năm 2023, họ buộc phải điều chỉnh cơ cấu cây trồng để tuân thủ pháp lý. Mặc dù vậy, việc tăng tỷ lệ bần trắng lên trên 25% để bù đắp cho việc giảm tỷ lệ Đước đôi (nhằm đáp ứng Nghị định 84) lại kích hoạt cơ chế cạnh tranh dinh dưỡng nghiêm trọng, làm suy giảm mật độ tảo silic - nguồn thức ăn chính của quần thể tôm he bản địa. Sự suy giảm quần thể tôm he dưới mức kiểm soát sẽ gián tiếp làm mất hiệu lực của Khoản b, Điều 4 trong cam kết với Hiệp hội Thủy sản Quốc tế (IFA) về việc bảo tồn sinh kế bền vững, từ đó dẫn đến việc đình chỉ khoản giải ngân trị giá 5 triệu USD dự kiến vào năm 2025."""

QUESTIONS = [
    (
        "Giả sử dự án TG-20 nhận ngân sách từ Quỹ Tài khóa Xanh và muốn đảm bảo tất cả "
        "các điều kiện sinh trưởng của cây trồng, mục tiêu đa dạng sinh học tối ưu (H' > 3.5), "
        "cũng như không làm mất khoản giải ngân 5 triệu USD từ IFA vào năm 2025. Tỷ lệ diện tích "
        "tối đa của loài Đước đôi (Rhizophora apiculata) có thể gieo trồng là bao nhiêu? "
        "A. 60% B. 25% C. 20% D. 15%",
        "B"
    ),
    (
        "Nếu vào cuối năm 2023, tài liệu phân loại thực vật địa phương được cập nhật và chính thức "
        "công nhận Đước đôi (Rhizophora apiculata) là 'loài bản địa thuần túy' (thay vì bán bản địa), "
        "điều này ảnh hưởng thế nào đến giới hạn gieo trồng của loài này trong dự án TG-20 (vẫn áp dụng "
        "Quỹ Tài khóa Xanh và các ràng buộc sinh thái khác giữ nguyên)? "
        "A. Dự án có thể tăng tỷ lệ Đước đôi lên tối đa 55% mà vẫn thỏa mãn tất cả các ràng buộc pháp lý, "
        "sinh thái và tài chính liên quan. B. Dự án có thể tăng tỷ lệ Đước đôi lên tối đa 60% vì lúc này "
        "Đước đôi đã là loài bản địa, giúp dễ dàng đạt yêu cầu của Nghị định 84. C. Giới hạn tối đa của "
        "Đước đôi vẫn phải giữ ở mức 25% để đảm bảo loài vẹt dù có thể sinh trưởng tốt. D. Dự án bắt buộc "
        "phải loại bỏ hoàn toàn bần trắng để tập trung tăng tỷ lệ Đước đôi lên tối đa nhằm tối ưu hóa chỉ số H'.",
        "A"
    ),
    (
        "Theo logic của văn bản, hành động nào sau đây là nguyên nhân gốc rễ dẫn đến việc IFA "
        "đình chỉ khoản giải ngân 5 triệu USD vào năm 2025? "
        "A. Không đạt được mục tiêu tái thiết lập 40% diện tích rừng ngập mặn trước năm 2026. "
        "B. Tăng tỷ lệ bần trắng vượt quá 25% nhằm bù đắp cho việc giảm Đước đôi để đáp ứng Nghị định 84. "
        "C. Sử dụng nguồn vốn từ Quỹ Tài khóa Xanh mà không thực hiện đánh giá tác động sinh thái đối với "
        "loài tảo silic. D. Sự xuất hiện của loài vẹt dù cạnh tranh dinh dưỡng trực tiếp với quần thể tôm he bản địa.",
        "B"
    ),
    (
        "Phát biểu nào sau đây là SAI dựa trên các thông tin được cung cấp trong văn bản? "
        "A. Đước đôi không được xem là loài bản địa thuần túy theo tài liệu phân loại của địa phương trước "
        "năm 2023. B. Nếu dự án không tiếp nhận ngân sách từ Quỹ Tài khóa Xanh, họ không bắt buộc phải "
        "duy trì tổng tỷ lệ bần trắng và vẹt dù tối thiểu là 75%. C. Chỉ cần duy trì tỷ lệ Đước đôi không quá "
        "60% và bần trắng tối thiểu 15% thì chỉ số đa dạng sinh học (H') chắc chắn sẽ đạt trên 3.5. D. Mật độ "
        "tảo silic là yếu tố trung gian liên kết giữa cơ cấu cây trồng của rừng ngập mặn và việc thực thi cam "
        "kết tài chính với IFA.",
        "C"
    )
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
