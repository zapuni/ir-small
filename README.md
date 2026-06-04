# Offline RAG Competition — Student Server

Hệ thống RAG trả lời câu hỏi trắc nghiệm (A/B/C/D), tối ưu cho **CPU laptop, không GPU**,
phản hồi nhanh cho từng request. Xây dựng theo đúng đặc tả trong `tmp/request.md`.

## Kiến trúc (tối ưu độ chính xác + tốc độ trên CPU)

```
          /upload                         /ask
 text ──► chunk (câu, ~180 từ, overlap 40)  question ─► tách stem + A/B/C/D
       └► embed (vietnamese-sbert, CPU)              └► hybrid retrieve:
       └► BM25 (pyvi word-seg)                            • dense (SBERT cosine)
       └► lưu vào index trong RAM                         • BM25 (lexical)
                                                          • RRF fusion + MMR
                                                       └► LLM (qua .env) ─► 1 ký tự
                                                       └► fallback embedding nếu LLM lỗi
```

Điểm chính:
- **Embedding local**: `keepitreal/vietnamese-sbert` (PhoBERT, 768d) tải sẵn về
  `src/models/`, chạy hẳn trên CPU, dùng hết số core.
- **Hybrid retrieval** (BM25 + dense, hợp nhất bằng Reciprocal Rank Fusion, đa dạng
  hoá bằng MMR) cho độ chính xác cao mà không cần reranker nặng.
- **Option-aware retrieval**: truy hồi theo cả câu hỏi lẫn từng đáp án A/B/C/D nên
  ngữ cảnh bao phủ mọi lựa chọn.
- **Tốc độ**: model nạp 1 lần lúc khởi động; cosine brute-force bằng numpy (corpus
  chỉ 1 tài liệu nên cực nhanh); không cần FAISS.
- **An toàn thời gian**: `/ask` có hạn chót wall-clock (`ASK_DEADLINE`, mặc định 45s
  < giới hạn 60s). LLM chạy trong thread riêng; nếu treo/quá hạn → tự động fallback.
- **Luôn trả về 1 ký tự hợp lệ** A/B/C/D, kể cả khi LLM offline (fallback bằng độ
  tương đồng embedding).

## Cấu trúc thư mục

```
ir/
├── .env                      # TẤT CẢ cấu hình (API key, model, tuning) — KHÔNG commit
├── .gitignore
├── pyproject.toml            # khai báo dependency (pin version) cho uv
├── uv.lock                   # khoá phiên bản chính xác — cài lại y hệt mọi máy
├── requirements.txt          # bản pin tương đương cho pip (fallback)
├── README.md
├── src/                      # ⭐ LÕI HỆ THỐNG — đọc thư mục này là hiểu toàn bộ logic
│   ├── config.py             # đọc .env, chọn LLM (generic vs Teacher proxy)
│   ├── embedder.py           # vietnamese-sbert trên CPU
│   ├── textutils.py          # chunking, tokenize tiếng Việt, parse trắc nghiệm
│   ├── retriever.py          # hybrid BM25 + dense + RRF + MMR
│   ├── llm.py                # gọi LLM + parse 1 ký tự + fallback
│   ├── app.py                # FastAPI: /upload, /ask, /health
│   └── models/               # model đã tải về (offline)
├── scripts/                  # tiện ích chạy QUANH hệ thống (không phải lõi)
│   ├── _bootstrap.py         # thêm src/ vào sys.path cho các script
│   ├── download_model.py     # tải model về local (chạy khi còn mạng)
│   ├── compete.py            # đăng ký/evaluate/reset với Teacher Server
│   ├── selftest.py           # test end-to-end (dùng LLM trong .env)
│   └── test_endpoints.py     # test HTTP khi server đang chạy
└── tmp/
    ├── request.md            # đặc tả gốc
    └── D22_CLC_Teaching_Slide (Next).pdf
```

> **Đọc code**: toàn bộ logic hệ thống nằm trong `src/`. Luồng đọc gợi ý:
> `config.py` → `embedder.py` → `textutils.py` → `retriever.py` → `llm.py` → `app.py`.

## Cài đặt (khi còn Internet) — dùng `uv`

Dự án quản lý môi trường bằng [uv](https://docs.astral.sh/uv/). Tất cả dependency
được **pin phiên bản chính xác** trong `pyproject.toml` và khoá trong `uv.lock`,
nên cài lại trên máy thi sẽ ra môi trường y hệt.

```powershell
# 0. Cài uv (chọn 1 cách)
#    - winget:    winget install astral-sh.uv
#    - pip:       pip install uv
#    - script:    https://docs.astral.sh/uv/getting-started/installation/

# 1. Tạo venv + cài đúng phiên bản đã khoá (đọc uv.lock)
uv sync

# 2. Tải model embedding về local (đã tải sẵn vào src/models/)
uv run python scripts/download_model.py
```

`uv sync` tự tạo `.venv/` và cài torch bản **CPU** (`2.12.0+cpu`) từ index chính
thức của PyTorch — không kéo về gói GPU. Mọi lệnh sau đều chạy qua `uv run` để
dùng đúng môi trường này (không cần activate thủ công).

> Không có/không muốn dùng uv? Vẫn cài được bằng pip: `pip install -r requirements.txt`
> (xem ghi chú về torch CPU ở đầu file đó).

## Cấu hình — chỉ sửa `.env`

Mở `ir/.env`:
- `STUDENT_ID` → đổi thành **MSSV viết hoa** của bạn.
- **LLM tạm thời** (đang bật): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`.
- **Chuyển sang Teacher proxy** khi thi: đặt `LLM_USE_PROXY=true` (hoặc xoá
  `LLM_BASE_URL`). Khi đó key chính là `STUDENT_ID`, model là `LLM_MODEL`.

> **Qwen3/3.5 (4B–8B, ngày thi):** đặt `LLM_ENABLE_THINKING=false` (mặc định khi
> model có tên `qwen`). Code tự tắt thinking qua `chat_template_kwargs`, prompt ngắn
> dạng `<context>`, và `LLM_MAX_TOKENS=96`. Có thể ghi đè endpoint trong `.env.local`.
> Reasoning model (mimo, …): profile `reasoning`, `LLM_MAX_TOKENS=512`.

## Chạy server (máy thi)

```powershell
uv run python src/app.py        # lắng nghe http://0.0.0.0:5000
```

Kiểm tra nhanh khi server đang chạy (mở terminal khác, đứng ở thư mục gốc `ir/`):

```powershell
uv run python scripts/test_endpoints.py     # /health, /upload, 3 câu /ask
uv run python scripts/selftest.py           # 5 câu, đo accuracy + thời gian
```

## Thi đấu (khi Teacher Server hoạt động)

Hiện Teacher Server đang **offline**; mọi thứ đã cấu hình sẵn nên khi nó bật lên
chỉ cần chạy (server `app.py` phải đang chạy ở terminal khác):

```powershell
uv run python scripts/compete.py register     # tự dò IP LAN, đăng ký server_url
uv run python scripts/compete.py evaluate      # bắt đầu thi (Teacher gọi /upload rồi 10x /ask)
uv run python scripts/compete.py result        # xem điểm/tiến độ
uv run python scripts/compete.py reset         # reset nếu cần
```

`register` tự phát hiện IP LAN; nếu cần ép tay: `uv run python scripts/compete.py register --ip 192.168.1.15 --port 5000`.

## API (Student Server)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/upload` | POST | `{doc_id, text}` → chunk + embed + index. Trả `{status, doc_id, chunks}` |
| `/ask` | POST | `{question}` → RAG. Trả `{answer: "A|B|C|D", sources: [...]}` |
| `/health` | GET | trạng thái sẵn sàng |

## Tinh chỉnh (trong `.env`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | 180 / 40 | kích thước & overlap chunk |
| `TOP_K_CONTEXT` | 5 | số chunk đưa vào LLM |
| `TOP_K_DENSE` / `TOP_K_BM25` | 12 / 12 | số ứng viên mỗi retriever |
| `MMR_LAMBDA` | 0.3 | cân bằng liên quan ↔ đa dạng |
| `ASK_DEADLINE` | 45 | hạn chót (giây) cho /ask trước khi fallback |
| `LLM_PROFILE` | auto | `qwen_small` / `reasoning` / `default` |
| `LLM_ENABLE_THINKING` | auto | `false` cho Qwen MCQ (nhanh, ~1s/câu) |
| `LLM_MAX_TOKENS` | 96 (Qwen) | 512 cho reasoning model |
| `CONTEXT_MAX_CHARS` | 5500 | cắt ngữ cảnh vừa seq len proxy |
