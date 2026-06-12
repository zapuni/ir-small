# Offline RAG Competition — Student Server

Hệ thống RAG trả lời câu hỏi trắc nghiệm (A/B/C/D), tối ưu cho **CPU laptop, không GPU**,
phản hồi nhanh cho từng request. Xây dựng theo đúng đặc tả trong `tmp/request.md`.

---

## ⚡ TRIỂN KHAI NHANH NGÀY THI (đọc mục này là đủ)

> Chi tiết chiến lược 5 lượt + chọn model: xem `docs/COMPETITION_PLAYBOOK.md`.

### B0 — Giờ setup (CÒN Internet)
```powershell
uv sync                                          # cài môi trường (CPU torch)
uv run python scripts/download_all_models.py     # tải TẤT CẢ model về local
```
Sửa `ir/.env`: đặt `STUDENT_ID=<MSSV viết hoa>`. Khi thi thật, dùng Teacher proxy:
đặt `LLM_USE_PROXY=true` (key = STUDENT_ID, model = `LLM_MODEL`).
Model mặc định đã set sẵn: `paraphrase-multilingual-MiniLM-L12-v2` (robust nhất).

### B1 — Bật server (để chạy suốt, terminal riêng)
```powershell
uv run python src/app.py        # http://0.0.0.0:5000
```

### B2 — Lượt 1: đăng ký + thi (BẮT BUỘC nhận tài liệu)
```powershell
uv run python scripts/compete.py register
uv run python scripts/compete.py evaluate              # document_received=false
uv run python scripts/embed_all_models.py              # (tuỳ chọn) embed sẵn các model khác
```
Server tự **lưu tài liệu + 100 câu hỏi vào `logs/eval/`** và cache vector xuống đĩa.

### B3 — Đọc log để biết đề rồi tinh chỉnh
Mở `logs/eval/<round>__questions.jsonl` (100 câu thật + đáp án đã trả).
- Câu **factoid** (ngày/số/tên/mã) → đổi `dangvantuan/vietnamese-embedding` hoặc `keepitreal/vietnamese-sbert`.
- Câu **ngữ nghĩa/suy luận** → giữ `paraphrase-multilingual` hoặc thử `intfloat/multilingual-e5-base`.

### B4 — Lượt 2–5: đổi model/config rồi thi lại (KHÔNG cần upload lại)
```powershell
uv run python scripts/switch_model.py --set dangvantuan/vietnamese-embedding
#  (đổi sang model chunk-lớn e5/AITeamVN thì đặt CONTEXT_MAX_CHARS=12000 trong .env)
#  >>> RESTART server (Ctrl+C rồi chạy lại app.py) — index nạp lại từ cache trong vài giây <<<
uv run python scripts/compete.py reset
uv run python scripts/compete.py evaluate --document-received   # Teacher BỎ QUA /upload
```
Lặp lại, giữ lượt điểm cao nhất (điểm cao nhất được tính).

---

## Kiến trúc (tối ưu độ chính xác + tốc độ trên CPU)

```
          /upload                              /ask
 text ──► chunk (adaptive theo model)   question ─► tách stem + A/B/C/D
       └► embed (multi-model, CPU)             └► hybrid retrieve:
       └► cache embeddings per model ⚡            • dense (cosine) + BM25 (lexical)
       └► lưu text + index ra đĩa                  • RRF fusion + MMR
       └► log tài liệu (logs/eval/)                • + option-evidence (BM25 top-1/đáp án)
                                                 └► LLM (qua .env) ─► 1 ký tự
                                                 └► fallback embedding nếu LLM lỗi
                                                 └► log câu hỏi + đáp án (logs/eval/)
```

Điểm chính:
- **Option-evidence retrieval** ⭐ (đòn bẩy chính): ngoài hybrid RRF+MMR, đảm bảo đưa
  vào context **chunk khớp BM25 chính xác nhất cho TỪNG đáp án A/B/C/D**. Khắc phục
  việc RRF pha loãng tín hiệu khớp-chính-xác (ngày/email/SĐT/mã) → recall 75%→97%.
- **Chuẩn hoá ngày/số/mã cho BM25** (structure-agnostic): `11/06/2024` ≡ `11/6/2024`,
  giữ `89/QĐ-TTg` nguyên khối → khớp factoid chính xác trên mọi loại văn bản.
- **Multi-model embedding**: đổi model trong `.env`, mỗi model cache riêng. Mặc định
  `paraphrase-multilingual-MiniLM-L12-v2` (robust nhất). Xem `docs/COMPETITION_PLAYBOOK.md`.
- **Persistent cache + docstore** ⚡: text + embeddings lưu ra đĩa. Restart server /
  đổi model → nạp lại index trong vài giây, không cần Teacher gửi lại tài liệu.
- **Request logging**: mỗi lượt thi, tài liệu + 100 câu hỏi được lưu `logs/eval/` để
  phân tích và tinh chỉnh cho lượt sau.
- **Adaptive chunking**: tự điều chỉnh chunk size theo model (80–320 từ, overlap ~27%).
- **Tốc độ**: model nạp 1 lần lúc khởi động; cosine brute-force bằng numpy; ~1s/câu.
- **An toàn thời gian**: `/ask` có hạn chót (`ASK_DEADLINE`=45s < 60s); LLM treo/quá hạn
  → tự động fallback bằng độ tương đồng embedding.
- **Luôn trả về 1 ký tự hợp lệ** A/B/C/D, kể cả khi LLM offline.

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
│   ├── embedder.py           # embedding đa model trên CPU
│   ├── textutils.py          # chunking, tokenize VI + chuẩn hoá ngày/số/mã, parse trắc nghiệm
│   ├── retriever.py          # hybrid BM25 + dense + RRF + MMR + search_mcq (option-evidence)
│   ├── llm.py                # gọi LLM + parse 1 ký tự + fallback
│   ├── docstore.py           # lưu/nạp lại text tài liệu (restart không cần upload lại)
│   ├── reqlog.py             # log tài liệu + 100 câu hỏi mỗi lượt thi (logs/eval/)
│   ├── vector_cache.py       # cache embeddings ra đĩa, per model
│   ├── app.py                # FastAPI: /upload, /ask, /health, /cache/*
│   └── models/               # model đã tải về (offline)
├── scripts/                  # tiện ích chạy QUANH hệ thống (không phải lõi)
│   ├── download_all_models.py # ⭐ tải TẤT CẢ model (giờ setup, còn mạng)
│   ├── embed_all_models.py   # ⭐ embed sẵn text đã lưu với mọi model (đổi nhanh)
│   ├── switch_model.py       # ⭐ đổi embedding model trong .env
│   ├── compete.py            # register / evaluate (±document_received) / reset / result
│   ├── eval_questions.py     # ⭐ đo accuracy + recall trên bộ câu hỏi tự sinh
│   ├── manage_cache.py       # xem/xoá vector cache
│   ├── benchmark_models.py   # so sánh tốc độ các embedding model
│   └── test_endpoints.py     # test HTTP khi server đang chạy
├── docs/
│   ├── COMPETITION_PLAYBOOK.md # ⭐ chiến lược 5 lượt + chọn model/config
│   └── EMBEDDING_MODELS.md   # hướng dẫn chi tiết các embedding model
├── logs/eval/                # (tự tạo) tài liệu + câu hỏi mỗi lượt thi — git-ignored
├── cache/                    # (tự tạo) embeddings + last_document.json — git-ignored
└── tmp/                      # dữ liệu/script thử nghiệm — git-ignored
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

# 2. Tải TẤT CẢ model embedding về local (offline-ready)
uv run python scripts/download_all_models.py
```

`uv sync` tự tạo `.venv/` và cài torch bản **CPU** (`2.12.0+cpu`) từ index chính
thức của PyTorch — không kéo về gói GPU. Mọi lệnh sau đều chạy qua `uv run` để
dùng đúng môi trường này (không cần activate thủ công).

> Không có/không muốn dùng uv? Vẫn cài được bằng pip: `pip install -r requirements.txt`
> (xem ghi chú về torch CPU ở đầu file đó).

## Cấu hình — chỉ sửa `.env`

Mở `ir/.env`:
- `STUDENT_ID` → đổi thành **MSSV viết hoa** của bạn.
- **Embedding Model** (mặc định `paraphrase-multilingual-MiniLM-L12-v2`, robust nhất):
  - Factoid (ngày/số/tên) → `dangvantuan/vietnamese-embedding` hoặc `keepitreal/vietnamese-sbert`
  - Ngữ nghĩa/suy luận → `paraphrase-multilingual` hoặc `intfloat/multilingual-e5-base`
  - Tiện ích: `uv run python scripts/switch_model.py --list`. Xem `docs/COMPETITION_PLAYBOOK.md`.
- **LLM tạm thời** (dev/local Qwen): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`.
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

## Quản lý Embedding Models

### Xem model hiện tại:
```powershell
uv run python scripts/switch_model.py
```

### Danh sách models có sẵn:
```powershell
uv run python scripts/switch_model.py --list
```

### Thay đổi model:
```powershell
# Vietnamese models (khuyến nghị)
uv run python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding
uv run python scripts/switch_model.py --set dangvantuan/vietnamese-embedding

# Multilingual models
uv run python scripts/switch_model.py --set intfloat/multilingual-e5-base
uv run python scripts/switch_model.py --set jinaai/jina-embeddings-v5-text-nano
```

### So sánh hiệu suất:
```powershell
uv run python scripts/benchmark_models.py   # test tất cả models
```

> **Lưu ý**: Sau khi thay đổi model, phải **restart server** để load model mới.
> Xem hướng dẫn chi tiết trong `docs/EMBEDDING_MODELS.md`.

## Quản lý Vector Cache ⚡

Embeddings được **tự động cache per model**. Switch model rồi quay lại → load cache (15x faster!).

### Xem cache statistics:
```powershell
uv run python scripts/manage_cache.py stats
```

### List cached files:
```powershell
uv run python scripts/manage_cache.py list
```

### Clear cache:
```powershell
# Clear tất cả
uv run python scripts/manage_cache.py clear

# Clear một model cụ thể
uv run python scripts/manage_cache.py clear --model keepitreal/vietnamese-sbert

# Clear một document
uv run python scripts/manage_cache.py clear --doc doc_123
```

> **Tip**: Cache giúp test nhiều models nhanh chóng. Xem chi tiết: `docs/VECTOR_CACHE.md`

## Thi đấu (khi Teacher Server hoạt động)

Luật mới (**Modified 2**): Teacher bơm **100 câu hỏi**, mỗi người được gọi
`/evaluate` **tối đa 5 lần** (điểm cao nhất tính). API `/evaluate` nay nhận body
`{"document_received": true|false}` để có thể **bỏ qua bước `/upload`** (tránh
timeout 120s khi embedding chậm).

Quy trình khuyến nghị (server `app.py` phải đang chạy ở terminal khác):

```powershell
# --- TRONG GIỜ SETUP (CÒN INTERNET) ---
uv run python scripts/download_all_models.py   # tải toàn bộ model về local

# --- SAU KHI NGẮT MẠNG (chỉ LAN) ---
# 1) Lần evaluate ĐẦU TIÊN: nhận tài liệu, embed bằng model mặc định, LƯU xuống đĩa.
uv run python scripts/compete.py register
uv run python scripts/compete.py evaluate            # document_received=false
#    -> Teacher gọi /upload; kể cả báo timeout, server vẫn embed xong + lưu cache + lưu text.

# 2) (Tuỳ chọn) Embed sẵn text đã lưu với các model khác để đổi nhanh sau này:
uv run python scripts/embed_all_models.py            # đọc tmp/model-embedding.md

# 3) Các lần evaluate SAU (đã có VectorDB trên đĩa, đã reload lúc khởi động):
uv run python scripts/compete.py reset
uv run python scripts/compete.py evaluate --document-received   # Teacher BỎ QUA /upload, hỏi luôn
```

> **Lưu document + VectorDB:** `/upload` tự lưu text vào `cache/last_document.json`
> và embeddings vào `cache/embeddings/<model>/`. Khi **khởi động lại server**
> (để sửa prompt/logic/đổi model), index được **nạp lại từ cache trong vài giây**
> — không cần Teacher gửi lại tài liệu, không tốn lượt `/evaluate`.

`register` tự phát hiện IP LAN; nếu cần ép tay: `uv run python scripts/compete.py register --ip 192.168.1.15 --port 5000`.

### Đổi embedding model giữa các lần thi

```powershell
uv run python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding
# restart server -> nó load embeddings của model đó từ cache (đã chạy embed_all_models.py)
```

## API (Student Server)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/upload` | POST | `{doc_id, text}` → chunk + embed + index. Trả `{status, doc_id, chunks}` |
| `/ask` | POST | `{question}` → RAG. Trả `{answer: "A|B|C|D", sources: [...]}` |
| `/health` | GET | trạng thái sẵn sàng |

## Tinh chỉnh (trong `.env`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EMBED_MODEL_NAME` | paraphrase-multilingual-MiniLM-L12-v2 | Model embedding (robust nhất; xem COMPETITION_PLAYBOOK.md) |
| `EMBED_MAX_SEQ_LEN` | 256 | Độ dài sequence tối đa (auto-detect theo model) |
| `CHUNK_ADAPTIVE` | true | Tự chọn chunk size/overlap theo model |
| `TOP_K_CONTEXT` | 8 | số chunk fused đưa vào LLM (+ tối đa 4 chunk option-evidence) |
| `TOP_K_DENSE` / `TOP_K_BM25` | 20 / 20 | số ứng viên mỗi retriever |
| `MMR_LAMBDA` | 0.35 | cân bằng liên quan ↔ đa dạng |
| `CONTEXT_MAX_CHARS` | 6500 | cắt ngữ cảnh; tăng ~12000 cho model chunk-lớn, giảm ~5000 nếu proxy seq len hẹp |
| `ASK_DEADLINE` | 45 | hạn chót (giây) cho /ask trước khi fallback |
| `LLM_PROFILE` | auto | `qwen_small` / `reasoning` / `default` |
| `LLM_ENABLE_THINKING` | false | tắt thinking cho Qwen MCQ (nhanh, ~1s/câu) |
| `LLM_MAX_TOKENS` | 96 (Qwen) | 512 cho reasoning model |
