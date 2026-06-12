# Competition Playbook — đạt accuracy cao khi KHÔNG biết trước đề

Bạn chỉ có **5 lượt `/evaluate`** và không biết domain/độ khó câu hỏi. Chiến lược
cốt lõi: dùng **lượt 1 để thu thập đề thật qua log**, rồi **lượt 2–5 chỉnh model/
config theo đúng loại câu hỏi** đã thấy. Đổi model = chỉ cần đổi `.env` + restart
(index nạp lại từ cache trong vài giây nhờ vector cache + docstore).

## 1. Kết quả nghiên cứu (vì sao chọn cấu hình này)

Test trên 380 câu trắc nghiệm tự sinh (grounded) từ dữ liệu PTIT, với pipeline đã
nâng cấp (option-evidence retrieval). Khi đáp án nằm trong context, LLM trả đúng
~97% → **nút thắt là retrieval recall**, đã giải quyết bằng *option-evidence*
(đảm bảo đưa vào context chunk khớp BM25 chính xác nhất cho TỪNG đáp án A/B/C/D).

So sánh model (file factoid, nhiều ngày/số):

| Model | chunk | Accuracy | Tốc độ | Ghi chú |
|---|---|---|---|---|
| dangvantuan/vietnamese-embedding | ~150 từ | 98.8% | 1.3s | tốt nhất cho factoid |
| keepitreal/vietnamese-sbert | ~160 từ | 96.2% | 1.3s | |
| **paraphrase-multilingual-MiniLM-L12-v2** | ~80 từ | 93.8% | 0.9s | **mặc định, robust nhất** |
| intfloat/multilingual-e5-base | ~320 từ | 81.2% | 1.9s | chunk lớn kém định vị |
| AITeamVN/Vietnamese_Embedding | ~320 từ | 80.0% | 3.0s | nặng, chậm |

Tổng accuracy với paraphrase: **91.8%** (349/380), mọi file 90–94%.

**Mâu thuẫn quan trọng**: trên đề thi THẬT (câu hỏi tự nhiên, ngữ nghĩa) của bạn:
paraphrase 10/10, e5 9/10, keepitreal 6/10 — tức câu *ngữ nghĩa* thì model
đa-ngôn-ngữ tốt hơn, còn câu *factoid* thì model chunk nhỏ tốt hơn. Vì chưa biết
đề → chọn model **robust nhất cho cả hai**.

## 2. Cấu hình mặc định khuyến nghị (đã set sẵn trong `.env`)

```dotenv
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOP_K_DENSE=20
TOP_K_BM25=20
TOP_K_CONTEXT=8
CONTEXT_MAX_CHARS=6500          # hợp model chunk nhỏ; tăng nếu đổi sang model chunk lớn
CONTEXT_MAX_CHUNK_CHARS=850
ASK_DEADLINE=45
LLM_ENABLE_THINKING=false       # Qwen MCQ: tắt thinking cho nhanh
LLM_MAX_TOKENS=96
```

Lý do chọn **paraphrase-multilingual** làm mặc định: robust nhất across question
types (10/10 trên đề thật + 93.8% trên factoid), **nhanh nhất** (0.9s/câu), nhẹ.

## 3. Quy trình 5 lượt (data-driven)

### Giờ setup (còn Internet)
```powershell
uv run python scripts/download_all_models.py     # tải mọi model
```

### Lượt 1 — thu thập đề (BẮT BUỘC document_received=false)
```powershell
# (server app.py đang chạy ở terminal khác)
uv run python scripts/compete.py register
uv run python scripts/compete.py evaluate         # Teacher gửi /upload + 100 câu
```
- Server embed bằng paraphrase, **lưu doc + 100 câu hỏi vào `logs/eval/`**.
- Ngay sau lượt 1, pre-embed các model khác để đổi nhanh:
```powershell
uv run python scripts/embed_all_models.py         # điền cache cho mọi model
```

### Phân tích log (mấu chốt)
Mở `logs/eval/<round>__questions.jsonl` — bạn THẤY 100 câu hỏi thật + đáp án hệ
thống đã trả. Tự đánh giá loại câu hỏi:
- **Factoid** (ngày/số/tên/email/quyết định) → đổi sang `dangvantuan/vietnamese-embedding`
  hoặc `keepitreal/vietnamese-sbert` (chunk nhỏ, ~96–99% factoid).
- **Ngữ nghĩa/suy luận/diễn giải** → giữ `paraphrase-multilingual` hoặc thử
  `intfloat/multilingual-e5-base`.
- **Hỗn hợp** → giữ paraphrase (robust nhất).

### Lượt 2–5 — tinh chỉnh (document_received=true, KHÔNG tốn embed)
```powershell
uv run python scripts/switch_model.py --set dangvantuan/vietnamese-embedding
# nếu đổi sang model chunk lớn (e5/AITeamVN): đặt CONTEXT_MAX_CHARS=12000 trong .env
# restart server  ->  index nạp lại từ cache (vài giây)
uv run python scripts/compete.py reset
uv run python scripts/compete.py evaluate --document-received
```
Lặp lại, chọn lượt cho điểm cao nhất (điểm cao nhất được tính).

## 4. Các lever chỉnh tay khi điểm chưa đạt

| Triệu chứng (từ log) | Chỉnh |
|---|---|
| Câu factoid sai, đáp án không có trong `sources` | tăng `TOP_K_CONTEXT` (10–12), đổi model chunk nhỏ |
| Đổi sang model chunk lớn (e5/AITeamVN) | tăng `CONTEXT_MAX_CHARS` (9000–12000) |
| Proxy LLM giới hạn ~2048 token | giảm `CONTEXT_MAX_CHARS` (~4500–5000) |
| Câu suy luận sai | giữ paraphrase/e5; cân nhắc nới prompt cho phép suy luận ngắn |
| Quá hạn 60s | giảm `ASK_DEADLINE`, giảm `TOP_K_CONTEXT` |

## 5. Tính linh hoạt sẵn có (domain-agnostic)

- **Hybrid BM25 + dense + RRF + MMR**: lexical lo factoid/đúng chữ; dense lo ngữ nghĩa.
- **Option-evidence**: đảm bảo bằng chứng cho từng đáp án — giúp mọi domain.
- **Chuẩn hoá ngày/số/mã** trong BM25: structure-agnostic.
- **Adaptive chunking**: tự điều chỉnh theo model.
- **Luôn trả về A/B/C/D** kể cả khi LLM lỗi (fallback embedding).
- **Persistent cache + docstore**: đổi model/restart không cần Teacher gửi lại.
