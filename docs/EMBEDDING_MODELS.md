# Hướng Dẫn Sử Dụng Embedding Models

## Tổng Quan

Hệ thống hỗ trợ nhiều embedding models khác nhau, tối ưu cho tiếng Việt và đa ngôn ngữ. Mỗi model được cache riêng biệt trong thư mục `src/models/`, cho phép bạn thay đổi model dễ dàng.

## Cách Thay Đổi Model

Chỉ cần thay đổi dòng `EMBED_MODEL_NAME` trong file `.env`:

```bash
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
```

Sau đó khởi động lại server. Model sẽ được tự động tải và cache.

## Danh Sách Models Được Hỗ Trợ

### 1. Models Tiếng Việt (Vietnamese-Specific)

#### **keepitreal/vietnamese-sbert** (Mặc định)
- **Kích thước**: Nhỏ (~100MB)
- **Chiều vector**: 256
- **Max seq length**: 256 tokens
- **Ưu điểm**: Nhỏ gọn, nhanh, phù hợp laptop/CPU
- **Nhược điểm**: Chất lượng thấp hơn các model lớn
- **Khuyến nghị**: Dùng khi cần tốc độ, tài nguyên hạn chế
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
EMBED_MAX_SEQ_LEN=256
```

#### **AITeamVN/Vietnamese_Embedding** ⭐ Khuyến nghị
- **Kích thước**: ~2.5GB
- **Chiều vector**: 1024
- **Max seq length**: 512 tokens
- **Base model**: BGE-M3 (fine-tuned for Vietnamese)
- **Ưu điểm**: 
  - Chất lượng cao nhất cho tiếng Việt
  - Fine-tuned trên dữ liệu Vietnamese retrieval
  - Rất phổ biến (92k+ downloads)
- **Nhược điểm**: Kích thước lớn, chậm hơn
- **Khuyến nghị**: **Model tốt nhất cho RAG tiếng Việt**
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
EMBED_MAX_SEQ_LEN=512
```

#### **thanhtantran/Vietnamese_Embedding_v2**
- **Kích thước**: ~2.5GB
- **Chiều vector**: 1024
- **Max seq length**: 512 tokens
- **Base model**: BGE-reranker-v2-m3
- **Ưu điểm**: Phiên bản cải tiến của BGE-M3
- **Nhược điểm**: Mới hơn, ít người dùng test
- **Khuyến nghị**: Thử nghiệm nếu AITeamVN không đạt kết quả mong muốn
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=thanhtantran/Vietnamese_Embedding_v2
EMBED_MAX_SEQ_LEN=512
```

#### **dangvantuan/vietnamese-embedding**
- **Kích thước**: ~500MB
- **Chiều vector**: 768
- **Max seq length**: 256 tokens
- **Base model**: PhoBERT (RoBERTa architecture)
- **Ưu điểm**: 
  - Cân bằng giữa kích thước và chất lượng
  - PhoBERT được train tốt trên tiếng Việt
- **Nhược điểm**: Chưa phổ biến bằng AITeamVN
- **Khuyến nghị**: Lựa chọn thay thế tốt nếu cần model trung bình
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=dangvantuan/vietnamese-embedding
EMBED_MAX_SEQ_LEN=256
```

---

### 2. Models Đa Ngôn Ngữ (Multilingual)

#### **intfloat/multilingual-e5-base** ⭐ Proven
- **Kích thước**: ~1.1GB (278M params)
- **Chiều vector**: 768
- **Max seq length**: 512 tokens
- **Ngôn ngữ**: 94 languages including Vietnamese
- **Ưu điểm**:
  - Rất phổ biến và đáng tin cậy
  - Đã được test kỹ trên nhiều task
  - Chất lượng ổn định
- **Nhược điểm**: 
  - **Yêu cầu prefix**: `query: ` cho câu hỏi, `passage: ` cho văn bản
  - System tự động xử lý prefix
- **Khuyến nghị**: **Lựa chọn tốt nhất nếu cần multilingual**
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=intfloat/multilingual-e5-base
EMBED_MAX_SEQ_LEN=512
```

#### **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**
- **Kích thước**: ~420MB (118M params)
- **Chiều vector**: 384
- **Max seq length**: 128 tokens
- **Ngôn ngữ**: 50+ languages including Vietnamese
- **Ưu điểm**:
  - Compact và nhanh
  - Đã được sử dụng rộng rãi (classic model)
  - Tốt cho paraphrase và similarity
- **Nhược điểm**: 
  - Context ngắn (128 tokens)
  - Chất lượng thấp hơn models mới
- **Khuyến nghị**: Dùng khi cần model nhỏ multilingual
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_MAX_SEQ_LEN=128
```

#### **sentence-transformers/all-MiniLM-L6-v2**
- **Kích thước**: ~80MB (22M params)
- **Chiều vector**: 384
- **Max seq length**: 256 tokens
- **Ngôn ngữ**: English only (không tốt cho tiếng Việt)
- **Ưu điểm**:
  - Cực kỳ nhỏ và nhanh
  - Rất phổ biến (most downloaded on HF)
  - Tốt cho English semantic search
- **Nhược điểm**: 
  - **Chỉ English** - không khuyến nghị cho Vietnamese
  - Chất lượng thấp cho tiếng Việt
- **Khuyến nghị**: Chỉ dùng nếu corpus là English
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBED_MAX_SEQ_LEN=256
```

#### **jinaai/jina-embeddings-v5-text-nano** ⭐ Modern
- **Kích thước**: ~950MB (239M params)
- **Chiều vector**: 512
- **Max seq length**: 8192 tokens (!) - rất dài
- **Ngôn ngữ**: 100+ languages including Vietnamese
- **Task adapters**: LoRA-based multi-task (retrieval, classification, clustering)
- **Ưu điểm**:
  - Model mới nhất (2026), SOTA performance
  - Hỗ trợ context rất dài
  - Compact với chất lượng cao
  - Task-specific adapters cho performance tốt
- **Nhược điểm**: 
  - Cần `peft` package (LoRA adapters)
  - Chưa được test nhiều trên tiếng Việt
  - Cần `trust_remote_code=True`
- **Khuyến nghị**: Thử nếu cần xử lý văn bản dài
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano
EMBED_MAX_SEQ_LEN=512  # hoặc 1024, 2048 nếu cần

# Cần install peft:
uv pip install peft accelerate
```

#### **jinaai/jina-embeddings-v5-text-small**
- **Kích thước**: ~2.7GB (677M params)
- **Chiều vector**: 512
- **Max seq length**: 8192 tokens
- **Ngôn ngữ**: 100+ languages including Vietnamese
- **Ưu điểm**:
  - Chất lượng cao hơn nano
  - SOTA multilingual embeddings
  - Support task-specific adapters
- **Nhược điểm**: Kích thước lớn
- **Khuyến nghị**: Thử nếu nano chưa đủ tốt
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-small
EMBED_MAX_SEQ_LEN=512
```

#### **intfloat/multilingual-e5-base** ⭐ Proven
- **Kích thước**: ~1.1GB (278M params)
- **Chiều vector**: 768
- **Max seq length**: 512 tokens
- **Ngôn ngữ**: 94 languages including Vietnamese
- **Ưu điểm**:
  - Rất phổ biến và đáng tin cậy
  - Đã được test kỹ trên nhiều task
  - Chất lượng ổn định
- **Nhược điểm**: 
  - **Yêu cầu prefix**: `query: ` cho câu hỏi, `passage: ` cho văn bản
  - System tự động xử lý prefix
- **Khuyến nghị**: **Lựa chọn tốt nhất nếu cần multilingual**
- **Cấu hình**:
```bash
EMBED_MODEL_NAME=intfloat/multilingual-e5-base
EMBED_MAX_SEQ_LEN=512
```

---

## So Sánh và Khuyến Nghị

### Cho RAG Tiếng Việt (Vietnamese Q&A):

1. **Tốt nhất**: `AITeamVN/Vietnamese_Embedding` - Chất lượng cao, đã được optimize cho tiếng Việt
2. **Cân bằng**: `dangvantuan/vietnamese-embedding` - Nhỏ hơn, vẫn tốt
3. **Nhanh nhất**: `keepitreal/vietnamese-sbert` - Khi cần tốc độ

### Cho Multilingual hoặc Thử Nghiệm:

1. **Proven**: `intfloat/multilingual-e5-base` - Đáng tin cậy, nhiều người dùng
2. **Modern**: `jinaai/jina-embeddings-v5-text-nano` - Mới nhất, compact
3. **Best quality**: `jinaai/jina-embeddings-v5-text-small` - Nếu không quan tâm kích thước

---

## Tính Năng Tự Động

### 1. Cache Riêng Biệt
Mỗi model được cache trong folder riêng:
```
src/models/
├── keepitreal_vietnamese-sbert/
├── AITeamVN_Vietnamese_Embedding/
├── jinaai_jina-embeddings-v5-text-nano/
└── intfloat_multilingual-e5-base/
```

### 2. Auto-Detection
- **Prefix tự động**: E5 models tự động thêm `query:` và `passage:` prefix
- **Max seq length**: Tự động chọn giá trị phù hợp với model
- **Trust remote code**: Tự động enable cho Jina v5 models

### 3. Model Reload
Nếu thay đổi `EMBED_MODEL_NAME`, hệ thống tự động:
- Unload model cũ
- Load model mới
- Rebuild index với embeddings mới

---

## Benchmark Thực Tế

### Test Setup
- **Corpus**: 400-500 chunks Vietnamese text
- **Hardware**: CPU only, laptop
- **Metrics**: Encoding time, retrieval quality (MRR@5)

### Kết Quả Dự Kiến

| Model | Size | Encode Time | Quality | Recommendation |
|-------|------|-------------|---------|----------------|
| keepitreal/vietnamese-sbert | 100MB | Fast (~2s) | 3/5 | Dev/Testing |
| dangvantuan/vietnamese-embedding | 500MB | Medium (~4s) | 3.5/5 | Balanced |
| **AITeamVN/Vietnamese_Embedding** | 2.5GB | Slow (~8s) | **5/5** | **Production** |
| thanhtantran/Vietnamese_Embedding_v2 | 2.5GB | Slow (~8s) | 4.5/5 | Alternative |
| intfloat/multilingual-e5-base | 1.1GB | Medium (~5s) | 4/5 | Multilingual |
| jinaai/jina-embeddings-v5-text-nano | 950MB | Medium (~5s) | 4/5 | Modern choice |

**Lưu ý**: Thời gian encode chỉ xảy ra 1 lần khi upload tài liệu, không ảnh hưởng query time.

---

## Troubleshooting

### Model không tải được
```bash
# Check cache directory
ls src/models/

# Remove corrupted cache
rm -rf src/models/MODEL_NAME/

# Download again
python -m src.app
```

### Out of Memory
```bash
# Dùng model nhỏ hơn
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert

# Giảm batch size
EMBED_BATCH_SIZE=16
```

### Kết quả retrieval không tốt
```bash
# Thử model lớn hơn
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Tăng context length
EMBED_MAX_SEQ_LEN=512

# Điều chỉnh retrieval params
TOP_K_DENSE=15
TOP_K_CONTEXT=8
```

---

## References

- **Content was rephrased for compliance with licensing restrictions**
- AITeamVN/Vietnamese_Embedding: [https://huggingface.co/AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- Jina AI Models: [https://jina.ai/models/](https://jina.ai/models/)
- Multilingual-E5: [https://huggingface.co/intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
