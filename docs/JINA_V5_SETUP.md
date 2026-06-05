# Jina Embeddings v5 - Setup Guide

## Giới Thiệu

Jina Embeddings v5 là model embedding mới nhất (2026) với nhiều tính năng tiên tiến:
- **Multi-task LoRA adapters**: Tối ưu cho retrieval, classification, clustering
- **Ultra-long context**: Hỗ trợ lên đến 8192 tokens
- **100+ languages**: Bao gồm tiếng Việt
- **SOTA performance**: Top performance trong class 239M-677M params

## Yêu Cầu Đặc Biệt

### 1. Package Dependencies

Jina v5 cần `peft` (Parameter-Efficient Fine-Tuning) cho LoRA adapters:

```bash
uv pip install peft accelerate
```

Hoặc nếu dùng pip:
```bash
pip install peft accelerate
```

### 2. Trust Remote Code

Jina v5 dùng custom modeling code, cần enable `trust_remote_code=True`.

**Hệ thống đã tự động enable!** Không cần config thêm.

### 3. Task Types

Khác với models khác dùng prefix (như E5), Jina v5 dùng `task` parameter:

```python
# Query embedding
model.encode(texts, task="retrieval.query")

# Passage embedding  
model.encode(texts, task="retrieval.passage")
```

**Hệ thống đã tự động xử lý!** Code sẽ:
- Detect Jina v5 models
- Auto-apply đúng task type
- Query → `retrieval.query`
- Passage → `retrieval.passage`

## Cài Đặt

### Bước 1: Install Dependencies

```bash
# Trong thư mục ir/
uv pip install peft accelerate
```

Output mong đợi:
```
Installed 3 packages in XXXms
 + accelerate==1.13.0
 + peft==0.19.1
 + psutil==7.2.2
```

### Bước 2: Configure Model

Sửa `.env`:

```bash
# Jina v5 nano (239M params, fast)
EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano

# Hoặc Jina v5 small (677M params, better quality)
# EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-small

# Context length (có thể tăng lên 8192)
EMBED_MAX_SEQ_LEN=512
```

### Bước 3: Restart Server

```bash
python src/app.py
```

Output mong đợi:
```
[embedder] Loading model from Hugging Face Hub: jinaai/jina-embeddings-v5-text-nano
[embedder] Cache directory: src/models/jinaai_jina-embeddings-v5-text-nano
[embedder] Ready. model=jinaai/jina-embeddings-v5-text-nano
[embedder] dim=512, max_seq_len=512
[embedder] Using task types: query='retrieval.query' passage='retrieval.passage'
```

## Troubleshooting

### Error: "No module named 'peft'"

**Nguyên nhân:** Chưa install peft package.

**Giải pháp:**
```bash
uv pip install peft accelerate
```

### Error: "Task must be specified before encoding"

**Nguyên nhân:** Model cần task parameter nhưng code chưa pass.

**Giải pháp:** ✅ **ĐÃ FIX!** Code giờ tự động detect Jina v5 và apply task.

Nếu vẫn lỗi:
1. Check model name trong `.env`:
   ```bash
   EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano
   ```
2. Restart server
3. Check log output có dòng "Using task types"

### Error: "trust_remote_code"

**Nguyên nhân:** Jina v5 cần custom code.

**Giải pháp:** ✅ **ĐÃ FIX!** Code tự động enable cho Jina v5.

### Download Chậm

**Nguyên nhân:** Model ~950MB-2.7GB, download lần đầu mất thời gian.

**Giải pháp:**
1. Kiên nhẫn đợi download hoàn tất
2. Lần sau sẽ load từ cache (fast!)
3. Check cache: `ls src/models/jinaai_jina-embeddings-v5-text-nano/`

### Out of Memory

**Nguyên nhân:** Jina v5 small (677M) lớn hơn models khác.

**Giải pháp:**
```bash
# Dùng nano thay vì small
EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano

# Giảm batch size
EMBED_BATCH_SIZE=16
```

## Performance Tuning

### Context Length

Jina v5 hỗ trợ lên đến 8192 tokens:

```bash
# Short documents (fast)
EMBED_MAX_SEQ_LEN=512

# Medium documents
EMBED_MAX_SEQ_LEN=1024

# Long documents  
EMBED_MAX_SEQ_LEN=2048

# Very long (research papers, books)
EMBED_MAX_SEQ_LEN=4096
```

**Trade-off:**
- Longer → Better context, slower
- Shorter → Faster, may lose context

### Batch Size

```bash
# Fast machine
EMBED_BATCH_SIZE=32

# Normal
EMBED_BATCH_SIZE=16

# Low memory
EMBED_BATCH_SIZE=8
```

### Chunking Strategy

Với Jina v5's ultra-long context:

```bash
# Enable adaptive chunking
CHUNK_ADAPTIVE=true

# Hoặc manual (larger chunks)
CHUNK_ADAPTIVE=false
CHUNK_SIZE_WORDS=600  # Tận dụng long context
CHUNK_OVERLAP_WORDS=160
```

## Comparison với Models Khác

| Feature | Jina v5 nano | Vietnamese SBERT | E5-base |
|---------|--------------|------------------|---------|
| Params | 239M | ~100M | 278M |
| Max context | **8192** | 256 | 512 |
| Task adapters | ✅ LoRA | ❌ | ❌ |
| Multilingual | ✅ 100+ | ❌ (VN only) | ✅ 94 |
| Setup complexity | Medium | Easy | Easy |
| Speed | Medium | Fast | Medium |

## Best Use Cases

### ✅ Khi Nào Dùng Jina v5

1. **Long documents**: Papers, books, technical docs
2. **Multilingual**: Cần support nhiều ngôn ngữ
3. **Latest tech**: Muốn dùng SOTA 2026
4. **Quality focus**: Chất lượng quan trọng hơn tốc độ

### ❌ Khi Nào KHÔNG Dùng

1. **Short texts**: SMS, tweets (overkill)
2. **Speed critical**: Cần < 1s response
3. **Limited resources**: RAM < 4GB
4. **Production tested**: Cần model đã được test kỹ

## Example Usage

```bash
# 1. Install
uv pip install peft accelerate

# 2. Configure
echo 'EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano' >> .env

# 3. Run
python src/app.py

# 4. Upload document
curl -X POST http://localhost:5000/upload \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "test", "text": "Long document..."}'

# 5. Query
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## Advanced: Task Types

Jina v5 hỗ trợ nhiều task types:

### Retrieval (Default)
```python
# Query: retrieval.query
# Passage: retrieval.passage
```

### Text Matching
```python
task="text-matching"  # For similarity tasks
```

### Classification
```python
task="classification"  # For text classification
```

### Clustering
```python
task="clustering"  # For document clustering
```

**Note:** Hiện tại code chỉ support retrieval tasks. Các task khác có thể add sau.

## References

**Content was rephrased for compliance with licensing restrictions**

- [Jina AI Models](https://jina.ai/models/)
- [Jina v5 Announcement](https://jina.ai/news/)
- [HuggingFace: jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
- [Paper: Task-Targeted Embedding Distillation](https://arxiv.org/abs/2602.15547)

## Summary

**Setup checklist:**
- ✅ Install: `uv pip install peft accelerate`
- ✅ Configure: `EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano`
- ✅ Restart server
- ✅ Check log: "Using task types"
- ✅ Test upload + query

**Benefits:**
- 🚀 SOTA 2026 performance
- 📚 Ultra-long context (8192 tokens)
- 🌍 100+ languages
- 🎯 Task-specific optimization

**Gotchas:**
- ⚠️ Needs `peft` package
- ⚠️ Slower than small models
- ⚠️ Larger download size

Enjoy the cutting-edge embeddings! 🎉
