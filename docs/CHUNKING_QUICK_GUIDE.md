# 🔪 Chunking Quick Guide

## TL;DR

**Adaptive chunking đã được enable mặc định!** Hệ thống tự động tối ưu chunk size dựa trên embedding model.

```bash
# Trong .env
CHUNK_ADAPTIVE=true  # ✅ Khuyến nghị (mặc định)
```

---

## 📊 Adaptive Chunking

### Model nhỏ (256 tokens)
```
keepitreal/vietnamese-sbert
dangvantuan/vietnamese-embedding

→ Chunk: 160 words, Overlap: 45 (27%)
```

### Model vừa (512 tokens)
```
AITeamVN/Vietnamese_Embedding
thanhtantran/Vietnamese_Embedding_v2
intfloat/multilingual-e5-base

→ Chunk: 320 words, Overlap: 85 (27%)
```

### Model lớn (8192 tokens)
```
jinaai/jina-embeddings-v5-text-nano
jinaai/jina-embeddings-v5-text-small

→ Chunk: 600 words, Overlap: 160 (27%)
```

---

## 🎛️ Manual Override

Nếu muốn tự điều chỉnh:

```bash
# Tắt adaptive
CHUNK_ADAPTIVE=false

# Set thủ công
CHUNK_SIZE_WORDS=200
CHUNK_OVERLAP_WORDS=50
```

---

## 🧪 Test Chunking

```bash
# Test với config hiện tại
python scripts/test_chunking.py

# So sánh các strategies
python scripts/test_chunking.py --compare

# Test với model cụ thể
python scripts/test_chunking.py --model AITeamVN/Vietnamese_Embedding

# Test với file riêng
python scripts/test_chunking.py --text path/to/document.txt
```

---

## 📐 Công Thức Tối Ưu

### Chunk Size
```
optimal_chunk_size = (model_max_tokens * 0.9) / 1.4

Trong đó:
- model_max_tokens: 256, 512, hoặc 8192
- 0.9: buffer 10% cho tokenizer overhead  
- 1.4: trung bình token/word cho tiếng Việt
```

### Overlap Ratio
```
overlap_ratio = 0.27 (27%)

Tại sao 27%?
- < 20%: Context bị mất
- 20-25%: Acceptable
- 25-30%: Optimal (theo research)
- > 30%: Redundant, tốn thời gian
```

---

## ✅ Best Practices

### 1. Dùng Adaptive (Khuyến nghị)
```bash
CHUNK_ADAPTIVE=true
```
✅ Tự động tối ưu cho model  
✅ Không cần điều chỉnh thủ công  
✅ Dễ switch model

### 2. Override khi cần
```bash
# Model lớn nhưng muốn chunks nhỏ (precision > overview)
CHUNK_ADAPTIVE=false
CHUNK_SIZE_WORDS=150

# Model nhỏ nhưng muốn chunks lớn (overview > precision)
CHUNK_ADAPTIVE=false
CHUNK_SIZE_WORDS=250
```

### 3. Tăng overlap nếu
- Corpus có nhiều cross-references
- Cần context preservation cao
- Queries phức tạp

```bash
CHUNK_OVERLAP_WORDS=60  # Tăng từ 40 lên 60
```

---

## 🔍 Troubleshooting

### Chunks quá nhỏ
```bash
# Kiểm tra
python scripts/test_chunking.py

# Nếu avg_words < 100, tăng lên
CHUNK_SIZE_WORDS=200
```

### Chunks quá lớn
```bash
# Nếu avg_words > 400 với model nhỏ
CHUNK_SIZE_WORDS=150
```

### Context bị mất
```bash
# Tăng overlap
CHUNK_OVERLAP_WORDS=60

# Hoặc tăng overlap ratio
overlap = chunk_size * 0.30
```

### Retrieval không tốt
```bash
# Thử chunks lớn hơn (overview)
CHUNK_SIZE_WORDS=300

# Hoặc chunks nhỏ hơn (precision)
CHUNK_SIZE_WORDS=120
```

---

## 📈 Performance Impact

| Config | Chunks | Retrieval Time | Quality |
|--------|--------|----------------|---------|
| Small (120w, 30o) | More | Slower | High Precision |
| **Adaptive (auto)** | **Optimal** | **Balanced** | **Recommended** |
| Large (400w, 100o) | Fewer | Faster | High Recall |

---

## 🎯 Recommendations by Use Case

### Academic Papers / Technical Docs
```bash
# Cần precision cao
CHUNK_SIZE_WORDS=160
CHUNK_OVERLAP_WORDS=50
```

### General Knowledge / FAQs
```bash
# Cân bằng
CHUNK_ADAPTIVE=true  # Dùng adaptive
```

### Long-form Content / Books
```bash
# Cần overview tốt
CHUNK_SIZE_WORDS=400
CHUNK_OVERLAP_WORDS=100
```

---

## 📚 Xem Thêm

- Chi tiết: [CHUNKING_STRATEGY.md](./CHUNKING_STRATEGY.md)
- Embedding models: [EMBEDDING_MODELS.md](./EMBEDDING_MODELS.md)
