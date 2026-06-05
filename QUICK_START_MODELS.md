# 🚀 Quick Start: Embedding Models

## Thay Đổi Model Trong 3 Bước

### Cách 1: Dùng Script (Khuyến nghị)

```bash
# Bước 1: Xem danh sách models
python scripts/switch_model.py --list

# Bước 2: Chọn model
python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding

# Bước 3: Restart server
python src/app.py
```

### Cách 2: Thủ Công

```bash
# Bước 1: Mở .env
notepad .env

# Bước 2: Sửa dòng này
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Bước 3: Restart server
python src/app.py
```

---

## 🎯 Chọn Model Nào?

### Cho Vietnamese RAG (Khuyến nghị)
```bash
# Tốt nhất (chất lượng cao)
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Cân bằng (vừa tốc độ vừa chất lượng)  
EMBED_MODEL_NAME=dangvantuan/vietnamese-embedding

# Nhanh nhất (dev/test)
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
```

### Cho Multilingual
```bash
# Đáng tin cậy nhất
EMBED_MODEL_NAME=intfloat/multilingual-e5-base

# Mới nhất, compact
EMBED_MODEL_NAME=jinaai/jina-embeddings-v5-text-nano
```

---

## 📊 So Sánh Nhanh

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **AITeamVN/Vietnamese_Embedding** | ⬛⬛⬛ | ⬛⬜⬜ | ⭐⭐⭐⭐⭐ | **Production** |
| dangvantuan/vietnamese-embedding | ⬛⬛⬜ | ⬛⬛⬜ | ⭐⭐⭐⭐ | Balanced |
| keepitreal/vietnamese-sbert | ⬛⬜⬜ | ⬛⬛⬛ | ⭐⭐⭐ | Dev/Test |
| intfloat/multilingual-e5-base | ⬛⬛⬜ | ⬛⬛⬜ | ⭐⭐⭐⭐ | Multilingual |
| jinaai/jina-embeddings-v5-text-nano | ⬛⬛⬜ | ⬛⬛⬜ | ⭐⭐⭐⭐ | Modern |

---

## 🔍 Test Models

```bash
# Xem model hiện tại
python scripts/switch_model.py

# Benchmark tất cả models
python scripts/benchmark_models.py

# Test với server
python scripts/selftest.py
```

---

## 📝 Models Cache

Mỗi model tự động cache riêng:
```
src/models/
├── keepitreal_vietnamese-sbert/      (100 MB)
├── AITeamVN_Vietnamese_Embedding/    (2.5 GB)
├── dangvantuan_vietnamese-embedding/ (500 MB)
└── intfloat_multilingual-e5-base/    (1.1 GB)
```

Xóa cache nếu cần:
```bash
# Xóa một model
rm -rf src/models/AITeamVN_Vietnamese_Embedding/

# Xóa tất cả
rm -rf src/models/*/
```

---

## 🆘 Troubleshooting

### Model không tải được?
```bash
rm -rf src/models/MODEL_NAME/
python src/app.py  # Tải lại
```

### Out of Memory?
```bash
# Dùng model nhỏ hơn
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
```

### Kết quả không tốt?
```bash
# Thử model tốt hơn
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
```

---

## 📚 Xem Thêm

- Chi tiết đầy đủ: `docs/EMBEDDING_MODELS.md`
- Thay đổi update: `docs/MULTI_MODEL_UPDATE.md`
- Main README: `README.md`

---

## ⚡ TL;DR

**Muốn kết quả tốt nhất?**
```bash
python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding
python src/app.py
```

Done! 🎉
