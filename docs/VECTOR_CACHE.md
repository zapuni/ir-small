# Vector Embeddings Cache

## Tổng Quan

Hệ thống giờ **tự động cache embeddings** cho mỗi document. Khi bạn:
1. Upload document lần đầu → Embed và cache
2. Switch sang model khác → Embed lại và cache riêng
3. Switch về model cũ → **Load từ cache**, không embed lại! ⚡

## Cấu Trúc Cache

```
cache/
├── embeddings/
│   ├── keepitreal_vietnamese-sbert/
│   │   ├── doc_123_abc1234.npz      # Document 123
│   │   └── doc_456_def5678.npz      # Document 456
│   ├── AITeamVN_Vietnamese_Embedding/
│   │   ├── doc_123_abc1234.npz      # Same doc, different embedding
│   │   └── doc_456_def5678.npz
│   └── intfloat_multilingual-e5-base/
│       └── doc_123_abc1234.npz
```

**Mỗi file .npz chứa:**
- `chunks`: Array các text chunks
- `embeddings`: Numpy array (N, embedding_dim)
- `metadata`: Thông tin (model, timestamp, config, etc.)

## Cách Hoạt Động

### 1. Upload Document (Lần Đầu)

```python
POST /upload
{
    "doc_id": "doc_123",
    "text": "..."
}
```

**Flow:**
1. Check cache: Không có
2. ⚙️ Chunk text (180 words, 40 overlap)
3. ⚙️ Embed chunks (mất 5-10s)
4. 💾 Save to cache: `cache/embeddings/MODEL_NAME/doc_123_HASH.npz`
5. Build BM25 index
6. Return response

### 2. Switch Model & Re-upload

```bash
# Switch to different model
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Restart server
python src/app.py
```

```python
POST /upload  # Same doc_id
{
    "doc_id": "doc_123",
    "text": "..."  # Same text
}
```

**Flow:**
1. Check cache: Không có (model khác)
2. ⚙️ Embed lại với model mới (mất 8-12s)
3. 💾 Save to new cache: `cache/embeddings/AITeamVN_Vietnamese_Embedding/doc_123_HASH.npz`
4. Return response

**Kết quả:** Giờ có 2 cache files, mỗi model 1 file!

### 3. Switch Về Model Cũ

```bash
# Switch back
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert

# Restart server
python src/app.py
```

```python
POST /upload  # Same doc_id, same text
{
    "doc_id": "doc_123",
    "text": "..."
}
```

**Flow:**
1. ✅ Check cache: **Có rồi!**
2. 📂 Load from cache (< 1s)
3. Skip embedding! ⚡
4. Build BM25 index
5. Return response

**Tiết kiệm:** 5-10 giây embedding time!

## Cache Key Strategy

Cache key = `doc_id` + `text_hash`

```python
doc_id_123_abc1234.npz
        ↑       ↑
     doc_id   text hash (16 chars)
```

**Tại sao dùng text hash?**
- Verify document content không đổi
- Nếu text thay đổi → hash khác → cache mới
- Tránh dùng nhầm cache của document cũ

## API Endpoints

### GET /cache/stats

Xem thống kê cache:

```json
{
    "total_files": 6,
    "total_size_mb": 45.3,
    "models": {
        "keepitreal_vietnamese-sbert": {
            "files": 2,
            "size_mb": 12.5
        },
        "AITeamVN_Vietnamese_Embedding": {
            "files": 4,
            "size_mb": 32.8
        }
    }
}
```

### GET /cache/list

List tất cả cache files:

```json
{
    "caches": [
        {
            "file": "doc_123_abc1234.npz",
            "model": "keepitreal_vietnamese-sbert",
            "doc_id": "doc_123",
            "chunks": 45,
            "dim": 768,
            "timestamp": 1704844800,
            "size_mb": 5.2
        }
    ]
}
```

### DELETE /cache/clear

Xóa cache:

```bash
# Xóa tất cả
DELETE /cache/clear

# Xóa một model cụ thể
DELETE /cache/clear?model=keepitreal_vietnamese-sbert

# Xóa một document cụ thể
DELETE /cache/clear?doc_id=doc_123

# Xóa doc cụ thể của model cụ thể
DELETE /cache/clear?model=MODEL&doc_id=DOC_ID
```

## CLI Management

### Xem statistics

```bash
python scripts/manage_cache.py stats
```

Output:
```
======================================================================
VECTOR CACHE STATISTICS
======================================================================

Total files:  6
Total size:   45.32 MB

Per-model breakdown:
Model                                              Files      Size (MB)      
----------------------------------------------------------------------
AITeamVN_Vietnamese_Embedding                      4          32.80          
keepitreal_vietnamese-sbert                        2          12.52          
======================================================================
```

### List cache files

```bash
# List all
python scripts/manage_cache.py list

# List for specific model
python scripts/manage_cache.py list --model keepitreal/vietnamese-sbert
```

### Clear cache

```bash
# Clear all (với confirmation)
python scripts/manage_cache.py clear

# Clear một model
python scripts/manage_cache.py clear --model keepitreal/vietnamese-sbert

# Clear một document
python scripts/manage_cache.py clear --doc doc_123

# Clear without confirmation
python scripts/manage_cache.py clear -y
```

## Use Cases

### Use Case 1: Thử Nghiệm Models

```bash
# 1. Upload document với model A
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
python src/app.py
# → Upload doc → Embed + cache (5s)

# 2. Test retrieval quality
python scripts/selftest.py
# → Accuracy: 70%

# 3. Switch to model B
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
python src/app.py
# → Upload doc → Embed + cache (10s)

# 4. Test retrieval quality
python scripts/selftest.py
# → Accuracy: 85%

# 5. Switch back to model A
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
python src/app.py
# → Upload doc → Load cache (< 1s) ⚡
```

**Lợi ích:** Test nhiều models nhanh chóng!

### Use Case 2: Competition Day

```bash
# Pre-competition: Test multiple models
for MODEL in model1 model2 model3; do
    EMBED_MODEL_NAME=$MODEL python src/app.py
    # Upload document → Cache for each model
done

# Competition: Use best model
EMBED_MODEL_NAME=best_model python src/app.py
# → Upload → Load from cache ⚡ (fast!)
```

### Use Case 3: Development

```bash
# Cache một lần
python src/app.py
# Upload document

# Sau đó restart nhiều lần (code changes)
python src/app.py  # Load cache
python src/app.py  # Load cache
python src/app.py  # Load cache

# → Không cần embed lại!
```

## Performance Impact

### Without Cache

```
Upload (first time):
- Chunking: 0.5s
- Embedding: 8.0s ⏱️
- BM25 build: 0.2s
Total: 8.7s
```

### With Cache (Hit)

```
Upload (cache hit):
- Check cache: 0.05s
- Load cache: 0.3s ⚡
- BM25 build: 0.2s
Total: 0.55s (15x faster!)
```

### Cache Size

| Model | Chunks | Embedding Dim | Cache Size |
|-------|--------|---------------|------------|
| vietnamese-sbert | 50 | 768 | ~5 MB |
| AITeamVN/Vietnamese_Embedding | 50 | 1024 | ~8 MB |
| E5-base | 50 | 768 | ~5 MB |

**Rule of thumb:** ~0.1 MB per chunk

## Cache Invalidation

Cache tự động invalidate khi:
1. ✅ Document text thay đổi (hash khác)
2. ✅ Model name khác
3. ❌ Chunk config đổi (WARNING: cache vẫn dùng)

**Note:** Nếu đổi `CHUNK_SIZE_WORDS`, cache cũ vẫn dùng được nhưng có thể không optimal. Khuyến nghị clear cache sau khi đổi chunking config.

## Best Practices

### ✅ DO

1. **Keep cache** cho models thường dùng
2. **Clear cache** khi đổi chunking config
3. **Monitor cache size** (không để quá lớn)
4. **Test với cache** trước khi competition

### ❌ DON'T

1. **Không commit** cache vào Git (đã ignore)
2. **Không delete** cache ngay trước competition
3. **Không dựa vào cache** nếu chưa test

## Troubleshooting

### Q: Cache không load?

```bash
# Check cache
python scripts/manage_cache.py list

# Rebuild cache
python scripts/manage_cache.py clear -y
# Then upload again
```

### Q: Cache quá lớn?

```bash
# Check size
python scripts/manage_cache.py stats

# Clear unused models
python scripts/manage_cache.py clear --model OLD_MODEL
```

### Q: Text đổi nhưng cache vẫn load?

```
Impossible! Hash verification sẽ fail và re-embed.
If happens, clear cache: python scripts/manage_cache.py clear -y
```

### Q: Switch model nhưng vẫn dùng cache cũ?

```
Check EMBED_MODEL_NAME trong .env
Restart server sau khi đổi model
```

## Technical Details

### Storage Format

`.npz` (numpy compressed archive):
- Efficient storage
- Fast loading
- Native numpy format

### Hash Function

SHA-256 của document text (16 chars):
- Collision-resistant
- Fast computation
- Deterministic

### Thread Safety

Cache operations are **thread-safe**:
- File-based locking
- Atomic writes
- Safe concurrent reads

## Monitoring

### Check cache health

```bash
# Quick check
curl http://localhost:5000/cache/stats

# Detailed check
python scripts/manage_cache.py stats
python scripts/manage_cache.py list
```

### Expected behavior

```
✓ Cache grows as you upload docs
✓ Each model has separate cache
✓ Cache persists across restarts
✓ Load time < 1s on cache hit
```

## Migration

### From old system (no cache)

1. **No action needed!** Cache tự động tạo
2. First upload sẽ chậm (embed)
3. Subsequent uploads sẽ nhanh (cache)

### Backup cache

```bash
# Backup
cp -r cache/ cache_backup/

# Restore
cp -r cache_backup/ cache/
```

## Summary

**Tính năng mới:**
- ✅ Auto cache embeddings per model
- ✅ 15x faster re-upload
- ✅ Test nhiều models dễ dàng
- ✅ No manual management needed

**Khi nào cache giúp ích:**
- 🔄 Switch models và quay lại
- 🧪 Development (restart nhiều)
- 🏆 Competition prep (test models)

**Khi nào cần clear:**
- 📝 Đổi chunking config
- 🗑️ Cache quá lớn
- 🐛 Debugging issues

Enjoy the speed boost! ⚡
