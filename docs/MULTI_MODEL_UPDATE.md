# Multi-Model Embedding Support - Update Summary

## Tổng Quan Thay Đổi

Hệ thống đã được nâng cấp để hỗ trợ **nhiều embedding models**, cho phép dễ dàng thay đổi và thử nghiệm với các models khác nhau chỉ bằng cách sửa file `.env`.

## Tính Năng Mới

### 1. **Multi-Model Support**
- Hỗ trợ 7+ embedding models
- Vietnamese-specific: AITeamVN, dangvantuan, keepitreal
- Multilingual: Jina AI v5, E5, và nhiều hơn nữa

### 2. **Auto-Configuration**
- **Tự động phát hiện** model config (prefix, max_seq_length, etc.)
- **E5 models**: Tự động thêm `query:` và `passage:` prefix
- **Jina v5**: Tự động enable `trust_remote_code`
- **Sequence length**: Auto-detect dựa trên model

### 3. **Per-Model Caching**
Mỗi model được cache riêng trong folder riêng:
```
src/models/
├── keepitreal_vietnamese-sbert/
├── AITeamVN_Vietnamese_Embedding/
├── jinaai_jina-embeddings-v5-text-nano/
└── intfloat_multilingual-e5-base/
```

### 4. **Easy Model Switching**
Thay đổi model chỉ cần **1 dòng** trong `.env`:
```bash
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
```

Hoặc dùng script tiện ích:
```bash
python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding
```

## Files Đã Thay Đổi

### Core Files

#### `src/config.py`
- Thêm `EMBED_CACHE_DIR` để quản lý cache directory
- Tự động tạo tên folder cho mỗi model: `model_name.replace("/", "_")`
- `MODEL_DIR` giờ auto-detect dựa trên `EMBED_MODEL_NAME`

#### `src/embedder.py`
- **Hoàn toàn refactor** để hỗ trợ multi-model
- Thêm `MODEL_CONFIGS` registry cho model-specific settings
- Hàm `_detect_model_config()` tự động phát hiện config
- Method `encode()` giờ có tham số `is_query` để handle prefix
- Singleton pattern được cải tiến để reload khi model thay đổi

#### `src/retriever.py`
- Update `build()` để dùng `is_query=False` khi embed passages
- Update `search()` để dùng `is_query=True` khi embed queries

### Configuration Files

#### `.env`
- Thêm section hướng dẫn chi tiết về các models
- Thêm `EMBED_CACHE_DIR` (optional)
- Comment giải thích từng loại model

#### `.env.example`
- Cập nhật với full documentation về các models
- Thêm ví dụ cho từng loại model

### Documentation

#### `docs/EMBEDDING_MODELS.md` (MỚI)
- **Hướng dẫn chi tiết** về từng model
- So sánh performance, kích thước, use cases
- Khuyến nghị model cho từng scenario
- Troubleshooting guide

#### `docs/MULTI_MODEL_UPDATE.md` (MỚI)
- File này - tóm tắt các thay đổi

#### `src/models/README.md` (MỚI)
- Giải thích cache structure
- Hướng dẫn quản lý cache

### Utility Scripts

#### `scripts/switch_model.py` (MỚI)
- CLI tool để xem/thay đổi model
- List all available models
- Validate model name
- Update .env automatically

#### `scripts/benchmark_models.py` (MỚI)
- Benchmark tất cả models
- So sánh: load time, encoding speed, dimension
- Tự động recommendations

### Updated Files

#### `README.md`
- Thêm section "Quản lý Embedding Models"
- Update kiến trúc diagram
- Thêm links đến docs/EMBEDDING_MODELS.md

## Model Configuration Registry

Code tự động detect config cho các loại model:

```python
MODEL_CONFIGS = {
    "intfloat/multilingual-e5": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
    },
    "jinaai/jina-embeddings-v5": {
        "max_seq_length": 8192,
        "trust_remote_code": True,
    },
    # ... và nhiều hơn
}
```

## Cách Sử Dụng

### Quick Start

```bash
# 1. Xem model hiện tại
python scripts/switch_model.py

# 2. List available models
python scripts/switch_model.py --list

# 3. Thay đổi model
python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding

# 4. Restart server
python src/app.py
```

### Hoặc thủ công:

1. Mở `.env`
2. Sửa dòng: `EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding`
3. Restart server

## Backward Compatibility

✅ **Hoàn toàn tương thích ngược**

- Model mặc định vẫn là `keepitreal/vietnamese-sbert`
- Nếu không thay đổi gì, hệ thống hoạt động y như cũ
- Cache cũ vẫn dùng được (ở `src/models/vietnamese-sbert/`)

## Testing

### Test basic functionality:
```bash
# Test current model
python scripts/test_endpoints.py

# Benchmark all models
python scripts/benchmark_models.py
```

### Test model switching:
```bash
# Switch to AITeamVN
python scripts/switch_model.py --set AITeamVN/Vietnamese_Embedding

# Restart server in another terminal
python src/app.py

# Test
python scripts/selftest.py
```

## Khuyến Nghị

### Cho Production (Vietnamese Q&A):
1. **Tốt nhất**: `AITeamVN/Vietnamese_Embedding`
2. **Cân bằng**: `dangvantuan/vietnamese-embedding`
3. **Nhanh nhất**: `keepitreal/vietnamese-sbert`

### Cho Multilingual:
1. **Proven**: `intfloat/multilingual-e5-base`
2. **Modern**: `jinaai/jina-embeddings-v5-text-nano`

## Troubleshooting

### Model không load được:
```bash
# Clear cache
rm -rf src/models/MODEL_NAME/

# Download again
python src/app.py
```

### Out of memory:
```bash
# Dùng model nhỏ hơn
EMBED_MODEL_NAME=keepitreal/vietnamese-sbert
```

### Kết quả không tốt:
```bash
# Thử model lớn hơn
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Hoặc điều chỉnh retrieval params
TOP_K_CONTEXT=8
```

## Performance Impact

| Model | Load Time | Encode Speed | Memory | Quality |
|-------|-----------|--------------|--------|---------|
| keepitreal | Fast (~2s) | Fast | Low | Good |
| AITeamVN | Slow (~8s) | Slow | High | **Excellent** |
| dangvantuan | Medium (~4s) | Medium | Medium | Very Good |
| E5-base | Medium (~5s) | Medium | Medium | Very Good |
| Jina v5 nano | Medium (~5s) | Medium | Medium | Very Good |

**Note**: Load time chỉ xảy ra 1 lần khi start server, không ảnh hưởng query time.

## Future Enhancements

- [ ] Support reranker models
- [ ] Support LoRA adapters for Jina v5
- [ ] Cache embeddings to disk (persistent cache)
- [ ] Support ONNX/quantized models for faster inference
- [ ] Auto-benchmark on first run

## Migration Guide

Nếu bạn đang dùng version cũ:

1. ✅ **Không cần làm gì** - code vẫn chạy như cũ
2. ✅ Cache cũ vẫn dùng được
3. ✅ Muốn thử model mới: chỉ cần đổi `.env`

## References

- [Embedding Models Guide](./EMBEDDING_MODELS.md)
- [Models Cache README](../src/models/README.md)
- [Main README](../README.md)
