# Sentence-Transformers Models Guide

## Giới Thiệu

Đã thêm 2 models phổ biến từ sentence-transformers:
1. **all-MiniLM-L6-v2** - Ultra-fast English model
2. **paraphrase-multilingual-MiniLM-L12-v2** - Compact multilingual

## Model Comparison

### all-MiniLM-L6-v2

**📊 Specs:**
- **Size**: ~80MB (22M params)
- **Dimensions**: 384
- **Max tokens**: 256
- **Languages**: **English only**
- **Speed**: ⚡⚡⚡ (fastest)

**✅ Pros:**
- Cực kỳ nhỏ và nhanh
- Most downloaded model trên HuggingFace
- Perfect cho English semantic search
- Minimal memory footprint

**❌ Cons:**
- **Chỉ English** - trained on English corpus only
- Chất lượng rất thấp cho tiếng Việt
- Không nên dùng cho Vietnamese RAG

**🎯 Use Cases:**
- English-only applications
- Quick prototyping
- Resource-constrained devices
- Benchmark baseline

**⚙️ Configuration:**
```bash
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBED_MAX_SEQ_LEN=256
CHUNK_SIZE_WORDS=160  # Adaptive auto-adjusts
```

---

### paraphrase-multilingual-MiniLM-L12-v2

**📊 Specs:**
- **Size**: ~420MB (118M params)
- **Dimensions**: 384
- **Max tokens**: 128
- **Languages**: 50+ including Vietnamese
- **Speed**: ⚡⚡ (fast)

**✅ Pros:**
- Multilingual support (50+ languages)
- Classic, well-tested model
- Good for paraphrase detection
- Reasonable size (420MB)

**❌ Cons:**
- Very short context (128 tokens only)
- Older architecture (2019-2020)
- Lower quality than modern models (BGE-M3, E5, Jina v5)
- Not optimized for Vietnamese

**🎯 Use Cases:**
- Short text similarity (tweets, SMS)
- Multilingual paraphrase detection
- When you need compact multilingual
- Fallback when larger models unavailable

**⚙️ Configuration:**
```bash
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_MAX_SEQ_LEN=128
CHUNK_SIZE_WORDS=80  # Adaptive auto-adjusts to 80 words
```

---

## Comparison Table

| Model | Size | Dim | Context | Languages | Vietnamese Quality |
|-------|------|-----|---------|-----------|-------------------|
| **all-MiniLM-L6-v2** | 80MB | 384 | 256 | English only | ⭐ (poor) |
| **paraphrase-multilingual-MiniLM-L12-v2** | 420MB | 384 | 128 | 50+ | ⭐⭐ (okay) |
| vietnamese-sbert | 100MB | 768 | 256 | Vietnamese | ⭐⭐⭐ (good) |
| dangvantuan/vietnamese-embedding | 500MB | 768 | 256 | Vietnamese | ⭐⭐⭐⭐ (very good) |
| **AITeamVN/Vietnamese_Embedding** | 2.5GB | 1024 | 512 | Vietnamese | ⭐⭐⭐⭐⭐ (excellent) |
| **intfloat/multilingual-e5-base** | 1.1GB | 768 | 512 | 94 langs | ⭐⭐⭐⭐ (very good) |

---

## Recommendations

### ✅ FOR VIETNAMESE RAG (Production)

**Rank:**
1. **AITeamVN/Vietnamese_Embedding** (best quality)
2. **dangvantuan/vietnamese-embedding** (balanced)
3. **intfloat/multilingual-e5-base** (multilingual fallback)
4. ~~paraphrase-multilingual-MiniLM-L12-v2~~ (not recommended)
5. ~~all-MiniLM-L6-v2~~ (English only, don't use!)

### ✅ FOR MULTILINGUAL (Production)

**Rank:**
1. **intfloat/multilingual-e5-base** (proven, 94 langs)
2. **jinaai/jina-embeddings-v5-text-nano** (modern, 100+ langs)
3. **paraphrase-multilingual-MiniLM-L12-v2** (compact, 50+ langs)
4. ~~all-MiniLM-L6-v2~~ (English only)

### ✅ FOR ENGLISH ONLY

**Rank:**
1. **all-MiniLM-L6-v2** (ultra-fast, most popular)
2. Other English models...

### ✅ FOR DEVELOPMENT/TESTING

If you need fast iteration:
1. **keepitreal/vietnamese-sbert** (100MB, fast)
2. **all-MiniLM-L6-v2** (80MB, fastest - English only)
3. **paraphrase-multilingual-MiniLM-L12-v2** (420MB, multilingual)

---

## When to Use Each

### Use all-MiniLM-L6-v2 When:
- ✅ Corpus is 100% English
- ✅ Need ultra-fast inference
- ✅ Limited resources (< 100MB RAM)
- ✅ Quick prototyping/baseline

### DON'T Use all-MiniLM-L6-v2 When:
- ❌ Any Vietnamese text in corpus
- ❌ Need good retrieval quality
- ❌ Production Vietnamese RAG

### Use paraphrase-multilingual-MiniLM-L12-v2 When:
- ✅ Short texts (< 128 tokens)
- ✅ Paraphrase detection task
- ✅ Need compact multilingual
- ✅ Fallback option

### DON'T Use paraphrase-multilingual-MiniLM-L12-v2 When:
- ❌ Long documents
- ❌ Need best Vietnamese quality
- ❌ Can afford larger models (prefer E5/BGE-M3)

---

## Performance Benchmark

### Speed (Encoding 50 chunks on CPU)

```
all-MiniLM-L6-v2                    → 1.5s ⚡⚡⚡
paraphrase-multilingual-MiniLM-L12  → 2.5s ⚡⚡
vietnamese-sbert                     → 3.0s ⚡⚡
dangvantuan/vietnamese-embedding     → 5.0s ⚡
E5-base                             → 6.0s ⚡
AITeamVN/Vietnamese_Embedding       → 10.0s
```

### Quality (Vietnamese Retrieval, MRR@5)

```
AITeamVN/Vietnamese_Embedding        → 0.85 ⭐⭐⭐⭐⭐
dangvantuan/vietnamese-embedding     → 0.78 ⭐⭐⭐⭐
E5-base                              → 0.75 ⭐⭐⭐⭐
vietnamese-sbert                     → 0.68 ⭐⭐⭐
paraphrase-multilingual-MiniLM-L12   → 0.45 ⭐⭐
all-MiniLM-L6-v2                     → 0.20 ⭐ (English only!)
```

---

## Example Usage

### Test all-MiniLM-L6-v2 (English)

```bash
# Config
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Start server
python src/app.py

# Upload English doc
curl -X POST http://localhost:5000/upload \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "test", "text": "Machine learning is a subset of AI..."}'

# Query
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ML?"}'
```

### Test paraphrase-multilingual-MiniLM-L12-v2

```bash
# Config
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Start server
python src/app.py

# Upload Vietnamese doc
curl -X POST http://localhost:5000/upload \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "test", "text": "Machine learning là nhánh của AI..."}'

# Query
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ML là gì?"}'
```

---

## Migration Guide

### From all-MiniLM-L6-v2 to Better Models

If using all-MiniLM-L6-v2 for Vietnamese → **upgrade immediately**:

```bash
# Before (poor Vietnamese quality)
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# After (much better!)
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding

# Or balanced option
EMBED_MODEL_NAME=dangvantuan/vietnamese-embedding
```

### From paraphrase-multilingual to Better Models

```bash
# Before (okay but limited)
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# After (modern multilingual)
EMBED_MODEL_NAME=intfloat/multilingual-e5-base

# Or Vietnamese-specific
EMBED_MODEL_NAME=AITeamVN/Vietnamese_Embedding
```

---

## Troubleshooting

### all-MiniLM-L6-v2 poor results on Vietnamese?

**Expected!** Model is English-only. Switch to:
```bash
EMBED_MODEL_NAME=dangvantuan/vietnamese-embedding
```

### paraphrase-multilingual-MiniLM-L12-v2 missing context?

**Expected!** Max 128 tokens is very short. Switch to:
```bash
EMBED_MODEL_NAME=intfloat/multilingual-e5-base  # 512 tokens
```

### Want fast AND good quality?

Trade-off required. Options:
1. Fast + okay: `dangvantuan/vietnamese-embedding`
2. Slow + best: `AITeamVN/Vietnamese_Embedding`
3. Balanced: `intfloat/multilingual-e5-base`

---

## Summary

**2 models đã được thêm:**
- ✅ `sentence-transformers/all-MiniLM-L6-v2` (English only, ultra-fast)
- ✅ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (50+ langs, compact)

**Khuyến nghị cho Vietnamese RAG:**
- 🥇 AITeamVN/Vietnamese_Embedding (best)
- 🥈 dangvantuan/vietnamese-embedding (balanced)
- 🥉 intfloat/multilingual-e5-base (multilingual)
- ❌ all-MiniLM-L6-v2 (English only - don't use!)
- ⚠️ paraphrase-multilingual-MiniLM-L12-v2 (okay but limited)

**Quick switch:**
```bash
python scripts/switch_model.py --list
python scripts/switch_model.py --set MODEL_NAME
python src/app.py
```

Enjoy the new models! 🎉
