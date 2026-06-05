# Chunking Strategy - Phân Tích và Tối Ưu

## Phân Tích Hiện Tại

### Implementation Hiện Tại (textutils.py)

**Chiến lược**: Sentence-aware sliding window với overlap

```python
CHUNK_SIZE_WORDS = 180      # ~180 từ/chunk
CHUNK_OVERLAP_WORDS = 40    # 40 từ overlap giữa các chunk
```

**Cách hoạt động**:
1. Chia text thành câu (dựa vào `.!?;:` và paragraph breaks)
2. Gộp câu cho đến khi đạt 180 từ
3. Bắt đầu chunk mới với **40 từ cuối** của chunk trước (overlap)
4. Deduplicate chunks giống nhau

### Ưu Điểm
✅ **Sentence-aware**: Không cắt giữa câu, giữ nguyên ý nghĩa  
✅ **Overlap**: Giữ context liên tục giữa các chunk  
✅ **Hard-split cho câu dài**: Câu > 180 từ được chia nhỏ  
✅ **Deduplication**: Loại bỏ chunks trùng lặp

### Vấn Đề Tiềm Ẩn

❌ **Overlap ratio thấp**: 40/180 = 22% (nên 25-30%)  
❌ **Context window hẹp**: 180 từ ≈ 250-300 tokens, có thể mất thông tin tổng quan  
❌ **Không có semantic boundary detection**: Chỉ dựa vào dấu câu  
❌ **Không xử lý đặc biệt cho lists, tables, code blocks**

---

## Phân Tích Kỹ Thuật

### 1. Chunk Size vs Embedding Model

| Model | Max Seq Length | Optimal Chunk Size (words) | Reasoning |
|-------|----------------|---------------------------|-----------|
| keepitreal/vietnamese-sbert | 256 tokens | **150-180** | ~0.7 token/word cho tiếng Việt |
| AITeamVN/Vietnamese_Embedding | 512 tokens | **300-350** | BGE-M3 hỗ trợ context dài hơn |
| dangvantuan/vietnamese-embedding | 256 tokens | **150-180** | PhoBERT tokenizer tương tự |
| intfloat/multilingual-e5-base | 512 tokens | **300-350** | XLM-RoBERTa tokenizer |
| jinaai/jina-embeddings-v5 | 8192 tokens | **500-1000+** | Ultra-long context support |

**Kết luận**: Chunk size nên **tự động điều chỉnh theo model**

### 2. Overlap Strategy

**Công thức tối ưu**: 
```
overlap_ratio = 0.25 - 0.30  (25-30%)
overlap_words = chunk_size * overlap_ratio
```

**Tại sao?**
- < 20%: Mất context giữa các chunk
- 20-25%: Acceptable
- **25-30%**: Optimal (khuyến nghị research)
- > 30%: Redundancy cao, tăng latency không đáng kể

### 3. Token Budget Analysis

**Với model 256 tokens (vietnamese-sbert, dangvantuan)**:
```
180 words × 1.4 token/word ≈ 252 tokens  ✅ Vừa khớp
40 overlap × 1.4 ≈ 56 tokens (22% overlap)
```

**Với model 512 tokens (BGE-M3, E5)**:
```
180 words × 1.4 ≈ 252 tokens  ⚠️ Dư nhiều (chỉ dùng ~50%)
Nên tăng lên 350 words × 1.4 ≈ 490 tokens  ✅ Tận dụng tốt
```

---

## Đề Xuất Cải Tiến

### 1. **Adaptive Chunking Based on Model**

```python
# Tự động điều chỉnh chunk size theo model
MODEL_CHUNK_CONFIGS = {
    "keepitreal/vietnamese-sbert": {
        "chunk_size": 150,
        "overlap": 40,
        "overlap_ratio": 0.27,
    },
    "AITeamVN/Vietnamese_Embedding": {
        "chunk_size": 320,
        "overlap": 85,
        "overlap_ratio": 0.27,
    },
    "dangvantuan/vietnamese-embedding": {
        "chunk_size": 150,
        "overlap": 40,
        "overlap_ratio": 0.27,
    },
    "intfloat/multilingual-e5-base": {
        "chunk_size": 320,
        "overlap": 85,
        "overlap_ratio": 0.27,
    },
    "jinaai/jina-embeddings-v5": {
        "chunk_size": 600,
        "overlap": 160,
        "overlap_ratio": 0.27,
    },
}
```

### 2. **Semantic Boundary Detection**

Thêm detection cho:
- **Headings**: `#`, `##`, `**Title**`
- **Lists**: Bullet points, numbered lists
- **Paragraph breaks**: `\n\n`
- **Section markers**: `===`, `---`

```python
def detect_semantic_boundaries(text: str) -> List[int]:
    """Phát hiện vị trí nên chia chunk (headings, lists, paragraphs)"""
    boundaries = []
    
    # Headings (markdown style)
    for m in re.finditer(r'^#{1,6}\s+.+$', text, re.MULTILINE):
        boundaries.append(m.start())
    
    # Strong paragraph breaks
    for m in re.finditer(r'\n\n+', text):
        boundaries.append(m.start())
    
    # List starts
    for m in re.finditer(r'^\s*[\-\*\+\d+\.]\s+', text, re.MULTILINE):
        boundaries.append(m.start())
    
    return sorted(set(boundaries))
```

### 3. **Context Preservation Strategy**

**Hierarchical Chunking**: Tạo 2 levels
- **Level 1 (Detail)**: Chunks nhỏ (150-320 words) cho dense retrieval
- **Level 2 (Overview)**: Chunks lớn (500-1000 words) cho cái nhìn tổng quan

```python
def hierarchical_chunk(text: str) -> dict:
    return {
        "detail_chunks": chunk_text(text, size=180, overlap=45),
        "overview_chunks": chunk_text(text, size=600, overlap=150),
    }
```

### 4. **Smart Overlap với Sentence Priority**

Thay vì carry overlap words máy móc, ưu tiên **complete sentences**:

```python
def smart_overlap(prev_chunk: str, overlap_words: int) -> List[str]:
    """Lấy overlap nhưng ưu tiên câu hoàn chỉnh"""
    words = prev_chunk.split()
    if len(words) <= overlap_words:
        return words
    
    # Lấy khoảng overlap_words, tìm sentence boundary gần nhất
    target_start = len(words) - overlap_words
    text = " ".join(words)
    
    # Tìm câu hoàn chỉnh gần target_start nhất
    sentences = split_sentences(text)
    # ... logic tìm boundary tốt nhất
```

---

## Khuyến Nghị Cấu Hình

### Cho keepitreal/vietnamese-sbert (hiện tại)
```bash
# Good (hiện tại)
CHUNK_SIZE_WORDS=180
CHUNK_OVERLAP_WORDS=40

# Better (tối ưu)
CHUNK_SIZE_WORDS=160
CHUNK_OVERLAP_WORDS=45
```

### Cho AITeamVN/Vietnamese_Embedding (BGE-M3)
```bash
# Recommended
CHUNK_SIZE_WORDS=320
CHUNK_OVERLAP_WORDS=85
```

### Cho jinaai/jina-embeddings-v5 (long context)
```bash
# Recommended
CHUNK_SIZE_WORDS=600
CHUNK_OVERLAP_WORDS=160
```

### Cho dangvantuan/vietnamese-embedding (PhoBERT)
```bash
# Recommended
CHUNK_SIZE_WORDS=150
CHUNK_OVERLAP_WORDS=40
```

### Cho intfloat/multilingual-e5-base
```bash
# Recommended
CHUNK_SIZE_WORDS=320
CHUNK_OVERLAP_WORDS=85
```

---

## Công Thức Tính Toán

### 1. Optimal Chunk Size
```python
def calculate_optimal_chunk_size(max_seq_len: int, avg_token_per_word: float = 1.4) -> int:
    """
    max_seq_len: từ model config (256, 512, 8192...)
    avg_token_per_word: ~1.4 cho tiếng Việt
    
    Giữ lại 10% buffer cho tokenizer overhead
    """
    safe_tokens = int(max_seq_len * 0.9)
    chunk_size_words = int(safe_tokens / avg_token_per_word)
    return chunk_size_words
```

### 2. Optimal Overlap
```python
def calculate_optimal_overlap(chunk_size: int, overlap_ratio: float = 0.27) -> int:
    """
    overlap_ratio: 0.25-0.30 (khuyến nghị 0.27)
    """
    return int(chunk_size * overlap_ratio)
```

### 3. Number of Chunks Estimation
```python
def estimate_num_chunks(text_words: int, chunk_size: int, overlap: int) -> int:
    """Ước tính số chunks sẽ được tạo"""
    if text_words <= chunk_size:
        return 1
    
    effective_step = chunk_size - overlap
    return 1 + math.ceil((text_words - chunk_size) / effective_step)
```

---

## Test Cases và Validation

### Test 1: No Context Loss
```python
def test_no_context_loss():
    """Đảm bảo overlap đủ để preserve context"""
    text = "Câu 1. Câu 2. Câu 3. Câu 4. Câu 5."
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    
    # Kiểm tra mỗi pair chunks kế tiếp có overlap
    for i in range(len(chunks) - 1):
        assert has_overlap(chunks[i], chunks[i+1])
```

### Test 2: Sentence Integrity
```python
def test_sentence_integrity():
    """Đảm bảo không cắt giữa câu"""
    text = "Đây là câu dài. Đây là câu ngắn."
    chunks = chunk_text(text)
    
    for chunk in chunks:
        # Mỗi chunk phải bắt đầu bằng chữ hoa (đầu câu)
        assert chunk[0].isupper()
        # Mỗi chunk phải kết thúc bằng dấu câu
        assert chunk[-1] in ".!?;:"
```

### Test 3: Optimal Coverage
```python
def test_optimal_coverage():
    """Đảm bảo retrieval coverage tốt"""
    text = load_test_corpus()
    chunks = chunk_text(text)
    
    # Với query bất kỳ, ít nhất 1 chunk phải có relevance > threshold
    queries = load_test_queries()
    for query in queries:
        scores = [similarity(query, chunk) for chunk in chunks]
        assert max(scores) > 0.5  # Ít nhất 1 chunk relevant
```

---

## Benchmarking

### Script để test chunking strategies

```python
def benchmark_chunking(text: str, strategies: dict):
    """
    So sánh các chunking strategies khác nhau
    
    Metrics:
    - Số chunks tạo ra
    - Average chunk size
    - Overlap ratio
    - Retrieval coverage (% queries tìm thấy answer)
    - Latency
    """
    results = {}
    
    for name, config in strategies.items():
        start = time.time()
        chunks = chunk_text(
            text, 
            chunk_size=config["size"], 
            overlap=config["overlap"]
        )
        latency = time.time() - start
        
        results[name] = {
            "num_chunks": len(chunks),
            "avg_size": sum(len(c.split()) for c in chunks) / len(chunks),
            "overlap_ratio": config["overlap"] / config["size"],
            "latency": latency,
        }
    
    return results
```

---

## Recommendations Summary

### ⭐ Quick Wins (Dễ implement)

1. **Tăng overlap ratio** lên 27%:
   ```bash
   CHUNK_SIZE_WORDS=180
   CHUNK_OVERLAP_WORDS=50  # từ 40 lên 50
   ```

2. **Add chunk size validation**:
   - Warning nếu chunk > 90% của max_seq_len
   - Auto-adjust nếu có thể

### 🚀 Advanced (Nếu có thời gian)

1. **Adaptive chunking theo model**
2. **Semantic boundary detection**
3. **Hierarchical chunking** (detail + overview)
4. **Smart sentence-aware overlap**

### 📊 Testing

1. Benchmark với corpus thực tế
2. Measure retrieval quality (MRR@5, Recall@10)
3. A/B test giữa strategies khác nhau

---

## Kết Luận

**Config hiện tại (180/40)** là **acceptable** cho model nhỏ (vietnamese-sbert 256 tokens).

**Khuyến nghị cải tiến**:
1. Tăng overlap lên **50 words** (27%)
2. Khi dùng model lớn (BGE-M3, E5), tăng chunk size lên **320 words**
3. Implement adaptive chunking trong tương lai

**Trade-offs**:
- Chunk lớn hơn: Cái nhìn tổng quan tốt hơn, nhưng noise nhiều hơn
- Chunk nhỏ hơn: Precision cao hơn, nhưng có thể miss context
- Overlap cao hơn: Context preservation tốt hơn, nhưng redundancy tăng

**Best practice**: Test với corpus và queries thực tế của bạn!
