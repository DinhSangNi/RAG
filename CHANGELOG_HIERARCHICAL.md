# Hierarchical RAG Implementation - Changelog

## Ngày: 2026-01-31

## Tóm tắt
Triển khai workflow hierarchical retrieval mới cho hệ thống RAG với 6 bước, giúp cải thiện độ chính xác và giảm nhiễu thông tin.

## Các thay đổi chính

### 1. Database Schema

#### Bảng mới: `summary_documents`
- **File**: `migrations/create_summary_documents_table.sql`
- **Cấu trúc**:
  ```sql
  - id (UUID): Primary key
  - document_id (UUID): Foreign key -> documents.id
  - summary_content (TEXT): Nội dung tóm tắt
  - embedding (vector(768)): Vector embedding
  - metadata (JSONB): Metadata
  - created_at, updated_at: Timestamps
  ```
- **Indexes**:
  - HNSW index trên embedding (semantic search)
  - ParadeDB BM25 index (full-text search)
  - Index trên document_id

### 2. Models (`app/database/models.py`)

#### Model mới: `SummaryDocument`
```python
class SummaryDocument(Base):
    __tablename__ = "summary_documents"
    - Lưu tóm tắt documents
    - Relationship với Document
    - Vector embedding
```

#### Cập nhật: `Document`
```python
- Thêm relationship: summary_documents
```

### 3. Search Service (`app/services/search_service.py`)

#### Methods mới:

1. **`bm25_search_summaries()`**
   - BM25 search trên summary documents
   - Sử dụng ParadeDB

2. **`semantic_search_summaries()`**
   - Semantic search trên summary documents
   - Sử dụng pgvector cosine similarity

3. **`hybrid_search_summaries()`**
   - Kết hợp BM25 + Semantic bằng RRF
   - Tìm kiếm summary documents
   - Trả về max score để check threshold

### 4. RAG Service (`app/services/rag_service.py`)

#### Prompts mới:

1. **`sufficiency_prompt`**
   - Check xem summary docs có đủ thông tin không
   - LLM trả về JSON: `{sufficient: bool, reason: str}`

#### Methods mới:

1. **`_format_summary_docs()`**
   - Format summary documents thành context
   
2. **`_check_info_sufficiency()`**
   - Gọi LLM để kiểm tra đủ thông tin
   - Returns: True/False

3. **`retrieve_hierarchical()`**
   - **Workflow 6 bước**:
     1. Search summary documents (hybrid)
     2. Check sufficiency với LLM
     3. Nếu đủ → return summaries; Nếu không → expand
     4. Search child chunks (scoped to relevant docs)
     5. RRF fusion của tất cả variants
     6. Return chunks
   
   - **Parameters**:
     - `summary_k`: Số summary docs (default: 5)
     - `chunk_k`: Số chunks trả về (default: 20)
     - `min_summary_score`: Threshold (default: 0.3)
   
   - **Returns**:
     ```python
     {
       'docs': List[Dict],
       'source': 'summary' | 'chunks_expanded' | 'chunks_fallback',
       'metadata': {...}
     }
     ```

#### Methods cập nhật:

1. **`chat()`**
   - Thêm parameter: `use_hierarchical` (default: True)
   - Tự động chọn workflow dựa vào flag
   - Xử lý cả summary docs và chunks
   - Trả về metadata chi tiết

### 5. Helper Scripts

#### `src/generate_summaries.py`
- Script tự động tạo summary documents
- Sử dụng LLM (Gemini) để summarize
- Tạo embeddings cho summaries
- Lưu vào database
- Usage:
  ```bash
  python src/generate_summaries.py
  ```

#### `src/test_hierarchical_rag.py`
- Script test workflow mới
- So sánh hierarchical vs legacy
- Test summary search riêng
- Usage:
  ```bash
  python src/test_hierarchical_rag.py --mode full
  python src/test_hierarchical_rag.py --mode summary
  ```

### 6. Documentation

#### `HIERARCHICAL_RAG_README.md`
- Hướng dẫn chi tiết về workflow
- Cách cài đặt và sử dụng
- Ví dụ code
- Bảng so sánh với workflow cũ
- Troubleshooting

## Workflow Chi tiết

```
User Question
     |
     v
[1] Search Summary Docs (Hybrid)
     |
     +---> Max score < 0.3? ---> [Fallback] Search all chunks --> Answer
     |                                
     v
[2] Check Sufficiency (LLM)
     |
     +---> Sufficient? ---> [3a] Generate answer from summaries
     |
     v
[3b] Query Expansion
     - Extract entities, aliases, keywords (LLM)
     - Generate query variants
     |
     v
[4] Search Child Chunks (Scoped)
     - For each variant
     - Only in relevant docs
     |
     v
[5] RRF Fusion
     - Combine all results
     - Calculate scores
     |
     v
[6] Generate Answer
     - Send top chunks to LLM
     - Return answer
```

## Breaking Changes

**KHÔNG CÓ** - Backward compatible

- Old workflow vẫn hoạt động: `chat(use_hierarchical=False)`
- Default behavior: Sử dụng hierarchical retrieval
- API không thay đổi

## Migration Steps

1. **Run SQL migration**:
   ```bash
   psql -h 127.0.0.1 -p 5433 -U rag_user -d rag_db -f migrations/create_summary_documents_table.sql
   ```

2. **Generate summaries**:
   ```bash
   python src/generate_summaries.py
   ```

3. **Test**:
   ```bash
   python src/test_hierarchical_rag.py
   ```

## Configuration

Các tham số mặc định (có thể customize):

```python
RAGService(
    # Retrieval
    top_k=20,              # Số chunks cuối
    first_pass_k=12,       # Chunks đầu tiên
    variant_count=5,       # Query variants
    
    # Weights
    bm25_weight=0.6,       # BM25 weight
    semantic_weight=0.4,   # Semantic weight
    
    # RRF
    rrf_k=60               # RRF parameter
)

retrieve_hierarchical(
    summary_k=5,           # Summary docs
    chunk_k=20,            # Chunks
    min_summary_score=0.3  # Score threshold
)
```

## Performance Impact

### Ưu điểm:
- ✅ Độ chính xác cao hơn (scope search)
- ✅ Ít nhiễu thông tin hơn
- ✅ Sufficiency check tránh expansion không cần thiết
- ✅ Fallback tự động khi không có summaries

### Nhược điểm:
- ⚠️ Cần tạo summaries trước (one-time cost)
- ⚠️ Thêm 1-2 LLM calls (sufficiency check)
- ⚠️ Database size tăng (summaries table)

## Dependencies

Không có dependency mới, sử dụng:
- PostgreSQL + pgvector
- ParadeDB (pg_search)
- LangChain
- Google Gemini API

## Testing

Tất cả code đã pass syntax check:
- ✅ `app/services/rag_service.py`
- ✅ `app/services/search_service.py`
- ✅ `app/database/models.py`

## Next Steps

1. Run migration SQL
2. Generate summary documents
3. Test với các câu hỏi thực tế
4. Fine-tune parameters (summary_k, min_score, etc.)
5. Monitor performance và accuracy

## Author
- GitHub Copilot
- Date: 2026-01-31
