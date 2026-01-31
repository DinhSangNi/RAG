# Hierarchical RAG Retrieval Workflow

## Tổng quan

Hệ thống RAG đã được nâng cấp với workflow hierarchical retrieval mới, giúp cải thiện độ chính xác và giảm nhiễu thông tin khi tìm kiếm.

## Kiến trúc Database

Hệ thống gồm 3 bảng chính:

1. **summary_documents**: Lưu tóm tắt của các parent documents
2. **documents**: Các parent documents (tài liệu gốc)
3. **chunks**: Các child chunks (đoạn nhỏ từ documents)

## Workflow 6 bước

### Bước 1: Tìm kiếm trên Summary Documents
- Sử dụng **hybrid search** (BM25 + Semantic) trên bảng `summary_documents`
- Mục tiêu: Xác định phạm vi tìm kiếm, tìm các documents liên quan nhất
- **Fallback**: Nếu không tìm thấy summary docs hoặc max score < 0.3 → tìm kiếm trực tiếp trên toàn bộ child chunks

### Bước 2: Kiểm tra đủ thông tin
- Format các summary docs tìm được thành context
- Gửi context + câu hỏi cho LLM để kiểm tra xem đã đủ thông tin chưa
- LLM trả về JSON: `{sufficient: true/false, reason: "..."}`

### Bước 3: Quyết định
- **Nếu đủ thông tin**: 
  - Sử dụng summary docs để tạo câu trả lời
  - Workflow kết thúc ở đây
  
- **Nếu không đủ**: 
  - Thực hiện query expansion
  - Trích xuất entities, aliases, keywords bằng LLM
  - Tạo query variants (biến thể câu hỏi)

### Bước 4: Tìm kiếm Child Chunks (có phạm vi)
- Sử dụng từng query variant để tìm kiếm
- **Chỉ tìm trong phạm vi** các child chunks thuộc về summary docs đã tìm ở bước 1
- Giảm nhiễu thông tin, tăng độ chính xác

### Bước 5: Tổng hợp kết quả (RRF)
- Áp dụng thuật toán **RRF (Reciprocal Rank Fusion)** 
- Tính điểm cho từng chunk từ các kết quả tìm kiếm
- Chọn các chunks có điểm cao nhất

### Bước 6: Tạo câu trả lời
- Gửi các chunks tốt nhất + câu hỏi cho LLM
- LLM tạo câu trả lời cuối cùng

## Cài đặt

### 1. Chạy Migration

```bash
# PowerShell
psql -h 127.0.0.1 -p 5433 -U rag_user -d rag_db -f migrations/create_summary_documents_table.sql

# Hoặc dùng script
.\run_migration.ps1
```

### 2. Sử dụng trong Code

```python
from app.services.rag_service import get_rag_service
from app.database.connection import get_db

# Khởi tạo service
db = next(get_db())
rag_service = get_rag_service(db)

# Sử dụng hierarchical retrieval (mặc định)
result = rag_service.chat(
    question="Hồ Chí Minh sinh năm nào?",
    use_hierarchical=True  # Mặc định là True
)

print(result['answer'])
print(result['metadata'])
```

### 3. Tạo Summary Documents

Bạn cần tạo summary documents cho các parent documents. Có thể sử dụng LLM để tạo tóm tắt:

```python
from app.database.models import SummaryDocument, Document
from app.services.embedding_service import get_embedding_service

embedding_service = get_embedding_service()

# Lấy document
doc = db.query(Document).first()

# Tạo summary (có thể dùng LLM)
summary_text = "Tóm tắt nội dung document..."

# Tạo embedding
embedding = embedding_service.embed_text(summary_text)

# Lưu vào DB
summary_doc = SummaryDocument(
    document_id=doc.id,
    summary_content=summary_text,
    embedding=embedding
)
db.add(summary_doc)
db.commit()
```

## Tham số cấu hình

```python
rag_service = RAGService(
    db=db,
    # LLM settings
    model_name="gemini-2.0-flash-exp",
    temperature=0.1,
    
    # Retrieval settings
    top_k=20,              # Số chunks trả về cuối cùng
    first_pass_k=12,       # Số chunks trong lần tìm kiếm đầu
    variant_count=5,       # Số query variants tạo ra
    
    # Hybrid search weights
    bm25_weight=0.6,       # Trọng số BM25
    semantic_weight=0.4,   # Trọng số semantic
    
    # RRF parameter
    rrf_k=60              # Tham số k của RRF
)

# Hierarchical retrieval settings
result = rag_service.retrieve_hierarchical(
    question="...",
    summary_k=5,           # Số summary docs tìm kiếm
    chunk_k=20,            # Số chunks trả về cuối
    min_summary_score=0.3  # Ngưỡng score tối thiểu
)
```

## Metadata trả về

```python
result = rag_service.chat(question="...")

# Metadata có thể bao gồm:
metadata = result['metadata']
# {
#     'retrieval_method': 'hierarchical',
#     'source': 'summary' | 'chunks_expanded' | 'chunks_fallback',
#     'summary_docs_count': 3,
#     'max_summary_score': 0.85,
#     'sufficient': True/False,
#     'variants_count': 5,
#     'chunks_returned': 20,
#     'scoped_to_docs': 3,
#     'chunks_used': 20,
#     'model': 'gemini-2.0-flash-exp'
# }
```

## So sánh với workflow cũ

| Aspect | Old Workflow | New Hierarchical Workflow |
|--------|-------------|---------------------------|
| Tìm kiếm | Trực tiếp trên chunks | Tìm summary trước → scope → chunks |
| Độ chính xác | Trung bình | Cao hơn (ít nhiễu) |
| Query expansion | Luôn thực hiện | Chỉ khi cần (không đủ info) |
| Phạm vi tìm kiếm | Toàn bộ chunks | Giới hạn theo summary docs |
| Sufficiency check | Không có | Có (LLM kiểm tra) |

## Lưu ý

1. **Cần tạo summary documents** trước khi sử dụng hierarchical retrieval
2. **Fallback tự động** nếu không có summary docs
3. **ParadeDB BM25 index** cần được tạo cho bảng `summary_documents`
4. Có thể tắt hierarchical retrieval bằng cách set `use_hierarchical=False` trong `chat()`

## Troubleshooting

### Lỗi: "No summary documents found"
→ Cần tạo summary documents cho các parent documents

### Max score quá thấp (< 0.3)
→ Hệ thống tự động fallback sang tìm kiếm toàn bộ chunks

### Performance chậm
→ Kiểm tra indexes trên bảng `summary_documents`:
- Vector index (HNSW)
- BM25 index (ParadeDB)
- Foreign key index trên `document_id`
