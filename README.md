# 🚀 RAG System - Hệ thống Hỏi Đáp Thông Minh

Hệ thống RAG (Retrieval-Augmented Generation) sử dụng Hybrid Search (BM25 + Semantic) và Google Gemini để trả lời câu hỏi dựa trên dữ liệu Wikipedia.

> **🔄 UPDATE**: Hệ thống đã được chuyển từ ChromaDB sang **Pinecone** - xem chi tiết tại [PINECONE_MIGRATION.md](PINECONE_MIGRATION.md)

## 📋 Yêu Cầu

```bash
pip install langchain langchain-community langchain-google-genai
pip install pinecone-client langchain-pinecone
pip install beautifulsoup4 requests
pip install rank_bm25
pip install python-dotenv
```

hoặc chạy

```bash
pip install -r requirements.txt
```

## ⚙️ Cấu Hình

1. **Tạo file `.env`** từ template:

   ```bash
   copy .env.example .env
   ```

2. **Điền API Keys** vào file `.env`:

   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Lấy Pinecone API Key**:
   - Đăng ký tại [https://www.pinecone.io/](https://www.pinecone.io/)
   - Tạo API key từ dashboard

## 🏃 Cách Chạy Chương Trình

### Chạy Menu Chính

```bash
python -m src.main
```

Lưu ý set lại GEMINII_API_KEY trong file text_chunker.py

Bạn sẽ thấy menu với 5 tùy chọn:

```
================================================================================
CHỌN CHỨC NĂNG:
================================================================================
1. 📝 Chuẩn bị data
2. 🔪 Chunking và lưu vào Vector DB
3. 💬 RAG Chat (Interactive)
4. 🚀 Chạy cả hai (Chunking → Chat)
0. ❌ Thoát
================================================================================
```

## 📖 Hướng Dẫn Sử Dụng

### Option 1: 📝 Chuẩn Bị Data

**Chức năng:** Lấy dữ liệu từ Wikipedia và xử lý thành Markdown

**Các bước:**

1. Nhập từ khóa tìm kiếm (ví dụ: "Hồ Chí Minh", "Võ Nguyên Giáp")
2. Hệ thống sẽ:
   - Tải HTML từ Wikipedia
   - Làm sạch HTML
   - Chuyển đổi sang Markdown chuẩn hóa

**Kết quả:** File `.md` được lưu trong `data/processed_data/`

---

### Option 2: 🔪 Chunking và Lưu vào Vector DB

**Chức năng:** Tách văn bản thành chunks và lưu vào Pinecone

**Cách hoạt động:**

- Đọc tất cả file `.md` trong `data/processed_data/`
- Tách theo Markdown headers (h1, h2)
- Lưu vào Pinecone với embedding Google AI
- Tạo file pickle cho BM25 search

**Kết quả:**

- Pinecone Index: `knowledge-base`
- Pickle file: `data/chunks/knowledge_base_chunks.pkl`

**Index name:** `knowledge-base` (mặc định)

---

### Option 3: 💬 RAG Chat (Interactive)

**Chức năng:** Hỏi đáp tương tác với AI

**Yêu cầu:** Phải chạy Option 2 trước để có vector DB

**Cách sử dụng:**

```
❓ Câu hỏi: Hồ Chí Minh sinh năm nào?
💡 TRẢ LỜI: Hồ Chí Minh sinh vào ngày 19 tháng 5 năm 1890.

❓ Câu hỏi: Võ Nguyên Giáp sinh ngày nào?
💡 TRẢ LỜI: Võ Nguyên Giáp sinh ngày 25 tháng 8 năm 1911.
```

**Lệnh đặc biệt:**

- `verbose` - Bật/tắt hiển thị context được retrieve (các chunks được get ra)
- `quit` hoặc `exit` - Thoát chương trình

**Tham số:**

- **Model:** gemini-2.5-flash-lite
- **Top K:** 10 chunks
- **BM25 Weight:** 0.5
- **Semantic Weight:** 0.5

---

### Option 4: 🚀 Chạy Cả Hai (Full Pipeline)

**Chức năng:** Chạy tuần tự Option 1 → Option 2 → Option 3

**Quy trình:**

1. Chuẩn bị data từ Wikipedia
2. Chunking và lưu vào Vector DB
3. Mở RAG Chat để hỏi đáp

**Phù hợp cho:** Lần đầu chạy hoặc muốn cập nhật toàn bộ dữ liệu

---

## 📁 Cấu Trúc Thư Mục

```
RAG/
├── .env.example                     # Template cho environment variables
├── .env                             # API Keys (không commit!)
├── data/
│   ├── raw_data/wikipedia/          # HTML gốc từ Wikipedia
│   ├── processed_data/              # File Markdown đã xử lý
│   └── chunks/                      # Pickle files cho BM25
│       └── knowledge_base_chunks.pkl
├── src/
│   ├── main.py                      # File chính
│   ├── rag_chat.py                  # RAG Chat logic
│   ├── chunking/
│   │   └── text_chunker.py          # Chunking logic
│   ├── ingestion/
│   │   └── get_data_from_wikipedia.py
│   └── preprocessing/
│       ├── html_cleaner.py
│       └── normalize_markdown.py
├── README.md
└── PINECONE_MIGRATION.md            # Hướng dẫn migration
```

## ⚙️ Cấu Hình

### Hybrid Search Weights

Trong `src/rag_chat.py`, dòng 152:

```python
bm25_weight=0.5,      # Keyword search
semantic_weight=0.5   # Semantic search
```

**Điều chỉnh:**

- Tăng `bm25_weight` → Ưu tiên khớp từ khóa chính xác
- Tăng `semantic_weight` → Ưu tiên hiểu nghĩa ngữ cảnh

### Chunk Size

Trong `src/main.py`, dòng 73:

```python
chunker = HybridSectionChunker(chunk_size=800, chunk_overlap=150)
```

**Tham số:**

- `chunk_size`: Kích thước chunk tối đa (ký tự)
- `chunk_overlap`: Số ký tự chồng lắp giữa các chunk

## 🐛 Xử Lý Lỗi

### Lỗi: File pkl không tồn tại

```
❌ Lỗi: [Errno 2] No such file or directory: 'data/chunks\\knowledge_base_chunks.pkl'
```

**Giải pháp:** Chạy Option 2 để tạo chunks và upload lên Pinecone

### Lỗi: Invalid Pinecone API Key

```
❌ Lỗi: Invalid API key
```

**Giải pháp:**

1. Kiểm tra file `.env`
2. Đảm bảo `PINECONE_API_KEY` được set đúng
3. Không có khoảng trắng thừa

### Lỗi: Không trả lời được câu hỏi

**Nguyên nhân:** Query không match với chunks

**Giải pháp:**

1. Bật `verbose` mode để xem context
2. Điều chỉnh weights (tăng semantic_weight)
3. Tăng `top_k` để retrieve nhiều chunks hơn

### Lỗi: Index not found

```
❌ Lỗi: Index 'knowledge-base' not found
```

**Giải pháp:** Chạy Option 2 để tạo index trên Pinecone

## 📝 Ví Dụ Câu Hỏi

```
✅ Hồ Chí Minh sinh năm nào?
✅ Võ Nguyên Giáp sinh ngày nào?
✅ Phạm Văn Đồng là ai?
✅ Hồ Chí Minh có tên khai sinh là gì?
✅ Võ Nguyên Giáp tham gia trận chiến nào?
✅ Hồ Chí Minh đã đi qua những nước nào?
```

## 🔧 API Key

File sử dụng Google Gemini API. API key được hardcode trong:

- `src/chunking/text_chunker.py` (line 12)
- `src/rag_chat.py` (line 11)

**Khuyến nghị:** Chuyển sang dùng biến môi trường:

```python
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

## 🎯 Kiến Trúc Hệ Thống

### 1. Data Pipeline

```
Wikipedia → HTML → Cleaned HTML → Normalized Markdown → Chunks
```

### 2. Chunking Strategy

- **Markdown Header Splitter**: Tách theo headers (h1, h2)
- **Recursive Character Splitter**: Tách sections lớn thành chunks nhỏ hơn
- **Metadata**: Lưu thông tin headers, source file, section ID

### 3. Hybrid Search

- **BM25 Retriever**: Keyword-based search (sparse retrieval)
- **Semantic Retriever**: Vector similarity search (dense retrieval)
- **Ensemble Retriever**: Kết hợp 2 phương pháp với weights

### 4. RAG Pipeline

```
Query → Query Expansion → Hybrid Search → Context Formatting → LLM Generation
```

## 🚀 Quick Start

**Chạy lần đầu:**

```bash
# Bước 1: Cài đặt dependencies
pip install -r requirements.txt

# Bước 2: Chạy chương trình
python -m src.main

# Bước 3: Chọn option 4 (Full Pipeline)
# Nhập từ khóa: Hồ Chí Minh
# Đợi xử lý...
# Bắt đầu hỏi đáp!
```

## 📊 Performance Tips

1. **Tăng retrieval quality:**
   - Tăng `top_k` lên 15-20
   - Tăng `semantic_weight` lên 0.6-0.7

2. **Giảm latency:**
   - Giảm `top_k` xuống 5
   - Cache chunks trong memory

3. **Cải thiện chunking:**
   - Giảm `chunk_size` xuống 500-600 (chunks nhỏ hơn, chính xác hơn)
   - Tăng `chunk_overlap` lên 200 (giữ context tốt hơn)

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:

1. ✅ Đã cài đặt đủ thư viện
2. ✅ Có API key hợp lệ
3. ✅ Đã chạy Option 2 trước khi chat
4. ✅ File markdown tồn tại trong `data/processed_data/`
5. ✅ Có kết nối internet (để gọi API)
