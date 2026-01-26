# 🚀 RAG System - Production-Ready Microservice

Hệ thống RAG (Retrieval-Augmented Generation) được đóng gói thành **Production-Ready Microservice** với FastAPI, PostgreSQL (pgvector), Redis Queue và Docker.

## 🌟 Highlights

- ✅ **RESTful API** với FastAPI
- ✅ **Vector Database** với PostgreSQL + pgvector
- ✅ **Background Processing** với Redis Queue (RQ)
- ✅ **Multi-file Upload** - Upload nhiều files cùng lúc qua multipart-form data
- ✅ **Duplicate Detection** - SHA256 hash để tránh trùng lặp
- ✅ **Batch Tracking** - Redis tracking tổng thời gian xử lý của batch files
- ✅ **Docker Ready** - docker-compose để deploy một lệnh
- ✅ **Auto Documentation** - Swagger UI tích hợp sẵn
- ✅ **Performance Logging** - Chi tiết timing từng phase (Ingest, Chunking, Embedding, Database)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env và thêm:
# GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start Services

```bash
docker-compose up -d --build
```

Hệ thống sẽ khởi động:

- **FastAPI** (port 8000)
- **PostgreSQL** với pgvector (port 5432)
- **Redis** (port 6379)
- **RQ Worker** (background processing)

### 3. Access API

Mở browser: **http://localhost:8000/docs**

**Xong!** 🎉 API đã sẵn sàng.

## � Database Migrations

Khi cần chạy migration cho database:

### Cách 1: PowerShell (Windows)

```powershell
Get-Content migrations/add_file_size_and_hash.sql | docker exec -i rag_postgres psql -U rag_user -d rag_db
```

### Cách 2: Bash (Linux/Mac)

```bash
cat migrations/add_file_size_and_hash.sql | docker exec -i rag_postgres psql -U rag_user -d rag_db
```

### Cách 3: Trực tiếp trong container

```bash
docker exec -i rag_postgres psql -U rag_user -d rag_db -f /migrations/add_file_size_and_hash.sql
```

> **💡 Tip**: Migration files nằm trong thư mục `migrations/`. Chạy theo thứ tự từ cũ đến mới.

## 📚 API Endpoints

### 1. Upload & Process Documents

**POST** `/api/v1/process`

Upload một hoặc nhiều files (HTML) để xử lý:

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@file1.html" \
  -F "files=@file2.html" \
  -F "chunk_size=800" \
  -F "chunk_overlap=150"
```

**Features**:

- ✅ Multi-file upload
- ✅ Content-length validation (max 50MB)
- ✅ SHA256 duplicate detection
- ✅ Background processing với RQ
- ✅ Batch timing tracking

**Response**:

```json
{
  "total_files": 2,
  "results": [
    {
      "filename": "file1.html",
      "status": "processing",
      "job_id": "job_abc123",
      "document_id": 1,
      "message": "File uploaded successfully"
    }
  ]
}
```

### 2. Check Job Status

**GET** `/api/v1/jobs/{job_id}/status`

```bash
curl http://localhost:8000/api/v1/jobs/job_abc123/status
```

### 3. Search Documents

**POST** `/api/v1/search`

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hồ Chí Minh sinh năm nao",
    "top_k": 5
  }'
```

### 4. RAG Chat

**POST** `/api/v1/chat`

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Hồ Chí Minh sinh năm nao",
    "top_k": 10
  }'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Browser/API)                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   FastAPI        │ ← Port 8000
                    │   (REST API)     │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
         ┌──────────┐  ┌─────────┐  ┌─────────┐
         │PostgreSQL│  │  Redis  │  │ Worker  │
         │+pgvector │  │  Queue  │  │  (RQ)   │
         └──────────┘  └─────────┘  └─────────┘
              │             │             │
              │             └─────────────┘
              │           Queue Jobs
              │
         ┌────┴─────┐
         │          │
    Documents    Chunks
    (Metadata)   (Vectors)
```

**Data Flow**:

1. Client upload file(s) → FastAPI
2. FastAPI lưu temp file, tạo document record, queue job
3. Worker nhận job từ Redis Queue
4. Worker: Ingest → Chunking → Embedding → Save to PostgreSQL
5. Worker update batch tracking trong Redis
6. Worker cuối cùng log tổng thời gian batch

## 🔧 Tech Stack

- **API Framework**: FastAPI
- **Vector DB**: PostgreSQL 17 + pgvector
- **Queue**: Redis + RQ (Redis Queue)
- **Embedding**: Google Gemini API
- **File Processing**: BeautifulSoup4, MarkItDown
- **Deployment**: Docker + Docker Compose

## 📊 Performance Monitoring

Hệ thống tự động log timing cho từng phase:

```
======================================================================
✅ PROCESSING COMPLETED - Summary
======================================================================
📊 Document ID: 123
📊 Total chunks created: 45
⏱️  TOTAL TIME: 12.34s

📈 Time Breakdown:
   • Ingest:    3.21s (26.0%)
   • Chunking:  2.10s (17.0%)
   • Embedding: 5.89s (47.7%)
   • Database:  1.14s (9.2%)
======================================================================
```

**Batch Processing Log**:

```
======================================================================
🎉 BATCH COMPLETED - All 3 file(s) processed
======================================================================
📊 Batch ID: batch_a1b2c3d4e5f6
⏱️  TOTAL BATCH TIME: 45.67s

📈 Total Time Breakdown (All Files):
   • Ingest:    12.34s (27.0%)
   • Chunking:  8.56s (18.7%)
   • Embedding: 21.45s (47.0%)
   • Database:  3.32s (7.3%)

⚡ Average per file: 15.22s
======================================================================
```

## 🔄 Original Script (Legacy)

> **Note**: Script tương tác cũ vẫn có tại `src/main.py` (dùng Pinecone) nhưng **không khuyến nghị** sử dụng. Hãy dùng API microservice mới.

## 📁 Project Structure

```
RAG/
├── app/                           # Microservice source code
│   ├── api/                       # API routes & schemas
│   │   ├── routes.py              # REST endpoints
│   │   └── schemas.py             # Pydantic models
│   ├── database/                  # Database layer
│   │   ├── models.py              # SQLAlchemy models
│   │   └── connection.py          # DB connection
│   ├── services/                  # Business logic
│   │   ├── chunking_service.py    # Text chunking
│   │   ├── embedding_service.py   # Vector embeddings
│   │   ├── queue_service.py       # Redis queue
│   │   ├── search_service.py      # Vector search
│   │   └── rag_service.py         # RAG pipeline
│   ├── workers/                   # Background workers
│   │   └── process_worker.py      # Document processing
│   ├── config.py                  # Configuration
│   └── main.py                    # FastAPI app
├── migrations/                    # SQL migrations
│   └── add_file_size_and_hash.sql
├── data/
│   └── temp/                      # Temporary upload files
│       └── .gitkeep
├── docker-compose.yml             # Docker orchestration
├── Dockerfile                     # API container
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md
```

## ⚙️ Configuration

File `.env` cần có:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Database
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_db

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
```

### Chunking Parameters

Trong API request, bạn có thể điều chỉnh:

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "files=@file.html" \
  -F "chunk_size=800" \      # Kích thước chunk (chars)
  -F "chunk_overlap=150"     # Overlap giữa chunks (chars)
```

**Khuyến nghị**:

- `chunk_size`: 600-1000 chars
- `chunk_overlap`: 100-200 chars (15-20% của chunk_size)

## 🐛 Troubleshooting

### 1. Container không start

```bash
# Check logs
docker-compose logs -f api
docker-compose logs -f postgres
docker-compose logs -f worker

# Restart services
docker-compose restart
```

### 2. Migration chưa chạy

```bash
# Chạy migration
Get-Content migrations/add_file_size_and_hash.sql | docker exec -i rag_postgres psql -U rag_user -d rag_db
```

### 3. Worker không xử lý jobs

```bash
# Check worker logs
docker-compose logs -f worker

# Restart worker
docker-compose restart worker
```

### 4. File upload quá 50MB

```
❌ Error: Request quá lớn. Tối đa 50MB
```

**Giải pháp**: Tăng giới hạn trong [routes.py](app/api/routes.py) hoặc split file nhỏ hơn.

### 5. Duplicate file detected

```json
{
  "status": "duplicate",
  "message": "File already exists",
  "document_id": 123
}
```

**Lý do**: SHA256 hash trùng với document hiện có (cùng nội dung).

## 📊 Monitoring

### Check Services Health

```bash
# API health check
curl http://localhost:8000/health

# Check PostgreSQL
docker exec rag_postgres psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM documents;"

# Check Redis queue
docker exec rag_redis redis-cli LLEN rq:queue:process
```

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
```

### Database Queries

```bash
# Connect to PostgreSQL
docker exec -it rag_postgres psql -U rag_user -d rag_db

# Example queries
SELECT id, title, status FROM documents;
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM chunks WHERE document_id = 1;
```

## 🚀 Production Deployment

### Environment Variables

Tạo `.env.production`:

```env
GEMINI_API_KEY=prod_key_here
POSTGRES_PASSWORD=strong_password_here
DATABASE_URL=postgresql://user:pass@prod-db:5432/rag_db
```

### Docker Compose Production

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start with production config
docker-compose -f docker-compose.prod.yml up -d
```

### Scaling Workers

```bash
# Scale to 3 workers
docker-compose up -d --scale worker=3
```

## 🎯 Development

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL & Redis (Docker)
docker-compose up -d postgres redis

# Run API locally
uvicorn app.main:app --reload --port 8000

# Run worker locally
rq worker process --url redis://localhost:6379/0
```

### Run Tests

```bash
# TODO: Add tests
pytest tests/
```

## 📈 Performance Tips

1. **Tăng retrieval quality**:
   - Tăng `top_k` lên 15-20
   - Giảm `chunk_size` xuống 600

2. **Giảm processing time**:
   - Scale workers: `docker-compose up -d --scale worker=3`
   - Tối ưu chunk_size và overlap

3. **Monitoring batch jobs**:
   - Xem worker logs để track batch timing
   - Redis batch tracking tự động cleanup sau 24h

## 📞 Support

Nếu gặp vấn đề:

1. ✅ Check logs: `docker-compose logs -f`
2. ✅ Verify `.env` có đầy đủ API keys
3. ✅ Đảm bảo migrations đã chạy
4. ✅ Check services đang chạy: `docker-compose ps`
5. ✅ Restart services: `docker-compose restart`
