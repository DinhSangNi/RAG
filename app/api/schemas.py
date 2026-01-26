from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessDocumentRequest(BaseModel):
    """Request schema cho process document API (ingest + chunk combined)"""
    file_path: str = Field(..., description="Đường dẫn đến file cần xử lý")
    source_type: str = Field(default="local", description="Loại source: local, cloud, wikipedia")
    chunk_size: Optional[int] = Field(default=800, description="Kích thước chunk")
    chunk_overlap: Optional[int] = Field(default=150, description="Overlap giữa các chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata bổ sung")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "data/raw_data/wikipedia/Hồ_Chí_Minh.html",
                "source_type": "local",
                "chunk_size": 800,
                "chunk_overlap": 150,
                "metadata": {"category": "history"}
            }
        }


class JobResponse(BaseModel):
    """Response schema cho job status"""
    job_id: str = Field(..., description="ID của job")
    status: JobStatus = Field(..., description="Trạng thái job")
    message: Optional[str] = Field(default=None, description="Thông báo")
    document_id: Optional[str] = Field(default=None, description="ID của document (UUID)")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Tiến trình xử lý")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123xyz",
                "status": "processing",
                "message": "Đang xử lý file...",
                "document_id": 1,
                "progress": {"current": 50, "total": 100}
            }
        }


class FileUploadResult(BaseModel):
    """Result cho từng file upload"""
    filename: str = Field(..., description="Tên file gốc")
    status: str = Field(..., description="Trạng thái: processing, duplicate, failed")
    job_id: Optional[str] = Field(default=None, description="ID của job xử lý")
    document_id: Optional[str] = Field(default=None, description="ID của document (UUID)")
    message: Optional[str] = Field(default=None, description="Thông báo")


class MultiFileUploadResponse(BaseModel):
    """Response schema cho multi-file upload"""
    total_files: int = Field(..., description="Tổng số files được upload")
    results: List[FileUploadResult] = Field(..., description="Kết quả cho từng file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 2,
                "results": [
                    {
                        "filename": "document1.html",
                        "status": "processing",
                        "job_id": "process_abc123",
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "message": "Document đang được xử lý"
                    },
                    {
                        "filename": "document2.html",
                        "status": "duplicate",
                        "document_id": "550e8400-e29b-41d4-a716-446655440001",
                        "message": "File đã tồn tại trong hệ thống"
                    }
                ]
            }
        }


class DocumentResponse(BaseModel):
    """Response schema cho document"""
    id: str  # UUID
    file_path: str
    file_name: str
    source_type: str
    status: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    chunk_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class ChunkResponse(BaseModel):
    """Response schema cho chunk"""
    id: int
    document_id: str  # UUID
    content: str
    chunk_index: int
    section_id: Optional[int]
    h1: Optional[str]
    h2: Optional[str]
    h3: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    """Request schema cho search API"""
    query: str = Field(..., description="Câu hỏi cần search")
    top_k: int = Field(default=10, description="Số lượng kết quả trả về")
    document_ids: Optional[List[str]] = Field(default=None, description="Lọc theo document IDs (UUIDs)")
    search_type: str = Field(default="hybrid", description="Loại search: bm25, semantic, hybrid")
    bm25_weight: float = Field(default=0.6, description="Trọng số BM25 cho hybrid search")
    semantic_weight: float = Field(default=0.4, description="Trọng số semantic cho hybrid search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Hồ Chí Minh sinh năm nào?",
                "top_k": 10,
                "search_type": "hybrid",
                "bm25_weight": 0.6,
                "semantic_weight": 0.4
            }
        }


class SearchResult(BaseModel):
    """Response schema cho search result"""
    id: int
    content: str
    score: float
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None
    document_id: str  # UUID
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response schema cho search"""
    query: str
    results: List[SearchResult]
    total: int
    search_type: str


class ChatRequest(BaseModel):
    """Request schema cho chat API"""
    question: str = Field(..., description="Câu hỏi")
    document_ids: Optional[List[str]] = Field(default=None, description="Lọc theo document IDs (UUIDs)")
    verbose: bool = Field(default=False, description="Hiển thị context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Hồ Chí Minh sinh năm nào?",
                "verbose": False
            }
        }


class ChatResponse(BaseModel):
    """Response schema cho chat"""
    question: str
    answer: str
    metadata: Dict[str, Any]
