"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessDocumentRequest(BaseModel):
    """
    Request schema for document processing API (ingest + chunk combined)
    """
    file_path: str = Field(..., description="Path to file to process")
    source_type: str = Field(default="local", description="Source type: local, cloud, wikipedia")
    chunk_size: Optional[int] = Field(default=800, description="Chunk size")
    chunk_overlap: Optional[int] = Field(default=150, description="Overlap between chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

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
    """
    Response schema for job status
    """
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Job status")
    message: Optional[str] = Field(default=None, description="Status message")
    document_id: Optional[str] = Field(default=None, description="Document ID (UUID)")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Processing progress")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123xyz",
                "status": "processing",
                "message": "Processing file...",
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "progress": {"current": 50, "total": 100}
            }
        }


class FileUploadResult(BaseModel):
    """
    Result for individual file upload
    """
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Status: processing, duplicate, failed")
    job_id: Optional[str] = Field(default=None, description="Processing job ID")
    document_id: Optional[str] = Field(default=None, description="Document ID (UUID)")
    message: Optional[str] = Field(default=None, description="Status message")


class UpdateSummaryResponse(BaseModel):
    """
    Response schema for update summary API
    """
    status: str = Field(..., description="Status: updated, unchanged, duplicate")
    summary_id: str = Field(..., description="Summary document ID (UUID)")
    document_id: str = Field(..., description="Document ID (UUID)")
    job_id: Optional[str] = Field(default=None, description="Processing job ID (if any)")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "updated",
                "summary_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_id": "660e8400-e29b-41d4-a716-446655440001",
                "job_id": "update_abc123",
                "message": "Summary document is being updated"
            }
        }


class MultiFileUploadResponse(BaseModel):
    """
    Response schema for multi-file upload
    """
    total_files: int = Field(..., description="Total number of uploaded files")
    results: List[FileUploadResult] = Field(..., description="Results for each file")

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
                        "message": "Document is being processed"
                    },
                    {
                        "filename": "document2.html",
                        "status": "duplicate",
                        "document_id": "550e8400-e29b-41d4-a716-446655440001",
                        "message": "File already exists in system"
                    }
                ]
            }
        }


class DocumentResponse(BaseModel):
    """
    Response schema for document
    """
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
    """
    Response schema for chunk
    """
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
    """
    Request schema for search API
    """
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results to return")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs (UUIDs)")
    search_type: str = Field(default="hybrid", description="Search type: bm25, semantic, hybrid")
    bm25_weight: float = Field(default=0.6, description="BM25 weight for hybrid search")
    semantic_weight: float = Field(default=0.4, description="Semantic weight for hybrid search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "When was Hồ Chí Minh born?",
                "top_k": 10,
                "search_type": "hybrid",
                "bm25_weight": 0.6,
                "semantic_weight": 0.4
            }
        }


class SearchResult(BaseModel):
    """
    Response schema for search result
    """
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
    """
    Response schema for search
    """
    query: str
    results: List[SearchResult]
    total: int
    search_type: str


class ChatRequest(BaseModel):
    """
    Request schema for chat API
    """
    question: str = Field(..., description="Question to ask")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs (UUIDs)")
    verbose: bool = Field(default=False, description="Show context in response")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "When was Hồ Chí Minh born?",
                "verbose": False
            }
        }


class ChatResponse(BaseModel):
    """
    Response schema for chat
    """
    question: str
    answer: str
    metadata: Dict[str, Any]
