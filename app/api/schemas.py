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


class UpdateSummaryTextRequest(BaseModel):
    """Request schema for directly updating summary content by text."""
    summary_text: str = Field(..., description="New summary text content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata fields to merge into summary metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "summary_text": "Vo Nguyen Giap la Dai tuong dau tien cua Quan doi Nhan dan Viet Nam...",
                "metadata": {"source": "manual_update"}
            }
        }


class UpdateSummaryTextResponse(BaseModel):
    """Response schema for direct summary text update endpoint."""
    status: str = Field(..., description="Status: updated, unchanged")
    summary_id: str = Field(..., description="Summary document ID (UUID)")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "updated",
                "summary_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Summary document updated successfully"
            }
        }


class WikipediaFetchRequest(BaseModel):
    """Request schema for fetching a Wikipedia page HTML by title."""
    title: str = Field(..., description="Wikipedia page title to fetch")
    language: str = Field(default="vi", description="Wikipedia language code")
    auto_suggest: bool = Field(default=True, description="Enable wikipedia auto-suggest for title resolution")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Hồ Chí Minh",
                "language": "vi",
                "auto_suggest": True
            }
        }


class WikipediaFetchResponse(BaseModel):
    """Response schema for fetched Wikipedia HTML file."""
    requested_title: str = Field(..., description="Original title requested by user")
    resolved_title: str = Field(..., description="Resolved Wikipedia page title")
    language: str = Field(..., description="Wikipedia language code used")
    page_url: str = Field(..., description="Resolved Wikipedia page URL")
    file_path: str = Field(..., description="Saved HTML file path")
    file_name: str = Field(..., description="Saved HTML filename")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "requested_title": "Hồ Chí Minh",
                "resolved_title": "Hồ Chí Minh",
                "language": "vi",
                "page_url": "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ch%C3%AD_Minh",
                "file_path": "data/raw_data/wikipedia/Hồ_Chí_Minh.html",
                "file_name": "Hồ_Chí_Minh.html",
                "message": "Wikipedia HTML fetched successfully"
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
    bm25_weight: float = Field(default=0.5, description="BM25 weight for hybrid search")
    semantic_weight: float = Field(default=0.5, description="Semantic weight for hybrid search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "When was Hồ Chí Minh born?",
                "top_k": 10,
                "search_type": "hybrid",
                "bm25_weight": 0.5,
                "semantic_weight": 0.5
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
