from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Request
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
import hashlib
from pathlib import Path
import aiofiles

from app.database import get_db
from app.database.models import Document, Chunk
from app.api.schemas import (
    JobResponse, 
    DocumentResponse,
    JobStatus,
    SearchRequest,
    SearchResult,
    SearchResponse,
    ChatRequest,
    ChatResponse,
    FileUploadResult,
    MultiFileUploadResponse
)
from app.services.queue_service import queue_process_job, get_job_status
from app.services.search_service import get_search_service
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/api/v1", tags=["documents"])


def normalize_file_path(file_path: str) -> str:
    # Nếu là đường dẫn Windows (có dấu \), chuyển về dấu / của Linux
    file_path = file_path.replace("\\", "/")

    # Nếu file_path đã là đường dẫn tuyệt đối bắt đầu bằng /app, giữ nguyên
    if file_path.startswith("/app/"):
        return file_path
    
    # Nếu chạy trong Docker, đảm bảo nó trỏ vào folder /app/data
    if os.path.exists('/app'):
        # Nếu user truyền "data/file.pdf", ta không muốn thành "/app/data/file.pdf" bị lặp 
        # nên ta làm sạch nó
        clean_path = file_path.lstrip('/')
        if not clean_path.startswith('app/'):
            return os.path.join("/app", clean_path)
            
    return file_path


def calculate_file_hash(file_path: str) -> str:
    """Tính SHA256 hash của file content"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Đọc file theo chunks để xử lý file lớn
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def calculate_upload_file_hash(file: UploadFile) -> str:
    """Tính SHA256 hash của upload file content"""
    sha256_hash = hashlib.sha256()
    # Reset file pointer về đầu
    await file.seek(0)
    # Đọc file theo chunks
    while chunk := await file.read(4096):
        sha256_hash.update(chunk)
    # Reset lại file pointer về đầu sau khi hash
    await file.seek(0)
    return sha256_hash.hexdigest()


def ensure_temp_directory() -> Path:
    """Đảm bảo thư mục temp tồn tại và trả về path"""
    # Kiểm tra xem có đang chạy trong Docker không
    if os.path.exists('/app'):
        temp_dir = Path('/app/data/temp')
    else:
        temp_dir = Path('data/temp')
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def check_duplicate_document(db: Session, file_path: str, file_size: int, content_hash: str) -> Document | None:
    """Kiểm tra xem đã có document trùng lặp dựa trên file_size và content_hash
    
    Returns:
        Document nếu tìm thấy duplicate, None nếu không có
    """
    # Tìm tất cả documents có cùng file_size
    candidates = db.query(Document).filter(
        Document.file_size == file_size,
        Document.file_path != file_path  # Loại trừ chính file này (nếu update)
    ).all()
    
    # Nếu có candidate, so sánh content_hash
    for candidate in candidates:
        # So sánh giá trị thực tế, không phải Column object
        if str(candidate.content_hash) == str(content_hash):
            return candidate
    
    return None


@router.post("/process", response_model=MultiFileUploadResponse)
async def process_document(
    request: Request,
    files: List[UploadFile] = File(...),
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    db: Session = Depends(get_db)
):
    """
    Xử lý toàn bộ document: ingest + chunk + embeddings
    - Nhận file(s) qua multipart-form data
    - Kiểm tra content-length <= 50MB
    - Stream files vào thư mục temp
    - Hash content và kiểm tra duplicate
    - Tạo document record
    - Queue job để xử lý: clean → markdown → chunk → embeddings → save
    """
    # Kiểm tra content-length header
    content_length = request.headers.get('content-length')
    if content_length:
        content_length_bytes = int(content_length)
        max_size = 50 * 1024 * 1024  # 50MB
        if content_length_bytes > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request quá lớn. Tối đa 50MB, nhận được {content_length_bytes / 1024 / 1024:.2f}MB"
            )
    
    if not files:
        raise HTTPException(status_code=400, detail="Không có file nào được upload")
    
    # Tạo batch_id để track tất cả files trong request này
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    # Đảm bảo thư mục temp tồn tại
    temp_dir = ensure_temp_directory()
    
    # Danh sách kết quả cho từng file
    results = []
    
    for upload_file in files:
        # Validate file name
        if not upload_file.filename:
            continue
            
        # Tạo unique filename với UUID để tránh conflict
        file_extension = Path(upload_file.filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        temp_file_path = temp_dir / unique_filename
        
        # Stream file vào thư mục temp
        try:
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                while content := await upload_file.read(4096):
                    await out_file.write(content)
        except Exception as e:
            # Xóa file nếu có lỗi
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file: {str(e)}")
        
        # Reset file pointer về đầu để hash
        await upload_file.seek(0)
        
        # Tính file_size và content_hash
        file_size = os.path.getsize(temp_file_path)
        content_hash = calculate_file_hash(str(temp_file_path))
        file_name = upload_file.filename
        
        # Normalize file path cho Docker/local compatibility
        normalized_path = normalize_file_path(str(temp_file_path))
        
        # Kiểm tra duplicate dựa trên file_size và content_hash
        duplicate_doc = check_duplicate_document(db, normalized_path, file_size, content_hash)
        
        if duplicate_doc:
            # File này đã có trong DB với nội dung giống hệt (khác path)
            # Xóa file temp
            if temp_file_path.exists():
                temp_file_path.unlink()
            
            results.append(FileUploadResult(
                filename=file_name,
                status="duplicate",
                message=f"File đã tồn tại trong hệ thống (document_id: {duplicate_doc.id})",
                document_id=str(duplicate_doc.id)
            ))
            continue
        
        # Kiểm tra xem document với path này đã tồn tại chưa
        existing_document = db.query(Document).filter(Document.file_path == normalized_path).first()
        
        if existing_document:
            # Kiểm tra xem nội dung file có thay đổi không
            existing_size = getattr(existing_document, 'file_size', None)
            existing_hash = getattr(existing_document, 'content_hash', None)
            file_unchanged = (existing_size == file_size and existing_hash == content_hash)
            
            if file_unchanged:
                # File không thay đổi, chỉ cập nhật status để xử lý lại
                setattr(existing_document, 'status', 'pending')
            else:
                # File đã thay đổi, cập nhật thông tin mới
                setattr(existing_document, 'status', 'pending')
                setattr(existing_document, 'file_size', file_size)
                setattr(existing_document, 'content_hash', content_hash)
            
            db.commit()
            db.refresh(existing_document)
            document = existing_document
        else:
            # Tạo document record mới
            document = Document(
                file_path=normalized_path,
                file_name=file_name,
                source_type="upload",
                status="pending",
                file_size=file_size,
                content_hash=content_hash,
                meta_data={"original_filename": file_name}
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
        
        # Tạo job ID
        job_id = f"process_{uuid.uuid4().hex[:8]}"
        
        # Get document ID as str (UUID)
        doc_id = str(document.id)  # type: ignore[arg-type]
        
        # Queue job để xử lý toàn bộ (ingest + chunk)
        queue_process_job(
            job_id=job_id,
            document_id=doc_id,
            file_path=normalized_path,
            source_type="upload",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_id=batch_id,
            total_files=len(files)
        )
        
        results.append(FileUploadResult(
            filename=file_name,
            status="processing",
            job_id=job_id,
            document_id=doc_id,
            message="Document đang được xử lý (ingest + chunk)"
        ))
    
    return MultiFileUploadResponse(
        total_files=len(files),
        results=results
    )


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status_endpoint(job_id: str):
    """
    Lấy trạng thái của một job
    """
    status = get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Job không tồn tại: {job_id}")
    
    # Ensure status is JobStatus enum
    status_value = status.get('status', 'pending')
    if isinstance(status_value, str):
        status_value = JobStatus(status_value)
    
    # Get document_id as str (UUID)
    doc_id = status.get('document_id')
    if doc_id is not None:
        doc_id = str(doc_id)
    
    # Get progress with proper type
    progress = status.get('progress')
    if progress is not None and not isinstance(progress, dict):
        progress = None
    
    return JobResponse(
        job_id=status['job_id'],
        status=status_value,
        message=status.get('message'),
        document_id=doc_id,
        progress=progress
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách documents
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    # Đếm chunks cho mỗi document
    result = []
    for doc in documents:
        chunk_count = db.query(Chunk).filter(Chunk.document_id == doc.id).count()
        doc_dict = {
            "id": str(doc.id),  # type: ignore[arg-type]
            "file_path": str(doc.file_path),  # type: ignore[arg-type]
            "file_name": str(doc.file_name),  # type: ignore[arg-type]
            "source_type": str(doc.source_type),  # type: ignore[arg-type]
            "status": str(doc.status),  # type: ignore[arg-type]
            "metadata": doc.meta_data,  # type: ignore[arg-type]
            "created_at": doc.created_at,  # type: ignore[arg-type]
            "chunk_count": chunk_count
        }
        result.append(DocumentResponse(**doc_dict))
    
    return result


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Lấy thông tin chi tiết một document
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document không tồn tại: {document_id}")
    
    chunk_count = db.query(Chunk).filter(Chunk.document_id == document.id).count()
    
    return DocumentResponse(
        id=str(document.id),  # type: ignore[arg-type]
        file_path=str(document.file_path),  # type: ignore[arg-type]
        file_name=str(document.file_name),  # type: ignore[arg-type]
        source_type=str(document.source_type),  # type: ignore[arg-type]
        status=str(document.status),  # type: ignore[arg-type]
        metadata=document.meta_data,  # type: ignore[arg-type]
        created_at=document.created_at,  # type: ignore[arg-type]
        chunk_count=chunk_count
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Xóa một document và tất cả chunks của nó
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document không tồn tại: {document_id}")
    
    db.delete(document)
    db.commit()
    
    return {"message": f"Document {document_id} đã được xóa"}


# ============================================================================
# SEARCH & RAG ENDPOINTS
# ============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search chunks using BM25, semantic, or hybrid search
    """
    search_service = get_search_service(db)
    
    if request.search_type == "bm25":
        results = search_service.bm25_search(
            query=request.query,
            k=request.top_k,
            document_ids=request.document_ids
        )
    elif request.search_type == "semantic":
        results = search_service.semantic_search(
            query=request.query,
            k=request.top_k,
            document_ids=request.document_ids
        )
    else:  # hybrid
        results = search_service.hybrid_search(
            query=request.query,
            k=request.top_k,
            bm25_weight=request.bm25_weight,
            semantic_weight=request.semantic_weight,
            document_ids=request.document_ids
        )
    
    # Format results
    search_results = [
        SearchResult(
            id=r['id'],
            content=r['content'],
            score=r.get('fused_score') or r.get('score', 0.0),
            h1=r.get('h1'),
            h2=r.get('h2'),
            h3=r.get('h3'),
            document_id=str(r['document_id']),
            chunk_index=r['chunk_index'],
            metadata=r.get('metadata')
        )
        for r in results
    ]
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        total=len(search_results),
        search_type=request.search_type
    )


@router.post("/chat", response_model=ChatResponse)
async def rag_chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    RAG chat with advanced retrieval and answer generation
    """
    rag_service = get_rag_service(db)
    
    result = rag_service.chat(
        question=request.question,
        document_ids=request.document_ids,
        verbose=request.verbose
    )
    
    return ChatResponse(
        question=request.question,
        answer=result['answer'],
        metadata=result['metadata']
    )

