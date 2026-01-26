import redis
from rq import Queue
from app.config import settings

# Tạo Redis connection
redis_conn = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True
)

# Tạo queue
process_queue = Queue('process', connection=redis_conn)


def queue_process_job(
    job_id: str, 
    document_id: str, 
    file_path: str, 
    source_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    batch_id: str = None,
    total_files: int = 1
):
    """
    Queue job để xử lý toàn bộ: ingest + chunk
    """
    from app.workers.process_worker import process_document
    
    # Nếu có batch_id, khởi tạo batch tracking trong Redis
    if batch_id:
        # Tạo hoặc increment counter cho batch
        batch_key = f"batch:{batch_id}"
        if not redis_conn.exists(batch_key):
            # Khởi tạo batch tracking - convert all values to string
            redis_conn.hset(batch_key, mapping={
                "total_files": str(total_files),
                "completed_files": "0",
                "ingest_total": "0.0",
                "chunking_total": "0.0",
                "embedding_total": "0.0",
                "database_total": "0.0",
                "processing_total": "0.0"
            })
            redis_conn.expire(batch_key, 86400)  # 24 hours
    
    job = process_queue.enqueue(
        process_document,
        kwargs={
            "job_id": job_id,
            "document_id": document_id,
            "file_path": file_path,
            "source_type": source_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "batch_id": batch_id
        },
        job_id=job_id,
        job_timeout='2h'
    )
    
    # Lưu mapping
    redis_conn.set(f"job:{job_id}", job.id, ex=86400)
    
    return job.id


def get_job_status(job_id: str):
    """
    Lấy trạng thái của một job từ Redis
    """
    from rq.job import Job
    
    # Lấy RQ job ID
    rq_job_id = redis_conn.get(f"job:{job_id}")
    
    if not rq_job_id:
        return None
    
    try:
        # Convert redis response to string if needed
        job_id_str = str(rq_job_id) if rq_job_id else None
        if not job_id_str:
            return None
            
        job = Job.fetch(job_id_str, connection=redis_conn)
        
        status_map = {
            'queued': 'pending',
            'started': 'processing',
            'finished': 'completed',
            'failed': 'failed'
        }
        
        status = status_map.get(job.get_status(), 'pending')
        
        result = {
            'job_id': job_id,
            'status': status,
            'message': None,
            'document_id': None,
            'progress': None
        }
        
        # Nếu job hoàn thành, lấy result
        if status == 'completed' and job.result:
            # Update with job result but ensure types are correct
            if isinstance(job.result, dict):
                if 'message' in job.result:
                    result['message'] = str(job.result['message'])
                if 'document_id' in job.result:
                    result['document_id'] = int(job.result['document_id']) if job.result['document_id'] else None
                if 'progress' in job.result and isinstance(job.result['progress'], dict):
                    result['progress'] = job.result['progress']
        
        # Nếu job failed, lấy error
        if status == 'failed':
            result['message'] = str(job.exc_info) if job.exc_info else "Job failed"
        
        # Lấy meta data (progress)
        if job.meta:
            if 'progress' in job.meta and isinstance(job.meta['progress'], dict):
                result['progress'] = job.meta['progress']
            if 'document_id' in job.meta:
                result['document_id'] = int(job.meta['document_id']) if job.meta['document_id'] else None
        
        return result
        
    except Exception as e:
        return {
            'job_id': job_id,
            'status': 'failed',
            'message': str(e)
        }
