"""
Combined worker for processing documents: ingest + chunk in one job
"""
import os
import re
import time
from pathlib import Path
from bs4 import BeautifulSoup
import markitdown
from rq import get_current_job
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database.models import Document, Chunk
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service


# ============================================================================
# HTML Cleaning Functions (from src/preprocessing/html_cleaner.py)
# ============================================================================

def clean_wikipedia_html(html_file_path):
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    content = soup.find("div", class_="mw-parser-output")

    if not content:
        print("Không tìm thấy vùng nội dung chính!")
        return
    
    # Remove unwanted tags and elements
    tags_without_class = ['audio', 'style', 'img', 'sup', 'link', 'input']
    for tag in tags_without_class:
        for element in content.find_all(tag):
            element.decompose()
    
    # Remove specific tags with certain classes
    tags_with_class = [
        ('ol', 'references'),
        ('span', 'mw-editsection'),
        ('div', 'hatnote'),
        ('div', 'navbox'),
        ('div', 'navbox-styles'),
        ('div', 'metadata'),
        ('div', 'toc'), ## remove table of contents (left side) 
        ('table', 'navbox-inner'),
        ('table', 'navbox'),
        ('table', 'sidebar'),
        ('table', 'infobox'), 
        ('table', 'metadata'),
        ('span', 'languageicon'),
        ('span', 'tocnumber'),
        ('span', 'toctext'),
        ('span', 'reference-accessdate'),
        ('span', 'Z3988'),
        ('cite', None),
    ]
    
    for tag, class_name in tags_with_class:
        if class_name:
            for element in content.find_all(tag, class_=class_name):
                element.decompose()
        else:
            for element in content.find_all(tag):
                element.decompose()

    for table in content.find_all('table', class_="cquote"):
        quote_text = table.get_text(separator=" ", strip=True)
        p = soup.new_tag('p')
        p.string = quote_text
        table.replace_with(p)
    
    for p in content.find_all('p'):
        if not p.get_text(strip=True):
            p.decompose()
    
    for span in content.find_all('span'):
        if span.get('id') and not span.get_text(strip=True):
            span.decompose()
    
    for figure in content.find_all('figure'):
        figcaption = figure.find('figcaption')
        if figcaption:
            new_p = soup.new_tag('p')
            new_p.string = f"[Hình ảnh: {figcaption.get_text(strip=True)}]"
            figure.replace_with(new_p)
        else:
            figure.decompose()

    for a_tag in content.find_all('a'):
        a_tag.unwrap()
    
    for span in content.find_all('span'):
        span.unwrap()
    
    for tag in content.find_all(['b']):
        tag.unwrap()

    sections_to_kill = [
        "Tham_khảo", 
        "Tài liệu tham khảo",
        "Chú giải",
        "Liên_kết_ngoài", 
        "Danh_mục", 
        "Ghi_chú", 
        "Thư_mục_hậu_cần",
        "Đọc_thêm",
        "Chú_thích",
        "Thư_mục",
        "Nguồn_thứ_cấp",
        "Nguồn_sơ_cấp",
        "Nguồn_trích_dẫn",
        "Diễn_văn_của_Hồ_Chí_Minh",
        "Tác_phẩm_của_Hồ_Chí_Minh",
        "Viết_về_Hồ_Chí_Minh",
        "Những_người_từng_gặp_Hồ_Chí_Minh_kể_về_ông"
    ]

    for section_id in sections_to_kill:
        header = content.find(['h2', 'h3'], id=section_id)
        if header:
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                sibling.decompose() 
            header.decompose()
    
    # Remove empty <li> elements
    for li in content.find_all('li'):
        if not li.get_text(strip=True):
            li.decompose()
    
    # Clean attributes
    for tag in content.find_all(True):
        if tag.has_attr('class'):
            del tag['class']
        if tag.has_attr('style'):
            del tag['style']
        if tag.has_attr('id') and tag.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            del tag['id']
        if tag.has_attr('dir'):
            del tag['dir']
        if tag.has_attr('lang'):
            del tag['lang']

    input_file_name = Path(html_file_path).stem
    temp_html_file_path = f"data/raw_data/wikipedia/temp_clean_html/{input_file_name}.html"
    os.makedirs(os.path.dirname(temp_html_file_path), exist_ok=True)
    with open(temp_html_file_path, "w", encoding="utf-8") as f:
        f.write(str(content))

    print(f"Đã lưu HTML đã làm sạch vào: {temp_html_file_path}")

    return temp_html_file_path


# ============================================================================
# Markdown Normalization Functions (from src/preprocessing/normalize_markdown.py)
# ============================================================================

def normalize_markdown(md_text):
    lines = md_text.split('\n')
    normalized_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Nếu là header, giữ nguyên
        if line.strip().startswith('#'):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Nếu là bullet point, table, hoặc dòng trống, giữ nguyên
        if (line.strip().startswith(('* ', '- ', '+ ', '|', '>')) or 
            re.match(r'^\s*\d+\.', line) or
            line.strip() == '' or
            line.strip().startswith(':')):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Gộp các dòng liên tiếp không phải là đoạn đặc biệt
        paragraph = line
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Dừng nếu gặp dòng trống, header, bullet, table
            if (next_line.strip() == '' or 
                next_line.strip().startswith(('#', '* ', '- ', '+ ', '|', '>', ':')) or
                re.match(r'^\s*\d+\.', next_line)):
                break
            # Gộp dòng
            paragraph += ' ' + next_line.strip()
            i += 1
        
        normalized_lines.append(paragraph)
    
    # Join và làm sạch khoảng trắng thừa
    result = '\n'.join(normalized_lines)
    
    # Chuẩn hóa bullet points: chuyển tất cả thành *
    result = re.sub(r'^\s*[-+]\s+', '* ', result, flags=re.MULTILINE)
    
    # Loại bỏ khoảng trắng thừa ở cuối dòng
    result = re.sub(r' +\n', '\n', result)
    
    # Chuẩn hóa block quotes: chuyển : thành >
    result = re.sub(r'^:   \*', '>   *', result, flags=re.MULTILINE)
    result = re.sub(r'^:\s+', '> ', result, flags=re.MULTILINE)
    
    # Đảm bảo có dòng trống trước header
    result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', result)
    # Loại bỏ nhiều dòng trống liên tiếp (giữ tối đa 2)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def convert_html_to_normalized_md(html_file_path, output_md_file_path=None):
    md = markitdown.MarkItDown()   
    rs = md.convert(html_file_path)
    normalized_md = normalize_markdown(rs.text_content)

    if output_md_file_path is None:
        name = Path(html_file_path).stem
        output_md_file_path = "data/processed_data/{}.md".format(name)
    
    os.makedirs(os.path.dirname(output_md_file_path), exist_ok=True)
    with open(output_md_file_path, "w", encoding="utf-8") as f:
        f.write(normalized_md)

    print(f"Đã lưu markdown đã chuẩn hóa vào: {output_md_file_path}")

    return output_md_file_path


# Tạo database session cho worker
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def process_document(
    job_id: str, 
    document_id: str, 
    file_path: str, 
    source_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    batch_id: str = None
):
    """
    Worker function để xử lý toàn bộ: ingest + chunk + embeddings
    Workflow:
    1. Clean HTML (nếu là HTML)
    2. Convert to Markdown
    3. Normalize Markdown
    4. Chunk document
    5. Tạo embeddings
    6. Lưu vào PostgreSQL
    """
    job = get_current_job()
    db = SessionLocal()
    
    # Import redis để update batch tracking
    import redis
    from app.config import settings
    redis_conn = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    
    # Initialize timing dictionary
    timing_stats = {
        'total_start': time.time(),
        'phases': {}
    }
    
    try:
        # Initialize progress
        if job:
            job.meta['document_id'] = document_id
            job.meta['status'] = 'processing'
            job.meta['progress'] = {'step': 'starting', 'current': 0, 'total': 100}
            job.save_meta()
        
        # Lấy document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"Document không tồn tại: {document_id}")
        
        document.status = "processing"  # type: ignore[assignment]
        db.commit()
        
        # ========================================================================
        # PHASE 1: INGEST (0-30%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"📦 PHASE 1: INGEST - Starting for document {document_id}")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 1: Clean HTML nếu file là .html
        if job:
            job.meta['progress'] = {'step': 'cleaning_html', 'current': 5, 'total': 100}
            job.save_meta()
        
        cleaned_file_path = file_path
        if file_path.endswith('.html'):
            step_start = time.time()
            print(f"🧹 Cleaning HTML: {file_path}")
            cleaned_file_path = clean_wikipedia_html(file_path)
            step_duration = time.time() - step_start
            print(f"✅ Cleaned HTML: {cleaned_file_path}")
            print(f"⏱️  HTML Cleaning took: {step_duration:.2f}s")
            timing_stats['phases']['html_cleaning'] = step_duration
        
        # Bước 2: Convert to Markdown
        if job:
            job.meta['progress'] = {'step': 'converting_to_markdown', 'current': 15, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        print(f"📝 Converting to Markdown: {cleaned_file_path}")
        md_file_path = convert_html_to_normalized_md(cleaned_file_path)
        step_duration = time.time() - step_start
        print(f"✅ Converted to Markdown: {md_file_path}")
        print(f"⏱️  Markdown Conversion took: {step_duration:.2f}s")
        timing_stats['phases']['markdown_conversion'] = step_duration
        
        # Bước 3: Update document metadata
        if job:
            job.meta['progress'] = {'step': 'updating_metadata', 'current': 30, 'total': 100}
            job.save_meta()
        
        meta_data = document.meta_data or {}  # type: ignore[assignment]
        if isinstance(meta_data, dict):
            meta_data['processed_file'] = md_file_path
            meta_data['original_file'] = file_path
            document.meta_data = meta_data  # type: ignore[assignment]
        db.commit()
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['ingest_total'] = phase_duration
        print(f"✅ Ingest phase completed")
        print(f"⏱️  Total INGEST time: {phase_duration:.2f}s")
        
        # ========================================================================
        # PHASE 2: CHUNKING (30-50%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"🔪 PHASE 2: CHUNKING - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 4: Load markdown content
        if job:
            job.meta['progress'] = {'step': 'loading_file', 'current': 35, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        with open(md_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        step_duration = time.time() - step_start
        print(f"📄 Loaded file: {md_file_path} ({len(text)} chars)")
        print(f"⏱️  File Loading took: {step_duration:.2f}s")
        timing_stats['phases']['file_loading'] = step_duration
        
        # Bước 5: Chunk document
        if job:
            job.meta['progress'] = {'step': 'chunking', 'current': 40, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        chunking_service = get_chunking_service(chunk_size, chunk_overlap)
        chunks = chunking_service.chunk_markdown(text, md_file_path)
        step_duration = time.time() - step_start
        print(f"🔪 Created {len(chunks)} chunks")
        print(f"⏱️  Chunking took: {step_duration:.2f}s")
        timing_stats['phases']['chunking'] = step_duration
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['chunking_total'] = phase_duration
        print(f"⏱️  Total CHUNKING time: {phase_duration:.2f}s")
        
        # ========================================================================
        # PHASE 3: EMBEDDING (50-95%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"🎯 PHASE 3: EMBEDDING - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 6: Tạo embeddings
        if job:
            job.meta['progress'] = {'step': 'creating_embeddings', 'current': 50, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        embedding_service = get_embedding_service()
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)
        step_duration = time.time() - step_start
        print(f"🎯 Created {len(embeddings)} embeddings")
        print(f"⏱️  Embedding generation took: {step_duration:.2f}s")
        print(f"⏱️  Average per chunk: {step_duration/len(chunks):.3f}s")
        timing_stats['phases']['embedding_generation'] = step_duration
        timing_stats['phases']['embedding_per_chunk'] = step_duration / len(chunks)
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['embedding_total'] = phase_duration
        print(f"⏱️  Total EMBEDDING time: {phase_duration:.2f}s")
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['embedding_total'] = phase_duration
        
        # ========================================================================
        # PHASE 4: DATABASE SAVE (95-100%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"💾 PHASE 4: DATABASE SAVE - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 7: Xóa chunks cũ (nếu có)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        db.commit()
        
        # Bước 8: Lưu chunks vào database
        if job:
            job.meta['progress'] = {'step': 'saving_to_db', 'current': 60, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        total_chunks = len(chunks)
        for idx, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                document_id=document_id,
                content=chunk_data['content'],
                embedding=embedding,
                chunk_index=chunk_data['chunk_index'],
                section_id=chunk_data['metadata'].get('section_id'),
                sub_chunk_id=chunk_data['metadata'].get('sub_chunk_id'),
                h1=chunk_data['metadata'].get('h1'),
                h2=chunk_data['metadata'].get('h2'),
                h3=chunk_data['metadata'].get('h3'),
                metadata=chunk_data['metadata']
            )
            db.add(chunk)
            
            # Update progress
            if job and idx % 10 == 0:
                progress = 60 + int((idx / total_chunks) * 35)
                job.meta['progress'] = {
                    'step': 'saving_to_db',
                    'current': progress,
                    'total': 100,
                    'chunks_saved': idx,
                    'total_chunks': total_chunks
                }
                job.save_meta()
        
        db.commit()
        step_duration = time.time() - step_start
        print(f"💾 Saved {len(chunks)} chunks to database")
        print(f"⏱️  Database save took: {step_duration:.2f}s")
        print(f"⏱️  Average per chunk: {step_duration/len(chunks):.3f}s")
        timing_stats['phases']['database_save'] = step_duration
        timing_stats['phases']['db_save_per_chunk'] = step_duration / len(chunks)
        
        # Bước 9: Update document metadata với chunk info
        if job:
            job.meta['progress'] = {'step': 'finalizing', 'current': 95, 'total': 100}
            job.save_meta()
        
        current_meta = document.meta_data or {}  # type: ignore[assignment]
        if isinstance(current_meta, dict):
            current_meta['chunk_count'] = len(chunks)
            current_meta['chunk_size'] = chunk_size
            current_meta['chunk_overlap'] = chunk_overlap
            # Add timing stats to metadata
            current_meta['processing_time'] = timing_stats
            document.meta_data = current_meta  # type: ignore[assignment]
        
        document.status = "completed"  # type: ignore[assignment]
        db.commit()
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['database_total'] = phase_duration
        print(f"⏱️  Total DATABASE time: {phase_duration:.2f}s")
        
        # Calculate total time
        total_duration = time.time() - timing_stats['total_start']
        timing_stats['total_duration'] = total_duration
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETED - Summary")
        print(f"{'='*70}")
        print(f"📊 Document ID: {document_id}")
        print(f"📊 Total chunks created: {len(chunks)}")
        print(f"⏱️  TOTAL TIME: {total_duration:.2f}s")
        print(f"\n📈 Time Breakdown:")
        print(f"   • Ingest:    {timing_stats['phases'].get('ingest_total', 0):.2f}s ({timing_stats['phases'].get('ingest_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Chunking:  {timing_stats['phases'].get('chunking_total', 0):.2f}s ({timing_stats['phases'].get('chunking_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Embedding: {timing_stats['phases'].get('embedding_total', 0):.2f}s ({timing_stats['phases'].get('embedding_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Database:  {timing_stats['phases'].get('database_total', 0):.2f}s ({timing_stats['phases'].get('database_total', 0)/total_duration*100:.1f}%)")
        print(f"{'='*70}\n")
        
        # Update batch tracking nếu có batch_id
        if batch_id:
            batch_key = f"batch:{batch_id}"
            pipe = redis_conn.pipeline()
            
            # Increment completed counter
            pipe.hincrby(batch_key, "completed_files", 1)
            
            # Add timing stats
            pipe.hincrbyfloat(batch_key, "ingest_total", timing_stats['phases'].get('ingest_total', 0))
            pipe.hincrbyfloat(batch_key, "chunking_total", timing_stats['phases'].get('chunking_total', 0))
            pipe.hincrbyfloat(batch_key, "embedding_total", timing_stats['phases'].get('embedding_total', 0))
            pipe.hincrbyfloat(batch_key, "database_total", timing_stats['phases'].get('database_total', 0))
            pipe.hincrbyfloat(batch_key, "processing_total", total_duration)
            
            # Get batch info
            pipe.hgetall(batch_key)
            
            results = pipe.execute()
            batch_info = results[-1]  # Last result is hgetall
            
            # Check if this is the last file
            completed = int(batch_info.get('completed_files', 0))
            total = int(batch_info.get('total_files', 1))
            
            if completed == total:
                # This is the last worker - print batch summary
                batch_ingest = float(batch_info.get('ingest_total', 0))
                batch_chunking = float(batch_info.get('chunking_total', 0))
                batch_embedding = float(batch_info.get('embedding_total', 0))
                batch_database = float(batch_info.get('database_total', 0))
                batch_total = float(batch_info.get('processing_total', 0))
                
                print(f"\n{'='*70}")
                print(f"🎉 BATCH COMPLETED - All {total} file(s) processed")
                print(f"{'='*70}")
                print(f"📊 Batch ID: {batch_id}")
                print(f"⏱️  TOTAL BATCH TIME: {batch_total:.2f}s")
                print(f"\n📈 Total Time Breakdown (All Files):")
                print(f"   • Ingest:    {batch_ingest:.2f}s ({batch_ingest/batch_total*100:.1f}%)")
                print(f"   • Chunking:  {batch_chunking:.2f}s ({batch_chunking/batch_total*100:.1f}%)")
                print(f"   • Embedding: {batch_embedding:.2f}s ({batch_embedding/batch_total*100:.1f}%)")
                print(f"   • Database:  {batch_database:.2f}s ({batch_database/batch_total*100:.1f}%)")
                print(f"\n⚡ Average per file: {batch_total/total:.2f}s")
                print(f"{'='*70}\n")
                
                # Clean up batch tracking
                redis_conn.delete(batch_key)
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'message': f'Đã xử lý document và tạo {len(chunks)} chunks thành công',
            'document_id': document_id,
            'progress': {
                'step': 'completed',
                'current': 100,
                'total': 100,
                'chunks_saved': len(chunks)
            },
            'timing': timing_stats
        }
        
    except Exception as e:
        # Update document status to failed
        if 'document' in locals() and document:
            document.status = "failed"  # type: ignore[assignment]
            meta_data = document.meta_data or {}  # type: ignore[assignment]
            if isinstance(meta_data, dict):
                meta_data['error'] = str(e)
                document.meta_data = meta_data  # type: ignore[assignment]
            db.commit()
        
        print(f"❌ Error processing document {document_id}: {str(e)}")
        raise
        
    finally:
        db.close()
