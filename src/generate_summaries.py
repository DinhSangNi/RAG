"""
Helper script to generate summary documents from existing documents
This script uses LLM to create summaries for hierarchical RAG retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.database.connection import get_db
from app.database.models import Document, ChildChunk, ParentChunk, SummaryDocument
from app.services.embedding_service import get_embedding_service
from app.config import settings


def create_summary_for_document(
    db: Session,
    document_id: str,
    llm: ChatGoogleGenerativeAI,
    embedding_service
) -> bool:
    """
    Create a summary document for a given parent document
    
    Args:
        db: Database session
        document_id: UUID of the document
        llm: LLM instance for summarization
        embedding_service: Embedding service instance
    
    Returns:
        True if successful, False otherwise
    """
    # Check if summary already exists
    existing = db.query(SummaryDocument).filter(
        SummaryDocument.document_id == document_id
    ).first()
    
    if existing:
        print(f"⚠️ Summary already exists for document {document_id}")
        return False
    
    # Get document info
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        print(f"❌ Document {document_id} not found")
        return False
    
    # Get all chunks for this document
    chunks = db.query(ChildChunk).filter(ChildChunk.document_id == document_id).order_by(ChildChunk.chunk_index).all()
    
    if not chunks:
        print(f"⚠️ No chunks found for document {document_id}")
        return False
    
    print(f"\n📄 Processing: {doc.file_name}")
    print(f"   Chunks: {len(chunks)}")
    
    # Combine chunk content (limit to avoid token overflow)
    max_chunks = 50  # Limit number of chunks to summarize
    chunk_contents = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        header = f"{chunk.h1 or ''} > {chunk.h2 or ''} > {chunk.h3 or ''}".strip(' > ')
        content = chunk.content or ''
        chunk_contents.append(f"[Chunk {i+1}] {header}\n{content}")
    
    full_content = "\n\n".join(chunk_contents)
    
    # Create summary prompt
    summary_prompt = ChatPromptTemplate.from_messages([
        ("human", """Bạn là chuyên gia tóm tắt tài liệu. Hãy tạo một bản tóm tắt chi tiết và toàn diện cho tài liệu sau.

Yêu cầu:
1. Tóm tắt phải bao gồm tất cả thông tin quan trọng
2. Giữ nguyên các tên riêng, số liệu, ngày tháng
3. Bao gồm tất cả các chủ đề chính được đề cập
4. Viết bằng tiếng Việt, rõ ràng, mạch lạc
5. Độ dài: khoảng 500-800 từ (tùy độ dài tài liệu)

TÀI LIỆU:
{content}

TÓM TẮT:""")
    ])
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    try:
        print("   🤖 Generating summary...")
        summary_text = summary_chain.invoke({"content": full_content})
        summary_text = summary_text.strip()
        
        if not summary_text or len(summary_text) < 50:
            print(f"   ❌ Generated summary too short")
            return False
        
        print(f"   ✅ Summary generated: {len(summary_text)} chars")
        
        # Generate embedding
        print("   🧮 Generating embedding...")
        embedding = embedding_service.embed_text(summary_text)
        
        # Create summary document
        summary_doc = SummaryDocument(
            document_id=document_id,
            summary_content=summary_text,
            embedding=embedding,
            meta_data={
                'chunks_count': len(chunks),
                'chunks_summarized': min(len(chunks), max_chunks),
                'source_file': doc.file_name
            }
        )
        
        db.add(summary_doc)
        db.commit()
        
        print(f"   💾 Summary saved to database")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        db.rollback()
        return False


def main():
    """Main function to generate summaries for all documents"""
    print("="*70)
    print("GENERATE SUMMARY DOCUMENTS FOR HIERARCHICAL RAG")
    print("="*70)
    
    # Initialize services
    db = next(get_db())
    embedding_service = get_embedding_service()
    
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        api_key=settings.GEMINI_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Get all documents
    documents = db.query(Document).filter(Document.status == "completed").all()
    
    print(f"\n📚 Found {len(documents)} completed documents\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing {doc.file_name}...")
        
        # Check if summary exists
        existing = db.query(SummaryDocument).filter(
            SummaryDocument.document_id == doc.id
        ).first()
        
        if existing:
            print(f"   ⏭️ Skipped (already exists)")
            skip_count += 1
            continue
        
        # Create summary
        success = create_summary_for_document(
            db=db,
            document_id=str(doc.id),
            llm=llm,
            embedding_service=embedding_service
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Success: {success_count}")
    print(f"⏭️ Skipped: {skip_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📊 Total: {len(documents)}")
    print("="*70)


if __name__ == "__main__":
    main()
