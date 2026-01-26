from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.database.connection import Base
from app.config import settings
import uuid


class Document(Base):
    """Bảng lưu trữ thông tin documents"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    file_path = Column(String(512), nullable=False, unique=True, index=True)
    file_name = Column(String(256), nullable=False)
    source_type = Column(String(50), default="local")  # local, cloud, wikipedia
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    file_size = Column(Integer, nullable=True, index=True)  # File size in bytes
    content_hash = Column(String(64), nullable=True, index=True)  # SHA256 hash of file content
    meta_data = Column(JSON, nullable=True, name="metadata")  # Use name to keep DB column as 'metadata'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Bảng lưu trữ chunks và vectors"""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    
    # Vector embedding
    embedding = Column(Vector(settings.DIMENSION_OF_MODEL))
    
    # Metadata
    section_id = Column(Integer, nullable=True)
    sub_chunk_id = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    meta_data = Column(JSON, nullable=True, name="metadata")  # Use name to keep DB column as 'metadata'
    
    # Headers from markdown
    h1 = Column(String(512), nullable=True)
    h2 = Column(String(512), nullable=True)
    h3 = Column(String(512), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
