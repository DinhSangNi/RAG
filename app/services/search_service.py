"""
Hybrid Search Service using PostgreSQL + pgvector
Implements RRF (Reciprocal Rank Fusion) for combining BM25 and semantic search
"""
import re
import math
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func, literal_column
from app.database.models import Chunk, Document
from app.services.embedding_service import get_embedding_service


class SearchService:
    """Service for hybrid search using BM25 + Semantic (pgvector)"""
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = get_embedding_service()
        self._auto_stopwords: Optional[set] = None
    
    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]
    
    def _build_auto_stopwords(
        self, 
        df_threshold: float = 0.35, 
        max_size: int = 250
    ) -> set:
        """
        Build stopwords from corpus based on document frequency
        """
        # Get all chunks
        chunks = self.db.query(Chunk).all()
        n_docs = max(1, len(chunks))
        
        df: Dict[str, int] = {}
        
        for chunk in chunks:
            content = str(chunk.content) if chunk.content else "" 
            tokens = set(self._tokenize_vi(content))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        
        threshold = int(math.ceil(df_threshold * n_docs))
        high_df = [(t, c) for t, c in df.items() if c >= threshold and len(t) >= 2]
        high_df.sort(key=lambda x: x[1], reverse=True)
        
        stop = set([t for t, _ in high_df[:max_size]])
        print(f"🧹 Auto-stopwords: {len(stop)} tokens (DF≥{threshold}/{n_docs})")
        return stop
    
    def get_stopwords(self) -> set:
        """Get or build stopwords"""
        if self._auto_stopwords is None:
            self._auto_stopwords = self._build_auto_stopwords()
        return self._auto_stopwords
    
    @staticmethod
    def _normalize_query_for_bm25(query: str) -> str:
        """
        Normalize query for ParadeDB BM25 search
        Removes special characters that break paradedb.parse()
        """
        if not query:
            return ""
        
        # Remove special characters that break ParadeDB parse
        # Keep only alphanumeric, Vietnamese characters, and spaces
        # ParadeDB parse() expects clean text without punctuation
        normalized = re.sub(r'[^\w\sÀ-ỹ]', ' ', query, flags=re.UNICODE)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ nghĩa thuần túy bằng pgvector
        """
        try:
            # 1. Tạo embedding từ query
            query_embedding = self.embedding_service.embed_text(query)
            
            # 2. Query dùng literal_column để tránh xung đột với thuộc tính .metadata của SQLAlchemy
            base_query = self.db.query(
                Chunk.id,
                Chunk.content,
                Chunk.document_id,
                Chunk.h1,
                Chunk.h2,
                Chunk.h3,
                Chunk.chunk_index,
                Chunk.section_id,
                Chunk.sub_chunk_id,
                literal_column("metadata").label("data_meta"), # Ép lấy cột metadata từ DB
                (1 - Chunk.embedding.cosine_distance(query_embedding)).label('similarity')
            )
            
            # 3. Metadata Filtering (Lọc theo danh sách document_ids nếu có)
            if document_ids:
                base_query = base_query.filter(Chunk.document_id.in_(document_ids))
            
            # 4. Sắp xếp theo similarity
            results = base_query.order_by(text('similarity DESC')).limit(k).all()
            
            # 5. Format kết quả trả về
            return [
                {
                    'id': r.id,
                    'content': r.content or "",
                    'document_id': str(r.document_id),
                    'h1': r.h1 or "",
                    'h2': r.h2 or "",
                    'h3': r.h3 or "",
                    'chunk_index': r.chunk_index,
                    'metadata': r.data_meta if r.data_meta is not None else {},
                    'score': float(r.similarity) if r.similarity is not None else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ Semantic Search Error: {e}")
            self.db.rollback() # Giải phóng transaction ngay khi lỗi
            return []
    
    def hybrid_search(self, query: str, k: int = 10, document_ids: Optional[List[str]] = None, **kwargs):
        """
        Đã hạ cấp xuống chỉ còn Semantic Search để chạy ổn định trên Azure
        """
        print(f"🎯 Thực hiện Semantic Search cho: {query}")
        return self.semantic_search(query, k=k, document_ids=document_ids)


def get_search_service(db: Session) -> SearchService:
    """Factory function for SearchService"""
    return SearchService(db)
