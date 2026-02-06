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

    def fts_search(
        self, 
        query: str, 
        k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Thay thế BM25 bằng PostgreSQL Native Full Text Search
        """
        try:
            # Chuẩn hóa query: Duy Tân là ai -> Duy & Tân & là & ai
            # Dùng toán tử & (AND) hoặc | (OR) tùy nhu cầu
            clean_query = self._normalize_query_for_bm25(query)
            formatted_query = " | ".join(clean_query.split())
            
            if not formatted_query:
                return []

            # SQL sử dụng ts_rank để lấy điểm số tương tự BM25
            search_query = text("""
                SELECT 
                    c.id, c.content, c.document_id, c.h1, c.h2, c.h3,
                    c.chunk_index,
                    literal_column("metadata").label("data_meta"),
                    ts_rank(c.search_vector, to_tsquery('simple', :query_text)) as rank_score
                FROM chunks c
                WHERE c.search_vector @@ to_tsquery('simple', :query_text)
                """ + ("AND c.document_id = ANY(:doc_ids) " if document_ids else "") + """
                ORDER BY rank_score DESC
                LIMIT :limit_k
            """)
            
            params = {"query_text": formatted_query, "limit_k": k}
            if document_ids:
                params["doc_ids"] = document_ids
                
            results = self.db.execute(search_query, params).fetchall()
            
            return [
                {
                    'id': r.id,
                    'content': r.content,
                    'score': float(r.rank_score),
                    # ... các trường khác giữ nguyên ...
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ FTS Error: {e}")
            self.db.rollback()
            return []
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10, 
        document_ids: Optional[List[str]] = None, 
        alpha: float = 0.5, # Trọng số giữa 2 phương pháp
        **kwargs
    ):        
        """
        Hybrid Search kết hợp FTS (Native Azure/Postgres) và Semantic Search (pgvector)
        Sử dụng thuật toán RRF (Reciprocal Rank Fusion)
        """
        print(f"🔍 Thực hiện Hybrid Search cho: {query}")
        
        # 1. Chạy song song hoặc tuần tự 2 phương pháp
        fts_results = self.fts_search(query, k=k*2, document_ids=document_ids)
        semantic_results = self.semantic_search(query, k=k*2, document_ids=document_ids)
        
        # 2. Thuật toán RRF để gộp kết quả
        rrf_scores = {} # {doc_id: score}
        doc_map = {}    # {doc_id: doc_object}
        
        # RRF Constant (thường là 60)
        K_RRF = 60
        
        # Tính điểm cho Full Text Search
        for rank, doc in enumerate(fts_results):
            doc_id = doc['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (K_RRF + rank + 1))
            doc_map[doc_id] = doc
            
        # Tính điểm cho Semantic Search
        for rank, doc in enumerate(semantic_results):
            doc_id = doc['id']
            # RRF score cộng dồn
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (K_RRF + rank + 1))
            # Nếu doc này chưa có trong map (từ FTS), thì thêm vào
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 3. Sắp xếp lại dựa trên điểm RRF tổng hợp
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, score in sorted_ids[:k]:
            doc = doc_map[doc_id]
            doc['hybrid_score'] = score # Gán điểm mới
            final_results.append(doc)
            
        print(f"✅ Hybrid Search hoàn tất: tìm thấy {len(final_results)} kết quả")
        return final_results

def get_search_service(db: Session) -> SearchService:
    """Factory function for SearchService"""
    return SearchService(db)
