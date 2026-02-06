"""
Hybrid Search Service using PostgreSQL + pgvector
Implements RRF (Reciprocal Rank Fusion) for combining BM25 and semantic search
"""
import re
import math
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
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
            content = str(chunk.content) if chunk.content else ""  # type: ignore[arg-type]
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
    
    def bm25_search(
        self, 
        query: str, 
        k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Native Postgres Full Text Search - Đã sửa tên cột metadata
        """
        print(f"🔍 FTS query: {query}")
        clean_query = self._normalize_query_for_bm25(query)
        # Sử dụng '|' (OR) thay vì '&' (AND) để tăng khả năng tìm thấy kết quả nếu query dài
        formatted_query = " | ".join(clean_query.split())
        
        if not formatted_query:
            return []

        search_query = text("""
            SELECT 
                c.id, c.content, c.document_id, c.h1, c.h2, c.h3,
                c.chunk_index, c.section_id, c.sub_chunk_id,
                c.metadata, -- ĐÃ SỬA: meta_data -> metadata
                ts_rank(to_tsvector('simple', COALESCE(c.content, '')), to_tsquery('simple', :query_text)) as rank
            FROM chunks c
            WHERE to_tsvector('simple', COALESCE(c.content, '')) @@ to_tsquery('simple', :query_text)
            """ + ("AND c.document_id = ANY(:doc_ids) " if document_ids else "") + """
            ORDER BY rank DESC
            LIMIT :limit_k
        """)
        
        try:
            params = {"query_text": formatted_query, "limit_k": k}
            if document_ids:
                params["doc_ids"] = document_ids
                
            results = self.db.execute(search_query, params).fetchall()
            
            return [
                {
                    'id': r.id,
                    'content': r.content or "",
                    'document_id': str(r.document_id),
                    'h1': r.h1 or "",
                    'h2': r.h2 or "",
                    'h3': r.h3 or "",
                    'chunk_index': r.chunk_index,
                    'section_id': r.section_id,
                    'sub_chunk_id': r.sub_chunk_id,
                    'metadata': r.metadata if r.metadata is not None else {},
                    'score': float(r.rank) if r.rank else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ FTS search error: {e}")
            self.db.rollback() 
            return []
    
    def semantic_search(self, query: str, k: int = 10, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_service.embed_text(query)
            
            # Sử dụng literal_column để ép SQLAlchemy lấy đúng tên cột trong DB
            from sqlalchemy import literal_column
            
            results = self.db.query(
                Chunk.id,
                Chunk.content,
                Chunk.document_id,
                Chunk.h1,
                Chunk.h2,
                Chunk.h3,
                Chunk.chunk_index,
                Chunk.section_id,
                Chunk.sub_chunk_id,
                literal_column("metadata").label("raw_meta"), # Lấy trực tiếp cột tên 'metadata'
                (1 - Chunk.embedding.cosine_distance(query_embedding)).label('similarity')
            )

            if document_ids:
                results = results.filter(Chunk.document_id.in_(document_ids))
            
            results = results.order_by(text('similarity DESC')).limit(k).all()
            
            return [
                {
                    'id': r.id,
                    'content': r.content or "",
                    'document_id': str(r.document_id),
                    'h1': r.h1 or "",
                    'h2': r.h2 or "",
                    'h3': r.h3 or "",
                    'chunk_index': r.chunk_index,
                    'metadata': r.raw_meta or {},
                    'score': float(r.similarity) if r.similarity is not None else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ Semantic Search Error: {e}")
            self.db.rollback()
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        document_ids: Optional[List[str]] = None, # Metadata Filtering ở đây
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion)
        Combines BM25 and semantic search results
        """
        bm25_res = self.bm25_search(query, k=k, document_ids=document_ids)
        semantic_res = self.semantic_search(query, k=k, document_ids=document_ids)
        
        print(f"\n🔍 HYBRID SEARCH")
        print(f"Query: {query}")
        # print(f"Weights: BM25={bm25_weight}, Semantic={semantic_weight}")
        # print(f"BM25_k={bm25_k}, Semantic_k={semantic_k}, RRF_k={rrf_k}")
        
        # Get BM25 results
        # bm25_results = self.bm25_search(query, k=bm25_k, document_ids=document_ids)
        # print(f"📄 BM25: {len(bm25_results)} results")
        
        # Get semantic results
        # semantic_results = self.semantic_search(query, k=semantic_k, document_ids=document_ids)
        # print(f"🎯 Semantic: {len(semantic_results)} results")
        
        # RRF fusion
        scores: Dict[int, Dict[str, Any]] = {}
        
        # Add BM25 scores
        # for rank, doc in enumerate(bm25_results, start=1):
        #     doc_id = doc['id']
        #     scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
        #     scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))
        
        # Add semantic scores
        # for rank, doc in enumerate(semantic_results, start=1):
        #     doc_id = doc['id']
        #     scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
        #     scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))
        
        # Sort by fused score
        fused = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        results = [x['doc'] for x in fused[:k]]
        
        # Add fused score to results
        for i, result in enumerate(results):
            result['fused_score'] = fused[i]['score']
        
        print(f"✅ Fused: {len(results)} results")
        if results:
            print(f"Top result: {results[0].get('h1', '')} / {results[0].get('h2', '')}")
            print(f"Top scores: {[round(r.get('fused_score', 0), 4) for r in results[:3]]}")
        
        return results


def get_search_service(db: Session) -> SearchService:
    """Factory function for SearchService"""
    return SearchService(db)
