"""
Hybrid Search Service using PostgreSQL + pgvector
Implements RRF (Reciprocal Rank Fusion) for combining BM25 and semantic search
"""
import re
import math
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from app.database.models import ChildChunk, ParentChunk, Document, SummaryDocument
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
        # Get all child chunks
        chunks = self.db.query(ChildChunk).all()
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
    
    def bm25_search(
        self, 
        query: str, 
        k: int = 10,
        summary_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        BM25 search on child_chunks using ParadeDB pg_search extension
        Filter by summary_ids if provided for scoped search
        """
        print(f"🔍 BM25 query: {query}")
        
        # Build WHERE clause dynamically
        # ParadeDB syntax: content @@@ 'search text'
        where_conditions = ["content @@@ :query_text"]
        params = {"query_text": query, "limit_k": k}
        
        if summary_ids:
            # Convert UUID strings to UUID array for PostgreSQL
            where_conditions.append("c.summary_id::text = ANY(:summary_ids)")
            params["summary_ids"] = summary_ids
            print(f"📌 Scoped to {len(summary_ids)} summary documents")
        
        where_clause = " AND ".join(where_conditions)
        
        # Build ParadeDB search query on child_chunks
        search_query = text(f"""
            SELECT 
                c.id,
                c.content,
                c.document_id,
                c.parent_id,
                c.summary_id,
                c.h1,
                c.h2,
                c.h3,
                c.chunk_index,
                c.section_id,
                c.sub_chunk_id,
                c.metadata as meta_data,
                paradedb.score(c.id) as rank
            FROM child_chunks c
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :limit_k
        """)
        
        # Execute query
        try:
            results = self.db.execute(search_query, params).fetchall()
            
            print(f"📊 BM25 results: {len(results)} child chunks")
            
            return [
                {
                    'id': r.id,
                    'content': r.content,
                    'document_id': str(r.document_id),
                    'parent_id': r.parent_id,
                    'summary_id': str(r.summary_id) if r.summary_id else None,
                    'h1': r.h1,
                    'h2': r.h2,
                    'h3': r.h3,
                    'chunk_index': r.chunk_index,
                    'section_id': r.section_id,
                    'sub_chunk_id': r.sub_chunk_id,
                    'metadata': r.meta_data,
                    'score': float(r.rank) if r.rank else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ BM25 search error: {e}")
            # Rollback transaction on error
            self.db.rollback()
            return []
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 10,
        summary_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on child_chunks using pgvector cosine similarity
        Filter by summary_ids if provided for scoped search
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Base query with cosine distance
        base_query = self.db.query(
            ChildChunk.id,
            ChildChunk.content,
            ChildChunk.document_id,
            ChildChunk.parent_id,
            ChildChunk.summary_id,
            ChildChunk.h1,
            ChildChunk.h2,
            ChildChunk.h3,
            ChildChunk.chunk_index,
            ChildChunk.section_id,
            ChildChunk.sub_chunk_id,
            ChildChunk.meta_data,
            (1 - ChildChunk.embedding.cosine_distance(query_embedding)).label('similarity')
        )
        
        # Filter by summary_ids if provided
        if summary_ids:
            base_query = base_query.filter(ChildChunk.summary_id.in_(summary_ids))
            print(f"📌 Scoped to {len(summary_ids)} summary documents")
        
        # Order by similarity and limit
        results = base_query.order_by(text('similarity DESC')).limit(k).all()
        
        return [
            {
                'id': r.id,
                'content': r.content,
                'document_id': str(r.document_id),
                'parent_id': r.parent_id,
                'summary_id': str(r.summary_id) if r.summary_id else None,
                'h1': r.h1,
                'h2': r.h2,
                'h3': r.h3,
                'chunk_index': r.chunk_index,
                'section_id': r.section_id,
                'sub_chunk_id': r.sub_chunk_id,
                'metadata': r.meta_data,
                'score': float(r.similarity) if r.similarity else 0.0
            }
            for r in results
        ]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        bm25_k: Optional[int] = None,
        semantic_k: Optional[int] = None,
        rrf_k: int = 60,
        summary_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search on child_chunks using RRF (Reciprocal Rank Fusion)
        Combines BM25 and semantic search results
        Filter by summary_ids if provided for scoped search
        """
        bm25_k = bm25_k or max(k, 20)
        semantic_k = semantic_k or max(k, 20)
        
        print(f"\n🔍 HYBRID SEARCH (Child Chunks)")
        print(f"Query: {query}")
        print(f"Weights: BM25={bm25_weight}, Semantic={semantic_weight}")
        print(f"BM25_k={bm25_k}, Semantic_k={semantic_k}, RRF_k={rrf_k}")
        
        # Get BM25 results
        bm25_results = self.bm25_search(query, k=bm25_k, summary_ids=summary_ids)
        print(f"📄 BM25: {len(bm25_results)} results")
        
        # Get semantic results
        semantic_results = self.semantic_search(query, k=semantic_k, summary_ids=summary_ids)
        print(f"🎯 Semantic: {len(semantic_results)} results")
        
        # RRF fusion
        scores: Dict[int, Dict[str, Any]] = {}
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))
        
        # Add semantic scores
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))
        
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
    
    def bm25_search_summaries(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        BM25 search on summary documents using ParadeDB
        """
        print(f"🔍 BM25 Summary query: {query}")
        
        search_query = text("""
            SELECT 
                sd.id,
                sd.summary_content,
                sd.document_id,
                sd.metadata as meta_data,
                paradedb.score(sd.id) as rank
            FROM summary_documents sd
            WHERE summary_content @@@ :query_text
            ORDER BY rank DESC
            LIMIT :limit_k
        """)
        
        try:
            results = self.db.execute(
                search_query, 
                {"query_text": query, "limit_k": k}
            ).fetchall()
            
            print(f"📊 BM25 Summary: {len(results)} results")
            
            return [
                {
                    'id': str(r.id),
                    'summary_content': r.summary_content,
                    'document_id': str(r.document_id),
                    'metadata': r.meta_data,
                    'score': float(r.rank) if r.rank else 0.0
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ BM25 Summary search error: {e}")
            # Rollback transaction on error
            self.db.rollback()
            return []
    
    def semantic_search_summaries(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on summary documents using pgvector
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Query with cosine similarity
        results = self.db.query(
            SummaryDocument.id,
            SummaryDocument.summary_content,
            SummaryDocument.document_id,
            SummaryDocument.meta_data,
            (1 - SummaryDocument.embedding.cosine_distance(query_embedding)).label('similarity')
        ).order_by(text('similarity DESC')).limit(k).all()
        
        return [
            {
                'id': str(r.id),
                'summary_content': r.summary_content,
                'document_id': str(r.document_id),
                'metadata': r.meta_data,
                'score': float(r.similarity) if r.similarity else 0.0
            }
            for r in results
        ]
    
    def hybrid_search_summaries(
        self,
        query: str,
        k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search on summary documents using RRF
        """
        print(f"\n🔍 HYBRID SEARCH SUMMARIES")
        print(f"Query: {query}")
        print(f"Weights: BM25={bm25_weight}, Semantic={semantic_weight}")
        
        # Get BM25 results
        bm25_results = self.bm25_search_summaries(query, k=max(k, 10))
        print(f"📄 BM25 Summary: {len(bm25_results)} results")
        
        # Get semantic results
        semantic_results = self.semantic_search_summaries(query, k=max(k, 10))
        print(f"🎯 Semantic Summary: {len(semantic_results)} results")
        
        # RRF fusion
        scores: Dict[str, Dict[str, Any]] = {}
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))
        
        # Add semantic scores
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc['id']
            scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
            scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))
        
        # Sort by fused score
        fused = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        results = [x['doc'] for x in fused[:k]]
        
        # Add fused score and max score to results
        for i, result in enumerate(results):
            result['fused_score'] = fused[i]['score']
        
        max_score = results[0]['fused_score'] if results else 0.0
        
        print(f"✅ Fused Summaries: {len(results)} results")
        print(f"Max score: {round(max_score, 4)}")
        
        return results


def get_search_service(db: Session) -> SearchService:
    """Factory function for SearchService"""
    return SearchService(db)
