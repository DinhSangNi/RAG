import json
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.services.search_service import SearchService


class RAGService:
    """Service for RAG chat with advanced retrieval strategies"""
    
    def __init__(
        self,
        db: Session,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1,
        top_k: int = 20,
        bm25_weight: float = 0.6,
        semantic_weight: float = 0.4,
        first_pass_k: int = 12,
        variant_count: int = 5,
        rrf_k: int = 60
    ):
        self.db = db
        self.search_service = SearchService(db)
        
        self.top_k = top_k
        self.first_pass_k = first_pass_k
        self.variant_count = variant_count
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Initialize LLM
        print(f"🤖 Initializing {model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
        # Main RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn là trợ lý AI. Trả lời câu hỏi dựa trên CONTEXT được cung cấp.

NGUYÊN TẮC:
1) CHỈ trả lời dựa trên thông tin trong CONTEXT
2) Nếu CONTEXT không có thông tin → trả lời đúng câu: "Tôi không tìm thấy thông tin này trong tài liệu."
3) Trả lời NGẮN GỌN, CHÍNH XÁC, bằng tiếng Việt
4) Nếu trong CONTEXT có nhiều tên gọi (bí danh / tên khai sinh / tên khác) của cùng một người, hãy coi chúng là 1 thực thể khi suy luận.
5) Không bịa đặt.

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""),
        ])
        
        # Alias extraction prompt
        self.alias_prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn sẽ trích xuất thông tin thực thể từ CONTEXT để hỗ trợ truy hồi.

Hãy trả về JSON hợp lệ theo schema:
{{
  "entity": "tên thực thể chính nếu xác định được, nếu không thì rỗng",
  "aliases": ["các tên gọi khác của cùng thực thể, nếu có"],
  "keywords": ["một vài từ khóa quan trọng liên quan đến câu hỏi (không quá chung chung)"]
}}

Ràng buộc:
- Chỉ dùng thông tin xuất hiện trong CONTEXT.
- Nếu không chắc entity là gì → entity = "".
- aliases: chỉ đưa alias thật sự cùng một người/tổ chức với entity trong CONTEXT.
- keywords: tối đa 8 từ/nhóm từ.
- Trả về JSON và CHỈ JSON, không thêm chữ nào khác.

CÂU HỎI: {question}

CONTEXT:
{context}
""")
        ])
        
        # Sufficiency check prompt
        self.sufficiency_prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn là chuyên gia đánh giá thông tin. Hãy kiểm tra xem CONTEXT có đủ thông tin để trả lời CÂU HỎI hay không.

Trả về JSON hợp lệ theo schema:
{{
  "sufficient": true/false,
  "reason": "giải thích ngắn gọn tại sao đủ hoặc không đủ"
}}

Nguyên tắc:
- sufficient = true: CONTEXT có đủ thông tin cụ thể để trả lời câu hỏi một cách chính xác
- sufficient = false: CONTEXT thiếu thông tin quan trọng, quá chung chung, hoặc không liên quan
- Chỉ trả về JSON, không thêm chữ nào khác

CONTEXT:
{context}

CÂU HỎI: {question}
""")
        ])
        
        # Build chains
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(x["docs"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.alias_chain = self.alias_prompt | self.llm | StrOutputParser()
        
        self.sufficiency_chain = self.sufficiency_prompt | self.llm | StrOutputParser()
        
        print("✅ RAG Service ready!")
    

    ## Tách chuỗi thành các token thân thiện với tiếng Việt
    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents for context"""
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = doc.get('h2') or doc.get('h1') or ''
            content = doc.get('content', '')
            formatted.append(f"--- Đoạn {i} | {header} ---\n{content}")
        
        return "\n\n".join(formatted)
    
    def _format_summary_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format summary documents for context"""
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.get('summary_content', '')
            formatted.append(f"--- Tóm tắt {i} ---\n{content}")
        
        return "\n\n".join(formatted)
    
    def _check_info_sufficiency(
        self,
        question: str,
        summary_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if summary documents have enough information to answer the question
        Returns True if sufficient, False otherwise
        """
        if not summary_docs:
            return False
        
        context = self._format_summary_docs(summary_docs)
        
        try:
            raw = self.sufficiency_chain.invoke({
                "question": question,
                "context": context
            })
            
            # Parse JSON response
            data = json.loads(raw)
            sufficient = data.get("sufficient", False)
            reason = data.get("reason", "")
            
            print(f"🧠 Sufficiency check: {sufficient} - {reason}")
            return sufficient
            
        except Exception as e:
            print(f"⚠️ Sufficiency check failed: {e}")
            # If check fails, assume not sufficient to be safe
            return False
    
    ## Trích xuất thực thể, bí danh, từ khóa từ ngữ cảnh bằng LLM
    def _extract_entity_info(
        self, 
        question: str, 
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        context = self._format_docs(docs[:min(len(docs), 8)])
        
        try:
            raw = self.alias_chain.invoke({
                "question": question, 
                "context": context
            })
            
            # Try to parse JSON
            data = json.loads(raw)
            entity = (data.get("entity") or "").strip()
            aliases = data.get("aliases") or []
            keywords = data.get("keywords") or []
            
            # Normalize
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
            
            # Remove duplicates
            aliases = list(dict.fromkeys(aliases))
            keywords = list(dict.fromkeys(keywords))
            
            return {
                "entity": entity,
                "aliases": aliases,
                "keywords": keywords
            }
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
            return {"entity": "", "aliases": [], "keywords": []}
    
    def _make_variants(
        self, 
        question: str, 
        info: Dict[str, Any]
    ) -> List[str]:
        """Generate query variants from entity/aliases/keywords"""
        entity = info.get("entity", "").strip()
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        
        # Filter stopwords
        stop = self.search_service.get_stopwords()
        q_tokens = self._tokenize_vi(question)
        q_core = [t for t in q_tokens if t not in stop]
        core_text = " ".join(q_core).strip()
        
        variants = [question]
        
        # Entity-based variants
        if entity:
            if core_text:
                variants.append(f"{entity} {core_text}")
            variants.append(f"{entity} {question}")
        
        # Alias-based variants
        for alias in aliases[:self.variant_count]:
            if core_text:
                variants.append(f"{alias} {core_text}")
            else:
                variants.append(f"{alias} {question}")
        
        # Keyword-based variants
        if entity and keywords:
            variants.append(f"{entity} " + " ".join(keywords[:8]))
        elif keywords:
            variants.append(" ".join(keywords[:8]))
        
        # Dedup
        deduped = []
        seen = set()
        for v in variants:
            v2 = v.strip()
            if not v2 or v2.lower() in seen:
                continue
            seen.add(v2.lower())
            deduped.append(v2)
        
        return deduped[:max(3, self.variant_count)]
    
    def _rrf_fuse(
        self, 
        list_of_results: List[List[Dict[str, Any]]], 
        rrf_k: int = 60, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for multiple result lists"""
        scores: Dict[int, Dict[str, Any]] = {}
        
        for results in list_of_results:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc['id']
                scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
                scores[doc_id]['score'] += 1.0 / (rrf_k + rank)
        
        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        fused = [x['doc'] for x in ranked[:top_k]]
        
        # Add fused scores
        for i, doc in enumerate(fused):
            doc['fused_score'] = ranked[i]['score']
        
        return fused
    
    def retrieve_hierarchical(
        self,
        question: str,
        summary_k: int = 5,
        chunk_k: int = 20,
        min_summary_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval workflow with 6 steps:
        
        Step 1: Query on summary documents (hybrid search) => get relevant documents (determine scope)
                If no summary docs found or max score < min_summary_score, search all child chunks
        Step 2: Format summary docs as context => Send to LLM to check if enough info to answer
        Step 3: If enough: return summary docs for answer generation
                If not enough: query expansion => extract entities, aliases, keywords
        Step 4: Use each query variant to search child chunks belonging to summary docs from step 1
                If step 1 found nothing, search all child chunks
        Step 5: Aggregate results, calculate RRF scores for chunks from step 4
        Step 6: Return chunks for answer generation
        
        Returns:
            {
                'docs': List[Dict],  # Documents to use for answer
                'source': str,  # 'summary' or 'chunks'
                'metadata': Dict
            }
        """
        print(f"\n{'='*70}")
        print(f"🔍 HIERARCHICAL RETRIEVAL")
        print(f"{'='*70}")
        print(f"Question: {question}")
        
        # STEP 1: Query on summary documents
        print(f"\n📋 STEP 1: Search summary documents")
        summary_docs = self.search_service.hybrid_search_summaries(
            query=question,
            k=summary_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k
        )
        
        max_score = summary_docs[0]['fused_score'] if summary_docs else 0.0
        print(f"Found {len(summary_docs)} summary docs, max score: {round(max_score, 4)}")
        
        # Check if we should fall back to full child chunk search
        if not summary_docs or max_score < min_summary_score:
            print(f"⚠️ No good summary docs found (max score < {min_summary_score})")
            print(f"Falling back to full child chunk search")
            
            # Search all child chunks directly
            all_chunks = self.search_service.hybrid_search(
                query=question,
                k=chunk_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k
            )
            
            return {
                'docs': all_chunks,
                'source': 'chunks_fallback',
                'metadata': {
                    'summary_docs_found': len(summary_docs),
                    'max_summary_score': max_score,
                    'chunks_returned': len(all_chunks)
                }
            }
        
        # Extract document IDs from summary docs for scoping
        relevant_doc_ids = list(set([doc['document_id'] for doc in summary_docs]))
        print(f"Scope: {len(relevant_doc_ids)} parent documents")
        
        # STEP 2: Check if summary docs have enough information
        print(f"\n🧠 STEP 2: Check information sufficiency")
        is_sufficient = self._check_info_sufficiency(question, summary_docs)
        
        # STEP 3: Decision point
        if is_sufficient:
            print(f"\n✅ STEP 3: Summary docs are sufficient - using them for answer")
            return {
                'docs': summary_docs,
                'source': 'summary',
                'metadata': {
                    'summary_docs_count': len(summary_docs),
                    'max_summary_score': max_score,
                    'sufficient': True
                }
            }
        else:
            print(f"\n🔄 STEP 3: Not sufficient - proceeding with query expansion")
            
            # Extract entity info for query expansion
            # First get some chunks from the relevant docs for context
            initial_chunks = self.search_service.hybrid_search(
                query=question,
                k=self.first_pass_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                document_ids=relevant_doc_ids
            )
            
            info = self._extract_entity_info(question, initial_chunks)
            entity = info.get("entity", "")
            aliases = info.get("aliases") or []
            keywords = info.get("keywords") or []
            
            print(f"🧠 Entity: {entity or '(none)'} | Aliases: {len(aliases)} | Keywords: {len(keywords)}")
            
            # Generate query variants
            variants = self._make_variants(question, info)
            print(f"🧩 Variants: {len(variants)}")
            for i, v in enumerate(variants, 1):
                print(f"   Q{i}: {v}")
            
            # STEP 4: Search child chunks with each variant (scoped to relevant docs)
            print(f"\n🔎 STEP 4: Search child chunks with variants (scoped)")
            all_results = []
            
            for i, variant in enumerate(variants, 1):
                print(f"Variant {i}/{len(variants)}: {variant[:60]}...")
                results = self.search_service.hybrid_search(
                    query=variant,
                    k=max(chunk_k, 20),
                    bm25_weight=self.bm25_weight,
                    semantic_weight=self.semantic_weight,
                    rrf_k=self.rrf_k,
                    document_ids=relevant_doc_ids
                )
                all_results.append(results)
                print(f"  → {len(results)} chunks")
            
            # STEP 5: RRF fusion of all variant results
            print(f"\n📊 STEP 5: RRF fusion of {len(all_results)} result sets")
            fused_chunks = self._rrf_fuse(all_results, rrf_k=self.rrf_k, top_k=chunk_k)
            print(f"✅ Fused: {len(fused_chunks)} chunks")
            
            # STEP 6: Return fused chunks for answer generation
            return {
                'docs': fused_chunks,
                'source': 'chunks_expanded',
                'metadata': {
                    'summary_docs_count': len(summary_docs),
                    'max_summary_score': max_score,
                    'sufficient': False,
                    'variants_count': len(variants),
                    'chunks_returned': len(fused_chunks),
                    'scoped_to_docs': len(relevant_doc_ids)
                }
            }
    
    def retrieve(
        self, 
        question: str,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-pass retrieval with query expansion:
        1. First pass: retrieve with original question
        2. Extract entity/aliases/keywords from context
        3. Generate query variants
        4. Retrieve for each variant
        5. RRF fusion of all results
        """
        print(f"\n🔍 Retrieve: '{question}'")
        
        # Pass 1: Initial retrieval
        first_pass = self.search_service.hybrid_search(
            query=question,
            k=self.first_pass_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            document_ids=document_ids
        )
        print(f"📦 Pass 1: {len(first_pass)} chunks")
        
        # Extract entity info from first pass
        info = self._extract_entity_info(question, first_pass)
        entity = info.get("entity", "")
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        
        print(f"🧠 Entity: {entity or '(none)'} | Aliases: {len(aliases)} | Keywords: {len(keywords)}")
        
        # Generate variants
        variants = self._make_variants(question, info)
        print(f"🧩 Variants: {len(variants)}")
        for i, v in enumerate(variants, 1):
            print(f"   Q{i}: {v}")
        
        # Pass 2: Retrieve for each variant
        all_results = [first_pass]
        
        for variant in variants:
            results = self.search_service.hybrid_search(
                query=variant,
                k=max(self.top_k, 20),
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                document_ids=document_ids
            )
            all_results.append(results)
        
        # Fuse all results
        fused = self._rrf_fuse(all_results, rrf_k=self.rrf_k, top_k=self.top_k)
        print(f"✅ Fused: {len(fused)} chunks")
        
        return fused
    
    def chat(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        verbose: bool = False,
        use_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """
        RAG chat: retrieve + generate answer
        
        Args:
            question: User question
            document_ids: Optional filter by document IDs (only works with old retrieve method)
            verbose: Print detailed context
            use_hierarchical: Use new hierarchical retrieval workflow (default: True)
        
        Returns:
            {
                'answer': str,
                'chunks': List[Dict],
                'metadata': Dict
            }
        """
        # Choose retrieval strategy
        if use_hierarchical:
            # New hierarchical retrieval workflow
            result = self.retrieve_hierarchical(question)
            docs = result['docs']
            source = result['source']
            metadata = result['metadata']
            metadata['retrieval_method'] = 'hierarchical'
        else:
            # Old two-pass retrieval with query expansion
            docs = self.retrieve(question, document_ids=document_ids)
            source = 'legacy'
            metadata = {'retrieval_method': 'legacy'}
        
        if not docs:
            return {
                'answer': "Tôi không tìm thấy thông tin này trong tài liệu.",
                'chunks': [],
                'metadata': {**metadata, 'chunks_used': 0}
            }
        
        if verbose:
            print(f"\n{'='*70}\nCONTEXT:\n{'='*70}")
            for i, doc in enumerate(docs[:5], 1):
                print(f"\n📄 Doc {i}:")
                if source == 'summary':
                    print(f"   Type: Summary Document")
                    print(f"   Preview: {doc.get('summary_content', '')[:200]}...")
                else:
                    print(f"   Type: Chunk")
                    print(f"   Headers: {doc.get('h1', '')} / {doc.get('h2', '')}")
                    print(f"   Preview: {doc.get('content', '')[:200]}...")
        
        # Generate answer based on source type
        print("\n💬 Generating answer...")
        
        if source == 'summary':
            # Use summary documents - format differently
            formatted_docs = []
            for doc in docs:
                formatted_docs.append({
                    'content': doc.get('summary_content', ''),
                    'h1': 'Summary',
                    'h2': ''
                })
            answer = self.rag_chain.invoke({"docs": formatted_docs, "question": question})
        else:
            # Use regular chunks
            answer = self.rag_chain.invoke({"docs": docs, "question": question})
        
        answer = (answer or "").strip()
        
        # Normalize fallback
        if not answer or ("không tìm thấy" in answer.lower() and "tài liệu" in answer.lower()):
            answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        
        return {
            'answer': answer,
            'chunks': docs[:10],  # Return top 10 for reference
            'metadata': {
                **metadata,
                'chunks_used': len(docs),
                'source': source,
                'model': getattr(self.llm, 'model', 'unknown')
            }
        }


def get_rag_service(db: Session) -> RAGService:
    """Factory function for RAGService"""
    return RAGService(db)
