"""
RAG Service for advanced retrieval-augmented generation
Implements hierarchical retrieval with query expansion and entity extraction
"""

import json
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.services.search_service import SearchService
from app.database.models import Document


class RAGService:
    """
    Service for RAG chat with advanced retrieval strategies
    Implements hierarchical retrieval, query expansion, and entity extraction
    """

    def __init__(
        self,
        db: Session,
        model_name: str | None = None,
        temperature: float = 0.1,
        top_k: int = 20,
        bm25_weight: float = 0.6,
        semantic_weight: float = 0.4,
        first_pass_k: int = 12,
        variant_count: int = 5,
        rrf_k: int = 60
    ):
        """
        Initialize RAG Service

        Args:
            db: Database session
            model_name: LLM model name (defaults to config)
            temperature: LLM temperature
            top_k: Number of results to return
            bm25_weight: Weight for BM25 search
            semantic_weight: Weight for semantic search
            first_pass_k: Results for first retrieval pass
            variant_count: Number of query variants to generate
            rrf_k: RRF parameter
        """
        self.db = db
        self.search_service = SearchService(db)

        # Use model from config if not provided
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        self.temperature = temperature

        self.top_k = top_k
        self.first_pass_k = first_pass_k
        self.variant_count = variant_count
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Initialize LLM
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_API_KEY,
            temperature=self.temperature,
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
#         self.alias_prompt = ChatPromptTemplate.from_messages([
#             ("human", """Bạn sẽ trích xuất thông tin thực thể từ CONTEXT để hỗ trợ truy hồi.

# Hãy trả về JSON hợp lệ theo schema:
# {{
#   "entity": "tên thực thể chính nếu xác định được, nếu không thì rỗng",
#   "aliases": ["các tên gọi khác của cùng thực thể, nếu có"],
#   "keywords": ["một vài từ khóa quan trọng liên quan đến câu hỏi (không quá chung chung)"]
# }}

# Ràng buộc:
# - Chỉ dùng thông tin xuất hiện trong CONTEXT.
# - Nếu không chắc entity là gì → entity = "".
# - aliases: chỉ đưa alias thật sự cùng một người/tổ chức với entity trong CONTEXT.
# - keywords: tối đa 8 từ/nhóm từ.
# - Trả về JSON và CHỈ JSON, không thêm chữ nào khác.

# CÂU HỎI: {question}

# CONTEXT:
# {context}
# """)
#         ])
        self.alias_prompt = ChatPromptTemplate.from_messages([
    ("human", """Bạn là một chuyên gia phân tích truy vấn (Query Parser). 
Nhiệm vụ của bạn là trích xuất các thành phần quan trọng từ CÂU HỎI để tạo query variants cho tìm kiếm.

RÀNG BUỘC QUAN TRỌNG:
1. 'entity' CHỈ được lấy từ tên riêng/danh từ chỉ người/tổ chức được NHẮC TRỰC TIẾP trong CÂU HỎI
2. KHÔNG được suy diễn entity từ CONTEXT (nếu context có "Trần Thuận Tông là vua thứ 12" mà câu hỏi chỉ hỏi "vua thứ 12" thì entity KHÔNG PHẢI là "Trần Thuận Tông")
3. Nếu câu hỏi dạng "Ai là...", "Khi nào...", "Ở đâu..." thì entity thường RỖNG hoặc là cụm mô tả từ câu hỏi
4. 'keywords' là các từ khóa đặc trưng, định danh cao từ CÂU HỎI (ví dụ: "thứ 12", "triều Nguyễn", "năm 1945")
5. 'aliases' CHỈ dùng nếu entity xuất hiện trong CÂU HỎI và có tên thay thế trong CONTEXT

SCHEMA TRẢ VỀ (JSON):
{{
  "entity": "tên riêng xuất hiện trong CÂU HỎI hoặc rỗng",
  "aliases": ["tên thay thế nếu có trong context"],
  "keywords": ["từ khóa quan trọng từ câu hỏi"]
}}

VÍ DỤ:
Q: "Ai là vị hoàng đế thứ 12 của triều Nguyễn?"
→ {{"entity": "", "aliases": [], "keywords": ["hoàng đế thứ 12", "triều Nguyễn", "vua thứ 12"]}}

Q: "Khải Định sinh năm nào?"
→ {{"entity": "Khải Định", "aliases": ["vua Khải Định", "Nguyễn Phúc Tuấn"], "keywords": ["sinh năm"]}}

Q: "Cha của Bảo Đại là ai?"
→ {{"entity": "Bảo Đại", "aliases": ["vua Bảo Đại"], "keywords": ["cha", "cha của Bảo Đại"]}}

CÂU HỎI: {question}

CONTEXT (chỉ để tìm aliases, KHÔNG dùng để suy diễn entity):
{context}

JSON:""")
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
            
            # Clean potential markdown code blocks
            clean_raw = raw.strip()
            if clean_raw.startswith("```"):
                lines = clean_raw.split("\n")
                clean_raw = "\n".join(lines[1:-1]) if len(lines) > 2 else clean_raw
                clean_raw = clean_raw.replace("```json", "").replace("```", "").strip()
            
            # Try to parse JSON response
            try:
                data = json.loads(clean_raw)
                sufficient = data.get("sufficient", False)
                reason = data.get("reason", "")
                print(f"🧠 Sufficiency check: {sufficient} (reason: {reason[:50]})")
                return sufficient
            except json.JSONDecodeError:
                # Fallback to simple YES/NO parsing
                answer = raw.strip().upper()
                sufficient = "YES" in answer or "CÓ" in answer or '"SUFFICIENT": TRUE' in answer
                print(f"🧠 Sufficiency check: {sufficient} (fallback parse, response: {raw[:50]})")
                return sufficient
            
        except Exception as e:
            print(f"⚠️ Sufficiency check failed: {e}")
            # If check fails, assume not sufficient to drill down
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
            
            # Debug output
            if not raw or not raw.strip():
                print(f"⚠️ Entity extraction: LLM returned empty response")
                return {"entity": "", "aliases": [], "keywords": []}
            
            # Clean potential markdown code blocks
            raw = raw.strip()
            if raw.startswith("```"):
                # Remove markdown code fence
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
                raw = raw.replace("```json", "").replace("```", "").strip()
            
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
        except json.JSONDecodeError as e:
            print(f"⚠️ Entity extraction JSON parse error: {e}")
            print(f"📝 Raw output: {raw[:200] if raw else '(empty)'}")
            return {"entity": "", "aliases": [], "keywords": []}
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
    
    def _get_parent_chunks_context(
        self, 
        child_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Lấy parent chunks từ danh sách child chunks
        Deduplicate parent_ids trước khi query
        
        Args:
            child_chunks: List of child chunk dicts with 'parent_id' field
        
        Returns:
            List of parent chunk dicts for context
        """
        from app.database.models import ParentChunk
        
        # Extract parent_ids from child chunks (filter out None)
        parent_ids = [chunk['parent_id'] for chunk in child_chunks if chunk.get('parent_id')]
        
        if not parent_ids:
            print(f"⚠️ No parent chunks found - using child chunks directly")
            return child_chunks
        
        # Deduplicate parent_ids while preserving order
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        print(f"📋 Extracting {len(unique_parent_ids)} unique parent chunks from {len(child_chunks)} child chunks")
        
        # Query parent chunks
        parent_chunks = self.db.query(ParentChunk).filter(
            ParentChunk.id.in_(unique_parent_ids)
        ).all()
        
        # Convert to dict format
        parent_chunks_dict = [
            {
                'id': pc.id,
                'content': pc.content,
                'document_id': str(pc.document_id),
                'h1': pc.h1,
                'h2': pc.h2,
                'h3': pc.h3,
                'chunk_index': pc.chunk_index,
                'metadata': pc.meta_data
            }
            for pc in parent_chunks
        ]
        
        print(f"✅ Retrieved {len(parent_chunks_dict)} parent chunks for context")
        return parent_chunks_dict
    
    def retrieve_hierarchical(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        summary_k: int = 5,
        chunk_k: int = 20,
        min_summary_score: float = 0.005
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval workflow with 6 steps:
        
        Args:
            question: User question
            document_ids: Optional list of document IDs to scope search (filters summaries)
            summary_k: Number of summary documents to retrieve
            chunk_k: Number of child chunks to retrieve
            min_summary_score: Minimum score threshold for summary documents
        
        Step 1: Query on summary documents (hybrid search) => get relevant documents (determine scope)
                If document_ids provided, only search summaries linked to those documents
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
        if document_ids:
            print(f"Filtering by document IDs: {document_ids}")
        
        # If document_ids provided, find associated summary_ids
        summary_ids = None
        if document_ids:
            from app.database.models import SummaryDocument
            from sqlalchemy import select
            
            # Query summary documents that are linked to the provided document_ids
            stmt = select(SummaryDocument.id).where(
                SummaryDocument.documents.any(Document.id.in_(document_ids))
            )
            result = self.db.execute(stmt)
            summary_ids = [row[0] for row in result.fetchall()]
            print(f"Found {len(summary_ids)} summary documents linked to provided documents")
            
            if not summary_ids:
                print("No summary documents found for provided document IDs, falling back to full search")
                summary_ids = None
        
        # STEP 1: Query on summary documents
        print(f"\n📋 STEP 1: Search summary documents")
        summary_docs = self.search_service.hybrid_search_summaries(
            query=question,
            k=summary_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            summary_ids=summary_ids
        )
        
        max_score = summary_docs[0]['fused_score'] if summary_docs else 0.0
        print(f"Found {len(summary_docs)} summary docs, max score: {round(max_score, 4)}")
        
        # Log chi tiết các summary documents
        if summary_docs:
            print(f"\n📄 Top {len(summary_docs)} Summary Documents:")
            for i, doc in enumerate(summary_docs, 1):
                summary_id = doc.get('id', 'N/A')
                score = doc.get('fused_score', 0.0)
                content_preview = doc.get('summary_content', '')[:200].replace('\n', ' ')
                print(f"\n  {i}. Summary ID: {summary_id}")
                print(f"     Score: {round(score, 4)}")
                print(f"     Preview: {content_preview}...")
        
        # Check if we should fall back to full child chunk search
        if not summary_docs or max_score < min_summary_score:
            print(f"⚠️ No good summary docs found (max score < {min_summary_score})")
            print(f"Falling back to full child chunk search with 2-pass retrieval")
            
            # PASS 1: Initial search on all child chunks
            print(f"\n📦 FALLBACK - Pass 1: Initial search")
            first_pass_chunks = self.search_service.hybrid_search(
                query=question,
                k=self.first_pass_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                summary_ids=summary_ids
            )
            print(f"Found {len(first_pass_chunks)} chunks in first pass")
            
            # Log the 12 chunks from pass 1 (fallback mode)
            print(f"\n📦 PASS 1: Initial {len(first_pass_chunks)} chunks from fallback search:")
            for i, chunk in enumerate(first_pass_chunks, 1):
                # Build headers string
                headers = []
                if chunk.get('h1'): headers.append(chunk['h1'])
                if chunk.get('h2'): headers.append(chunk['h2'])
                if chunk.get('h3'): headers.append(chunk['h3'])
                headers_str = " / ".join(headers) if headers else ""
                
                content_preview = chunk.get('content', '')[:150].replace('\n', ' ')
                print(f"📄 Doc {i}:")
                print(f"   Type: Child Chunk")
                print(f"   Headers: {headers_str}")
                print(f"   Preview: {content_preview}...")
                print()
            
            # Extract entity info for query expansion
            print(f"\n🔄 FALLBACK - Query expansion")
            info = self._extract_entity_info(question, first_pass_chunks)
            entity = info.get("entity", "")
            aliases = info.get("aliases") or []
            keywords = info.get("keywords") or []
            
            print(f"🧠 Entity: {entity or '(none)'} | Aliases: {len(aliases)} | Keywords: {len(keywords)}")
            
            # Generate query variants
            variants = self._make_variants(question, info)
            print(f"🧩 Variants: {len(variants)}")
            for i, v in enumerate(variants, 1):
                print(f"   Q{i}: {v}")
            
            # PASS 2: Search with variants on all child chunks
            print(f"\n🔎 FALLBACK - Pass 2: Search with variants")
            all_results = [first_pass_chunks]
            variant_results = []  # Track results for each variant
            
            for i, variant in enumerate(variants, 1):
                print(f"Variant {i}/{len(variants)}: {variant[:60]}...")
                results = self.search_service.hybrid_search(
                    query=variant,
                    k=max(chunk_k, 20),
                    bm25_weight=self.bm25_weight,
                    semantic_weight=self.semantic_weight,
                    rrf_k=self.rrf_k,
                    summary_ids=summary_ids
                )
                all_results.append(results)
                variant_results.append({
                    'variant': variant,
                    'top_3_chunks': results[:3]  # Store top 3 for this variant
                })
                print(f"  → {len(results)} chunks")
            
            # RRF fusion
            print(f"\n📊 FALLBACK - RRF fusion of {len(all_results)} result sets")
            fused_chunks = self._rrf_fuse(all_results, rrf_k=self.rrf_k, top_k=chunk_k)
            print(f"✅ Fused: {len(fused_chunks)} chunks")
            
            return {
                'docs': fused_chunks,
                'source': 'chunks_fallback',
                'metadata': {
                    'summary_docs_found': len(summary_docs),
                    'max_summary_score': max_score,
                    'chunks_returned': len(fused_chunks),
                    'variants_count': len(variants),
                    'variants': variants,
                    'variant_results': variant_results,
                    'fallback_mode': 'two_pass'
                }
            }
        
        # Extract summary IDs from summary docs for scoped search
        summary_ids = [doc['id'] for doc in summary_docs]
        print(f"Scope: {len(summary_ids)} summary documents")
        
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
            # First get some child chunks from the relevant summaries for context
            initial_chunks = self.search_service.hybrid_search(
                query=question,
                k=self.first_pass_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                summary_ids=summary_ids
            )
            
            # Log the 12 chunks from pass 1
            print(f"\n📦 PASS 1: Initial {len(initial_chunks)} chunks from scoped search:")
            for i, chunk in enumerate(initial_chunks, 1):
                # Build headers string
                headers = []
                if chunk.get('h1'): headers.append(chunk['h1'])
                if chunk.get('h2'): headers.append(chunk['h2'])
                if chunk.get('h3'): headers.append(chunk['h3'])
                headers_str = " / ".join(headers) if headers else ""
                
                content_preview = chunk.get('content', '')[:150].replace('\n', ' ')
                print(f"📄 Doc {i}:")
                print(f"   Type: Child Chunk")
                print(f"   Headers: {headers_str}")
                print(f"   Preview: {content_preview}...")
                print()
            
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
            
            # STEP 4: Search child chunks with each variant (scoped to summaries)
            print(f"\n🔎 STEP 4: Search child chunks with variants (scoped to summaries)")
            all_results = []
            variant_results = []  # Track results for each variant
            
            for i, variant in enumerate(variants, 1):
                print(f"Variant {i}/{len(variants)}: {variant[:60]}...")
                results = self.search_service.hybrid_search(
                    query=variant,
                    k=max(chunk_k, 20),
                    bm25_weight=self.bm25_weight,
                    semantic_weight=self.semantic_weight,
                    rrf_k=self.rrf_k,
                    summary_ids=summary_ids
                )
                all_results.append(results)
                variant_results.append({
                    'variant': variant,
                    'top_3_chunks': results[:3]  # Store top 3 for this variant
                })
                print(f"  → {len(results)} child chunks")
            
            # STEP 5: RRF fusion of all variant results
            print(f"\n📊 STEP 5: RRF fusion of {len(all_results)} result sets")
            fused_child_chunks = self._rrf_fuse(all_results, rrf_k=self.rrf_k, top_k=chunk_k)
            print(f"✅ Fused: {len(fused_child_chunks)} child chunks")
            
            # STEP 5.5: Get parent chunks from child chunks for context
            print(f"\n📋 STEP 5.5: Retrieve parent chunks for context")
            parent_chunks = self._get_parent_chunks_context(fused_child_chunks)
            
            # STEP 6: Return parent chunks for answer generation
            return {
                'docs': parent_chunks,
                'source': 'parent_chunks_from_children',
                'metadata': {
                    'summary_docs_count': len(summary_docs),
                    'max_summary_score': max_score,
                    'sufficient': False,
                    'variants_count': len(variants),
                    'variants': variants,
                    'variant_results': variant_results,
                    'child_chunks_found': len(fused_child_chunks),
                    'parent_chunks_returned': len(parent_chunks),
                    'scoped_to_summaries': len(summary_ids)
                }
            }
    
    def chat(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        RAG chat: retrieve + generate answer using hierarchical retrieval
        
        Args:
            question: User question
            document_ids: Optional list of document IDs to scope search
            verbose: Print detailed context
        
        Returns:
            {
                'answer': str,
                'chunks': List[Dict],
                'metadata': Dict
            }
        """
        # Use hierarchical retrieval workflow
        result = self.retrieve_hierarchical(question, document_ids=document_ids)
        docs = result['docs']
        source = result['source']
        metadata = result['metadata']
        metadata['retrieval_method'] = 'hierarchical'
        
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
                elif source == 'parent_chunks_from_children':
                    print(f"   Type: Parent Chunk")
                    print(f"   Headers: {doc.get('h1', '')} / {doc.get('h2', '')}")
                    print(f"   Preview: {doc.get('content', '')[:200]}...")
                else:
                    print(f"   Type: Child Chunk")
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
        elif source == 'parent_chunks_from_children':
            # Use parent chunks (already in correct format)
            answer = self.rag_chain.invoke({"docs": docs, "question": question})
        else:
            # Use child chunks or legacy chunks
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
