"""
RAG (Retrieval-Augmented Generation) Service
Implements two-pass retrieval with query expansion and RRF fusion
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


class RAGService:
    """Service for RAG chat with advanced retrieval strategies"""
    
    def __init__(
        self,
        db: Session,
        model_name: str = settings.GEMINI_MODEL_NAME,
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
        
        # Cập nhật trong __init__
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn là trợ lý chuyên gia phân tích tài liệu. Trả lời câu hỏi dựa trên CONTEXT được cung cấp.

        NGUYÊN TẮC:
        1) ƯU TIÊN: Nếu có câu trả lời trực tiếp, hãy trả lời ngắn gọn, chính xác.
        2) NẾU DỮ LIỆU YẾU: Nếu không có câu trả lời trực tiếp nhưng CONTEXT có nhắc đến các thực thể hoặc từ khóa liên quan, hãy tóm tắt những gì tài liệu có về chúng và bắt đầu bằng: "Dựa trên tài liệu tôi tìm thấy, dù không có thông tin cụ thể về [vấn đề], nhưng có một vài chi tiết liên quan là..."
        3) TRƯỜNG HỢP KHÔNG CÓ THÔNG TIN: Chỉ khi CONTEXT hoàn toàn không có từ khóa nào liên quan, hãy trả lời đúng câu: "Tôi không tìm thấy thông tin này trong tài liệu."
        4) Coi các bí danh/tên gọi khác là cùng một thực thể.
        5) Tuyệt đối không bịa đặt thông tin nằm ngoài CONTEXT.

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
        
        print("✅ RAG Service ready!")

    def _calculate_confidence(self, question: str, docs: List[Dict[str, Any]], info: Dict[str, Any]) -> float:
        """Tính toán mức độ tự tin của dữ liệu tìm được (0.0 - 1.0)"""
        if not docs:
            return 0.0
        
        # Lấy danh sách từ khóa chính từ info (do LLM trích xuất ở Pass 1)
        keywords = info.get("keywords") or []
        entity = info.get("entity", "")
        all_terms = [entity.lower()] + [k.lower() for k in keywords] if entity else [k.lower() for k in keywords]
        
        # 1. Kiểm tra mức độ khớp từ khóa trong Context
        context_text = " ".join([d.get('content', '').lower() for d in docs])
        matches = sum(1 for term in all_terms if term in context_text)
        keyword_score = matches / len(all_terms) if all_terms else 0.5
        
        # 2. Kiểm tra điểm RRF cao nhất (tín hiệu từ Search Engine)
        # RRF score thường nhỏ, chúng ta có thể normalize hoặc check max_score
        max_fused_score = docs[0].get('fused_score', 0)
        
        # Trọng số: 70% khớp từ khóa, 30% điểm tìm kiếm
        confidence = (keyword_score * 0.7) + (min(max_fused_score * 10, 1.0) * 0.3)
        return confidence
    
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
            return "Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.get('content') or "Nội dung trống"
            # Ưu tiên lấy H2, không có thì lấy H1
            title = doc.get('h2') or doc.get('h1') or f"Tài liệu {i}"
            formatted.append(f"--- {title} ---\n{content}")
        
        return "\n\n".join(formatted)
    
    def _extract_entity_info(
        self, 
        question: str, 
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract entity, aliases, and keywords using LLM"""
        context = self._format_docs(docs[:min(len(docs), 8)])
        
        try:
            raw = self.alias_chain.invoke({
                "question": question, 
                "context": context
            })
            
            # Clean response - handle markdown code blocks
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            
            if raw.endswith("```"):
                raw = raw[:-3]
            
            raw = raw.strip()
            
            # Handle empty response
            if not raw:
                print(f"⚠️ Entity extraction: Empty response from LLM")
                return {"entity": "", "aliases": [], "keywords": []}
            
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
            print(f"⚠️ Entity extraction failed - Invalid JSON: {e}")
            print(f"   Raw response: {raw[:200] if raw else '(empty)'}...")
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
        verbose: bool = False
    ) -> Dict[str, Any]:
        # 1. Truy hồi dữ liệu
        # Lưu ý: Retrieve đã chạy extract_entity_info nội bộ, 
        # nhưng để lấy info ta có thể gọi nhẹ lại hoặc refactor retrieve để trả về cả info.
        # Ở đây tôi giả định retrieve trả về list docs như cũ.
        docs = self.retrieve(question, document_ids=document_ids)
        
        # 2. Đánh giá nhanh nội dung (Kiểm tra xem có keywords liên quan không)
        # Bạn có thể lấy info từ logic extract_entity_info
        # Để tối ưu, hãy lưu 'info' từ trong self.retrieve vào biến instance hoặc trả về kèm docs
        
        if not docs:
            return {
                'answer': "Tôi không tìm thấy thông tin này trong tài liệu.",
                'chunks': [],
                'metadata': {'chunks_used': 0, 'confidence': 0}
            }

        # 3. Sinh câu trả lời qua LLM
        print("\n💬 Generating answer...")
        answer = self.rag_chain.invoke({"docs": docs, "question": question})
        answer = (answer or "").strip()
        
        # 4. Hậu xử lý: Nếu Gemini vẫn "cứng nhắc" báo không thấy dù có data
        # Ta có thể kiểm tra nếu answer quá ngắn và chứa phrase từ chối
        if "không tìm thấy thông tin" in answer.lower() and len(docs) > 0:
            # Nếu có docs nhưng Gemini từ chối, ta có thể thử một prompt "ép" tóm tắt
            # Hoặc giữ nguyên để đảm bảo tính an toàn (tránh hallucination)
            pass

        return {
            'answer': answer,
            'chunks': docs[:10],
            'metadata': {
                'chunks_used': len(docs),
                'model': getattr(self.llm, 'model', 'unknown'),
                # 'confidence': confidence # Nếu bạn triển khai bước 2
            }
        }


def get_rag_service(db: Session) -> RAGService:
    """Factory function for RAGService"""
    return RAGService(db)
