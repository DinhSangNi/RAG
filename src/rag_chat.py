import os
import re
import json
import math
import pickle
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.chunking.text_chunker import HybridSectionChunker

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================
# Utility helpers (general)
# =========================

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)

def _tokenize_vi(text: str) -> List[str]:
    """Very light tokenizer: Vietnamese-friendly word chunks."""
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]

def _doc_key(doc: Document) -> str:
    """Stable key for dedup across retrievers/variants."""
    md = doc.metadata or {}
    src = md.get("source", "")
    sec = md.get("section_id", "")
    sub = md.get("sub_chunk_id", "")
    docname = md.get("document", "")
    # include small content hash to avoid collisions
    content = (doc.page_content or "")[:2000]
    return f"{docname}|{src}|{sec}|{sub}|{hash(content)}"

def _rrf_fuse(list_of_doclists: List[List[Document]], rrf_k: int = 60, top_k: int = 10) -> List[Document]:
    """
    Reciprocal Rank Fusion on multiple ranked lists.
    Score(doc) = sum(1/(rrf_k + rank))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for docs in list_of_doclists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = [doc_map[k] for k, _ in ranked[:top_k]]
    return fused


# =========================
# RAG Chat (Option 2)
# =========================

class RAGChat:
    def __init__(
        self,
        index_name: str = "knowledge-base",
        chunks_dir: str = "data/chunks",
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.1,
        top_k: int = 20,
        bm25_weight: float = 0.6,
        semantic_weight: float = 0.4,
        # retrieval strategy knobs (general, not entity-specific)
        first_pass_k: int = 12,
        variant_count: int = 5,
        rrf_k: int = 60,
        auto_stop_df_threshold: float = 0.35,  # token appears in >=35% docs -> stop
        auto_stop_max_size: int = 250,         # cap size to avoid over-filter
    ):
        self.index_name = index_name
        self.chunks_dir = chunks_dir

        self.top_k = top_k
        self.first_pass_k = first_pass_k
        self.variant_count = variant_count
        self.rrf_k = rrf_k

        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        self.auto_stop_df_threshold = auto_stop_df_threshold
        self.auto_stop_max_size = auto_stop_max_size
        self._auto_stopwords: Optional[set] = None

        # Chunker (retriever)
        self.chunker = HybridSectionChunker(chunk_size=600, chunk_overlap=150)

        # LLM
        print(f"🤖 Khởi tạo {model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        # Prompt answering
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

        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(x["docs"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # A tiny chain for alias extraction (LLM-based, no regex)
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
        self.alias_chain = self.alias_prompt | self.llm | StrOutputParser()

        print("✅ RAG Chat sẵn sàng!")

    # -------------------------
    # Auto stopwords from corpus
    # -------------------------

    def _load_all_chunks(self) -> List[Document]:
        chunks_file = os.path.join(self.chunks_dir, f"{self.index_name.replace('-', '_')}_chunks.pkl")
        with open(chunks_file, "rb") as f:
            return pickle.load(f)

    def _build_auto_stopwords(self) -> set:
        """
        Build stopword list from corpus:
        token is stopword if it appears in >= threshold fraction of docs (DF-based).
        """
        try:
            chunks = self._load_all_chunks()
        except Exception as e:
            print(f"⚠️ Không thể load chunks để build auto-stopwords: {e}")
            return set()

        n_docs = max(1, len(chunks))
        df: Dict[str, int] = {}

        for doc in chunks:
            tokens = set(_tokenize_vi(doc.page_content or ""))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        threshold = int(math.ceil(self.auto_stop_df_threshold * n_docs))
        # tokens with high DF
        high_df = [(t, c) for t, c in df.items() if c >= threshold and len(t) >= 2]
        # sort by DF desc
        high_df.sort(key=lambda x: x[1], reverse=True)

        stop = set([t for t, _ in high_df[: self.auto_stop_max_size]])
        print(f"🧹 Auto-stopwords built: {len(stop)} tokens (DF≥{threshold}/{n_docs})")
        return stop

    def _get_stopwords(self) -> set:
        if self._auto_stopwords is None:
            self._auto_stopwords = self._build_auto_stopwords()
        return self._auto_stopwords

    # -------------------------
    # Formatting context
    # -------------------------

    def _format_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        formatted = []
        for i, doc in enumerate(docs, 1):
            md = doc.metadata or {}
            header = md.get("h2", md.get("h1", "")) or ""
            source = md.get("document", "") or os.path.basename(md.get("source", "") or "")
            formatted.append(f"--- Đoạn {i} | {source} | {header} ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    # -------------------------
    # Query variant generation (general, no hardcode)
    # -------------------------

    def _extract_entity_aliases_keywords(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """
        Use LLM to extract entity + aliases + keywords from retrieved context.
        No regex hardcode.
        """
        context = self._format_docs(docs[: min(len(docs), 8)])
        try:
            raw = self.alias_chain.invoke({"question": question, "context": context})
            data = json.loads(raw)
            entity = (data.get("entity") or "").strip()
            aliases = data.get("aliases") or []
            keywords = data.get("keywords") or []
            # normalize
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
            # avoid duplicates
            aliases = list(dict.fromkeys(aliases))
            keywords = list(dict.fromkeys(keywords))
            return {"entity": entity, "aliases": aliases, "keywords": keywords}
        except Exception as e:
            print(f"⚠️ Alias extraction failed, fallback no-alias. Error: {e}")
            return {"entity": "", "aliases": [], "keywords": []}

    def _make_variants(self, question: str, info: Dict[str, Any]) -> List[str]:
        """
        Build query variants from entity/aliases/keywords.
        Still general: it works for any person/org as long as context includes the alias.
        """
        entity = (info.get("entity") or "").strip()
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []

        # Base keywords (filtered by auto-stopwords)
        stop = self._get_stopwords()
        q_tokens = _tokenize_vi(question)
        q_core = [t for t in q_tokens if t not in stop]
        # keep order-ish by joining original filtered tokens
        core_text = " ".join(q_core).strip()

        variants = []
        variants.append(question)

        # If we have entity, combine it with core question
        if entity:
            if core_text:
                variants.append(f"{entity} {core_text}")
            variants.append(f"{entity} {question}")

        # Alias-based variants
        for a in aliases[: max(0, self.variant_count)]:
            if core_text:
                variants.append(f"{a} {core_text}")
            else:
                variants.append(f"{a} {question}")

        # Keyword-based variants (if LLM gave good keywords)
        if entity and keywords:
            variants.append(f"{entity} " + " ".join(keywords[:8]))
        elif keywords:
            variants.append(" ".join(keywords[:8]))

        # Dedup while preserving order
        deduped = []
        seen = set()
        for v in variants:
            v2 = v.strip()
            if not v2:
                continue
            if v2.lower() in seen:
                continue
            seen.add(v2.lower())
            deduped.append(v2)

        # cap variants
        return deduped[: max(3, self.variant_count)]

    # -------------------------
    # Retrieval orchestration
    # -------------------------

    def _retrieve_once(self, query: str, k: int) -> List[Document]:
        return self.chunker.query_with_hybrid_search(
            query=query,
            index_name=self.index_name,
            chunks_dir=self.chunks_dir,
            k=k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight
        )

    def retrieve(self, question: str) -> List[Document]:
        """
        Two-pass retrieval:
        - Pass 1: retrieve with raw question (small k)
        - Extract entity/aliases/keywords via LLM from pass1 context
        - Generate variants, retrieve each variant
        - RRF fuse all results -> final top_k
        """
        print(f"\n🔍 Retrieve: '{question}'")

        # Pass 1 (small)
        first = self._retrieve_once(question, k=self.first_pass_k)
        print(f"📦 Pass1 got: {len(first)} chunks")

        # LLM extracts entity+aliases+keywords from pass1 context
        info = self._extract_entity_aliases_keywords(question, first)

        entity = info.get("entity") or ""
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        print(f"🧠 Entity: {entity or '(none)'} | Aliases: {len(aliases)} | Keywords: {len(keywords)}")

        # Build variants
        variants = self._make_variants(question, info)
        print(f"🧩 Variants: {len(variants)}")
        for i, v in enumerate(variants, 1):
            print(f"   Q{i}: {v}")

        # Pass 2: retrieve for each variant
        lists = []
        for v in variants:
            docs = self._retrieve_once(v, k=max(self.top_k, 20))
            lists.append(docs)

        # Include pass1 in fusion as well
        lists.insert(0, first)

        fused = _rrf_fuse(lists, rrf_k=self.rrf_k, top_k=self.top_k)
        print(f"✅ Fused retrieve: {len(fused)} chunks")

        return fused

    # -------------------------
    # Chat
    # -------------------------

    def chat(self, question: str, verbose: bool = False) -> str:
        docs = self.retrieve(question)

        if not docs:
            return "Tôi không tìm thấy thông tin này trong tài liệu."

        if verbose:
            print(f"\n{'='*70}\nCONTEXT ĐƯỢC RETRIEVE:\n{'='*70}")
            for i, doc in enumerate(docs, 1):
                md = doc.metadata or {}
                print(f"\n📄 Chunk {i}:")
                print(f"   Headers: {md.get('h1', '')} / {md.get('h2', '')}")
                print(f"   Preview: {(doc.page_content or '')[:220]}...")
                print(f"   {'-'*70}")

        print("\n💬 Đang generate câu trả lời...")
        answer = self.rag_chain.invoke({"docs": docs, "question": question})
        answer = (answer or "").strip()

        # Enforce exact fallback sentence
        if not answer:
            return "Tôi không tìm thấy thông tin này trong tài liệu."
        if "không tìm thấy" in answer.lower() and "tài liệu" in answer.lower():
            # normalize to exact required sentence
            return "Tôi không tìm thấy thông tin này trong tài liệu."
        return answer


# ============================================================================
# INTERACTIVE CHAT
# ============================================================================

def interactive_chat():
    rag = RAGChat(
        index_name="knowledge-base",
        chunks_dir="data/chunks",
        model_name="gemini-2.5-flash-lite",
        temperature=0.1,
        top_k=20,
        bm25_weight=0.6,
        semantic_weight=0.4,
        first_pass_k=12,
        variant_count=5,
        rrf_k=60,
        auto_stop_df_threshold=0.35,
        auto_stop_max_size=250
    )

    print(f"\n{'='*70}")
    print("🤖 RAG CHAT - Hỏi đáp")
    print(f"{'='*70}")
    print("Nhập 'quit' hoặc 'exit' để thoát")
    print("Nhập 'verbose' để bật/tắt hiển thị context")
    print(f"{'='*70}\n")

    verbose = False

    while True:
        question = input("❓ Câu hỏi: ").strip()

        if question.lower() in ["quit", "exit", "thoát"]:
            print("👋 Tạm biệt!")
            break

        if question.lower() == "verbose":
            verbose = not verbose
            print(f"✅ Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        if not question:
            continue

        try:
            answer = rag.chat(question, verbose=verbose)
            print(f"\n💡 TRẢ LỜI: {answer}\n")
        except Exception as e:
            print(f"❌ Lỗi: {e}\n")


if __name__ == "__main__":
    interactive_chat()
