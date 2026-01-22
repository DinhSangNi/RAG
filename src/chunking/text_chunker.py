from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone, ServerlessSpec
import os
import pickle
import time
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DIMENSION_OF_MODEL = os.getenv("DIMENSION_OF_MODEL")


class HybridSectionChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 1) Split theo markdown headers
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        # ✅ SỬA: giữ headers trong content để tăng tín hiệu retrieval
        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

        # 2) Split tiếp nếu section lớn
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=GEMINI_API_KEY
        )

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

    def _get_or_create_index(self, index_name):
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"🔧 Tạo Pinecone index mới: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=int(DIMENSION_OF_MODEL),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print(f"✅ Index {index_name} đã sẵn sàng!")
        else:
            print(f"✅ Sử dụng index có sẵn: {index_name}")

        return self.pc.Index(index_name)

    def chunk_and_save_to_db(self, md_file_path, index_name="knowledge-base",
                             chunks_dir="data/chunks", reset=False):
        print(f"\n🔪 HYBRID SECTION CHUNKING")
        print("=" * 70)
        print(f"📄 File: {md_file_path}")
        print(f"📏 Chunk size: {self.chunk_size}")
        print(f"🔗 Chunk overlap: {self.chunk_overlap}")

        os.makedirs(chunks_dir, exist_ok=True)

        if reset:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if index_name in existing_indexes:
                print(f"🗑️  Xóa index cũ: {index_name}")
                self.pc.delete_index(index_name)
                time.sleep(1)

        with open(md_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n📑 Bước 1: Tách theo markdown headers...")
        section_docs = self.section_splitter.split_text(text)
        print(f"→ {len(section_docs)} sections")

        print(f"🔨 Bước 2: Recursive split cho sections lớn...")
        final_chunks = []

        for idx, section_doc in enumerate(section_docs):
            section_doc.metadata.update({
                "source": md_file_path,
                "section_id": idx,
                "document": os.path.basename(md_file_path).replace(".md", "")
            })

            if len(section_doc.page_content) > self.chunk_size:
                sub_chunks = self.recursive_splitter.split_documents([section_doc])
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata.update({
                        "sub_chunk_id": sub_idx,
                        "total_sub_chunks": len(sub_chunks)
                    })
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(section_doc)

        print(f"   → {len(final_chunks)} chunks cuối cùng")

        print(f"\n💾 Bước 3: Lưu vào Pinecone...")
        self._get_or_create_index(index_name)

        PineconeVectorStore.from_documents(
            documents=final_chunks,
            embedding=self.embeddings,
            index_name=index_name
        )

        chunks_file = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        with open(chunks_file, "wb") as f:
            pickle.dump(final_chunks, f)
        print(f"💾 Đã lưu chunks vào {chunks_file}")

        print(f"\n✅ HOÀN TẤT!")
        print(f"📊 {len(section_docs)} sections → {len(final_chunks)} chunks")
        print(f"💾 Lưu vào Pinecone index: {index_name}")

    # ==============================
    # ✅ SỬA CHUNG: HYBRID SEARCH = BM25 + SEMANTIC dùng RRF
    # ==============================
    def query_with_hybrid_search(self, query, index_name="knowledge-base",
                                 chunks_dir="data/chunks", k=10,
                                 bm25_weight=0.5, semantic_weight=0.5,
                                 bm25_k=None, semantic_k=None, semantic_fetch_k=50,
                                 rrf_k=60):
        """
        Hybrid retrieval tổng quát bằng Reciprocal Rank Fusion (RRF).
        - Không hardcode theo câu hỏi
        - Ổn định hơn EnsembleRetriever
        """

        bm25_k = bm25_k or max(k, 10)
        semantic_k = semantic_k or max(k, 10)

        print(f"\n🔍 HYBRID SEARCH QUERY")
        print("=" * 70)
        print(f"❓ Query: {query}")
        print(f"🎯 Final Top K: {k}")
        print(f"⚖️ Weights: BM25={bm25_weight}, Semantic={semantic_weight}")
        print(f"📌 Internal: bm25_k={bm25_k}, semantic_k={semantic_k}, fetch_k={semantic_fetch_k}, rrf_k={rrf_k}")

        # load vectorstore
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings
        )

        # load chunks for BM25
        chunks_file = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = bm25_k

        # ✅ Semantic retriever dùng MMR + fetch_k lớn để tăng recall
        semantic_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": semantic_k, "fetch_k": semantic_fetch_k}
        )

        bm25_docs = bm25_retriever.invoke(query) or []
        sem_docs = semantic_retriever.invoke(query) or []

        print(f"📄 BM25 got: {len(bm25_docs)} docs | Semantic got: {len(sem_docs)} docs")

        # ---- RRF fusion ----
        # key doc by stable identifier (ưu tiên source + section_id + sub_chunk_id + hash content)
        def doc_key(d):
            src = d.metadata.get("source", "")
            sid = d.metadata.get("section_id", "")
            sub = d.metadata.get("sub_chunk_id", "")
            # nội dung có thể trùng, nên thêm 1 phần content
            return f"{src}|{sid}|{sub}|{d.page_content[:80]}"

        scores = {}

        def add_rrf(docs, w):
            for rank, d in enumerate(docs, start=1):
                key = doc_key(d)
                # RRF score
                scores.setdefault(key, {"doc": d, "score": 0.0})
                scores[key]["score"] += w * (1.0 / (rrf_k + rank))

        add_rrf(bm25_docs, bm25_weight)
        add_rrf(sem_docs, semantic_weight)

        fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        results = [x["doc"] for x in fused[:k]]

        print("✅ Fused results:", len(results))
        if results:
            # 3 dòng log gọn để debug
            print("LOG1 top1 header:", results[0].metadata.get("h1", "") or results[0].metadata.get("h2", ""))
            print("LOG2 top1 preview:", results[0].page_content[:120].replace("\n", " "))
            print("LOG3 top scores:", [round(x["score"], 6) for x in fused[:3]])

        return results