from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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
    
    def __init__(self, chunk_size=1000, chunk_overlap=100, child_chunk_size=400):
        self.chunk_size = chunk_size  # Parent chunk size threshold
        self.chunk_overlap = chunk_overlap
        self.child_chunk_size = child_chunk_size  # Child chunk size
        
        # 1. Section splitter - tách theo markdown headers
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            # ("####", "h4")
        ]
        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Giữ headers trong content
        )
        
        # 2. Parent splitter - tách section lớn thành parent chunks
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 3. Child splitter - tách parent chunks thành child chunks
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap // 2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=GEMINI_API_KEY
        )
        
        # Khởi tạo Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
    
    def _get_or_create_index(self, index_name):
        """Tạo hoặc lấy Pinecone index"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"🔧 Tạo Pinecone index mới: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=int(DIMENSION_OF_MODEL), 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Đợi index được tạo xong
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print(f"✅ Index {index_name} đã sẵn sàng!")
        else:
            print(f"✅ Sử dụng index có sẵn: {index_name}")
        
        return self.pc.Index(index_name)
    
    def chunk_and_save_to_db(self, md_file_path, index_name="knowledge-base", 
                             chunks_dir="data/chunks", reset=False):
        
        print(f"\n🔪 SMART PARENT-CHILD CHUNKING")
        print(f"="*70)
        print(f"📄 File: {md_file_path}")
        print(f"👨 Parent chunk threshold: {self.chunk_size} chars")
        print(f"👶 Child chunk size: {self.child_chunk_size} chars")
        print(f"🔗 Overlap: {self.chunk_overlap} chars")
        
        # Tạo thư mục lưu chunks pickle
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Xóa index cũ nếu reset
        if reset:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if index_name in existing_indexes:
                print(f"🗑️  Xóa index cũ: {index_name}")
                self.pc.delete_index(index_name)
                time.sleep(1)
        
        # Load document
        with open(md_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Step 1: Tách theo headers (sections)
        print(f"\n📑 Bước 1: Tách theo markdown headers...")
        section_docs = self.section_splitter.split_text(text)
        print(f"→ {len(section_docs)} sections")
        
        # Step 2: Phân loại và xử lý sections
        print(f"\n🔍 Bước 2: Phân loại và chunking sections...")
        chunks_to_save = []  # Chunks để lưu vào DB (child chunks + small sections)
        parent_map = {}  # Map parent_id -> parent document
        
        stats = {"small": 0, "large": 0, "total_child_chunks": 0}
        
        for section_idx, section_doc in enumerate(section_docs):
            section_size = len(section_doc.page_content)
            section_id = f"{os.path.basename(md_file_path).replace('.md', '')}_section_{section_idx}"
            
            # Thêm metadata cơ bản
            section_doc.metadata.update({
                "source": md_file_path,
                "section_id": section_idx,
                "document": os.path.basename(md_file_path).replace('.md', ''),
                "section_size": section_size
            })
            
            # Case 1: Section nhỏ (≤ child_chunk_size) - Lưu trực tiếp
            if section_size <= self.child_chunk_size:
                section_doc.metadata.update({
                    "chunk_type": "small_section",
                    "chunk_id": section_id
                })
                chunks_to_save.append(section_doc)
                stats["small"] += 1
                print(f"   ✓ Section {section_idx}: {section_size} chars → Small (lưu nguyên)")
            
            # Case 2: Section lớn (> child_chunk_size) - Parent-Child chunking
            else:
                # Step 3a: Tách thành parent chunks
                parent_chunks = self.parent_splitter.split_documents([section_doc])
                
                print(f"   📦 Section {section_idx}: {section_size} chars → Large → {len(parent_chunks)} parents", end="")
                
                # Step 3b: Tách mỗi parent thành child chunks
                child_count = 0
                for parent_idx, parent_chunk in enumerate(parent_chunks):
                    parent_id = f"{section_id}_parent_{parent_idx}"
                    parent_text = parent_chunk.page_content
                    
                    # Lưu parent document để reconstruct
                    parent_map[parent_id] = {
                        'text': parent_text,
                        'metadata': {
                            **parent_chunk.metadata,
                            'parent_id': parent_id,
                            'parent_idx': parent_idx,
                            'total_parents': len(parent_chunks)
                        }
                    }
                    
                    # Tách parent thành child chunks
                    child_chunks = self.child_splitter.split_documents([parent_chunk])
                    
                    # Thêm metadata cho child chunks
                    for child_idx, child_chunk in enumerate(child_chunks):
                        child_id = f"{parent_id}_child_{child_idx}"
                        child_chunk.metadata.update({
                            "chunk_type": "child",
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "child_idx": child_idx,
                            "total_children": len(child_chunks)
                        })
                        chunks_to_save.append(child_chunk)
                        child_count += 1
                
                stats["large"] += 1
                stats["total_child_chunks"] += child_count
                print(f" → {child_count} children")
        
        print(f"\n📊 Thống kê:")
        print(f"   • Small sections (≤{self.child_chunk_size}): {stats['small']} (lưu nguyên)")
        print(f"   • Large sections (>{self.child_chunk_size}): {stats['large']} → {stats['total_child_chunks']} child chunks")
        print(f"   • Tổng chunks để lưu: {len(chunks_to_save)}")
        
        # Step 3: Lưu vào Pinecone (chỉ lưu chunks_to_save)
        print(f"\n💾 Bước 3: Lưu {len(chunks_to_save)} chunks vào Pinecone...")
        self._get_or_create_index(index_name)
        
        PineconeVectorStore.from_documents(
            documents=chunks_to_save,
            embedding=self.embeddings,
            index_name=index_name
        )
        
        # Step 4: Lưu data vào pickle (cho BM25 và parent reconstruction)
        chunks_file = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        storage_data = {
            'chunks': chunks_to_save,
            'parent_map': parent_map,
            'stats': stats
        }
        with open(chunks_file, 'wb') as f:
            pickle.dump(storage_data, f)
        print(f"💾 Đã lưu data vào {chunks_file}")
        
        print(f"\n✅ HOÀN TẤT!")
        print(f"📊 {len(section_docs)} sections → {len(chunks_to_save)} chunks (small sections + child chunks)")
        print(f"💾 Lưu vào Pinecone index: {index_name}")
    
    def query_with_hybrid_search(self, query, index_name="knowledge-base", 
                                  chunks_dir="data/chunks", k=5,
                                  bm25_weight=0.5, semantic_weight=0.5,
                                  return_parent=True):
        """
        Hybrid search với smart parent-child retrieval
        
        Args:
            return_parent: Nếu True, child chunks sẽ được expand về parent chunks
        """
        
        print(f"\n🔍 SMART HYBRID SEARCH")
        print(f"="*70)
        print(f"❓ Query: {query}")
        print(f"🎯 Top K: {k}")
        print(f"⚖️ Weights: BM25={bm25_weight}, Semantic={semantic_weight}")
        print(f"👨‍👦 Return parent: {return_parent}")
        
        # Load vectorstore
        print(f"\n📂 Đang load Pinecone vectorstore...")
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings
        )
        
        # Load data
        chunks_file = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        with open(chunks_file, 'rb') as f:
            storage_data = pickle.load(f)
        
        chunks = storage_data['chunks']
        parent_map = storage_data.get('parent_map', {})
        
        print(f"📂 Đã load {len(chunks)} chunks và {len(parent_map)} parent documents")
        
        # Tạo BM25 retriever
        print(f"🔤 Khởi tạo BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = k * 2  # Lấy nhiều hơn để có thể expand
        
        # Tạo Semantic retriever
        print(f"🧠 Khởi tạo Semantic retriever...")
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})
        
        # Ensemble retriever
        print(f"🔀 Tạo Ensemble retriever...")
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[bm25_weight, semantic_weight]
        )
        
        # Query
        print(f"\n🔎 Đang search...")
        results = hybrid_retriever.invoke(query)
        
        # Xử lý kết quả
        if return_parent:
            print(f"👨‍👦 Expanding child chunks về parent chunks...")
            final_results = []
            seen_parents = set()
            seen_chunks = set()
            
            for result in results:
                chunk_type = result.metadata.get('chunk_type')
                chunk_id = result.metadata.get('chunk_id')
                
                # Bỏ qua duplicate
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                
                # Case 1: Child chunk → expand về parent
                if chunk_type == 'child':
                    parent_id = result.metadata.get('parent_id')
                    
                    # Nếu chưa thấy parent này
                    if parent_id and parent_id not in seen_parents and parent_id in parent_map:
                        parent_info = parent_map[parent_id]
                        
                        # Tạo document từ parent
                        parent_doc = Document(
                            page_content=parent_info['text'],
                            metadata={
                                **parent_info['metadata'],
                                'matched_child_id': chunk_id,
                                'matched_child_text': result.page_content[:100] + "..."
                            }
                        )
                        final_results.append(parent_doc)
                        seen_parents.add(parent_id)
                        
                        if len(final_results) >= k:
                            break
                
                # Case 2: Small/Medium section → giữ nguyên
                else:
                    final_results.append(result)
                    if len(final_results) >= k:
                        break
            
            print(f"   → Expanded {len([r for r in final_results if 'matched_child_id' in r.metadata])} child chunks về parent")
        else:
            # Trả về raw chunks
            final_results = results[:k]
        
        print(f"✅ Tìm thấy {len(final_results)} kết quả!")
        
        return final_results

