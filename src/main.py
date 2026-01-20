from src.chunking.text_chunker import HybridSectionChunker
from src.rag_chat import interactive_chat
import os
import pickle
import traceback

from src.ingestion.get_data_from_wikipedia import get_html_page_from_wikipedia
from src.preprocessing.html_cleaner import clean_wikipedia_html
from src.preprocessing.normalize_markdown import convert_html_to_normalized_md

def print_banner():
    print("\n" + "="*80)
    print(" "*25 + "🚀 RAG SYSTEM - HỒ CHÍ MINH KB")
    print("="*80)


def print_menu():
    print(f"\n{'='*80}")
    print("CHỌN CHỨC NĂNG:")
    print(f"{'='*80}")
    print("1. 📝 Chuẩn bị data")
    print("2. 🔪 Chunking và lưu vào Vector DB")
    print("3. 💬 RAG Chat (Interactive)")
    print("4. 🚀 Chạy cả hai (Chunking → Chat)")
    print("0. ❌ Thoát")
    print(f"{'='*80}")

def prepare_data():
    search_keyword = input("Hãy nhập keyword tìm kiếm trên Wikipedia (mặc định 'Hồ Chí Minh') và nhấn Enter: ").strip()
    if not search_keyword:
        search_keyword = "Hồ Chí Minh"

    # Lấy các file html từ wikipedia
    html_file_path = get_html_page_from_wikipedia(search_keyword)

    # Làm sạch html
    cleaned_html_file_path = clean_wikipedia_html(html_file_path)

    # Chuyển html thành markdown và chuẩn hóa markdown
    convert_html_to_normalized_md(cleaned_html_file_path)

# Hàm chunking các file markdown và lưu vào Chroma DB
def chunking():
    print("\n" + "="*80)
    print("🔪 CHUNKING TẤT CẢ FILE MARKDOWN VÀ LƯU VÀO VECTOR DB")
    print("="*80)
    
    md_dir = "data/processed_data"
    
    if not os.path.exists(md_dir):
        print(f"\n❌ Thư mục không tồn tại: {md_dir}")
        return None
    
    # Lấy tất cả file .md
    md_files = [f for f in os.listdir(md_dir) if f.endswith('.md')]
    
    if not md_files:
        print(f"\n❌ Không tìm thấy file markdown nào trong {md_dir}")
        return None
    
    print(f"\n💫 {len(md_files)} file markdown:")
    for i, f in enumerate(md_files, 1):
        print(f"  {i}. {f}")
    
    index_name = "knowledge-base"
    chunks_dir = "data/chunks"
    
    try:
        chunker = HybridSectionChunker(chunk_size=800, chunk_overlap=150)
        all_chunks = []
        
        print(f"\n🔄 Bắt đầu chunking...")
        
        for idx, md_file in enumerate(md_files, 1):
            md_file_path = os.path.join(md_dir, md_file)
            file_name = md_file.replace('.md', '')
            
            print(f"\n📄 [{idx}/{len(md_files)}] Đang xử lý: {md_file}")
            
            # Load document
            with open(md_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tách theo headers
            section_docs = chunker.section_splitter.split_text(text)
            print(f"   → {len(section_docs)} sections")
            
            # Recursive split cho sections lớn
            file_chunks = []
            for section_idx, section_doc in enumerate(section_docs):
                # Thêm metadata: source file name
                section_doc.metadata.update({
                    "source": md_file_path,
                    "section_id": section_idx,
                    "document": file_name
                })
                
                # Nếu section quá lớn, tách tiếp
                if len(section_doc.page_content) > chunker.chunk_size:
                    sub_chunks = chunker.recursive_splitter.split_documents([section_doc])
                    
                    # Thêm sub_chunk_id và giữ source_file metadata
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        sub_chunk.metadata.update({
                            "sub_chunk_id": sub_idx,
                            "total_sub_chunks": len(sub_chunks),
                        })
                    file_chunks.extend(sub_chunks)
                else:
                    file_chunks.append(section_doc)
            
            print(f"   → {len(file_chunks)} chunks")
            all_chunks.extend(file_chunks)
        
        print(f"\n📊 TỔNG KẾT:")
        print(f"   - Tổng số file: {len(md_files)}")
        print(f"   - Tổng số chunks: {len(all_chunks)}")
        
        # Lưu vào Pinecone
        print(f"\n💾 Đang lưu vào Pinecone...")
        
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Import Pinecone components
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
        import time
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY_HERE")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Xóa index cũ nếu có
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name in existing_indexes:
            print(f"🗑️  Xóa index cũ: {index_name}")
            pc.delete_index(index_name)
            time.sleep(1)
        
        # Tạo index mới
        print(f"🔧 Tạo Pinecone index mới: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Google text-embedding-004
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        # Đợi index được tạo xong
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"✅ Index {index_name} đã sẵn sàng!")
        
        PineconeVectorStore.from_documents(
            documents=all_chunks,
            embedding=chunker.embeddings,
            index_name=index_name
        )
        
        # Lưu chunks vào pickle
        chunks_file_path = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        with open(chunks_file_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        print(f"💾 Đã lưu chunks vào {chunks_file_path}")
        
        print(f"\n✅ ĐÃ HOÀN THÀNH!")
        print(f"   📦 Index: {index_name}")
        print(f"   💾 Chunks lưu tại: {chunks_dir}")
        
        return index_name
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        traceback.print_exc()
        return None


def rag_chat():
    """Giai đoạn 4: RAG Chat"""
    print("\n" + "="*80)
    print("💬 GIAI ĐOẠN 4: RAG CHAT")
    print("="*80)
    
    try:
        interactive_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Đã thoát chat!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")


def run_full_pipeline():
    print("\n" + "="*80)
    print("🚀 CHẠY TOÀN BỘ PIPELINE")
    print("="*80)
    
    # Chuẩn bị data (wiki => html => cleaned html => normalized md)
    prepare_data()

    # Chunking và lưu vào vector DB
    chunking()
    
    # Bật rag chat trong terminal
    input("\n✅ Pipeline hoàn tất! Nhấn Enter để vào RAG Chat...")
    rag_chat()

def main():
    print_banner()
    
    while True:
        print_menu()
        choice = input("\n👉 Chọn giai đoạn (0-4): ").strip()
        
        if choice == '0':
            print("\n👋 Tạm biệt!")
            break
        elif choice == '1':
            prepare_data()
        elif choice == '2':
            chunking()
        elif choice == '3':
            rag_chat()
        elif choice == '4':
            run_full_pipeline()
        else:
            print("\n❌ Lựa chọn không hợp lệ!")
        
        if choice != '0':
            input("\n⏸️  Nhấn Enter để tiếp tục...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tạm biệt!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
