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
        import time
        
        chunker = HybridSectionChunker(chunk_size=1000, chunk_overlap=150, child_chunk_size=300)
        
        print(f"\n🔄 Bắt đầu chunking với chiến lược parent-child...")
        print(f"   👨 Parent threshold: {chunker.chunk_size} chars")
        print(f"   👶 Child chunk size: {chunker.child_chunk_size} chars")
        
        # ============================================================
        # BƯỚC 1: XÓA TOÀN BỘ DB CŨ (1 LẦN DUY NHẤT)
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🗑️  BƯỚC 1: XÓA TOÀN BỘ DATABASE CŨ")
        print(f"{'='*70}")
        
        # Xóa Pinecone index
        existing_indexes = [idx.name for idx in chunker.pc.list_indexes()]
        if index_name in existing_indexes:
            print(f"   🗑️  Đang xóa Pinecone index: {index_name}")
            chunker.pc.delete_index(index_name)
            print(f"   ✅ Đã xóa Pinecone index")
            time.sleep(2)  # Đợi Pinecone xóa xong
        else:
            print(f"   ℹ️  Không có index cũ cần xóa")
        
        # Xóa file pickle cũ
        chunks_file = os.path.join(chunks_dir, f"{index_name.replace('-', '_')}_chunks.pkl")
        if os.path.exists(chunks_file):
            os.remove(chunks_file)
            print(f"   ✅ Đã xóa file pickle: {chunks_file}")
        else:
            print(f"   ℹ️  Không có file pickle cũ cần xóa")
        
        print(f"\n✅ Đã xóa sạch database cũ!\n")
        
        # ============================================================
        # BƯỚC 2: CHUNKING TẤT CẢ FILE VÀ TẠO DB MỚI
        # ============================================================
        print(f"{'='*70}")
        print(f"🔪 BƯỚC 2: CHUNKING VÀ TẠO DATABASE MỚI")
        print(f"{'='*70}\n")
        
        # Xử lý TẤT CẢ file với reset=False (vì đã xóa sạch ở bước 1)
        for idx, md_file in enumerate(md_files, 1):
            md_file_path = os.path.join(md_dir, md_file)
            print(f"\n{'='*70}")
            print(f"📄 [{idx}/{len(md_files)}] {md_file}")
            print(f"{'='*70}")
            
            chunker.chunk_and_save_to_db(
                md_file_path=md_file_path,
                index_name=index_name,
                chunks_dir=chunks_dir,
                reset=False  # KHÔNG reset vì đã xóa sạch ở bước 1 rồi
            )
        
        print(f"\n{'='*80}")
        print(f"✅ ĐÃ HOÀN THÀNH TẤT CẢ {len(md_files)} FILE!")
        print(f"📦 Index: {index_name}")
        print(f"💾 Chunks lưu tại: {chunks_dir}")
        print(f"{'='*80}")
        
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
