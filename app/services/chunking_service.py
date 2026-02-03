from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document as LangChainDocument


class ChunkingService:
    """Service để chunking documents"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Markdown header splitter
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        
        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # Recursive character splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_markdown(self, text: str, source_file: str = "") -> Dict[str, Any]:
        """
        Chunk một markdown document với hierarchical structure
        
        Returns:
            Dict with:
            {
                'parent_chunks': List[Dict] - Sections lớn bị chia nhỏ,
                'child_chunks': List[Dict] - Tất cả chunks (có parent_id nếu thuộc parent)
            }
        """
        # Bước 1: Split theo headers
        section_docs = self.section_splitter.split_text(text)
        
        parent_chunks = []
        child_chunks = []
        child_chunk_index = 0
        
        # Bước 2: Xử lý từng section
        for section_idx, section_doc in enumerate(section_docs):
            # Lấy headers từ metadata
            headers = {
                'h1': section_doc.metadata.get('h1', ''),
                'h2': section_doc.metadata.get('h2', ''),
                'h3': section_doc.metadata.get('h3', ''),
            }
            
            # Nếu section quá lớn, tách tiếp
            if len(section_doc.page_content) > self.chunk_size:
                # Lưu parent chunk (section gốc)
                parent_chunk = {
                    'content': section_doc.page_content,
                    'chunk_index': section_idx,
                    'metadata': {
                        **headers,
                        'source': source_file
                    },
                    'section_id': section_idx
                }
                parent_chunks.append(parent_chunk)
                
                # Chia section thành sub-chunks
                sub_chunks = self.recursive_splitter.split_documents([section_doc])
                
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    child_chunks.append({
                        'content': sub_chunk.page_content,
                        'chunk_index': child_chunk_index,
                        'parent_section_id': section_idx,  # Đánh dấu thuộc parent nào
                        'metadata': {
                            **headers,
                            'section_id': section_idx,
                            'sub_chunk_id': sub_idx,
                            'source': source_file
                        }
                    })
                    child_chunk_index += 1
            else:
                # Section nhỏ, lưu trực tiếp vào child_chunks (không có parent)
                child_chunks.append({
                    'content': section_doc.page_content,
                    'chunk_index': child_chunk_index,
                    'parent_section_id': None,  # Không có parent
                    'metadata': {
                        **headers,
                        'section_id': section_idx,
                        'sub_chunk_id': None,
                        'source': source_file
                    }
                })
                child_chunk_index += 1
        
        return {
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks
        }


def get_chunking_service(chunk_size: int = 800, chunk_overlap: int = 150) -> ChunkingService:
    """Factory function để tạo ChunkingService"""
    return ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
