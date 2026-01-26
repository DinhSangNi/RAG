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
    
    def chunk_markdown(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Chunk một markdown document
        
        Returns:
            List of dicts với format:
            {
                'content': str,
                'metadata': {
                    'h1': str,
                    'h2': str,
                    'h3': str,
                    'section_id': int,
                    'sub_chunk_id': int,
                    'source': str
                }
            }
        """
        # Bước 1: Split theo headers
        section_docs = self.section_splitter.split_text(text)
        
        chunks = []
        chunk_index = 0
        
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
                sub_chunks = self.recursive_splitter.split_documents([section_doc])
                
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'content': sub_chunk.page_content,
                        'chunk_index': chunk_index,
                        'metadata': {
                            **headers,
                            'section_id': section_idx,
                            'sub_chunk_id': sub_idx,
                            'source': source_file
                        }
                    })
                    chunk_index += 1
            else:
                # Section nhỏ, giữ nguyên
                chunks.append({
                    'content': section_doc.page_content,
                    'chunk_index': chunk_index,
                    'metadata': {
                        **headers,
                        'section_id': section_idx,
                        'sub_chunk_id': None,
                        'source': source_file
                    }
                })
                chunk_index += 1
        
        return chunks


def get_chunking_service(chunk_size: int = 800, chunk_overlap: int = 150) -> ChunkingService:
    """Factory function để tạo ChunkingService"""
    return ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
