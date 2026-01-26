from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from pydantic import SecretStr
import os


class EmbeddingService:
    """Service để tạo embeddings"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=SecretStr(settings.GEMINI_API_KEY)
        )
    
    def embed_text(self, text: str) -> list:
        """
        Tạo embedding cho một đoạn text
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list) -> list:
        """
        Tạo embeddings cho nhiều documents
        """
        return self.embeddings.embed_documents(texts)


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton instance của EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
