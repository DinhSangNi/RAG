from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from pydantic import SecretStr
import numpy as np
from typing import List
import os


class EmbeddingService:
    """Service để tạo embeddings với Matryoshka truncation và normalization"""
    
    def __init__(self):
        # Sử dụng gemini-embedding-001 với output_dimensionality=768
        # Model này tạo 3072 dimensions nhưng truncate về 768 dimensions
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=SecretStr(settings.GEMINI_API_KEY)
        )
        # Set output dimensionality for Matryoshka truncation
        self.output_dimensionality = settings.DIMENSION_OF_MODEL
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector để đảm bảo chất lượng với smaller dimensions.
        Cần normalize khi dùng dimensions < 3072 (như 768, 1536).
        """
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            normalized = embedding_np / norm
            return normalized.tolist()
        return embedding
    
    def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một đoạn text với truncation và normalization
        """
        # Tạo embedding với output_dimensionality
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        result = client.models.embed_content(
            model=settings.EMBEDDING_MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            embedding = list(result.embeddings[0].values)
            # Normalize để đảm bảo chất lượng
            return self._normalize_embedding(embedding)
        
        raise ValueError("Failed to generate embedding")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều documents với truncation và normalization
        """
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Batch embed với output_dimensionality
        embeddings: List[List[float]] = []
        for text in texts:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
            )
            
            if result.embeddings and len(result.embeddings) > 0:
                embedding = list(result.embeddings[0].values)
                # Normalize từng embedding
                normalized = self._normalize_embedding(embedding)
                embeddings.append(normalized)
            else:
                raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")
        
        return embeddings


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton instance của EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
