"""
Application Configuration Settings
Uses Pydantic BaseSettings for environment variable management
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file
    """

    # Application Settings
    APP_NAME: str = "RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database Configuration
    DATABASE_URL: str = "postgresql://rag_user:rag_password@127.0.0.1:5433/rag_db"

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = "redis://localhost:6379/0"

    # Google AI Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # Embedding model using Matryoshka truncation to 768 dimensions
    # Model generates 3072d but truncated to 768d for storage efficiency
    EMBEDDING_MODEL_NAME: str = "models/gemini-embedding-001"
    DIMENSION_OF_MODEL: int = 768

    # Pinecone (Legacy - optional, for backward compatibility)
    PINECONE_API_KEY: Optional[str] = None

    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # File Storage Paths
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed_data"
    RAW_DIR: str = "./data/raw_data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Docker environment and adjust paths
        if os.path.exists('/app'):
            self.DATA_DIR = "/app/data"
            self.UPLOAD_DIR = "/app/data/uploads"
            self.PROCESSED_DIR = "/app/data/processed_data"
            self.RAW_DIR = "/app/data/raw_data"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
