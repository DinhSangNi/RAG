from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Tạo engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

# Tạo SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class cho models
Base = declarative_base()


def get_db():
    """Dependency để lấy database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
