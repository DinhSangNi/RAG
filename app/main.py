from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings
from app.database.connection import engine, Base

# Tạo tables
Base.metadata.create_all(bind=engine)

# Tạo FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG Service API với PostgreSQL, pgvector và Redis Queue"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "RAG Service API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
