from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.database import engine, Base
from app.routers import documents, audit

# Initialize database schema tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="AI-Based Fake Identity & Document Screening System — FastAPI Backend"
)

# CORS middleware for React SPA frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(documents.router)
app.include_router(audit.router)

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "version": "1.0.0",
        "endpoints": [
            "/api/documents/verify",
            "/api/audit",
            "/api/audit/{id}"
        ]
    }
