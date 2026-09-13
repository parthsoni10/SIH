import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.database import engine, Base
from app.routers import documents, audit
from app.modules.risk_scoring import self_test_model
from app.modules.ocr import get_ocr_engine
from app.modules.face_match import warmup_deepface_model

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # App Startup
    logger.info("Initializing database schema tables...")
    Base.metadata.create_all(bind=engine)

    # Auto-migrate missing columns for existing SQLite databases
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("PRAGMA table_info(audit_logs);")).fetchall()
            column_names = [row[1] for row in result] if result else []
            if "layout_score" not in column_names:
                conn.execute(text("ALTER TABLE audit_logs ADD COLUMN layout_score FLOAT;"))
            if "layout_anomalies" not in column_names:
                conn.execute(text("ALTER TABLE audit_logs ADD COLUMN layout_anomalies JSON;"))
            if "face_match_missing" not in column_names:
                conn.execute(text("ALTER TABLE audit_logs ADD COLUMN face_match_missing BOOLEAN;"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Auto-migration notice: {str(e)}")

    logger.info("Running risk model self-test...")
    model_ok = self_test_model()
    if not model_ok:
        logger.warning("Risk model self-test returned unexpected values!")
    else:
        logger.info("Risk model self-test passed.")

    logger.info("Pre-warming OCR engine & DeepFace models...")
    try:
        get_ocr_engine()
        warmup_deepface_model()
    except Exception as e:
        logger.warning(f"Model pre-warmup non-critical warning: {str(e)}")

    yield

    # App Shutdown
    logger.info("Shutting down AI Screening Pipeline backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="AI-Based Fake Identity & Document Screening System — FastAPI Backend",
    lifespan=lifespan
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
@app.get("/api/health")
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
