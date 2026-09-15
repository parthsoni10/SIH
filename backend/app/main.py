import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.database import engine, Base
from app.routers import documents, audit
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

            new_cols = [
                ("layout_score", "FLOAT"),
                ("layout_anomalies", "JSON"),
                ("face_match_missing", "BOOLEAN"),
                ("synthetic_generation_score", "FLOAT"),
                ("synthetic_reasons", "JSON"),
                ("ai_probability", "FLOAT"),
                ("ai_model_version", "VARCHAR(50)"),
                ("frequency_score", "FLOAT"),
                ("synthetic_noise_score", "FLOAT"),
                ("synthetic_confidence", "FLOAT"),
                ("strong_signal_count", "INTEGER"),
                ("synthetic_status", "VARCHAR(50)"),
                ("decision_reason_codes", "JSON"),
                ("synthetic_analysis", "JSON"),
                ("decision_status", "VARCHAR(50)"),
                ("decision_confidence", "FLOAT"),
                ("document_validity_score", "FLOAT"),
                ("ai_generation_probability", "FLOAT"),
                ("tampering_probability", "FLOAT"),
                ("frequency_anomaly", "FLOAT"),
                ("noise_anomaly", "FLOAT"),
                ("patch_ai_probability", "FLOAT"),
                ("corroborated", "BOOLEAN"),
                ("quality_score", "FLOAT"),
                ("gemini_summary", "TEXT"),
                ("reason_codes", "JSON"),
                ("model_versions", "JSON"),
                ("full_forensic_json", "JSON"),
            ]

            for col_name, col_type in new_cols:
                if col_name not in column_names:
                    conn.execute(text(f"ALTER TABLE audit_logs ADD COLUMN {col_name} {col_type};"))
            conn.commit()

    except Exception as e:
        logger.warning(f"Auto-migration notice: {str(e)}")

    logger.info("Pre-loading EfficientNet-B0 AI Detector & Forensic Engines...")
    try:
        from app.modules.ai_image_detector import AIDetectorService
        AIDetectorService.get_instance()
        logger.info("AI Detector Service loaded successfully at startup.")
    except Exception as e:
        logger.warning(f"AI Detector startup pre-loading warning: {str(e)}")

    logger.info("Pre-warming OCR engine & DeepFace models...")
    try:
        get_ocr_engine()
        warmup_deepface_model()
    except Exception as e:
        logger.warning(f"Model pre-warmup non-critical warning: {str(e)}")

    yield

    logger.info("Shutting down AI Screening Pipeline backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="AI-Based Fake Identity & Document Screening System — FastAPI Backend (V2 Target Architecture)",
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
        "version": "2.0.0",
        "endpoints": [
            "/api/documents/verify",
            "/api/audit",
            "/api/audit/{id}"
        ]
    }
