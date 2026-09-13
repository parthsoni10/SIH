from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from datetime import datetime
from app.db.database import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    document_type = Column(String(50), nullable=False, index=True)
    filename = Column(String(255), nullable=True)
    
    # Model & Risk outputs
    risk_score = Column(Integer, nullable=False, index=True)
    prediction = Column(String(20), nullable=False, index=True) # genuine / fraudulent
    probability = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)

    # Feature signals
    ocr_confidence = Column(Float, nullable=False)
    validation_pass_rate = Column(Float, nullable=False)
    id_checksum_valid = Column(Boolean, nullable=False)
    expiry_valid = Column(Boolean, nullable=False)
    tampering_score = Column(Float, nullable=False)
    face_match_score = Column(Float, nullable=True)
    face_match_missing = Column(Boolean, nullable=True)
    blacklist_hit = Column(Boolean, nullable=False)

    # Rich JSON details
    extracted_fields = Column(JSON, nullable=True)
    failed_rules = Column(JSON, nullable=True)
    tampering_signals = Column(JSON, nullable=True)
    feature_vector = Column(JSON, nullable=True)
    raw_text = Column(JSON, nullable=True)
    llm_validation = Column(JSON, nullable=True)
    preprocessing_warnings = Column(JSON, nullable=True)
    tampering_flags = Column(JSON, nullable=True)
    layout_score = Column(Float, nullable=True)
    layout_anomalies = Column(JSON, nullable=True)
