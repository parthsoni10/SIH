from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class VerificationResponse(BaseModel):
    id: int
    created_at: datetime
    document_type: str
    filename: Optional[str] = None
    
    risk_score: int
    prediction: str
    probability: float
    explanation: str

    ocr_confidence: float
    validation_pass_rate: float
    id_checksum_valid: bool
    expiry_valid: bool
    tampering_score: float
    face_match_score: Optional[float] = None
    face_match_missing: Optional[bool] = None
    blacklist_hit: bool

    extracted_fields: Dict[str, Any]
    failed_rules: List[str]
    tampering_signals: Dict[str, float]
    feature_vector: Dict[str, float]
    raw_text: Optional[List[str]] = None
    llm_validation: Optional[Dict[str, Any]] = None
    preprocessing_warnings: Optional[List[str]] = None
    tampering_flags: Optional[List[str]] = None
    layout_score: Optional[float] = None
    layout_anomalies: Optional[List[str]] = None

    class Config:
        from_attributes = True

class AuditLogItem(BaseModel):
    id: int
    created_at: datetime
    document_type: str
    filename: Optional[str] = None
    risk_score: int
    prediction: str
    explanation: Optional[str] = None

    class Config:
        from_attributes = True

class PaginatedAuditLogResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[VerificationResponse]
