from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.schemas.forensic import (
    ImageQualityResult,
    AIAnalysisResult,
    TamperingAnalysisResult,
    DocumentValidityResult,
    FaceVerificationResult,
    EvidenceFusionResult,
    GeminiAdvisoryResult
)
from app.schemas.decision import FinalDecisionPayload

class VerificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    document_type: str
    filename: Optional[str] = None

    # Canonical V2 Architecture Payload
    decision: FinalDecisionPayload
    document_validity: DocumentValidityResult
    ai_analysis: AIAnalysisResult
    tampering_analysis: TamperingAnalysisResult
    image_quality: ImageQualityResult
    face_verification: FaceVerificationResult
    fusion: EvidenceFusionResult
    gemini: GeminiAdvisoryResult
    model_versions: Dict[str, str]

    # Preserved Top-Level Fields for Audit & Frontend Compatibility
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
    synthetic_generation_score: Optional[float] = None
    synthetic_reasons: Optional[List[str]] = None
    layout_score: Optional[float] = None
    layout_anomalies: Optional[List[str]] = None

    # Legacy nested dictionaries
    risk: Optional[Dict[str, Any]] = None
    synthetic_analysis: Optional[Dict[str, Any]] = None
    ai_probability: Optional[float] = None
    frequency_score: Optional[float] = None
    strong_signal_count: Optional[int] = None
    synthetic_status: Optional[str] = None
    decision_reason_codes: Optional[List[str]] = None

class AuditLogItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    document_type: str
    filename: Optional[str] = None
    risk_score: int
    prediction: str
    decision_confidence: Optional[float] = None
    explanation: Optional[str] = None

class PaginatedAuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total: int
    page: int
    limit: int
    items: List[AuditLogItem]

