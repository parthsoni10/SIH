from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional

class ImageQualityResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    score: float
    blur_score: float
    resolution_ok: bool
    brightness_ok: bool
    contrast_score: float
    saturation_score: float
    jpeg_quality_estimate: float
    noise_level: float
    warnings: List[str]
    quality_warning: bool
    quality_too_low_for_forensics: bool

class AIAnalysisResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # Canonical AI probability — single authoritative field
    probability: Optional[float] = None
    ai_probability: Optional[float] = None
    global_probability: Optional[float] = None

    # Patch statistics
    patch_mean_probability: Optional[float] = None
    patch_median_probability: Optional[float] = None
    patch_topk_probability: Optional[float] = None

    # Supporting forensic signals
    frequency_anomaly: Optional[float] = None
    noise_anomaly: Optional[float] = None
    metadata_anomaly: Optional[float] = None

    # Corroboration
    strong_signal_count: int = 0
    strong_signals: List[str] = []
    corroborated: bool = False

    # Model state
    model_version: str = "convnext_base_ai_detector_v1"
    model_name: str = "ConvNeXt-Base"
    model_loaded: bool = False
    is_weights_loaded: bool = False  # Legacy alias for model_loaded
    trained: bool = False
    calibrated: bool = False
    calibration_method: Optional[str] = None
    evidence_quality: Optional[str] = None

    # Reason codes
    reasons: List[str] = []
    status: Optional[str] = "UNAVAILABLE"

class TamperingAnalysisResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    probability: Optional[float] = None
    tampering_probability: Optional[float] = None
    ela_score: Optional[float] = None
    noise_inconsistency: Optional[float] = None
    copy_move_score: Optional[float] = None
    splicing_score: Optional[float] = None
    metadata_score: Optional[float] = None
    confidence: Optional[float] = None
    flags: List[str] = []


class DocumentValidityResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    score: float
    ocr_confidence: float
    layout_score: float
    checksum_valid: bool
    expiry_valid: bool
    required_fields_present: bool = True
    structural_validity: bool = True
    failed_rules: List[str] = []

class FaceVerificationResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    available: bool
    score: Optional[float] = None
    status: str
    face_detected_in_doc: bool = False
    face_detected_live: bool = False

class EvidenceFusionResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ai_evidence: Optional[float] = None
    tampering_evidence: Optional[float] = None
    document_validity: Optional[float] = None
    overall_confidence: Optional[float] = None

class GeminiAdvisoryResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    available: bool
    role: str = "EXPLANATION_ONLY"
    summary: str
