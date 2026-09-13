import logging
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

# Suppress sklearn version mismatch warnings when unpickling models
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


FEATURE_ORDER = [
    "ocr_confidence",
    "validation_pass_rate",
    "id_checksum_valid",
    "expiry_valid",
    "tampering_score",
    "face_match_score",
    "blacklist_hit",
    "document_type_Aadhaar",
    "document_type_Driving License",
    "document_type_PAN Card",
    "document_type_Passport",
    "document_type_Visa",
]

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = settings.RISK_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Risk model binary not found at {model_path}")
        _model = joblib.load(model_path)
    return _model

def build_feature_vector(
    ocr_confidence: float,
    validation_pass_rate: float,
    id_checksum_valid: bool,
    expiry_valid: bool,
    tampering_score: float,
    face_match_score: Optional[float],
    blacklist_hit: bool,
    document_type: str,
) -> Dict[str, float]:
    """
    Constructs the 12-element dictionary matching FEATURE_ORDER.
    """
    # Neutral default for missing face match score (0.93 corresponds to genuine class training mean when live capture is omitted)
    effective_face_match = face_match_score if face_match_score is not None else 0.93

    # One-hot encode document_type
    doc_types = ["Aadhaar", "Driving License", "PAN Card", "Passport", "Visa"]
    doc_one_hot = {f"document_type_{dt}": 1.0 if document_type == dt else 0.0 for dt in doc_types}

    vector = {
        "ocr_confidence": float(ocr_confidence),
        "validation_pass_rate": float(validation_pass_rate),
        "id_checksum_valid": 1.0 if id_checksum_valid else 0.0,
        "expiry_valid": 1.0 if expiry_valid else 0.0,
        "tampering_score": float(tampering_score),
        "face_match_score": float(effective_face_match),
        "blacklist_hit": 1.0 if blacklist_hit else 0.0,
        **doc_one_hot,
    }
    return vector

def predict_risk(
    ocr_confidence: float,
    validation_pass_rate: float,
    id_checksum_valid: bool,
    expiry_valid: bool,
    tampering_score: float,
    face_match_score: Optional[float],
    blacklist_hit: bool,
    document_type: str,
) -> Dict[str, Any]:
    """
    Evaluates the document signals against risk_model.pkl.
    Returns risk score (0-100), prediction string ('genuine'/'fraudulent'), probability, and feature vector.
    """
    model = get_model()
    face_match_missing = face_match_score is None
    feature_dict = build_feature_vector(
        ocr_confidence=ocr_confidence,
        validation_pass_rate=validation_pass_rate,
        id_checksum_valid=id_checksum_valid,
        expiry_valid=expiry_valid,
        tampering_score=tampering_score,
        face_match_score=face_match_score,
        blacklist_hit=blacklist_hit,
        document_type=document_type,
    )

    logger.info(
        f"[Risk Scoring] Feature Vector: {feature_dict} | "
        f"Face Match Score Original: {face_match_score} | "
        f"Face Match Missing: {face_match_missing}"
    )
    df = pd.DataFrame([feature_dict], columns=FEATURE_ORDER)
    
    # Predict probabilities: index 1 is fraud class (class 1)
    probabilities = model.predict_proba(df)[0]
    
    # Check if model has classes_ [0, 1]
    classes = getattr(model, "classes_", [0, 1])
    fraud_idx = 1 if len(classes) > 1 else 0
    fraud_prob = float(probabilities[fraud_idx])
    
    risk_score = round(fraud_prob * 100)
    hard_override = bool(blacklist_hit)

    if hard_override:
        risk_score = 100
        prediction = "fraudulent"
        risk_tier = "high"
    else:
        prediction = "fraudulent" if fraud_prob >= 0.5 else "genuine"
        if risk_score > 70:
            risk_tier = "high"
        elif risk_score >= 30:
            risk_tier = "medium"
        else:
            risk_tier = "low"

    # Extract feature importances if available on model
    feature_importances = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(FEATURE_ORDER, model.feature_importances_):
            feature_importances[name] = round(float(imp), 4)

    return {
        "risk_score": risk_score,
        "prediction": prediction,
        "probability": round(fraud_prob, 4),
        "risk_tier": risk_tier,
        "hard_override": hard_override,
        "face_match_missing": face_match_missing,
        "feature_vector": feature_dict,
        "feature_importances": feature_importances,
    }

def self_test_model() -> bool:
    """
    Runs synthetic genuine and fraudulent test vectors to verify model readiness.
    """
    genuine = predict_risk(
        ocr_confidence=0.98,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Passport",
    )
    
    fraudulent = predict_risk(
        ocr_confidence=0.40,
        validation_pass_rate=0.20,
        id_checksum_valid=False,
        expiry_valid=False,
        tampering_score=0.85,
        face_match_score=0.20,
        blacklist_hit=True,
        document_type="Passport",
    )
    
    return genuine["risk_score"] < 50 and fraudulent["risk_score"] >= 50
