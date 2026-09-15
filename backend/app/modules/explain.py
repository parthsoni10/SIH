import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def generate_template_explanation(
    risk_score: int,
    prediction: str,
    failed_rules: List[str],
    tampering_score: float,
    face_match_score: Optional[float],
    blacklist_hit: bool,
    document_type: str,
    synthetic_generation_score: float = 0.0,
    synthetic_reasons: Optional[List[str]] = None,
) -> str:
    """
    Local plain-language explanation generator.
    Used as fallback when the unified LLM call is unavailable (offline mode) or returns no explanation.
    """
    pred_upper = (prediction or "").upper()

    if pred_upper == "INCOMPLETE_SUBMISSION":
        return f"Incomplete Submission ({risk_score}% risk): Physical document not fully presented. Upload required secondary side (e.g. back side) to complete screening."

    reasons = []

    if blacklist_hit:
        reasons.append("Document ID matches known watch-list / blacklist record")

    if synthetic_generation_score >= 0.5:
        details = f" ({', '.join(synthetic_reasons)})" if synthetic_reasons else ""
        reasons.append(f"document shows strong signs of being AI-generated rather than a physical photo{details}")

    if "invalid_id_checksum" in failed_rules or "mrz_checksum_mismatch" in failed_rules:
        reasons.append("checksum verification failed on document ID/MRZ")

    if "expired_document" in failed_rules:
        reasons.append("document has expired")

    if tampering_score >= 0.35:
        reasons.append(f"digital image anomalies indicated potential tampering (score: {tampering_score:.2f})")

    if face_match_score is not None and face_match_score < 0.60:
        reasons.append(f"low facial similarity with live capture ({int(face_match_score * 100)}%)")

    if not reasons and failed_rules:
        reasons.append(f"failed validation rules: {', '.join(failed_rules)}")

    if pred_upper in ["AI_GENERATED", "ALTERED", "AI_GENERATED_AND_ALTERED", "SUSPICIOUS", "FRAUDULENT"] or risk_score >= 70:
        if reasons:
            return f"Flagged ({risk_score}% risk): {'; '.join(reasons)}."
        return f"Flagged ({risk_score}% risk): Elevated anomaly index across document signals."
    elif pred_upper == "MANUAL_REVIEW" or risk_score >= 30:
        if reasons:
            return f"Manual Review Required ({risk_score}% risk): {'; '.join(reasons)}."
        return f"Manual Review Required ({risk_score}% risk): Uncalibrated model signals or unread fields require officer verification."
    else:
        return f"Verified authentic {document_type} ({risk_score}% risk): All document security checks, checksums, and structural rules passed."
