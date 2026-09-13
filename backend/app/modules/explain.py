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
    document_type: str
) -> str:
    """
    Local plain-language explanation generator.
    Used as fallback when the unified LLM call is unavailable or returns no explanation.
    """
    reasons = []

    if blacklist_hit:
        reasons.append("Document ID matches known watch-list / blacklist record")

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

    if prediction == "fraudulent" or risk_score >= 50:
        if reasons:
            return f"Flagged ({risk_score}% risk): {'; '.join(reasons)}."
        return f"Flagged ({risk_score}% risk): Elevated anomaly index across document signals."
    else:
        return f"Verified ({risk_score}% risk): All document security checks, checksums, and structural rules passed."
