import os
from typing import Dict, Any, List, Optional
from app.config import settings

HAS_GENAI = False
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def generate_template_explanation(
    risk_score: int,
    prediction: str,
    failed_rules: List[str],
    tampering_score: float,
    face_match_score: Optional[float],
    blacklist_hit: bool,
    document_type: str
) -> str:
    """Fallback plain language generator when Gemini API is unavailable."""
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

def generate_officer_explanation(
    risk_score: int,
    prediction: str,
    failed_rules: List[str],
    tampering_signals: Dict[str, float],
    tampering_score: float,
    face_match_score: Optional[float],
    blacklist_hit: bool,
    document_type: str
) -> str:
    """
    Calls Gemini API (or template fallback) to produce a concise 1-sentence officer explanation.
    """
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    if not api_key or not HAS_GENAI:
        return generate_template_explanation(
            risk_score=risk_score,
            prediction=prediction,
            failed_rules=failed_rules,
            tampering_score=tampering_score,
            face_match_score=face_match_score,
            blacklist_hit=blacklist_hit,
            document_type=document_type
        )

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
Given border screening document verification results:
- Document Type: {document_type}
- Risk Score: {risk_score}/100 ({prediction})
- Failed Validation Rules: {failed_rules}
- Tampering Score: {tampering_score} (ELA: {tampering_signals.get('ela_score', 0)}, EXIF Metadata: {tampering_signals.get('metadata_score', 0)})
- Face Match Similarity: {face_match_score if face_match_score is not None else 'N/A'}
- Blacklist Hit: {blacklist_hit}

Write exactly ONE concise, professional sentence (under 28 words) for a border officer explaining why this document was flagged or passed.
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        if response and response.text:
            cleaned = response.text.strip().replace("\n", " ")
            return cleaned

    except Exception:
        pass

    # Fallback to local template generator
    return generate_template_explanation(
        risk_score=risk_score,
        prediction=prediction,
        failed_rules=failed_rules,
        tampering_score=tampering_score,
        face_match_score=face_match_score,
        blacklist_hit=blacklist_hit,
        document_type=document_type
    )
