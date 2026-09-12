import os
import io
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schema import AuditLog
from app.schemas.document import VerificationResponse

from app.modules.preprocessing import preprocess_image
from app.modules.ocr import extract_ocr_data
from app.modules.validation import validate_document
from app.modules.tampering import detect_tampering
from app.modules.face_match import verify_face_match
from app.modules.risk_scoring import predict_risk
from app.modules.explain import generate_officer_explanation

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/verify", response_model=VerificationResponse)
async def verify_document(
    document_file: UploadFile = File(...),
    live_capture_file: Optional[UploadFile] = File(None),
    document_type: str = Form("Passport"),
    db: Session = Depends(get_db)
):
    """
    Executes full synchronous AI screening pipeline per document upload.
    """
    # 1. Read document bytes
    doc_bytes = await document_file.read()
    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Empty document file uploaded.")

    live_bytes = None
    if live_capture_file:
        live_bytes = await live_capture_file.read()

    # Module 1: Preprocessing
    try:
        preprocessed = preprocess_image(doc_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    # Module 2: OCR Extraction
    ocr_result = extract_ocr_data(preprocessed.image, document_type=document_type)

    # Module 3 & 4: Validation & Tampering (run in parallel/sequence)
    validation_result = validate_document(ocr_result, document_type=document_type)
    tampering_result = detect_tampering(doc_bytes, preprocessed.image, preprocessed.exif_dict)

    # Module 5: Face Verification
    face_result = verify_face_match(preprocessed.image, live_bytes)

    # Module 6: Risk Scoring
    risk_result = predict_risk(
        ocr_confidence=ocr_result["ocr_confidence"],
        validation_pass_rate=validation_result["validation_pass_rate"],
        id_checksum_valid=ocr_result["id_checksum_valid"],
        expiry_valid=validation_result["expiry_valid"],
        tampering_score=tampering_result["tampering_score"],
        face_match_score=face_result["face_match_score"],
        blacklist_hit=validation_result["blacklist_hit"],
        document_type=document_type
    )

    # Module 7: Gemini Explanation Layer
    explanation_text = generate_officer_explanation(
        risk_score=risk_result["risk_score"],
        prediction=risk_result["prediction"],
        failed_rules=validation_result["failed_rules"],
        tampering_signals=tampering_result["signals"],
        tampering_score=tampering_result["tampering_score"],
        face_match_score=face_result["face_match_score"],
        blacklist_hit=validation_result["blacklist_hit"],
        document_type=document_type
    )

    # Module 8: Save to Audit DB
    audit_entry = AuditLog(
        document_type=document_type,
        filename=document_file.filename,
        risk_score=risk_result["risk_score"],
        prediction=risk_result["prediction"],
        probability=risk_result["probability"],
        explanation=explanation_text,
        ocr_confidence=ocr_result["ocr_confidence"],
        validation_pass_rate=validation_result["validation_pass_rate"],
        id_checksum_valid=ocr_result["id_checksum_valid"],
        expiry_valid=validation_result["expiry_valid"],
        tampering_score=tampering_result["tampering_score"],
        face_match_score=face_result["face_match_score"],
        blacklist_hit=validation_result["blacklist_hit"],
        extracted_fields=ocr_result["fields"],
        failed_rules=validation_result["failed_rules"],
        tampering_signals=tampering_result["signals"],
        feature_vector=risk_result["feature_vector"]
    )

    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    return audit_entry
