import os
import io
import time
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schema import AuditLog
from app.schemas.document import VerificationResponse

from app.modules.preprocessing import preprocess_image
from app.modules.ocr import extract_ocr_data
from app.modules.validation import validate_document, rule_name_present
from app.modules.tampering import detect_tampering
from app.modules.face_match import verify_face_match
from app.modules.risk_scoring import predict_risk
from app.modules.explain import generate_template_explanation
from app.modules.layout_validator import validate_document_layout
from app.modules.llm_field_validator import run_unified_llm_analysis, merge_llm_with_regex

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_DOC_TYPES = {"Passport", "Visa", "Aadhaar", "PAN Card", "Driving License", "Other"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

@router.post("/verify", response_model=VerificationResponse)
async def verify_document(
    document_file: UploadFile = File(...),
    live_capture_file: Optional[UploadFile] = File(None),
    document_type: str = Form("Passport"),
    db: Session = Depends(get_db)
):
    """
    Executes full AI screening pipeline with concurrency and per-module failure isolation.
    Uses a SINGLE LLM call at the end for field validation + officer explanation.
    """
    start_time = time.time()

    # Pre-validation: document_type check
    if document_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported document_type: '{document_type}'. Must be one of {list(ALLOWED_DOC_TYPES)}"
        )

    # 1. Read document bytes and validate size
    doc_bytes = await document_file.read()
    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Empty document file uploaded.")
    
    if len(doc_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit of 10MB.")

    live_bytes = None
    if live_capture_file:
        live_bytes = await live_capture_file.read()

    # Module 1: Preprocessing (local)
    t0 = time.time()
    try:
        preprocessed = preprocess_image(doc_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    logger.info(f"[Pipeline] Module 1 - Preprocessing: {int((time.time()-t0)*1000)}ms")

    # Module 2: OCR Extraction (local only — no LLM)
    t0 = time.time()
    try:
        ocr_result = extract_ocr_data(preprocessed.image, document_type=document_type)
    except Exception as e:
        logger.error(f"OCR module error: {str(e)}")
        ocr_result = {
            "fields": {},
            "ocr_confidence": 0.0,
            "id_checksum_valid": False,
            "mrz": {"present": False},
            "raw_text": [],
            "llm_validation": {"llm_available": False, "mismatches_or_anomalies": [f"OCR module failure: {str(e)}"]}
        }
    logger.info(f"[Pipeline] Module 2 - OCR Extraction (local): {int((time.time()-t0)*1000)}ms")

    # Module 3: Layout Validation (local only — algorithmic, no LLM)
    t0 = time.time()
    try:
        layout_result = validate_document_layout(
            preprocessed.image,
            document_type=document_type,
            ocr_text_lines=ocr_result.get("raw_text", [])
        )
    except Exception as e:
        logger.error(f"Layout validator error: {str(e)}")
        layout_result = {
            "layout_valid": True,
            "layout_score": 0.80,
            "aspect_ratio": 1.5,
            "layout_anomalies": [f"Layout validator warning: {str(e)}"]
        }
    logger.info(f"[Pipeline] Module 3 - Layout Validation (local): {int((time.time()-t0)*1000)}ms")

    # Attach layout_result to ocr_result so validation rules can evaluate layout_score
    ocr_result["layout_result"] = layout_result

    # Module 4 & 5: Concurrent execution of Validation & Tampering Detection (local)
    t0 = time.time()
    try:
        validation_task = asyncio.to_thread(validate_document, ocr_result, document_type)
        tampering_task = asyncio.to_thread(detect_tampering, doc_bytes, preprocessed.image, preprocessed.exif_dict)
        
        validation_result, tampering_result = await asyncio.gather(validation_task, tampering_task)
    except Exception as e:
        logger.error(f"Validation or tampering execution error: {str(e)}")
        validation_result = validate_document(ocr_result, document_type)
        tampering_result = {
            "tampering_score": 0.20,
            "signals": {"ela_score": 0.0, "metadata_score": 0.2, "noise_inconsistency": 0.0},
            "flags": [f"Tampering check partial failure: {str(e)}"]
        }
    logger.info(f"[Pipeline] Module 4&5 - Validation+Tampering (local): {int((time.time()-t0)*1000)}ms")

    # Module 6: Face Verification (local)
    t0 = time.time()
    try:
        face_result = verify_face_match(preprocessed.image, live_bytes, document_type=document_type)
    except Exception as e:
        logger.error(f"Face verification module error: {str(e)}")
        face_result = {
            "face_match_score": None,
            "face_detected_in_doc": False,
            "face_detected_live": False,
            "distance": None
        }
    logger.info(f"[Pipeline] Module 6 - Face Verification (local): {int((time.time()-t0)*1000)}ms")

    # Module 7: Risk Scoring (local)
    t0 = time.time()
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
    logger.info(f"[Pipeline] Module 7 - Risk Scoring (local): {int((time.time()-t0)*1000)}ms")

    # ═══════════════════════════════════════════════════════════════
    # Module 8: ★ THE SINGLE LLM CALL ★
    # Sends raw OCR text + document type + full pipeline context
    # Returns: extracted fields, anomalies, and officer explanation
    # ═══════════════════════════════════════════════════════════════
    t0 = time.time()
    try:
        unified_llm_result = await asyncio.to_thread(
            run_unified_llm_analysis,
            raw_text_lines=ocr_result.get("raw_text", []),
            document_type=document_type,
            risk_score=risk_result["risk_score"],
            prediction=risk_result["prediction"],
            tampering_score=tampering_result["tampering_score"],
            face_match_score=face_result["face_match_score"],
            failed_rules=validation_result["failed_rules"],
            blacklist_hit=validation_result["blacklist_hit"],
            preprocessing_warnings=preprocessed.warnings,
            tampering_flags=tampering_result.get("flags", []),
        )
    except Exception as e:
        logger.error(f"Unified LLM analysis error: {str(e)}")
        unified_llm_result = {
            "llm_available": False,
            "extracted_fields": {},
            "mismatches_or_anomalies": [],
            "schema_matched": False,
            "raw_llm_response": "",
            "officer_explanation": ""
        }
    logger.info(f"[Pipeline] Module 8 - Unified LLM Analysis: {int((time.time()-t0)*1000)}ms")

    # Merge LLM-extracted fields back into OCR fields
    ocr_result["fields"] = merge_llm_with_regex(ocr_result["fields"], unified_llm_result)
    ocr_result["llm_validation"] = {
        "llm_available": unified_llm_result.get("llm_available", False),
        "extracted_fields": unified_llm_result.get("extracted_fields", {}),
        "mismatches_or_anomalies": unified_llm_result.get("mismatches_or_anomalies", []),
        "schema_matched": unified_llm_result.get("schema_matched", False),
        "raw_llm_response": unified_llm_result.get("raw_llm_response", "")
    }

    # Re-validate missing name if LLM supplied holder name after Module 4
    if "missing_name" in validation_result["failed_rules"]:
        name_ok, _, _ = rule_name_present(ocr_result["fields"], document_type, ocr_result)
        if name_ok:
            validation_result["failed_rules"].remove("missing_name")
            rules_total = len(validation_result.get("rule_details", {})) or 7
            rules_passed = rules_total - len(validation_result["failed_rules"])
            validation_result["validation_pass_rate"] = round(rules_passed / float(rules_total), 4)

            # Re-predict risk with updated pass rate
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
            logger.info(
                f"[Pipeline] Post-LLM re-validation: name found! "
                f"Updated pass_rate to {validation_result['validation_pass_rate']}, "
                f"risk_score to {risk_result['risk_score']}"
            )

    # Use LLM officer explanation, or fall back to local template
    explanation_text = unified_llm_result.get("officer_explanation", "")
    if not explanation_text:
        try:
            explanation_text = generate_template_explanation(
                risk_score=risk_result["risk_score"],
                prediction=risk_result["prediction"],
                failed_rules=validation_result["failed_rules"],
                tampering_score=tampering_result["tampering_score"],
                face_match_score=face_result["face_match_score"],
                blacklist_hit=validation_result["blacklist_hit"],
                document_type=document_type
            )
        except Exception:
            explanation_text = f"Flagged ({risk_result['risk_score']}% risk): Anomalies evaluated across document security signals."

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"[Pipeline] TOTAL processing time: {processing_time_ms}ms")

    # Module 9: Save to Audit DB
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
        face_match_missing=risk_result.get("face_match_missing", False),
        blacklist_hit=validation_result["blacklist_hit"],
        extracted_fields=ocr_result["fields"],
        failed_rules=validation_result["failed_rules"],
        tampering_signals=tampering_result["signals"],
        feature_vector=risk_result["feature_vector"],
        raw_text=ocr_result.get("raw_text", []),
        llm_validation=ocr_result.get("llm_validation", {}),
        preprocessing_warnings=preprocessed.warnings,
        tampering_flags=tampering_result.get("flags", []),
        layout_score=layout_result.get("layout_score", 0.85),
        layout_anomalies=layout_result.get("layout_anomalies", [])
    )

    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    return audit_entry
