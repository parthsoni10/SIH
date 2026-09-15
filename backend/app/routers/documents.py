import os
import io
import re
import time
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schema import AuditLog
from app.schemas.document import VerificationResponse

from app.modules.preprocessing import preprocess_image
from app.modules.image_quality import assess_image_quality
from app.modules.ocr import extract_ocr_data
from app.modules.validation import validate_document, rule_name_present
from app.modules.tampering_detector import detect_digital_tampering
from app.modules.ai_image_detector import predict_ai_image_probability
from app.modules.frequency_analysis import compute_frequency_analysis
from app.modules.noise_analysis import estimate_spatial_noise_anomalies
from app.modules.synthetic_detection import check_camera_metadata_plausibility
from app.modules.face_match import verify_face_match
from app.modules.forensic_fusion import fuse_forensic_evidence
from app.modules.decision_engine import evaluate_final_decision
from app.modules.provenance import analyse_provenance
from app.modules.qr_integrity import analyse_qr
from app.modules.texture_forensics import analyse_texture
from app.modules.forensic_fusion_v3 import fuse as fuse_v3
from app.modules.decision_engine_v3 import decide as decide_v3
from app.modules.document_spec import get_spec, SIDE_FRONT, SIDE_BACK
from app.modules.anchor_verifier import verify_anchors
from app.modules.alteration_detector import detect_alteration
from app.modules.field_resolver import resolve_structure
from app.modules.decision_engine_v31 import decide as decide_v31
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
    document_back_file: Optional[UploadFile] = File(None),
    live_capture_file: Optional[UploadFile] = File(None),
    document_type: str = Form("Passport"),
    db: Session = Depends(get_db)
):
    """
    Executes 20-Stage Target Architecture Screening Pipeline with concurrency and module failure isolation.
    Separates Document Validity, Image Quality, Digital Tampering, and AI Generation.
    Returns canonical V2 response payload.
    """
    start_time = time.time()

    # Stage 4: Document type check
    if document_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported document_type: '{document_type}'. Must be one of {list(ALLOWED_DOC_TYPES)}"
        )

    # Stage 1: Read files and validate size
    doc_bytes = await document_file.read()
    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Empty document file uploaded.")
    
    if len(doc_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit of 10MB.")

    back_bytes = None
    if document_back_file:
        back_bytes = await document_back_file.read()
        if back_bytes and len(back_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Back-side file size exceeds maximum allowed limit of 10MB.")
        if not back_bytes:
            back_bytes = None

    live_bytes = None
    if live_capture_file:
        live_bytes = await live_capture_file.read()

    # Stage 3: Image Normalization — Front side
    t0 = time.time()
    try:
        preprocessed = preprocess_image(doc_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    logger.info(f"[Pipeline] Stage 3 - Preprocessing (front): {int((time.time()-t0)*1000)}ms")

    # Stage 2: Image Quality Assessment & Gating
    t0 = time.time()
    try:
        image_quality = assess_image_quality(preprocessed.image)
    except Exception as e:
        logger.error(f"Image quality module error: {e}")
        image_quality = {
            "score": 0.50,
            "blur_score": 50.0,
            "resolution_ok": True,
            "brightness_ok": True,
            "contrast_score": 40.0,
            "saturation_score": 20.0,
            "jpeg_quality_estimate": 75.0,
            "noise_level": 5.0,
            "warnings": [f"Quality assessment error: {str(e)}"],
            "quality_warning": True,
            "quality_too_low_for_forensics": False,
        }
    logger.info(f"[Pipeline] Stage 2 - Image Quality: score={image_quality['score']}, blur={image_quality['blur_score']}")

    # Preprocessing — Back side (if provided)
    preprocessed_back = None
    if back_bytes:
        try:
            preprocessed_back = preprocess_image(back_bytes)
        except Exception as e:
            logger.warning(f"Back-side preprocessing failed: {str(e)}")

    # Stage 5 & 6: OCR & Document Field Extraction — Front side
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

    # OCR — Back side (merge if provided)
    if preprocessed_back is not None:
        try:
            ocr_result_back = extract_ocr_data(preprocessed_back.image, document_type=document_type)
            ocr_result["raw_text"] = ocr_result.get("raw_text", []) + ocr_result_back.get("raw_text", [])
            back_fields = ocr_result_back.get("fields", {})
            for f_key, f_val in back_fields.items():
                if f_key not in ocr_result["fields"] or not ocr_result["fields"][f_key].get("value"):
                    ocr_result["fields"][f_key] = f_val
            if ocr_result_back.get("id_checksum_valid"):
                ocr_result["id_checksum_valid"] = True
            if not ocr_result.get("mrz", {}).get("present") and ocr_result_back.get("mrz", {}).get("present"):
                ocr_result["mrz"] = ocr_result_back["mrz"]
        except Exception as e:
            logger.warning(f"Back-side OCR merge warning: {str(e)}")

    # Stage 8: Layout & Geometry Validation
    t0 = time.time()
    try:
        layout_result = validate_document_layout(
            preprocessed.image,
            document_type=document_type,
            ocr_text_lines=ocr_result.get("raw_text", [])
        )
    except Exception as e:
        logger.error(f"Layout validator error: {str(e)}")
        layout_result = {"layout_valid": True, "layout_score": 0.80, "aspect_ratio": 1.5, "layout_anomalies": []}
    ocr_result["layout_result"] = layout_result

    # Stage 7 & 9: Document Structural Validity
    t0 = time.time()
    validation_res = validate_document(ocr_result, document_type)
    document_validity = validation_res["document_validity"]

    # Concurrent Execution of Independent Forensic Branches:
    # - Stage 10: Digital Tampering Forensics
    # - Stage 11 & 14: AI Image Detection (Global + Multi-scale Patches)
    # - Stage 12: Frequency Domain Forensics
    # - Stage 13: Spatial Noise Forensics
    # - Stage 15: Face Verification
    t0 = time.time()
    metadata_res = check_camera_metadata_plausibility(preprocessed.exif_dict)

    async def run_tampering():
        return detect_digital_tampering(doc_bytes, preprocessed.image, preprocessed.exif_dict)

    async def run_ai_detector():
        return predict_ai_image_probability(preprocessed.image)

    async def run_frequency():
        return compute_frequency_analysis(preprocessed.image)

    async def run_noise():
        return estimate_spatial_noise_anomalies(preprocessed.image)

    async def run_face():
        return verify_face_match(preprocessed.image, live_bytes, document_type=document_type)

    forensic_results = await asyncio.gather(
        asyncio.to_thread(detect_digital_tampering, doc_bytes, preprocessed.image, preprocessed.exif_dict, document_type),
        asyncio.to_thread(predict_ai_image_probability, preprocessed.image),
        asyncio.to_thread(compute_frequency_analysis, preprocessed.image),
        asyncio.to_thread(estimate_spatial_noise_anomalies, preprocessed.image),
        asyncio.to_thread(verify_face_match, preprocessed.image, live_bytes, document_type=document_type),
        return_exceptions=True
    )

    tampering_analysis = forensic_results[0] if not isinstance(forensic_results[0], Exception) else {
        "available": False, "status": "ERROR", "probability": 0.0, "tampering_probability": 0.0,
        "ela_score": 0.0, "noise_inconsistency": 0.0, "copy_move_score": 0.0, "splicing_score": 0.0,
        "metadata_score": 0.0, "confidence": 0.0, "flags": ["TAMPERING_MODULE_ERROR"]
    }
    if "probability" not in tampering_analysis:
        tampering_analysis["probability"] = tampering_analysis.get("tampering_probability")

    ai_detector_res = forensic_results[1] if not isinstance(forensic_results[1], Exception) else {
        "status": "MODEL_ERROR", "is_weights_loaded": False, "ai_probability": None, "global_probability": None,
        "genuine_probability": None, "altered_probability": None, "patch_mean_probability": None,
        "patch_median_probability": None, "patch_topk_probability": None, "confidence": None,
        "model_version": "ai_detector_v2", "reasons": ["MODEL_INFERENCE_ERROR"]
    }
    frequency_res = forensic_results[2] if not isinstance(forensic_results[2], Exception) else {
        "available": False, "status": "ERROR", "frequency_anomaly_score": None, "frequency_score": None
    }
    noise_res = forensic_results[3] if not isinstance(forensic_results[3], Exception) else {
        "available": False, "status": "ERROR", "noise_anomaly_score": None, "suspiciously_smooth": None
    }
    face_verification = forensic_results[4] if not isinstance(forensic_results[4], Exception) else {
        "available": False, "score": None, "status": "ERROR", "face_detected_in_doc": False, "face_detected_live": False
    }

    logger.info(f"[Pipeline] Stage 10-15 - Concurrent Forensics: {int((time.time()-t0)*1000)}ms")

    # Stage 3b & 16: V3.3 Signal Analysis & Forensic Fusion
    doc_spec = get_spec(document_type)

    provenance_res = analyse_provenance(doc_bytes)
    qr_res = analyse_qr(preprocessed.image, doc_spec, ocr_result.get("fields", {}))
    texture_res = analyse_texture(preprocessed.image)

    anchor_res = verify_anchors(preprocessed.image, doc_spec, side=SIDE_FRONT, warped=False)
    alteration_res = detect_alteration(preprocessed.image, anchor_res)

    sides_present = {SIDE_FRONT, SIDE_BACK} if back_bytes else {SIDE_FRONT}

    # Extract MRZ lines from ocr_result or raw_text if present
    mrz_lines = []
    if isinstance(ocr_result.get("mrz"), dict) and ocr_result["mrz"].get("lines"):
        mrz_lines = ocr_result["mrz"]["lines"]
    elif isinstance(ocr_result.get("raw_text"), list):
        mrz_lines = [line for line in ocr_result["raw_text"] if isinstance(line, str) and re.match(r"^[A-Z0-9<]{30,44}$", line.strip())]

    # Extract checksum_input (document_number)
    doc_num_val = ""
    f_doc = ocr_result.get("fields", {}).get("document_number")
    if isinstance(f_doc, dict):
        doc_num_val = str(f_doc.get("value", "") or "")
    elif isinstance(f_doc, str):
        doc_num_val = f_doc

    ocr_raw = ocr_result.get("raw_text", [])
    ocr_lines = []
    if ocr_raw and isinstance(ocr_raw, list) and len(ocr_raw) > 0 and isinstance(ocr_raw[0], dict):
        ocr_lines = ocr_raw
    elif isinstance(ocr_raw, list):
        for idx, text_str in enumerate(ocr_raw):
            ocr_lines.append({
                "text": str(text_str),
                "conf": 0.85,
                "x0": 0.1, "y0": 0.1 + (idx * 0.05),
                "x1": 0.9, "y1": 0.15 + (idx * 0.05),
            })

    struct_res = resolve_structure(
        spec=doc_spec,
        lines=ocr_lines,
        sides_present=sides_present,
        extra_fields=ocr_result.get("fields", {}),
        checksum_input=doc_num_val,
        mrz_lines=mrz_lines,
        checksum_valid=ocr_result.get("id_checksum_valid") if ocr_result.get("id_checksum_valid") else None,
        layout_score=layout_result.get("layout_score", 0.85)
    )

    fusion_res_obj = fuse_v3(
        provenance=provenance_res,
        qr=qr_res,
        texture=texture_res,
        learned=ai_detector_res,
        tampering=tampering_analysis,
    )

    fusion_result = fuse_forensic_evidence(
        ai_detector_res=ai_detector_res,
        frequency_res=frequency_res,
        noise_res=noise_res,
        metadata_res=metadata_res,
        tampering_res=tampering_analysis,
        validity_res=validation_res,
        provenance_res=provenance_res,
        qr_res=qr_res,
        texture_res=texture_res,
    )

    ai_analysis = fusion_result["ai_analysis"]
    fusion = fusion_result["fusion"]

    # Stage 17: Calibrated Decision Engine V3.1
    decision_obj = decide_v31(
        fusion=fusion_res_obj,
        structure=struct_res,
        alteration=alteration_res,
        image_quality=image_quality,
        blacklist_hit=validation_res["blacklist_hit"],
    )
    decision = decision_obj.to_dict()
    decision["final_decision"] = decision_obj.status
    decision["decision_confidence"] = decision_obj.confidence

    status_to_risk = {
        "GENUINE": 10,
        "INCOMPLETE_SUBMISSION": 35,
        "MANUAL_REVIEW": 60,
        "SUSPICIOUS": 75,
        "ALTERED": 90,
        "AI_GENERATED": 95,
        "AI_GENERATED_AND_ALTERED": 100,
    }
    risk_score_int = status_to_risk.get(decision["status"], 60)

    # Stage 18: Single Unified Gemini AI Analysis Call (Executes At The End)
    t0 = time.time()
    try:
        unified_llm_result = await asyncio.to_thread(
            run_unified_llm_analysis,
            raw_text_lines=ocr_result.get("raw_text", []),
            document_type=document_type,
            risk_score=risk_score_int,
            prediction=decision["status"],
            tampering_score=tampering_analysis.get("probability") or 0.0,
            face_match_score=face_verification.get("score"),
            failed_rules=validation_res["failed_rules"],
            blacklist_hit=validation_res["blacklist_hit"],
            preprocessing_warnings=preprocessed.warnings,
            tampering_flags=tampering_analysis.get("flags", []),
            synthetic_generation_score=ai_detector_res.get("ai_probability") if ai_detector_res.get("ai_probability") is not None else 0.0,
            synthetic_reasons=ai_detector_res.get("reasons", []),
            document_image_bgr=preprocessed.image,
        )
    except Exception as e:
        logger.error(f"Unified LLM analysis error: {str(e)}")
        unified_llm_result = {
            "llm_available": False,
            "extracted_fields": {},
            "mismatches_or_anomalies": [],
            "schema_matched": False,
            "raw_llm_response": "",
            "officer_explanation": "",
            "visual_authenticity_score": None,
            "visual_evidence": [],
        }
    logger.info(f"[Pipeline] Stage 18 - Single Unified Gemini Analysis: {int((time.time()-t0)*1000)}ms")

    # Merge LLM-extracted fields back into OCR fields
    ocr_result["fields"] = merge_llm_with_regex(ocr_result["fields"], unified_llm_result)

    # Re-validate missing name if LLM supplied holder name
    if "missing_name" in validation_res["failed_rules"]:
        name_ok, _, _ = rule_name_present(ocr_result["fields"], document_type, ocr_result)
        if name_ok:
            validation_res["failed_rules"].remove("missing_name")
            rules_total = len(validation_res.get("rule_details", {})) or 7
            rules_passed = rules_total - len(validation_res["failed_rules"])
            validation_res["validation_pass_rate"] = round(rules_passed / float(rules_total), 4)
            document_validity["score"] = validation_res["validation_pass_rate"]
            document_validity["failed_rules"] = validation_res["failed_rules"]

    # Explanations
    explanation_text = unified_llm_result.get("officer_explanation", "")
    if not explanation_text:
        try:
            explanation_text = generate_template_explanation(
                risk_score=risk_score_int,
                prediction=decision["status"],
                failed_rules=validation_res["failed_rules"],
                tampering_score=tampering_analysis.get("probability") or 0.0,
                face_match_score=face_verification.get("score"),
                blacklist_hit=validation_res["blacklist_hit"],
                document_type=document_type,
                synthetic_generation_score=ai_analysis.get("probability") or 0.0,
                synthetic_reasons=ai_analysis.get("reasons", []),
            )
        except Exception:
            explanation_text = f"Evaluated verdict {decision['status']}: Screening complete across security modules."

    gemini = {
        "available": unified_llm_result.get("llm_available", False),
        "role": "EXPLANATION_ONLY",
        "summary": explanation_text,
    }

    model_versions = {
        "ai_detector": ai_analysis.get("model_version", "convnext_base_ai_detector_v1"),
        "tampering_detector": "tampering_v2",
        "fusion": "fusion_v3",
        "decision_engine": "decision_engine_v3",
    }

    # Build combined filename showing both sides if applicable
    combined_filename = document_file.filename or "document"
    if back_bytes and document_back_file:
        back_name = document_back_file.filename or "back"
        combined_filename = f"{combined_filename} + {back_name}"

    # Feature vector for technical inspector UI modal
    feature_vector = {
        "ocr_confidence": ocr_result["ocr_confidence"],
        "validation_pass_rate": validation_res["validation_pass_rate"],
        "id_checksum_valid": 1.0 if ocr_result["id_checksum_valid"] else 0.0,
        "expiry_valid": 1.0 if validation_res["expiry_valid"] else 0.0,
        "tampering_probability": tampering_analysis.get("probability") or 0.0,
        "ai_generation_probability": ai_analysis.get("probability") if ai_analysis.get("probability") is not None else 0.0,
        "frequency_anomaly": ai_analysis.get("frequency_anomaly") or 0.0,
        "noise_anomaly": ai_analysis.get("noise_anomaly") or 0.0,
        "image_quality_score": image_quality.get("score") or 0.85,
        "blacklist_hit": 1.0 if validation_res["blacklist_hit"] else 0.0,
    }

    # Stage 19: Save to Audit DB
    status_to_risk = {
        "GENUINE": 10,
        "INCOMPLETE_SUBMISSION": 35,
        "MANUAL_REVIEW": 60,
        "SUSPICIOUS": 75,
        "ALTERED": 90,
        "AI_GENERATED": 95,
        "AI_GENERATED_AND_ALTERED": 100,
    }
    risk_score_int = status_to_risk.get(decision["status"], 60)
    audit_entry = AuditLog(
        document_type=document_type,
        filename=combined_filename,
        risk_score=risk_score_int,
        prediction=decision["status"],
        probability=ai_analysis["probability"] if ai_analysis["probability"] is not None else 0.0,
        explanation=explanation_text,
        ocr_confidence=ocr_result["ocr_confidence"],
        validation_pass_rate=validation_res["validation_pass_rate"],
        id_checksum_valid=ocr_result["id_checksum_valid"],
        expiry_valid=validation_res["expiry_valid"],
        tampering_score=float(tampering_analysis.get("tampering_probability") if tampering_analysis.get("tampering_probability") is not None else (tampering_analysis.get("probability") or 0.0)),
        face_match_score=face_verification.get("score"),
        face_match_missing=not face_verification.get("available", False),
        blacklist_hit=validation_res["blacklist_hit"],
        extracted_fields=ocr_result["fields"],
        failed_rules=validation_res["failed_rules"],
        tampering_signals={
            "ela_score": tampering_analysis.get("ela_score"),
            "noise_inconsistency": tampering_analysis.get("noise_inconsistency"),
            "copy_move_score": tampering_analysis.get("copy_move_score"),
            "splicing_score": tampering_analysis.get("splicing_score"),
            "metadata_score": tampering_analysis.get("metadata_score"),
        },
        feature_vector=feature_vector,
        raw_text=ocr_result.get("raw_text", []),
        llm_validation=ocr_result.get("llm_validation", {}),
        preprocessing_warnings=preprocessed.warnings + image_quality["warnings"],
        tampering_flags=tampering_analysis.get("flags", []),
        synthetic_generation_score=ai_analysis["probability"] if ai_analysis["probability"] is not None else 0.0,
        synthetic_reasons=ai_analysis["reasons"],
        layout_score=layout_result.get("layout_score", 0.85),
        layout_anomalies=layout_result.get("layout_anomalies", []),
        ai_probability=ai_analysis["probability"] if ai_analysis["probability"] is not None else 0.0,
        ai_model_version=ai_analysis["model_version"],
        frequency_score=ai_analysis["frequency_anomaly"],
        synthetic_noise_score=ai_analysis["noise_anomaly"],
        synthetic_confidence=decision.get("confidence"),
        strong_signal_count=ai_analysis["strong_signal_count"],
        synthetic_status=decision["status"],
        decision_reason_codes=decision["reason_codes"],
        synthetic_analysis=ai_analysis,
        decision_status=decision["status"],
        decision_confidence=decision["confidence"],
        document_validity_score=document_validity["score"],
        ai_generation_probability=ai_analysis["probability"],
        tampering_probability=tampering_analysis.get("tampering_probability") if tampering_analysis.get("tampering_probability") is not None else (tampering_analysis.get("probability") or 0.0),
        frequency_anomaly=ai_analysis["frequency_anomaly"],
        noise_anomaly=ai_analysis["noise_anomaly"],
        patch_ai_probability=ai_analysis["patch_topk_probability"],
        corroborated=ai_analysis["corroborated"],
        quality_score=image_quality["score"],
        gemini_summary=explanation_text,
        reason_codes=decision["reason_codes"],
        model_versions=model_versions,
        full_forensic_json={
            "decision": decision,
            "document_validity": document_validity,
            "ai_analysis": ai_analysis,
            "tampering_analysis": tampering_analysis,
            "image_quality": image_quality,
            "face_verification": face_verification,
            "fusion": fusion,
            "gemini": gemini,
            "model_versions": model_versions,
        }
    )

    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    # Attach Pydantic V2 response payload attributes directly on ORM object
    setattr(audit_entry, "decision", decision)
    setattr(audit_entry, "document_validity", document_validity)
    setattr(audit_entry, "ai_analysis", ai_analysis)
    setattr(audit_entry, "tampering_analysis", tampering_analysis)
    setattr(audit_entry, "image_quality", image_quality)
    setattr(audit_entry, "face_verification", face_verification)
    setattr(audit_entry, "fusion", fusion)
    setattr(audit_entry, "gemini", gemini)
    setattr(audit_entry, "model_versions", model_versions)

    # Attach legacy backwards compatibility properties
    setattr(audit_entry, "risk", {
        "score": risk_score_int,
        "probability": ai_analysis["probability"] if ai_analysis["probability"] is not None else 0.0,
        "tier": "high" if risk_score_int >= 70 else ("medium" if risk_score_int >= 30 else "low"),
    })
    setattr(audit_entry, "synthetic_analysis", ai_analysis)
    setattr(audit_entry, "ai_probability", ai_analysis["probability"] if ai_analysis["probability"] is not None else 0.0)
    setattr(audit_entry, "frequency_score", ai_analysis["frequency_anomaly"])
    setattr(audit_entry, "strong_signal_count", ai_analysis["strong_signal_count"])
    setattr(audit_entry, "synthetic_status", decision["status"])
    setattr(audit_entry, "decision_reason_codes", decision["reason_codes"])

    return audit_entry
