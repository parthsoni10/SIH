import pytest
from app.modules.decision_engine import evaluate_final_decision
from app.modules.forensic_fusion import fuse_forensic_evidence
from app.modules.face_match import verify_face_match
import numpy as np

def test_1_structurally_valid_uncertain_ai_returns_manual_review():
    """
    TEST 1: Structurally valid + uncertain AI
    Expected: MANUAL_REVIEW with exact reason codes
    """
    ai_analysis = {
        "probability": 0.45,
        "global_probability": 0.45,
        "patch_topk_probability": 0.52,
        "frequency_anomaly": 0.15,
        "noise_anomaly": 0.40,
        "strong_signal_count": 0,
        "corroborated": False,
        "status": "INCONCLUSIVE",
        "is_weights_loaded": True,
    }
    tampering_analysis = {"probability": 0.06}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
        blacklist_hit=False
    )

    assert res["status"] == "MANUAL_REVIEW"
    assert res["requires_manual_review"] is True
    assert "DOCUMENT_STRUCTURE_VALID" in res["reason_codes"]
    assert "AI_EVIDENCE_INCONCLUSIVE" in res["reason_codes"]
    assert "TAMPERING_EVIDENCE_LOW" in res["reason_codes"]
    assert "AUTHENTICITY_NOT_ESTABLISHED" in res["reason_codes"]


def test_2_high_ai_with_corroboration_returns_ai_generated():
    """
    TEST 2: High AI probability + 2 strong corroborating signals
    Expected: AI_GENERATED
    """
    ai_analysis = {
        "probability": 0.85,
        "global_probability": 0.82,
        "patch_topk_probability": 0.88,
        "frequency_anomaly": 0.65,
        "noise_anomaly": 0.70,
        "strong_signal_count": 3,
        "corroborated": True,
        "status": "LIKELY_AI_GENERATED",
        "reasons": ["AI_MODEL_STRONG", "PATCH_LEVEL_AI_CORROBORATION", "FREQUENCY_ANOMALY"],
        "is_weights_loaded": True,
    }
    tampering_analysis = {"probability": 0.10}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "AI_GENERATED"
    assert res["requires_manual_review"] is False


def test_3_high_tampering_low_ai_returns_altered():
    """
    TEST 3: High tampering + low AI
    Expected: ALTERED
    """
    ai_analysis = {
        "probability": 0.15,
        "strong_signal_count": 0,
        "corroborated": False,
        "is_weights_loaded": True,
    }
    tampering_analysis = {
        "probability": 0.82,
        "flags": ["ELA_HIGH_DISCREPANCY", "COPY_MOVE_MATCH"]
    }
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "ALTERED"
    assert res["requires_manual_review"] is False


def test_4_high_ai_and_high_tampering_returns_combined():
    """
    TEST 4: High AI + high tampering
    Expected: AI_GENERATED_AND_ALTERED
    """
    ai_analysis = {
        "probability": 0.88,
        "strong_signal_count": 2,
        "corroborated": True,
        "is_weights_loaded": True,
    }
    tampering_analysis = {"probability": 0.80}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "AI_GENERATED_AND_ALTERED"


def test_5_invalid_checksum_or_layout_returns_suspicious():
    """
    TEST 5: Invalid checksum/layout
    Expected: SUSPICIOUS
    """
    ai_analysis = {
        "probability": 0.10,
        "strong_signal_count": 0,
        "corroborated": False,
        "is_weights_loaded": True,
    }
    tampering_analysis = {"probability": 0.10}
    document_validity = {
        "score": 0.40,
        "structural_validity": False,
        "failed_rules": ["invalid_checksum", "layout_mismatch"]
    }
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "SUSPICIOUS"
    assert res["requires_manual_review"] is True


def test_6_very_poor_image_quality_returns_manual_review():
    """
    TEST 6: Very poor image quality
    Expected: MANUAL_REVIEW
    """
    ai_analysis = {"probability": 0.10, "is_weights_loaded": True}
    tampering_analysis = {"probability": 0.10}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {
        "score": 0.30,
        "quality_too_low_for_forensics": True,
        "warnings": ["EXTREME_BLUR"]
    }

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "MANUAL_REVIEW"
    assert "QUALITY_TOO_LOW_FOR_FORENSICS" in res["reason_codes"]


def test_7_missing_exif_does_not_imply_ai():
    """
    TEST 7: Missing EXIF
    Expected: Metadata anomaly triggers, but NOT automatically classified as AI_GENERATED without corroboration
    """
    ai_detector_res = {"ai_probability": 0.20, "global_probability": 0.20, "patch_topk_probability": 0.20}
    freq_res = {"frequency_anomaly_score": 0.10}
    noise_res = {"noise_anomaly_score": 0.15}
    metadata_res = {"no_camera_origin_evidence": True}  # Missing EXIF
    tampering_res = {"tampering_probability": 0.05}
    validity_res = {"validation_pass_rate": 1.0}

    fused = fuse_forensic_evidence(
        ai_detector_res=ai_detector_res,
        frequency_res=freq_res,
        noise_res=noise_res,
        metadata_res=metadata_res,
        tampering_res=tampering_res,
        validity_res=validity_res
    )

    # Missing EXIF alone is 1 weak signal, corroborated must be False
    assert fused["ai_analysis"]["corroborated"] is False
    # AI generation probability should remain below strong threshold
    assert fused["ai_analysis"]["probability"] < 0.60


def test_8_missing_live_capture_returns_null_face_score():
    """
    TEST 8: Missing live capture
    Expected: face score = None/null, available = False, status = NOT_PROVIDED
    """
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    face_res = verify_face_match(doc_image_bgr=dummy_img, live_capture_bytes=None)

    assert face_res["score"] is None
    assert face_res["available"] is False
    assert face_res["status"] == "NOT_PROVIDED"


def test_9_ai_model_unavailable_returns_manual_review():
    """
    TEST 9: Valid document structure but AI model unavailable / weights not loaded
    Expected: MANUAL_REVIEW / MODEL_NOT_READY (NOT GENUINE)
    """
    ai_analysis = {
        "probability": 0.0,
        "status": "MODEL_NOT_READY",
        "is_weights_loaded": False,
    }
    tampering_analysis = {"probability": 0.05}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "MANUAL_REVIEW"
    assert "MODEL_NOT_READY" in res["reason_codes"]


def test_10_gemini_authentic_with_inconclusive_local_returns_manual_review():
    """
    TEST 10: Gemini says "authentic" (e.g. score 0.95) while local evidence is inconclusive (AI prob 0.45)
    Expected: MANUAL_REVIEW (Gemini must not override local decision)
    """
    ai_analysis = {
        "probability": 0.45,
        "strong_signal_count": 0,
        "corroborated": False,
        "is_weights_loaded": True,
    }
    tampering_analysis = {"probability": 0.05}
    document_validity = {"score": 1.0, "structural_validity": True, "failed_rules": []}
    image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}

    # Even if Gemini says authentic, evaluate_final_decision only takes local evidence inputs
    res = evaluate_final_decision(
        ai_analysis=ai_analysis,
        tampering_analysis=tampering_analysis,
        document_validity=document_validity,
        image_quality=image_quality,
    )

    assert res["status"] == "MANUAL_REVIEW"
    assert res["status"] != "GENUINE"
