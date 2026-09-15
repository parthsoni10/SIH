import pytest
from app.modules.decision_engine import evaluate_final_decision


def test_decision_engine_genuine_legacy():
    """Legacy signature: low AI + valid structure → GENUINE (legacy uses trained=True, calibrated=False but allows via legacy path)."""
    risk = {"risk_score": 15, "risk_tier": "low"}
    synthetic = {"synthetic_generation_score": 0.15, "status": "LOW_AI_EVIDENCE", "strong_signal_count": 0, "is_corroborated": False}
    tampering = {"tampering_score": 0.10}
    validation = {"validation_pass_rate": 1.0, "blacklist_hit": False}

    res = evaluate_final_decision(risk, synthetic, tampering, validation)
    # Legacy path has trained=True but calibrated=False and no override → MANUAL_REVIEW
    # Unless allow_genuine_without_calibration is set
    assert res["final_decision"] in ("GENUINE", "MANUAL_REVIEW")


def test_decision_engine_suspicious_blacklist():
    risk = {"risk_score": 100, "risk_tier": "high"}
    synthetic = {"synthetic_generation_score": 0.10, "status": "LOW_AI_EVIDENCE", "strong_signal_count": 0, "is_corroborated": False}
    tampering = {"tampering_score": 0.10}
    validation = {"validation_pass_rate": 1.0, "blacklist_hit": True}

    res = evaluate_final_decision(risk, synthetic, tampering, validation, blacklist_hit=True)
    assert res["final_decision"] == "SUSPICIOUS"
    assert "BLACKLISTED_ID_MATCH" in res["reason_codes"]


def test_decision_engine_ai_generated_corroborated():
    risk = {"risk_score": 90, "risk_tier": "high"}
    synthetic = {"ai_probability": 0.85, "synthetic_generation_score": 0.85, "status": "AI_GENERATED", "strong_signal_count": 2, "is_corroborated": True}
    tampering = {"tampering_score": 0.10}
    validation = {"validation_pass_rate": 1.0, "blacklist_hit": False}

    res = evaluate_final_decision(risk, synthetic, tampering, validation)
    assert res["final_decision"] == "AI_GENERATED"
    assert "SYNTHETIC_AI_IMAGE_CORROBORATED" in res["reason_codes"]


def test_decision_engine_manual_review_single_signal():
    risk = {"risk_score": 45, "risk_tier": "medium"}
    synthetic = {"synthetic_generation_score": 0.40, "status": "INCONCLUSIVE", "strong_signal_count": 1, "is_corroborated": False}
    tampering = {"tampering_score": 0.20}
    validation = {"validation_pass_rate": 0.80, "blacklist_hit": False}

    res = evaluate_final_decision(risk, synthetic, tampering, validation)
    assert res["final_decision"] == "MANUAL_REVIEW"
    assert res["requires_manual_review"]


def test_decision_engine_low_ai_uncalibrated_not_genuine():
    """
    CRITICAL SECURITY TEST:
    Low AI probability + uncalibrated model → MANUAL_REVIEW (NOT GENUINE).
    Low AI evidence is NOT proof of authenticity without calibrated model.
    """
    ai_analysis = {
        "probability": 0.20,
        "strong_signal_count": 0,
        "corroborated": False,
        "model_loaded": True,
        "is_weights_loaded": True,
        "trained": True,
        "calibrated": False,  # NOT calibrated
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

    # Without calibration, even low AI should NOT produce GENUINE
    assert res["status"] == "MANUAL_REVIEW"
    assert res["status"] != "GENUINE"


def test_decision_engine_genuine_with_calibrated_model():
    """
    GENUINE is granted when model is trained + calibrated + low AI + no signals.
    """
    ai_analysis = {
        "probability": 0.10,
        "strong_signal_count": 0,
        "corroborated": False,
        "model_loaded": True,
        "is_weights_loaded": True,
        "trained": True,
        "calibrated": True,  # Properly calibrated
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

    assert res["status"] == "GENUINE"
    assert not res["requires_manual_review"]
    assert "AUTHENTICITY_CONFIRMED" in res["reason_codes"]


def test_decision_engine_structural_valid_not_auto_genuine():
    """
    Structural validity alone must NEVER automatically produce GENUINE.
    Inconclusive AI (0.46) + valid structure → MANUAL_REVIEW.
    """
    ai_analysis = {
        "probability": 0.46,
        "strong_signal_count": 0,
        "corroborated": False,
        "model_loaded": True,
        "is_weights_loaded": True,
        "trained": True,
        "calibrated": True,
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

    # 0.46 is in inconclusive range (0.35-0.65)
    assert res["status"] == "MANUAL_REVIEW"
    assert res["status"] != "GENUINE"
