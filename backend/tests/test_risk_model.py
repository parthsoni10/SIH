import pytest
from app.modules.risk_scoring import predict_risk, predict_risk_v2


def test_predict_risk_v1_baseline():
    res = predict_risk(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Passport"
    )
    assert res["risk_score"] < 50
    assert res["prediction"] in ("genuine", "fraudulent")


def test_predict_risk_v2():
    res = predict_risk_v2(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Passport",
        synthetic_generation_score=0.10,
        ai_probability=0.05,
        frequency_score=0.10,
        synthetic_noise_score=0.05,
    )
    assert res["model_version"] == "risk_model_v2"
    assert res["risk_score"] < 50
    assert "ai_probability" in res["feature_vector"]
    assert "frequency_score" in res["feature_vector"]
