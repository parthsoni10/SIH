from app.modules.risk_scoring import predict_risk, build_feature_vector, self_test_model, FEATURE_ORDER

def test_feature_vector_structure():
    vector = build_feature_vector(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.1,
        face_match_score=0.9,
        blacklist_hit=False,
        document_type="Passport"
    )
    
    assert len(vector) == 12
    for feature in FEATURE_ORDER:
        assert feature in vector
    assert vector["document_type_Passport"] == 1.0
    assert vector["document_type_Visa"] == 0.0

def test_predict_risk_genuine():
    res = predict_risk(
        ocr_confidence=0.98,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Passport"
    )
    
    assert "risk_score" in res
    assert "prediction" in res
    assert res["prediction"] in ("genuine", "fraudulent")
    assert res["risk_score"] < 50

def test_predict_risk_fraudulent_blacklist():
    res = predict_risk(
        ocr_confidence=0.3,
        validation_pass_rate=0.2,
        id_checksum_valid=False,
        expiry_valid=False,
        tampering_score=0.8,
        face_match_score=0.2,
        blacklist_hit=True,
        document_type="Aadhaar"
    )
    
    assert res["prediction"] == "fraudulent"

def test_model_self_test():
    assert self_test_model() is True
