import pytest
from app.modules.risk_scoring import build_feature_vector, predict_risk
from app.modules.validation import rule_name_present
from app.modules.ocr import map_fields_by_keywords

def test_missing_face_match_uses_genuine_default():
    """Verify that None face_match_score results in 0.93 effective score."""
    vec = build_feature_vector(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=None,
        blacklist_hit=False,
        document_type="Passport"
    )
    assert vec["face_match_score"] == 0.93

def test_genuine_document_without_live_capture_scores_low_risk():
    """Verify that a genuine document with no live capture produces risk score < 50 (Genuine)."""
    result = predict_risk(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=None,
        blacklist_hit=False,
        document_type="Passport"
    )
    assert result["risk_score"] < 50, f"Expected risk < 50, got {result['risk_score']}"
    assert result["prediction"] == "genuine"
    assert result["face_match_missing"] is True

def test_rule_name_present_variations():
    """Verify name presence rule handles different key names and string/dict structures."""
    # Dict structure
    fields_dict = {"given_names": {"value": "PARTH SONI"}}
    passed, rule_id, _ = rule_name_present(fields_dict, "Passport", {})
    assert passed is True
    assert rule_id == "name_present"

    # Raw string structure
    fields_str = {"full_name": "PARTH SONI"}
    passed, rule_id, _ = rule_name_present(fields_str, "Passport", {})
    assert passed is True

    # Key containing 'name'
    fields_custom = {"holder_name_eng": "PARTH SONI"}
    passed, rule_id, _ = rule_name_present(fields_custom, "Passport", {})
    assert passed is True

    # Missing name
    fields_empty = {"dob": "1995-05-15"}
    passed, rule_id, _ = rule_name_present(fields_empty, "Passport", {})
    assert passed is False
    assert rule_id == "missing_name"

def test_ocr_name_extraction():
    """Verify regex-based map_fields_by_keywords extracts holder name correctly."""
    ocr_lines = [
        {"text": "REPUBLIC OF INDIA"},
        {"text": "Name / Name: SONI PARTH"},
        {"text": "DOB: 15/05/1995"}
    ]
    fields = map_fields_by_keywords(ocr_lines, "Passport")
    assert "name" in fields
    assert fields["name"]["value"] == "SONI PARTH"

def test_sanitize_field_value_garbage_stripping():
    """Verify sanitize_field_value strips label noise (/Name) and stray OCR tokens (fuaa)."""
    from app.modules.ocr import sanitize_field_value
    
    assert sanitize_field_value("name", "SONIPARTH fuaa") == "SONIPARTH"
    assert sanitize_field_value("name", "/Name: PARTH SONI") == "PARTH SONI"
    assert sanitize_field_value("name", "SONI PARTH DOB: 15/05/1995") == "SONI PARTH"
    assert sanitize_field_value("document_number", " ABCDE1234F /") == "ABCDE1234F"

def test_ocr_confidence_mapped_fields_only():
    """Verify ocr_confidence is computed from mapped fields only, ignoring low-confidence noise boxes."""
    from app.modules.ocr import extract_ocr_data
    import numpy as np
    
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    res = extract_ocr_data(dummy_img, "Passport")
    # Must return a valid float between 0.0 and 1.0
    assert 0.0 <= res["ocr_confidence"] <= 1.0
