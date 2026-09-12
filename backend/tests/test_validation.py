from app.modules.validation import validate_document, parse_date_safely

def test_parse_date_safely():
    dt = parse_date_safely("1990-05-12")
    assert dt is not None
    assert dt.year == 1990
    assert dt.month == 5
    assert dt.day == 12

def test_validate_document_valid_passport():
    ocr_res = {
        "fields": {
            "document_number": {"value": "A8765432"},
            "dob": {"value": "1990-01-01"},
            "expiry_date": {"value": "2030-12-31"},
            "name": {"value": "JOHN DOE"}
        },
        "id_checksum_valid": True
    }
    
    res = validate_document(ocr_res, document_type="Passport")
    assert res["validation_pass_rate"] > 0.80
    assert res["expiry_valid"] is True
    assert res["blacklist_hit"] is False

def test_validate_document_expired_passport():
    ocr_res = {
        "fields": {
            "document_number": {"value": "A8765432"},
            "dob": {"value": "1990-01-01"},
            "expiry_date": {"value": "2020-01-01"},
            "name": {"value": "JOHN DOE"}
        },
        "id_checksum_valid": True
    }
    
    res = validate_document(ocr_res, document_type="Passport")
    assert res["expiry_valid"] is False
    assert "expired_document" in res["failed_rules"]

def test_validate_document_blacklisted_id():
    ocr_res = {
        "fields": {
            "document_number": {"value": "A1234567"},
            "dob": {"value": "1990-01-01"},
            "name": {"value": "BAD ACTOR"}
        },
        "id_checksum_valid": True
    }
    
    res = validate_document(ocr_res, document_type="Passport")
    assert res["blacklist_hit"] is True
    assert "blacklist_hit" in res["failed_rules"]
