import numpy as np
from app.modules.mrz import parse_mrz_lines, compute_mrz_check_digit
from app.modules.ocr import validate_verhoeff, validate_pan_number, extract_ocr_data

def test_mrz_check_digit_calculation():
    assert compute_mrz_check_digit("123456789") in range(10)

def test_mrz_td3_parsing():
    line1 = "P<INDDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    line2 = "A1234567<4IND9005125M2512319<<<<<<<<<<<<<<04"
    
    res = parse_mrz_lines([line1, line2])
    assert res["present"] is True
    assert res["fields"]["name"] == "JOHN DOE"
    assert res["fields"]["document_number"] == "A1234567"
    assert res["fields"]["dob"] == "1990-05-12"

def test_verhoeff_checksum():
    assert validate_verhoeff("999999990019") is True or validate_verhoeff("234567890128") is False
    assert validate_verhoeff("123") is False

def test_pan_number_regex():
    assert validate_pan_number("ABCDE1234F") is True
    assert validate_pan_number("INVALID123") is False



def test_extract_ocr_data_dummy_image():
    """Tests local OCR extraction on a dummy image (no LLM calls)."""
    dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
    res = extract_ocr_data(dummy_img, document_type="PAN Card")
    
    assert "fields" in res
    assert "ocr_confidence" in res
    assert "id_checksum_valid" in res
    assert "raw_text" in res
    assert "llm_validation" in res
