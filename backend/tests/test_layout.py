import pytest
import numpy as np
import cv2
from app.modules.layout_validator import (
    normalize_card_perspective,
    check_card_aspect_ratio,
    validate_document_layout
)

def test_check_card_aspect_ratio_all_doc_types():
    # Test valid aspect ratio for Aadhaar (475x300 => 1.583)
    img_aadhaar = np.zeros((300, 475, 3), dtype=np.uint8)
    ratio, is_ok, msg = check_card_aspect_ratio(img_aadhaar, "Aadhaar")
    assert is_ok is True
    assert 1.38 <= ratio <= 1.78

    # Test valid aspect ratio for Passport (568x400 => 1.42)
    img_passport = np.zeros((400, 568, 3), dtype=np.uint8)
    ratio, is_ok, msg = check_card_aspect_ratio(img_passport, "Passport")
    assert is_ok is True

    # Test valid aspect ratio for PAN Card (475x300 => 1.583)
    img_pan = np.zeros((300, 475, 3), dtype=np.uint8)
    ratio, is_ok, msg = check_card_aspect_ratio(img_pan, "PAN Card")
    assert is_ok is True

    # Test invalid aspect ratio (e.g. extremely tall image 800x200 => 4.0)
    img_tall = np.zeros((800, 200, 3), dtype=np.uint8)
    ratio, is_ok, msg = check_card_aspect_ratio(img_tall, "Aadhaar")
    assert is_ok is False

def test_normalize_card_perspective():
    # Create a synthetic white card contour on dark canvas
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    pts = np.array([[150, 100], [650, 120], [620, 450], [120, 400]], np.int32)
    cv2.fillPoly(canvas, [pts], (255, 255, 255))

    warped = normalize_card_perspective(canvas)
    assert warped is not None
    assert warped.shape[0] > 0 and warped.shape[1] > 0

def test_validate_document_layout_synthetic():
    """Tests layout validation on a synthetic card (purely algorithmic, no LLM)."""
    # Create synthetic card image (800x500 - aspect ratio 1.60)
    card_img = np.ones((500, 800, 3), dtype=np.uint8) * 220
    cv2.rectangle(card_img, (10, 10), (790, 490), (50, 50, 50), 3)

    res = validate_document_layout(card_img, "Aadhaar")
    assert "layout_score" in res
    assert "layout_anomalies" in res
    assert "aspect_ratio_ok" in res
    assert res["aspect_ratio_ok"] is True
    assert 0.0 <= res["layout_score"] <= 1.0

def test_validate_document_layout_corrupt_or_empty():
    empty_img = np.array([], dtype=np.uint8)
    res = validate_document_layout(empty_img, "PAN Card")
    assert "layout_score" in res
    assert res["aspect_ratio_ok"] is False
