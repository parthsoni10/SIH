import pytest
import numpy as np
from app.modules.synthetic_detection import detect_synthetic_document


def test_synthetic_detection_end_to_end_genuine():
    blank = np.ones((300, 300, 3), dtype=np.uint8) * 200
    ocr_lines = ["GOVERNMENT OF INDIA", "Name: Rahul Sharma", "DOB: 15/08/1990"]
    exif_dict = {}

    res = detect_synthetic_document(blank, ocr_lines, exif_dict, "Aadhaar")
    assert "synthetic_generation_score" in res
    assert "synthetic_analysis" in res
    assert "ai_probability" in res
    assert "frequency_score" in res
    assert "strong_signal_count" in res
    assert res["synthetic_generation_score"] <= 0.55
    assert not res["is_corroborated"]



def test_synthetic_detection_end_to_end_ai_tool_exif():
    blank = np.ones((300, 300, 3), dtype=np.uint8) * 200
    ocr_lines = ["GOVERNMENT OF INDIA", "Name: Rahul Sharma"]
    exif_dict = {"Software": "Adobe Photoshop 2023"}

    res = detect_synthetic_document(blank, ocr_lines, exif_dict, "Aadhaar")
    assert "exif_ai_software_tag_detected" in res["synthetic_reasons"] or "no_camera_metadata_evidence" in res["synthetic_reasons"]
