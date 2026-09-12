import numpy as np
from app.modules.tampering import detect_tampering, compute_ela_score, compute_metadata_score

def test_detect_tampering_structure():
    dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    res = detect_tampering(b"dummybytes", dummy_img, exif_dict={})
    
    assert "tampering_score" in res
    assert 0.0 <= res["tampering_score"] <= 1.0
    assert "signals" in res
    assert "ela_score" in res["signals"]
    assert "metadata_score" in res["signals"]
    assert "noise_inconsistency" in res["signals"]

def test_metadata_editing_software():
    score = compute_metadata_score({"Software": "Adobe Photoshop CS6"}, b"dummy")
    assert score >= 0.70
