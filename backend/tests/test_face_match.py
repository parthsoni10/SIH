import numpy as np
from app.modules.face_match import verify_face_match, detect_and_crop_face

def test_face_match_no_live_capture():
    dummy_doc = np.zeros((200, 200, 3), dtype=np.uint8)
    res = verify_face_match(dummy_doc, live_capture_bytes=None)
    
    assert res["face_match_score"] is None
    assert res["face_detected_in_doc"] is False
    assert res["face_detected_live"] is False

def test_detect_and_crop_face_blank_image():
    blank = np.ones((300, 300, 3), dtype=np.uint8) * 200
    found, crop = detect_and_crop_face(blank)
    assert found is False
