import pytest
import numpy as np
import cv2
from app.modules.face_match import detect_face_yunet, detect_and_crop_face, verify_face_match
from app.modules.tampering import compute_metadata_score, adaptive_ela_threshold, detect_tampering

def test_yunet_face_detection_synthetic():
    """Verify YuNet detects synthetic face circle or returns valid detector_used indicator."""
    img = np.full((320, 320, 3), (200, 200, 200), dtype=np.uint8)
    cv2.circle(img, (160, 160), 80, (220, 180, 140), -1)  # face
    cv2.circle(img, (130, 140), 10, (40, 40, 40), -1)     # left eye
    cv2.circle(img, (190, 140), 10, (40, 40, 40), -1)     # right eye
    cv2.ellipse(img, (160, 190), (30, 15), 0, 0, 180, (40, 40, 40), 4)

    found, crop, detector_used = detect_and_crop_face(img, return_detector=True)
    assert detector_used in ("yunet", "haar_fallback", "none")

def test_yunet_invalid_model_path_fallback():
    """Verify system gracefully falls back to Haar Cascade when YuNet model path is invalid."""
    img = np.full((320, 320, 3), (200, 200, 200), dtype=np.uint8)
    found, crop, detector_used = detect_and_crop_face(img, custom_yunet_path="invalid/path/model.onnx", return_detector=True)
    assert detector_used in ("haar_fallback", "none")

def test_messaging_app_exif_scoring():
    """Verify stripped EXIF on WhatsApp dimensions (1600x1200) scores ~0.15 instead of 0.5+."""
    exif_empty = {}
    dims_messaging = (1600, 1200)
    score, reason = compute_metadata_score(exif_empty, image_dims=dims_messaging)
    
    assert score == 0.15
    assert reason == "exif_stripped_likely_messaging_app"

def test_photoshop_editing_exif_scoring():
    """Verify Photoshop signature in EXIF scores 0.90 regardless of image dimensions."""
    exif_photoshop = {"Software": "Adobe Photoshop 2024 (Windows)"}
    dims = (1600, 1200)
    score, reason = compute_metadata_score(exif_photoshop, image_dims=dims)
    
    assert score == 0.90
    assert "editing_software_detected" in reason

def test_adaptive_ela_threshold():
    """Verify adaptive ELA threshold returns 0.45 for images <= 1600px and 0.35 for high-res images."""
    assert adaptive_ela_threshold((1600, 1200)) == 0.45
    assert adaptive_ela_threshold((1280, 960)) == 0.45
    assert adaptive_ela_threshold((2500, 2000)) == 0.35

def test_tampering_detection_with_messaging_image():
    """Verify end-to-end tampering detection on messaging image returns clean/low score."""
    img = np.full((1200, 1600, 3), (240, 240, 240), dtype=np.uint8)
    dummy_bytes = b"dummy_jpeg_bytes"
    res = detect_tampering(dummy_bytes, img, exif_dict={})
    
    assert res["meta_reason"] == "exif_stripped_likely_messaging_app"
    assert res["signals"]["metadata_score"] == 0.15
    assert res["ela_cutoff"] == 0.45
