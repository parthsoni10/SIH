import pytest
import cv2
import numpy as np

from app.modules.image_quality import assess_image_quality
from app.modules.ai_image_detector import predict_ai_image_probability
from app.modules.frequency_analysis import compute_frequency_analysis
from app.modules.noise_analysis import estimate_spatial_noise_anomalies
from app.modules.tampering_detector import detect_digital_tampering
from app.modules.face_match import verify_face_match
from app.modules.forensic_fusion import fuse_forensic_evidence
from app.modules.decision_engine import evaluate_final_decision

def create_dummy_image(w=800, h=600, color=(240, 240, 240)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.putText(img, "INCOME TAX DEPARTMENT", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "PERMANENT ACCOUNT NUMBER", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "ABCDE1234F", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.rectangle(img, (20, 20), (w-20, h-20), (50, 50, 50), 3)
    noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def test_1_valid_genuine_pan():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.15, "global_probability": 0.12, "patch_topk_probability": 0.18, "calibrated": True}
    freq = {"frequency_anomaly_score": 0.10}
    sp_noise = {"noise_anomaly_score": 0.10}
    meta = {"no_camera_origin_evidence": False}
    tamp = {"tampering_probability": 0.10}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "GENUINE"
    assert not dec["requires_manual_review"]

def test_2_ai_generated_pan():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.88, "global_probability": 0.85, "patch_topk_probability": 0.90}
    freq = {"frequency_anomaly_score": 0.75}
    sp_noise = {"noise_anomaly_score": 0.70}
    meta = {"no_camera_origin_evidence": True}
    tamp = {"tampering_probability": 0.10}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "AI_GENERATED"
    assert not dec["requires_manual_review"]

def test_3_genuine_pan_with_edited_name():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.15, "global_probability": 0.12, "patch_topk_probability": 0.18}
    freq = {"frequency_anomaly_score": 0.10}
    sp_noise = {"noise_anomaly_score": 0.10}
    meta = {"no_camera_origin_evidence": False}
    tamp = {"tampering_probability": 0.82, "flags": ["Strong ELA diff anomaly"]}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "ALTERED"

def test_4_ai_generated_pan_with_edited_name():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.88, "global_probability": 0.85, "patch_topk_probability": 0.90}
    freq = {"frequency_anomaly_score": 0.75}
    sp_noise = {"noise_anomaly_score": 0.70}
    meta = {"no_camera_origin_evidence": True}
    tamp = {"tampering_probability": 0.85, "flags": ["Copy-move forgery"]}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "AI_GENERATED_AND_ALTERED"

def test_5_blurry_genuine_pan_never_ai():
    img = create_dummy_image(color=(200, 200, 200))
    # Simulate heavy blur
    blurry_img = cv2.GaussianBlur(img, (25, 25), 0)
    quality = assess_image_quality(blurry_img)

    ai_det = {"ai_probability": 0.20, "global_probability": 0.20, "patch_topk_probability": 0.20}
    freq = {"frequency_anomaly_score": 0.10}
    sp_noise = {"noise_anomaly_score": 0.10}
    meta = {"no_camera_origin_evidence": False}
    tamp = {"tampering_probability": 0.15}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] in ("GENUINE", "MANUAL_REVIEW")
    assert dec["status"] != "AI_GENERATED"

def test_7_missing_exif_genuine_pan_never_ai():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.15, "global_probability": 0.15, "patch_topk_probability": 0.15, "calibrated": True}
    freq = {"frequency_anomaly_score": 0.05}
    sp_noise = {"noise_anomaly_score": 0.05}
    meta = {"no_camera_origin_evidence": False}
    tamp = {"tampering_probability": 0.10}
    val = {"validation_pass_rate": 1.0, "document_validity": {"score": 1.0, "structural_validity": True, "failed_rules": []}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "GENUINE"

def test_8_invalid_pan_format_suspicious():
    img = create_dummy_image()
    quality = assess_image_quality(img)
    ai_det = {"ai_probability": 0.15, "global_probability": 0.15, "patch_topk_probability": 0.15}
    freq = {"frequency_anomaly_score": 0.05}
    sp_noise = {"noise_anomaly_score": 0.05}
    meta = {"no_camera_origin_evidence": False}
    tamp = {"tampering_probability": 0.10}
    val = {"validation_pass_rate": 0.40, "document_validity": {"score": 0.40, "structural_validity": False, "failed_rules": ["invalid_document_number_format"]}, "blacklist_hit": False}

    fusion = fuse_forensic_evidence(ai_det, freq, sp_noise, meta, tamp, val)
    dec = evaluate_final_decision(fusion["ai_analysis"], tamp, val["document_validity"], quality, False)

    assert dec["status"] == "SUSPICIOUS"

def test_10_no_live_face_image_null_score():
    img = create_dummy_image()
    res = verify_face_match(img, None)
    assert not res["available"]
    assert res["score"] is None
    assert res["status"] == "NOT_PROVIDED"
