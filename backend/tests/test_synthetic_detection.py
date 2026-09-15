import pytest
import numpy as np
from app.modules.synthetic_detection import (
    compute_garbled_text_ratio,
    check_qr_authenticity,
    check_camera_metadata_plausibility,
    compute_noise_floor,
    compute_synthetic_generation_score,
    detect_synthetic_document,
)
from app.modules.risk_scoring import predict_risk


def test_garbled_text_ratio_clean():
    lines = ["GOVERNMENT OF INDIA", "Name: Rahul Sharma", "DOB: 15/08/1990", "Gender: MALE"]
    labels = ["Government of India", "Name", "DOB", "Gender", "Aadhaar"]
    res = compute_garbled_text_ratio(lines, labels)
    assert res["garbled_ratio"] == 0.0
    assert not res["flag"]


def test_garbled_text_ratio_bilingual_genuine():
    # Real bilingual Aadhaar/PAN OCR fragments containing Hindi/English combined labels
    lines = [
        "INCOME TAX DEPARTMENT",
        "/Name",
        "fuaa/Father'sName",
        "DILIPKUMAR SHANTILAL SONI",
        "ata/Signature",
        "04/01/2005",
    ]
    labels = ["INCOME TAX DEPARTMENT", "Name", "Father's Name", "Signature", "Date of Birth"]
    res = compute_garbled_text_ratio(lines, labels)
    # Bilingual label fragments should fuzzy-match expected labels and NOT be flagged
    assert res["garbled_ratio"] == 0.0
    assert not res["flag"]


def test_garbled_text_ratio_synthetic():
    lines = [
        "GOVERNMENT OF INDIA",
        "faa/",
        "S.Partk",
        "Name: Rahul Sharma",
    ]
    labels = ["Government of India", "Name", "DOB", "Gender", "Aadhaar"]
    res = compute_garbled_text_ratio(lines, labels)
    assert res["garbled_ratio"] >= 0.25
    assert res["flag"]


def test_qr_authenticity_synthetic_vs_real():
    def make_qr_img(fuzzy=False):
        img = np.ones((200, 200, 3), dtype=np.uint8) * (128 if fuzzy else 255)
        def make_finder(top_left):
            y, x = top_left
            img[y:y+35, x:x+35] = 80 if fuzzy else 0
            img[y+5:y+30, x+5:x+30] = 160 if fuzzy else 255
            img[y+10:y+25, x+10:x+25] = 80 if fuzzy else 0
        make_finder((20, 20))
        make_finder((20, 145))
        make_finder((145, 20))
        return img

    fuzzy_img = make_qr_img(fuzzy=True)
    res_fuzzy = check_qr_authenticity(fuzzy_img, "Aadhaar")
    assert res_fuzzy["qr_found"]
    assert not res_fuzzy["looks_like_real_qr"]

    bw_img = make_qr_img(fuzzy=False)
    res_bw = check_qr_authenticity(bw_img, "Aadhaar")
    assert res_bw["qr_found"]
    assert res_bw["looks_like_real_qr"]


def test_camera_metadata_plausibility():
    clean_exif = {"Make": "Apple", "Model": "iPhone 13", "FNumber": 1.6}
    assert not check_camera_metadata_plausibility(clean_exif)["no_camera_origin_evidence"]

    empty_exif = {}
    assert not check_camera_metadata_plausibility(empty_exif)["no_camera_origin_evidence"]

    photoshop_exif = {"Software": "Adobe Photoshop 2023"}
    assert check_camera_metadata_plausibility(photoshop_exif)["no_camera_origin_evidence"]


def test_qr_authenticity_detector_failure_no_crop_fallback():
    # Plain image with no QR code detected should return qr_found = False, not fail with fallback crop false positive
    plain_image = np.ones((200, 300, 3), dtype=np.uint8) * 128
    res = check_qr_authenticity(plain_image, "PAN Card")
    assert not res["qr_found"]
    assert res["looks_like_real_qr"]


def test_single_signal_no_corroboration():
    # Only metadata missing (e.g. WhatsApp photo)
    garbled_res = {"flag": False, "garbled_ratio": 0.0, "garbled_fragments": []}
    qr_res = {"qr_found": True, "looks_like_real_qr": True}
    meta_res = {"no_camera_origin_evidence": True}
    noise_res = {"suspiciously_smooth": False}

    score_dict = compute_synthetic_generation_score(garbled_res, qr_res, meta_res, noise_res)
    assert not score_dict["is_corroborated"]
    assert score_dict["synthetic_generation_score"] <= 0.35


def test_two_numeric_signals_alone_not_corroborated():
    # 2 numeric signals alone → NOT corroborated (requires 3 without LLM visual)
    garbled_res = {"flag": False, "garbled_ratio": 0.0, "garbled_fragments": []}
    qr_res = {"qr_found": True, "looks_like_real_qr": False}
    meta_res = {"no_camera_origin_evidence": False}
    noise_res = {"suspiciously_smooth": True}

    score_dict = compute_synthetic_generation_score(garbled_res, qr_res, meta_res, noise_res)
    assert not score_dict["is_corroborated"]
    assert score_dict["synthetic_generation_score"] <= 0.35


def test_three_numeric_signals_triggers_corroboration():
    # 3 strong numeric signals → corroborated even without LLM
    garbled_res = {"flag": True, "garbled_ratio": 0.20, "garbled_fragments": ["foo"]}
    qr_res = {"qr_found": True, "looks_like_real_qr": False}
    meta_res = {"no_camera_origin_evidence": True}
    noise_res = {"suspiciously_smooth": False}

    score_dict = compute_synthetic_generation_score(garbled_res, qr_res, meta_res, noise_res)
    assert score_dict["is_corroborated"]
    assert score_dict["synthetic_generation_score"] >= 0.65


def test_llm_visual_plus_one_numeric_triggers_corroboration():
    # LLM says synthetic (score=0.25) + 1 numeric signal → corroborated
    garbled_res = {"flag": False, "garbled_ratio": 0.0, "garbled_fragments": []}
    qr_res = {"qr_found": True, "looks_like_real_qr": False}
    meta_res = {"no_camera_origin_evidence": False}
    noise_res = {"suspiciously_smooth": False}

    score_dict = compute_synthetic_generation_score(
        garbled_res, qr_res, meta_res, noise_res,
        visual_authenticity_score=0.25
    )
    assert score_dict["is_corroborated"]
    assert score_dict["synthetic_generation_score"] >= 0.60
    assert score_dict["llm_visual_available"] is True


def test_llm_visual_alone_no_numeric_not_corroborated():
    # LLM says synthetic but NO numeric signal → not corroborated (requires 1 numeric corroborator)
    garbled_res = {"flag": False, "garbled_ratio": 0.0, "garbled_fragments": []}
    qr_res = {"qr_found": False}
    meta_res = {"no_camera_origin_evidence": False}
    noise_res = {"suspiciously_smooth": False}

    score_dict = compute_synthetic_generation_score(
        garbled_res, qr_res, meta_res, noise_res,
        visual_authenticity_score=0.20
    )
    assert not score_dict["is_corroborated"]
    assert score_dict["synthetic_generation_score"] <= 0.35



def test_risk_scoring_synthetic_hard_override():
    res = predict_risk(
        ocr_confidence=0.95,
        validation_pass_rate=1.0,
        id_checksum_valid=True,
        expiry_valid=True,
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Aadhaar",
        synthetic_generation_score=0.75,
    )
    assert res["hard_override"]
    assert res["override_reason"] == "synthetic_image_suspected"
    assert res["risk_score"] >= 90
    assert res["prediction"] == "fraudulent"
