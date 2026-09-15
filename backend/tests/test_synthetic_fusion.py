import pytest
from app.modules.forensic_fusion import fuse_synthetic_forensics


def test_corroboration_rule_single_signal_not_corroborated():
    ai_detector = {"ai_probability": 0.85}
    frequency = {"frequency_score": 0.10}
    noise = {"suspiciously_smooth": False, "median_noise_variance": 10.0}
    metadata = {"no_camera_origin_evidence": False}
    garbled = {"flag": False, "garbled_ratio": 0.0}

    res = fuse_synthetic_forensics(ai_detector, frequency, noise, metadata, garbled)
    assert not res["is_corroborated"]
    assert res["prediction"] == "suspicious"
    assert res["strong_signal_count"] == 1


def test_corroboration_rule_two_strong_signals_corroborated():
    ai_detector = {"ai_probability": 0.80}
    frequency = {"frequency_score": 0.60}
    noise = {"suspiciously_smooth": False, "median_noise_variance": 10.0}
    metadata = {"no_camera_origin_evidence": False}
    garbled = {"flag": False, "garbled_ratio": 0.0}

    res = fuse_synthetic_forensics(ai_detector, frequency, noise, metadata, garbled)
    assert res["is_corroborated"]
    assert res["prediction"] == "likely_ai_generated"
    assert res["strong_signal_count"] == 2
    assert res["synthetic_score"] >= 0.65


def test_missing_exif_alone_never_causes_synthetic_override():
    ai_detector = {"ai_probability": 0.10}
    frequency = {"frequency_score": 0.05}
    noise = {"suspiciously_smooth": False, "median_noise_variance": 12.0}
    metadata = {"no_camera_origin_evidence": False}
    garbled = {"flag": False, "garbled_ratio": 0.0}

    res = fuse_synthetic_forensics(ai_detector, frequency, noise, metadata, garbled)
    assert not res["is_corroborated"]
    assert res["prediction"] == "likely_genuine"
    assert res["synthetic_score"] <= 0.30
