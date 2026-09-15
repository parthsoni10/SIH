import pytest
import numpy as np
from app.modules.ai_image_detector import predict_ai_image_probability, AIDetectorService, AIImageDetector


def test_ai_detector_empty_image():
    """Empty image → ai_probability=None, not 0.0."""
    res = predict_ai_image_probability(np.array([]))
    assert res["ai_probability"] is None
    assert not res["is_ai_generated"]
    assert res["model_loaded"] is False or res["ai_probability"] is None


def test_ai_detector_valid_image():
    """Valid image → returns dict with ai_probability field."""
    dummy_img = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)
    res = predict_ai_image_probability(dummy_img)
    assert "ai_probability" in res
    # ai_probability is either None (model not loaded) or in [0, 1]
    if res["ai_probability"] is not None:
        assert 0.0 <= res["ai_probability"] <= 1.0
    assert "latency_ms" in res
    assert "model_version" in res


def test_ai_detector_singleton():
    s1 = AIDetectorService.get_instance()
    s2 = AIDetectorService.get_instance()
    assert s1 is s2


def test_ai_detector_unavailable_returns_null():
    """When model is unavailable, ai_probability must be None, not 0.0."""
    res = predict_ai_image_probability(np.array([]))
    assert res["ai_probability"] is None
    assert res["status"] in ("IMAGE_MISSING", "MODEL_NOT_READY", "MODEL_INFERENCE_ERROR")


def test_ai_detector_model_version_consistency():
    """Model version should be convnext_base_* format."""
    detector = AIImageDetector.get_instance()
    assert "convnext" in detector.model_version.lower() or "ai_detector" in detector.model_version.lower()
    # If model_loaded, version must match what metadata says
    res = predict_ai_image_probability(
        np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    )
    assert res["model_version"] == detector.model_version
