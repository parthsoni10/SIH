import pytest
import numpy as np
from app.modules.frequency_analysis import compute_frequency_analysis


def test_frequency_analysis_empty_image():
    res = compute_frequency_analysis(np.array([]))
    assert res["frequency_score"] == 0.0
    assert not res["is_spectral_anomaly"]


def test_frequency_analysis_synthetic_checkerboard():
    # Checkerboard image has high frequency spectral spikes
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    res = compute_frequency_analysis(img)
    assert res["frequency_score"] > 0.0
    assert "low_freq_energy" in res
    assert "high_freq_energy" in res
    assert "spectral_entropy" in res


def test_frequency_analysis_smooth_photo():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    res = compute_frequency_analysis(img)
    assert res["frequency_score"] < 0.50
