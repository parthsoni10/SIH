import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def temperature_scale_logits(logits: np.ndarray, temperature: float = 1.25) -> np.ndarray:
    """Applies temperature scaling calibration to raw logit outputs."""
    scaled_logits = logits / max(0.1, temperature)
    return 1.0 / (1.0 + np.exp(-scaled_logits))

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes mean squared difference between predicted probabilities and true binary labels."""
    return float(np.mean((y_prob - y_true) ** 2))

def calibrate_model_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates probability calibration quality and Brier score.
    """
    brier = compute_brier_score(y_true, y_prob)
    return {
        "brier_score": round(brier, 4),
        "temperature": 1.25,
        "calibration_version": "calibration_v1",
    }
