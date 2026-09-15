import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def estimate_spatial_noise_anomalies(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Module: Spatial Noise Analysis Engine
    Measures spatial noise residual variance, Gaussian denoising residual,
    and noise consistency across document regions.
    Outputs noise_anomaly_score (0.0 to 1.0).
    CRITICAL: A low noise score alone MUST NEVER classify an image as AI-generated.
    """
    empty_result = {
        "noise_anomaly_score": 0.0,
        "residual_variance": 0.0,
        "gaussian_residual_std": 0.0,
        "local_variance_median": 0.0,
        "noise_consistency_ratio": 1.0,
        "suspiciously_smooth": False,
    }

    if image_bgr is None or image_bgr.size == 0:
        return empty_result

    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape

        # 1. Denoising Residual Analysis (High-frequency noise component)
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_residual = gray - denoised
        residual_variance = round(float(np.var(noise_residual)), 3)
        gaussian_residual_std = round(float(np.std(noise_residual)), 3)

        # 2. Local 5x5 Box Variance
        local_mean = cv2.blur(gray, (5, 5))
        local_sqmean = cv2.blur(gray ** 2, (5, 5))
        local_var = np.maximum(0.0, local_sqmean - (local_mean ** 2))
        local_var_median = round(float(np.median(local_var)), 3)

        # 3. Patch Noise Consistency across 4x4 Grid
        patch_h, patch_w = h // 4, w // 4
        patch_vars = []
        if patch_h > 10 and patch_w > 10:
            for i in range(4):
                for j in range(4):
                    p_res = noise_residual[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    patch_vars.append(np.var(p_res))

        if patch_vars and np.mean(patch_vars) > 0:
            noise_consistency_ratio = round(float(np.std(patch_vars) / (np.mean(patch_vars) + 1e-5)), 3)
        else:
            noise_consistency_ratio = 0.0

        # Unnaturally smooth AI gradients: residual_variance < 0.8 and local_var_median < 1.2
        suspiciously_smooth = residual_variance < 0.8 and local_var_median < 1.2

        # Compute noise_anomaly_score (0.0 to 1.0)
        anomaly = 0.0
        if suspiciously_smooth:
            anomaly += 0.40

        if noise_consistency_ratio > 1.8:
            # High spatial inconsistency across document sections
            anomaly += 0.35
        elif noise_consistency_ratio > 1.2:
            anomaly += 0.20

        if residual_variance > 45.0:
            # Excessive unnatural noise injection
            anomaly += 0.25

        noise_anomaly_score = round(min(1.0, anomaly), 4)

        return {
            "noise_anomaly_score": noise_anomaly_score,
            "residual_variance": residual_variance,
            "gaussian_residual_std": gaussian_residual_std,
            "local_variance_median": local_var_median,
            "noise_consistency_ratio": noise_consistency_ratio,
            "suspiciously_smooth": suspiciously_smooth,
        }
    except Exception as e:
        logger.error(f"Error computing spatial noise analysis: {e}")
        return empty_result
