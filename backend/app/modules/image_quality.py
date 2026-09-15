import cv2
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def assess_image_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Module: Image Quality Assessment & Gating
    Evaluates image quality metrics to determine forensic reliability.
    Poor image quality yields quality_too_low_for_forensics=True and routes to MANUAL_REVIEW.
    CRITICAL: Low image quality MUST NOT directly label an image as AI-generated.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "score": 0.0,
            "blur_score": 0.0,
            "resolution_ok": False,
            "brightness_ok": False,
            "contrast_score": 0.0,
            "saturation_score": 0.0,
            "jpeg_quality_estimate": 0.0,
            "noise_level": 0.0,
            "warnings": ["EMPTY_OR_UNREADABLE_IMAGE"],
            "quality_warning": True,
            "quality_too_low_for_forensics": True,
        }

    h, w = image_bgr.shape[:2]
    min_edge = min(h, w)
    aspect_ratio = round(w / float(h), 3)
    warnings: List[str] = []

    # 1. Blur score (Laplacian variance)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    if blur_score < 60.0:
        warnings.append("SEVERE_IMAGE_BLUR")
    elif blur_score < 100.0:
        warnings.append("SLIGHT_IMAGE_BLUR")

    # 2. Brightness & Contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    brightness_ok = True
    if brightness < 40.0:
        warnings.append("IMAGE_TOO_DARK")
        brightness_ok = False
    elif brightness > 220.0:
        warnings.append("IMAGE_OVEREXPOSED_GLARE")
        brightness_ok = False

    # 3. Saturation (HSV)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = round(float(np.mean(hsv[:, :, 1])), 2)

    # 4. Resolution score
    resolution_ok = min_edge >= 600
    if not resolution_ok:
        warnings.append(f"LOW_RESOLUTION ({min_edge}px, min 600px required)")

    # 5. Noise level (high-pass residual)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blurred)
    noise_level = round(float(np.mean(residual)), 2)

    # 6. JPEG quality estimate & 8x8 blockiness
    # Estimate blockiness across 8x8 boundary differences
    diff_h = np.abs(gray[8::8, :] - gray[7:-1:8, :])
    diff_v = np.abs(gray[:, 8::8] - gray[:, 7:-1:8])
    blockiness = round(float((np.mean(diff_h) + np.mean(diff_v)) / 2.0), 2)
    jpeg_quality_estimate = round(max(10.0, min(100.0, 100.0 - (blockiness * 3.5))), 1)
    if jpeg_quality_estimate < 50.0:
        warnings.append("HIGH_JPEG_COMPRESSION")

    # Composite Quality Score Calculation (0.0 to 1.0)
    # Blur weight (35%), Resolution (25%), Brightness/Contrast (20%), JPEG Quality (20%)
    blur_norm = min(1.0, blur_score / 250.0)
    res_norm = min(1.0, min_edge / 800.0)
    bright_norm = 1.0 if brightness_ok else 0.5
    jpeg_norm = jpeg_quality_estimate / 100.0

    quality_score = round(
        (blur_norm * 0.35) + (res_norm * 0.25) + (bright_norm * 0.20) + (jpeg_norm * 0.20),
        3
    )

    quality_warning = len(warnings) > 0
    quality_too_low_for_forensics = quality_score < 0.45 or min_edge < 300 or blur_score < 30.0

    if quality_too_low_for_forensics:
        warnings.append("QUALITY_TOO_LOW_FOR_FORENSICS")

    return {
        "score": quality_score,
        "blur_score": blur_score,
        "resolution_ok": resolution_ok,
        "brightness_ok": brightness_ok,
        "contrast_score": round(contrast, 2),
        "saturation_score": saturation,
        "jpeg_quality_estimate": jpeg_quality_estimate,
        "noise_level": noise_level,
        "warnings": warnings,
        "quality_warning": quality_warning,
        "quality_too_low_for_forensics": quality_too_low_for_forensics,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
    }
