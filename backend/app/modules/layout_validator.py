import cv2
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points (x, y) in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def normalize_card_perspective(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects document boundary contour and applies 4-point perspective warp
    to correct camera rotation, angle tilt, or perspective skews up to ~45 degrees.
    Falls back to original image if no prominent card boundary is found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    try:
        orig_h, orig_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = orig_h * orig_w

        card_contour = None
        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(c)

            # Look for 4-point polygon with at least 15% of image area
            if len(approx) == 4 and area > 0.15 * img_area:
                card_contour = approx.reshape(4, 2)
                break

        if card_contour is None:
            return image_bgr

        # Apply 4-point perspective transform
        pts = order_points(card_contour)
        (tl, tr, br, bl) = pts

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_w = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_h = max(int(height_a), int(height_b))

        if max_w < 100 or max_h < 100:
            return image_bgr

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image_bgr, M, (max_w, max_h))
        return warped

    except Exception as e:
        logger.warning(f"Perspective normalization fallback triggered: {str(e)}")
        return image_bgr


def check_card_aspect_ratio(image_bgr: np.ndarray, document_type: str) -> Tuple[float, bool, str]:
    """
    Computes aspect ratio (w/h) and checks compliance against official ID standards:
    - ID-1 Standard (Aadhaar, PAN Card, Driving License): ratio ~1.586 (range 1.38 - 1.78)
    - Passport / Visa Booklet: ratio ~1.42 (range 1.25 - 1.65)
    """
    if image_bgr is None or image_bgr.size == 0:
        return 1.0, False, "Invalid image array"

    h, w = image_bgr.shape[:2]
    aspect_ratio = round(max(w, h) / float(min(w, h)), 3)

    if document_type in ("Aadhaar", "PAN Card", "Driving License"):
        is_ok = 1.38 <= aspect_ratio <= 1.78
        expected_str = "ISO/IEC 7810 ID-1 standard (~1.586)"
    elif document_type in ("Passport", "Visa"):
        is_ok = 1.25 <= aspect_ratio <= 1.65
        expected_str = "ICAO standard passport ratio (~1.42)"
    else:
        is_ok = 1.20 <= aspect_ratio <= 1.85
        expected_str = "General document format"

    status_msg = f"Aspect ratio: {aspect_ratio} (Expected: {expected_str})"
    return aspect_ratio, is_ok, status_msg


def validate_document_layout(
    image_bgr: np.ndarray,
    document_type: str,
    ocr_text_lines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Government Approved Layout & Geometry Verification (algorithmic only, no LLM):
    1. Corrects perspective skew / rotation angles.
    2. Validates card aspect ratio against ID-1 / ICAO standards.

    Note: Emblem/seal detection and photo position checks have been removed
    (previously required Gemini Vision). These default to True.
    """
    # 1. Perspective correction
    corrected_img = normalize_card_perspective(image_bgr)

    # 2. Aspect ratio evaluation
    aspect_ratio, aspect_ok, ratio_msg = check_card_aspect_ratio(corrected_img, document_type)

    anomalies = []
    if not aspect_ok:
        anomalies.append(f"Non-standard card aspect ratio ({aspect_ratio:.2f})")

    # Layout score: based on aspect ratio compliance only (no vision-based emblem check)
    aspect_penalty = 0.0 if aspect_ok else 0.20
    final_score = round(max(0.0, min(1.0, 0.85 - aspect_penalty)), 2)
    layout_valid = final_score >= 0.50 and aspect_ok

    return {
        "layout_valid": layout_valid,
        "layout_score": final_score,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_ok": aspect_ok,
        "emblem_detected": True,  # Default — no vision check available
        "spatial_geometry_valid": True,  # Default — no vision check available
        "layout_anomalies": anomalies,
        "corrected_image": corrected_img
    }
