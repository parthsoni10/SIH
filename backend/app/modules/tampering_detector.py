import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from app.modules.tampering import (
    analyze_error_level_analysis,
    inspect_exif_editing_software,
    compute_block_noise_inconsistency
)
from app.modules.alteration_detector import detect_alteration
from app.modules.anchor_verifier import verify_anchors
from app.modules.document_spec import get_spec, SIDE_FRONT

logger = logging.getLogger(__name__)

def detect_copy_move_and_splicing(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Evaluates copy-move forgery and splicing edge boundary discontinuities using ORB feature matching
    and Canny edge gradient variances across text bounding zones.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"copy_move_score": 0.0, "splicing_score": 0.0}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. ORB Keypoint Matching for Copy-Move Detection
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(gray, None)
    copy_move_score = 0.0

    if des is not None and len(des) > 10:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des, des, k=2)

        duplicate_pairs = 0
        for m_pair in matches:
            if len(m_pair) == 2:
                m1, m2 = m_pair
                if m1.distance < 0.6 * m2.distance:
                    pt1 = np.array(kp[m1.queryIdx].pt)
                    pt2 = np.array(kp[m1.trainIdx].pt)
                    dist = np.linalg.norm(pt1 - pt2)
                    if dist > 30:
                        duplicate_pairs += 1

        copy_move_score = min(1.0, duplicate_pairs / 25.0)

    # 2. Splicing Edge Discontinuity Analysis
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel)

    patch_h, patch_w = h // 4, w // 4
    edge_vars = []
    for i in range(4):
        for j in range(4):
            p_edge = dilated_edges[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            edge_vars.append(np.var(p_edge))

    splicing_score = 0.0
    if edge_vars and np.mean(edge_vars) > 0:
        edge_variance_ratio = float(np.std(edge_vars) / (np.mean(edge_vars) + 1e-5))
        if edge_variance_ratio > 1.5:
            splicing_score = min(1.0, (edge_variance_ratio - 1.5) / 2.0)

    return {
        "copy_move_score": round(copy_move_score, 4),
        "splicing_score": round(splicing_score, 4),
    }

def detect_digital_tampering(
    image_bytes: bytes,
    image_bgr: np.ndarray,
    exif_dict: Dict[str, Any],
    document_type: str = "Aadhaar",
    anchor_result: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Module: Independent Digital Tampering Detector (V3.1 Enhanced)
    Combines V3.1 alteration detection & anchor verification with legacy ELA, EXIF, and noise analysis.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "tampering_probability": 0.0,
            "probability": 0.0,
            "ela_score": 0.0,
            "noise_inconsistency": 0.0,
            "copy_move_score": 0.0,
            "splicing_score": 0.0,
            "metadata_score": 0.0,
            "confidence": 0.0,
            "flags": ["EMPTY_IMAGE_TAMPERING_CHECK_FAILED"],
            "v31_alteration": None
        }

    # Run anchor verification if not supplied
    if anchor_result is None:
        spec = get_spec(document_type)
        anchor_result = verify_anchors(image_bgr, spec, side=SIDE_FRONT, warped=False)

    # Run V3.1 alteration detector
    v31_alt = detect_alteration(image_bgr, anchor_result)

    # 1. Error Level Analysis (ELA)
    ela_res = analyze_error_level_analysis(image_bytes, image_bgr)
    ela_score = round(float(ela_res["ela_score"]), 4)

    # 2. EXIF Editing Software Inspection
    exif_res = inspect_exif_editing_software(exif_dict)
    metadata_score = round(float(exif_res["metadata_score"]), 4)

    # 3. Block Noise Inconsistency
    noise_inc = compute_block_noise_inconsistency(image_bgr)
    noise_inconsistency = round(float(noise_inc), 4)

    # 4. Copy-Move & Splicing
    forgery_res = detect_copy_move_and_splicing(image_bgr)
    copy_move_score = forgery_res["copy_move_score"]
    splicing_score = forgery_res["splicing_score"]

    flags: List[str] = list(ela_res.get("flags", [])) + list(exif_res.get("flags", [])) + list(v31_alt.reasons)

    if copy_move_score > 0.40:
        flags.append(f"Copy-move feature duplication detected (score={copy_move_score:.2f})")
    if splicing_score > 0.45:
        flags.append(f"Splicing edge boundary anomaly detected (score={splicing_score:.2f})")

    # Combine V3.1 probability with legacy score (taking maximum if anchor erasure established)
    legacy_prob = round(
        (ela_score * 0.40) +
        (max(copy_move_score, splicing_score) * 0.30) +
        (noise_inconsistency * 0.20) +
        (metadata_score * 0.10),
        4
    )

    final_prob = max(legacy_prob, v31_alt.probability)
    confidence = max(v31_alt.confidence, round(min(1.0, max(0.5, final_prob * 1.2)), 4))

    return {
        "tampering_probability": final_prob,
        "probability": final_prob,
        "ela_score": ela_score,
        "noise_inconsistency": noise_inconsistency,
        "copy_move_score": copy_move_score,
        "splicing_score": splicing_score,
        "metadata_score": metadata_score,
        "confidence": confidence,
        "flags": list(dict.fromkeys(flags)),
        "corroborated": v31_alt.corroborated,
        "strong_cues": v31_alt.strong_cues,
        "suspect_regions": v31_alt.suspect_regions,
        "v31_alteration": v31_alt.to_dict(),
        "anchor_result": anchor_result.to_dict() if hasattr(anchor_result, "to_dict") else anchor_result,
    }
