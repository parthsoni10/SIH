import re
import cv2
import difflib
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

CAMERA_SPECIFIC_TAGS = [
    "Make", "Model", "FNumber", "ISOSpeedRatings",
    "ExposureTime", "GPSInfo", "FocalLength"
]

KNOWN_AI_EDITING_TOOLS = [
    "photoshop", "gimp", "canva", "stable diffusion", "stablediffusion",
    "midjourney", "dall-e", "dalle", "comfyui", "automatic1111", "firefly", "generative fill"
]

EXPECTED_LABELS_BY_DOC: Dict[str, List[str]] = {
    "Aadhaar": [
        "Government of India", "Bharat Sarkar", "Aadhaar", "Name", "DOB",
        "Year of Birth", "Gender", "Male", "Female", "Father", "Father's Name",
        "Address", "Signature", "Enrolment", "Help"
    ],
    "PAN Card": [
        "INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number Card",
        "Name", "Father's Name", "Date of Birth", "Signature"
    ],
    "Passport": [
        "Passport", "Republic of India", "Type", "Code", "Country Code",
        "Passport No", "Surname", "Given Name", "Nationality", "Date of Birth",
        "Sex", "Place of Birth", "Place of Issue", "Date of Issue", "Date of Expiry",
        "P<IND"
    ],
    "Driving License": [
        "Driving Licence", "Union Territory", "State", "DL No", "Name",
        "DOB", "Address", "Issue Date", "Validity", "Authorisation"
    ],
    "Visa": [
        "Visa", "Republic of India", "Passport No", "Visa No", "Type",
        "Entries", "Date of Issue", "Date of Expiry"
    ],
    "Other": [
        "Name", "DOB", "Document No", "Signature", "Date"
    ]
}


def is_fuzzy_label_match(text: str, expected_labels: List[str], threshold: float = 0.55) -> bool:
    """
    Checks if an OCR line or sub-fragment fuzzy matches any expected document label.
    Handles bilingual card labels (e.g. English labels appended to unparsed Hindi text like 'fuaa/Father'sName' or '/Name').
    """
    clean_text = text.lower().strip()
    if not clean_text:
        return False

    # Direct substring containment check
    for label in expected_labels:
        lbl_clean = label.lower().strip()
        if lbl_clean in clean_text or clean_text in lbl_clean:
            return True

    # Check sub-tokens split by slashes, dots, spaces, or colons
    sub_tokens = [t.strip() for t in re.split(r'[/.\s:]+', clean_text) if len(t.strip()) >= 2]
    for label in expected_labels:
        lbl_clean = label.lower().strip()
        for token in sub_tokens:
            if lbl_clean in token or token in lbl_clean:
                return True
            ratio = difflib.SequenceMatcher(None, token, lbl_clean).ratio()
            if ratio >= threshold:
                return True

        full_ratio = difflib.SequenceMatcher(None, clean_text, lbl_clean).ratio()
        if full_ratio >= threshold:
            return True

    return False


def compute_garbled_text_ratio(
    raw_ocr_lines: List[str],
    expected_labels: List[str]
) -> Dict[str, Any]:
    """
    Signal 1 — Garbled/nonsense micro-text density.
    AI generators mangle small print (labels, signatures, microtext).
    Uses fuzzy label matching so bilingual card OCR fragments (e.g. Hindi/English combined labels) are NOT falsely flagged.
    """
    if not raw_ocr_lines:
        return {
            "garbled_fragments": [],
            "garbled_ratio": 0.0,
            "flag": False,
        }

    garbled = []
    for line in raw_ocr_lines:
        cleaned = line.strip()
        if len(cleaned) < 2:
            continue

        # Check if line or sub-token fuzzy matches an expected document label (bilingual handling)
        if is_fuzzy_label_match(cleaned, expected_labels):
            continue

        # Truly garbled microtext: fails label fuzzy match AND matches mangled symbol/character patterns
        if re.search(r'[a-zA-Z]{1,}[/.][a-zA-Z]{1,}', cleaned) or cleaned.endswith('/') or cleaned.startswith('/'):
            garbled.append(cleaned)
        elif len(cleaned) <= 6 and not cleaned.isdigit() and not cleaned.isalpha() and any(c.isalpha() for c in cleaned):
            garbled.append(cleaned)

    ratio = len(garbled) / max(len(raw_ocr_lines), 1)
    return {
        "garbled_fragments": garbled,
        "garbled_ratio": round(ratio, 3),
        "flag": ratio > 0.15,
    }


def check_qr_authenticity(
    image_bgr: np.ndarray,
    document_type: str
) -> Dict[str, Any]:
    """
    Signal 2 — QR/barcode structural validity check.
    Real printed QR codes have sharp black & white pixel histogram extremes.
    AI-generated QR codes are fuzzy/gray textures with broad middle brightness.
    Threshold loosened to extreme_mass > 0.30 for real phone photos under blur/lighting variance.
    """
    empty_res = {
        "extreme_pixel_mass": 0.0,
        "middle_pixel_mass": 0.0,
        "looks_like_real_qr": True,
        "qr_found": False,
    }

    if image_bgr is None or image_bgr.size == 0:
        return empty_res

    # QR codes are expected primarily on Aadhaar and PAN Card
    if document_type not in ("Aadhaar", "PAN Card"):
        return empty_res

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Attempt 1: Detect actual QR region using OpenCV QRCodeDetector
    qr_crop = None
    try:
        detector = cv2.QRCodeDetector()
        retval, points = detector.detect(gray)
        if retval and points is not None and len(points) > 0:
            pts = points[0].astype(int)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            x_min = max(0, x_min - 5)
            y_min = max(0, y_min - 5)
            x_max = min(w, x_max + 5)
            y_max = min(h, y_max + 5)
            if (x_max - x_min) > 20 and (y_max - y_min) > 20:
                qr_crop = gray[y_min:y_max, x_min:x_max]
    except Exception as e:
        logger.debug(f"OpenCV QRCodeDetector check error: {e}")

    # Attempt 2: If detector failed to find QR code, return no signal rather than cropping card
    if qr_crop is None or qr_crop.size == 0:
        return empty_res

    # Bimodality check: calculate 16-bin histogram
    hist, _ = np.histogram(qr_crop, bins=16, range=(0, 255))
    total = float(hist.sum())
    if total == 0:
        return empty_res

    # Extreme mass: near-black (bins 0,1) + near-white (bins 14,15)
    extreme_mass = (hist[0] + hist[1] + hist[-1] + hist[-2]) / total
    middle_mass = hist[6:10].sum() / total

    # Real QR codes: extreme_mass > 0.30 (calibrated for real phone photo lighting/blur)
    looks_like_real_qr = extreme_mass > 0.30

    return {
        "extreme_pixel_mass": round(float(extreme_mass), 3),
        "middle_pixel_mass": round(float(middle_mass), 3),
        "looks_like_real_qr": looks_like_real_qr,
        "qr_found": True,
    }


def check_camera_metadata_plausibility(exif_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Signal 3 — Camera-metadata plausibility.
    Checks for explicit AI tool or photo editing software tags in EXIF.
    Empty or missing EXIF (common in browser uploads, scans, social media) is ambiguous
    and is NOT flagged as synthetic evidence.
    """
    if not exif_dict:
        return {
            "camera_tags_present": [],
            "camera_tag_count": 0,
            "no_camera_origin_evidence": False,
        }

    present = [tag for tag in CAMERA_SPECIFIC_TAGS if tag in exif_dict]
    software = str(exif_dict.get("Software", "") or exif_dict.get("ProcessingSoftware", "")).lower()
    has_ai_editing_tool = any(tool in software for tool in KNOWN_AI_EDITING_TOOLS)

    return {
        "camera_tags_present": present,
        "camera_tag_count": len(present),
        "no_camera_origin_evidence": has_ai_editing_tool,
    }


def compute_noise_floor(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Signal 4 — Noise-floor check.
    Measures overall median local noise variance using 5x5 box filter.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "median_noise_variance": 10.0,
            "suspiciously_smooth": False,
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    local_mean = cv2.blur(gray, (5, 5))
    local_sqmean = cv2.blur(gray ** 2, (5, 5))
    local_var = np.maximum(0.0, local_sqmean - (local_mean ** 2))
    median_noise = float(np.median(local_var))

    return {
        "median_noise_variance": round(median_noise, 3),
        "suspiciously_smooth": median_noise < 2.0,
    }


def compute_synthetic_generation_score(
    garbled_result: Dict[str, Any],
    qr_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
    noise_result: Dict[str, Any],
    visual_authenticity_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Combines Signals 1–4 (numeric) + Signal 5 (Gemini visual) into synthetic_generation_score (0.0 to 1.0).

    Corroboration rules:
    - LLM visual judgment (visual_authenticity_score < 0.4) + at least 1 numeric signal → override fires
    - Numeric only fallback: at least 3 strong numeric signals → override fires (when LLM unavailable)
    - Single numeric signal alone: never triggers override (capped at <= 0.35)
    """
    reasons = []

    sig_garbled = garbled_result.get("flag", False) and garbled_result.get("garbled_ratio", 0.0) > 0.15
    sig_qr_invalid = qr_result and qr_result.get("qr_found", False) and not qr_result.get("looks_like_real_qr", True)
    sig_no_meta = metadata_result.get("no_camera_origin_evidence", False)
    sig_smooth_noise = noise_result.get("suspiciously_smooth", False)
    sig_llm_visual = (visual_authenticity_score is not None) and (visual_authenticity_score < 0.35)

    numeric_strong_count = sum([sig_garbled, sig_qr_invalid, sig_no_meta, sig_smooth_noise])

    score = 0.0

    if sig_garbled:
        score += 0.15
        count = len(garbled_result.get("garbled_fragments", []))
        reasons.append(f"garbled_microtext_detected ({count} fragments)")

    if sig_qr_invalid:
        score += 0.30
        reasons.append("qr_code_structurally_invalid")

    if sig_no_meta:
        score += 0.10
        reasons.append("no_camera_metadata_evidence")

    if sig_smooth_noise:
        score += 0.10
        reasons.append("noise_floor_below_expected")

    if sig_llm_visual:
        score += 0.35
        reasons.append(f"llm_visual_judgment_synthetic (score={visual_authenticity_score:.2f})")

    # --- CORROBORATION RULE ---
    # Path 1: LLM visual says synthetic + at least 1 numeric corroborating signal
    is_corroborated = sig_llm_visual and numeric_strong_count >= 1
    # Path 2: Numeric-only fallback — requires 3 strong signals (when LLM unavailable)
    if not is_corroborated and not sig_llm_visual:
        is_corroborated = numeric_strong_count >= 3

    if is_corroborated:
        final_score = round(min(1.0, score + 0.30), 3)
    else:
        # Cap at 0.35 — never triggers hard override on its own
        final_score = round(min(0.35, score), 3)

    return {
        "synthetic_generation_score": final_score,
        "strong_signal_count": numeric_strong_count,
        "llm_visual_available": visual_authenticity_score is not None,
        "is_corroborated": is_corroborated,
        "reasons": reasons,
    }


from app.modules.ai_image_detector import predict_ai_image_probability
from app.modules.frequency_analysis import compute_frequency_analysis
from app.modules.forensic_fusion import fuse_synthetic_forensics


def detect_synthetic_document(
    image_bgr: np.ndarray,
    raw_ocr_lines: List[str],
    exif_dict: Dict[str, Any],
    document_type: str,
    visual_authenticity_score: Optional[float] = None,
    visual_evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Module 5b Master Entry Point (Upgraded):
    Orchestrates Deep Learning AI Detector (EfficientNet-B0), 2D FFT Frequency Analysis,
    Spatial Noise Floor, Metadata Inspection, and Microtext Density into the Forensic Fusion Engine.
    Enforces the 2-signal corroboration rule.
    """
    expected_labels = EXPECTED_LABELS_BY_DOC.get(document_type, EXPECTED_LABELS_BY_DOC["Other"])

    # 1. Deep Learning AI Detector (Primary signal)
    ai_detector_res = predict_ai_image_probability(image_bgr)

    # 2. 2D FFT Frequency Analysis
    frequency_res = compute_frequency_analysis(image_bgr)

    # 3. Spatial Noise Floor & Microtext & Metadata
    garbled_res = compute_garbled_text_ratio(raw_ocr_lines, expected_labels)
    qr_res = check_qr_authenticity(image_bgr, document_type)
    metadata_res = check_camera_metadata_plausibility(exif_dict)
    noise_res = compute_noise_floor(image_bgr)

    # 4. Multi-Signal Forensic Fusion
    fusion_res = fuse_synthetic_forensics(
        ai_detector_result=ai_detector_res,
        frequency_result=frequency_res,
        noise_result=noise_res,
        metadata_result=metadata_res,
        garbled_result=garbled_res,
        visual_authenticity_score=visual_authenticity_score,
    )

    # 5. Legacy hybrid score check (for backward compatibility)
    legacy_score_res = compute_synthetic_generation_score(
        garbled_res, qr_res, metadata_res, noise_res,
        visual_authenticity_score=visual_authenticity_score,
    )

    final_synthetic_score = max(fusion_res["synthetic_score"], legacy_score_res["synthetic_generation_score"])
    all_reasons = list(dict.fromkeys(fusion_res["reasons"] + legacy_score_res["reasons"]))

    return {
        "synthetic_generation_score": round(final_synthetic_score, 4),
        "synthetic_reasons": all_reasons,
        "prediction": fusion_res["prediction"],
        "status": fusion_res["status"],
        "confidence": fusion_res["confidence"],
        "strong_signal_count": fusion_res["strong_signal_count"],
        "is_corroborated": fusion_res["is_corroborated"],
        "ai_probability": ai_detector_res.get("ai_probability", 0.0),
        "frequency_score": frequency_res.get("frequency_score", 0.0),
        "synthetic_noise_score": fusion_res["signals"]["noise_score"],
        "metadata_score": fusion_res["signals"]["metadata_score"],
        "signals": {
            "ai_detector": ai_detector_res,
            "frequency_analysis": frequency_res,
            "garbled_microtext": garbled_res,
            "qr_authenticity": qr_res,
            "camera_metadata": metadata_res,
            "noise_floor": noise_res,
            "llm_visual": {
                "visual_authenticity_score": visual_authenticity_score,
                "visual_evidence": visual_evidence or [],
                "flag": (visual_authenticity_score is not None) and visual_authenticity_score < 0.35,
            },
        },
        "synthetic_analysis": {
            "score": round(final_synthetic_score, 4),
            "prediction": fusion_res["prediction"],
            "ai_prediction": fusion_res["prediction"],
            "ai_probability": ai_detector_res.get("ai_probability", 0.0),
            "frequency_score": frequency_res.get("frequency_score", 0.0),
            "fft_score": frequency_res.get("frequency_score", 0.0),
            "noise_score": fusion_res["signals"]["noise_score"],
            "metadata_score": fusion_res["signals"]["metadata_score"],
            "strong_signal_count": fusion_res["strong_signal_count"],
            "confidence": fusion_res["confidence"],
            "reasons": all_reasons,
            "model_version": "ai_detector_v1",
        }
    }



