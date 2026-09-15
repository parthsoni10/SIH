import io
import cv2
import numpy as np
import logging
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "affinity", "pixlr",
    "lightroom", "adobe", "snapseed", "paint.net", "picsart", "fotor"
]

# Common dimensions for social media & messaging app re-compression (e.g. WhatsApp, Telegram)
KNOWN_MESSAGING_DIMENSIONS = {
    (1600, 1200), (1200, 1600),
    (1600, 900), (900, 1600),
    (1280, 960), (960, 1280),
    (1280, 720), (720, 1280),
    (1024, 768), (768, 1024),
}

def compute_ela_score(image_bgr: np.ndarray, original_bytes: Optional[bytes] = None, quality: int = 90, scale: int = 15) -> float:
    """
    Performs Error Level Analysis (ELA) by re-compressing image at specified JPEG quality
    and measuring the diff intensity across modified/unmodified areas.
    """
    try:
        if original_bytes and not original_bytes.startswith(b"%PDF"):
            try:
                pil_img = Image.open(io.BytesIO(original_bytes))
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
            except Exception:
                rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
        else:
            rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed_pil = Image.open(buffer)

        diff = ImageChops.difference(pil_img, recompressed_pil)
        diff_arr = np.array(diff, dtype=np.float32)
        
        diff_amplified = diff_arr * scale
        diff_clipped = np.clip(diff_amplified, 0, 255)

        mean_intensity = float(np.mean(diff_clipped) / 255.0)
        return min(max(mean_intensity, 0.0), 1.0)
    except Exception:
        return 0.0

def adaptive_ela_threshold(image_dims: Tuple[int, int]) -> float:
    """
    Returns resolution-adaptive ELA threshold to prevent false-positives on upstream-compressed photos.
    """
    if not image_dims:
        return 0.35
    longest_edge = max(image_dims)
    if longest_edge <= 1600:
        return 0.45  # Looser threshold for upstream compressed images (e.g. WhatsApp/Telegram)
    return 0.35      # Standard threshold for high-res scans

def compute_metadata_score(
    exif_dict: Dict[str, Any],
    file_bytes: Optional[bytes] = None,
    image_dims: Tuple[int, int] = (0, 0)
) -> Tuple[float, str]:
    """
    Analyzes EXIF metadata for explicit editing software tags vs messaging app EXIF stripping.
    Returns (metadata_score, reason_category).
    """
    if not exif_dict:
        # Check if dimensions match standard messaging app re-compression output
        if image_dims in KNOWN_MESSAGING_DIMENSIONS:
            return 0.15, "exif_stripped_likely_messaging_app"
        return 0.35, "exif_missing_unknown_cause"

    software_tag = str(exif_dict.get("Software", "")).lower()
    processing_software = str(exif_dict.get("ProcessingSoftware", "")).lower()
    combined_tags = f"{software_tag} {processing_software}"

    # Check for explicit editing software signatures
    for kw in EDITING_SOFTWARE_KEYWORDS:
        if kw in combined_tags:
            return 0.90, f"editing_software_detected_{kw}"

    # Check for missing original timestamp on present EXIF
    if "DateTimeOriginal" not in exif_dict and "DateTimeDigitized" not in exif_dict:
        return 0.25, "exif_present_missing_timestamps"

    return 0.05, "exif_present_clean"

def compute_noise_inconsistency(image_bgr: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> float:
    """
    Splits image into grid blocks and measures variance-of-variance in local noise estimates.
    Spliced/edited regions typically exhibit non-uniform noise distribution.
    """
    try:
        from skimage.restoration import estimate_sigma
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        rows, cols = grid_size
        block_h, block_w = h // rows, w // cols

        if block_h < 10 or block_w < 10:
            return 0.0

        block_sigmas = []
        for r in range(rows):
            for c in range(cols):
                block = gray[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                sigma = estimate_sigma(block, average_sigmas=True)
                block_sigmas.append(sigma)

        if not block_sigmas:
            return 0.0

        sigma_std = float(np.std(block_sigmas))
        sigma_mean = float(np.mean(block_sigmas)) + 1e-6

        relative_fluctuation = sigma_std / sigma_mean
        score = min(relative_fluctuation, 1.0)
        return round(score, 4)
    except Exception:
        return 0.0

def detect_tampering(
    original_bytes: bytes,
    image_bgr: np.ndarray,
    exif_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combines Error Level Analysis (ELA), EXIF metadata inspection, and noise inconsistency
    into an overall tampering_score (0.0 = clean, 1.0 = highly suspicious).
    """
    image_dims = (image_bgr.shape[1], image_bgr.shape[0]) if image_bgr is not None and image_bgr.size > 0 else (0, 0)

    ela_score = round(compute_ela_score(image_bgr, original_bytes=original_bytes), 4)
    metadata_score, meta_reason = compute_metadata_score(exif_dict, original_bytes, image_dims=image_dims)
    metadata_score = round(metadata_score, 4)
    noise_inconsistency = round(compute_noise_inconsistency(image_bgr), 4)

    # Weighted combination
    tampering_score = round(
        0.45 * ela_score + 0.35 * metadata_score + 0.20 * noise_inconsistency, 4
    )

    ela_cutoff = adaptive_ela_threshold(image_dims)

    flags = []
    if ela_score > ela_cutoff:
        flags.append(f"High Error Level Analysis (ELA) compression variance (score: {ela_score}, cutoff: {ela_cutoff})")
    if metadata_score > 0.50:
        flags.append(f"Editing software signature detected in EXIF metadata ({meta_reason})")
    if noise_inconsistency > 0.30:
        flags.append(f"Inconsistent background noise patterns detected (score: {noise_inconsistency})")

    return {
        "tampering_score": tampering_score,
        "signals": {
            "ela_score": ela_score,
            "metadata_score": metadata_score,
            "noise_inconsistency": noise_inconsistency
        },
        "meta_reason": meta_reason,
        "ela_cutoff": ela_cutoff,
        "flags": flags
    }

def analyze_error_level_analysis(original_bytes: bytes, image_bgr: np.ndarray) -> Dict[str, Any]:
    score = compute_ela_score(image_bgr, original_bytes=original_bytes)
    flags = []
    if score > 0.40:
        flags.append(f"High ELA compression variance (score={score:.4f})")
    return {"ela_score": score, "flags": flags}

def inspect_exif_editing_software(exif_dict: Dict[str, Any]) -> Dict[str, Any]:
    score, reason = compute_metadata_score(exif_dict)
    flags = []
    if score > 0.50:
        flags.append(f"Editing software signature ({reason})")
    return {"metadata_score": score, "reason": reason, "flags": flags}

def compute_block_noise_inconsistency(image_bgr: np.ndarray) -> float:
    return compute_noise_inconsistency(image_bgr)

