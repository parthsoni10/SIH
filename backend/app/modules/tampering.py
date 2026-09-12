import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, Tuple
from skimage.restoration import estimate_sigma

EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "paint.net", "picsart",
    "lightroom", "adobe", "snapseed", "pixlr", "fotor"
]

def compute_ela_score(image_bgr: np.ndarray, quality: int = 90, scale: int = 15) -> float:
    """
    Performs Error Level Analysis (ELA) by re-compressing image at specified JPEG quality
    and measuring the diff intensity across modified/unmodified areas.
    """
    try:
        # Convert BGR array to PIL Image
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Re-save to JPEG in memory buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed_pil = Image.open(buffer)

        # Compute absolute difference
        diff = ImageChops.difference(pil_img, recompressed_pil)
        
        # Convert diff to numpy array
        diff_arr = np.array(diff, dtype=np.float32)
        
        # Scale up diff intensity
        diff_amplified = diff_arr * scale
        diff_clipped = np.clip(diff_amplified, 0, 255)

        # Mean normalized diff intensity scalar
        mean_intensity = float(np.mean(diff_clipped) / 255.0)
        return min(max(mean_intensity, 0.0), 1.0)
    except Exception:
        return 0.0

def compute_metadata_score(exif_dict: Dict[str, Any], file_bytes: bytes) -> float:
    """
    Analyzes EXIF data for editing software tags or metadata manipulation.
    """
    if not exif_dict:
        # If no EXIF data present on uploaded file, slight penalty (0.20)
        return 0.20

    score = 0.0
    software_tag = str(exif_dict.get("Software", "")).lower()
    processing_software = str(exif_dict.get("ProcessingSoftware", "")).lower()

    # Check for editing software signatures
    for kw in EDITING_SOFTWARE_KEYWORDS:
        if kw in software_tag or kw in processing_software:
            score += 0.70
            break

    # Check for missing original timestamp
    if "DateTimeOriginal" not in exif_dict and "DateTimeDigitized" not in exif_dict:
        score += 0.15

    return min(score, 1.0)

def compute_noise_inconsistency(image_bgr: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> float:
    """
    Splits image into grid blocks and measures variance-of-variance in local noise estimates.
    Spliced/edited regions typically exhibit non-uniform noise distribution.
    """
    try:
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

        # Variance of block noise estimates
        sigma_std = float(np.std(block_sigmas))
        sigma_mean = float(np.mean(block_sigmas)) + 1e-6

        # Relative noise fluctuation score (normalized)
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
    into a overall tampering_score (0.0 = clean, 1.0 = highly suspicious).
    """
    ela_score = round(compute_ela_score(image_bgr), 4)
    metadata_score = round(compute_metadata_score(exif_dict, original_bytes), 4)
    noise_inconsistency = round(compute_noise_inconsistency(image_bgr), 4)

    # Weighted combination
    tampering_score = round(
        0.45 * ela_score + 0.35 * metadata_score + 0.20 * noise_inconsistency, 4
    )

    return {
        "tampering_score": tampering_score,
        "signals": {
            "ela_score": ela_score,
            "metadata_score": metadata_score,
            "noise_inconsistency": noise_inconsistency
        }
    }
