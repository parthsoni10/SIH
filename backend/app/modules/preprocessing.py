import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from app.config import settings

@dataclass
class PreprocessedImage:
    image: np.ndarray  # BGR format np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    format: str
    exif_present: bool
    exif_dict: Dict[str, Any] = field(default_factory=dict)
    low_quality: bool = False
    blur_score: float = 0.0

def extract_exif(pil_img: Image.Image) -> Tuple[bool, Dict[str, Any]]:
    """Extracts raw EXIF dictionary from PIL Image."""
    try:
        raw_exif = pil_img._getexif()
        if not raw_exif:
            return False, {}
        
        exif_dict = {}
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Convert non-serializable bytes/tuples if necessary
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore")
                except Exception:
                    value = str(value)
            exif_dict[tag_name] = value
            
        return True, exif_dict
    except Exception:
        return False, {}

def orient_and_convert_pil(pil_img: Image.Image) -> Image.Image:
    """Auto-orients image based on EXIF orientation tag."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img

def preprocess_image(
    file_bytes: bytes,
    max_edge: int = settings.MAX_IMAGE_EDGE,
    blur_threshold: float = 100.0
) -> PreprocessedImage:
    """
    Normalizes uploaded image bytes (JPEG/PNG/PDF):
    1. Extracts EXIF and auto-orients.
    2. Decodes to BGR numpy array.
    3. Resizes longest edge to max_edge without upscaling.
    4. Computes blur score (Laplacian variance) and flags low_quality.
    """
    if not file_bytes:
        raise ValueError("Empty file bytes provided.")

    # 1. Attempt PDF conversion if magic bytes indicate PDF (%PDF)
    image_format = "JPEG"
    if file_bytes.startswith(b"%PDF"):
        image_format = "PDF"
        if HAS_PDF2IMAGE:
            pages = convert_from_bytes(file_bytes, first_page=1, last_page=1)
            if not pages:
                raise ValueError("Could not rasterize PDF file.")
            pil_img = pages[0]
        else:
            raise RuntimeError("pdf2image library is not installed to process PDF files.")
    else:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            image_format = pil_img.format or "JPEG"
        except Exception as e:
            raise ValueError(f"Failed to decode image file: {str(e)}")

    # 2. Extract EXIF before orientation transformation
    exif_present, exif_dict = extract_exif(pil_img)

    # 3. Auto-orient image
    pil_img = orient_and_convert_pil(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_width, orig_height = pil_img.size

    # Convert to OpenCV BGR array
    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # 4. Resize longest edge bound
    h, w = bgr_array.shape[:2]
    longest_edge = max(h, w)

    if longest_edge > max_edge:
        scale = max_edge / float(longest_edge)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr_array = cv2.resize(bgr_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    proc_height, proc_width = bgr_array.shape[:2]

    # 5. Compute blur score via Laplacian variance
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low_quality = blur_score < blur_threshold

    return PreprocessedImage(
        image=bgr_array,
        original_size=(orig_width, orig_height),
        processed_size=(proc_width, proc_height),
        format=image_format,
        exif_present=exif_present,
        exif_dict=exif_dict,
        low_quality=low_quality,
        blur_score=round(blur_score, 2),
    )
