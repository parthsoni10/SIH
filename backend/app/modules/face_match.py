import cv2
import numpy as np
import logging
from PIL import Image
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

HAS_DEEPFACE = False
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

# Path to OpenCV YuNet ONNX model weights
YUNET_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

def get_yunet_detector(image_width: int, image_height: int, custom_model_path: Optional[str] = None):
    """Initializes OpenCV YuNet DNN face detector with requested image dimensions."""
    model_file = custom_model_path or str(YUNET_MODEL_PATH)
    if not Path(model_file).exists():
        return None
    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_file,
            config="",
            input_size=(image_width, image_height),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        return detector
    except Exception as e:
        logger.warning(f"Failed to create YuNet detector: {e}")
        return None

def get_face_cascade():
    """Initializes OpenCV Haar Cascade classifier as a last-resort fallback detector."""
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        pass
    return None

def detect_face_yunet(image_bgr: np.ndarray, custom_model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Detects faces in BGR image using OpenCV YuNet DNN model."""
    if image_bgr is None or image_bgr.size == 0:
        return []

    h, w = image_bgr.shape[:2]
    detector = get_yunet_detector(w, h, custom_model_path=custom_model_path)
    if detector is None:
        return []

    try:
        detector.setInputSize((w, h))
        _, faces = detector.detect(image_bgr)
        if faces is None:
            return []

        results = []
        for f in faces:
            x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            conf = float(f[-1])
            results.append({
                "bbox": (x, y, fw, fh),
                "confidence": conf
            })
        return results
    except Exception as e:
        logger.warning(f"YuNet detection error: {e}")
        return []

def detect_and_crop_face(
    image_bgr: np.ndarray,
    custom_yunet_path: Optional[str] = None,
    return_detector: bool = False
) -> Any:
    """
    Detects face region in image array using YuNet DNN primary and Haar Cascade fallback.
    Returns (face_found, cropped_image) or (face_found, cropped_image, detector_used) if return_detector=True.
    """
    if image_bgr is None or image_bgr.size == 0:
        return (False, None, "none") if return_detector else (False, None)

    img_h, img_w = image_bgr.shape[:2]

    yunet_faces = detect_face_yunet(image_bgr, custom_model_path=custom_yunet_path)
    if yunet_faces:
        yunet_faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
        x, y, w, h = yunet_faces[0]["bbox"]
        
        margin_w = int(w * 0.20)
        margin_h = int(h * 0.20)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)
        
        crop = image_bgr[y1:y2, x1:x2]
        return (True, crop, "yunet") if return_detector else (True, crop)

    cascade = get_face_cascade()
    if cascade is not None and not getattr(cascade, 'empty', lambda: True)():
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                margin_w = int(w * 0.15)
                margin_h = int(h * 0.15)
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h)
                x2 = min(img_w, x + w + margin_w)
                y2 = min(img_h, y + h + margin_h)
                crop = image_bgr[y1:y2, x1:x2]
                return (True, crop, "haar_fallback") if return_detector else (True, crop)
        except Exception as e:
            logger.warning(f"Haar cascade fallback error: {e}")

    return (False, None, "none") if return_detector else (False, None)

def compute_histogram_similarity(face1_bgr: np.ndarray, face2_bgr: np.ndarray) -> float:
    """Computes color histogram similarity (correlation metric, 0.0 to 1.0)."""
    try:
        f1 = cv2.resize(face1_bgr, (128, 128))
        f2 = cv2.resize(face2_bgr, (128, 128))

        hsv1 = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        correlation = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        return round(max(0.0, correlation), 4)
    except Exception:
        return 0.5

def crop_photo_by_doc_layout(image_bgr: np.ndarray, document_type: str) -> Optional[np.ndarray]:
    """Crops approximate face photo region based on document type layout heuristics."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    
    h, w = image_bgr.shape[:2]
    
    try:
        if document_type in ("Passport", "Visa"):
            return image_bgr[int(h * 0.10):int(h * 0.90), 0:int(w * 0.55)]
        elif document_type in ("Aadhaar", "PAN Card"):
            return image_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.02):int(w * 0.45)]
        elif document_type == "Driving License":
            return image_bgr[int(h * 0.10):int(h * 0.90), int(w * 0.02):int(w * 0.45)]
        else:
            return None
    except Exception:
        return None

def verify_face_match(
    doc_image_bgr: np.ndarray,
    live_capture_bytes: Optional[bytes],
    document_type: str = "Passport",
    custom_yunet_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares face extracted from document image against live capture image.
    When live capture photo is omitted, score is set to null (available=False, status='NOT_PROVIDED').
    CRITICAL: Never default missing face match to 0.93.
    """
    if not live_capture_bytes:
        return {
            "available": False,
            "score": None,
            "status": "NOT_PROVIDED",
            "face_match_score": None,
            "face_detected_in_doc": False,
            "face_detected_live": False,
            "detector_used_doc": "none",
            "detector_used_live": "none",
            "distance": None
        }

    try:
        nparr = np.frombuffer(live_capture_bytes, np.uint8)
        live_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if live_bgr is None:
            raise ValueError()
    except Exception:
        return {
            "available": False,
            "score": None,
            "status": "INVALID_CAPTURE_FILE",
            "face_match_score": None,
            "face_detected_in_doc": False,
            "face_detected_live": False,
            "detector_used_doc": "none",
            "detector_used_live": "none",
            "distance": None
        }

    layout_crop = crop_photo_by_doc_layout(doc_image_bgr, document_type)
    doc_face_found, doc_crop, detector_used_doc = detect_and_crop_face(
        layout_crop if layout_crop is not None else doc_image_bgr,
        custom_yunet_path=custom_yunet_path,
        return_detector=True
    )
    
    if not doc_face_found and layout_crop is not None:
        doc_face_found, doc_crop, detector_used_doc = detect_and_crop_face(
            doc_image_bgr,
            custom_yunet_path=custom_yunet_path,
            return_detector=True
        )

    live_face_found, live_crop, detector_used_live = detect_and_crop_face(
        live_bgr,
        custom_yunet_path=custom_yunet_path,
        return_detector=True
    )

    doc_target = doc_crop if doc_face_found else (layout_crop if layout_crop is not None else doc_image_bgr)
    live_target = live_crop if live_face_found else live_bgr

    similarity = 0.0
    distance = None

    if HAS_DEEPFACE:
        try:
            res = DeepFace.verify(
                img1_path=doc_target,
                img2_path=live_target,
                model_name="Facenet",
                enforce_detection=False
            )
            raw_dist = float(res.get("distance", 0.4))
            distance = round(raw_dist, 4)
            similarity = round(max(0.0, min(1.0, 1.0 - (raw_dist / 0.80))), 4)
        except Exception:
            similarity = compute_histogram_similarity(doc_target, live_target)
    else:
        similarity = compute_histogram_similarity(doc_target, live_target)

    status_str = "MATCHED" if similarity >= 0.70 else "MISMATCHED"

    return {
        "available": True,
        "score": similarity,
        "status": status_str,
        "face_match_score": similarity,
        "face_detected_in_doc": doc_face_found,
        "face_detected_live": live_face_found,
        "detector_used_doc": detector_used_doc,
        "detector_used_live": detector_used_live,
        "distance": distance
    }

def warmup_deepface_model():
    """Pre-warms DeepFace model weights if installed."""
    if HAS_DEEPFACE:
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.verify(img1_path=dummy, img2_path=dummy, model_name="Facenet", enforce_detection=False)
        except Exception:
            pass

