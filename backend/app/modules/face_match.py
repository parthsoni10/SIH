import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Any, Optional, Tuple

HAS_DEEPFACE = False
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

def get_face_cascade():
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        pass
    return None

def detect_and_crop_face(image_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Detects face region in image array and returns cropped face array."""
    if image_bgr is None or image_bgr.size == 0:
        return False, None

    cascade = get_face_cascade()
    if cascade is None or getattr(cascade, 'empty', lambda: True)():
        return False, None

    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            return False, None

        # Pick largest face box
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add margin padding around face box
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)
        
        img_h, img_w = image_bgr.shape[:2]
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        crop = image_bgr[y1:y2, x1:x2]
        return True, crop
    except Exception:
        return False, None

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

def verify_face_match(
    doc_image_bgr: np.ndarray,
    live_capture_bytes: Optional[bytes]
) -> Dict[str, Any]:
    """
    Compares face extracted from document image against live capture image.
    Returns face_match_score, face_detected_in_doc, face_detected_live.
    """
    if not live_capture_bytes:
        return {
            "face_match_score": None,
            "face_detected_in_doc": False,
            "face_detected_live": False,
            "distance": None
        }

    # Decode live capture bytes
    try:
        nparr = np.frombuffer(live_capture_bytes, np.uint8)
        live_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if live_bgr is None:
            raise ValueError()
    except Exception:
        return {
            "face_match_score": None,
            "face_detected_in_doc": False,
            "face_detected_live": False,
            "distance": None
        }

    # Detect faces in doc and live capture
    doc_face_found, doc_crop = detect_and_crop_face(doc_image_bgr)
    live_face_found, live_crop = detect_and_crop_face(live_bgr)

    # Use entire image if face cascade missed face
    doc_target = doc_crop if doc_face_found else doc_image_bgr
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
            similarity = round(max(0.0, min(1.0, 1.0 - raw_dist)), 4)
        except Exception:
            similarity = compute_histogram_similarity(doc_target, live_target)
    else:
        similarity = compute_histogram_similarity(doc_target, live_target)

    return {
        "face_match_score": similarity,
        "face_detected_in_doc": doc_face_found,
        "face_detected_live": live_face_found,
        "distance": distance
    }
