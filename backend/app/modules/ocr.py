import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from app.modules.mrz import parse_mrz_lines

# Lazy paddleocr loading
_paddle_ocr_instance = None
HAS_PADDLE = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

def get_ocr_engine():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None and HAS_PADDLE:
        try:
            _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except Exception:
            _paddle_ocr_instance = None
    return _paddle_ocr_instance

# Verhoeff algorithm for Aadhaar validation
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(number_str: str) -> bool:
    clean_num = re.sub(r'\D', '', number_str)
    if len(clean_num) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(clean_num)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(item)]]
    return c == 0

def validate_pan_number(pan: str) -> bool:
    """Validates 10-character PAN pattern: 5 letters, 4 digits, 1 letter."""
    return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan.strip().upper()))

def extract_ocr_data(image: np.ndarray, document_type: str) -> Dict[str, Any]:
    """
    Runs OCR engine on preprocessed image and extracts mapped fields, confidence scores,
    and ID checksum status.
    """
    ocr_engine = get_ocr_engine()
    ocr_results = []
    
    if ocr_engine:
        try:
            raw_ocr = ocr_engine.ocr(image, cls=True)
            if raw_ocr and raw_ocr[0]:
                for line in raw_ocr[0]:
                    bbox, (text, conf) = line
                    ocr_results.append({
                        "bbox": bbox,
                        "text": text.strip(),
                        "confidence": float(conf)
                    })
        except Exception:
            pass

    # Extract all text strings for pattern matching
    all_lines = [item["text"] for item in ocr_results]

    # MRZ check for Passport & Visa
    mrz_res = parse_mrz_lines(all_lines)
    
    # Initialize fields dict and checksum
    fields: Dict[str, Dict[str, Any]] = {}
    id_checksum_valid = False

    # Extract fields based on document type & MRZ
    if mrz_res["present"]:
        mrz_fields = mrz_res["fields"]
        id_checksum_valid = mrz_res["checksum_valid"]
        for k, v in mrz_fields.items():
            fields[k] = {"value": v, "confidence": 0.95}
    else:
        fields = map_fields_by_keywords(ocr_results, document_type)
        id_checksum_valid = verify_checksum_by_doc_type(fields, document_type, all_lines)

    # Compute overall OCR confidence score
    if ocr_results:
        conf_scores = [item["confidence"] for item in ocr_results]
        ocr_confidence = round(float(np.mean(conf_scores)), 4)
    else:
        ocr_confidence = 0.0

    return {
        "fields": fields,
        "ocr_confidence": ocr_confidence,
        "id_checksum_valid": id_checksum_valid,
        "mrz": mrz_res,
        "raw_text": all_lines
    }

def map_fields_by_keywords(ocr_results: List[Dict[str, Any]], document_type: str) -> Dict[str, Dict[str, Any]]:
    """Maps raw OCR lines to key-value pairs using label heuristics and regex."""
    fields = {}
    full_text = " ".join([r["text"] for r in ocr_results])
    
    # Date of Birth pattern
    dob_match = re.search(r'(?:DOB|Date of Birth|Birth|DOB:)\s*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4}[/-][0-9]{2}[/-][0-9]{2})', full_text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = {"value": dob_match.group(1), "confidence": 0.90}
    else:
        # Fallback date search
        generic_date = re.search(r'\b([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})\b', full_text)
        if generic_date:
            fields["dob"] = {"value": generic_date.group(1), "confidence": 0.80}

    # Expiry Date pattern
    exp_match = re.search(r'(?:Expiry|Exp|Valid Until|Date of Expiry)\s*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4}[/-][0-9]{2}[/-][0-9]{2})', full_text, re.IGNORECASE)
    if exp_match:
        fields["expiry_date"] = {"value": exp_match.group(1), "confidence": 0.90}

    # Specific Doc Type patterns
    if document_type == "Aadhaar":
        aadhaar_match = re.search(r'\b[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\b', full_text)
        if aadhaar_match:
            fields["document_number"] = {"value": aadhaar_match.group(0).replace(" ", ""), "confidence": 0.92}
    elif document_type == "PAN Card":
        pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', full_text)
        if pan_match:
            fields["document_number"] = {"value": pan_match.group(0), "confidence": 0.95}
    elif document_type == "Driving License":
        dl_match = re.search(r'\b[A-Z]{2}[0-9]{2}\s?[0-9]{11}\b', full_text)
        if dl_match:
            fields["document_number"] = {"value": dl_match.group(0), "confidence": 0.90}
    elif document_type in ("Passport", "Visa"):
        pass_match = re.search(r'\b[A-Z][0-9]{7}\b', full_text)
        if pass_match:
            fields["document_number"] = {"value": pass_match.group(0), "confidence": 0.91}

    return fields

def verify_checksum_by_doc_type(fields: Dict[str, Dict[str, Any]], document_type: str, all_lines: List[str]) -> bool:
    """Verifies checksum / pattern validity based on document type."""
    doc_num = fields.get("document_number", {}).get("value", "")

    if document_type == "Aadhaar":
        if not doc_num:
            # Search all_lines for 12 digits
            for line in all_lines:
                clean = re.sub(r'\D', '', line)
                if len(clean) == 12:
                    doc_num = clean
                    break
        return validate_verhoeff(doc_num) if doc_num else False

    elif document_type == "PAN Card":
        if not doc_num:
            for line in all_lines:
                match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', line.upper())
                if match:
                    doc_num = match.group(0)
                    break
        return validate_pan_number(doc_num) if doc_num else False

    elif document_type in ("Passport", "Visa"):
        # If no MRZ was found, pattern match passport number
        return bool(re.match(r'^[A-Z][0-9]{7}$', doc_num.strip().upper())) if doc_num else False

    elif document_type == "Driving License":
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}\s?[0-9]{11}$', doc_num.strip().upper())) if doc_num else False

    return False
