"""
SENTINEL V3.3 — qr_integrity.py PATCH
=======================================
Only change from V3: `QR_MANDATORY = {"PAN Card", "Aadhaar"}` was a hardcoded
set that silently excluded every other document type from ever being asked
to produce a QR, even ones you might add in future. `analyse_qr` now derives
mandatoriness from `DocumentSpec.qr_expected_side` instead, so it
automatically covers Passport/Visa/Driving License/Other correctly (all
`None` => optional) without another hardcoded list to maintain, and would
correctly start enforcing on a new type the moment you set
`qr_expected_side` on its spec.

Apply by replacing the `QR_MANDATORY` constant and the `analyse_qr` signature
in the existing qr_integrity.py with the versions below. Everything else in
that file (decode ladder, payload parsing, cross-check) is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as _zbar_decode
    _HAS_ZBAR = True
except Exception:
    _zbar_decode = None
    _HAS_ZBAR = False

MIN_CROP_LAPLACIAN = 120.0
MIN_CROP_PIXELS = 400

PAN_RE = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
AADHAAR_RE = re.compile(r"\b[2-9][0-9]{11}\b")


@dataclass
class QRResult:
    evidence_available: bool = False
    qr_expected: bool = False
    region_found: bool = False
    region_sharpness: float = 0.0
    region_pixels: int = 0
    decoded: bool = False
    symbol_type: str | None = None
    payload_length: int = 0
    payload_fields: dict[str, str] = field(default_factory=dict)
    cross_check_performed: bool = False
    cross_check_passed: bool | None = None
    mismatched_fields: list[str] = field(default_factory=list)
    synthetic_qr_score: float = 0.0
    payload_mismatch_score: float = 0.0
    genuine_evidence: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_regions(bgr: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        ok, pts = cv2.QRCodeDetector().detect(gray)
        if ok and pts is not None:
            p = pts.reshape(-1, 2).astype(int)
            x0, y0 = max(p[:, 0].min(), 0), max(p[:, 1].min(), 0)
            x1, y1 = min(p[:, 0].max(), w), min(p[:, 1].max(), h)
            if x1 - x0 > 40 and y1 - y0 > 40:
                pad = int(0.08 * (x1 - x0))
                out.append(bgr[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad])
    except Exception:
        pass
    try:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dense = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 60 or ch < 60:
                continue
            if 0.75 < cw / float(ch) < 1.33 and cw * ch > 0.01 * w * h:
                pad = int(0.10 * cw)
                out.append(bgr[max(y - pad, 0):y + ch + pad, max(x - pad, 0):x + cw + pad])
    except Exception:
        pass
    out.append(bgr[int(0.15 * h):int(0.75 * h), int(0.60 * w):int(1.00 * w)])
    return [c for c in out if c.size and c.shape[0] > 40 and c.shape[1] > 40]


def _try_decode(crop: np.ndarray):
    if not _HAS_ZBAR:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    sharp_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for scale in (1, 2, 3, 4):
        base = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants = [
            base, clahe.apply(base), cv2.filter2D(base, -1, sharp_k),
            cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 5),
        ]
        for v in variants:
            try:
                res = _zbar_decode(v)
            except Exception:
                continue
            if res:
                return res[0].data, res[0].type
    return None


def _parse_payload(raw: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return fields
    m = PAN_RE.search(text.upper())
    if m:
        fields["document_number"] = m.group(0)
    m = AADHAAR_RE.search(text)
    if m:
        fields["document_number"] = m.group(0)
    m = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
    if m:
        fields["dob"] = m.group(1)
    return fields


def analyse_qr(bgr: np.ndarray,
               spec,                     # DocumentSpec, not a raw string
               side: str,
               ocr_fields: dict[str, Any] | None = None) -> QRResult:
    """`spec` replaces the old `document_type: str` parameter. Mandatoriness
    is read from `spec.qr_expected_side` (set only on Aadhaar/PAN today, but
    works for any future type without touching this file)."""
    r = QRResult()
    r.qr_expected = (spec.qr_expected_side == side)
    ocr_fields = ocr_fields or {}

    if not _HAS_ZBAR:
        r.reasons.append("QR_DECODER_UNAVAILABLE_INSTALL_PYZBAR")
        return r
    if bgr is None or bgr.size == 0:
        return r

    best_crop, best_sharp = None, 0.0
    for crop in _candidate_regions(bgr):
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        s = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if s > best_sharp:
            best_crop, best_sharp = crop, s
        hit = _try_decode(crop)
        if hit:
            payload, sym = hit
            r.region_found = True
            r.decoded = True
            r.symbol_type = sym
            r.payload_length = len(payload)
            r.payload_fields = _parse_payload(payload)
            r.region_sharpness = s
            r.region_pixels = min(crop.shape[:2])
            break

    if best_crop is not None:
        r.region_found = True
        r.region_sharpness = max(r.region_sharpness, best_sharp)
        r.region_pixels = max(r.region_pixels, min(best_crop.shape[:2]))

    if r.decoded:
        r.evidence_available = True
        r.genuine_evidence = 0.90
        r.reasons.append("MACHINE_READABLE_PAYLOAD_DECODED")
        printed = str(ocr_fields.get("document_number", "")).strip().upper()
        encoded = str(r.payload_fields.get("document_number", "")).strip().upper()
        if printed and encoded:
            r.cross_check_performed = True
            if printed == encoded:
                r.cross_check_passed = True
                r.genuine_evidence = 0.97
                r.reasons.append("QR_PAYLOAD_MATCHES_PRINTED_FIELDS")
            else:
                r.cross_check_passed = False
                r.mismatched_fields.append("document_number")
                r.payload_mismatch_score = 0.95
                r.genuine_evidence = 0.0
                r.reasons.append("QR_PAYLOAD_CONTRADICTS_PRINTED_NUMBER")
        return r

    if not r.qr_expected:
        r.reasons.append("QR_NOT_REQUIRED_FOR_DOCUMENT_TYPE")
        return r

    sharp_enough = (r.region_sharpness >= MIN_CROP_LAPLACIAN
                    and r.region_pixels >= MIN_CROP_PIXELS)

    if not r.region_found:
        r.evidence_available = True
        r.synthetic_qr_score = 0.80
        r.reasons.append("MANDATORY_QR_REGION_ABSENT")
        return r

    if sharp_enough:
        r.evidence_available = True
        r.synthetic_qr_score = 0.92
        r.reasons.append("QR_STRUCTURE_PRESENT_BUT_NO_VALID_CODEWORD")
    else:
        r.evidence_available = False
        r.reasons.append("QR_UNDECODABLE_DUE_TO_CAPTURE_QUALITY")
        r.reasons.append("RECOMMEND_RECAPTURE_QR_REGION")

    return r
