"""
SENTINEL V3.1 — Mandatory Anchor Verification
==============================================
Detects REMOVED or BLANKED printed artwork, which is the alteration class that
ELA, noise analysis and the ConvNeXt detector all miss entirely.

Prerequisite
------------
Run the 4-point deskew/crop warp from layout_validator BEFORE calling this, or
every normalised bbox lands in the wrong place. If the warp is unavailable,
pass `warped=False` and the module widens its tolerance and reports reduced
confidence rather than emitting false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

from .document_spec import AnchorSpec, DocumentSpec, SIDE_FRONT


@dataclass
class AnchorFinding:
    key: str
    description: str
    present: bool
    ink_coverage: float
    colour_coverage: float
    required_ink: float
    required_colour: float
    margin: float          # how far above/below the requirement, signed


@dataclass
class AnchorResult:
    evidence_available: bool = False
    confidence: float = 0.0
    warped: bool = True

    findings: list[dict] = field(default_factory=list)
    missing_anchors: list[str] = field(default_factory=list)
    anchors_checked: int = 0

    # 0..1 -- high means mandatory artwork has been removed
    artwork_removal_score: float = 0.0
    # structural completeness of the printed template, 0..1
    template_completeness: float = 1.0

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _measure(region_bgr: np.ndarray) -> tuple[float, float]:
    """Return (ink_coverage, colour_coverage) for a document region.

    'Ink' = pixel is either dark or chromatic. Both are needed: black text is
    dark but unsaturated, a saffron brush band is bright but saturated.
    """
    if region_bgr is None or region_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    ink = float(((val < 200) | (sat > 60)).mean())
    colour = float((sat > 80).mean())
    return ink, colour


def _crop(bgr: np.ndarray, bbox, inset: float = 0.0) -> np.ndarray:
    """Sample an anchor region, optionally INSET (shrunk inward).

    When no deskew warp is available the correct response is to shrink the box,
    not pad it. Padding pulls in neighbouring artwork and destroys the measurement.
    Shrinking keeps the sample inside the anchor core, where a few percent of
    registration error still lands on the artwork.
    """
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    dx = (x1 - x0) * inset
    dy = (y1 - y0) * inset
    x0, x1 = x0 + dx, x1 - dx
    y0, y1 = y0 + dy, y1 - dy
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = bbox
    return bgr[int(max(0.0, y0) * h):int(min(1.0, y1) * h),
               int(max(0.0, x0) * w):int(min(1.0, x1) * w)]


def verify_anchors(bgr: np.ndarray,
                   spec: DocumentSpec,
                   side: str = SIDE_FRONT,
                   warped: bool = True) -> AnchorResult:
    r = AnchorResult(warped=warped)

    if bgr is None or bgr.size == 0 or spec is None:
        r.reasons.append("ANCHOR_CHECK_NO_INPUT")
        return r

    anchors = [a for a in spec.anchors if a.side == side]
    if not anchors:
        r.evidence_available = True
        r.confidence = 1.0
        r.reasons.append(f"NO_ANCHORS_DECLARED_FOR_SIDE_{side}")
        return r

    # Without a deskew warp the boxes drift, so sample the anchor CORE
    # (inset) and demand a bigger shortfall before declaring anything missing.
    inset = 0.0 if warped else 0.12
    slack = 1.0 if warped else 0.45

    missing: list[AnchorFinding] = []
    for a in anchors:
        region = _crop(bgr, a.bbox, inset)
        ink, colour = _measure(region)

        need_ink = a.min_ink_coverage * slack
        need_col = a.min_colour_coverage * slack
        ok = (ink >= need_ink) and (colour >= need_col)

        f = AnchorFinding(
            key=a.key, description=a.description, present=ok,
            ink_coverage=round(ink, 4), colour_coverage=round(colour, 4),
            required_ink=round(need_ink, 4), required_colour=round(need_col, 4),
            margin=round(ink - need_ink, 4),
        )
        r.findings.append(asdict(f))
        if not ok:
            missing.append(f)
            r.missing_anchors.append(a.key)

    r.anchors_checked = len(anchors)
    r.evidence_available = True
    r.confidence = 1.0 if warped else 0.55
    r.template_completeness = round(
        1.0 - len(missing) / float(len(anchors)), 4)

    if missing:
        worst = min(f.ink_coverage / max(f.required_ink, 1e-6) for f in missing)
        if worst < 0.10:
            r.artwork_removal_score = 0.92
            r.reasons.append("MANDATORY_ARTWORK_ERASED")
        elif worst < 0.50:
            r.artwork_removal_score = 0.70
            r.reasons.append("MANDATORY_ARTWORK_DEGRADED_OR_COVERED")
        else:
            r.artwork_removal_score = 0.45
            r.reasons.append("MANDATORY_ARTWORK_BELOW_EXPECTED_DENSITY")

        if not warped:
            if worst < 0.10:
                r.artwork_removal_score *= 0.95
                r.reasons.append("ANCHOR_VOID_DECISIVE_DESPITE_NO_WARP")
            else:
                r.artwork_removal_score *= 0.60
                r.reasons.append("ANCHOR_SCORE_DISCOUNTED_NO_DESKEW_WARP")

        for f in missing:
            r.reasons.append(f"ANCHOR_MISSING_{f.key.upper()}")
    else:
        r.reasons.append("ALL_MANDATORY_ANCHORS_PRESENT")

    return r
