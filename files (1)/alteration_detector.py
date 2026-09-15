"""
SENTINEL V3.1 — Core Alteration / Tampering Detector
=====================================================
Replaces the scoring in `tampering_detector.py`, which returned 4.3% overall
(ELA 2%) on a card with an erased government logo.

Why ELA failed, and why it will keep failing
--------------------------------------------
Error Level Analysis assumes the image is a single-generation JPEG whose
edited region has a different recompression history from its surroundings.
Both preconditions were violated on sample #207:

  * The document went through a scanner app that flattened the paper
    background to pure white. Measured: background pixels read EXACTLY 255
    with ZERO local variance across 86% of the near-white area, in BOTH the
    genuine and the altered card. There is no noise floor left to be
    inconsistent.
  * The file was then sent through WhatsApp, which strips EXIF and
    re-encodes the WHOLE frame uniformly. Any per-region compression history
    is erased in one pass.

So the honest conclusion is: for scanned-and-messaged documents, pixel-level
recompression forensics are dead on arrival. Do not tune them. Replace them.

The five cues below each report `available` independently and several
SELF-DISABLE when their preconditions fail. That is the important property --
a cue that cannot measure must say so rather than returning a small
reassuring number, which is precisely how 4.3% got rendered as
"LOW TAMPERING" on a forged card.

Measured cue performance on the reference Aadhaar pair:

    cue                     genuine    altered    verdict
    anchor_removal           0.000      0.920     FIRES (decisive)
    pasted_region_geometry   0 edges    4 edges    FIRES (decisive)
    noise_void               n/a        n/a        SELF-DISABLED (bg flat)
    ela                      0.02       0.02       useless, down-weighted
    colour_discontinuity     low        low        no signal here
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class Cue:
    name: str
    available: bool
    score: float = 0.0
    detail: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class AlterationResult:
    probability: float = 0.0
    confidence: float = 0.0
    corroborated: bool = False
    strong_cues: list[str] = field(default_factory=list)
    available_cue_count: int = 0

    cues: dict[str, dict] = field(default_factory=dict)
    suspect_regions: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    # Surfaced so the decision engine can gate on it separately.
    background_flattened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRONG = 0.60

# Cues that survive scanner-app flattening and messenger recompression.
ROBUST_CUES = {"anchor_removal", "pasted_region_geometry"}

WEIGHTS = {
    "anchor_removal":         0.40,
    "pasted_region_geometry": 0.30,
    "noise_void":             0.15,
    "colour_discontinuity":   0.08,
    "ela":                    0.07,
}


# ---------------------------------------------------------------------------
# Precondition probe
# ---------------------------------------------------------------------------
def _background_flatness(g: np.ndarray) -> float:
    """Fraction of near-white pixels with effectively zero local variance.

    Above ~0.5 the document has been through a scanner-app 'document mode'
    that clamps paper to pure white. Every noise-based cue is then blind.
    """
    k = np.ones((5, 5), np.float64) / 25.0
    m = cv2.filter2D(g, -1, k)
    v = cv2.filter2D((g - m) ** 2, -1, k)
    white = g > 235
    if white.sum() < 1000:
        return 0.0
    return float((v[white] < 0.5).mean())


# ---------------------------------------------------------------------------
# Cue 1 — anchor removal (delegated to anchor_verifier)
# ---------------------------------------------------------------------------
def _cue_anchor_removal(anchor_result) -> Cue:
    if anchor_result is None or not getattr(anchor_result, "evidence_available", False):
        return Cue("anchor_removal", False,
                   reasons=["ANCHOR_VERIFICATION_NOT_RUN"])
    return Cue(
        "anchor_removal", True,
        score=float(getattr(anchor_result, "artwork_removal_score", 0.0)),
        detail={
            "missing_anchors": list(getattr(anchor_result, "missing_anchors", [])),
            "template_completeness": getattr(anchor_result, "template_completeness", 1.0),
        },
        reasons=list(getattr(anchor_result, "reasons", [])),
    )


# ---------------------------------------------------------------------------
# Cue 2 — pasted region geometry
# ---------------------------------------------------------------------------
def _cue_pasted_geometry(bgr: np.ndarray, anchor_result) -> Cue:
    """Ask whether a MISSING anchor region has been covered by a uniform fill.

    An earlier version of this cue searched the whole image for axis-aligned
    edges. It fired strongly on the GENUINE reference card, because a document
    is full of legitimate straight lines: the card border, the dashed cut
    line, the red rule, and -- on Aadhaar specifically -- the genuine white
    rounded panel behind the 12-digit number. Clustering those into one
    bounding box produced a confident false positive.

    The cue is therefore targeted, not exploratory. It only evaluates regions
    that anchor verification already flagged as missing artwork, and asks a
    narrow question: is this region a flat fill bounded by pixel-perfect
    straight edges? That converts it from an independent detector into a
    corroborating explanation of HOW the artwork disappeared, which is what it
    is actually good at.

    Consequence: with no missing anchors the cue reports 0.0 rather than
    hunting for trouble. That is intended.
    """
    missing = list(getattr(anchor_result, "missing_anchors", []) or [])
    findings = {f["key"]: f for f in getattr(anchor_result, "findings", [])}
    if not getattr(anchor_result, "evidence_available", False):
        return Cue("pasted_region_geometry", False, 0.0, {},
                   ["GEOMETRY_CUE_REQUIRES_ANCHOR_VERIFICATION"])
    if not missing:
        return Cue("pasted_region_geometry", True, 0.0,
                   {"evaluated_regions": 0},
                   ["NO_MISSING_ANCHORS_TO_EXPLAIN"])

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]

    from .document_spec import REGISTRY  # local import avoids a cycle
    bbox_by_key: dict[str, tuple] = {}
    for spec in REGISTRY.values():
        for a in spec.anchors:
            bbox_by_key.setdefault(a.key, a.bbox)

    regions: list[dict] = []
    best = 0.0
    for key in missing:
        bb = bbox_by_key.get(key)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        # Widen slightly so the fill box's own borders fall inside the crop.
        px0, py0 = int(max(0.0, x0 - 0.02) * w), int(max(0.0, y0 - 0.02) * h)
        px1, py1 = int(min(1.0, x1 + 0.02) * w), int(min(1.0, y1 + 0.02) * h)
        sub = g[py0:py1, px0:px1]
        if sub.size == 0 or min(sub.shape) < 20:
            continue

        edges = cv2.Canny(sub, 40, 120)
        min_len = max(40, int(0.35 * min(sub.shape)))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=min_len, maxLineGap=3)
        n_axis = 0
        if lines is not None:
            for a1, b1, a2, b2 in lines[:, 0]:
                if abs(a1 - a2) <= 1 or abs(b1 - b2) <= 1:
                    n_axis += 1

        interior = g[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.float64)
        flat = float(cv2.Laplacian(interior, cv2.CV_64F).var()) if interior.size else 0.0
        uniform = float((interior > 235).mean()) if interior.size else 0.0

        if n_axis >= 3 and flat < 400:
            s = 0.88
        elif uniform > 0.90 and flat < 200:
            s = 0.80          # blanked to paper white with no border drawn
        elif n_axis >= 3:
            s = 0.55
        else:
            s = 0.30

        regions.append({
            "anchor": key,
            "norm_bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
            "axis_aligned_edges": n_axis,
            "interior_laplacian_var": round(flat, 2),
            "uniform_white_fraction": round(uniform, 4),
            "ink_coverage": findings.get(key, {}).get("ink_coverage"),
            "score": s,
        })
        best = max(best, s)

    reasons = []
    if best >= STRONG:
        reasons.append("MISSING_ANCHOR_COVERED_BY_UNIFORM_FILL")
    return Cue("pasted_region_geometry", True, best,
               {"evaluated_regions": len(regions), "regions": regions}, reasons)


# ---------------------------------------------------------------------------
# Cue 3 — noise void (self-disabling)
# ---------------------------------------------------------------------------
def _cue_noise_void(g: np.ndarray, bg_flat: float) -> Cue:
    if bg_flat >= 0.50:
        return Cue("noise_void", False, 0.0,
                   {"background_flatness": round(bg_flat, 4)},
                   ["NOISE_VOID_CUE_DISABLED_BACKGROUND_PREFLATTENED"])

    k = np.ones((7, 7), np.float64) / 49.0
    m = cv2.filter2D(g, -1, k)
    v = cv2.filter2D((g - m) ** 2, -1, k)
    void = (v < 0.5).astype(np.uint8)
    void = cv2.morphologyEx(void, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(void)
    total_px = g.size
    biggest = 0
    for i in range(1, n):
        biggest = max(biggest, int(stats[i, cv2.CC_STAT_AREA]))
    frac = biggest / float(total_px)
    score = float(np.clip(frac / 0.02, 0.0, 1.0))
    return Cue("noise_void", True, score,
               {"largest_void_fraction": round(frac, 5)},
               ["NOISE_VOID_REGION_DETECTED"] if score >= STRONG else [])


# ---------------------------------------------------------------------------
# Cue 4 — colour discontinuity
# ---------------------------------------------------------------------------
def _cue_colour_discontinuity(bgr: np.ndarray) -> Cue:
    """Compare the WHITE POINT of paper-background tiles only.

    An earlier version measured cast spread across all tiles and scored 0.863
    on the genuine reference card against 0.846 on the altered one -- i.e. it
    ranked them backwards and would have condemned every colourful document.
    The reason is obvious in hindsight: an Aadhaar card legitimately contains
    a saffron/green brush band, a red rule and a blue portrait, so high
    chromatic variance across the whole frame is the NORMAL state.

    Paper background is the only surface whose colour SHOULD be uniform, so
    that is the only surface worth measuring. A region filled in an editor
    typically uses pure #FFFFFF, while scanned paper carries a slight warm or
    cool cast, making the filled tile an outlier in a*/b*.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    gh, gw = L.shape
    step_y, step_x = max(gh // 10, 1), max(gw // 10, 1)

    casts = []
    for y in range(0, gh - step_y + 1, step_y):
        for x in range(0, gw - step_x + 1, step_x):
            tl = L[y:y + step_y, x:x + step_x]
            paper = tl > 230
            # Only tiles that are overwhelmingly background.
            if paper.mean() < 0.85:
                continue
            ta = a[y:y + step_y, x:x + step_x][paper]
            tb = b[y:y + step_y, x:x + step_x][paper]
            if ta.size:
                casts.append((ta.mean(), tb.mean()))

    if len(casts) < 6:
        return Cue("colour_discontinuity", False, 0.0,
                   {"paper_tiles": len(casts)},
                   ["COLOUR_CUE_DISABLED_INSUFFICIENT_PAPER_AREA"])

    casts = np.asarray(casts)
    med = np.median(casts, axis=0)
    dev = np.hypot(casts[:, 0] - med[0], casts[:, 1] - med[1])
    mad = float(np.median(np.abs(dev - np.median(dev))))
    worst = float(dev.max())

    # THIRD self-disabling precondition, and the subtlest one. A pure z-score
    # divides by a spread that a scanner app has already driven to zero: on
    # the reference pair every paper tile sat at exactly the same white point,
    # giving MAD ~ 0 and a nominal z of ~240,000 on BOTH cards from a worst
    # deviation of only 0.6 LAB units. That is a divide-by-zero wearing a
    # statistic's clothing, and it scored 1.0 on the genuine card.
    #
    # Guard with an ABSOLUTE floor: a deviation that small is not visible and
    # not evidence, whatever its z-score.
    if mad < 0.05 or worst < 2.0:
        return Cue("colour_discontinuity", False, 0.0,
                   {"paper_tiles": int(len(casts)),
                    "worst_tile_deviation": round(worst, 3),
                    "mad": round(mad, 4)},
                   ["COLOUR_CUE_DISABLED_WHITE_POINT_CLAMPED_BY_SCANNER"])

    z = worst / (mad * 1.4826)
    # Require both statistical AND absolute significance.
    score = float(np.clip((z - 6.0) / 10.0, 0.0, 1.0)) * \
        float(np.clip((worst - 2.0) / 4.0, 0.0, 1.0))
    return Cue("colour_discontinuity", True, round(score, 4),
               {"paper_tiles": int(len(casts)),
                "worst_tile_deviation": round(worst, 3),
                "robust_z": round(z, 2)},
               ["PAPER_WHITE_POINT_DISCONTINUITY"] if score >= STRONG else [])


# ---------------------------------------------------------------------------
# Cue 5 — ELA (retained, heavily down-weighted, self-disabling)
# ---------------------------------------------------------------------------
def _cue_ela(bgr: np.ndarray, bg_flat: float) -> Cue:
    if bg_flat >= 0.50:
        return Cue("ela", False, 0.0, {},
                   ["ELA_CUE_DISABLED_BACKGROUND_PREFLATTENED"])
    try:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=90)
        buf.seek(0)
        recomp = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)
        diff = cv2.absdiff(bgr, recomp).astype(np.float64).max(2)
        score = float(np.clip(diff.std() / 12.0, 0.0, 1.0))
        return Cue("ela", True, score, {"ela_std": round(float(diff.std()), 3)}, [])
    except Exception as exc:
        return Cue("ela", False, 0.0, {}, [f"ELA_FAILED_{type(exc).__name__}"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_alteration(bgr: np.ndarray, anchor_result=None) -> AlterationResult:
    r = AlterationResult()
    if bgr is None or bgr.size == 0:
        r.reasons.append("ALTERATION_NO_INPUT")
        return r

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bg_flat = _background_flatness(g)
    r.background_flattened = bg_flat >= 0.50

    cues = [
        _cue_anchor_removal(anchor_result),
        _cue_pasted_geometry(bgr, anchor_result),
        _cue_noise_void(g, bg_flat),
        _cue_colour_discontinuity(bgr),
        _cue_ela(bgr, bg_flat),
    ]

    num = den = 0.0
    for c in cues:
        r.cues[c.name] = asdict(c) | {"strong": c.available and c.score >= STRONG}
        if not c.available:
            continue
        w = WEIGHTS[c.name]
        num += w * c.score
        den += w
        r.reasons.extend(c.reasons)
        if c.score >= STRONG:
            r.strong_cues.append(c.name)
        if c.name == "pasted_region_geometry":
            r.suspect_regions.extend(c.detail.get("regions", []))

    r.available_cue_count = sum(1 for c in cues if c.available)
    r.probability = round(num / den, 4) if den > 0 else 0.0

    # Corroboration: two strong cues, at least one recapture-robust. Mirrors
    # the V3 family rule so a single artefact cannot condemn a document.
    robust_strong = [c for c in r.strong_cues if c in ROBUST_CUES]
    r.corroborated = len(r.strong_cues) >= 2 and bool(robust_strong)

    # A single erased mandatory anchor is a presence assertion, not a
    # statistical hint. It stands alone.
    if "anchor_removal" in r.strong_cues and \
       r.cues["anchor_removal"]["score"] >= 0.85:
        r.probability = max(r.probability, 0.90)
        r.corroborated = True
        r.reasons.append("ALTERATION_ESTABLISHED_BY_ANCHOR_ERASURE")
    elif len(r.strong_cues) == 1 and not robust_strong:
        r.probability = min(r.probability, 0.35)
        r.reasons.append("SINGLE_FRAGILE_CUE_CAPPED")

    coverage = den / sum(WEIGHTS.values())
    r.confidence = round(min(1.0, 0.5 * coverage +
                             0.5 * abs(r.probability - 0.5) * 2), 4)

    if r.background_flattened:
        r.reasons.append("PIXEL_FORENSICS_LIMITED_SCANNER_APP_PREPROCESSING")

    r.reasons = list(dict.fromkeys(r.reasons))
    return r
