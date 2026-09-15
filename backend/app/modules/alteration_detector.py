"""
SENTINEL V3.1 — Core Alteration / Tampering Detector
=====================================================
Evaluates five independent alteration cues:
  1. anchor_removal (mandatory artwork presence assertion)
  2. pasted_region_geometry (targeted check for flat fill / straight edges over missing artwork)
  3. noise_void (spatial noise consistency, self-disables on flat backgrounds)
  4. colour_discontinuity (paper white point consistency)
  5. ela (error level analysis, self-disables on flat backgrounds)
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

    background_flattened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRONG = 0.60
ROBUST_CUES = {"anchor_removal", "pasted_region_geometry"}

WEIGHTS = {
    "anchor_removal":         0.40,
    "pasted_region_geometry": 0.30,
    "noise_void":             0.15,
    "colour_discontinuity":   0.08,
    "ela":                    0.07,
}


def _background_flatness(g: np.ndarray) -> float:
    k = np.ones((5, 5), np.float64) / 25.0
    m = cv2.filter2D(g, -1, k)
    v = cv2.filter2D((g - m) ** 2, -1, k)
    white = g > 235
    if white.sum() < 1000:
        return 0.0
    return float((v[white] < 0.5).mean())


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


def _cue_pasted_geometry(bgr: np.ndarray, anchor_result) -> Cue:
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

    from .document_spec import REGISTRY
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
        if lines is not None and len(lines) > 0:
            lines_2d = lines.reshape(-1, 4)
            for a1, b1, a2, b2 in lines_2d:
                if abs(a1 - a2) <= 1 or abs(b1 - b2) <= 1:
                    n_axis += 1

        interior = g[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.float64)
        flat = float(cv2.Laplacian(interior, cv2.CV_64F).var()) if interior.size else 0.0
        uniform = float((interior > 235).mean()) if interior.size else 0.0

        if n_axis >= 3 and flat < 400:
            s = 0.88
        elif uniform > 0.90 and flat < 200:
            s = 0.80
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


def _cue_colour_discontinuity(bgr: np.ndarray) -> Cue:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    gh, gw = L.shape
    step_y, step_x = max(gh // 10, 1), max(gw // 10, 1)

    casts = []
    for y in range(0, gh - step_y + 1, step_y):
        for x in range(0, gw - step_x + 1, step_x):
            tl = L[y:y + step_y, x:x + step_x]
            paper = tl > 230
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

    if mad < 0.05 or worst < 2.0:
        return Cue("colour_discontinuity", False, 0.0,
                   {"paper_tiles": int(len(casts)),
                    "worst_tile_deviation": round(worst, 3),
                    "mad": round(mad, 4)},
                   ["COLOUR_CUE_DISABLED_WHITE_POINT_CLAMPED_BY_SCANNER"])

    z = worst / (mad * 1.4826)
    score = float(np.clip((z - 6.0) / 10.0, 0.0, 1.0)) * \
        float(np.clip((worst - 2.0) / 4.0, 0.0, 1.0))
    return Cue("colour_discontinuity", True, round(score, 4),
               {"paper_tiles": int(len(casts)),
                "worst_tile_deviation": round(worst, 3),
                "robust_z": round(z, 2)},
               ["PAPER_WHITE_POINT_DISCONTINUITY"] if score >= STRONG else [])


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

    robust_strong = [c for c in r.strong_cues if c in ROBUST_CUES]
    r.corroborated = len(r.strong_cues) >= 2 and bool(robust_strong)

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
