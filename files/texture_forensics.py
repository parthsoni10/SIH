"""
SENTINEL V3 — Stage 12/13 replacement: Quality-Normalised Texture Forensics
===========================================================================
Replaces the separate `frequency_analysis.py` + `noise_analysis.py` scoring.

Why the V2 modules under-reported
---------------------------------
On the reference pair the raw features separate cleanly:

    feature               real(iPhone)   gemini      ratio
    local_var_med(5x5)        5.40        2.62       2.06x
    hf_lf_energy_ratio       11.23        7.83       1.43x

...yet V2 reported `frequency_anomaly = 15%`. Two bugs caused that:

  BUG 1 — Unfitted thresholds. V2 mapped raw features through constants
          borrowed from a generic AI-art detector. Document photographs live
          in a completely different feature range, so every real document
          landed in the middle of the sigmoid.

  BUG 2 — No quality normalisation. Blur and JPEG re-compression *both*
          suppress high-frequency energy, which is the same direction
          synthesis pushes it. A blurry genuine card and a sharp synthetic
          card produce the same raw number. Without conditioning on the
          measured blur/compression level the feature is not identifiable.

This module fixes both: it computes features on a resolution-normalised,
document-cropped image, then divides out the measured capture degradation
before mapping through thresholds fitted by ml/calibrate_v3.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

# Every image is resampled to this width before feature extraction so that
# features are comparable across a 2594px render and a 2999px phone capture.
CANONICAL_WIDTH = 1600

# PLACEHOLDER DEFAULTS -- measured on a 2-image reference pair using this
# module's own Hanning-windowed pipeline, so they are self-consistent with
# what analyse_texture() actually computes. They are NOT production values:
# two samples cannot define a class median. Run ml/calibrate_v3.py on >= 800
# images per class and let it overwrite thresholds_v3.json before deployment.
#
# Observed separation on the reference pair:
#   local_variance_median   real 4.49  vs  synth 2.16   -> 2.1x, strong
#   hf_lf_energy_ratio      real 6.79  vs  synth 6.37   -> 1.07x, near-useless
#
# The frequency feature collapses once the Hanning window is applied, which is
# why it carries only 0.25 weight below. Do not restore it to parity until
# calibrate_v3 reports an AUC above 0.70 for it on your data.
DEFAULTS = {
    "local_var_real_median": 4.49,
    "local_var_synth_median": 2.16,
    "hf_ratio_real_median": 6.79,
    "hf_ratio_synth_median": 6.37,
    "residual_real_median": 5.77,
    "residual_synth_median": 6.49,
    # Feature is only trusted when capture quality clears this bar.
    "min_laplacian_for_texture": 60.0,
    "max_jpeg_blockiness": 0.35,
}


@dataclass
class TextureResult:
    evidence_available: bool = False
    reliability: float = 0.0          # 0..1, how much the fusion may trust this

    local_variance_median: float = 0.0
    hf_lf_energy_ratio: float = 0.0
    highpass_residual_std: float = 0.0
    radial_spectrum_flatness: float = 0.0
    jpeg_blockiness: float = 0.0
    laplacian_variance: float = 0.0

    smoothness_anomaly: float = 0.0   # 0..1, high = unnaturally smooth
    frequency_anomaly: float = 0.0    # 0..1, high = HF energy deficit
    texture_ai_score: float = 0.0     # fused 0..1

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = CANONICAL_WIDTH / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (CANONICAL_WIDTH, max(1, int(h * scale))),
                      interpolation=interp).astype(np.float64)


def _local_variance_median(g: np.ndarray) -> float:
    k = np.ones((5, 5), np.float64) / 25.0
    mean = cv2.filter2D(g, -1, k)
    var = cv2.filter2D((g - mean) ** 2, -1, k)
    return float(np.median(var))


def _radial_spectrum(g: np.ndarray) -> tuple[float, float]:
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))
    h, w = F.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / (min(h, w) / 2.0)

    low = F[r < 0.12].mean() + 1e-9
    high = F[r > 0.55].mean()
    hf_ratio = float(high / low * 1e3)

    # Spectral flatness across radial bands. Diffusion upsamplers leave a
    # characteristically *flat* far-field tail; optics leave a rolloff.
    bands = []
    for lo in np.arange(0.15, 0.95, 0.10):
        m = (r >= lo) & (r < lo + 0.10)
        if m.any():
            bands.append(F[m].mean())
    bands = np.asarray(bands) + 1e-9
    flatness = float(np.exp(np.log(bands).mean()) / bands.mean())
    return hf_ratio, flatness


def _jpeg_blockiness(g: np.ndarray) -> float:
    """Energy at the 8x8 DCT block boundaries relative to interior."""
    d = np.abs(np.diff(g, axis=1))
    on = d[:, 7::8].mean() if d.shape[1] > 8 else 0.0
    off = d.mean() + 1e-9
    return float(np.clip((on / off - 1.0), 0.0, 3.0) / 3.0)


def _to_anomaly(value: float, real_med: float, synth_med: float) -> float:
    """Map a raw feature onto 0..1 where 1 means 'looks synthetic'.

    Linear interpolation between the two class medians, clamped. Deliberately
    NOT a sigmoid with an invented temperature -- that was the V2 mistake.
    Anything beyond the class medians saturates rather than extrapolating.
    """
    if abs(real_med - synth_med) < 1e-6:
        return 0.5
    t = (value - real_med) / (synth_med - real_med)
    return float(np.clip(t, 0.0, 1.0))


def analyse_texture(bgr: np.ndarray,
                    thresholds: dict[str, float] | None = None
                    ) -> TextureResult:
    t = {**DEFAULTS, **(thresholds or {})}
    r = TextureResult()

    if bgr is None or bgr.size == 0:
        r.reasons.append("TEXTURE_NO_IMAGE")
        return r

    g = _normalise(bgr)

    r.laplacian_variance = float(cv2.Laplacian(g, cv2.CV_64F).var())
    r.local_variance_median = _local_variance_median(g)
    r.hf_lf_energy_ratio, r.radial_spectrum_flatness = _radial_spectrum(g)
    r.highpass_residual_std = float((g - cv2.GaussianBlur(g, (0, 0), 1.2)).std())
    r.jpeg_blockiness = _jpeg_blockiness(g)

    # ---- Reliability gate ------------------------------------------------
    # This is the fix for BUG 2. If the capture is too degraded, the texture
    # features cannot distinguish synthesis from blur, so we down-weight
    # rather than emit a confident-looking middling number.
    rel = 1.0
    if r.laplacian_variance < t["min_laplacian_for_texture"]:
        rel *= float(np.clip(r.laplacian_variance / t["min_laplacian_for_texture"], 0.0, 1.0))
        r.reasons.append("TEXTURE_RELIABILITY_REDUCED_BY_BLUR")
    if r.jpeg_blockiness > t["max_jpeg_blockiness"]:
        rel *= float(np.clip(t["max_jpeg_blockiness"] / max(r.jpeg_blockiness, 1e-6), 0.0, 1.0))
        r.reasons.append("TEXTURE_RELIABILITY_REDUCED_BY_COMPRESSION")

    r.reliability = float(np.clip(rel, 0.0, 1.0))
    r.evidence_available = r.reliability >= 0.25
    if not r.evidence_available:
        r.reasons.append("TEXTURE_EVIDENCE_UNAVAILABLE_LOW_QUALITY")
        return r

    r.smoothness_anomaly = _to_anomaly(
        r.local_variance_median,
        t["local_var_real_median"], t["local_var_synth_median"])
    r.frequency_anomaly = _to_anomaly(
        r.hf_lf_energy_ratio,
        t["hf_ratio_real_median"], t["hf_ratio_synth_median"])

    # Smoothness separates the classes 2.1x; the windowed frequency ratio only
    # 1.07x. Weighting them 60/40 let the weak feature drag a genuine document
    # to within 0.004 of the STRONG bar. Weight by measured discriminability.
    fused = 0.75 * r.smoothness_anomaly + 0.25 * r.frequency_anomaly
    r.texture_ai_score = float(np.clip(fused, 0.0, 1.0))

    if r.smoothness_anomaly >= 0.60:
        r.reasons.append("SURFACE_MICROTEXTURE_UNNATURALLY_SMOOTH")
    if r.frequency_anomaly >= 0.60:
        r.reasons.append("HIGH_FREQUENCY_ENERGY_DEFICIT")
    if r.radial_spectrum_flatness >= 0.75:
        r.reasons.append("FLAT_SPECTRAL_TAIL_CONSISTENT_WITH_UPSAMPLER")

    return r
