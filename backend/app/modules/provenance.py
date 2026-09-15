"""
SENTINEL V3 — Stage 3b: Container & Provenance Forensics
=========================================================
This is the single highest-AUC recapture-robust signal for the ID-card domain.

Rationale
---------
A genuine border-checkpoint capture is a photograph taken by a physical camera.
It carries an irreversible provenance chain: camera Make/Model, exposure
triplet, an ICC profile, JFIF/MPO structure, and a camera-specific JPEG
quantisation table.

A diffusion-model render (Gemini / Midjourney / SDXL / Flux) is serialised
straight out of a tensor. It has NO camera chain. Even when an attacker strips
EXIF from a real photo, they cannot *fabricate* a self-consistent one, and the
absence pattern differs: a stripped JPEG keeps its quantisation table and
chroma subsampling; a PNG render has neither.

Measured on the reference pair:
    real  -> MPO / RGB  / Make=Apple Model=iPhone 14 Pro / ICC + XMP + MPO
    gemini-> PNG / RGBA / no EXIF whatsoever / no ancillary chunks

IMPORTANT: absence of provenance is NOT proof of synthesis (WhatsApp strips
EXIF, scanners produce bare TIFFs). It is scored as evidence, and the fusion
layer requires corroboration from a different signal family before it can
drive an AI_GENERATED verdict. See forensic_fusion_v3.py.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field, asdict
from typing import Any

from PIL import Image, ImageFile
from PIL.ExifTags import TAGS

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Containers a physical camera or scanner actually emits.
CAMERA_CONTAINERS = {"JPEG", "MPO", "HEIF", "HEIC", "TIFF", "DNG", "WEBP"}
# Containers a generative model emits by default.
SYNTH_CONTAINERS = {"PNG", "BMP"}

# EXIF tags that only exist if a real optical sensor produced the frame.
OPTICAL_TAGS = {
    "Make", "Model", "LensModel", "LensMake", "FNumber", "ExposureTime",
    "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength",
    "FocalLengthIn35mmFilm", "ShutterSpeedValue", "ApertureValue",
    "ExposureBiasValue", "MeteringMode", "Flash", "SubjectDistance",
    "WhiteBalance", "DigitalZoomRatio", "SceneCaptureType",
}

# Software strings that indicate a render pipeline or an editor, not a camera.
GENERATOR_SOFTWARE = (
    "gemini", "imagen", "dall", "midjourney", "stable diffusion", "sdxl",
    "comfyui", "automatic1111", "invokeai", "flux", "firefly", "nano banana",
    "openai", "runway", "leonardo", "ideogram", "grok",
)
EDITOR_SOFTWARE = (
    "photoshop", "gimp", "canva", "pixelmator", "affinity", "lightroom",
    "snapseed", "picsart", "inkscape", "paint.net", "krita", "figma",
)

# Diffusion models sample on a latent lattice; the decoded side is a multiple
# of 8 (usually 16 or 64). Real sensors have idiosyncratic dimensions.
LATENT_MULTIPLES = (64, 32, 16, 8)


@dataclass
class ProvenanceResult:
    evidence_available: bool = True

    container: str = ""
    mode: str = ""
    width: int = 0
    height: int = 0

    has_exif: bool = False
    has_optical_exif: bool = False
    optical_tag_count: int = 0
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None

    has_icc_profile: bool = False
    has_jfif: bool = False
    has_mpo: bool = False
    has_xmp: bool = False
    has_c2pa_hint: bool = False

    has_alpha: bool = False
    latent_aligned: bool = False
    latent_multiple: int = 0

    quant_table_present: bool = False
    chroma_subsampling: str | None = None

    # 0.0 = fully camera-consistent, 1.0 = fully render-consistent
    synthetic_provenance_score: float = 0.0
    # 0.0 = untouched, 1.0 = editor fingerprint present
    editor_provenance_score: float = 0.0

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _latent_alignment(w: int, h: int) -> tuple[bool, int]:
    for m in LATENT_MULTIPLES:
        if w % m == 0 and h % m == 0:
            return (m >= 16), m
    return False, 0


def analyse_provenance(image_bytes: bytes) -> ProvenanceResult:
    """Inspect the raw uploaded bytes. Never touch the decoded pixel array
    here -- re-encoding destroys exactly the evidence we want."""
    r = ProvenanceResult()

    try:
        im = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # pragma: no cover
        r.evidence_available = False
        r.reasons.append(f"PROVENANCE_UNREADABLE:{type(exc).__name__}")
        return r

    r.container = (im.format or "UNKNOWN").upper()
    r.mode = im.mode
    r.width, r.height = im.size
    r.has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info

    info = im.info or {}
    r.has_icc_profile = bool(info.get("icc_profile"))
    r.has_jfif = "jfif" in info
    r.has_mpo = "mp" in info or r.container == "MPO"
    r.has_xmp = bool(info.get("xmp")) or "XML:com.adobe.xmp" in info
    r.has_c2pa_hint = any(k.lower().startswith(("c2pa", "jumbf")) for k in info)

    # ---- EXIF ------------------------------------------------------------
    tags: dict[str, Any] = {}
    try:
        exif = im.getexif()
        if exif:
            for k, v in exif.items():
                tags[TAGS.get(k, str(k))] = v
            try:  # the Exif IFD holds the optical parameters
                for k, v in exif.get_ifd(0x8769).items():
                    tags[TAGS.get(k, str(k))] = v
            except Exception:
                pass
    except Exception:
        pass

    r.has_exif = bool(tags)
    optical = [t for t in tags if t in OPTICAL_TAGS]
    r.optical_tag_count = len(optical)
    r.has_optical_exif = len(optical) >= 3

    r.camera_make = str(tags.get("Make", "")).strip() or None
    r.camera_model = str(tags.get("Model", "")).strip() or None
    r.software = str(tags.get("Software", "")).strip() or None

    # ---- JPEG-internal structure ----------------------------------------
    try:
        r.quant_table_present = bool(getattr(im, "quantization", None))
        sub = getattr(im, "layer", None)
        if sub:
            r.chroma_subsampling = str(sub)
    except Exception:
        pass

    r.latent_aligned, r.latent_multiple = _latent_alignment(r.width, r.height)

    # ---- Scoring ---------------------------------------------------------
    # Weights are additive evidence, clamped at the end. They were chosen so
    # that no single weak indicator can cross the 0.60 "strong" bar alone.
    score = 0.0

    if r.container in SYNTH_CONTAINERS:
        score += 0.30
        r.reasons.append("CONTAINER_NOT_CAMERA_NATIVE")
    elif r.container in CAMERA_CONTAINERS:
        score -= 0.10

    if not r.has_exif:
        score += 0.28
        r.reasons.append("EXIF_BLOCK_ABSENT")
    elif not r.has_optical_exif:
        score += 0.14
        r.reasons.append("EXIF_PRESENT_BUT_NO_OPTICAL_PARAMS")
    else:
        score -= 0.30
        r.reasons.append("OPTICAL_CAPTURE_CHAIN_PRESENT")

    if r.camera_make and r.camera_model:
        score -= 0.20
        r.reasons.append("CAMERA_IDENTITY_PRESENT")

    if not r.has_icc_profile and r.container in SYNTH_CONTAINERS:
        score += 0.08
        r.reasons.append("NO_COLOUR_PROFILE")

    if r.has_alpha:
        # A photograph has no alpha channel. A tensor dump frequently does.
        score += 0.12
        r.reasons.append("ALPHA_CHANNEL_ON_OPAQUE_DOCUMENT")

    if r.latent_aligned and not r.has_optical_exif:
        score += 0.10
        r.reasons.append(f"LATENT_ALIGNED_DIMENSIONS_x{r.latent_multiple}")

    if r.container == "JPEG" and not r.quant_table_present:
        score += 0.06
        r.reasons.append("NO_JPEG_QUANTISATION_HISTORY")

    sw = (r.software or "").lower()
    if any(g in sw for g in GENERATOR_SOFTWARE):
        score += 0.55
        r.reasons.append("GENERATOR_SOFTWARE_TAG")
    if r.has_c2pa_hint:
        score += 0.35
        r.reasons.append("C2PA_PROVENANCE_MANIFEST_PRESENT")

    r.synthetic_provenance_score = max(0.0, min(1.0, score))

    # ---- Editor (tampering) provenance, kept on a separate axis ---------
    edit = 0.0
    if any(e in sw for e in EDITOR_SOFTWARE):
        edit += 0.60
        r.reasons.append("EDITOR_SOFTWARE_TAG")
    if r.has_exif and "DateTime" in tags and "DateTimeOriginal" in tags:
        if str(tags["DateTime"]) != str(tags["DateTimeOriginal"]):
            edit += 0.25
            r.reasons.append("EXIF_TIMESTAMP_DIVERGENCE")
    r.editor_provenance_score = max(0.0, min(1.0, edit))

    return r
