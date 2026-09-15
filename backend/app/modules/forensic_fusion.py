"""
SENTINEL V3 Forensic Evidence Fusion Adapter
Delegates to forensic_fusion_v3 while supporting legacy V2 test signatures.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.modules.provenance import ProvenanceResult
from app.modules.qr_integrity import QRResult
from app.modules.texture_forensics import TextureResult
from app.modules.forensic_fusion_v3 import fuse as fuse_v3, FusionResult

logger = logging.getLogger(__name__)


def fuse_forensic_evidence(
    ai_detector_res: Dict[str, Any],
    frequency_res: Dict[str, Any],
    noise_res: Dict[str, Any],
    metadata_res: Dict[str, Any],
    tampering_res: Optional[Dict[str, Any]] = None,
    validity_res: Optional[Dict[str, Any]] = None,
    visual_authenticity_score: Optional[float] = None,
    provenance_res: Optional[Any] = None,
    qr_res: Optional[Any] = None,
    texture_res: Optional[Any] = None,
) -> Dict[str, Any]:
    tampering_res = tampering_res or {}
    validity_res = validity_res or {}
    ai_detector_res = ai_detector_res or {}
    frequency_res = frequency_res or {}
    noise_res = noise_res or {}
    metadata_res = metadata_res or {}

    # Build Provenance object if not passed
    if provenance_res is None:
        prov = ProvenanceResult()
        if metadata_res.get("no_camera_origin_evidence"):
            prov.synthetic_provenance_score = 0.65
            prov.evidence_available = True
            prov.reasons.append("CONTAINER_NOT_CAMERA_NATIVE")
        elif metadata_res.get("camera_tags_present") or metadata_res.get("camera_tag_count", 0) > 0:
            prov.synthetic_provenance_score = 0.0
            prov.has_optical_exif = True
            prov.evidence_available = True
            prov.camera_model = "Camera"
            prov.reasons.append("OPTICAL_CAPTURE_CHAIN_PRESENT")
        else:
            prov.evidence_available = False
        provenance_res = prov

    # Build QR object if not passed
    if qr_res is None:
        provenance_res_qr = QRResult()
    else:
        provenance_res_qr = qr_res

    # Build Texture object if not passed
    if texture_res is None:
        tex = TextureResult()
        freq_score = float(frequency_res.get("frequency_anomaly_score") or frequency_res.get("frequency_score") or 0.0) if frequency_res else 0.0
        noise_score = float(noise_res.get("noise_anomaly_score") or (0.7 if noise_res.get("suspiciously_smooth") else 0.0)) if noise_res else 0.0
        tex.frequency_anomaly = freq_score
        tex.smoothness_anomaly = noise_score
        tex.texture_ai_score = max(freq_score, noise_score)
        tex.evidence_available = bool((frequency_res and frequency_res.get("available", True)) or (noise_res and noise_res.get("available", True)))
        tex.reliability = 1.0
        texture_res = tex

    calibrated_specified = ("is_calibrated" in ai_detector_res) or ("calibrated" in ai_detector_res)
    calibrated_val = bool(ai_detector_res.get("is_calibrated", ai_detector_res.get("calibrated", True if not calibrated_specified else False)))
    loaded_val = bool(ai_detector_res.get("is_weights_loaded", ai_detector_res.get("model_loaded", ai_detector_res.get("model_ready", True))))

    learned_dict = {
        "global_probability": ai_detector_res.get("global_probability", ai_detector_res.get("ai_probability", 0.0)),
        "patch_topk_probability": ai_detector_res.get("patch_topk_probability", ai_detector_res.get("ai_probability", 0.0)),
        "is_weights_loaded": loaded_val,
        "is_calibrated": calibrated_val,
    }

    fusion_res = fuse_v3(
        provenance=provenance_res,
        qr=provenance_res_qr,
        texture=texture_res,
        learned=learned_dict,
        tampering=tampering_res,
    )

    ai_prob = fusion_res.ai_probability
    corroborated = fusion_res.corroborated or (fusion_res.strong_signal_count >= 2)
    strong_count = fusion_res.strong_signal_count

    if not loaded_val:
        ai_status = "UNAVAILABLE"
    elif not calibrated_val:
        ai_status = "EXCLUDED_UNCALIBRATED"
    elif corroborated and ai_prob >= 0.50:
        ai_status = "AI_GENERATED"
    elif ai_prob <= 0.35:
        ai_status = "LOW_AI_EVIDENCE"
    else:
        ai_status = "INCONCLUSIVE"

    ai_analysis = {
        "status": ai_status,
        "probability": ai_prob,
        "ai_probability": ai_prob,
        "global_probability": ai_detector_res.get("global_probability", ai_prob),
        "patch_topk_probability": ai_detector_res.get("patch_topk_probability", ai_prob),
        "frequency_anomaly": getattr(texture_res, "frequency_anomaly", 0.0),
        "noise_anomaly": getattr(texture_res, "smoothness_anomaly", 0.0),
        "strong_signal_count": strong_count,
        "strong_signals": fusion_res.strong_families,
        "corroborated": corroborated,
        "model_loaded": loaded_val,
        "is_weights_loaded": loaded_val,
        "trained": loaded_val,
        "calibrated": calibrated_val,
        "model_version": ai_detector_res.get("model_version", "convnext_base_ai_detector_v1"),
        "reasons": fusion_res.reasons,
    }

    fusion_dict = {
        "model_ready": loaded_val,
        "strong_signal_count": strong_count,
        "strong_signals": fusion_res.strong_families,
        "corroborated": corroborated,
        "ai_evidence": ai_prob,
        "tampering_evidence": fusion_res.tampering_probability,
        "genuine_evidence": fusion_res.genuine_evidence,
        "v3_fusion": fusion_res.to_dict(),
    }

    return {
        "ai_analysis": ai_analysis,
        "fusion": fusion_dict,
        "strong_signal_count": strong_count,
        "corroborated": corroborated,
        "v3_fusion_object": fusion_res,
    }


def fuse_synthetic_forensics(*args, **kwargs) -> Dict[str, Any]:
    ai_det = kwargs.get("ai_detector_res") or kwargs.get("ai_detector_result") or (args[0] if len(args) > 0 else {})
    freq = kwargs.get("frequency_res") or kwargs.get("frequency_result") or (args[1] if len(args) > 1 else {})
    noise = kwargs.get("noise_res") or kwargs.get("noise_result") or (args[2] if len(args) > 2 else {})
    meta = kwargs.get("metadata_res") or kwargs.get("metadata_result") or (args[3] if len(args) > 3 else {})
    tamp = kwargs.get("tampering_res") or kwargs.get("tampering_result") or kwargs.get("garbled_result") or (args[4] if len(args) > 4 else {})
    val = kwargs.get("validity_res") or kwargs.get("validity_result") or (args[5] if len(args) > 5 else {})
    vis_score = kwargs.get("visual_authenticity_score") or (args[6] if len(args) > 6 else None)

    res = fuse_forensic_evidence(
        ai_detector_res=ai_det,
        frequency_res=freq,
        noise_res=noise,
        metadata_res=meta,
        tampering_res=tamp,
        validity_res=val,
        visual_authenticity_score=vis_score,
    )
    ai = res["ai_analysis"]
    strong_count = res["strong_signal_count"]
    corroborated = res["corroborated"]

    if (corroborated or strong_count >= 2) and ai["probability"] >= 0.50:
        prediction = "likely_ai_generated"
    elif strong_count == 1:
        prediction = "suspicious"
    elif ai["status"] == "LOW_AI_EVIDENCE" or ai["probability"] <= 0.35:
        prediction = "likely_genuine"
    else:
        prediction = "inconclusive"

    synth_score = max(ai["probability"], 0.65) if (corroborated or strong_count >= 2) else (ai["probability"] if ai["probability"] is not None else 0.0)

    res["prediction"] = prediction
    res["synthetic_score"] = round(synth_score, 4)
    res["synthetic_probability"] = round(synth_score, 4)
    res["status"] = ai["status"]
    res["confidence"] = None
    res["reasons"] = ai["reasons"]
    res["is_corroborated"] = corroborated
    res["strong_signal_count"] = strong_count
    res["signals"] = {
        "ai_probability": ai["probability"] if ai["probability"] is not None else 0.0,
        "frequency_score": ai["frequency_anomaly"],
        "noise_score": ai["noise_anomaly"],
        "metadata_score": 0.0,
    }
    return res
