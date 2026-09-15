"""
SENTINEL V3 Calibrated Decision Engine Adapter
Delegates to decision_engine_v3 while maintaining full backwards compatibility.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from app.modules.decision_engine_v3 import decide as decide_v3, Decision
from app.modules.forensic_fusion_v3 import FusionResult

logger = logging.getLogger(__name__)


def evaluate_final_decision(
    *args,
    ai_analysis: Optional[Dict[str, Any]] = None,
    tampering_analysis: Optional[Dict[str, Any]] = None,
    document_validity: Optional[Dict[str, Any]] = None,
    image_quality: Optional[Dict[str, Any]] = None,
    blacklist_hit: bool = False,
    fusion: Optional[Any] = None,
) -> Dict[str, Any]:
    # Extract positional/legacy args if present
    if len(args) >= 4 and isinstance(args[0], dict) and ("risk_score" in args[0] or "risk_tier" in args[0]):
        _synthetic = args[1] if isinstance(args[1], dict) else {}
        _tampering = args[2] if isinstance(args[2], dict) else {}
        _validation = args[3] if isinstance(args[3], dict) else {}
        ai_analysis = {
            "probability": _synthetic.get("ai_probability", _synthetic.get("synthetic_generation_score", 0.0)),
            "is_calibrated": _synthetic.get("calibrated", False),
            "is_weights_loaded": True,
            "corroborated": _synthetic.get("is_corroborated", False),
            "strong_signal_count": _synthetic.get("strong_signal_count", 0),
        }
        tampering_analysis = {"probability": _tampering.get("tampering_score", _tampering.get("tampering_probability", 0.0))}
        document_validity = {"score": _validation.get("validation_pass_rate", 1.0)}
        image_quality = {"score": 0.85, "quality_too_low_for_forensics": False}
        if len(args) >= 5:
            blacklist_hit = bool(args[4])
    else:
        ai_analysis = ai_analysis or (args[0] if len(args) > 0 and isinstance(args[0], dict) else {})
        tampering_analysis = tampering_analysis or (args[1] if len(args) > 1 and isinstance(args[1], dict) else {})
        document_validity = document_validity or (args[2] if len(args) > 2 and isinstance(args[2], dict) else {})
        image_quality = image_quality or (args[3] if len(args) > 3 and isinstance(args[3], dict) else {})
        if len(args) >= 5 and isinstance(args[4], bool):
            blacklist_hit = args[4]

    # Build FusionResult object if fusion not directly passed
    if fusion is None or not isinstance(fusion, FusionResult):
        f = FusionResult()
        f.ai_probability = float(ai_analysis.get("probability", ai_analysis.get("ai_probability", 0.0)) or 0.0)
        f.tampering_probability = float(tampering_analysis.get("probability", tampering_analysis.get("tampering_probability", 0.0)) or 0.0)
        f.corroborated = bool(ai_analysis.get("corroborated", False))
        f.strong_signal_count = int(ai_analysis.get("strong_signal_count", 0))
        f.genuine_evidence = float(ai_analysis.get("genuine_evidence", 0.0))

        calibrated = bool(ai_analysis.get("calibrated", ai_analysis.get("is_calibrated", False)))
        loaded = bool(ai_analysis.get("model_loaded", ai_analysis.get("is_weights_loaded", True)))

        if calibrated and loaded:
            f.available_family_count = 2
            f.ai_confidence = 0.80
        else:
            f.available_family_count = 1 if loaded else 0
            f.ai_confidence = 0.30

        fusion = f

    decision_res: Decision = decide_v3(
        fusion=fusion,
        document_validity=document_validity or {"score": 1.0},
        image_quality=image_quality or {"score": 0.85},
        blacklist_hit=blacklist_hit,
    )

    result_dict = decision_res.to_dict()
    rc = result_dict.get("reason_codes", [])

    # Legacy reason code aliases
    if ai_analysis and not bool(ai_analysis.get("is_weights_loaded", ai_analysis.get("model_loaded", True))):
        if "MODEL_NOT_READY" not in rc:
            rc.append("MODEL_NOT_READY")
    if "BLACKLIST_WATCHLIST_MATCH" in rc and "BLACKLISTED_ID_MATCH" not in rc:
        rc.append("BLACKLISTED_ID_MATCH")
    if "AI_GENERATION_CORROBORATED" in rc and "SYNTHETIC_AI_IMAGE_CORROBORATED" not in rc:
        rc.append("SYNTHETIC_AI_IMAGE_CORROBORATED")
    if any("AUTHENTICITY_CONFIRMED" in code for code in rc) and "AUTHENTICITY_CONFIRMED" not in rc:
        rc.append("AUTHENTICITY_CONFIRMED")

    result_dict["final_decision"] = decision_res.status
    result_dict["decision_confidence"] = decision_res.confidence
    return result_dict
