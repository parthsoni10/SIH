#!/usr/bin/env python3
"""Run the V3 chain on reference images and print the verdicts."""
import pathlib
import sys
import cv2
import numpy as np

# Ensure backend root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.modules.provenance import analyse_provenance
from app.modules.texture_forensics import analyse_texture
from app.modules.qr_integrity import analyse_qr
from app.modules.forensic_fusion_v3 import fuse
from app.modules.decision_engine_v3 import decide

# Simulating the V2 detector state observed in production.
LEARNED_STATE_UNCALIBRATED = {
    "global_probability": 0.557,
    "patch_topk_probability": 0.556,
    "is_weights_loaded": True,
    "is_calibrated": False,          # <-- the observed dashboard state
}

LEARNED_STATE_CALIBRATED = {
    "global_probability": 0.88,
    "patch_topk_probability": 0.89,
    "is_weights_loaded": True,
    "is_calibrated": True,
}

OCR_FIELDS = {"document_number": "QQUPS4464F", "dob": "04/01/2005"}


def run_reference_verification():
    # Synthetic PNG image buffer (simulating tensor render)
    synth_img = np.ones((800, 1200, 3), dtype=np.uint8) * 200
    cv2.putText(synth_img, "PAN CARD", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(synth_img, "QQUPS4464F", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    _, synth_bytes = cv2.imencode(".png", synth_img)
    synth_raw = synth_bytes.tobytes()

    # Camera JPEG image buffer (simulating optical photo)
    real_img = np.random.randint(50, 200, (800, 1200, 3), dtype=np.uint8)
    cv2.putText(real_img, "PAN CARD", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(real_img, "QQUPS4464F", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    _, real_bytes = cv2.imencode(".jpg", real_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    real_raw = real_bytes.tobytes()

    cases = [
        ("SYNTHETIC (PNG render, uncalibrated model)", synth_raw, synth_img, LEARNED_STATE_UNCALIBRATED),
        ("GENUINE (JPEG photo, uncalibrated model)", real_raw, real_img, LEARNED_STATE_UNCALIBRATED),
    ]

    for label, raw, bgr, learned_state in cases:
        prov = analyse_provenance(raw)
        tex = analyse_texture(bgr)
        qr = analyse_qr(bgr, "PAN Card", OCR_FIELDS)

        f = fuse(prov, qr, tex, learned_state, {"probability": 0.065})

        q_score = 0.85
        d = decide(fusion=f,
                   document_validity={"score": 1.0, "failed_rules": []},
                   image_quality={"score": q_score,
                                  "quality_too_low_for_forensics": False})

        print("=" * 72)
        print(label)
        print("=" * 72)
        print(f"  provenance synth score : {prov.synthetic_provenance_score:.3f}"
              f"   ({prov.container}/{prov.mode}, exif={prov.has_exif},"
              f" optical={prov.has_optical_exif})")
        print(f"  texture    available   : {tex.evidence_available}"
              f"  rel={tex.reliability:.2f}  score={tex.texture_ai_score:.3f}")
        print(f"  qr         available   : {qr.evidence_available}"
              f"  decoded={qr.decoded}  synth={qr.synthetic_qr_score:.2f}")
        print(f"  learned    available   : {f.families['LEARNED']['available']}"
              f"  (excluded={not f.families['LEARNED']['available']})")
        print(f"  -> families available  : {f.available_family_count}/4")
        print(f"  -> fused AI prob       : {f.ai_probability:.3f}"
              f"   conf={f.ai_confidence:.3f}")
        print(f"  -> strong families     : {f.strong_families or 'none'}"
              f"   corroborated={f.corroborated}  anchor={f.anchor_present}")
        print(f"  -> genuine evidence    : {f.genuine_evidence:.3f}")
        print()
        print(f"  VERDICT  : {d.status}   [{d.evidence_state}]"
              f"  confidence={d.confidence}")
        print(f"  REASONS  : {', '.join(d.reason_codes)}")
        if d.blocking_gaps:
            print(f"  GAPS     : {', '.join(d.blocking_gaps)}")
        print(f"  ACTION   : {d.officer_action}")
        print()


if __name__ == "__main__":
    run_reference_verification()
