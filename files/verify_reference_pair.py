#!/usr/bin/env python3
"""Run the V3 chain on the two reference images and print the verdicts."""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from modules.provenance import analyse_provenance
from modules.texture_forensics import analyse_texture
from modules.qr_integrity import analyse_qr
from modules.forensic_fusion_v3 import fuse
from modules.decision_engine_v3 import decide

CASES = [
    ("GENUINE  (iPhone capture)", "/mnt/user-data/uploads/IMG_7926_JPG.jpeg"),
    ("SYNTHETIC (Gemini render)", "/mnt/user-data/uploads/Gemini_Generated_Image_r7ksznr7ksznr7ks.png"),
]

# Simulating the V2 detector state observed in production.
LEARNED_STATE = {
    "global_probability": 0.557,
    "patch_topk_probability": 0.556,
    "is_weights_loaded": True,
    "is_calibrated": False,          # <-- the observed dashboard state
}

OCR_FIELDS = {"document_number": "QQUPS4464F", "dob": "04/01/2005"}

for label, path in CASES:
    raw = pathlib.Path(path).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    prov = analyse_provenance(raw)
    tex = analyse_texture(bgr)
    qr = analyse_qr(bgr, "PAN Card", OCR_FIELDS)

    f = fuse(prov, qr, tex, LEARNED_STATE, {"probability": 0.065})

    q_score = 0.58   # matches the dashboard reading
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
          f"  rel={tex.reliability:.2f}  score={tex.texture_ai_score:.3f}"
          f"  (localvar={tex.local_variance_median:.2f},"
          f" hf={tex.hf_lf_energy_ratio:.2f})")
    print(f"  qr         available   : {qr.evidence_available}"
          f"  decoded={qr.decoded}  synth={qr.synthetic_qr_score:.2f}")
    print(f"  learned    available   : {f.families['LEARNED']['available']}"
          f"  (excluded: uncalibrated)")
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
