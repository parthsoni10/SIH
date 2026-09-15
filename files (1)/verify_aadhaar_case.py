#!/usr/bin/env python3
"""Reproduce verification #207 and confirm each bug is now caught."""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from modules.document_spec import get_spec, SIDE_BACK, SIDE_FRONT
from modules.anchor_verifier import verify_anchors
from modules.alteration_detector import detect_alteration
from modules.field_resolver import resolve_structure
from modules.provenance import analyse_provenance
from modules.texture_forensics import analyse_texture
from modules.qr_integrity import analyse_qr
from modules.forensic_fusion_v3 import fuse
from modules.decision_engine_v31 import decide

U = "/mnt/user-data/uploads/"
CASES = [
    ("GENUINE Aadhaar", U + "WhatsApp_Image_2026-09-15_at_15_13_10.jpeg"),
    ("ALTERED Aadhaar (logo erased)", U + "WhatsApp_Image_2026-09-15_at_15_32_34.jpeg"),
]

LEARNED = {"global_probability": 0.484, "patch_topk_probability": 0.535,
           "is_weights_loaded": True, "is_calibrated": False}

# OCR lines as #207 actually saw them, with normalised boxes.
# The issue date is VERTICAL on the left edge -- that is the trap.
OCR_LINES = [
    {"text": "Government of India", "conf": 0.94, "x0": .30, "y0": .09, "x1": .59, "y1": .16},
    {"text": "Issue Date : 03/01/2014", "conf": 0.80, "x0": .020, "y0": .26, "x1": .055, "y1": .72},
    {"text": "Dhruv Prajapati", "conf": 0.91, "x0": .34, "y0": .27, "x1": .54, "y1": .32},
    {"text": "DOB : 17/08/2005", "conf": 0.88, "x0": .34, "y0": .34, "x1": .74, "y1": .39},
    {"text": "Male", "conf": 0.90, "x0": .34, "y0": .40, "x1": .48, "y1": .45},
    {"text": "2778 3746 3380", "conf": 0.92, "x0": .31, "y0": .76, "x1": .69, "y1": .84},
]

spec = get_spec("Aadhaar")

for label, path in CASES:
    raw = pathlib.Path(path).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    anchors = verify_anchors(bgr, spec, SIDE_FRONT, warped=False)
    alt = detect_alteration(bgr, anchors)
    prov = analyse_provenance(raw)
    tex = analyse_texture(bgr)
    qr = analyse_qr(bgr, "Aadhaar", {"document_number": "277837463380"})
    f = fuse(prov, qr, tex, LEARNED, {"probability": alt.probability})

    # Front side only -- exactly how #207 was submitted.
    struct = resolve_structure(spec=spec, lines=OCR_LINES,
                               sides_present={SIDE_FRONT},
                               extra_fields={"document_number": "277837463380",
                                             "gender": "Male"},
                               checksum_valid=True, layout_score=0.85)

    d = decide(fusion=f, structure=struct, alteration=alt,
               image_quality={"score": 0.72})

    print("=" * 74)
    print(label)
    print("=" * 74)
    print(f"  anchors : {anchors.anchors_checked} checked, "
          f"missing={anchors.missing_anchors or 'none'}, "
          f"completeness={anchors.template_completeness}")
    for fd in anchors.findings:
        flag = "" if fd["present"] else "   <-- MISSING"
        print(f"      {fd['key']:20s} ink={fd['ink_coverage']:.4f} "
              f"(need {fd['required_ink']:.4f}){flag}")
    print(f"  alteration: prob={alt.probability:.3f} corroborated={alt.corroborated} "
          f"strong={alt.strong_cues or 'none'} cues_avail={alt.available_cue_count}/5")
    print(f"      bg_flattened={alt.background_flattened} "
          f"(noise_void + ELA self-disabled)")
    print(f"  structure : validity={struct.validity_score} "
          f"complete={struct.submission_complete} "
          f"missing_sides={struct.sides_missing}")
    print(f"      dob resolved -> {struct.fields.get('dob',{}).get('value')} "
          f"[{struct.fields.get('dob',{}).get('source')}]")
    print(f"      issue_date   -> {struct.fields.get('issue_date',{}).get('value')}")
    print(f"      full_name    -> {struct.fields.get('full_name',{}).get('value')}")
    print()
    print(f"  VERDICT : {d.status}  [{d.evidence_state}]  conf={d.confidence}")
    print(f"  REASONS : {', '.join(d.reason_codes[:6])}")
    if d.blocking_gaps:
        print(f"  GAPS    : {', '.join(d.blocking_gaps)}")
    print(f"  ACTION  : {d.officer_action}")
    print()
