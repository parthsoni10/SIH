import pytest
import numpy as np
import cv2

from app.modules.document_spec import get_spec, AADHAAR, PAN, PASSPORT, VISA, DRIVING_LICENSE, OTHER, SIDE_FRONT, SIDE_BACK
from app.modules.anchor_verifier import verify_anchors
from app.modules.alteration_detector import detect_alteration
from app.modules.field_resolver import resolve_dates, resolve_name, resolve_structure
from app.modules.decision_engine_v31 import decide, Decision
from app.modules.forensic_fusion_v3 import FusionResult

def test_document_specs_registry():
    assert get_spec("Aadhaar").doc_type == "Aadhaar"
    assert get_spec("PAN Card").doc_type == "PAN Card"
    assert get_spec("Passport").doc_type == "Passport"
    assert get_spec("Visa").doc_type == "Visa"
    assert get_spec("Driving License").doc_type == "Driving License"
    assert get_spec("UnknownDoc").is_fallback is True

def test_anchor_verifier_erased_logo():
    # Create white canvas (representing erased artwork)
    white_img = np.full((1073, 1600, 3), 255, dtype=np.uint8)
    spec = AADHAAR
    res = verify_anchors(white_img, spec, side=SIDE_FRONT, warped=False)

    assert res.evidence_available is True
    assert len(res.missing_anchors) > 0
    assert res.artwork_removal_score >= 0.70

def test_date_resolution_vertical_issue_date():
    lines = [
        {"text": "Government of India", "conf": 0.94, "x0": .30, "y0": .09, "x1": .59, "y1": .16},
        {"text": "Issue Date : 03/01/2014", "conf": 0.80, "x0": .020, "y0": .26, "x1": .055, "y1": .72}, # Vertical text (h/w > 2.5)
        {"text": "DOB : 17/08/2005", "conf": 0.88, "x0": .34, "y0": .34, "x1": .74, "y1": .39},
    ]
    spec = AADHAAR
    dates = resolve_dates(lines, spec)

    assert "dob" in dates
    assert dates["dob"].value == "17/08/2005"
    assert dates["dob"].source == "LABEL_PROXIMITY"
    assert dates["issue_date"].value == "03/01/2014"

def test_name_resolution_unlabelled_aadhaar_line():
    lines = [
        {"text": "Government of India", "conf": 0.94, "x0": .30, "y0": .09, "x1": .59, "y1": .16},
        {"text": "Dhruv Prajapati", "conf": 0.91, "x0": .34, "y0": .27, "x1": .54, "y1": .32},
        {"text": "DOB : 17/08/2005", "conf": 0.88, "x0": .34, "y0": .34, "x1": .74, "y1": .39},
    ]
    spec = AADHAAR
    nm = resolve_name(lines, spec)

    assert nm.value == "Dhruv Prajapati"
    assert nm.source == "POSITIONAL"

def test_incomplete_submission_front_only_aadhaar():
    lines = [
        {"text": "Government of India", "conf": 0.94, "x0": .30, "y0": .09, "x1": .59, "y1": .16},
        {"text": "Dhruv Prajapati", "conf": 0.91, "x0": .34, "y0": .27, "x1": .54, "y1": .32},
        {"text": "DOB : 17/08/2005", "conf": 0.88, "x0": .34, "y0": .34, "x1": .74, "y1": .39},
        {"text": "Male", "conf": 0.90, "x0": .34, "y0": .40, "x1": .48, "y1": .45},
    ]
    spec = AADHAAR
    struct = resolve_structure(
        spec=spec,
        lines=lines,
        sides_present={SIDE_FRONT},
        extra_fields={"document_number": "277837463380"},
        checksum_valid=True
    )

    assert struct.submission_complete is False
    assert SIDE_BACK in struct.sides_missing

    fusion = FusionResult(ai_probability=0.05, genuine_evidence=0.85, available_family_count=2)
    class CleanAltMock:
        probability = 0.0
        corroborated = False
        strong_cues = []
        cues = {}
        suspect_regions = []

    d = decide(fusion=fusion, structure=struct, alteration=CleanAltMock(), image_quality={"score": 0.85})

    assert d.status == "INCOMPLETE_SUBMISSION"
    assert "UPLOAD_BACK_SIDE" in d.blocking_gaps

def test_decision_engine_corroborated_alteration():
    fusion = FusionResult(ai_probability=0.10)
    class StructMock:
        validity_score = 0.85
        submission_complete = True
        missing_required = []
        role_conflicts = []
        structurally_valid = True

    class AltMock:
        probability = 0.90
        corroborated = True
        strong_cues = ["anchor_removal", "pasted_region_geometry"]
        cues = {"anchor_removal": {"detail": {"missing_anchors": ["aadhaar_logo"]}}}
        suspect_regions = []

    d = decide(fusion=fusion, structure=StructMock(), alteration=AltMock(), image_quality={"score": 0.80})

    assert d.status == "ALTERED"
    assert d.confidence >= 0.90
    assert "ALTERATION_CORROBORATED" in d.reason_codes
