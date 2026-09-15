import pytest
from app.modules.checksum import verify as verify_checksum, _mrz_td3, _mrz_td1
from app.modules.document_spec import get_spec, AADHAAR, PAN, PASSPORT, VISA, DRIVING_LICENSE, OTHER, SIDE_FRONT, SIDE_BACK
from app.modules.field_resolver import resolve_structure
from app.modules.qr_integrity import analyse_qr
import numpy as np

def test_icao_9303_official_worked_example():
    # Official ICAO 9303 TD3 Passport MRZ example (Eriksson)
    mrz_lines = [
        "P<DTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36DTO7408122F1204159ZE184226B<<<<<10"
    ]
    res = verify_checksum("MRZ_TD3", mrz_lines)
    assert res.applicable is True
    assert res.valid is True
    assert res.algorithm == "MRZ_TD3"

def test_passport_structure_resolution_with_mrz():
    mrz_lines = [
        "P<DTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36DTO7408122F1204159ZE184226B<<<<<10"
    ]
    spec = PASSPORT
    lines = [
        {"text": "Passport No: L898902C3", "conf": 0.90, "x0": 0.1, "y0": 0.2, "x1": 0.8, "y1": 0.25},
        {"text": "Name: Eriksson Anna Maria", "conf": 0.90, "x0": 0.1, "y0": 0.3, "x1": 0.8, "y1": 0.35},
        {"text": "DOB: 12/08/1974", "conf": 0.90, "x0": 0.1, "y0": 0.4, "x1": 0.8, "y1": 0.45},
    ]
    struct = resolve_structure(
        spec=spec,
        lines=lines,
        sides_present={SIDE_FRONT},
        mrz_lines=mrz_lines,
        checksum_input="L898902C3",
        extra_fields={"document_number": "L898902C3"}
    )

    assert struct.checksum_applicable is True
    assert struct.checksum_valid is True
    assert struct.validity_score >= 0.85
    assert struct.structurally_valid is True

def test_verhoeff_and_pan_checksums():
    v_res = verify_checksum("VERHOEFF", "277837463380")
    assert v_res.valid is True

    v_bad = verify_checksum("VERHOEFF", "277837463381")
    assert v_bad.valid is False

    pan_res = verify_checksum("PAN_ENTITY_CHAR", "ABCPE1234F")
    assert pan_res.valid is True

    pan_bad = verify_checksum("PAN_ENTITY_CHAR", "ABCZE1234F")
    assert pan_bad.valid is False

def test_driving_license_state_code():
    dl_ok = verify_checksum("DL_STATE_CODE", "DL1420110012345")
    assert dl_ok.valid is True

    dl_bad = verify_checksum("DL_STATE_CODE", "ZZ0000000000000")
    assert dl_bad.valid is False

def test_spec_fallback_for_unknown_type():
    spec = get_spec("TotallyUnknownDocType")
    assert spec.is_fallback is True
    assert spec.doc_type == "TotallyUnknownDocType"
    assert len(spec.anchors) == 0

    struct = resolve_structure(
        spec=spec,
        lines=[],
        sides_present={SIDE_FRONT}
    )
    assert struct.checksum_applicable is False
    assert struct.structurally_valid is True

def test_dynamic_qr_expected():
    spec_aadhaar = get_spec("Aadhaar")
    canvas = np.full((300, 300, 3), 200, dtype=np.uint8)
    qr_a = analyse_qr(canvas, spec_aadhaar, side=SIDE_BACK)
    assert qr_a.qr_expected is True

    spec_passport = get_spec("Passport")
    qr_p = analyse_qr(canvas, spec_passport, side=SIDE_FRONT)
    assert qr_p.qr_expected is False
