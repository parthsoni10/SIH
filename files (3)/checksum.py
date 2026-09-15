"""
SENTINEL V3.3 — Checksum Algorithm Registry
=============================================
Every previous version referenced `spec.checksum = "VERHOEFF"` etc. as a
label, but no code ever actually ran the algorithm — field_resolver only
accepted a pre-computed `checksum_valid: bool` from outside. This module
supplies the actual verifiers, keyed by the same string labels, so the
pipeline can compute (not just record) checksum validity for every document
type in the README.

Supported:
    VERHOEFF          -- Aadhaar 12-digit number
    PAN_ENTITY_CHAR    -- PAN 4th-character entity code sanity check
    MRZ_TD3            -- Passport, ICAO 9303 2-line 44-char MRZ
    MRZ_TD1             -- Visa / ID-1 3-line 30-char MRZ
    DL_STATE_CODE      -- Driving License: known Indian state/UT prefix + length
    NONE               -- no algorithm; always reports "not applicable"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INDIAN_STATE_CODES = {
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HR","HP",
    "JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","OR",
    "PB","PY","RJ","SK","TN","TR","TS","UK","UP","UA","WB",
}

MRZ_WEIGHTS = (7, 3, 1)
MRZ_CHARSET = "<0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class ChecksumResult:
    algorithm: str
    applicable: bool
    valid: bool | None      # None = applicable but could not be evaluated
    detail: str = ""


# ---------------------------------------------------------------------------
# Verhoeff — Aadhaar
# ---------------------------------------------------------------------------
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
     [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
     [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
     [9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
     [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
     [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def _verhoeff(number: str) -> ChecksumResult:
    digits = re.sub(r"\D", "", number or "")
    if len(digits) != 12:
        return ChecksumResult("VERHOEFF", True, None,
                              "expected 12 digits, got " + str(len(digits)))
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return ChecksumResult("VERHOEFF", True, c == 0)


# ---------------------------------------------------------------------------
# PAN — 4th character entity code
# ---------------------------------------------------------------------------
_PAN_ENTITY_CODES = set("PCHFATBLJG")   # Individual/Company/HUF/Firm/AOP/Trust/BOI/LA/AJP/Govt


def _pan_entity_char(pan: str) -> ChecksumResult:
    p = re.sub(r"\s", "", (pan or "")).upper()
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", p):
        return ChecksumResult("PAN_ENTITY_CHAR", True, False, "pattern mismatch")
    return ChecksumResult("PAN_ENTITY_CHAR", True, p[3] in _PAN_ENTITY_CODES,
                          f"4th char '{p[3]}'")


# ---------------------------------------------------------------------------
# MRZ — ICAO 9303 check-digit algorithm, shared by TD3 (passport) and
# TD1 (visa / ID-1) formats. The check-digit computation itself is identical;
# only line layout differs.
# ---------------------------------------------------------------------------
def _mrz_char_value(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - ord("A") + 10
    return 0


def _mrz_check_digit(data: str) -> int:
    total = 0
    for i, c in enumerate(data):
        total += _mrz_char_value(c) * MRZ_WEIGHTS[i % 3]
    return total % 10


def _mrz_td3(lines: list[str]) -> ChecksumResult:
    """Passport, 2 lines x 44 chars."""
    if len(lines) < 2 or any(len(ln) < 44 for ln in lines):
        return ChecksumResult("MRZ_TD3", True, None, "malformed MRZ block")
    l2 = lines[1]
    doc_num, doc_cd = l2[0:9], l2[9]
    dob, dob_cd = l2[13:19], l2[19]
    exp, exp_cd = l2[21:27], l2[27]
    composite = l2[0:10] + l2[13:20] + l2[21:43]
    comp_cd = l2[43]

    checks = {
        "document_number": _mrz_check_digit(doc_num) == _mrz_char_value(doc_cd),
        "dob": _mrz_check_digit(dob) == _mrz_char_value(dob_cd),
        "expiry": _mrz_check_digit(exp) == _mrz_char_value(exp_cd),
        "composite": _mrz_check_digit(composite) == _mrz_char_value(comp_cd),
    }
    return ChecksumResult("MRZ_TD3", True, all(checks.values()), str(checks))


def _mrz_td1(lines: list[str]) -> ChecksumResult:
    """Visa / ID-1, 3 lines x 30 chars."""
    if len(lines) < 3 or any(len(ln) < 30 for ln in lines):
        return ChecksumResult("MRZ_TD1", True, None, "malformed MRZ block")
    l1, l2 = lines[0], lines[1]
    doc_num, doc_cd = l1[5:14], l1[14]
    dob, dob_cd = l2[0:6], l2[6]
    exp, exp_cd = l2[8:14], l2[14]

    checks = {
        "document_number": _mrz_check_digit(doc_num) == _mrz_char_value(doc_cd),
        "dob": _mrz_check_digit(dob) == _mrz_char_value(dob_cd),
        "expiry": _mrz_check_digit(exp) == _mrz_char_value(exp_cd),
    }
    return ChecksumResult("MRZ_TD1", True, all(checks.values()), str(checks))


# ---------------------------------------------------------------------------
# Driving License — state prefix + fixed length (no true check digit exists
# nationally; this is a plausibility check, not a cryptographic one).
# ---------------------------------------------------------------------------
def _dl_state_code(dl_number: str) -> ChecksumResult:
    v = re.sub(r"[\s-]", "", (dl_number or "")).upper()
    if len(v) < 4:
        return ChecksumResult("DL_STATE_CODE", True, False, "too short")
    state = v[:2]
    rest_ok = bool(re.match(r"^\d{2,4}\d{6,9}$", v[2:])) or bool(re.match(r"^\d{11,13}$", v[2:]))
    return ChecksumResult("DL_STATE_CODE", True,
                          state in INDIAN_STATE_CODES and rest_ok,
                          f"state='{state}' known={state in INDIAN_STATE_CODES}")


def _none(_value) -> ChecksumResult:
    return ChecksumResult("NONE", False, None, "no checksum defined for this document type")


REGISTRY = {
    "VERHOEFF": _verhoeff,
    "PAN_ENTITY_CHAR": _pan_entity_char,
    "MRZ_TD3": _mrz_td3,
    "MRZ_TD1": _mrz_td1,
    "DL_STATE_CODE": _dl_state_code,
    "NONE": _none,
}


def verify(algorithm: str | None, value) -> ChecksumResult:
    """Single entry point. `value` is a str for VERHOEFF/PAN/DL, or a
    list[str] of MRZ lines for MRZ_TD3/MRZ_TD1."""
    fn = REGISTRY.get((algorithm or "NONE").upper(), _none)
    try:
        return fn(value)
    except Exception as exc:
        return ChecksumResult(algorithm or "NONE", True, None,
                              f"evaluation error: {exc}")
