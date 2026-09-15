"""
SENTINEL V3.1 — Declarative Document Structure Specification
=============================================================
This file is the single source of truth for "what must a valid X look like".

Your V2/V3 pipeline had no such thing. `validation.py` scored structure from a
loose bag of regexes, which is why verification #207 reported
`Document Validity 86% / STRUCTURALLY VALID` on a card whose holder name was
never extracted and whose mandatory Aadhaar logo had been erased.

A specification declares four things per document type:

  1. SIDES      -- which physical sides must be submitted before a verdict is
                   admissible. Aadhaar's QR and address live on the back; a
                   front-only submission is INCOMPLETE, not GENUINE.
  2. FIELDS     -- required fields, which side carries them, their semantic
                   ROLE, and a plausibility predicate. Roles matter: #207
                   extracted `dob = 03/01/2014`, which is the *Issue Date*
                   printed vertically on the left edge. The real DOB is
                   17/08/2005. A date-role resolver plus an age-plausibility
                   check catches that; a bare date regex never will.
  3. ANCHORS    -- mandatory printed artwork at fixed normalised positions.
                   This is what detects the erased Aadhaar logo: the region is
                   declared to carry saturated colour ink, and an altered card
                   shows ink coverage 0.0011 against a genuine 0.1652.
  4. CHECKSUM   -- the number validation algorithm.

Normalised coordinates are fractions of the deskewed, cropped document. Run
layout_validator's 4-point warp BEFORE anchor verification or every box will
be offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Field roles. The bug in #207 was treating every date as interchangeable.
# ---------------------------------------------------------------------------
ROLE_DOB = "DATE_OF_BIRTH"
ROLE_ISSUE = "DATE_OF_ISSUE"
ROLE_EXPIRY = "DATE_OF_EXPIRY"
ROLE_NUMBER = "DOCUMENT_NUMBER"
ROLE_NAME = "HOLDER_NAME"
ROLE_PARENT = "PARENT_NAME"
ROLE_GENDER = "GENDER"
ROLE_ADDRESS = "ADDRESS"

SIDE_FRONT = "FRONT"
SIDE_BACK = "BACK"
SIDE_EITHER = "EITHER"


@dataclass
class FieldSpec:
    key: str
    role: str
    side: str = SIDE_FRONT
    required: bool = True
    # Labels printed next to the value, used to bind a value to its role.
    # Multilingual on purpose -- Aadhaar prints Gujarati/Hindi + English.
    labels: tuple[str, ...] = ()
    # Regex the raw value must satisfy after normalisation.
    pattern: str | None = None
    # Extra semantic gate, e.g. "a DOB implying age < 5 on an adult card".
    plausibility: Callable[[str], bool] | None = None
    note: str = ""


@dataclass
class AnchorSpec:
    """A region of mandatory printed artwork."""
    key: str
    side: str
    # normalised (x0, y0, x1, y1) on the deskewed document
    bbox: tuple[float, float, float, float]
    # Minimum fraction of pixels that must read as ink (dark or saturated).
    min_ink_coverage: float
    # Minimum fraction that must be strongly chromatic. 0.0 = greyscale ok.
    min_colour_coverage: float = 0.0
    description: str = ""


@dataclass
class DocumentSpec:
    doc_type: str
    sides_required: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    anchors: tuple[AnchorSpec, ...]
    checksum: str | None = None
    aspect_ratio: tuple[float, float] | None = None
    qr_expected_side: str | None = None
    notes: str = ""

    def required_fields(self, sides_present: set[str]) -> list[FieldSpec]:
        """Only demand fields whose side was actually submitted."""
        return [f for f in self.fields
                if f.required and (f.side == SIDE_EITHER or f.side in sides_present)]

    def unsubmitted_sides(self, sides_present: set[str]) -> list[str]:
        return [s for s in self.sides_required if s not in sides_present]


# ---------------------------------------------------------------------------
# Plausibility predicates
# ---------------------------------------------------------------------------
def _parse_date(v: str) -> date | None:
    v = (v or "").strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def dob_plausible(v: str) -> bool:
    """A DOB must be in the past and imply an age in [0, 120].

    #207 assigned the card's Issue Date (03/01/2014) to the DOB field, which
    implies a 12-year-old holder on a card whose printed DOB is 17/08/2005.
    Age plausibility alone does not catch that -- 12 is a legal age for an
    Aadhaar -- which is exactly why role binding by LABEL PROXIMITY is the
    primary mechanism and plausibility is only the backstop.
    """
    d = _parse_date(v)
    if d is None:
        return False
    age = (date.today() - d).days / 365.25
    return 0 <= age <= 120


def issue_date_plausible(v: str) -> bool:
    d = _parse_date(v)
    return d is not None and d <= date.today()


# ---------------------------------------------------------------------------
# AADHAAR
# Anchor boxes measured on the reference scan (1600x1073). The Aadhaar logo
# box maps to x 0.783-0.980, y 0.047-0.201 -- exactly the region erased in the
# altered sample.
# ---------------------------------------------------------------------------
AADHAAR = DocumentSpec(
    doc_type="Aadhaar",
    sides_required=(SIDE_FRONT, SIDE_BACK),
    checksum="VERHOEFF",
    aspect_ratio=(1.45, 1.65),
    qr_expected_side=SIDE_BACK,
    notes=("Front carries identity fields; back carries address and the signed "
           "QR payload. Confirm QR side against your own genuine reference "
           "stock before enabling the mandatory-QR rule -- letter cut-outs and "
           "PVC cards differ."),
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("आधार", "આધાર", "Aadhaar"),
                  pattern=r"^[2-9][0-9]{11}$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT,
                  labels=("Name", "नाम", "નામ"),
                  note="Printed in Latin AND the state script. Latin line is "
                       "the reliable extraction target."),
        FieldSpec("dob", ROLE_DOB, SIDE_FRONT,
                  labels=("DOB", "Date of Birth", "जन्म तारीख", "જન્મ તારીખ"),
                  pattern=r"^\d{2}[/-]\d{2}[/-]\d{4}$",
                  plausibility=dob_plausible),
        FieldSpec("gender", ROLE_GENDER, SIDE_FRONT,
                  labels=("Male", "Female", "पुरुष", "પુરુષ", "સ્ત્રી")),
        FieldSpec("issue_date", ROLE_ISSUE, SIDE_FRONT, required=False,
                  labels=("Issue Date", "Download Date"),
                  plausibility=issue_date_plausible,
                  note="Printed vertically on the left edge. Must NOT be "
                       "bound to the DOB field -- this was the #207 bug."),
        FieldSpec("address", ROLE_ADDRESS, SIDE_BACK,
                  labels=("Address", "सरनामा", "સરનામું")),
    ),
    anchors=(
        AnchorSpec("state_emblem", SIDE_FRONT, (0.030, 0.030, 0.115, 0.190),
                   min_ink_coverage=0.08,
                   description="Ashoka lion capital, top-left"),
        AnchorSpec("tricolour_govt_band", SIDE_FRONT, (0.240, 0.040, 0.730, 0.180),
                   min_ink_coverage=0.10, min_colour_coverage=0.06,
                   description="Saffron/green brush band + 'Government of India'"),
        AnchorSpec("aadhaar_logo", SIDE_FRONT, (0.783, 0.040, 0.985, 0.205),
                   min_ink_coverage=0.05, min_colour_coverage=0.03,
                   description="UIDAI fingerprint/sun mark, top-right. "
                               "Genuine ink coverage ~0.165; erased ~0.001."),
        AnchorSpec("portrait", SIDE_FRONT, (0.060, 0.200, 0.330, 0.670),
                   min_ink_coverage=0.30,
                   description="Holder photograph"),
        AnchorSpec("number_band", SIDE_FRONT, (0.280, 0.730, 0.715, 0.870),
                   min_ink_coverage=0.06,
                   description="12-digit Aadhaar number"),
        AnchorSpec("slogan_rule", SIDE_FRONT, (0.150, 0.860, 0.850, 0.995),
                   min_ink_coverage=0.05,
                   description="Red rule + 'My Aadhaar, my identity' slogan"),
    ),
)

# ---------------------------------------------------------------------------
# PAN CARD
# ---------------------------------------------------------------------------
PAN = DocumentSpec(
    doc_type="PAN Card",
    sides_required=(SIDE_FRONT,),
    checksum="PAN_ENTITY_CHAR",
    aspect_ratio=(1.45, 1.70),
    qr_expected_side=SIDE_FRONT,
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("Permanent Account Number",),
                  pattern=r"^[A-Z]{5}[0-9]{4}[A-Z]$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT, labels=("Name", "नाम")),
        FieldSpec("father_name", ROLE_PARENT, SIDE_FRONT,
                  labels=("Father's Name", "पिता का नाम")),
        FieldSpec("dob", ROLE_DOB, SIDE_FRONT,
                  labels=("Date of Birth", "जन्म की तारीख"),
                  pattern=r"^\d{2}[/-]\d{2}[/-]\d{4}$",
                  plausibility=dob_plausible),
    ),
    anchors=(
        AnchorSpec("state_emblem", SIDE_FRONT, (0.460, 0.020, 0.580, 0.220),
                   min_ink_coverage=0.07, description="Ashoka emblem, centre-top"),
        AnchorSpec("dept_header", SIDE_FRONT, (0.030, 0.030, 0.430, 0.200),
                   min_ink_coverage=0.10, description="INCOME TAX DEPARTMENT"),
        AnchorSpec("govt_header", SIDE_FRONT, (0.620, 0.030, 0.985, 0.200),
                   min_ink_coverage=0.10, description="GOVT. OF INDIA"),
        AnchorSpec("qr_block", SIDE_FRONT, (0.680, 0.250, 0.980, 0.680),
                   min_ink_coverage=0.25, description="Signed QR symbol"),
        AnchorSpec("portrait", SIDE_FRONT, (0.040, 0.240, 0.240, 0.560),
                   min_ink_coverage=0.30, description="Holder photograph"),
    ),
)

REGISTRY: dict[str, DocumentSpec] = {
    "Aadhaar": AADHAAR,
    "PAN Card": PAN,
}


def get_spec(doc_type: str) -> DocumentSpec | None:
    return REGISTRY.get(doc_type)
