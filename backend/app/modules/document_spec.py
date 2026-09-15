"""
SENTINEL V3.3 — Global Document Specification Registry
=========================================================
Extends specifications to cover all supported document types:
    Passport, Visa, Aadhaar, PAN Card, Driving License, Other

Design Principle
----------------
Every spec ships with `calibrated=False`. Anchor/geometry evidence is
attached and visible but cannot solely gate a decision until you run
ml/calibrate_anchors.py on your own confirmed-genuine samples for that
specific document type.

get_spec() never returns None. An unrecognised `doc_type` string returns
the same minimal fallback as `Other`, tagged with `is_fallback=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Optional

ROLE_DOB = "DATE_OF_BIRTH"
ROLE_ISSUE = "DATE_OF_ISSUE"
ROLE_EXPIRY = "DATE_OF_EXPIRY"
ROLE_NUMBER = "DOCUMENT_NUMBER"
ROLE_NAME = "HOLDER_NAME"
ROLE_PARENT = "PARENT_NAME"
ROLE_GENDER = "GENDER"
ROLE_ADDRESS = "ADDRESS"
ROLE_NATIONALITY = "NATIONALITY"
ROLE_MRZ_LINE = "MRZ_RAW_LINE"

SIDE_FRONT = "FRONT"
SIDE_BACK = "BACK"
SIDE_EITHER = "EITHER"


@dataclass
class FieldSpec:
    key: str
    role: str
    side: str = SIDE_FRONT
    required: bool = True
    labels: tuple[str, ...] = ()
    pattern: Optional[str] = None
    plausibility: Optional[Callable[[str], bool]] = None
    note: str = ""


@dataclass
class AnchorSpec:
    key: str
    side: str
    bbox: tuple[float, float, float, float]
    min_ink_coverage: float
    min_colour_coverage: float = 0.0
    description: str = ""


@dataclass
class DocumentSpec:
    doc_type: str
    sides_required: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    anchors: tuple[AnchorSpec, ...]
    checksum: Optional[str] = None          # key into checksum.REGISTRY
    aspect_ratio: Optional[tuple[float, float]] = None
    qr_expected_side: Optional[str] = None  # None = QR not expected/mandatory
    calibrated: bool = False
    min_calibration_samples: int = 25
    is_fallback: bool = False            # True only for Other/unrecognised
    notes: str = ""

    def required_fields(self, sides_present: set[str]) -> list[FieldSpec]:
        return [f for f in self.fields
                if f.required and (f.side == SIDE_EITHER or f.side in sides_present)]

    def unsubmitted_sides(self, sides_present: set[str]) -> list[str]:
        return [s for s in self.sides_required if s not in sides_present]


def _parse_date(v: str) -> Optional[date]:
    v = (v or "").strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def dob_plausible(v: str) -> bool:
    d = _parse_date(v)
    if d is None:
        return False
    age = (date.today() - d).days / 365.25
    return 0 <= age <= 120


def issue_date_plausible(v: str) -> bool:
    d = _parse_date(v)
    return d is not None and d <= date.today()


def expiry_not_yet_passed(v: str) -> bool:
    d = _parse_date(v)
    return d is not None and d >= date.today()


# ===========================================================================
# AADHAAR
# ===========================================================================
AADHAAR = DocumentSpec(
    doc_type="Aadhaar",
    sides_required=(SIDE_FRONT, SIDE_BACK),
    checksum="VERHOEFF",
    aspect_ratio=(1.45, 1.65),
    qr_expected_side=SIDE_BACK,
    calibrated=False,
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("आधार", "આધાર", "Aadhaar"),
                  pattern=r"^[2-9][0-9]{11}$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT,
                  labels=("Name", "नाम", "નામ")),
        FieldSpec("dob", ROLE_DOB, SIDE_FRONT,
                  labels=("DOB", "Date of Birth", "जन्म तारीख", "જન્મ તારીખ"),
                  pattern=r"^\d{2}[/-]\d{2}[/-]\d{4}$",
                  plausibility=dob_plausible),
        FieldSpec("gender", ROLE_GENDER, SIDE_FRONT,
                  labels=("Male", "Female", "पुरुष", "પુરુષ", "સ્ત્રી")),
        FieldSpec("issue_date", ROLE_ISSUE, SIDE_FRONT, required=False,
                  labels=("Issue Date", "Download Date"),
                  plausibility=issue_date_plausible),
        FieldSpec("address", ROLE_ADDRESS, SIDE_BACK,
                  labels=("Address", "सरनामा", "સરનામું")),
    ),
    anchors=(
        AnchorSpec("state_emblem", SIDE_FRONT, (0.030, 0.030, 0.115, 0.190),
                   min_ink_coverage=0.08, description="Ashoka lion capital"),
        AnchorSpec("tricolour_govt_band", SIDE_FRONT, (0.240, 0.040, 0.730, 0.180),
                   min_ink_coverage=0.10, min_colour_coverage=0.06,
                   description="Saffron/green brush band"),
        AnchorSpec("aadhaar_logo", SIDE_FRONT, (0.783, 0.040, 0.985, 0.205),
                   min_ink_coverage=0.05, min_colour_coverage=0.03,
                   description="UIDAI mark, top-right"),
        AnchorSpec("portrait", SIDE_FRONT, (0.060, 0.200, 0.330, 0.670),
                   min_ink_coverage=0.30, description="Holder photograph"),
        AnchorSpec("number_band", SIDE_FRONT, (0.280, 0.730, 0.715, 0.870),
                   min_ink_coverage=0.06, description="12-digit number"),
        AnchorSpec("slogan_rule", SIDE_FRONT, (0.150, 0.860, 0.850, 0.995),
                   min_ink_coverage=0.05, description="Red rule + slogan"),
    ),
)

# ===========================================================================
# PAN CARD
# ===========================================================================
PAN = DocumentSpec(
    doc_type="PAN Card",
    sides_required=(SIDE_FRONT,),
    checksum="PAN_ENTITY_CHAR",
    aspect_ratio=(1.45, 1.70),
    qr_expected_side=SIDE_FRONT,
    calibrated=False,
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
                   min_ink_coverage=0.07, description="Ashoka emblem"),
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

# ===========================================================================
# PASSPORT (Indian, TD3 booklet bio page)
# ===========================================================================
PASSPORT = DocumentSpec(
    doc_type="Passport",
    sides_required=(SIDE_FRONT,),
    checksum="MRZ_TD3",
    aspect_ratio=(1.30, 1.50),
    qr_expected_side=None,
    calibrated=False,
    notes="Anchor boxes are unmeasured placeholders.",
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("Passport No", "Passport Number"),
                  pattern=r"^[A-Z0-9]{7,9}$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT,
                  labels=("Surname", "Given Name", "Name")),
        FieldSpec("nationality", ROLE_NATIONALITY, SIDE_FRONT, required=False,
                  labels=("Nationality",)),
        FieldSpec("dob", ROLE_DOB, SIDE_FRONT,
                  labels=("Date of Birth", "DOB"),
                  plausibility=dob_plausible),
        FieldSpec("expiry_date", ROLE_EXPIRY, SIDE_FRONT, required=False,
                  labels=("Date of Expiry",),
                  plausibility=expiry_not_yet_passed),
        FieldSpec("mrz_line1", ROLE_MRZ_LINE, SIDE_FRONT, required=False,
                  labels=(), pattern=r"^[A-Z0-9<]{44}$"),
        FieldSpec("mrz_line2", ROLE_MRZ_LINE, SIDE_FRONT, required=False,
                  labels=(), pattern=r"^[A-Z0-9<]{44}$"),
    ),
    anchors=(
        AnchorSpec("national_emblem", SIDE_FRONT, (0.400, 0.030, 0.600, 0.220),
                   min_ink_coverage=0.06, description="National emblem"),
        AnchorSpec("portrait", SIDE_FRONT, (0.050, 0.220, 0.350, 0.650),
                   min_ink_coverage=0.25, description="Holder photograph"),
        AnchorSpec("mrz_zone", SIDE_FRONT, (0.030, 0.850, 0.970, 0.980),
                   min_ink_coverage=0.20, description="Bottom OCR-B MRZ band"),
    ),
)

# ===========================================================================
# VISA (TD1-style MRZ)
# ===========================================================================
VISA = DocumentSpec(
    doc_type="Visa",
    sides_required=(SIDE_FRONT,),
    checksum="MRZ_TD1",
    aspect_ratio=(1.20, 1.60),
    qr_expected_side=None,
    calibrated=False,
    notes="Anchor boxes are unmeasured placeholders.",
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("Visa No", "Visa Number"),
                  pattern=r"^[A-Z0-9]{7,9}$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT, labels=("Name",)),
        FieldSpec("nationality", ROLE_NATIONALITY, SIDE_FRONT, required=False,
                  labels=("Nationality",)),
        FieldSpec("expiry_date", ROLE_EXPIRY, SIDE_FRONT, required=False,
                  labels=("Valid Until", "Date of Expiry"),
                  plausibility=expiry_not_yet_passed),
        FieldSpec("mrz_line1", ROLE_MRZ_LINE, SIDE_FRONT, required=False,
                  labels=(), pattern=r"^[A-Z0-9<]{30}$"),
        FieldSpec("mrz_line2", ROLE_MRZ_LINE, SIDE_FRONT, required=False,
                  labels=(), pattern=r"^[A-Z0-9<]{30}$"),
        FieldSpec("mrz_line3", ROLE_MRZ_LINE, SIDE_FRONT, required=False,
                  labels=(), pattern=r"^[A-Z0-9<]{30}$"),
    ),
    anchors=(
        AnchorSpec("header_band", SIDE_FRONT, (0.050, 0.030, 0.950, 0.180),
                   min_ink_coverage=0.10, description="VISA header"),
        AnchorSpec("mrz_zone", SIDE_FRONT, (0.030, 0.780, 0.970, 0.970),
                   min_ink_coverage=0.20, description="3-line MRZ band"),
    ),
)

# ===========================================================================
# DRIVING LICENSE
# ===========================================================================
DRIVING_LICENSE = DocumentSpec(
    doc_type="Driving License",
    sides_required=(SIDE_FRONT,),
    checksum="DL_STATE_CODE",
    aspect_ratio=(1.45, 1.70),
    qr_expected_side=None,
    calibrated=False,
    notes="DL_STATE_CODE is a state prefix + length plausibility check.",
    fields=(
        FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                  labels=("DL No", "Licence No", "License No"),
                  pattern=r"^[A-Z]{2}[0-9]{11,15}$"),
        FieldSpec("full_name", ROLE_NAME, SIDE_FRONT, labels=("Name",)),
        FieldSpec("dob", ROLE_DOB, SIDE_FRONT,
                  labels=("Date of Birth", "DOB"),
                  plausibility=dob_plausible),
        FieldSpec("expiry_date", ROLE_EXPIRY, SIDE_FRONT, required=False,
                  labels=("Valid Till", "Date of Expiry"),
                  plausibility=expiry_not_yet_passed),
    ),
    anchors=(
        AnchorSpec("state_transport_header", SIDE_FRONT,
                   (0.030, 0.030, 0.700, 0.180), min_ink_coverage=0.10,
                   description="State transport dept header"),
        AnchorSpec("portrait", SIDE_FRONT, (0.040, 0.220, 0.280, 0.600),
                   min_ink_coverage=0.25, description="Holder photograph"),
    ),
)


def _make_generic_fallback(doc_type: str) -> DocumentSpec:
    return DocumentSpec(
        doc_type=doc_type,
        sides_required=(SIDE_FRONT,),
        checksum="NONE",
        aspect_ratio=None,
        qr_expected_side=None,
        calibrated=False,
        is_fallback=True,
        notes="Generic fallback spec.",
        fields=(
            FieldSpec("document_number", ROLE_NUMBER, SIDE_FRONT,
                      required=False, labels=()),
        ),
        anchors=(),
    )


OTHER = _make_generic_fallback("Other")

REGISTRY: dict[str, DocumentSpec] = {
    "Aadhaar": AADHAAR,
    "PAN Card": PAN,
    "Passport": PASSPORT,
    "Visa": VISA,
    "Driving License": DRIVING_LICENSE,
    "Other": OTHER,
}


def get_spec(doc_type: str) -> DocumentSpec:
    spec = REGISTRY.get(doc_type)
    if spec is not None:
        return spec
    return _make_generic_fallback(doc_type)
