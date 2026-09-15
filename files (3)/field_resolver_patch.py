"""
SENTINEL V3.3 — field_resolver.py PATCH (checksum wiring only)
================================================================
Everything from V3.1's field_resolver.py (date-role binding, name
resolution, completeness gating) is UNCHANGED. This patch replaces only the
checksum handling in `resolve_structure`, which previously required the
caller to pass a pre-computed `checksum_valid: bool` -- meaning no checksum
algorithm ever actually ran inside the pipeline for ANY document type.

New signature: `resolve_structure(..., checksum_input=..., mrz_lines=...)`
replaces `checksum_valid: bool`. `checksum.verify()` is called internally
using `spec.checksum`, so Aadhaar/PAN/Passport/Visa/Driving License/Other
all get real checksum evaluation without the caller needing to know which
algorithm applies to which type.

  - VERHOEFF / PAN_ENTITY_CHAR / DL_STATE_CODE -> pass the document number
    string as `checksum_input`
  - MRZ_TD3 / MRZ_TD1                          -> pass `mrz_lines` (a list
    of the raw MRZ line strings); `checksum_input` is ignored
  - NONE (Other/fallback)                      -> both ignored, always
    reports "not applicable", which the validity-score gate below already
    treats as neutral rather than as a failure

If `checksum.verify()` returns `valid=None` (algorithm applicable but could
not be evaluated -- e.g. malformed MRZ block, wrong digit count), that is
treated as MISSING evidence, not as a failure: it should push toward
MANUAL_REVIEW ("please improve the scan"), not toward SUSPICIOUS ("this
number is wrong"). Only an explicit `valid=False` counts as a checksum
failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from . import checksum as checksum_mod
from .document_spec import DocumentSpec, ROLE_DOB, ROLE_ISSUE, ROLE_NAME, SIDE_FRONT

# ... (DATE_RE, LATIN_NAME_RE, NAME_STOPWORDS, ResolvedField, resolve_dates,
#      resolve_name, _parse_date, _norm, _box_distance, _is_vertical are
#      UNCHANGED from V3.1's field_resolver.py -- import or keep them as-is.
#      Only FieldResolution and resolve_structure change below.)

from .field_resolver import (        # re-use everything else unchanged
    resolve_dates, resolve_name, ResolvedField,
)


@dataclass
class FieldResolution:
    fields: dict[str, dict] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    role_conflicts: list[str] = field(default_factory=list)

    sides_present: list[str] = field(default_factory=list)
    sides_missing: list[str] = field(default_factory=list)
    submission_complete: bool = False

    completeness: float = 0.0
    validity_score: float = 0.0
    structurally_valid: bool = False

    checksum_algorithm: str = "NONE"
    checksum_applicable: bool = False
    checksum_valid: bool | None = None
    checksum_detail: str = ""

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_structure(*,
                      spec: DocumentSpec,
                      lines: list[dict],
                      sides_present: set[str],
                      extra_fields: dict[str, Any] | None = None,
                      checksum_input: str | None = None,
                      mrz_lines: list[str] | None = None,
                      layout_score: float = 1.0) -> FieldResolution:
    r = FieldResolution()
    extra_fields = extra_fields or {}
    r.sides_present = sorted(sides_present)

    resolved: dict[str, ResolvedField] = {}
    resolved.update(resolve_dates(lines, spec))
    nm = resolve_name(lines, spec)
    resolved[nm.key] = nm

    for k, v in extra_fields.items():
        if k not in resolved and v:
            spc = next((f for f in spec.fields if f.key == k), None)
            resolved[k] = ResolvedField(
                k, spc.role if spc else "UNKNOWN", str(v), 0.9,
                "REGEX_FALLBACK",
                bool(spc.plausibility(str(v)) if (spc and spc.plausibility) else True))

    # ---- checksum: ACTUALLY COMPUTED now, not passed in as a bool --------
    algo = (spec.checksum or "NONE").upper()
    r.checksum_algorithm = algo
    if algo in ("MRZ_TD3", "MRZ_TD1"):
        cr = checksum_mod.verify(algo, mrz_lines or [])
    else:
        doc_num = checksum_input
        if doc_num is None:
            doc_num_field = resolved.get("document_number") or extra_fields.get("document_number")
            doc_num = (doc_num_field.value if isinstance(doc_num_field, ResolvedField)
                      else doc_num_field)
        cr = checksum_mod.verify(algo, doc_num)
    r.checksum_applicable = cr.applicable
    r.checksum_valid = cr.valid
    r.checksum_detail = cr.detail
    if cr.applicable and cr.valid is False:
        r.reasons.append(f"CHECKSUM_FAILED_{algo}")
    elif cr.applicable and cr.valid is None:
        r.reasons.append(f"CHECKSUM_INDETERMINATE_{algo}")

    # ---- side completeness -------------------------------------------------
    r.sides_missing = spec.unsubmitted_sides(sides_present)
    r.submission_complete = not r.sides_missing
    if r.sides_missing:
        r.reasons.append("INCOMPLETE_SUBMISSION_MISSING_SIDE")
        for s in r.sides_missing:
            r.reasons.append(f"SIDE_NOT_PROVIDED_{s}")

    # ---- required fields, scoped to submitted sides ------------------------
    required = spec.required_fields(sides_present)
    got = 0
    for fs in required:
        rf = resolved.get(fs.key)
        if rf is None or not rf.value:
            r.missing_required.append(fs.key)
            r.reasons.append(f"REQUIRED_FIELD_MISSING_{fs.key.upper()}")
            continue
        if fs.pattern and not re.match(fs.pattern, str(rf.value).replace(" ", "")):
            rf.issues.append("PATTERN_MISMATCH")
            r.missing_required.append(fs.key)
            r.reasons.append(f"REQUIRED_FIELD_MALFORMED_{fs.key.upper()}")
            continue
        if not rf.plausible:
            rf.issues.append("IMPLAUSIBLE_VALUE")
            r.role_conflicts.append(fs.key)
            r.reasons.append(f"FIELD_ROLE_CONFLICT_{fs.key.upper()}")
            continue
        got += 1

    r.fields = {k: asdict(v) for k, v in resolved.items()}
    r.completeness = round(got / float(len(required)), 4) if required else 1.0

    # ---- gated validity score -----------------------------------------------
    base = 0.55 * r.completeness + 0.25 * float(layout_score)
    if r.checksum_applicable:
        if r.checksum_valid is True:
            base += 0.20
        elif r.checksum_valid is False:
            base = min(base, 0.35)
        # checksum_valid is None (indeterminate): no bonus, no penalty --
        # treated as missing evidence, pushed toward review via completeness
        # gates below rather than toward SUSPICIOUS.
    else:
        # No checksum defined for this type (Other/fallback, or DL when you
        # haven't wired a real algorithm) -- don't punish structural score
        # for something that was never verifiable to begin with.
        base += 0.10

    if r.missing_required:
        base = min(base, 0.60)
    if r.role_conflicts:
        base = min(base, 0.65)
    if not r.submission_complete:
        base = min(base, 0.70)
    if r.checksum_applicable and r.checksum_valid is None:
        base = min(base, 0.75)

    r.validity_score = round(max(0.0, min(1.0, base)), 4)
    r.structurally_valid = (
        r.validity_score >= 0.85
        and not r.missing_required
        and not r.role_conflicts
        and r.submission_complete
        and not (r.checksum_applicable and r.checksum_valid is False)
    )

    if r.structurally_valid:
        r.reasons.append("DOCUMENT_STRUCTURE_VALID")
    r.reasons = list(dict.fromkeys(r.reasons))
    return r
