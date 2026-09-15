"""
SENTINEL V3.1 — Field Role Resolution & Structural Completeness
================================================================
Fixes three distinct bugs visible in verification #207.

BUG A — Date role confusion
---------------------------
#207 reported `dob = 03/01/2014` at 80% confidence. That value is the card's
ISSUE DATE, printed vertically along the left edge. The printed DOB is
17/08/2005. A bare `\\d{2}/\\d{2}/\\d{4}` regex returns whichever date the OCR
happened to emit first, and vertical text is often read first because it sits
at a low x-coordinate.

Fix: bind every candidate value to a role by LABEL PROXIMITY using OCR bounding
boxes, not by scan order. Plausibility is only the backstop -- age alone cannot
distinguish these two dates, since a 12-year-old Aadhaar holder is perfectly
legal. Geometry is what disambiguates.

BUG B — Missing required field did not block clearance
-------------------------------------------------------
#207 raised `Holder Name Missing or Unreadable in OCR`, extracted 2 of the 4
required front fields, and still returned `Document Validity 86% ·
STRUCTURALLY VALID · GENUINE`. A flagged anomaly that does not move the
verdict is decoration.

Fix: completeness is computed against the declarative spec and becomes a hard
gate. A missing REQUIRED field caps validity and forces MANUAL_REVIEW.

The name failed to extract because PaddleOCR was running a Latin-only model
against a card printed in Gujarati and English. The Latin line
("Dhruv Prajapati") is extractable, but the label-based regex looked for a
"Name" token that Aadhaar never prints -- Aadhaar puts the name on a bare line
above the DOB with no label. Hence the positional fallback below.

BUG C — Single-side submission cleared as complete
---------------------------------------------------
Aadhaar's address and signed QR live on the back. A front-only upload can
never satisfy the spec, but #207 reported no gap. Fix: `sides_required` from
the spec is checked explicitly and a missing side yields
INCOMPLETE_SUBMISSION, which is distinct from both GENUINE and SUSPICIOUS --
the document is not accused of anything, it simply has not been fully
presented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any

from .document_spec import (DocumentSpec, FieldSpec, ROLE_DOB, ROLE_ISSUE,
                            ROLE_NAME, SIDE_BACK, SIDE_EITHER, SIDE_FRONT)

DATE_RE = re.compile(r"\b(\d{2})\s*[/\-.]\s*(\d{2})\s*[/\-.]\s*(\d{4})\b")
LATIN_NAME_RE = re.compile(r"^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}$")

# Tokens that must never be accepted as a holder name.
NAME_STOPWORDS = {
    "government", "india", "male", "female", "dob", "date", "birth", "issue",
    "aadhaar", "aadhar", "uidai", "income", "tax", "department", "permanent",
    "account", "number", "card", "signature", "father", "name", "address",
}


@dataclass
class ResolvedField:
    key: str
    role: str
    value: str | None
    confidence: float
    source: str            # LABEL_PROXIMITY | POSITIONAL | REGEX_FALLBACK
    plausible: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class FieldResolution:
    fields: dict[str, dict] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    role_conflicts: list[str] = field(default_factory=list)

    sides_present: list[str] = field(default_factory=list)
    sides_missing: list[str] = field(default_factory=list)
    submission_complete: bool = False

    completeness: float = 0.0     # fraction of required fields resolved
    validity_score: float = 0.0   # completeness gated structural score
    structurally_valid: bool = False

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
def _parse_date(v: str) -> date | None:
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(v.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _box_distance(a, b) -> float:
    """Centre distance between two OCR boxes, normalised by page diagonal."""
    ax = (a["x0"] + a["x1"]) / 2.0
    ay = (a["y0"] + a["y1"]) / 2.0
    bx = (b["x0"] + b["x1"]) / 2.0
    by = (b["y0"] + b["y1"]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _is_vertical(box) -> bool:
    """Vertical text has height much greater than width. The Aadhaar issue
    date is printed this way, and this alone excludes it from DOB binding."""
    w = max(box["x1"] - box["x0"], 1e-6)
    h = max(box["y1"] - box["y0"], 1e-6)
    return (h / w) > 2.5


# ---------------------------------------------------------------------------
def resolve_dates(lines: list[dict], spec: DocumentSpec) -> dict[str, ResolvedField]:
    """Bind each date on the page to a role using label proximity.

    `lines` = [{"text": str, "conf": float,
                "x0","y0","x1","y1": float (normalised)}, ...]
    """
    out: dict[str, ResolvedField] = {}

    dob_spec = next((f for f in spec.fields if f.role == ROLE_DOB), None)
    iss_spec = next((f for f in spec.fields if f.role == ROLE_ISSUE), None)

    candidates = []
    for ln in lines:
        for m in DATE_RE.finditer(ln.get("text", "")):
            candidates.append({
                "value": f"{m.group(1)}/{m.group(2)}/{m.group(3)}",
                "line": ln,
                "vertical": _is_vertical(ln),
            })
    if not candidates:
        return out

    def label_hit(text: str, labels) -> bool:
        t = text.lower()
        return any(lb.lower() in t for lb in labels)

    used: set[int] = set()

    # Pass 1 -- a date sharing its line with a role label binds immediately.
    for spc, key in ((dob_spec, "dob"), (iss_spec, "issue_date")):
        if spc is None:
            continue
        for i, c in enumerate(candidates):
            if i in used:
                continue
            if label_hit(c["line"].get("text", ""), spc.labels):
                out[key] = ResolvedField(
                    key=key, role=spc.role, value=c["value"],
                    confidence=float(c["line"].get("conf", 0.8)),
                    source="LABEL_PROXIMITY",
                    plausible=bool(spc.plausibility(c["value"])
                                   if spc.plausibility else True),
                )
                used.add(i)
                break

    # Pass 2 -- nearest unbound label elsewhere on the page.
    for spc, key in ((dob_spec, "dob"), (iss_spec, "issue_date")):
        if spc is None or key in out:
            continue
        label_boxes = [ln for ln in lines
                       if label_hit(ln.get("text", ""), spc.labels)]
        if not label_boxes:
            continue
        best, best_d = None, 1e9
        for i, c in enumerate(candidates):
            if i in used:
                continue
            d = min(_box_distance(c["line"], lb) for lb in label_boxes)
            if d < best_d:
                best, best_d, best_i = c, d, i
        if best is not None and best_d < 0.25:
            out[key] = ResolvedField(
                key=key, role=spc.role, value=best["value"],
                confidence=float(best["line"].get("conf", 0.7)) * 0.9,
                source="LABEL_PROXIMITY",
                plausible=bool(spc.plausibility(best["value"])
                               if spc.plausibility else True),
            )
            used.add(best_i)

    # Pass 3 -- geometric fallback. A VERTICAL date is an issue/print date,
    # never a DOB. This is the specific rule that would have stopped #207.
    if "dob" not in out and dob_spec is not None:
        horiz = [(i, c) for i, c in enumerate(candidates)
                 if i not in used and not c["vertical"]]
        if horiz:
            i, c = horiz[0]
            out["dob"] = ResolvedField(
                key="dob", role=ROLE_DOB, value=c["value"],
                confidence=0.55, source="POSITIONAL",
                plausible=bool(dob_spec.plausibility(c["value"])
                               if dob_spec.plausibility else True),
                issues=["DOB_BOUND_WITHOUT_LABEL_LOW_CONFIDENCE"],
            )
            used.add(i)
        else:
            vert = [(i, c) for i, c in enumerate(candidates) if i not in used]
            if vert:
                out["dob"] = ResolvedField(
                    key="dob", role=ROLE_DOB, value=None, confidence=0.0,
                    source="POSITIONAL", plausible=False,
                    issues=["ONLY_VERTICAL_DATES_FOUND_REFUSING_DOB_BINDING"],
                )

    # Sanity -- DOB must precede issue date.
    if "dob" in out and "issue_date" in out and out["dob"].value and out["issue_date"].value:
        d1, d2 = _parse_date(out["dob"].value), _parse_date(out["issue_date"].value)
        if d1 and d2 and d1 > d2:
            out["dob"].issues.append("DOB_AFTER_ISSUE_DATE")
            out["dob"].plausible = False

    return out


def resolve_name(lines: list[dict], spec: DocumentSpec) -> ResolvedField:
    """Aadhaar prints the holder name on an unlabelled line above the DOB.

    A label-driven extractor finds nothing, which is why #207 reported
    'Holder Name Missing or Unreadable in OCR'. Strategy: take Latin-script
    title-case lines that are not template boilerplate, preferring one that
    sits directly above a DOB line.
    """
    name_spec = next((f for f in spec.fields if f.role == ROLE_NAME), None)
    labels = name_spec.labels if name_spec else ()

    # Labelled form first (PAN prints "Name").
    for ln in lines:
        t = _norm(ln.get("text", ""))
        for lb in labels:
            if lb and t.lower().startswith(lb.lower()):
                val = _norm(t[len(lb):].lstrip(":/ -"))
                if val and val.lower() not in NAME_STOPWORDS:
                    return ResolvedField("full_name", ROLE_NAME, val,
                                         float(ln.get("conf", 0.85)),
                                         "LABEL_PROXIMITY", True)

    dob_y = None
    for ln in lines:
        if DATE_RE.search(ln.get("text", "")) and not _is_vertical(ln):
            dob_y = ln["y0"]
            break

    best, best_score = None, -1.0
    for ln in lines:
        t = _norm(ln.get("text", ""))
        if not LATIN_NAME_RE.match(t):
            continue
        if any(w in t.lower() for w in NAME_STOPWORDS):
            continue
        score = float(ln.get("conf", 0.5))
        if dob_y is not None and ln["y1"] <= dob_y:
            score += 0.5 * (1.0 - min(abs(dob_y - ln["y1"]), 0.3) / 0.3)
        if score > best_score:
            best, best_score = ln, score

    if best is not None:
        return ResolvedField("full_name", ROLE_NAME, _norm(best["text"]),
                             round(min(best_score, 0.95), 3),
                             "POSITIONAL", True,
                             ["NAME_BOUND_WITHOUT_LABEL"])

    return ResolvedField("full_name", ROLE_NAME, None, 0.0,
                         "REGEX_FALLBACK", False,
                         ["NAME_NOT_FOUND",
                          "CHECK_OCR_SCRIPT_MODEL_SUPPORTS_DOCUMENT_LANGUAGE"])


# ---------------------------------------------------------------------------
def resolve_structure(*,
                      spec: DocumentSpec,
                      lines: list[dict],
                      sides_present: set[str],
                      extra_fields: dict[str, Any] | None = None,
                      checksum_valid: bool | None = None,
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

    # ---- side completeness ------------------------------------------------
    r.sides_missing = spec.unsubmitted_sides(sides_present)
    r.submission_complete = not r.sides_missing
    if r.sides_missing:
        r.reasons.append("INCOMPLETE_SUBMISSION_MISSING_SIDE")
        for s in r.sides_missing:
            r.reasons.append(f"SIDE_NOT_PROVIDED_{s}")

    # ---- required fields, scoped to submitted sides ----------------------
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
    r.completeness = round(got / float(len(required)), 4) if required else 0.0

    # ---- gated validity score --------------------------------------------
    # THE #207 FIX. Completeness is a multiplier, not an addend. Two of four
    # required fields can no longer yield 86%.
    base = 0.55 * r.completeness + 0.25 * float(layout_score)
    if checksum_valid is True:
        base += 0.20
    elif checksum_valid is False:
        base = min(base, 0.35)
        r.reasons.append("CHECKSUM_FAILED")

    if r.missing_required:
        base = min(base, 0.60)
    if r.role_conflicts:
        base = min(base, 0.65)
    if not r.submission_complete:
        base = min(base, 0.70)

    r.validity_score = round(max(0.0, min(1.0, base)), 4)
    r.structurally_valid = (
        r.validity_score >= 0.85
        and not r.missing_required
        and not r.role_conflicts
        and r.submission_complete
    )

    if r.structurally_valid:
        r.reasons.append("DOCUMENT_STRUCTURE_VALID")
    r.reasons = list(dict.fromkeys(r.reasons))
    return r
