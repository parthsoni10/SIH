"""
SENTINEL V3.1 — Decision Engine
================================
Extends decision_engine_v3 with the three gates that verification #207 lacked.

New status
----------
`INCOMPLETE_SUBMISSION` is added as a seventh outcome. It is deliberately NOT
folded into MANUAL_REVIEW or SUSPICIOUS. A front-only Aadhaar is not
suspicious and does not need an officer's judgement -- it needs the back side.
Conflating "you haven't shown me everything" with "this looks forged" is what
makes review queues unusable.

Gate ordering (first match wins)
--------------------------------
  1. blacklist
  2. corroborated alteration        <- NEW, catches the erased Aadhaar logo
  3. structural invalidity
  4. incomplete submission          <- NEW, catches front-only two-sided docs
  5. required-field gaps            <- NEW, catches the unread holder name
  6. capture quality
  7. corroborated synthesis
  8. contradiction
  9. positive clearance
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

THRESHOLDS = {
    "quality_floor":            0.45,
    "quality_forensic_floor":   0.60,
    "doc_validity_genuine":     0.85,
    "doc_validity_suspicious":  0.50,
    "ai_strong":                0.80,
    "ai_low":                   0.35,
    "alteration_strong":        0.75,
    "alteration_low":           0.50,
    "genuine_evidence_min":     0.60,
}

STATUSES = ("GENUINE", "AI_GENERATED", "ALTERED", "AI_GENERATED_AND_ALTERED",
            "SUSPICIOUS", "MANUAL_REVIEW", "INCOMPLETE_SUBMISSION")


@dataclass
class Decision:
    status: str = "MANUAL_REVIEW"
    evidence_state: str = "INSUFFICIENT"
    confidence: float = 0.0
    requires_manual_review: bool = True
    reason_codes: list[str] = field(default_factory=list)
    officer_action: str = ""
    blocking_gaps: list[str] = field(default_factory=list)
    suspect_regions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(*,
           fusion,
           structure,
           alteration,
           image_quality: dict[str, Any],
           blacklist_hit: bool = False,
           thresholds: dict[str, float] | None = None) -> Decision:
    """
    fusion        : FusionResult        (forensic_fusion_v3)
    structure     : FieldResolution     (field_resolver)
    alteration    : AlterationResult    (alteration_detector)
    image_quality : dict from image_quality.py
    """
    T = {**THRESHOLDS, **(thresholds or {})}
    d = Decision()
    rc = d.reason_codes

    ai = float(fusion.ai_probability)
    ai_corroborated = bool(fusion.corroborated)
    genuine_ev = float(fusion.genuine_evidence)

    alt = float(alteration.probability)
    alt_corroborated = bool(alteration.corroborated)
    d.suspect_regions = list(alteration.suspect_regions)

    dv = float(structure.validity_score)
    q = float(image_quality.get("score", 0.0))

    # ---- Gate 1: watchlist ----------------------------------------------
    if blacklist_hit:
        d.status = "SUSPICIOUS"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = 0.99
        rc.append("BLACKLIST_WATCHLIST_MATCH")
        d.officer_action = "Detain document. Watchlist match on document number."
        return d

    # ---- Gate 2: corroborated alteration --------------------------------
    alt_strong = alt_corroborated and alt >= T["alteration_strong"]
    ai_strong = ai_corroborated and ai >= T["ai_strong"]

    if alt_strong and ai_strong:
        d.status = "AI_GENERATED_AND_ALTERED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, (ai + alt) / 2.0), 3)
        rc += ["AI_GENERATION_CORROBORATED", "ALTERATION_CORROBORATED"]
        rc += [f"ALTERATION_CUE_{c.upper()}" for c in alteration.strong_cues]
        d.officer_action = "Synthetic document with post-generation editing. Seize and report."
        return d

    if alt_strong:
        d.status = "ALTERED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, alt), 3)
        rc.append("ALTERATION_CORROBORATED")
        rc += [f"ALTERATION_CUE_{c.upper()}" for c in alteration.strong_cues]
        for a in alteration.cues.get("anchor_removal", {}).get("detail", {}).get("missing_anchors", []):
            rc.append(f"ANCHOR_MISSING_{a.upper()}")
        d.officer_action = ("Genuine document stock with removed or overwritten "
                            "security artwork. Seize and refer to secondary inspection.")
        return d

    # ---- Gate 3: structural invalidity -----------------------------------
    if dv < T["doc_validity_suspicious"]:
        d.status = "SUSPICIOUS"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(1.0 - dv, 3)
        rc.append("DOCUMENT_STRUCTURE_INVALID")
        rc.extend(structure.reasons[:8])
        d.officer_action = "Structural validation failed. Refer to secondary inspection."
        return d

    # ---- Gate 4: incomplete submission -----------------------------------
    if not structure.submission_complete:
        d.status = "INCOMPLETE_SUBMISSION"
        d.evidence_state = "INSUFFICIENT"
        d.confidence = 0.0
        rc.append("INCOMPLETE_SUBMISSION_MISSING_SIDE")
        for s in structure.sides_missing:
            rc.append(f"SIDE_NOT_PROVIDED_{s}")
            d.blocking_gaps.append(f"UPLOAD_{s}_SIDE")
        d.officer_action = ("Document not fully presented. Capture the "
                            + " and ".join(s.lower() for s in structure.sides_missing)
                            + " side and resubmit. No verdict is admissible until then.")
        return d

    # ---- Gate 5: required-field gaps -------------------------------------
    if structure.missing_required or structure.role_conflicts:
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "INSUFFICIENT"
        d.confidence = 0.35
        for k in structure.missing_required:
            rc.append(f"REQUIRED_FIELD_MISSING_{k.upper()}")
            d.blocking_gaps.append(f"FIELD_{k.upper()}")
        for k in structure.role_conflicts:
            rc.append(f"FIELD_ROLE_CONFLICT_{k.upper()}")
            d.blocking_gaps.append(f"VERIFY_FIELD_{k.upper()}")
        d.officer_action = ("Mandatory fields could not be read or failed "
                            "plausibility. Verify printed values manually: "
                            + ", ".join(structure.missing_required
                                        + structure.role_conflicts))
        return d

    # ---- Gate 6: capture quality -----------------------------------------
    if q < T["quality_floor"] or image_quality.get("quality_too_low_for_forensics"):
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "INSUFFICIENT"
        d.confidence = 0.30
        rc.append("IMAGE_QUALITY_BELOW_FORENSIC_FLOOR")
        d.blocking_gaps.append("RECAPTURE_REQUIRED")
        d.officer_action = ("Recapture: flat surface, no glare, fill the frame, "
                            "keep security artwork and any QR in focus.")
        return d

    # ---- Gate 7: corroborated synthesis ----------------------------------
    if ai_strong:
        d.status = "AI_GENERATED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, ai), 3)
        rc.append("AI_GENERATION_CORROBORATED")
        rc += [f"STRONG_FAMILY_{f}" for f in fusion.strong_families]
        d.officer_action = "Document is a synthetic render. Seize and report."
        return d

    # ---- Gate 8: contradiction -------------------------------------------
    if dv >= T["doc_validity_genuine"] and fusion.anchor_present and ai >= 0.50:
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "CONTRADICTORY"
        d.confidence = 0.40
        rc.append("STRUCTURE_VALID_BUT_PROVENANCE_ANOMALOUS")
        rc += [f"STRONG_FAMILY_{f}" for f in fusion.strong_families]
        d.officer_action = ("Fields consistent but capture chain anomalous. "
                            "Verify against the issuing authority portal.")
        return d

    if alt >= T["alteration_low"] and not alt_corroborated:
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "CONTRADICTORY"
        d.confidence = 0.40
        rc.append("ALTERATION_EVIDENCE_UNCORROBORATED")
        d.blocking_gaps.append("REVIEW_SUSPECT_REGIONS")
        d.officer_action = "Possible localised editing. Inspect highlighted regions."
        return d

    # ---- Gate 9: positive clearance --------------------------------------
    low_ai = ai <= T["ai_low"]
    low_alt = alt < T["alteration_low"]
    sound = structure.structurally_valid

    route_a = sound and low_ai and low_alt and genuine_ev >= T["genuine_evidence_min"]
    route_b = (sound and low_ai and low_alt
               and fusion.available_family_count >= 2
               and alteration.available_cue_count >= 2
               and q >= T["quality_forensic_floor"])

    if route_a or route_b:
        d.status = "GENUINE"
        d.evidence_state = "CONCLUSIVE"
        d.requires_manual_review = False
        d.confidence = round(min(0.98, 0.55 + 0.30 * fusion.ai_confidence
                                 + 0.15 * alteration.confidence), 3)
        rc += ["DOCUMENT_STRUCTURE_VALID", "ALL_REQUIRED_FIELDS_RESOLVED",
               "AI_EVIDENCE_LOW", "ALTERATION_EVIDENCE_LOW"]
        rc.append("AUTHENTICITY_CONFIRMED_BY_PROVENANCE" if route_a
                  else "AUTHENTICITY_CONFIRMED_BY_MULTI_FAMILY_CONSENSUS")
        if alteration.background_flattened:
            rc.append("NOTE_PIXEL_FORENSICS_LIMITED_BY_SCANNER_PREPROCESSING")
        d.officer_action = "Clear for processing."
        return d

    # ---- Fallback ---------------------------------------------------------
    d.status = "MANUAL_REVIEW"
    d.evidence_state = "INSUFFICIENT"
    d.confidence = round(0.30 + 0.30 * fusion.ai_confidence, 3)
    if fusion.available_family_count < 2:
        rc.append("INSUFFICIENT_INDEPENDENT_EVIDENCE_FAMILIES")
    if alteration.available_cue_count < 2:
        rc.append("INSUFFICIENT_ALTERATION_CUES")
        d.blocking_gaps.append("ALTERATION_CUES_UNAVAILABLE")
    if q < T["quality_forensic_floor"]:
        rc.append("QUALITY_LIMITS_FORENSIC_CERTAINTY")
        d.blocking_gaps.append("QUALITY_BELOW_FORENSIC_CONFIDENCE")
    if not rc:
        rc.append("NO_DECISIVE_EVIDENCE")
    d.officer_action = ("Manual adjudication required: "
                        + "; ".join(d.blocking_gaps[:3] or ["review evidence panel"]))
    return d
