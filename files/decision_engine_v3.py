"""
SENTINEL V3 — Stage 17 replacement: Decision Engine
====================================================

The V2 failure mode
-------------------
Your screenshot shows:
    Document Validity 100%  |  AI 55.7% INCONCLUSIVE  |  Tampering 6.5%
    -> REQUIRES REVIEW / UNCALIBRATED MODEL
    -> reason codes include AUTHENTICITY_NOT_ESTABLISHED

That is a *degenerate* engine. With `is_calibrated=False` every input reaches
MANUAL_REVIEW, because:
    - GENUINE required "positive genuine evidence", which nothing could emit
    - AI_GENERATED required corroboration, which the collapsed model blocked
    - the 0.35 < p < 0.80 inconclusive band swallowed the constant 0.557

A fail-closed system that closes on 100% of traffic has a throughput of zero
and provides no security -- officers learn to rubber-stamp the review queue.

The V3 contract
---------------
Fail-closed is preserved, but the engine distinguishes three states that V2
conflated into one:

    CONCLUSIVE        -> evidence supports a verdict
    INSUFFICIENT      -> we could not measure (recapture / enable a module)
    CONTRADICTORY     -> families disagree; a human must adjudicate

and it can reach GENUINE on *structural + provenance* evidence alone when the
AI family is offline. The AI detector becomes an escalation channel, not a
gate on the whole pipeline.
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
    "tamper_strong":            0.75,
    "tamper_low":               0.50,
    "genuine_evidence_min":     0.60,
}

STATUSES = ("GENUINE", "AI_GENERATED", "ALTERED", "AI_GENERATED_AND_ALTERED",
            "SUSPICIOUS", "MANUAL_REVIEW")


@dataclass
class Decision:
    status: str = "MANUAL_REVIEW"
    evidence_state: str = "INSUFFICIENT"   # CONCLUSIVE|INSUFFICIENT|CONTRADICTORY
    confidence: float = 0.0
    requires_manual_review: bool = True
    reason_codes: list[str] = field(default_factory=list)
    officer_action: str = ""
    blocking_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(*,
           fusion,
           document_validity: dict[str, Any],
           image_quality: dict[str, Any],
           blacklist_hit: bool = False,
           thresholds: dict[str, float] | None = None) -> Decision:
    T = {**THRESHOLDS, **(thresholds or {})}
    d = Decision()
    rc = d.reason_codes

    ai = float(fusion.ai_probability)
    tamper = float(fusion.tampering_probability)
    genuine_ev = float(fusion.genuine_evidence)
    corroborated = bool(fusion.corroborated)

    dv = float(document_validity.get("score", 0.0))
    q = float(image_quality.get("score", 0.0))

    # ---- Gate 1: hard structural / watchlist failures --------------------
    if blacklist_hit:
        d.status = "SUSPICIOUS"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = 0.99
        d.requires_manual_review = True
        rc.append("BLACKLIST_WATCHLIST_MATCH")
        d.officer_action = "Detain document. Watchlist match on document number."
        return d

    if dv < T["doc_validity_suspicious"]:
        d.status = "SUSPICIOUS"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(1.0 - dv, 3)
        rc.append("DOCUMENT_STRUCTURE_INVALID")
        rc.extend(document_validity.get("failed_rules", [])[:6])
        d.officer_action = "Structural validation failed. Refer to secondary inspection."
        return d

    # ---- Gate 2: capture quality ----------------------------------------
    if q < T["quality_floor"] or image_quality.get("quality_too_low_for_forensics"):
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "INSUFFICIENT"
        d.confidence = 0.30
        rc.append("IMAGE_QUALITY_BELOW_FORENSIC_FLOOR")
        d.blocking_gaps.append("RECAPTURE_REQUIRED")
        d.officer_action = ("Recapture the document: flat surface, no glare, "
                            "fill the frame, keep the QR region in focus.")
        return d

    # ---- Gate 3: corroborated synthesis / tampering ----------------------
    ai_strong = corroborated and ai >= T["ai_strong"]
    tamper_strong = tamper >= T["tamper_strong"]

    if ai_strong and tamper_strong:
        d.status = "AI_GENERATED_AND_ALTERED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, (ai + tamper) / 2.0), 3)
        rc += ["AI_GENERATION_CORROBORATED", "DIGITAL_TAMPERING_DETECTED"]
        rc += [f"STRONG_FAMILY_{f}" for f in fusion.strong_families]
        d.officer_action = "Synthetic document with post-generation editing. Seize and report."
        return d

    if ai_strong:
        d.status = "AI_GENERATED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, ai), 3)
        rc.append("AI_GENERATION_CORROBORATED")
        rc += [f"STRONG_FAMILY_{f}" for f in fusion.strong_families]
        d.officer_action = "Document is a synthetic render. Seize and report."
        return d

    if tamper_strong:
        d.status = "ALTERED"
        d.evidence_state = "CONCLUSIVE"
        d.confidence = round(min(0.99, tamper), 3)
        rc.append("DIGITAL_TAMPERING_DETECTED")
        d.officer_action = "Genuine stock with altered fields. Refer to secondary inspection."
        return d

    # ---- Gate 4: contradiction -------------------------------------------
    # High structural validity but an anchor family screaming synthesis, or
    # vice versa. Never auto-clear a contradiction.
    if dv >= T["doc_validity_genuine"] and fusion.anchor_present and ai >= 0.50:
        d.status = "MANUAL_REVIEW"
        d.evidence_state = "CONTRADICTORY"
        d.confidence = 0.40
        rc += ["STRUCTURE_VALID_BUT_PROVENANCE_ANOMALOUS"]
        rc += [f"STRONG_FAMILY_{f}" for f in fusion.strong_families]
        d.officer_action = ("Fields are internally consistent but the capture "
                            "chain is anomalous. Verify against the issuing "
                            "authority portal.")
        return d

    # ---- Gate 5: positive clearance --------------------------------------
    # THE FIX for AUTHENTICITY_NOT_ESTABLISHED. Two independent routes to
    # GENUINE so a single offline module cannot deadlock the pipeline.
    low_ai = ai <= T["ai_low"]
    low_tamper = tamper < T["tamper_low"]
    structurally_sound = dv >= T["doc_validity_genuine"]

    route_a = structurally_sound and low_ai and low_tamper and \
        genuine_ev >= T["genuine_evidence_min"]
    route_b = structurally_sound and low_ai and low_tamper and \
        fusion.available_family_count >= 2 and q >= T["quality_forensic_floor"]

    if route_a or route_b:
        d.status = "GENUINE"
        d.evidence_state = "CONCLUSIVE"
        d.requires_manual_review = False
        d.confidence = round(min(0.98, 0.55 + 0.35 * fusion.ai_confidence +
                                 0.10 * genuine_ev), 3)
        rc += ["DOCUMENT_STRUCTURE_VALID", "AI_EVIDENCE_LOW",
               "TAMPERING_EVIDENCE_LOW"]
        rc.append("AUTHENTICITY_CONFIRMED_BY_PROVENANCE" if route_a
                  else "AUTHENTICITY_CONFIRMED_BY_MULTI_FAMILY_CONSENSUS")
        d.officer_action = "Clear for processing."
        return d

    # ---- Gate 6: genuine remainder -> explain WHY we cannot clear --------
    d.status = "MANUAL_REVIEW"
    d.evidence_state = "INSUFFICIENT"
    d.confidence = round(0.30 + 0.30 * fusion.ai_confidence, 3)

    if fusion.available_family_count < 2:
        rc.append("INSUFFICIENT_INDEPENDENT_EVIDENCE_FAMILIES")
    for name, fam in fusion.families.items():
        if not fam["available"]:
            d.blocking_gaps.append(f"FAMILY_UNAVAILABLE_{name}")
    if T["ai_low"] < ai < T["ai_strong"]:
        rc.append("AI_EVIDENCE_EQUIVOCAL")
    if q < T["quality_forensic_floor"]:
        d.blocking_gaps.append("QUALITY_BELOW_FORENSIC_CONFIDENCE")
        rc.append("QUALITY_LIMITS_FORENSIC_CERTAINTY")
    if not d.blocking_gaps and not rc:
        rc.append("NO_DECISIVE_EVIDENCE")

    d.officer_action = ("Manual adjudication required: "
                        + "; ".join(d.blocking_gaps[:3] or ["review evidence panel"]))
    return d
