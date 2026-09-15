"""
SENTINEL V3 — Stage 16 replacement: Signal-Family Corroboration Fusion
======================================================================

What was wrong with the V2 fusion
---------------------------------
V2 summed five weighted signals and required "2 strong signals" to corroborate.
On the reference case it produced `strong_signal_count = 0/4` and
`ai_probability = 55.7%`. Three structural defects:

  DEFECT 1 — An uncalibrated model was still allowed to contribute 50% of the
             weight. `is_weights_loaded=True` and `is_calibrated=False` were
             treated as the same state. A model whose output is a near-constant
             0.55 dragged the fused score to the exact centre of the decision
             space, which is the one place where no verdict is reachable.

  DEFECT 2 — Signal independence was assumed but not enforced. `global_prob`,
             `patch_topk`, and `noise_anomaly` are not independent: all three
             measure high-frequency surface statistics. Counting them as three
             corroborating signals is double-counting one physical measurement.
             Conversely the reference case showed them agreeing at ~0.55 — so
             the "2 signal" rule could never fire from one family alone.

  DEFECT 3 — Missing evidence was scored as 0.0 (benign) rather than excluded.
             A module that could not measure anything voted "looks genuine".

The V3 rule
-----------
Signals are partitioned into four *physically independent families*:

  A. PROVENANCE  — container, EXIF, ICC, quantisation history
  B. SEMANTIC    — QR codeword validity, payload/print cross-check, checksums
  C. TEXTURE     — local variance, spectral rolloff  (one family, not three)
  D. LEARNED     — ConvNeXt global + patch        (one family, not two)

Corroboration requires strong signals from >= 2 DIFFERENT families, and at
least one of those must be from A or B. Families C and D are the ones that
degrade under recapture; A and B do not. That constraint is what stops a
blurry genuine card from being labelled AI_GENERATED.

Every family reports `available` independently. Unavailable families are
renormalised out of the weighted mean instead of voting zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Weight within the fused AI score, applied only to AVAILABLE families.
FAMILY_WEIGHTS = {
    "PROVENANCE": 0.35,
    "SEMANTIC":   0.35,
    "TEXTURE":    0.20,
    "LEARNED":    0.10,   # deliberately small until the model is calibrated
}

# A family contributes a "strong signal" above this level.
STRONG = 0.60
# Families that survive recapture and may anchor a corroborated verdict.
ANCHOR_FAMILIES = {"PROVENANCE", "SEMANTIC"}


@dataclass
class FamilySignal:
    name: str
    available: bool
    score: float = 0.0          # 0..1 synthesis evidence
    reliability: float = 1.0    # 0..1 confidence in this measurement
    reasons: list[str] = field(default_factory=list)

    @property
    def strong(self) -> bool:
        return self.available and self.score >= STRONG and self.reliability >= 0.5


@dataclass
class FusionResult:
    ai_probability: float = 0.0
    ai_confidence: float = 0.0
    corroborated: bool = False
    strong_signal_count: int = 0
    strong_families: list[str] = field(default_factory=list)
    anchor_present: bool = False

    genuine_evidence: float = 0.0
    tampering_probability: float = 0.0

    families: dict[str, dict] = field(default_factory=dict)
    available_family_count: int = 0
    total_weight_available: float = 0.0

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fuse(provenance,
         qr,
         texture,
         learned: dict[str, Any] | None,
         tampering: dict[str, Any] | None = None) -> FusionResult:
    """
    provenance : ProvenanceResult   (modules.provenance)
    qr         : QRResult           (modules.qr_integrity)
    texture    : TextureResult      (modules.texture_forensics)
    learned    : dict with keys global_probability, patch_topk_probability,
                 is_weights_loaded, is_calibrated
    tampering  : dict from modules.tampering_detector
    """
    r = FusionResult()
    learned = learned or {}
    tampering = tampering or {}

    sigs: list[FamilySignal] = []

    # ---- A. PROVENANCE ---------------------------------------------------
    sigs.append(FamilySignal(
        name="PROVENANCE",
        available=bool(getattr(provenance, "evidence_available", False)),
        score=float(getattr(provenance, "synthetic_provenance_score", 0.0)),
        reliability=1.0,
        reasons=list(getattr(provenance, "reasons", [])),
    ))

    # ---- B. SEMANTIC -----------------------------------------------------
    sem_available = bool(getattr(qr, "evidence_available", False))
    sem_score = max(float(getattr(qr, "synthetic_qr_score", 0.0)),
                    float(getattr(qr, "payload_mismatch_score", 0.0)))
    sigs.append(FamilySignal(
        name="SEMANTIC",
        available=sem_available,
        score=sem_score,
        reliability=1.0,
        reasons=list(getattr(qr, "reasons", [])),
    ))

    # ---- C. TEXTURE ------------------------------------------------------
    sigs.append(FamilySignal(
        name="TEXTURE",
        available=bool(getattr(texture, "evidence_available", False)),
        score=float(getattr(texture, "texture_ai_score", 0.0)),
        reliability=float(getattr(texture, "reliability", 0.0)),
        reasons=list(getattr(texture, "reasons", [])),
    ))

    # ---- D. LEARNED ------------------------------------------------------
    # THE CENTRAL FIX. An uncalibrated model is not evidence.
    loaded = bool(learned.get("is_weights_loaded", False))
    calibrated = bool(learned.get("is_calibrated", False))
    g = float(learned.get("global_probability", 0.0) or 0.0)
    p = float(learned.get("patch_topk_probability", 0.0) or 0.0)

    learned_reasons: list[str] = []
    if not loaded:
        learned_available, learned_score, learned_rel = False, 0.0, 0.0
        learned_reasons.append("AI_MODEL_WEIGHTS_NOT_LOADED")
    elif not calibrated:
        learned_available, learned_score, learned_rel = False, 0.0, 0.0
        learned_reasons.append("AI_MODEL_UNCALIBRATED_EXCLUDED_FROM_FUSION")
    else:
        learned_available = True
        # Global and patch are the SAME family; take the max, never sum.
        learned_score = max(g, p)
        # Agreement between scales raises confidence; divergence lowers it.
        learned_rel = 1.0 - min(abs(g - p), 0.5)
        learned_reasons.append("AI_MODEL_CALIBRATED")

    sigs.append(FamilySignal("LEARNED", learned_available,
                             learned_score, learned_rel, learned_reasons))

    # ---- Weighted mean over AVAILABLE families only ----------------------
    num = den = 0.0
    for s in sigs:
        r.families[s.name] = asdict(s) | {"strong": s.strong}
        if not s.available:
            continue
        w = FAMILY_WEIGHTS[s.name] * max(s.reliability, 0.1)
        num += w * s.score
        den += w
        r.reasons.extend(s.reasons)

    r.available_family_count = sum(1 for s in sigs if s.available)
    r.total_weight_available = round(den, 4)
    r.ai_probability = round(num / den, 4) if den > 0 else 0.0

    # ---- Corroboration ---------------------------------------------------
    strong = [s for s in sigs if s.strong]
    r.strong_families = [s.name for s in strong]
    r.strong_signal_count = len(strong)
    r.anchor_present = any(s.name in ANCHOR_FAMILIES for s in strong)
    r.corroborated = (len(strong) >= 2 and r.anchor_present)

    if r.strong_signal_count == 1 and not r.corroborated:
        # Single uncorroborated signal is capped -- unchanged policy from V2,
        # but now it actually has a chance of being corroborated because the
        # families are independent.
        r.ai_probability = min(r.ai_probability, 0.35)
        r.reasons.append("SINGLE_UNCORROBORATED_SIGNAL_CAPPED")

    if r.strong_signal_count >= 2 and not r.anchor_present:
        r.ai_probability = min(r.ai_probability, 0.55)
        r.reasons.append("NO_RECAPTURE_ROBUST_ANCHOR_SIGNAL")

    # ---- Positive genuineness evidence -----------------------------------
    # V2 had no way to ever establish authenticity, which is why every
    # document ended at AUTHENTICITY_NOT_ESTABLISHED.
    gen = 0.0
    if getattr(qr, "genuine_evidence", 0.0) > 0:
        gen = max(gen, float(qr.genuine_evidence))
    if getattr(provenance, "has_optical_exif", False) and \
       getattr(provenance, "camera_model", None):
        gen = max(gen, 0.75)
        r.reasons.append("OPTICAL_CAPTURE_CHAIN_ESTABLISHED")
    r.genuine_evidence = round(gen, 4)

    # ---- Confidence ------------------------------------------------------
    # How much of the total evidence budget did we actually collect?
    coverage = den / sum(FAMILY_WEIGHTS.values())
    decisiveness = abs(r.ai_probability - 0.5) * 2.0
    r.ai_confidence = round(min(1.0, 0.5 * coverage + 0.5 * decisiveness), 4)

    # ---- Tampering axis stays independent of the AI axis -----------------
    tamper = float(tampering.get("probability", 0.0) or 0.0)
    editor = float(getattr(provenance, "editor_provenance_score", 0.0))
    r.tampering_probability = round(max(tamper, 0.85 * editor), 4)

    r.reasons = list(dict.fromkeys(r.reasons))
    return r
