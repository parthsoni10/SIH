from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

DecisionStatus = Literal[
    "GENUINE",
    "AI_GENERATED",
    "ALTERED",
    "AI_GENERATED_AND_ALTERED",
    "SUSPICIOUS",
    "MANUAL_REVIEW",
    "INCOMPLETE_SUBMISSION",
]

class FinalDecisionPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: DecisionStatus

    # Confidence in the FINAL DECISION, not AI probability.
    # None means confidence is unavailable / not calibrated.
    decision_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence in the final decision. None if unavailable.",
    )

    # Deprecated alias for decision_confidence.
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for decision_confidence."
    )

    requires_manual_review: bool

    reason_codes: List[str] = Field(default_factory=list)

    # Explicitly separate structure from authenticity.
    structural_validity: Optional[bool] = None

    authenticity_status: Optional[str] = None

    authenticity_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )
