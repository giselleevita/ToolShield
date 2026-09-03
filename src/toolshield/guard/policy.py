"""Deterministic enforcement policy layered on top of detector scores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
ToolRisk = Literal["read", "write", "privileged"]


@dataclass(frozen=True)
class EnforcementPolicy:
    version: str = "1.0.0"
    block_threshold: float = 0.70
    review_margin: float = 0.15
    write_adjustment: float = 0.10
    privileged_adjustment: float = 0.20
    fail_closed: bool = True

    def config_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    effective_threshold: float
    review_threshold: float
    policy_version: str
    policy_hash: str
    reasons: tuple[str, ...]


def evaluate_policy(
    score: float | None,
    *,
    tool_risk: ToolRisk = "read",
    policy: EnforcementPolicy | None = None,
) -> PolicyDecision:
    """Convert a detector score into an auditable tool-execution decision."""
    selected = policy or EnforcementPolicy()
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        decision: Decision = "BLOCK" if selected.fail_closed else "REVIEW"
        return PolicyDecision(
            decision=decision,
            effective_threshold=selected.block_threshold,
            review_threshold=max(0.0, selected.block_threshold - selected.review_margin),
            policy_version=selected.version,
            policy_hash=selected.config_hash(),
            reasons=("invalid_or_missing_detector_score", "fail_closed" if selected.fail_closed else "manual_review"),
        )

    adjustment = {"read": 0.0, "write": selected.write_adjustment, "privileged": selected.privileged_adjustment}[tool_risk]
    block_at = max(0.0, min(1.0, selected.block_threshold - adjustment))
    review_at = max(0.0, block_at - selected.review_margin)

    if score >= block_at:
        decision = "BLOCK"
        reasons = ("detector_score_exceeds_risk_adjusted_threshold", f"tool_risk:{tool_risk}")
    elif score >= review_at:
        decision = "REVIEW"
        reasons = ("detector_score_in_review_band", f"tool_risk:{tool_risk}")
    else:
        decision = "ALLOW"
        reasons = ("detector_score_below_review_band", f"tool_risk:{tool_risk}")

    return PolicyDecision(
        decision=decision,
        effective_threshold=block_at,
        review_threshold=review_at,
        policy_version=selected.version,
        policy_hash=selected.config_hash(),
        reasons=reasons,
    )
