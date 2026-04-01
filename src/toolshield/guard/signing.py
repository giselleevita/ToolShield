from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VerificationInfo(BaseModel):
    algorithm: str = "hmac-sha256"
    key_id: str = "env:toolshield-signing-secret"


class PolicyArtifact(BaseModel):
    model_path: str
    model_config_hash: str
    thresholds: dict[str, float]
    selected_budget: float


class DecisionProvenance(BaseModel):
    decision: str
    score: float
    threshold: float
    explanation: str
    prompt_hash: str
    tool_name: str | None = None
    role_sequence: list[str] | None = None


class SignedDecisionRecord(BaseModel):
    audit_id: str
    signed_at: str
    payload_hash: str
    signature: str
    verification: VerificationInfo = Field(default_factory=VerificationInfo)
    policy_artifact: PolicyArtifact
    provenance: DecisionProvenance


class GuardSigner:
    def __init__(self, secret: str | None) -> None:
        self._secret = secret.encode("utf-8") if secret else None

    @property
    def enabled(self) -> bool:
        return self._secret is not None

    @classmethod
    def from_environment(cls) -> "GuardSigner":
        return cls(os.getenv("TOOLSHIELD_SIGNING_SECRET"))

    def sign(
        self,
        *,
        audit_id: str,
        prompt_hash: str,
        decision: str,
        score: float,
        threshold: float,
        explanation: str,
        fpr_budget: float,
        tool_name: str | None,
        role_sequence: list[str] | None,
        model_path: str,
        thresholds: dict[float, float],
    ) -> SignedDecisionRecord | None:
        if not self._secret:
            return None
        provenance = DecisionProvenance(
            decision=decision,
            score=score,
            threshold=threshold,
            explanation=explanation,
            prompt_hash=prompt_hash,
            tool_name=tool_name,
            role_sequence=role_sequence,
        )
        policy_artifact = PolicyArtifact(
            model_path=model_path,
            model_config_hash=self._model_config_hash(model_path),
            thresholds={str(key): float(value) for key, value in sorted(thresholds.items())},
            selected_budget=fpr_budget,
        )
        payload = {
            "audit_id": audit_id,
            "policy_artifact": policy_artifact.model_dump(),
            "provenance": provenance.model_dump(),
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload_hash = hashlib.sha256(payload_json).hexdigest()
        signature = hmac.new(self._secret, payload_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        return SignedDecisionRecord(
            audit_id=audit_id,
            signed_at=datetime.now(timezone.utc).isoformat(),
            payload_hash=payload_hash,
            signature=signature,
            policy_artifact=policy_artifact,
            provenance=provenance,
        )

    def verify(self, record: SignedDecisionRecord) -> bool:
        if not self._secret:
            return False
        expected = hmac.new(self._secret, record.payload_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, record.signature)

    def _model_config_hash(self, model_path: str) -> str:
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return "missing"
        raw = config_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()
