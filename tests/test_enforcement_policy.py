from fastapi.testclient import TestClient

from toolshield.demo.app import app
from toolshield.guard.policy import EnforcementPolicy, evaluate_policy


def test_privileged_tools_use_a_stricter_threshold() -> None:
    read = evaluate_policy(0.54, tool_risk="read")
    privileged = evaluate_policy(0.54, tool_risk="privileged")
    assert read.decision == "ALLOW"
    assert privileged.decision == "BLOCK"
    assert privileged.effective_threshold < read.effective_threshold


def test_policy_has_deterministic_configuration_hash() -> None:
    first = EnforcementPolicy().config_hash()
    assert first == EnforcementPolicy().config_hash()
    assert first != EnforcementPolicy(block_threshold=0.8).config_hash()


def test_missing_or_invalid_scores_fail_closed() -> None:
    assert evaluate_policy(None).decision == "BLOCK"
    assert evaluate_policy(float("nan")).decision == "BLOCK"
    assert evaluate_policy(1.1).decision == "BLOCK"


def test_review_band_requires_human_decision() -> None:
    result = evaluate_policy(0.60, tool_risk="read")
    assert result.decision == "REVIEW"
    assert result.review_threshold <= 0.60 < result.effective_threshold


def test_policy_api_returns_auditable_metadata() -> None:
    response = TestClient(app).post(
        "/policy/evaluate", json={"score": 0.55, "tool_risk": "privileged"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "BLOCK"
    assert payload["policy_version"] == "1.0.0"
    assert len(payload["policy_hash"]) == 64
