"""Build the static, machine-readable public results report."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolshield.guard.policy import EnforcementPolicy  # noqa: E402
SOURCE = ROOT / "data/reports/experiments/results.json"
OUTPUT = ROOT / "docs/results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: object) -> object:
    """Replace non-finite experiment values with JSON null.

    Python's encoder otherwise emits NaN, which is not valid JSON and breaks the
    browser explorer. Missing measurements are intentionally represented as null.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    source = json.loads(SOURCE.read_text())
    config_files = sorted((ROOT / "configs").rglob("*.yaml"))
    config_hash = hashlib.sha256("".join(sha256(p) for p in config_files).encode()).hexdigest()
    split_files = sorted((ROOT / "data/splits_longschema").rglob("*.jsonl"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    results = []
    for row in source["summary"]:
        item = json_safe(dict(row))
        assert isinstance(item, dict)
        item["seed_status"] = (
            "effective single seed" if "transformer" in item["model"] else "independent seeds"
        )
        results.append(item)
    policy = EnforcementPolicy()
    report = {
        "schema_version": "1.2.0",
        "source": "synthetic ToolShield benchmark",
        "source_license": "MIT",
        "dataset_provenance": "Deterministically generated from versioned templates; no user data.",
        "commit": commit,
        "configuration_sha256": config_hash,
        "source_artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "data/reports/experiments").glob("*.csv")) + [SOURCE]
        },
        "split_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in split_files},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "enforcement_policy": {
            **json_safe(policy.__dict__),
            "configuration_sha256": policy.config_hash(),
            "tool_risks": ["read", "write", "privileged"],
        },
        "limitations": [
            "Neural-model cross-seed variance is not established for the committed results.",
            "Synthetic performance does not establish production robustness."
        ],
        "results": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
