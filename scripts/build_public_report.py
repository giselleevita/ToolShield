"""Build the static, machine-readable public results report."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/reports/experiments/results.json"
OUTPUT = ROOT / "docs/results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        item = dict(row)
        item["seed_status"] = (
            "effective single seed" if "transformer" in item["model"] else "independent seeds"
        )
        results.append(item)
    report = {
        "schema_version": "1.0.0",
        "source": "synthetic ToolShield benchmark",
        "source_license": "MIT",
        "dataset_provenance": "Deterministically generated from versioned templates; no user data.",
        "commit": commit,
        "configuration_sha256": config_hash,
        "split_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in split_files},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "limitations": [
            "Neural-model cross-seed variance is not established for the committed results.",
            "Synthetic performance does not establish production robustness."
        ],
        "results": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
