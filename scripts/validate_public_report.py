"""Validate the committed browser report without loading any ML dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if report.get("schema_version") != "1.2.0":
        raise ValueError("unsupported public-report schema")
    if not report.get("results"):
        raise ValueError("public report has no results")
    pairs = [(row.get("protocol"), row.get("model")) for row in report["results"]]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate protocol/model result")
    protocols = {protocol for protocol, _ in pairs}
    expected = {"S_random", "S_attack_holdout", "S_tool_holdout"}
    if protocols != expected:
        raise ValueError(f"expected protocols {sorted(expected)}, got {sorted(protocols)}")
    checksums = {**report.get("split_sha256", {}), **report.get("source_artifacts_sha256", {})}
    if not checksums or any(len(value) != 64 for value in checksums.values()):
        raise ValueError("missing or malformed artifact checksum")
    policy = report.get("enforcement_policy", {})
    if policy.get("tool_risks") != ["read", "write", "privileged"]:
        raise ValueError("missing enforcement risk levels")
    if len(policy.get("configuration_sha256", "")) != 64:
        raise ValueError("missing enforcement policy checksum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("docs/results.json"))
    args = parser.parse_args()
    validate(args.path)
    print(f"Validated {args.path}")


if __name__ == "__main__":
    main()
