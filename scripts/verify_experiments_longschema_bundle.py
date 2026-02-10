#!/usr/bin/env python3
"""Verify the long-schema experiment bundle is complete and consistent.

Programmatically checks every thesis claim against the generated artifacts.
Exits with code 1 on any failure, printing clear diagnostics.

Usage:
    python scripts/verify_experiments_longschema_bundle.py
    python scripts/verify_experiments_longschema_bundle.py --root data/reports/experiments_longschema
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── Helpers ──────────────────────────────────────────────────────────────────

MODELS = [
    "context_transformer_keep_prompt_longschema",
    "context_transformer_naive_longschema",
]

KEEP_PROMPT_MODEL = "context_transformer_keep_prompt_longschema"
NAIVE_MODEL = "context_transformer_naive_longschema"


def fail(msg: str) -> None:
    """Print a FAIL message to stderr."""
    print(f"[FAIL] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    """Print an OK message."""
    print(f"[OK] {msg}")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV and return list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Check A: summary.csv ────────────────────────────────────────────────────

def check_summary(root: Path, protocols: list[str], n_seeds: int) -> bool:
    """Verify summary.csv exists and contains expected models/metrics."""
    path = root / "summary.csv"
    if not path.exists():
        fail(f"summary.csv not found at {path}")
        return False

    rows = read_csv(path)
    errors: list[str] = []

    for protocol in protocols:
        for model in MODELS:
            matches = [r for r in rows if r["protocol"] == protocol and r["model"] == model]
            if not matches:
                errors.append(f"Missing row: protocol={protocol}, model={model}")
                continue

            row = matches[0]

            # Check n_seeds
            actual_seeds = int(row["n_seeds"])
            if actual_seeds != n_seeds:
                errors.append(
                    f"{protocol}/{model}: n_seeds={actual_seeds}, expected {n_seeds}"
                )

            # Check ROC-AUC regression bounds
            roc_auc = float(row["roc_auc_mean"])
            if model == KEEP_PROMPT_MODEL:
                if roc_auc < 0.95:
                    errors.append(
                        f"{protocol}/{model}: roc_auc_mean={roc_auc:.4f} < 0.95 "
                        f"(keep_prompt should be high)"
                    )
            elif model == NAIVE_MODEL:
                if roc_auc > 0.60:
                    errors.append(
                        f"{protocol}/{model}: roc_auc_mean={roc_auc:.4f} > 0.60 "
                        f"(naive should be low under long schemas)"
                    )

    if errors:
        for e in errors:
            fail(f"summary.csv: {e}")
        return False

    ok("summary.csv — all models present, n_seeds correct, ROC-AUC bounds satisfied")
    return True


# ── Check B: truncation_stats.csv ───────────────────────────────────────────

def check_truncation_stats(root: Path, protocols: list[str]) -> bool:
    """Verify truncation_stats.csv has expected truncation patterns."""
    path = root / "truncation_stats.csv"
    if not path.exists():
        fail(f"truncation_stats.csv not found at {path}")
        return False

    rows = read_csv(path)
    errors: list[str] = []
    tol = 1e-6

    for protocol in protocols:
        for strategy, expected_pct, expected_retention in [
            ("naive", 1.0, 0.0),
            ("keep_prompt", 0.0, 1.0),
        ]:
            matches = [
                r for r in rows
                if r["protocol"] == protocol and r["strategy"] == strategy
            ]
            if not matches:
                errors.append(f"Missing row: protocol={protocol}, strategy={strategy}")
                continue

            row = matches[0]
            pct = float(row["pct_prompt_truncated"])
            retention = float(row["prompt_retention_ratio_mean"])

            if abs(pct - expected_pct) > tol:
                errors.append(
                    f"{protocol}/{strategy}: pct_prompt_truncated={pct}, "
                    f"expected {expected_pct}"
                )
            if abs(retention - expected_retention) > tol:
                errors.append(
                    f"{protocol}/{strategy}: prompt_retention_ratio_mean={retention}, "
                    f"expected {expected_retention}"
                )

    if errors:
        for e in errors:
            fail(f"truncation_stats.csv: {e}")
        return False

    ok("truncation_stats.csv — truncation patterns match expected values")
    return True


# ── Check C: split_hygiene.md + guards.json ─────────────────────────────────

def check_split_hygiene(root: Path) -> bool:
    """Verify split hygiene report shows all PASS and guards.json is empty."""
    md_path = root / "split_hygiene.md"
    guards_path = root / "guards.json"
    errors: list[str] = []

    if not md_path.exists():
        fail(f"split_hygiene.md not found at {md_path}")
        return False

    content = md_path.read_text()

    # Check that no FAIL status exists in the markdown table
    if "| FAIL |" in content:
        errors.append("split_hygiene.md contains FAIL entries")

    # Check that PASS entries exist
    if "| PASS |" not in content:
        errors.append("split_hygiene.md contains no PASS entries")

    if not guards_path.exists():
        errors.append(f"guards.json not found at {guards_path}")
    else:
        guards = json.loads(guards_path.read_text())
        if not isinstance(guards, list):
            errors.append("guards.json is not a list")
        elif len(guards) != 0:
            errors.append(f"guards.json has {len(guards)} warning(s), expected 0")

    if errors:
        for e in errors:
            fail(f"split hygiene: {e}")
        return False

    ok("split hygiene — all checks PASS, guards.json empty")
    return True


# ── Check D: appendix_truncation_example.md ─────────────────────────────────

def check_appendix_example(root: Path) -> bool:
    """Verify appendix truncation example exists and contains evidence strings."""
    path = root / "appendix_truncation_example.md"
    if not path.exists():
        fail(f"appendix_truncation_example.md not found at {path}")
        return False

    content = path.read_text()
    errors: list[str] = []

    # Evidence strings that prove both strategies are compared
    evidence = [
        ("Prompt retention ratio", "retention comparison table"),
        ("naive", "naive strategy reference"),
        ("keep_prompt", "keep_prompt strategy reference"),
    ]
    for marker, desc in evidence:
        if marker not in content:
            errors.append(f"Missing evidence string '{marker}' ({desc})")

    if errors:
        for e in errors:
            fail(f"appendix example: {e}")
        return False

    ok("appendix example — contains both strategy comparisons")
    return True


# ── Check E: raw metrics.json for seed_0 ────────────────────────────────────

def check_raw_metrics(root: Path) -> bool:
    """Verify raw metrics.json exists for seed_0/S_attack_holdout for both models."""
    errors: list[str] = []

    for model in MODELS:
        path = root / "seed_0" / "S_attack_holdout" / model / "metrics.json"
        if not path.exists():
            errors.append(f"metrics.json not found: {path}")
            continue

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Invalid JSON in {path}: {exc}")
            continue

        if "metrics" not in data:
            errors.append(f"No 'metrics' key in {path}")
        elif "roc_auc" not in data["metrics"]:
            errors.append(f"No 'roc_auc' in metrics of {path}")

    if errors:
        for e in errors:
            fail(f"raw metrics: {e}")
        return False

    ok("raw metrics present — seed_0/S_attack_holdout has both model metrics")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify long-schema experiment bundle completeness and consistency"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "experiments_longschema",
        help="Root directory of experiment artifacts",
    )
    parser.add_argument(
        "--expect-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Expected seed indices",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["S_random", "S_attack_holdout"],
        help="Expected protocols",
    )
    args = parser.parse_args()

    root: Path = args.root
    protocols: list[str] = args.protocols
    n_seeds: int = len(args.expect_seeds)

    print(f"Verifying long-schema bundle: {root}")
    print(f"  Protocols: {protocols}")
    print(f"  Expected seeds: {args.expect_seeds} (n={n_seeds})")
    print()

    results = [
        check_summary(root, protocols, n_seeds),
        check_truncation_stats(root, protocols),
        check_split_hygiene(root),
        check_appendix_example(root),
        check_raw_metrics(root),
    ]

    print()
    n_pass = sum(results)
    n_total = len(results)

    if all(results):
        print(f"All {n_total} verification checks passed.")
        return 0
    else:
        n_fail = n_total - n_pass
        print(f"FAILED: {n_fail}/{n_total} check(s) failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
