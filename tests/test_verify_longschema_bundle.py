"""Tests for scripts/verify_experiments_longschema_bundle.py.

Creates a temporary directory with minimal fixture files and asserts:
- Verifier passes on valid fixtures.
- Verifier fails if each required file is missing.
- Verifier fails if truncation stats don't match expected pattern.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "verify_experiments_longschema_bundle.py"

PROTOCOLS = ["S_random", "S_attack_holdout"]
MODELS = [
    "context_transformer_keep_prompt_longschema",
    "context_transformer_naive_longschema",
]


# ── Fixture helpers ──────────────────────────────────────────────────────────


def write_summary_csv(root: Path, *, roc_override: dict[str, float] | None = None):
    """Write a valid summary.csv with expected structure."""
    path = root / "summary.csv"
    fieldnames = ["protocol", "model", "n_seeds", "roc_auc_mean", "roc_auc_std"]
    rows = []
    for protocol in PROTOCOLS:
        for model in MODELS:
            default_roc = 0.998 if "keep_prompt" in model else 0.45
            roc = roc_override.get(model, default_roc) if roc_override else default_roc
            rows.append({
                "protocol": protocol,
                "model": model,
                "n_seeds": "3",
                "roc_auc_mean": str(roc),
                "roc_auc_std": "0.001",
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_truncation_stats_csv(
    root: Path,
    *,
    naive_pct: float = 1.0,
    naive_retention: float = 0.0,
    kp_pct: float = 0.0,
    kp_retention: float = 1.0,
):
    """Write a valid truncation_stats.csv."""
    path = root / "truncation_stats.csv"
    fieldnames = [
        "protocol", "strategy", "n_samples",
        "pct_prompt_truncated", "prompt_retention_ratio_mean",
    ]
    rows = []
    for protocol in PROTOCOLS:
        rows.append({
            "protocol": protocol,
            "strategy": "naive",
            "n_samples": "224",
            "pct_prompt_truncated": str(naive_pct),
            "prompt_retention_ratio_mean": str(naive_retention),
        })
        rows.append({
            "protocol": protocol,
            "strategy": "keep_prompt",
            "n_samples": "224",
            "pct_prompt_truncated": str(kp_pct),
            "prompt_retention_ratio_mean": str(kp_retention),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_hygiene(root: Path, *, inject_fail: bool = False):
    """Write split_hygiene.md and guards.json."""
    status = "FAIL" if inject_fail else "PASS"
    md = (
        "# Split Hygiene Report\n\n"
        "| Protocol | Check | Status | Detail |\n"
        "|----------|-------|--------|--------|\n"
        f"| S_attack_holdout | template_leakage | {status} | detail |\n"
        f"| S_random | template_leakage | PASS | detail |\n"
    )
    (root / "split_hygiene.md").write_text(md)
    guards = [{"issue": "test"}] if inject_fail else []
    (root / "guards.json").write_text(json.dumps(guards))


def write_appendix_example(root: Path, *, complete: bool = True):
    """Write appendix_truncation_example.md."""
    if complete:
        content = (
            "# Truncation Example\n\n"
            "| Metric | naive | keep_prompt |\n"
            "|--------|-------|-------------|\n"
            "| Prompt retention ratio | 0.0 | 1.0 |\n"
        )
    else:
        content = "# Truncation Example\n\nIncomplete.\n"
    (root / "appendix_truncation_example.md").write_text(content)


def write_raw_metrics(root: Path):
    """Write seed_0/S_attack_holdout/<model>/metrics.json for both models."""
    for model in MODELS:
        d = root / "seed_0" / "S_attack_holdout" / model
        d.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": "test",
            "protocol": "S_attack_holdout",
            "metrics": {"roc_auc": 0.99 if "keep_prompt" in model else 0.45},
        }
        (d / "metrics.json").write_text(json.dumps(data))


def build_full_fixture(root: Path):
    """Create a complete valid fixture set."""
    root.mkdir(parents=True, exist_ok=True)
    write_summary_csv(root)
    write_truncation_stats_csv(root)
    write_split_hygiene(root)
    write_appendix_example(root)
    write_raw_metrics(root)


def run_verifier(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the verifier script and return the result."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root", str(root),
            "--expect-seeds", "0", "1", "2",
            "--protocols", "S_random", "S_attack_holdout",
        ],
        capture_output=True,
        text=True,
    )


# ── Tests: valid fixtures ───────────────────────────────────────────────────


class TestVerifierPassesOnValidFixtures:
    """Verifier should pass on a complete, correct fixture set."""

    def test_passes_on_full_fixture(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        result = run_verifier(root)
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "[OK]" in result.stdout

    def test_all_five_ok_lines(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        result = run_verifier(root)
        ok_lines = [l for l in result.stdout.splitlines() if l.startswith("[OK]")]
        assert len(ok_lines) == 5, f"Expected 5 OK lines, got {len(ok_lines)}: {ok_lines}"


# ── Tests: missing files ────────────────────────────────────────────────────


class TestVerifierFailsOnMissingFiles:
    """Verifier should fail if any required file is missing."""

    def test_missing_summary_csv(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "summary.csv").unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "summary.csv" in result.stderr

    def test_missing_truncation_stats_csv(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "truncation_stats.csv").unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "truncation_stats.csv" in result.stderr

    def test_missing_split_hygiene_md(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "split_hygiene.md").unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "split_hygiene.md" in result.stderr

    def test_missing_guards_json(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "guards.json").unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "guards.json" in result.stderr

    def test_missing_appendix_example(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "appendix_truncation_example.md").unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "appendix_truncation_example.md" in result.stderr

    def test_missing_raw_metrics(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        # Remove one model's metrics
        p = root / "seed_0" / "S_attack_holdout" / MODELS[0] / "metrics.json"
        p.unlink()
        result = run_verifier(root)
        assert result.returncode == 1
        assert "metrics.json" in result.stderr


# ── Tests: truncation stats mismatch ────────────────────────────────────────


class TestVerifierFailsOnBadTruncationStats:
    """Verifier should fail if truncation invariants are violated."""

    def test_naive_not_fully_truncated(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        # Overwrite with bad naive stats (pct < 1.0)
        write_truncation_stats_csv(root, naive_pct=0.5)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "pct_prompt_truncated" in result.stderr

    def test_keep_prompt_has_nonzero_truncation(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_truncation_stats_csv(root, kp_pct=0.3)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "pct_prompt_truncated" in result.stderr

    def test_naive_has_nonzero_retention(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_truncation_stats_csv(root, naive_retention=0.5)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "prompt_retention_ratio_mean" in result.stderr

    def test_keep_prompt_has_low_retention(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_truncation_stats_csv(root, kp_retention=0.5)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "prompt_retention_ratio_mean" in result.stderr


# ── Tests: summary ROC-AUC regression ───────────────────────────────────────


class TestVerifierFailsOnROCAUCRegression:
    """Verifier should fail if ROC-AUC values violate expected bounds."""

    def test_keep_prompt_low_roc(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_summary_csv(root, roc_override={
            "context_transformer_keep_prompt_longschema": 0.80,
            "context_transformer_naive_longschema": 0.45,
        })
        result = run_verifier(root)
        assert result.returncode == 1
        assert "roc_auc_mean" in result.stderr

    def test_naive_high_roc(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_summary_csv(root, roc_override={
            "context_transformer_keep_prompt_longschema": 0.998,
            "context_transformer_naive_longschema": 0.75,
        })
        result = run_verifier(root)
        assert result.returncode == 1
        assert "roc_auc_mean" in result.stderr


# ── Tests: split hygiene content ────────────────────────────────────────────


class TestVerifierFailsOnSplitHygieneIssues:
    """Verifier should fail if split hygiene has FAIL entries or non-empty guards."""

    def test_fail_in_hygiene_md(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_split_hygiene(root, inject_fail=True)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "FAIL" in result.stderr or "guards.json" in result.stderr

    def test_nonempty_guards_json(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        (root / "guards.json").write_text(json.dumps([{"issue": "leaked"}]))
        result = run_verifier(root)
        assert result.returncode == 1
        assert "guards.json" in result.stderr


# ── Tests: appendix evidence strings ────────────────────────────────────────


class TestVerifierFailsOnIncompleteAppendix:
    """Verifier should fail if appendix is missing evidence strings."""

    def test_incomplete_appendix(self, tmp_path: Path):
        root = tmp_path / "experiments"
        build_full_fixture(root)
        write_appendix_example(root, complete=False)
        result = run_verifier(root)
        assert result.returncode == 1
        assert "appendix" in result.stderr.lower() or "evidence" in result.stderr.lower()
