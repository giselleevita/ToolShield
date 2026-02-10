#!/usr/bin/env python3
"""Verify dataset determinism, split hygiene, and balance constraints.

This script performs comprehensive verification of the generated dataset
and all split protocols:

1. Determinism: Regenerating the dataset with seed=1337 produces identical records.
2. Split hygiene: template_id sets are disjoint across train/val/test for all protocols.
3. Protocol constraints:
   - S_attack_holdout: AF4 must not appear in train/val
   - S_tool_holdout: exportReport must not appear in train/val
4. Balance sanity: ~50% benign, ~50% attack; AF1-4 roughly balanced.

Usage:
    python scripts/verify_dataset.py

Exit codes:
    0: All checks pass
    1: One or more checks failed
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolshield.data.generate_dataset import generate_dataset
from toolshield.data.make_splits import create_splits, SplitProtocol
from toolshield.data.schema import DatasetRecord
from toolshield.utils.io import load_jsonl


console = Console()


def compute_dataset_hash(records: list[DatasetRecord]) -> str:
    """Compute a deterministic hash of dataset records.
    
    Args:
        records: List of DatasetRecord objects.
        
    Returns:
        SHA256 hash of sorted record IDs.
    """
    # Sort by ID for determinism
    sorted_ids = sorted(r.id for r in records)
    combined = "|".join(sorted_ids)
    return hashlib.sha256(combined.encode()).hexdigest()


def check_determinism(data_dir: Path) -> tuple[bool, str]:
    """Check that dataset generation is deterministic.
    
    Args:
        data_dir: Path to data directory containing dataset.jsonl.
        
    Returns:
        Tuple of (passed, message).
    """
    dataset_path = data_dir / "dataset.jsonl"
    
    if not dataset_path.exists():
        return False, f"Dataset file not found: {dataset_path}"
    
    # Load existing dataset
    existing_records_raw = load_jsonl(dataset_path)
    existing_records = [DatasetRecord(**r) for r in existing_records_raw]
    existing_hash = compute_dataset_hash(existing_records)
    n_existing = len(existing_records)
    
    # Regenerate with same seed
    regenerated_records, _ = generate_dataset(seed=1337, n_samples=n_existing)
    regenerated_hash = compute_dataset_hash(regenerated_records)
    
    if existing_hash == regenerated_hash:
        return True, f"Determinism verified (hash: {existing_hash[:16]}...)"
    else:
        return False, (
            f"Determinism FAILED!\n"
            f"  Existing hash:     {existing_hash[:32]}...\n"
            f"  Regenerated hash:  {regenerated_hash[:32]}..."
        )


def check_template_disjointness(
    splits: dict[str, list[DatasetRecord]], protocol: str
) -> tuple[bool, str]:
    """Check that template_ids are disjoint across splits.

    For S_tool_holdout, only train/val disjointness is required because
    the same attack template can generate records for both the held-out
    tool (→ test) and other tools (→ train/val).
    
    Args:
        splits: Dictionary mapping split names to records.
        protocol: Protocol name for reporting.
        
    Returns:
        Tuple of (passed, message).
    """
    train_templates = {r.template_id for r in splits["train"]}
    val_templates = {r.template_id for r in splits["val"]}
    test_templates = {r.template_id for r in splits["test"]}
    
    issues = []
    
    # Train / val disjointness is always required
    train_val_overlap = train_templates & val_templates
    if train_val_overlap:
        issues.append(f"train/val overlap: {len(train_val_overlap)} templates")
    
    # For S_tool_holdout, train-test and val-test template overlap is expected
    if protocol != SplitProtocol.S_TOOL_HOLDOUT.value:
        train_test_overlap = train_templates & test_templates
        val_test_overlap = val_templates & test_templates
        if train_test_overlap:
            issues.append(f"train/test overlap: {len(train_test_overlap)} templates")
        if val_test_overlap:
            issues.append(f"val/test overlap: {len(val_test_overlap)} templates")
    
    if issues:
        return False, f"{protocol}: Template leakage! {', '.join(issues)}"
    
    msg = f"{protocol}: No template leakage (train={len(train_templates)}, val={len(val_templates)}, test={len(test_templates)})"
    if protocol == SplitProtocol.S_TOOL_HOLDOUT.value:
        msg += " [train/val checked; train-test overlap expected]"
    return True, msg


def check_attack_holdout_constraints(
    splits: dict[str, list[DatasetRecord]]
) -> tuple[bool, str]:
    """Check S_attack_holdout constraints: AF4 must not be in train/val.
    
    Args:
        splits: Dictionary mapping split names to records.
        
    Returns:
        Tuple of (passed, message).
    """
    train_families = {r.attack_family for r in splits["train"] if r.attack_family}
    val_families = {r.attack_family for r in splits["val"] if r.attack_family}
    test_families = {r.attack_family for r in splits["test"] if r.attack_family}
    
    issues = []
    if "AF4" in train_families:
        af4_count = sum(1 for r in splits["train"] if r.attack_family == "AF4")
        issues.append(f"AF4 found in train ({af4_count} records)")
    if "AF4" in val_families:
        af4_count = sum(1 for r in splits["val"] if r.attack_family == "AF4")
        issues.append(f"AF4 found in val ({af4_count} records)")
    
    if issues:
        return False, f"S_attack_holdout: Constraint violation! {', '.join(issues)}"
    
    # Verify AF4 is in test
    if "AF4" not in test_families:
        return False, "S_attack_holdout: AF4 not found in test (should be there)"
    
    af4_test_count = sum(1 for r in splits["test"] if r.attack_family == "AF4")
    return True, f"S_attack_holdout: AF4 correctly held out (test has {af4_test_count} AF4 samples)"


def check_tool_holdout_constraints(
    splits: dict[str, list[DatasetRecord]]
) -> tuple[bool, str]:
    """Check S_tool_holdout constraints: exportReport must not be in train/val.
    
    Args:
        splits: Dictionary mapping split names to records.
        
    Returns:
        Tuple of (passed, message).
    """
    train_tools = {r.tool_name for r in splits["train"]}
    val_tools = {r.tool_name for r in splits["val"]}
    test_tools = {r.tool_name for r in splits["test"]}
    
    issues = []
    if "exportReport" in train_tools:
        count = sum(1 for r in splits["train"] if r.tool_name == "exportReport")
        issues.append(f"exportReport found in train ({count} records)")
    if "exportReport" in val_tools:
        count = sum(1 for r in splits["val"] if r.tool_name == "exportReport")
        issues.append(f"exportReport found in val ({count} records)")
    
    if issues:
        return False, f"S_tool_holdout: Constraint violation! {', '.join(issues)}"
    
    # Verify exportReport is in test
    if "exportReport" not in test_tools:
        return False, "S_tool_holdout: exportReport not found in test (should be there)"
    
    test_count = sum(1 for r in splits["test"] if r.tool_name == "exportReport")
    return True, f"S_tool_holdout: exportReport correctly held out (test has {test_count} samples)"


def check_balance(records: list[DatasetRecord]) -> tuple[bool, str]:
    """Check dataset balance: ~50% benign, ~50% attack, AF1-4 balanced.
    
    Args:
        records: All dataset records.
        
    Returns:
        Tuple of (passed, message).
    """
    label_counts = Counter(r.label_binary for r in records)
    n_total = len(records)
    
    n_benign = label_counts.get(0, 0)
    n_attack = label_counts.get(1, 0)
    
    benign_ratio = n_benign / n_total if n_total > 0 else 0
    attack_ratio = n_attack / n_total if n_total > 0 else 0
    
    issues = []
    
    # Check benign/attack balance (allow 45-55% range)
    if not (0.45 <= benign_ratio <= 0.55):
        issues.append(f"benign ratio {benign_ratio:.1%} out of range (45-55%)")
    if not (0.45 <= attack_ratio <= 0.55):
        issues.append(f"attack ratio {attack_ratio:.1%} out of range (45-55%)")
    
    # Check AF1-4 balance within attacks
    attack_records = [r for r in records if r.label_binary == 1]
    family_counts = Counter(r.attack_family for r in attack_records)
    
    # Each family should be roughly 25% of attacks (allow 15-35% range)
    for family in ["AF1", "AF2", "AF3", "AF4"]:
        family_count = family_counts.get(family, 0)
        family_ratio = family_count / len(attack_records) if attack_records else 0
        if not (0.15 <= family_ratio <= 0.35):
            issues.append(f"{family} ratio {family_ratio:.1%} out of range (15-35%)")
    
    if issues:
        return False, f"Balance issues: {', '.join(issues)}"
    
    return True, (
        f"Balance OK: {benign_ratio:.1%} benign, {attack_ratio:.1%} attack; "
        f"AF1-4 each ~{100/4:.0f}%"
    )


def load_or_create_splits(
    data_dir: Path, protocol: str, records: list[DatasetRecord]
) -> dict[str, list[DatasetRecord]]:
    """Load existing splits or create them if not present.
    
    Args:
        data_dir: Data directory.
        protocol: Split protocol name.
        records: All dataset records.
        
    Returns:
        Dictionary mapping split names to records.
    """
    split_dir = data_dir / "splits" / protocol
    
    if split_dir.exists():
        # Load existing splits
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_path = split_dir / f"{split_name}.jsonl"
            if split_path.exists():
                raw = load_jsonl(split_path)
                splits[split_name] = [DatasetRecord(**r) for r in raw]
        
        if len(splits) == 3:
            return splits
    
    # Create splits in memory for verification
    return create_splits(records, protocol, seed=2026)


def main() -> int:
    """Run all verification checks.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    console.print("\n[bold blue]ToolShield Dataset Verification[/bold blue]\n")
    
    data_dir = Path("data")
    
    results: list[tuple[str, bool, str]] = []
    
    # 1. Check determinism
    console.print("[yellow]Checking determinism...[/yellow]")
    passed, msg = check_determinism(data_dir)
    results.append(("Determinism", passed, msg))
    
    # Load dataset for further checks
    dataset_path = data_dir / "dataset.jsonl"
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
        console.print("[dim]Run 'make data' first to generate the dataset.[/dim]")
        return 1
    
    records_raw = load_jsonl(dataset_path)
    records = [DatasetRecord(**r) for r in records_raw]
    
    # 2. Check balance
    console.print("[yellow]Checking dataset balance...[/yellow]")
    passed, msg = check_balance(records)
    results.append(("Balance", passed, msg))
    
    # 3. Check splits for each protocol
    for protocol in [SplitProtocol.S_RANDOM, SplitProtocol.S_ATTACK_HOLDOUT, SplitProtocol.S_TOOL_HOLDOUT]:
        protocol_name = protocol.value
        console.print(f"[yellow]Checking {protocol_name}...[/yellow]")
        
        splits = load_or_create_splits(data_dir, protocol_name, records)
        
        # Check template disjointness
        passed, msg = check_template_disjointness(splits, protocol_name)
        results.append((f"{protocol_name} leakage", passed, msg))
        
        # Check protocol-specific constraints
        if protocol == SplitProtocol.S_ATTACK_HOLDOUT:
            passed, msg = check_attack_holdout_constraints(splits)
            results.append((f"{protocol_name} constraints", passed, msg))
        elif protocol == SplitProtocol.S_TOOL_HOLDOUT:
            passed, msg = check_tool_holdout_constraints(splits)
            results.append((f"{protocol_name} constraints", passed, msg))
        
        # Check both classes in every split
        both_ok = True
        for split_name in ["train", "val", "test"]:
            labels = {r.label_binary for r in splits[split_name]}
            if len(labels) < 2:
                both_ok = False
                results.append((
                    f"{protocol_name} {split_name} classes",
                    False,
                    f"{split_name} is single-class (labels={labels})",
                ))
        if both_ok:
            results.append((f"{protocol_name} both-classes", True, "All splits have both classes"))
    
    # Print results table
    console.print("\n")
    table = Table(title="Verification Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    all_passed = True
    for check_name, passed, msg in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        if not passed:
            all_passed = False
        table.add_row(check_name, status, msg)
    
    console.print(table)
    
    # Print summary
    console.print()
    if all_passed:
        console.print("[bold green]✓ All checks passed![/bold green]")
        return 0
    else:
        n_failed = sum(1 for _, passed, _ in results if not passed)
        console.print(f"[bold red]✗ {n_failed} check(s) failed![/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
