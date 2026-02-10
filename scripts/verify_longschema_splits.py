#!/usr/bin/env python3
"""Verify split hygiene and class balance for the longschema experiment.

Checks:
1. Template leakage: no overlap of template_id between train and test.
2. AF4 holdout: for S_attack_holdout, AF4 absent in train/val, present in test.
3. Tool holdout: auto-detected for protocols containing "tool".
4. Class balance: each train/val has both label_binary classes.

Outputs:
    {output_dir}/split_hygiene.csv
    {output_dir}/split_hygiene.md
    {output_dir}/guards.json

Usage:
    python scripts/verify_longschema_splits.py
    python scripts/verify_longschema_splits.py --splits-dir data/splits_longschema
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolshield.data.schema import DatasetRecord
from toolshield.utils.io import load_jsonl

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_split(split_dir: Path, name: str) -> list[DatasetRecord]:
    path = split_dir / f"{name}.jsonl"
    if not path.exists():
        return []
    raw = load_jsonl(path)
    return [DatasetRecord(**r) for r in raw]


def load_protocol(splits_dir: Path, protocol: str) -> dict[str, list[DatasetRecord]]:
    sd = splits_dir / protocol
    return {s: load_split(sd, s) for s in ["train", "val", "test"]}


# ── Check functions ──────────────────────────────────────────────────────────

def check_template_leakage(
    splits: dict[str, list[DatasetRecord]], protocol: str, is_tool_holdout: bool
) -> tuple[bool, str]:
    """Check template_id disjointness across splits."""
    train_t = {r.template_id for r in splits["train"]}
    val_t = {r.template_id for r in splits["val"]}
    test_t = {r.template_id for r in splits["test"]}

    issues = []
    tv = train_t & val_t
    if tv:
        issues.append(f"train/val overlap: {len(tv)}")

    # For tool-holdout, train-test overlap may be expected
    if not is_tool_holdout:
        tt = train_t & test_t
        vt = val_t & test_t
        if tt:
            issues.append(f"train/test overlap: {len(tt)}")
        if vt:
            issues.append(f"val/test overlap: {len(vt)}")

    if issues:
        return False, f"Template leakage: {'; '.join(issues)}"
    return True, f"No template leakage (train={len(train_t)}, val={len(val_t)}, test={len(test_t)})"


def check_af4_holdout(splits: dict[str, list[DatasetRecord]]) -> tuple[bool, str]:
    """Check AF4 absent in train/val, present in test."""
    issues = []
    for s in ["train", "val"]:
        af4 = [r for r in splits[s] if r.attack_family == "AF4"]
        if af4:
            issues.append(f"AF4 in {s}: {len(af4)} records")
    af4_test = [r for r in splits["test"] if r.attack_family == "AF4"]
    if not af4_test:
        issues.append("AF4 not in test")
    if issues:
        return False, f"AF4 holdout violation: {'; '.join(issues)}"
    return True, f"AF4 correctly held out (test has {len(af4_test)} AF4 samples)"


def check_tool_holdout(
    splits: dict[str, list[DatasetRecord]], tool: str = "exportReport"
) -> tuple[bool, str]:
    """Check held-out tool absent in train/val, present in test."""
    issues = []
    for s in ["train", "val"]:
        found = [r for r in splits[s] if r.tool_name == tool]
        if found:
            issues.append(f"{tool} in {s}: {len(found)} records")
    test_found = [r for r in splits["test"] if r.tool_name == tool]
    if not test_found:
        issues.append(f"{tool} not in test")
    if issues:
        return False, f"Tool holdout violation: {'; '.join(issues)}"
    return True, f"{tool} correctly held out (test has {len(test_found)} samples)"


def check_class_balance(
    splits: dict[str, list[DatasetRecord]], protocol: str
) -> list[tuple[bool, str, dict | None]]:
    """Check that train and val have both classes. Returns list of (passed, detail, guard_entry)."""
    results = []
    for s in ["train", "val"]:
        labels = Counter(r.label_binary for r in splits[s])
        has_both = 0 in labels and 1 in labels
        detail = f"{s}: benign={labels.get(0,0)}, attack={labels.get(1,0)}"
        guard = None
        if not has_both:
            guard = {
                "protocol": protocol,
                "split": s,
                "check": "class_balance",
                "counts": dict(labels),
                "reason": "single-class training set",
            }
        results.append((has_both, detail, guard))
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify longschema split hygiene")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits_longschema")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "experiments_longschema")
    args = parser.parse_args()

    splits_dir = args.splits_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not splits_dir.exists():
        console.print(f"[red]Splits dir not found: {splits_dir}[/red]")
        return 1

    # Discover protocols
    protocols = sorted(p.name for p in splits_dir.iterdir() if p.is_dir())
    if not protocols:
        console.print(f"[red]No protocols found in {splits_dir}[/red]")
        return 1

    console.print(f"[bold blue]Long-Schema Split Verification[/bold blue]")
    console.print(f"Splits dir: {splits_dir}")
    console.print(f"Protocols:  {protocols}\n")

    rows: list[dict] = []          # for CSV
    guards: list[dict] = []        # for guards.json
    any_hard_fail = False

    for protocol in protocols:
        splits = load_protocol(splits_dir, protocol)
        if not all(splits.values()):
            console.print(f"[yellow]  {protocol}: incomplete splits, skipping[/yellow]")
            continue

        is_tool = "tool" in protocol.lower()

        # 1. Template leakage
        ok, detail = check_template_leakage(splits, protocol, is_tool)
        rows.append({"protocol": protocol, "check": "template_leakage", "passed": ok, "detail": detail})
        if not ok:
            any_hard_fail = True

        # 2. AF4 holdout (if applicable)
        if "attack_holdout" in protocol.lower():
            ok, detail = check_af4_holdout(splits)
            rows.append({"protocol": protocol, "check": "af4_holdout", "passed": ok, "detail": detail})
            if not ok:
                any_hard_fail = True

        # 3. Tool holdout (if applicable)
        if is_tool:
            ok, detail = check_tool_holdout(splits)
            rows.append({"protocol": protocol, "check": "tool_holdout", "passed": ok, "detail": detail})
            if not ok:
                any_hard_fail = True

        # 4. Class balance
        balance_results = check_class_balance(splits, protocol)
        all_balanced = True
        for ok, detail, guard in balance_results:
            if guard:
                guards.append(guard)
                all_balanced = False
            rows.append({"protocol": protocol, "check": "class_balance", "passed": ok, "detail": detail})
        if not all_balanced:
            # Class imbalance is a soft warning, not a hard failure
            console.print(f"  [yellow]  {protocol}: class balance warning — see guards.json[/yellow]")

    # ── Print table ──────────────────────────────────────────────────────
    table = Table(title="Split Hygiene Report")
    table.add_column("Protocol", style="cyan")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail")

    for r in rows:
        status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"
        table.add_row(r["protocol"], r["check"], status, r["detail"])

    console.print(table)

    # ── Save CSV ─────────────────────────────────────────────────────────
    import csv
    csv_path = output_dir / "split_hygiene.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nCSV saved: {csv_path}")

    # ── Save markdown ────────────────────────────────────────────────────
    md_path = output_dir / "split_hygiene.md"
    md_lines = ["# Split Hygiene Report\n"]
    md_lines.append("| Protocol | Check | Status | Detail |")
    md_lines.append("|----------|-------|--------|--------|")
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        md_lines.append(f"| {r['protocol']} | {r['check']} | {status} | {r['detail']} |")
    md_lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    console.print(f"Markdown saved: {md_path}")

    # ── Save guards.json ─────────────────────────────────────────────────
    guards_path = output_dir / "guards.json"
    with open(guards_path, "w") as f:
        json.dump(guards, f, indent=2)
    console.print(f"Guards saved: {guards_path} ({len(guards)} issue{'s' if len(guards) != 1 else ''})")

    # ── Summary ──────────────────────────────────────────────────────────
    n_pass = sum(1 for r in rows if r["passed"])
    n_fail = sum(1 for r in rows if not r["passed"])

    if any_hard_fail:
        console.print(f"\n[bold red]FAIL: {n_fail} check(s) failed, {n_pass} passed[/bold red]")
        return 1
    else:
        console.print(f"\n[bold green]ALL {n_pass} checks passed[/bold green]")
        if guards:
            console.print(f"[yellow]  ({len(guards)} soft warning(s) in guards.json)[/yellow]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
