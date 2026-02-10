#!/usr/bin/env python3
"""Aggregate experiment results from evaluation outputs.

This script discovers all evaluation outputs and rebuilds:
- data/reports/experiments/raw_results.csv
- data/reports/experiments/results.json
- data/reports/experiments/summary.csv

Source of truth: per-run evaluation JSON files under data/reports/

Usage:
    python scripts/aggregate_experiments.py
    python scripts/aggregate_experiments.py --expect-seeds 0 1 2
    python scripts/aggregate_experiments.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Expected protocols and models for completeness checks
EXPECTED_PROTOCOLS = ["S_random", "S_attack_holdout", "S_tool_holdout"]
EXPECTED_MODELS = ["heuristic", "heuristic_score", "tfidf_lr", "transformer", "context_transformer"]
BUDGET_LEVELS = [1, 3, 5]  # percentage


@dataclass
class EvalResult:
    """Parsed evaluation result."""
    seed: int
    protocol: str
    model: str
    source_path: str
    
    # Core metrics
    roc_auc: float
    pr_auc: float
    fpr_at_tpr_90: float
    fpr_at_tpr_95: float
    asr_reduction_90: float
    asr_reduction_95: float
    blocked_benign_rate_90: float
    blocked_benign_rate_95: float
    
    # Latency
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    
    # Budget results: {budget_pct: {threshold, fpr, tpr, asr, blocked_benign}}
    budget_results: dict[int, dict[str, float]] = field(default_factory=dict)


def _scan_seed_dirs(root: Path, results: list[tuple[Path, int, str, str]]) -> None:
    """Scan a directory containing seed_* subdirs for evaluation outputs."""
    for seed_dir in root.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        for protocol_dir in seed_dir.iterdir():
            if not protocol_dir.is_dir():
                continue
            protocol = protocol_dir.name

            for model_dir in protocol_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name

                # Look for metrics.json first, then config.json
                metrics_file = model_dir / "metrics.json"
                if metrics_file.exists():
                    results.append((metrics_file, seed, protocol, model))
                else:
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        results.append((config_file, seed, protocol, model))


def discover_eval_outputs(reports_dir: Path) -> list[tuple[Path, int, str, str]]:
    """Discover all evaluation JSON files.
    
    Returns list of (path, seed, protocol, model) tuples.
    
    Sources:
    1. {reports_dir}/{protocol}/{model}_metrics.json (seed=0, from make eval)
    2. {reports_dir}/experiments/seed_{n}/{protocol}/{model}/metrics.json
    3. {reports_dir}/seed_{n}/{protocol}/{model}/metrics.json (direct seed dirs)
    """
    results = []
    
    # Source 1: Direct protocol directories (from make eval)
    for protocol in EXPECTED_PROTOCOLS:
        protocol_dir = reports_dir / protocol
        if not protocol_dir.exists():
            continue
        
        for metrics_file in protocol_dir.glob("*_metrics.json"):
            # Extract model name from filename
            model = metrics_file.stem.replace("_metrics", "")
            results.append((metrics_file, 0, protocol, model))
    
    # Source 2: Experiments subdirectory
    experiments_dir = reports_dir / "experiments"
    if experiments_dir.exists():
        _scan_seed_dirs(experiments_dir, results)

    # Source 3: Direct seed_* dirs in reports_dir (for custom experiment roots)
    if any(reports_dir.glob("seed_*")):
        _scan_seed_dirs(reports_dir, results)
    
    return results


def parse_eval_json(
    path: Path,
    seed: int,
    protocol: str,
    model: str,
) -> EvalResult | None:
    """Parse an evaluation JSON file into EvalResult."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        console.print(f"[yellow]Warning: Could not parse {path}: {e}[/yellow]")
        return None
    
    # Handle different JSON formats
    metrics = data.get("metrics", data)
    
    # Extract core metrics with fallbacks
    roc_auc = metrics.get("roc_auc", 0.0)
    pr_auc = metrics.get("pr_auc", 0.0)
    fpr_at_tpr_90 = metrics.get("fpr_at_tpr_90", 1.0)
    fpr_at_tpr_95 = metrics.get("fpr_at_tpr_95", 1.0)
    asr_reduction_90 = metrics.get("asr_reduction_90", 0.0)
    asr_reduction_95 = metrics.get("asr_reduction_95", 0.0)
    blocked_benign_rate_90 = metrics.get("blocked_benign_rate_90", fpr_at_tpr_90)
    blocked_benign_rate_95 = metrics.get("blocked_benign_rate_95", fpr_at_tpr_95)
    
    # Skip if no meaningful metrics (likely a config file without results)
    if roc_auc == 0.0 and pr_auc == 0.0:
        return None
    
    # Latency
    latency = data.get("latency", metrics)
    latency_p50 = latency.get("latency_p50_ms")
    latency_p95 = latency.get("latency_p95_ms")
    
    # Budget results
    budget_results = {}
    for budget_data in data.get("budget_results", []):
        budget_pct = int(budget_data.get("budget", 0) * 100)
        if budget_pct in BUDGET_LEVELS:
            budget_results[budget_pct] = {
                "threshold": budget_data.get("threshold", 0.0),
                "fpr": budget_data.get("actual_fpr", budget_data.get("fpr", 0.0)),
                "tpr": budget_data.get("tpr", 0.0),
                "asr": budget_data.get("asr_after", 0.0),
                "blocked_benign": budget_data.get("blocked_benign_rate", 0.0),
            }
    
    return EvalResult(
        seed=seed,
        protocol=protocol,
        model=model,
        source_path=str(path),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr_at_tpr_90=fpr_at_tpr_90,
        fpr_at_tpr_95=fpr_at_tpr_95,
        asr_reduction_90=asr_reduction_90,
        asr_reduction_95=asr_reduction_95,
        blocked_benign_rate_90=blocked_benign_rate_90,
        blocked_benign_rate_95=blocked_benign_rate_95,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        budget_results=budget_results,
    )


def deduplicate_results(results: list[EvalResult]) -> list[EvalResult]:
    """Deduplicate results, preferring more complete entries."""
    # Group by (seed, protocol, model)
    groups: dict[tuple[int, str, str], list[EvalResult]] = {}
    for r in results:
        key = (r.seed, r.protocol, r.model)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    # Pick best from each group (most budget results, then highest ROC-AUC)
    deduped = []
    for key, group in groups.items():
        best = max(group, key=lambda r: (len(r.budget_results), r.roc_auc))
        deduped.append(best)
    
    return deduped


def validate_completeness(
    results: list[EvalResult],
    expected_seeds: list[int] | None,
    strict: bool,
) -> list[str]:
    """Validate completeness and return warnings."""
    warnings = []
    
    # Group results
    seeds = sorted(set(r.seed for r in results))
    protocols = sorted(set(r.protocol for r in results))
    models = sorted(set(r.model for r in results))
    
    # Check only one protocol
    if len(protocols) == 1:
        warnings.append(f"Only one protocol present: {protocols[0]}")
    
    # Check expected seeds
    if expected_seeds is not None:
        missing_seeds = set(expected_seeds) - set(seeds)
        if missing_seeds:
            warnings.append(f"Missing expected seeds: {sorted(missing_seeds)}")
    
    # Check transformer models
    transformer_models = {"transformer", "context_transformer"}
    present_transformers = transformer_models.intersection(models)
    missing_transformers = transformer_models - set(models)
    if missing_transformers:
        warnings.append(f"Transformer models missing: {sorted(missing_transformers)}")
    
    # Check model consistency across protocols
    model_by_protocol = {p: set() for p in protocols}
    for r in results:
        model_by_protocol[r.protocol].add(r.model)
    
    all_models = set.union(*model_by_protocol.values()) if model_by_protocol else set()
    for protocol, pmodels in model_by_protocol.items():
        missing = all_models - pmodels
        if missing:
            warnings.append(f"Protocol {protocol} missing models: {sorted(missing)}")
    
    return warnings


def build_raw_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Build raw results DataFrame."""
    rows = []
    for r in results:
        row = {
            "seed": r.seed,
            "protocol": r.protocol,
            "model": r.model,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "fpr_at_tpr_90": r.fpr_at_tpr_90,
            "fpr_at_tpr_95": r.fpr_at_tpr_95,
            "asr_reduction_90": r.asr_reduction_90,
            "asr_reduction_95": r.asr_reduction_95,
            "blocked_benign_rate_90": r.blocked_benign_rate_90,
            "blocked_benign_rate_95": r.blocked_benign_rate_95,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
        }
        
        # Budget columns
        for budget_pct in BUDGET_LEVELS:
            if budget_pct in r.budget_results:
                b = r.budget_results[budget_pct]
                row[f"budget_{budget_pct}_threshold"] = b.get("threshold", np.nan)
                row[f"budget_{budget_pct}_fpr"] = b.get("fpr", np.nan)
                row[f"budget_{budget_pct}_tpr"] = b.get("tpr", np.nan)
                row[f"budget_{budget_pct}_asr"] = b.get("asr", np.nan)
                row[f"budget_{budget_pct}_blocked_benign"] = b.get("blocked_benign", np.nan)
            else:
                row[f"budget_{budget_pct}_threshold"] = np.nan
                row[f"budget_{budget_pct}_fpr"] = np.nan
                row[f"budget_{budget_pct}_tpr"] = np.nan
                row[f"budget_{budget_pct}_asr"] = np.nan
                row[f"budget_{budget_pct}_blocked_benign"] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort for determinism
    df = df.sort_values(["seed", "protocol", "model"]).reset_index(drop=True)
    
    return df


def build_summary_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw results into summary with mean ± std."""
    if raw_df.empty:
        return pd.DataFrame()
    
    # Metrics to aggregate
    metric_cols = [
        "roc_auc", "pr_auc", "fpr_at_tpr_90", "fpr_at_tpr_95",
        "asr_reduction_90", "asr_reduction_95",
        "blocked_benign_rate_90", "blocked_benign_rate_95",
        "latency_p50_ms", "latency_p95_ms",
    ]
    
    # Add budget columns
    for budget_pct in BUDGET_LEVELS:
        metric_cols.extend([
            f"budget_{budget_pct}_threshold",
            f"budget_{budget_pct}_fpr",
            f"budget_{budget_pct}_tpr",
            f"budget_{budget_pct}_asr",
            f"budget_{budget_pct}_blocked_benign",
        ])
    
    agg_rows = []
    for (protocol, model), group in raw_df.groupby(["protocol", "model"]):
        agg_row = {
            "protocol": protocol,
            "model": model,
            "n_seeds": len(group),
        }
        
        for col in metric_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    agg_row[f"{col}_mean"] = values.mean()
                    agg_row[f"{col}_std"] = values.std() if len(values) > 1 else 0.0
                else:
                    agg_row[f"{col}_mean"] = np.nan
                    agg_row[f"{col}_std"] = np.nan
        
        agg_rows.append(agg_row)
    
    summary_df = pd.DataFrame(agg_rows)
    
    # Sort for determinism
    summary_df = summary_df.sort_values(["protocol", "model"]).reset_index(drop=True)
    
    return summary_df


def build_results_json(
    results: list[EvalResult],
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build comprehensive results.json."""
    return {
        "metadata": {
            "n_results": len(results),
            "seeds": sorted(set(r.seed for r in results)),
            "protocols": sorted(set(r.protocol for r in results)),
            "models": sorted(set(r.model for r in results)),
        },
        "raw_results": raw_df.to_dict(orient="records"),
        "summary": summary_df.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/reports"),
        help="Path to reports directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/experiments"),
        help="Output directory for aggregated files",
    )
    parser.add_argument(
        "--expect-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Expected seeds to validate (e.g., --expect-seeds 0 1 2)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on validation warnings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    
    args = parser.parse_args()
    
    console.print("[bold]Aggregating Experiment Results[/bold]")
    console.print(f"Reports directory: {args.reports_dir}")
    console.print(f"Output directory: {args.output_dir}")
    console.print()
    
    # Discover evaluation outputs
    console.print("[cyan]Discovering evaluation outputs...[/cyan]")
    discovered = discover_eval_outputs(args.reports_dir)
    console.print(f"Found {len(discovered)} evaluation files")
    
    if args.verbose:
        for path, seed, protocol, model in discovered:
            console.print(f"  - seed={seed}, {protocol}/{model}: {path}")
    
    # Parse all results
    console.print("[cyan]Parsing evaluation results...[/cyan]")
    results = []
    for path, seed, protocol, model in discovered:
        result = parse_eval_json(path, seed, protocol, model)
        if result is not None:
            results.append(result)
    
    console.print(f"Parsed {len(results)} valid results")
    
    # Deduplicate
    results = deduplicate_results(results)
    console.print(f"After deduplication: {len(results)} results")
    
    if not results:
        console.print("[red]No valid results found![/red]")
        sys.exit(1)
    
    # Validate completeness
    console.print("[cyan]Validating completeness...[/cyan]")
    warnings = validate_completeness(results, args.expect_seeds, args.strict)
    
    for w in warnings:
        console.print(f"[yellow]Warning: {w}[/yellow]")
    
    if args.strict and warnings:
        console.print("[red]Strict mode: failing due to warnings[/red]")
        sys.exit(1)
    
    # Build outputs
    console.print("[cyan]Building output files...[/cyan]")
    
    raw_df = build_raw_dataframe(results)
    summary_df = build_summary_dataframe(raw_df)
    results_json = build_results_json(results, raw_df, summary_df)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files (always overwrite)
    raw_csv_path = args.output_dir / "raw_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    console.print(f"  Saved: {raw_csv_path}")
    
    summary_csv_path = args.output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    console.print(f"  Saved: {summary_csv_path}")
    
    results_json_path = args.output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    console.print(f"  Saved: {results_json_path}")
    
    # Print summary table
    console.print()
    table = Table(title="Aggregation Summary")
    table.add_column("Protocol")
    table.add_column("Model")
    table.add_column("Seeds")
    table.add_column("ROC-AUC")
    table.add_column("PR-AUC")
    table.add_column("FPR@TPR90")
    
    for _, row in summary_df.iterrows():
        roc = f"{row.get('roc_auc_mean', 0):.4f}"
        if row.get("roc_auc_std", 0) > 0:
            roc += f" ± {row['roc_auc_std']:.4f}"
        
        pr = f"{row.get('pr_auc_mean', 0):.4f}"
        if row.get("pr_auc_std", 0) > 0:
            pr += f" ± {row['pr_auc_std']:.4f}"
        
        fpr = f"{row.get('fpr_at_tpr_90_mean', 0):.4f}"
        if row.get("fpr_at_tpr_90_std", 0) > 0:
            fpr += f" ± {row['fpr_at_tpr_90_std']:.4f}"
        
        table.add_row(
            str(row["protocol"]),
            str(row["model"]),
            str(int(row["n_seeds"])),
            roc,
            pr,
            fpr,
        )
    
    console.print(table)
    
    console.print()
    console.print("[bold green]Aggregation complete![/bold green]")


if __name__ == "__main__":
    main()
