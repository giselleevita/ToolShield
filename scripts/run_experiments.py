#!/usr/bin/env python3
"""Run full experiment suite with multiple seeds and aggregate results.

This script runs the complete experimental pipeline:
1. Train all models on all split protocols
2. Evaluate on test sets with budget-based thresholds
3. Aggregate results across seeds (mean ± std)
4. Generate summary CSV for thesis tables

Usage:
    # Single seed MVT run
    python scripts/run_experiments.py --seeds 0

    # Full 3-seed run
    python scripts/run_experiments.py --seeds 0 1 2

    # Specific protocol only
    python scripts/run_experiments.py --seeds 0 --protocols S_random
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolshield.data.generate_dataset import generate_dataset
from toolshield.data.make_splits import create_splits, SplitProtocol
from toolshield.data.schema import DatasetRecord
from toolshield.evaluation.evaluator import ModelEvaluator, load_records_from_split
from toolshield.evaluation.metrics import compute_all_metrics, measure_latency
from toolshield.models.heuristic import HeuristicClassifier
from toolshield.models.heuristic_score import ScoredHeuristicClassifier
from toolshield.models.tfidf_lr import TfidfLRClassifier
from toolshield.models.transformer import TransformerClassifier
from toolshield.models.context_transformer import ContextTransformerClassifier
from toolshield.utils.io import save_jsonl, load_jsonl

console = Console()

PROTOCOLS = ["S_random", "S_attack_holdout", "S_tool_holdout"]
MODELS = ["heuristic", "heuristic_score", "tfidf_lr", "transformer", "context_transformer"]
BUDGETS = [0.01, 0.03, 0.05]


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    seed: int
    protocol: str
    model: str
    roc_auc: float
    pr_auc: float
    fpr_at_tpr_90: float
    fpr_at_tpr_95: float
    asr_reduction_90: float
    asr_reduction_95: float
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    budget_results: dict[float, dict[str, float]] = field(default_factory=dict)


def train_model(
    model_type: str,
    train_records: list[DatasetRecord],
    val_records: list[DatasetRecord],
    seed: int,
    output_dir: Path,
) -> Any:
    """Train a model and return it."""
    
    if model_type == "heuristic":
        model = HeuristicClassifier()
        model.train(train_records, val_records)
    elif model_type == "heuristic_score":
        model = ScoredHeuristicClassifier()
        model.train(train_records, val_records)
    elif model_type == "tfidf_lr":
        config = {"random_state": seed}
        model = TfidfLRClassifier(config=config)
        model.train(train_records, val_records)
    elif model_type == "transformer":
        config = {
            "model_name": "distilroberta-base",
            "max_length": 256,
            "random_state": seed,
            "epochs": 1,  # Quick training for MVT
            "batch_size": 16,
        }
        model = TransformerClassifier(config=config)
        model.train(train_records, val_records)
    elif model_type == "context_transformer":
        config = {
            "model_name": "distilroberta-base",
            "max_length": 384,
            "random_state": seed,
            "epochs": 1,
            "batch_size": 16,
        }
        model = ContextTransformerClassifier(config=config)
        model.train(train_records, val_records)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    
    return model


def evaluate_model(
    model: Any,
    model_type: str,
    test_records: list[DatasetRecord],
    val_records: list[DatasetRecord],
) -> ExperimentResult:
    """Evaluate a model and return results."""
    
    # Get predictions - models take records, not prompts
    labels = [r.label_binary for r in test_records]
    attack_goals = [r.attack_goal for r in test_records]
    scores = model.predict_scores(test_records)
    
    # Ensure scores is the right shape
    scores = np.asarray(scores).flatten()
    if len(scores) != len(labels):
        raise ValueError(f"Score shape mismatch: {len(scores)} scores vs {len(labels)} labels")
    
    # Compute main metrics
    metrics = compute_all_metrics(
        y_true=labels,
        y_scores=scores,
        attack_goals=attack_goals,
    )
    
    # Measure latency (sample for speed)
    sample_records = test_records[:100]
    latency = measure_latency(model, sample_records)
    
    # Budget-based evaluation using validation data
    evaluator = ModelEvaluator(budgets=BUDGETS)
    budget_results = {}
    
    val_labels = [r.label_binary for r in val_records]
    val_scores = model.predict_scores(val_records)
    val_scores = np.asarray(val_scores).flatten()
    
    test_attack_goals = [r.attack_goal for r in test_records]
    
    for budget in BUDGETS:
        threshold, actual_fpr, actual_tpr = evaluator._find_threshold_for_fpr_budget(
            val_labels, val_scores, budget
        )
        budget_metrics = evaluator._compute_metrics_at_threshold(
            np.array(labels), np.array(scores), test_attack_goals, threshold
        )
        budget_results[budget] = {
            "threshold": threshold,
            "fpr": budget_metrics["fpr"],
            "tpr": budget_metrics["tpr"],
            "asr_after": budget_metrics["asr"],
            "blocked_benign": budget_metrics["blocked_benign"],
        }
    
    return ExperimentResult(
        seed=0,  # Will be set by caller
        protocol="",  # Will be set by caller
        model=model_type,
        roc_auc=metrics.roc_auc,
        pr_auc=metrics.pr_auc,
        fpr_at_tpr_90=metrics.fpr_at_tpr_90,
        fpr_at_tpr_95=metrics.fpr_at_tpr_95,
        asr_reduction_90=metrics.asr_reduction_90,
        asr_reduction_95=metrics.asr_reduction_95,
        latency_p50_ms=latency.p50_ms if latency else None,
        latency_p95_ms=latency.p95_ms if latency else None,
        budget_results=budget_results,
    )


def run_single_seed(
    seed: int,
    protocols: list[str],
    models: list[str],
    data_dir: Path,
    output_dir: Path,
) -> list[ExperimentResult]:
    """Run experiments for a single seed."""
    
    results = []
    
    # Load or generate dataset
    dataset_path = data_dir / "dataset.jsonl"
    if dataset_path.exists():
        raw = load_jsonl(dataset_path)
        records = [DatasetRecord(**r) for r in raw]
    else:
        records, _ = generate_dataset(seed=1337, n_samples=1000)
        save_jsonl([r.model_dump() for r in records], dataset_path)
    
    for protocol in protocols:
        console.print(f"\n[bold blue]Protocol: {protocol}[/bold blue]")
        
        # Create splits
        split_dir = data_dir / "splits" / protocol
        if split_dir.exists():
            train_raw = load_jsonl(split_dir / "train.jsonl")
            val_raw = load_jsonl(split_dir / "val.jsonl")
            test_raw = load_jsonl(split_dir / "test.jsonl")
            train_records = [DatasetRecord(**r) for r in train_raw]
            val_records = [DatasetRecord(**r) for r in val_raw]
            test_records = [DatasetRecord(**r) for r in test_raw]
        else:
            splits = create_splits(records, protocol, seed=2026)
            train_records = splits["train"]
            val_records = splits["val"]
            test_records = splits["test"]
        
        # Check if train data has both classes
        train_labels = set(r.label_binary for r in train_records)
        if len(train_labels) < 2:
            console.print(f"  [yellow]Skipping: train set has only {train_labels} classes[/yellow]")
            console.print(f"  [dim]This protocol requires data generation fixes.[/dim]")
            continue
        
        for model_type in models:
            console.print(f"  [cyan]Training {model_type}...[/cyan]")
            
            model_dir = output_dir / f"seed_{seed}" / protocol / model_type
            
            try:
                # Train
                model = train_model(
                    model_type=model_type,
                    train_records=train_records,
                    val_records=val_records,
                    seed=seed,
                    output_dir=model_dir,
                )
                
                # Evaluate
                result = evaluate_model(
                    model=model,
                    model_type=model_type,
                    test_records=test_records,
                    val_records=val_records,
                )
                result.seed = seed
                result.protocol = protocol
                
                results.append(result)
                
                console.print(
                    f"    ROC-AUC: {result.roc_auc:.4f}, "
                    f"FPR@TPR90: {result.fpr_at_tpr_90:.4f}"
                )
                
            except Exception as e:
                console.print(f"    [red]Error: {e}[/red]")
                continue
    
    return results


def aggregate_results(results: list[ExperimentResult]) -> pd.DataFrame:
    """Aggregate results across seeds to compute mean ± std."""
    
    if not results:
        console.print("[yellow]No results to aggregate[/yellow]")
        return pd.DataFrame()
    
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
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
        }
        # Add budget results
        for budget, metrics in r.budget_results.items():
            row[f"budget_{int(budget*100)}_fpr"] = metrics["fpr"]
            row[f"budget_{int(budget*100)}_tpr"] = metrics["tpr"]
            row[f"budget_{int(budget*100)}_asr"] = metrics["asr_after"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Compute aggregates
    metrics_cols = [
        "roc_auc", "pr_auc", "fpr_at_tpr_90", "fpr_at_tpr_95",
        "asr_reduction_90", "asr_reduction_95", "latency_p50_ms", "latency_p95_ms",
    ]
    # Add budget columns
    for budget in [1, 3, 5]:
        metrics_cols.extend([
            f"budget_{budget}_fpr", f"budget_{budget}_tpr", f"budget_{budget}_asr"
        ])
    
    agg_rows = []
    for (protocol, model), group in df.groupby(["protocol", "model"]):
        agg_row = {"protocol": protocol, "model": model, "n_seeds": len(group)}
        for col in metrics_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    agg_row[f"{col}_mean"] = values.mean()
                    agg_row[f"{col}_std"] = values.std() if len(values) > 1 else 0.0
        agg_rows.append(agg_row)
    
    return pd.DataFrame(agg_rows)


def save_summary(results: list[ExperimentResult], agg_df: pd.DataFrame, output_dir: Path):
    """Save results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw results
    raw_rows = []
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
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
        }
        for budget, metrics in r.budget_results.items():
            row[f"budget_{int(budget*100)}_threshold"] = metrics["threshold"]
            row[f"budget_{int(budget*100)}_fpr"] = metrics["fpr"]
            row[f"budget_{int(budget*100)}_tpr"] = metrics["tpr"]
            row[f"budget_{int(budget*100)}_asr"] = metrics["asr_after"]
            row[f"budget_{int(budget*100)}_blocked_benign"] = metrics["blocked_benign"]
        raw_rows.append(row)
    
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "raw_results.csv", index=False)
    
    # Aggregated summary
    agg_df.to_csv(output_dir / "summary.csv", index=False)
    
    # JSON for programmatic access
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "raw": raw_rows,
            "aggregated": agg_df.to_dict(orient="records"),
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {output_dir}[/green]")


def print_summary_table(agg_df: pd.DataFrame):
    """Print a nice summary table."""
    
    table = Table(title="Experiment Results Summary")
    table.add_column("Protocol", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("PR-AUC", justify="right")
    table.add_column("FPR@TPR90", justify="right")
    table.add_column("FPR@TPR95", justify="right")
    table.add_column("ASR Red.", justify="right")
    
    for _, row in agg_df.iterrows():
        n = row.get("n_seeds", 1)
        
        def fmt(col):
            mean = row.get(f"{col}_mean", row.get(col, 0))
            std = row.get(f"{col}_std", 0)
            if n > 1 and std > 0:
                return f"{mean:.3f}±{std:.3f}"
            return f"{mean:.3f}"
        
        table.add_row(
            row["protocol"],
            row["model"],
            fmt("roc_auc"),
            fmt("pr_auc"),
            fmt("fpr_at_tpr_90"),
            fmt("fpr_at_tpr_95"),
            fmt("asr_reduction_90"),
        )
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run ToolShield experiments")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seeds to run (default: 0)"
    )
    parser.add_argument(
        "--protocols", type=str, nargs="+", default=PROTOCOLS,
        help=f"Protocols to run (default: {PROTOCOLS})"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=MODELS,
        help=f"Models to train (default: {MODELS})"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/reports/experiments"),
        help="Output directory (default: data/reports/experiments)"
    )
    parser.add_argument(
        "--skip-transformers", action="store_true",
        help="Skip transformer models (faster MVT)"
    )
    
    args = parser.parse_args()
    
    if args.skip_transformers:
        args.models = [m for m in args.models if "transformer" not in m]
    
    console.print(f"\n[bold]ToolShield Experiment Runner[/bold]")
    console.print(f"Seeds: {args.seeds}")
    console.print(f"Protocols: {args.protocols}")
    console.print(f"Models: {args.models}")
    
    all_results = []
    
    for seed in args.seeds:
        console.print(f"\n[bold yellow]===== Seed {seed} =====[/bold yellow]")
        results = run_single_seed(
            seed=seed,
            protocols=args.protocols,
            models=args.models,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        all_results.extend(results)
    
    # Aggregate
    agg_df = aggregate_results(all_results)
    
    # Save
    save_summary(all_results, agg_df, args.output_dir)
    
    # Print
    print_summary_table(agg_df)
    
    console.print("\n[bold green]Experiments complete![/bold green]")


if __name__ == "__main__":
    main()
