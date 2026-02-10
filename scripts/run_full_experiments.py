#!/usr/bin/env python3
"""Run full experiment suite across all protocols, seeds, and models.

This script orchestrates the complete experimental pipeline:
1. For each seed × protocol × model combination:
   - Generate splits (if needed)
   - Train model (if needed)
   - Evaluate on test set
2. Aggregate all results into summary CSV/JSON
3. Print completeness report

Usage:
    # Full 3-seed run (thesis command)
    python scripts/run_full_experiments.py

    # Single seed MVT
    python scripts/run_full_experiments.py --seeds 0

    # Specific protocols/models
    python scripts/run_full_experiments.py --seeds 0 1 --protocols S_random --models heuristic tfidf_lr

    # Skip transformer models for quick testing
    python scripts/run_full_experiments.py --seeds 0 --skip-models transformer context_transformer
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

# Defaults
ALL_PROTOCOLS = ["S_random", "S_attack_holdout", "S_tool_holdout"]
ALL_MODELS = ["heuristic", "heuristic_score", "tfidf_lr", "transformer", "context_transformer"]
DEFAULT_SEEDS = [0, 1, 2]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = DATA_DIR / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class RunResult:
    """Result of a single experiment run."""
    seed: int
    protocol: str
    model: str
    status: str  # "success", "skipped", "failed"
    duration_s: float = 0.0
    error: str | None = None
    metrics_path: str | None = None


def get_split_dir(protocol: str) -> Path:
    """Get path to split directory for a protocol."""
    return DATA_DIR / "splits" / protocol


def get_model_output_dir(seed: int, protocol: str, model: str) -> Path:
    """Get path to model output directory."""
    return EXPERIMENTS_DIR / f"seed_{seed}" / protocol / model


def get_metrics_path(seed: int, protocol: str, model: str) -> Path:
    """Get path to metrics JSON file."""
    return get_model_output_dir(seed, protocol, model) / "metrics.json"


def splits_exist(protocol: str) -> bool:
    """Check if splits exist for a protocol."""
    split_dir = get_split_dir(protocol)
    return all((split_dir / f"{split}.jsonl").exists() for split in ["train", "val", "test"])


def model_trained(seed: int, protocol: str, model: str) -> bool:
    """Check if model artifacts exist."""
    model_dir = get_model_output_dir(seed, protocol, model)
    return (model_dir / "config.json").exists()


def metrics_exist(seed: int, protocol: str, model: str) -> bool:
    """Check if evaluation metrics exist."""
    return get_metrics_path(seed, protocol, model).exists()


def run_command(cmd: list[str], description: str, cwd: Path | None = None) -> tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 1 hour"
    except Exception as e:
        return False, str(e)


def ensure_splits(protocol: str) -> bool:
    """Ensure splits exist for a protocol, creating if needed."""
    if splits_exist(protocol):
        return True
    
    console.print(f"  [cyan]Creating splits for {protocol}...[/cyan]")
    
    # Check dataset exists
    dataset_path = DATA_DIR / "dataset.jsonl"
    if not dataset_path.exists():
        console.print(f"  [red]Error: Dataset not found at {dataset_path}[/red]")
        console.print("  [yellow]Run 'make generate' first to create the dataset.[/yellow]")
        return False
    
    cmd = [
        sys.executable, "-m", "toolshield.cli", "split",
        "--protocol", protocol,
        "--input", str(dataset_path),
        "--output", str(get_split_dir(protocol)),
    ]
    
    success, output = run_command(cmd, f"Split {protocol}")
    if not success:
        console.print(f"  [red]Failed to create splits: {output[:200]}[/red]")
    return success


def train_model(seed: int, protocol: str, model: str) -> tuple[bool, str]:
    """Train a model for a specific seed/protocol combination."""
    model_dir = get_model_output_dir(seed, protocol, model)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    split_dir = get_split_dir(protocol)
    config_path = CONFIGS_DIR / "training" / f"{model}.yaml"
    
    # Use default config if specific one doesn't exist
    if not config_path.exists():
        config_path = CONFIGS_DIR / "training" / "heuristic.yaml"  # fallback
    
    cmd = [
        sys.executable, "-m", "toolshield.cli", "train",
        "--model", model,
        "--split", str(split_dir),
        "--output", str(model_dir),
    ]
    
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])
    
    return run_command(cmd, f"Train {model}")


def evaluate_model(seed: int, protocol: str, model: str) -> tuple[bool, str]:
    """Evaluate a model and save metrics."""
    model_dir = get_model_output_dir(seed, protocol, model)
    split_dir = get_split_dir(protocol)
    metrics_path = get_metrics_path(seed, protocol, model)
    
    cmd = [
        sys.executable, "-m", "toolshield.cli", "eval",
        "--model", str(model_dir),
        "--test", str(split_dir / "test.jsonl"),
        "--output", str(metrics_path),
    ]
    
    return run_command(cmd, f"Eval {model}")


def run_single_experiment(
    seed: int,
    protocol: str,
    model: str,
    force: bool = False,
) -> RunResult:
    """Run a single experiment (train + eval) for one seed/protocol/model."""
    start_time = time.time()
    
    # Check if already complete
    if not force and metrics_exist(seed, protocol, model):
        return RunResult(
            seed=seed,
            protocol=protocol,
            model=model,
            status="skipped",
            metrics_path=str(get_metrics_path(seed, protocol, model)),
        )
    
    # Ensure splits exist
    if not ensure_splits(protocol):
        return RunResult(
            seed=seed,
            protocol=protocol,
            model=model,
            status="failed",
            error="Failed to create splits",
            duration_s=time.time() - start_time,
        )
    
    # Train if needed
    if not model_trained(seed, protocol, model) or force:
        success, output = train_model(seed, protocol, model)
        if not success:
            return RunResult(
                seed=seed,
                protocol=protocol,
                model=model,
                status="failed",
                error=f"Training failed: {output[:200]}",
                duration_s=time.time() - start_time,
            )
    
    # Evaluate
    success, output = evaluate_model(seed, protocol, model)
    if not success:
        return RunResult(
            seed=seed,
            protocol=protocol,
            model=model,
            status="failed",
            error=f"Evaluation failed: {output[:200]}",
            duration_s=time.time() - start_time,
        )
    
    return RunResult(
        seed=seed,
        protocol=protocol,
        model=model,
        status="success",
        duration_s=time.time() - start_time,
        metrics_path=str(get_metrics_path(seed, protocol, model)),
    )


def run_aggregate() -> bool:
    """Run the aggregation script."""
    console.print("\n[bold cyan]Aggregating results...[/bold cyan]")
    
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_experiments.py"),
        "--verbose",
    ]
    
    success, output = run_command(cmd, "Aggregate")
    if success:
        console.print(output)
    else:
        console.print(f"[red]Aggregation failed: {output}[/red]")
    
    return success


def print_completeness_report(
    results: list[RunResult],
    expected_seeds: list[int],
    expected_protocols: list[str],
    expected_models: list[str],
) -> None:
    """Print a completeness report showing what ran and what's missing."""
    console.print("\n[bold]Completeness Report[/bold]")
    console.print("=" * 60)
    
    # Count by status
    by_status = {"success": 0, "skipped": 0, "failed": 0}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    
    total = len(results)
    expected_total = len(expected_seeds) * len(expected_protocols) * len(expected_models)
    
    console.print(f"\nTotal runs: {total} / {expected_total} expected")
    console.print(f"  [green]Success: {by_status['success']}[/green]")
    console.print(f"  [yellow]Skipped (already existed): {by_status['skipped']}[/yellow]")
    console.print(f"  [red]Failed: {by_status['failed']}[/red]")
    
    # Show failed runs
    failed = [r for r in results if r.status == "failed"]
    if failed:
        console.print("\n[red]Failed runs:[/red]")
        for r in failed:
            console.print(f"  - seed={r.seed}, {r.protocol}/{r.model}: {r.error}")
    
    # Check for missing combinations
    completed = {(r.seed, r.protocol, r.model) for r in results if r.status in ("success", "skipped")}
    missing = []
    for seed in expected_seeds:
        for protocol in expected_protocols:
            for model in expected_models:
                if (seed, protocol, model) not in completed:
                    missing.append((seed, protocol, model))
    
    if missing:
        console.print(f"\n[yellow]Missing combinations ({len(missing)}):[/yellow]")
        for seed, protocol, model in missing[:10]:
            console.print(f"  - seed={seed}, {protocol}/{model}")
        if len(missing) > 10:
            console.print(f"  ... and {len(missing) - 10} more")
    
    # Summary table by protocol
    console.print("\n[bold]Results by Protocol:[/bold]")
    table = Table()
    table.add_column("Protocol")
    table.add_column("Seeds")
    table.add_column("Models Evaluated")
    table.add_column("Status")
    
    for protocol in expected_protocols:
        protocol_results = [r for r in results if r.protocol == protocol and r.status in ("success", "skipped")]
        seeds_done = len(set(r.seed for r in protocol_results))
        models_done = set(r.model for r in protocol_results)
        
        expected_models_for_protocol = len(expected_models)
        actual_models = len(models_done)
        
        if actual_models == expected_models_for_protocol:
            status = "[green]Complete[/green]"
        else:
            status = f"[red]{actual_models}/{expected_models_for_protocol}[/red]"
        
        table.add_row(protocol, f"{seeds_done}/{len(expected_seeds)}", ", ".join(sorted(models_done)) or "-", status)
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 3-seed thesis run
  python scripts/run_full_experiments.py

  # Single seed MVT
  python scripts/run_full_experiments.py --seeds 0

  # Quick test (skip transformers)
  python scripts/run_full_experiments.py --seeds 0 --skip-models transformer context_transformer
        """,
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=["all"],
        help=f"Protocols to run (default: all = {ALL_PROTOCOLS})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        help=f"Models to run (default: all = {ALL_MODELS})",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        nargs="+",
        default=[],
        help="Models to skip (useful for excluding slow transformers)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if artifacts exist",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip final aggregation step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    
    args = parser.parse_args()
    
    # Resolve 'all' keywords
    protocols = ALL_PROTOCOLS if "all" in args.protocols else args.protocols
    models = ALL_MODELS if "all" in args.models else args.models
    
    # Apply skip
    models = [m for m in models if m not in args.skip_models]
    
    # Validate
    for p in protocols:
        if p not in ALL_PROTOCOLS:
            console.print(f"[red]Unknown protocol: {p}[/red]")
            console.print(f"Valid protocols: {ALL_PROTOCOLS}")
            sys.exit(1)
    
    for m in models:
        if m not in ALL_MODELS:
            console.print(f"[red]Unknown model: {m}[/red]")
            console.print(f"Valid models: {ALL_MODELS}")
            sys.exit(1)
    
    # Print configuration
    total_runs = len(args.seeds) * len(protocols) * len(models)
    console.print("[bold]Full Experiment Runner[/bold]")
    console.print("=" * 60)
    console.print(f"Seeds: {args.seeds}")
    console.print(f"Protocols: {protocols}")
    console.print(f"Models: {models}")
    console.print(f"Total combinations: {total_runs}")
    console.print(f"Force re-run: {args.force}")
    console.print("=" * 60)
    
    if args.dry_run:
        console.print("\n[yellow]DRY RUN - would execute:[/yellow]")
        for seed in args.seeds:
            for protocol in protocols:
                for model in models:
                    exists = metrics_exist(seed, protocol, model)
                    status = "[yellow]SKIP (exists)[/yellow]" if exists and not args.force else "[green]RUN[/green]"
                    console.print(f"  seed={seed}, {protocol}/{model}: {status}")
        sys.exit(0)
    
    # Check dataset exists
    dataset_path = DATA_DIR / "dataset.jsonl"
    if not dataset_path.exists():
        console.print(f"\n[red]Error: Dataset not found at {dataset_path}[/red]")
        console.print("[yellow]Run 'make generate' first to create the dataset.[/yellow]")
        sys.exit(1)
    
    # Run experiments
    results: list[RunResult] = []
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments...", total=total_runs)
        
        for seed in args.seeds:
            for protocol in protocols:
                for model in models:
                    progress.update(task, description=f"seed={seed}, {protocol}/{model}")
                    
                    result = run_single_experiment(
                        seed=seed,
                        protocol=protocol,
                        model=model,
                        force=args.force,
                    )
                    results.append(result)
                    
                    # Log result
                    if result.status == "success":
                        console.print(f"  [green]✓[/green] seed={seed}, {protocol}/{model} ({result.duration_s:.1f}s)")
                    elif result.status == "skipped":
                        console.print(f"  [yellow]○[/yellow] seed={seed}, {protocol}/{model} (skipped - exists)")
                    else:
                        console.print(f"  [red]✗[/red] seed={seed}, {protocol}/{model}: {result.error}")
                    
                    progress.advance(task)
    
    total_time = time.time() - start_time
    console.print(f"\n[bold]Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)[/bold]")
    
    # Aggregate results
    if not args.no_aggregate:
        run_aggregate()
    
    # Print completeness report
    print_completeness_report(results, args.seeds, protocols, models)
    
    # Exit with error if any failed
    failed_count = sum(1 for r in results if r.status == "failed")
    
    if failed_count > 0:
        console.print(f"\n[red]{failed_count} experiments failed[/red]")
        sys.exit(1)
    
    console.print("\n[bold green]All experiments completed successfully![/bold green]")


if __name__ == "__main__":
    main()
