#!/usr/bin/env python3
"""Run controlled truncation ablation: naive vs keep_prompt.

Compares two context_transformer configurations that differ ONLY in
truncation strategy, holding all other hyperparameters constant.

Experiment matrix:
  Models:    context_transformer_naive, context_transformer_keep_prompt
  Protocols: S_random, S_attack_holdout (default; S_tool_holdout optional)
  Seeds:     0, 1, 2

Output structure:
  data/reports/{experiment_tag}/seed_{s}/{protocol}/{model_name}/
    config.json, metrics.json, model/, tokenizer/

Usage:
    # Full ablation (default)
    python scripts/run_truncation_ablation.py --seeds 0 1 2

    # Quick single-seed smoke test
    python scripts/run_truncation_ablation.py --seeds 0

    # Long-schema stress test (separate experiment tag + splits dir)
    python scripts/run_truncation_ablation.py --seeds 0 1 2 \\
        --experiment-tag experiments_longschema \\
        --splits-dir data/splits_longschema \\
        --model-set longschema
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Ablation model definitions ───────────────────────────────────────────────
# Maps run name -> (CLI model arg, config YAML path)
# CLI --model must be a name accepted by toolshield train (see cli.py)

ABLATION_MODEL_SETS: dict[str, dict[str, tuple[str, Path]]] = {
    "default": {
        "context_transformer_naive": (
            "context_transformer_naive",
            CONFIGS_DIR / "training" / "context_transformer_naive.yaml",
        ),
        "context_transformer_keep_prompt": (
            "context_transformer_keep_prompt",
            CONFIGS_DIR / "training" / "context_transformer.yaml",
        ),
    },
    "longschema": {
        "context_transformer_naive_longschema": (
            "context_transformer_naive_longschema",
            CONFIGS_DIR / "training" / "context_transformer_naive_longschema.yaml",
        ),
        "context_transformer_keep_prompt_longschema": (
            "context_transformer_keep_prompt_longschema",
            CONFIGS_DIR / "training" / "context_transformer_keep_prompt_longschema.yaml",
        ),
    },
}

DEFAULT_PROTOCOLS = ["S_random", "S_attack_holdout"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_EXPERIMENT_TAG = "experiments"


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    seed: int
    protocol: str
    model: str
    status: str  # "success", "skipped", "failed"
    duration_s: float = 0.0
    error: str | None = None


# These are set by main() from CLI args before any run
_splits_dir: Path = DATA_DIR / "splits"
_experiments_dir: Path = DATA_DIR / "reports" / "experiments"


def get_split_dir(protocol: str) -> Path:
    return _splits_dir / protocol


def get_model_dir(seed: int, protocol: str, model_name: str) -> Path:
    return _experiments_dir / f"seed_{seed}" / protocol / model_name


def get_metrics_path(seed: int, protocol: str, model_name: str) -> Path:
    return get_model_dir(seed, protocol, model_name) / "metrics.json"


def metrics_exist(seed: int, protocol: str, model_name: str) -> bool:
    return get_metrics_path(seed, protocol, model_name).exists()


def model_trained(seed: int, protocol: str, model_name: str) -> bool:
    return (get_model_dir(seed, protocol, model_name) / "config.json").exists()


def splits_exist(protocol: str) -> bool:
    sd = get_split_dir(protocol)
    return all((sd / f"{s}.jsonl").exists() for s in ["train", "val", "test"])


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a subprocess, return (success, combined output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Timed out after 1 hour"
    except Exception as e:
        return False, str(e)


# ── Train + eval ─────────────────────────────────────────────────────────────

def train_model(
    seed: int, protocol: str, model_name: str, cli_model: str, config_path: Path
) -> tuple[bool, str]:
    model_dir = get_model_dir(seed, protocol, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    split_dir = get_split_dir(protocol)

    cmd = [
        sys.executable, "-m", "toolshield.cli", "train",
        "--model", cli_model,
        "--split", str(split_dir),
        "--output", str(model_dir),
        "--config", str(config_path),
    ]
    return run_command(cmd, f"Train {model_name}")


def evaluate_model(seed: int, protocol: str, model_name: str) -> tuple[bool, str]:
    model_dir = get_model_dir(seed, protocol, model_name)
    split_dir = get_split_dir(protocol)
    metrics_path = get_metrics_path(seed, protocol, model_name)

    cmd = [
        sys.executable, "-m", "toolshield.cli", "eval",
        "--model", str(model_dir),
        "--test", str(split_dir / "test.jsonl"),
        "--val", str(split_dir / "val.jsonl"),
        "--output", str(metrics_path),
        "--latency-mode", "warm",
    ]
    return run_command(cmd, f"Eval {model_name}")


def run_single(
    seed: int, protocol: str, model_name: str,
    ablation_models: dict[str, tuple[str, Path]],
    force: bool = False,
) -> RunResult:
    start = time.time()

    # Skip if already complete
    if not force and metrics_exist(seed, protocol, model_name):
        return RunResult(seed, protocol, model_name, "skipped")

    # Ensure splits
    if not splits_exist(protocol):
        return RunResult(
            seed, protocol, model_name, "failed",
            error="Splits not found — run make splits first",
        )

    cli_model, config_path = ablation_models[model_name]

    # Train if needed
    if not model_trained(seed, protocol, model_name) or force:
        ok, out = train_model(seed, protocol, model_name, cli_model, config_path)
        if not ok:
            return RunResult(
                seed, protocol, model_name, "failed",
                duration_s=time.time() - start,
                error=f"Training failed: {out[:500]}",
            )

    # Evaluate
    ok, out = evaluate_model(seed, protocol, model_name)
    if not ok:
        return RunResult(
            seed, protocol, model_name, "failed",
            duration_s=time.time() - start,
            error=f"Evaluation failed: {out[:500]}",
        )

    return RunResult(
        seed, protocol, model_name, "success",
        duration_s=time.time() - start,
    )


# ── Completeness report ─────────────────────────────────────────────────────

def print_report(results: list[RunResult]) -> None:
    table = Table(title="Truncation Ablation — Completeness Report")
    table.add_column("Seed", style="cyan")
    table.add_column("Protocol", style="cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Time (s)", justify="right")

    n_ok = n_skip = n_fail = 0
    for r in results:
        if r.status == "success":
            status_str = "[green]success[/green]"
            n_ok += 1
        elif r.status == "skipped":
            status_str = "[yellow]skipped[/yellow]"
            n_skip += 1
        else:
            status_str = f"[red]failed[/red]"
            n_fail += 1
        table.add_row(
            str(r.seed), r.protocol, r.model, status_str, f"{r.duration_s:.1f}"
        )

    console.print(table)
    total = len(results)
    console.print(
        f"\nTotal: {total} | "
        f"[green]Success: {n_ok}[/green] | "
        f"[yellow]Skipped: {n_skip}[/yellow] | "
        f"[red]Failed: {n_fail}[/red]"
    )
    if n_fail > 0:
        console.print("\n[red]Failed runs:[/red]")
        for r in results:
            if r.status == "failed":
                console.print(f"  seed={r.seed}, {r.protocol}/{r.model}: {r.error}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _splits_dir, _experiments_dir

    parser = argparse.ArgumentParser(description="Truncation ablation: naive vs keep_prompt")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip final aggregation step")
    parser.add_argument(
        "--experiment-tag", default=DEFAULT_EXPERIMENT_TAG,
        help="Experiment output folder name under data/reports/ (default: experiments)",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=None,
        help="Custom splits directory (default: data/splits)",
    )
    parser.add_argument(
        "--model-set", choices=list(ABLATION_MODEL_SETS.keys()), default="default",
        help="Which model config set to use (default or longschema)",
    )
    args = parser.parse_args()

    # Configure paths from CLI args
    _experiments_dir = DATA_DIR / "reports" / args.experiment_tag
    if args.splits_dir:
        _splits_dir = args.splits_dir
    else:
        _splits_dir = DATA_DIR / "splits"

    ablation_models = ABLATION_MODEL_SETS[args.model_set]
    models = list(ablation_models.keys())
    total = len(args.seeds) * len(args.protocols) * len(models)

    console.print("[bold]Truncation Ablation Runner[/bold]")
    console.print("=" * 60)
    console.print(f"Seeds:     {args.seeds}")
    console.print(f"Protocols: {args.protocols}")
    console.print(f"Models:    {models}")
    console.print(f"Model set: {args.model_set}")
    console.print(f"Splits:    {_splits_dir}")
    console.print(f"Output:    {_experiments_dir}")
    console.print(f"Total:     {total} combinations")
    console.print(f"Force:     {args.force}")
    console.print("=" * 60)

    results: list[RunResult] = []

    for seed in args.seeds:
        for protocol in args.protocols:
            for model_name in models:
                r = run_single(seed, protocol, model_name, ablation_models, force=args.force)
                results.append(r)

                # Print inline progress
                if r.status == "success":
                    console.print(f"  [green]✓[/green] seed={seed}, {protocol}/{model_name} ({r.duration_s:.1f}s)")
                elif r.status == "skipped":
                    console.print(f"  [yellow]–[/yellow] seed={seed}, {protocol}/{model_name} (skipped)")
                else:
                    console.print(f"  [red]✗[/red] seed={seed}, {protocol}/{model_name}: {r.error[:120] if r.error else ''}")

    print_report(results)

    # Aggregate
    if not args.skip_aggregate:
        console.print("\n[bold]Aggregating results...[/bold]")
        ok, out = run_command(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_experiments.py"),
             "--verbose",
             "--reports-dir", str(_experiments_dir),
             "--output-dir", str(_experiments_dir)],
            "Aggregate",
        )
        if ok:
            console.print("[green]Aggregation complete.[/green]")
        else:
            console.print(f"[red]Aggregation failed:[/red] {out[:300]}")


if __name__ == "__main__":
    main()
