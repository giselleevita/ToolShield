"""Command-line interface for ToolShield.

Provides subcommands for:
- generate: Generate synthetic dataset
- split: Create train/val/test splits
- train: Train classifiers
- eval: Evaluate trained models

Usage:
    toolshield generate --config configs/dataset.yaml --output data/
    toolshield split --protocol S_random --input data/dataset.jsonl --output data/splits/S_random/
    toolshield train --model tfidf_lr --split data/splits/S_random/ --output outputs/tfidf_lr/
    toolshield eval --model outputs/tfidf_lr/ --test data/splits/S_random/test.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from toolshield.data.generate_dataset import generate_and_save
from toolshield.data.make_splits import SplitProtocol, create_and_save_splits
from toolshield.data.schema import DatasetRecord
from toolshield.evaluation.metrics import compute_all_metrics, print_metrics
from toolshield.utils.io import load_config, load_jsonl, save_manifest

# Initialize Typer app
app = typer.Typer(
    name="toolshield",
    help="Prompt injection detection for tool-using LLM agents",
    add_completion=False,
)

console = Console()


# =============================================================================
# Generate Command
# =============================================================================


@app.command()
def generate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to dataset configuration YAML"
    ),
    output: Path = typer.Option(
        Path("data/"), "--output", "-o", help="Output directory for dataset"
    ),
    seed: int = typer.Option(1337, "--seed", "-s", help="Random seed for generation"),
    n_samples: int = typer.Option(1000, "--n-samples", "-n", help="Number of samples to generate"),
) -> None:
    """Generate synthetic dataset for prompt injection detection."""
    console.print("[bold blue]ToolShield Dataset Generator[/bold blue]")

    # Load config if provided
    if config and config.exists():
        cfg = load_config(config)
        seed = cfg.get("seed", seed)
        n_samples = cfg.get("n_samples", n_samples)
        console.print(f"Loaded config from: {config}")

    console.print(f"Generating dataset with seed={seed}, n_samples={n_samples}")

    dataset_path, manifest_path = generate_and_save(
        output_dir=output,
        seed=seed,
        n_samples=n_samples,
    )

    console.print(f"[green]Dataset saved to:[/green] {dataset_path}")
    console.print(f"[green]Manifest saved to:[/green] {manifest_path}")

    # Display summary
    with manifest_path.open() as f:
        manifest = json.load(f)

    table = Table(title="Dataset Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Records", str(manifest["total_records"]))
    table.add_row("Seed", str(manifest["seed"]))

    for label, count in manifest["label_counts"].items():
        label_name = "Benign" if label == "0" else "Attack"
        table.add_row(f"  {label_name}", str(count))

    console.print(table)


# =============================================================================
# Split Command
# =============================================================================


@app.command()
def split(
    protocol: str = typer.Option(
        ..., "--protocol", "-p", help="Split protocol (S_random, S_attack_holdout, S_tool_holdout)"
    ),
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Input dataset.jsonl path"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for splits"
    ),
    seed: int = typer.Option(2026, "--seed", "-s", help="Random seed for splitting"),
) -> None:
    """Create train/val/test splits using specified protocol."""
    console.print("[bold blue]ToolShield Split Generator[/bold blue]")

    # Validate protocol
    valid_protocols = [p.value for p in SplitProtocol]
    if protocol not in valid_protocols:
        console.print(f"[red]Error:[/red] Invalid protocol '{protocol}'")
        console.print(f"Valid protocols: {valid_protocols}")
        raise typer.Exit(1)

    console.print(f"Creating {protocol} split with seed={seed}")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output}")

    paths = create_and_save_splits(
        input_path=input_path,
        output_dir=output,
        protocol=protocol,
        seed=seed,
    )

    console.print("[green]Splits created successfully![/green]")

    # Display summary
    table = Table(title="Split Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Records", style="green")
    table.add_column("Path", style="dim")

    for name, path in paths.items():
        records = load_jsonl(path)
        table.add_row(name, str(len(records)), str(path))

    console.print(table)


# =============================================================================
# Train Command
# =============================================================================


def _load_records(path: Path) -> list[DatasetRecord]:
    """Load records from JSONL file."""
    raw = load_jsonl(path)
    return [DatasetRecord(**r) for r in raw]


@app.command()
def train(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model type (heuristic, tfidf_lr, transformer, context_transformer)"
    ),
    split_dir: Path = typer.Option(
        ..., "--split", "-s", help="Directory containing train.jsonl and val.jsonl"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for model artifacts"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to model configuration YAML"
    ),
) -> None:
    """Train a classifier on the specified split."""
    console.print("[bold blue]ToolShield Model Training[/bold blue]")

    # Load config if provided
    model_config: dict[str, Any] = {}
    if config and config.exists():
        model_config = load_config(config)
        console.print(f"Loaded config from: {config}")

    # Load data
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"

    if not train_path.exists():
        console.print(f"[red]Error:[/red] Train file not found: {train_path}")
        raise typer.Exit(1)

    console.print(f"Loading training data from: {train_path}")
    train_records = _load_records(train_path)

    val_records = None
    if val_path.exists():
        console.print(f"Loading validation data from: {val_path}")
        val_records = _load_records(val_path)

    # Initialize model
    console.print(f"Initializing {model} model...")

    if model == "heuristic":
        from toolshield.models.heuristic import HeuristicClassifier
        classifier = HeuristicClassifier(config=model_config)
    elif model == "heuristic_score":
        from toolshield.models.heuristic_score import ScoredHeuristicClassifier
        classifier = ScoredHeuristicClassifier(config=model_config)
    elif model == "tfidf_lr":
        from toolshield.models.tfidf_lr import TfidfLRClassifier
        classifier = TfidfLRClassifier(config=model_config)
    elif model == "transformer":
        from toolshield.models.transformer import TransformerClassifier
        classifier = TransformerClassifier(config=model_config)
    elif model in (
        "context_transformer", "context_transformer_naive", "context_transformer_keep_prompt",
        "context_transformer_naive_longschema", "context_transformer_keep_prompt_longschema",
    ):
        from toolshield.models.context_transformer import ContextTransformerClassifier
        classifier = ContextTransformerClassifier(config=model_config)
    else:
        console.print(f"[red]Error:[/red] Unknown model type: {model}")
        console.print(
            "Valid models: heuristic, heuristic_score, tfidf_lr, transformer, "
            "context_transformer, context_transformer_naive, context_transformer_keep_prompt, "
            "context_transformer_naive_longschema, context_transformer_keep_prompt_longschema"
        )
        raise typer.Exit(1)

    # Train
    console.print("Training...")
    train_result = classifier.train(train_records, val_records)

    # Save model
    console.print(f"Saving model to: {output}")
    classifier.save(output)

    # Save training results
    save_manifest(train_result, output / "training_results.json")

    console.print("[green]Training complete![/green]")

    # Display summary
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in train_result.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


# =============================================================================
# Eval Command
# =============================================================================


def _load_model(model_path: Path) -> Any:
    """Load a model from its directory."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open() as f:
        model_config = json.load(f)

    model_type = model_config.get("model_type", "unknown")

    if model_type == "heuristic":
        from toolshield.models.heuristic import HeuristicClassifier
        return HeuristicClassifier.load(model_path)
    elif model_type == "heuristic_score":
        from toolshield.models.heuristic_score import ScoredHeuristicClassifier
        return ScoredHeuristicClassifier.load(model_path)
    elif model_type == "tfidf_lr":
        from toolshield.models.tfidf_lr import TfidfLRClassifier
        return TfidfLRClassifier.load(model_path)
    elif model_type == "transformer":
        from toolshield.models.transformer import TransformerClassifier
        return TransformerClassifier.load(model_path)
    elif model_type == "context_transformer":
        from toolshield.models.context_transformer import ContextTransformerClassifier
        return ContextTransformerClassifier.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@app.command(name="eval")
def evaluate(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to trained model directory"
    ),
    test_path: Path = typer.Option(
        ..., "--test", "-t", help="Path to test.jsonl file"
    ),
    val_path: Optional[Path] = typer.Option(
        None, "--val", "-v", help="Path to val.jsonl for budget threshold selection"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for evaluation results JSON"
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Classification threshold"
    ),
    measure_latency: bool = typer.Option(
        True, "--latency/--no-latency", help="Measure inference latency"
    ),
    latency_mode: str = typer.Option(
        "warm", "--latency-mode", help="Latency measurement mode: 'warm' (default, with warmup) or 'cold'"
    ),
) -> None:
    """Evaluate a trained model on test data."""
    console.print("[bold blue]ToolShield Model Evaluation[/bold blue]")

    # Validate latency_mode
    if latency_mode not in ("warm", "cold"):
        console.print(f"[red]Error:[/red] Invalid latency mode: {latency_mode}")
        console.print("Valid modes: warm, cold")
        raise typer.Exit(1)

    # Load model
    try:
        classifier = _load_model(model_path)
        model_type = classifier.config.get("model_type", "unknown")
        console.print(f"Loaded {model_type} model from: {model_path}")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Load test data
    console.print(f"Loading test data from: {test_path}")
    test_records = _load_records(test_path)
    console.print(f"Test samples: {len(test_records)}")

    # Load validation data if provided
    val_records = None
    if val_path and val_path.exists():
        console.print(f"Loading validation data from: {val_path}")
        val_records = _load_records(val_path)
        console.print(f"Validation samples: {len(val_records)}")

    # Use evaluator for full evaluation
    from toolshield.evaluation.evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator(measure_latency=measure_latency, latency_mode=latency_mode)
    
    # Infer protocol from path
    protocol = "S_random"
    path_str = str(test_path)
    if "S_attack_holdout" in path_str:
        protocol = "S_attack_holdout"
    elif "S_tool_holdout" in path_str:
        protocol = "S_tool_holdout"
    
    result = evaluator.evaluate_model(
        model=classifier,
        test_records=test_records,
        val_records=val_records,
        model_name=model_type,
        protocol=protocol,
    )

    # Print metrics
    print_metrics(result.metrics)

    # Print budget results if available
    if result.budget_results:
        console.print("\n[bold]Budget-based Threshold Evaluation:[/bold]")
        table = Table()
        table.add_column("Budget", style="cyan")
        table.add_column("Threshold")
        table.add_column("TPR")
        table.add_column("ASR")
        table.add_column("Blocked Benign")
        
        for br in result.budget_results:
            table.add_row(
                f"{br.budget:.2%}",
                f"{br.threshold:.4f}",
                f"{br.test_tpr:.4f}",
                f"{br.test_asr:.4f}",
                f"{br.test_blocked_benign:.4f}",
            )
        console.print(table)

    # Save results if output specified
    if output:
        results = result.to_dict()
        results["model_path"] = str(model_path)
        results["test_path"] = str(test_path)
        results["n_samples"] = len(test_records)
        save_manifest(results, output)
        console.print(f"[green]Results saved to:[/green] {output}")


# =============================================================================
# Report Command
# =============================================================================


@app.command()
def report(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing evaluation JSON files"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for reports"
    ),
) -> None:
    """Generate combined reports (tables.csv, metrics.json) from evaluation results."""
    console.print("[bold blue]ToolShield Report Generator[/bold blue]")
    
    from toolshield.evaluation.evaluator import EvaluationResult, ModelEvaluator
    
    # Find all metrics JSON files in input directory
    json_files = list(input_dir.glob("*_metrics.json"))
    
    if not json_files:
        console.print(f"[yellow]No metrics files found in {input_dir}[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(json_files)} evaluation files")
    
    # Load results
    results = []
    for json_file in json_files:
        with json_file.open() as f:
            data = json.load(f)
        
        # Reconstruct EvaluationResult from JSON
        from toolshield.evaluation.metrics import MetricsResult
        
        metrics_data = data.get("metrics", {})
        metrics = MetricsResult(
            roc_auc=metrics_data.get("roc_auc", 0.0),
            pr_auc=metrics_data.get("pr_auc", 0.0),
            fpr_at_tpr_90=metrics_data.get("fpr_at_tpr_90", 0.0),
            fpr_at_tpr_95=metrics_data.get("fpr_at_tpr_95", 0.0),
            threshold_at_tpr_90=metrics_data.get("threshold_at_tpr_90", 0.5),
            threshold_at_tpr_95=metrics_data.get("threshold_at_tpr_95", 0.5),
            asr_before=metrics_data.get("asr_before", 1.0),
            asr_after_90=metrics_data.get("asr_after_90", 0.0),
            asr_after_95=metrics_data.get("asr_after_95", 0.0),
            asr_reduction_90=metrics_data.get("asr_reduction_90", 0.0),
            asr_reduction_95=metrics_data.get("asr_reduction_95", 0.0),
            blocked_benign_rate_90=metrics_data.get("blocked_benign_rate_90", 0.0),
            blocked_benign_rate_95=metrics_data.get("blocked_benign_rate_95", 0.0),
            latency_p50_ms=metrics_data.get("latency_p50_ms"),
            latency_p95_ms=metrics_data.get("latency_p95_ms"),
        )
        
        # Extract model name from filename or data
        model_name = data.get("model_name", json_file.stem.replace("_metrics", ""))
        protocol = data.get("protocol", "S_random")
        
        result = EvaluationResult(
            model_name=model_name,
            protocol=protocol,
            metrics=metrics,
            budget_results=[],
        )
        results.append(result)
    
    # Generate reports
    evaluator = ModelEvaluator()
    paths = evaluator.generate_all_reports(results, output)
    
    console.print("[green]Reports generated:[/green]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")


# =============================================================================
# Info Command
# =============================================================================


@app.command()
def info() -> None:
    """Display information about ToolShield."""
    from toolshield import __version__

    console.print("[bold blue]ToolShield - Prompt Injection Detection[/bold blue]")
    console.print(f"Version: {__version__}")
    console.print()
    console.print("Available commands:")
    console.print("  generate  - Generate synthetic dataset")
    console.print("  split     - Create train/val/test splits")
    console.print("  train     - Train a classifier")
    console.print("  eval      - Evaluate a trained model")
    console.print()
    console.print("Split protocols:")
    console.print("  S_random         - Stratified random (no template leakage)")
    console.print("  S_attack_holdout - Hold out AF4 attack family")
    console.print("  S_tool_holdout   - Hold out exportReport tool")
    console.print()
    console.print("Model types:")
    console.print("  heuristic           - Rule-based keyword/pattern matching")
    console.print("  tfidf_lr            - TF-IDF + Logistic Regression")
    console.print("  transformer         - DistilRoBERTa (text only)")
    console.print("  context_transformer - DistilRoBERTa (with context)")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
