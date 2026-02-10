#!/usr/bin/env python3
"""Write a run manifest documenting the experiment configuration.

Captures dataset config, seeds, protocols, models, git hash, and
exact commands for reproducibility.

Usage:
    python scripts/write_run_manifest.py \
        --experiment-root data/reports/experiments_longschema \
        --dataset-config configs/dataset_longschema.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown (not a git repo)"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write run manifest")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocols", nargs="+", default=["S_random", "S_attack_holdout"])
    parser.add_argument("--models", nargs="+", default=[
        "context_transformer_naive_longschema",
        "context_transformer_keep_prompt_longschema",
    ])
    args = parser.parse_args()

    # Load dataset config
    with open(args.dataset_config) as f:
        ds_cfg = yaml.safe_load(f)

    max_schema = ds_cfg.get("inflate_schema_to", ds_cfg.get("max_schema_length", "N/A"))
    git_hash = get_git_hash()

    manifest = f"""# Run Manifest — Long-Schema Stress Test
# ========================================
# Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

Validate H1 (truncation bias) empirically: naive right-truncation
disproportionately removes prompt tokens when enterprise-length tool
schemas consume most of the token budget, while keep_prompt preserves
the attacker-controlled signal via a dedicated prompt token reservation.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset config | `{args.dataset_config}` |
| inflate_schema_to | {max_schema} characters |
| n_samples | {ds_cfg.get('n_samples', 'N/A')} |
| seed (dataset) | {ds_cfg.get('seed', 'N/A')} |
| Seeds (training) | {args.seeds} |
| Protocols | {args.protocols} |
| Models | {args.models} |
| max_length (tokens) | 256 |
| prompt_min_tokens | 128 |
| max_schema_length (model) | 5000 (no char-level pre-truncation) |
| Git commit | `{git_hash}` |

## Model Configs

- Naive: `configs/training/context_transformer_naive_longschema.yaml`
- Keep-prompt: `configs/training/context_transformer_keep_prompt_longschema.yaml`

## Controlled Variables

Only ONE variable changes between baseline and stress test:
- **Schema character length**: ~150–310 chars (baseline) → ~4000 chars (stress)

Everything else is held constant:
- Same prompts, labels, attack families, tools
- Same token budget (max_length=256)
- Same model architecture, learning rate, epochs, batch size
- Same split protocols and random seeds

## Exact Commands to Reproduce

```bash
# 1. Generate base dataset
toolshield generate --config configs/dataset.yaml --output data/

# 2. Inflate schemas to enterprise length
python scripts/inflate_schemas.py \\
    --input data/dataset.jsonl \\
    --output data/dataset_longschema.jsonl \\
    --target-chars {max_schema} --seed {ds_cfg.get('seed', 1337)}

# 3. Generate splits
toolshield split --protocol S_random \\
    --input data/dataset_longschema.jsonl \\
    --output data/splits_longschema/S_random/
toolshield split --protocol S_attack_holdout \\
    --input data/dataset_longschema.jsonl \\
    --output data/splits_longschema/S_attack_holdout/

# 4. Run truncation ablation
python scripts/run_truncation_ablation.py \\
    --seeds {' '.join(str(s) for s in args.seeds)} \\
    --protocols {' '.join(args.protocols)} \\
    --experiment-tag experiments_longschema \\
    --splits-dir data/splits_longschema \\
    --model-set longschema

# 5. Aggregate results
python scripts/aggregate_experiments.py --verbose \\
    --reports-dir data/reports/experiments_longschema \\
    --output-dir data/reports/experiments_longschema

# 6. Compute truncation statistics + figure
python scripts/report_truncation_stats.py \\
    --splits-dir data/splits_longschema \\
    --experiment-root data/reports/experiments_longschema \\
    --figures-dir data/reports/figures \\
    --config-set longschema \\
    --figure-suffix _longschema

# 7. Generate LaTeX tables
python scripts/generate_latex_from_summary.py \\
    --summary-csv data/reports/experiments_longschema/summary.csv \\
    > data/reports/experiments_longschema/latex_tables.txt

# Or simply:
make compare_truncation_longschema
```

## Expected Outputs

- `{args.experiment_root}/summary.csv` — aggregated KPIs
- `{args.experiment_root}/raw_results.csv` — per-run results
- `{args.experiment_root}/truncation_stats.csv` — token retention stats
- `{args.experiment_root}/truncation_stats.json` — detailed per-sample stats
- `{args.experiment_root}/latex_tables.txt` — thesis-ready LaTeX tables
- `data/reports/figures/truncation_bias_prompt_retention_vs_schema_longschema.png`

## Expected Results (H1 Validation)

Under enterprise-length schemas (~4000 chars → ~800+ tokens):
- **Naive truncation**: prompt tokens are completely clipped (retention ≈ 0%)
- **Keep-prompt**: prompt tokens are preserved (retention > 0%, up to ~128 tokens)
- The figure should show a clear visible separation between the two strategies.
"""

    output_path = args.experiment_root / "run_manifest.md"
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(manifest)

    print(f"Manifest written: {output_path}")


if __name__ == "__main__":
    main()
