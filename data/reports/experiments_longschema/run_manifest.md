# Run Manifest — Long-Schema Stress Test
# ========================================
# Generated: 2026-07-20T18:58:35

## Purpose

Validate H1 (truncation bias) empirically: naive right-truncation
disproportionately removes prompt tokens when enterprise-length tool
schemas consume most of the token budget, while keep_prompt preserves
the attacker-controlled signal via a dedicated prompt token reservation.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset config | `configs/dataset_longschema.yaml` |
| inflate_schema_to | 4000 characters |
| n_samples | 1000 |
| seed (dataset) | 1337 |
| Seeds (training) | [0, 1, 2] |
| Protocols | ['S_random', 'S_attack_holdout'] |
| Models | ['context_transformer_naive_longschema', 'context_transformer_keep_prompt_longschema'] |
| max_length (tokens) | 256 |
| prompt_min_tokens | 128 |
| max_schema_length (model) | 5000 (no char-level pre-truncation) |
| Git commit | `1ddd4333bbd21f2120c4a1a73928e603ea0c89d8` |

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
python scripts/inflate_schemas.py \
    --input data/dataset.jsonl \
    --output data/dataset_longschema.jsonl \
    --target-chars 4000 --seed 1337

# 3. Generate splits
toolshield split --protocol S_random \
    --input data/dataset_longschema.jsonl \
    --output data/splits_longschema/S_random/
toolshield split --protocol S_attack_holdout \
    --input data/dataset_longschema.jsonl \
    --output data/splits_longschema/S_attack_holdout/

# 4. Run truncation ablation
python scripts/run_truncation_ablation.py \
    --seeds 0 1 2 \
    --protocols S_random S_attack_holdout \
    --experiment-tag experiments_longschema \
    --splits-dir data/splits_longschema \
    --model-set longschema

# 5. Aggregate results
python scripts/aggregate_experiments.py --verbose \
    --reports-dir data/reports/experiments_longschema \
    --output-dir data/reports/experiments_longschema

# 6. Compute truncation statistics + figure
python scripts/report_truncation_stats.py \
    --splits-dir data/splits_longschema \
    --experiment-root data/reports/experiments_longschema \
    --figures-dir data/reports/figures \
    --config-set longschema \
    --figure-suffix _longschema

# 7. Generate LaTeX tables
python scripts/generate_latex_from_summary.py \
    --summary-csv data/reports/experiments_longschema/summary.csv \
    > data/reports/experiments_longschema/latex_tables.txt

# Or simply:
make compare_truncation_longschema
```

## Expected Outputs

- `data/reports/experiments_longschema/summary.csv` — aggregated KPIs
- `data/reports/experiments_longschema/raw_results.csv` — per-run results
- `data/reports/experiments_longschema/truncation_stats.csv` — token retention stats
- `data/reports/experiments_longschema/truncation_stats.json` — detailed per-sample stats
- `data/reports/experiments_longschema/latex_tables.txt` — thesis-ready LaTeX tables
- `data/reports/figures/truncation_bias_prompt_retention_vs_schema_longschema.png`

## Expected Results (H1 Validation)

Under enterprise-length schemas (~4000 chars → ~800+ tokens):
- **Naive truncation**: prompt tokens are completely clipped (retention ≈ 0%)
- **Keep-prompt**: prompt tokens are preserved (retention > 0%, up to ~128 tokens)
- The figure should show a clear visible separation between the two strategies.
