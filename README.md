# ToolShield: Prompt Injection Detection for Tool-Using LLM Agents

A research-grade repository for detecting prompt injection attacks in tool-using LLM agents.
Developed for bachelor's thesis research on LLM security.

## Overview

This project implements:

1. **Dataset Generation**: Synthetic dataset of benign and adversarial prompts targeting enterprise tools
2. **Split Protocols**: Three evaluation protocols to test generalization
3. **Baseline Models**: From rule-based heuristics to transformer classifiers
4. **Comprehensive Metrics**: Including operational metrics like FPR@TPR and ASR reduction

## Quick Start

```bash
# Install dependencies
make install-dev

# Generate dataset (uses seed=1337 for reproducibility)
make generate

# Create all splits (uses seed=2026)
make split-all

# Train all models
make train-all

# Evaluate
make eval-all

# Or run the full pipeline
make pipeline
```

## Project Structure

```
ToolShield/
├── configs/               # Configuration files
│   ├── dataset.yaml      # Dataset generation config
│   ├── splits.yaml       # Split generation config
│   └── training/         # Model training configs
├── src/toolshield/       # Main package
│   ├── cli.py            # CLI entry point
│   ├── data/             # Data generation and splitting
│   ├── models/           # Classifier implementations
│   ├── evaluation/       # Metrics computation
│   └── utils/            # I/O utilities
├── tests/                # Unit tests
├── data/                 # Generated data (gitignored)
└── outputs/              # Model artifacts (gitignored)
```

## Dataset Schema

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier (deterministic from seed) |
| `language` | str | Language code ("en") |
| `role_sequence` | list[str] | Conversation roles |
| `tool_name` | str | Target tool name |
| `tool_schema` | dict | JSON schema of tool |
| `tool_description` | str | Natural language description |
| `prompt` | str | The prompt text |
| `label_binary` | int | 0=benign, 1=attack |
| `attack_family` | str\|None | AF1-AF4 or None |
| `attack_goal` | str\|None | Specific attack goal |
| `template_id` | str | Template identifier |
| `variant_id` | str | Variant within template |
| `seed` | int | Generation seed |

## Attack Families

| ID | Family | Goal |
|----|--------|------|
| AF1 | Instruction Override | policy_bypass |
| AF2 | Data Exfiltration | data_exfiltration |
| AF3 | Tool Hijacking | tool_hijack |
| AF4 | Indirect Injection | privilege_misuse |

## Split Protocols

### S_random
Stratified random split (70/15/15) with **no template leakage** - 
no `template_id` appears in more than one split.

### S_attack_holdout
Tests generalization to unseen attack types:
- Train/Val: AF1, AF2, AF3 + benign
- Test: AF4 + benign subset

### S_tool_holdout
Tests generalization to unseen tools:
- Train/Val: All tools except `exportReport`
- Test: All `exportReport` samples

## Evaluation Metrics

- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **FPR@TPR(0.90)**: False positive rate at 90% true positive rate
- **FPR@TPR(0.95)**: False positive rate at 95% true positive rate
- **ASR Reduction**: Attack success rate reduction (operational metric)

## CLI Reference

```bash
# Generate dataset
toolshield generate --config configs/dataset.yaml --output data/

# Create splits
toolshield split --protocol S_random --input data/dataset.jsonl --output data/splits/S_random/

# Train model
toolshield train --model tfidf_lr --split data/splits/S_random/ --output outputs/tfidf_lr/

# Evaluate
toolshield eval --model outputs/tfidf_lr/ --test data/splits/S_random/test.jsonl
```

## Reproducibility

All random operations use fixed seeds:
- Dataset generation: `seed=1337`
- Split generation: `seed=2026`

Seeds are stored in config files and embedded in output manifests.

## Running Tests

```bash
# Run all tests
make test

# Run only split leakage tests
make test-splits

# Run with coverage
pytest tests/ -v --cov=src/toolshield
```

## Reproducing Results

```bash
make compare_truncation_longschema
make verify_longschema_results
```

## Long-Schema Stress Test (Enterprise Truncation Bias)

An ablation study comparing **naive right-truncation** vs. **prompt-preserving truncation**
under enterprise-length tool schemas (~4 000 chars). This is the key experiment backing
the thesis claim that naive tokenization silently destroys the prompt signal.

### Verification Output

```
$ make verify_longschema_results
Verifying long-schema bundle: data/reports/experiments_longschema
  Protocols: ['S_random', 'S_attack_holdout']
  Expected seeds: [0, 1, 2] (n=3)

[OK] summary.csv — all models present, n_seeds correct, ROC-AUC bounds satisfied
[OK] truncation_stats.csv — truncation patterns match expected values
[OK] split hygiene — all checks PASS, guards.json empty
[OK] appendix example — contains both strategy comparisons
[OK] raw metrics present — seed_0/S_attack_holdout has both model metrics

All 5 verification checks passed.
```

### Key Results

| Strategy | ROC-AUC (S_attack_holdout) | Prompt Retention |
|----------|---------------------------|-----------------|
| `keep_prompt` | **0.998** | 100% |
| `naive` | 0.455 | 0% |

All artifacts are in `data/reports/experiments_longschema/`. The raw verification
log is saved at `data/reports/experiments_longschema/verification_output.txt`.

## License

MIT License - See LICENSE file for details.
