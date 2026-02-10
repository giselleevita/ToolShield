# Thesis Artifacts Index

All artifacts produced by the long-schema truncation stress test.
Reproduce with `make compare_truncation_longschema && make verify_longschema_results`.

## 1. Aggregated Results (Main Proof)

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/summary.csv`](data/reports/experiments_longschema/summary.csv) | Per-protocol, per-model aggregated metrics (ROC-AUC, PR-AUC, FPR@TPR, ASR reduction) |

## 2. Truncation Mechanism Proof

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/truncation_stats.csv`](data/reports/experiments_longschema/truncation_stats.csv) | Per-strategy prompt retention ratios and truncation counts |

## 3. Split Hygiene / No Leakage

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/split_hygiene.md`](data/reports/experiments_longschema/split_hygiene.md) | Template leakage, AF4 holdout, and class balance checks |
| [`data/reports/experiments_longschema/guards.json`](data/reports/experiments_longschema/guards.json) | Machine-readable warnings (empty = all clear) |

## 4. Concrete Appendix Example

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/appendix_truncation_example.md`](data/reports/experiments_longschema/appendix_truncation_example.md) | Token-level walkthrough of naive vs keep_prompt on a single record |

## 5. Verification

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/verification_output.txt`](data/reports/experiments_longschema/verification_output.txt) | Terminal output of `make verify_longschema_results` |

## 6. Reproducibility

| File | Description |
|------|-------------|
| [`data/reports/experiments_longschema/run_manifest.md`](data/reports/experiments_longschema/run_manifest.md) | Git hash, commands, seeds, and configuration used |
| [`data/reports/environment_freeze.txt`](data/reports/environment_freeze.txt) | `pip freeze` snapshot of the Python environment |
