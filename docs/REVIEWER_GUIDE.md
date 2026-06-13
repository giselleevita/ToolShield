# ToolShield — Reviewer Guide

**Private repository — available on request.** This guide helps recruiters and senior engineers evaluate the thesis work in about 15 minutes without running full training pipelines.

## What problem it demonstrates

ToolShield researches **prompt-injection detection for tool-using LLM agents**: synthetic enterprise-tool datasets, split protocols that test generalization, and a ladder of baselines from heuristics to context-augmented transformers.

## Architecture (60 seconds)

- **CLI / library:** `src/toolshield/` — dataset generation, splits, models, evaluation
- **Models:** heuristic → TF-IDF+LR → text-only transformer → context transformer
- **Metrics:** ASR reduction, FPR@TPR, per-attack-family breakdown
- **Demo:** FastAPI guard service (`src/toolshield/demo/`)
- **CI:** Python 3.10–3.12 matrix, `make security-scan`, Docker build

## Fastest local path

```bash
make install-dev
make test          # 238 tests; no GPU required for most unit tests
make lint && make typecheck
```

Optional quick demo (no training):

```bash
make demo        # FastAPI guard on localhost:8000
```

Full thesis pipeline (`make pipeline`) downloads ML deps and trains all models — skip for a quick review.

## 15-minute review checklist

| Step | Where to look | What to verify |
|------|---------------|----------------|
| 1 | `README.md` | Problem framing, attack families AF1–AF4, split protocols |
| 2 | `src/toolshield/data/` | Dataset schema, template leakage controls |
| 3 | `src/toolshield/models/` | Baseline ladder and context-augmented classifier |
| 4 | `src/toolshield/evaluation/metrics.py` | Operational metrics beyond accuracy |
| 5 | `tests/test_splits.py`, `tests/test_context_truncation.py` | Split hygiene and truncation ablation coverage |
| 6 | `.github/workflows/ci.yml` | Test matrix + `make security-scan` gate |

**Tests to skim:** `pytest tests/test_splits.py tests/test_metrics.py tests/test_heuristic_score.py -q`

## What this is / is not

- **Is:** Reproducible research codebase with 200+ tests and ablation tooling
- **Is not:** A production SaaS guard; torch/transformers are research dependencies

## Request access

Contact via GitHub profile or portfolio site. Reviewers typically receive read access plus this guide and the main README.
