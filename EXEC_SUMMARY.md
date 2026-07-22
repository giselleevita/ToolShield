# ToolShield: Prompt Injection Detection for Enterprise LLM Agents

**Executive Summary**

---

## The Problem

Enterprise deployments of LLM agents with tool access face a critical security vulnerability: **prompt injection attacks**. These attacks manipulate agent behavior through malicious inputs, potentially leading to:

- **Data exfiltration** (customer records, API keys)
- **Unauthorized actions** (MFA resets, report exports)
- **System compromise** (indirect injection via document/tool results)

Current evaluation methods using random train/test splits **overestimate robustness** by up to 37 percentage points of FPR@TPR90 (TF-IDF+LR on novel attack families; transformer models instead degrade by 31–35 points on novel tools — see below), leaving enterprises falsely confident in their defenses.

---

## Key Insight

**Random evaluation splits are misleading.**

When attack patterns seen during training also appear in testing, models appear more robust than they actually are. ToolShield introduces *holdout splits* that simulate realistic deployment scenarios. FPR@TPR90 (lower is better) by model and split:

| Model | S_random (baseline) | S_attack_holdout (novel attacks) | S_tool_holdout (novel tools) |
|-------|:---:|:---:|:---:|
| TF-IDF + LR | 0% | **37%** | 17% |
| Transformer (text-only) | 0% | 0% | **31%** |
| Context-Transformer | 0% | 0% | **35%** |

**The generalization gap is real but model-dependent.** Under the industry-standard random split every strong model looks near-perfect (0% FPR@TPR90). Holdout splits reveal the true gap: the TF-IDF baseline degrades most on **novel attack families** (0% → 37%), while transformer models resist novel attacks (0%) but degrade on **novel tools** (0% → 31–35%). No single model is robust to both distribution shifts — the shift enterprises must plan for depends on the model they deploy.

> Figures are FPR@TPR90 from the committed `seed_0` results. Cross-seed variance for the neural models is not yet established — see the note under *Main Results*.

---

## What ToolShield Provides

### 1. Benchmark Dataset
- 1,000 samples: 50% benign, 50% attack
- 4 attack families (AF1-4): instruction override, data exfiltration, tool hijacking, indirect injection
- 4 enterprise tools: customer records, ticket search, report export, MFA reset

### 2. Evaluation Protocols
- **Template-isolated splits** (no data leakage)
- **Attack family holdout** (test on unseen attack types)
- **Tool holdout** (test on unseen tools)

### 3. Operationally-Relevant Metrics
- **FPR@TPR(0.90, 0.95)**: False positive rate at fixed detection rates
- **ASR reduction**: Attack success rate before/after deployment
- **Budget-based evaluation**: Threshold selection for 1%, 3%, 5% FPR budgets
- **Latency P50/P95**: Inference speed for production SLAs

### 4. Production-Ready Guard Service
- FastAPI endpoint (`POST /guard`)
- Configurable FPR budgets
- Audit logging (hashed prompts for privacy)
- Sub-millisecond latency

---

## Main Results

| Model | ROC-AUC (attack holdout) | FPR@TPR90 (random) | FPR@TPR90 (attack holdout) | Δ Gap |
|-------|:---:|:---:|:---:|:---:|
| Heuristic | 0.85 | 100% | 100% | 0% |
| TF-IDF + LR | 0.94 | 0% | 37% | **+37%** |

**Key takeaway**: The TF-IDF model that appears "perfect" on random splits degrades significantly when facing novel attack families.

> **Reproducibility note.** All figures above are from the committed `seed_0` results and are exactly reproducible from `data/reports/experiments/`. The reported runs used a fixed *effective* training seed for the transformer models (a seed-propagation defect, now fixed in `scripts/run_experiments.py` and `models/transformer.py`), so the current zero cross-seed variance for those models is an artifact, not evidence of seed-robustness. Re-run `python scripts/run_experiments.py --seeds 0 1 2` after the fix to obtain genuine mean ± std. See the thesis *Threats to Validity* chapter.

---

## Integration Guide

### Quick Start
```bash
# Install
pip install toolshield

# Run guard service
make demo
```

### API Usage
```python
import requests

response = requests.post(
    "http://localhost:8000/guard",
    json={
        "prompt": "Show me customer 12345",
        "tool_name": "getCustomerRecord",
        "fpr_budget": 0.03  # 3% false positive budget
    }
)

# Response
{
    "decision": "ALLOW",
    "score": 0.12,
    "threshold": 0.45,
    "audit_id": "abc123",
    "explanation": "Score 0.12 is below threshold 0.45 for 3% FPR budget"
}
```

### Enterprise Deployment
1. **Choose FPR budget** based on acceptable false positive rate
2. **Deploy guard service** as sidecar to LLM agent
3. **Monitor audit logs** for detection patterns
4. **Retrain periodically** as new attacks emerge

---

## Contact

This research was conducted as part of a Bachelor's thesis.

For questions about enterprise deployment or custom evaluation protocols, contact the thesis author.

---

*Generated: 2026-02-06 | Version: thesis-v1*
