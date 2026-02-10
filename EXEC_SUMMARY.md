# ToolShield: Prompt Injection Detection for Enterprise LLM Agents

**Executive Summary for Netcompany**

---

## The Problem

Enterprise deployments of LLM agents with tool access face a critical security vulnerability: **prompt injection attacks**. These attacks manipulate agent behavior through malicious inputs, potentially leading to:

- **Data exfiltration** (customer records, API keys)
- **Unauthorized actions** (MFA resets, report exports)
- **System compromise** (indirect injection via document/tool results)

Current evaluation methods using random train/test splits **overestimate robustness** by up to 37% (see Table 5), leaving enterprises falsely confident in their defenses.

---

## Key Insight

**Random evaluation splits are misleading.**

When attack patterns seen during training also appear in testing, models appear more robust than they actually are. ToolShield introduces *holdout splits* that simulate realistic deployment scenarios:

| Split Protocol | Purpose | Key Finding |
|----------------|---------|-------------|
| S_random | Baseline (industry standard) | FPR@TPR90 = 0% |
| S_attack_holdout | Novel attack types | FPR@TPR90 = 37% |
| S_tool_holdout | Novel tools | FPR@TPR90 = TBD |

**The 37% degradation reveals the true generalization gap** that enterprises must plan for.

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

| Model | ROC-AUC | FPR@TPR90 (random) | FPR@TPR90 (holdout) | Δ Gap |
|-------|---------|---------------------|----------------------|-------|
| Heuristic | 0.81 | 100% | 100% | 0% |
| TF-IDF + LR | 1.00 | 0% | 37% | **+37%** |

**Key takeaway**: The TF-IDF model that appears "perfect" on random splits degrades significantly when facing novel attack families.

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

This research was conducted as part of a Bachelor's thesis in collaboration with Netcompany.

For questions about enterprise deployment or custom evaluation protocols, contact the thesis author.

---

*Generated: 2026-02-06 | Version: thesis-v1*
