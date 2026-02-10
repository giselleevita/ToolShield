# Legacy Configuration Files

This directory contains configuration files that are **not recommended** for holdout evaluation
but are preserved for traceability and ablation studies.

## Files

### `heuristic_policy.yaml`
Tool-aware heuristic configuration (H2) that includes tool-specific keywords.

**WARNING**: Using this configuration for tool-holdout evaluation is "cheating" because
the detector has hardcoded knowledge of held-out tool names (e.g., "resetusermfa", 
"exportReport", etc.).

Use only for:
- Ablation studies comparing generic vs. tool-aware detection
- Appendix results showing the effect of tool-specific knowledge
- Understanding the upper bound of keyword-based detection

## Recommended Configurations

For fair holdout evaluation, use the configurations in `configs/training/`:
- `heuristic.yaml` - Generic injection heuristic (H1, holdout-safe)
- `heuristic_score.yaml` - Scored heuristic with continuous output (holdout-safe)
