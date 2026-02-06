"""Evaluation metrics and model evaluation for prompt injection detection."""

from toolshield.evaluation.metrics import (
    LatencyResult,
    MetricsResult,
    compute_all_metrics,
    compute_asr_metrics,
    compute_fpr_at_tpr,
    compute_pr_auc,
    compute_roc_auc,
    measure_latency,
    measure_per_sample_latency,
)
from toolshield.evaluation.evaluator import (
    BudgetResult,
    EvaluationResult,
    ModelEvaluator,
    load_records_from_split,
)

__all__ = [
    # Metrics
    "LatencyResult",
    "MetricsResult",
    "compute_all_metrics",
    "compute_asr_metrics",
    "compute_fpr_at_tpr",
    "compute_pr_auc",
    "compute_roc_auc",
    "measure_latency",
    "measure_per_sample_latency",
    # Evaluator
    "BudgetResult",
    "EvaluationResult",
    "ModelEvaluator",
    "load_records_from_split",
]
