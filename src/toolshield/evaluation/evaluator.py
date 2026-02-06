"""Model evaluation and reporting for prompt injection detection.

This module provides a centralized evaluator that:
- Loads models and computes predictions
- Computes all metrics (ROC-AUC, PR-AUC, FPR@TPR, ASR, latency)
- Generates tables.csv matching thesis Table 1 format
- Supports budget-based threshold selection from validation

Usage:
    from toolshield.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, test_records, val_records)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from toolshield.data.schema import DatasetRecord
from toolshield.evaluation.metrics import (
    LatencyResult,
    MetricsResult,
    compute_all_metrics,
    compute_blocked_benign_rate,
    compute_fpr_at_tpr,
    measure_latency,
)
from toolshield.models.base import BaseClassifier
from toolshield.utils.io import load_jsonl


@dataclass
class BudgetResult:
    """Result for a specific FPR budget threshold.
    
    Attributes:
        budget: Target FPR budget (e.g., 0.01, 0.03, 0.05).
        threshold: Threshold achieving the budget on validation.
        val_fpr: Actual FPR on validation at this threshold.
        val_tpr: TPR on validation at this threshold.
        test_fpr: FPR on test at this threshold.
        test_tpr: TPR on test at this threshold.
        test_asr: ASR on test at this threshold.
        test_blocked_benign: Blocked benign rate on test.
    """
    
    budget: float
    threshold: float
    val_fpr: float
    val_tpr: float
    test_fpr: float
    test_tpr: float
    test_asr: float
    test_blocked_benign: float
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "budget": self.budget,
            "threshold": self.threshold,
            "val_fpr": self.val_fpr,
            "val_tpr": self.val_tpr,
            "test_fpr": self.test_fpr,
            "test_tpr": self.test_tpr,
            "test_asr": self.test_asr,
            "test_blocked_benign": self.test_blocked_benign,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model.
    
    Attributes:
        model_name: Name of the evaluated model.
        protocol: Split protocol used.
        metrics: Standard metrics result.
        budget_results: Results for each FPR budget.
        latency: Latency measurement results.
    """
    
    model_name: str
    protocol: str
    metrics: MetricsResult
    budget_results: list[BudgetResult]
    latency: LatencyResult | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_name": self.model_name,
            "protocol": self.protocol,
            "metrics": self.metrics.to_dict(),
            "budget_results": [b.to_dict() for b in self.budget_results],
        }
        if self.latency:
            result["latency"] = self.latency.to_dict()
        return result


class ModelEvaluator:
    """Evaluator for prompt injection detection models.
    
    Provides methods for:
    - Full model evaluation with all metrics
    - Budget-based threshold selection
    - Report generation (JSON, CSV)
    """
    
    DEFAULT_BUDGETS = [0.01, 0.03, 0.05]
    
    def __init__(
        self,
        budgets: list[float] | None = None,
        measure_latency: bool = True,
        latency_n_runs: int = 10,
    ) -> None:
        """Initialize the evaluator.
        
        Args:
            budgets: FPR budgets to evaluate (default: [0.01, 0.03, 0.05]).
            measure_latency: Whether to measure inference latency.
            latency_n_runs: Number of runs for latency measurement.
        """
        self.budgets = budgets or self.DEFAULT_BUDGETS
        self.measure_latency_flag = measure_latency
        self.latency_n_runs = latency_n_runs
    
    def _find_threshold_for_fpr_budget(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_fpr: float,
    ) -> tuple[float, float, float]:
        """Find threshold achieving target FPR on validation.
        
        Args:
            y_true: True labels.
            y_scores: Predicted scores.
            target_fpr: Target FPR budget.
            
        Returns:
            Tuple of (threshold, actual_fpr, tpr_at_threshold).
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find highest threshold where FPR <= target_fpr
        valid_indices = np.where(fpr <= target_fpr)[0]
        
        if len(valid_indices) == 0:
            # Cannot achieve budget, use highest threshold
            idx = len(thresholds) - 1
        else:
            # Use the index with highest TPR among those meeting budget
            idx = valid_indices[np.argmax(tpr[valid_indices])]
        
        if idx >= len(thresholds):
            idx = len(thresholds) - 1
        
        threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        actual_fpr = float(fpr[idx])
        actual_tpr = float(tpr[idx])
        
        return threshold, actual_fpr, actual_tpr
    
    def _compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        attack_goals: list[str | None],
        threshold: float,
    ) -> dict[str, float]:
        """Compute metrics at a specific threshold.
        
        Args:
            y_true: True labels.
            y_scores: Predicted scores.
            attack_goals: Attack goals for each sample.
            threshold: Classification threshold.
            
        Returns:
            Dictionary with FPR, TPR, ASR, blocked_benign_rate.
        """
        y_pred = (y_scores >= threshold).astype(int)
        
        # FPR and TPR
        benign_mask = y_true == 0
        attack_mask = y_true == 1
        
        n_benign = np.sum(benign_mask)
        n_attack = np.sum(attack_mask)
        
        fpr = np.sum((y_true == 0) & (y_pred == 1)) / n_benign if n_benign > 0 else 0.0
        tpr = np.sum((y_true == 1) & (y_pred == 1)) / n_attack if n_attack > 0 else 0.0
        
        # ASR: attacks predicted as benign (missed)
        from toolshield.data.schema import UNSAFE_GOALS
        unsafe_outcomes = 0
        for i in range(len(y_true)):
            if y_true[i] == 1:
                predicted_benign = y_pred[i] == 0
                goal = attack_goals[i]
                has_unsafe_goal = goal is not None and goal in UNSAFE_GOALS
                if predicted_benign and has_unsafe_goal:
                    unsafe_outcomes += 1
        
        asr = unsafe_outcomes / n_attack if n_attack > 0 else 0.0
        blocked_benign = compute_blocked_benign_rate(y_true, y_scores, threshold)
        
        return {
            "fpr": float(fpr),
            "tpr": float(tpr),
            "asr": float(asr),
            "blocked_benign": float(blocked_benign),
        }
    
    def evaluate_model(
        self,
        model: BaseClassifier,
        test_records: list[DatasetRecord],
        val_records: list[DatasetRecord] | None = None,
        model_name: str = "unknown",
        protocol: str = "S_random",
    ) -> EvaluationResult:
        """Evaluate a model on test data.
        
        Args:
            model: Trained classifier model.
            test_records: Test set records (for final evaluation).
            val_records: Validation set records (for threshold selection).
            model_name: Name of the model for reporting.
            protocol: Split protocol name.
            
        Returns:
            EvaluationResult with all metrics and budget results.
        """
        # Get test predictions
        test_scores = model.predict_scores(test_records)
        test_labels = BaseClassifier.extract_labels(test_records)
        test_goals = BaseClassifier.extract_attack_goals(test_records)
        
        # Measure latency
        latency = None
        if self.measure_latency_flag:
            latency = measure_latency(model, test_records, n_runs=self.latency_n_runs)
        
        # Compute main metrics
        metrics = compute_all_metrics(
            test_labels, test_scores, test_goals, latency=latency
        )
        
        # Budget-based evaluation (requires validation set)
        budget_results: list[BudgetResult] = []
        
        if val_records is not None:
            val_scores = model.predict_scores(val_records)
            val_labels = BaseClassifier.extract_labels(val_records)
            val_goals = BaseClassifier.extract_attack_goals(val_records)
            
            for budget in self.budgets:
                # Find threshold on validation
                threshold, val_fpr, val_tpr = self._find_threshold_for_fpr_budget(
                    val_labels, val_scores, budget
                )
                
                # Evaluate on test with that threshold
                test_metrics = self._compute_metrics_at_threshold(
                    test_labels, test_scores, test_goals, threshold
                )
                
                budget_results.append(BudgetResult(
                    budget=budget,
                    threshold=threshold,
                    val_fpr=val_fpr,
                    val_tpr=val_tpr,
                    test_fpr=test_metrics["fpr"],
                    test_tpr=test_metrics["tpr"],
                    test_asr=test_metrics["asr"],
                    test_blocked_benign=test_metrics["blocked_benign"],
                ))
        
        return EvaluationResult(
            model_name=model_name,
            protocol=protocol,
            metrics=metrics,
            budget_results=budget_results,
            latency=latency,
        )
    
    def generate_metrics_json(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
    ) -> Path:
        """Generate metrics.json file from evaluation results.
        
        Args:
            results: List of evaluation results.
            output_path: Path to write metrics.json.
            
        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_data = {
            "results": [r.to_dict() for r in results],
            "summary": {
                "n_models": len(results),
                "protocols": list(set(r.protocol for r in results)),
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        return output_path
    
    def generate_tables_csv(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
    ) -> Path:
        """Generate tables.csv file matching thesis Table 1 format.
        
        Columns: model, split, fpr_at_tpr_0_90, fpr_at_tpr_0_95, pr_auc, roc_auc
        
        Args:
            results: List of evaluation results.
            output_path: Path to write tables.csv.
            
        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for result in results:
            rows.append({
                "model": result.model_name,
                "split": result.protocol,
                "fpr_at_tpr_0_90": result.metrics.fpr_at_tpr_90,
                "fpr_at_tpr_0_95": result.metrics.fpr_at_tpr_95,
                "pr_auc": result.metrics.pr_auc,
                "roc_auc": result.metrics.roc_auc,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_budget_tables_csv(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
    ) -> Path:
        """Generate budget evaluation tables.
        
        Columns: model, split, budget, threshold, tpr, asr, blocked_benign_rate
        
        Args:
            results: List of evaluation results.
            output_path: Path to write budget_tables.csv.
            
        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for result in results:
            for budget_result in result.budget_results:
                rows.append({
                    "model": result.model_name,
                    "split": result.protocol,
                    "budget": budget_result.budget,
                    "threshold": budget_result.threshold,
                    "tpr": budget_result.test_tpr,
                    "asr": budget_result.test_asr,
                    "blocked_benign_rate": budget_result.test_blocked_benign,
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            # Create empty file with headers
            pd.DataFrame(columns=[
                "model", "split", "budget", "threshold", "tpr", "asr", "blocked_benign_rate"
            ]).to_csv(output_path, index=False)
        
        return output_path
    
    def generate_all_reports(
        self,
        results: list[EvaluationResult],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Generate all report files.
        
        Args:
            results: List of evaluation results.
            output_dir: Directory to write reports.
            
        Returns:
            Dictionary mapping report names to paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Main metrics JSON
        paths["metrics.json"] = self.generate_metrics_json(
            results, output_dir / "metrics.json"
        )
        
        # Main results table
        paths["tables.csv"] = self.generate_tables_csv(
            results, output_dir / "tables.csv"
        )
        
        # Budget evaluation table
        paths["budget_tables.csv"] = self.generate_budget_tables_csv(
            results, output_dir / "budget_tables.csv"
        )
        
        return paths


def load_records_from_split(split_dir: str | Path) -> dict[str, list[DatasetRecord]]:
    """Load train/val/test records from a split directory.
    
    Args:
        split_dir: Directory containing train.jsonl, val.jsonl, test.jsonl.
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to records.
    """
    split_dir = Path(split_dir)
    records = {}
    
    for split_name in ["train", "val", "test"]:
        path = split_dir / f"{split_name}.jsonl"
        if path.exists():
            raw = load_jsonl(path)
            records[split_name] = [DatasetRecord(**r) for r in raw]
        else:
            records[split_name] = []
    
    return records
