"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from toolshield.evaluation.metrics import (
    compute_all_metrics,
    compute_asr_metrics,
    compute_blocked_benign_rate,
    compute_fpr_at_tpr,
    compute_pr_auc,
    compute_roc_auc,
)


class TestROCAUC:
    """Tests for ROC-AUC computation."""

    def test_perfect_classifier(self) -> None:
        """Test ROC-AUC for a perfect classifier."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        roc_auc = compute_roc_auc(y_true, y_scores)
        assert roc_auc == 1.0

    def test_random_classifier(self) -> None:
        """Test ROC-AUC for a random classifier (should be ~0.5)."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.random.rand(10)

        roc_auc = compute_roc_auc(y_true, y_scores)
        # Random should be around 0.5, allow some variance
        assert 0.0 <= roc_auc <= 1.0

    def test_single_class(self) -> None:
        """Test ROC-AUC with only one class (edge case)."""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        roc_auc = compute_roc_auc(y_true, y_scores)
        assert roc_auc == 0.5  # Undefined, return baseline


class TestPRAUC:
    """Tests for PR-AUC computation."""

    def test_perfect_classifier(self) -> None:
        """Test PR-AUC for a perfect classifier."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        pr_auc = compute_pr_auc(y_true, y_scores)
        assert pr_auc == 1.0

    def test_bounds(self) -> None:
        """Test that PR-AUC is within [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)

        pr_auc = compute_pr_auc(y_true, y_scores)
        assert 0.0 <= pr_auc <= 1.0


class TestFPRAtTPR:
    """Tests for FPR@TPR computation."""

    def test_perfect_classifier_fpr_at_tpr_90(self) -> None:
        """Test FPR@TPR=0.90 for a perfect classifier (should be 0)."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])

        fpr, threshold = compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.90)
        assert fpr == 0.0

    def test_fpr_at_tpr_bounds(self) -> None:
        """Test that FPR is within [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)

        fpr, threshold = compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.90)
        assert 0.0 <= fpr <= 1.0
        assert 0.0 <= threshold <= 1.0

    def test_higher_tpr_means_higher_or_equal_fpr(self) -> None:
        """Test that requiring higher TPR generally means accepting higher FPR."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)

        fpr_90, _ = compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.90)
        fpr_95, _ = compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)

        # Higher TPR requirement should mean equal or higher FPR
        assert fpr_95 >= fpr_90 - 0.01  # Small tolerance for edge cases


class TestASRMetrics:
    """Tests for ASR (Attack Success Rate) metrics."""

    def test_perfect_detection(self) -> None:
        """Test ASR for perfect attack detection."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.9, 0.95, 0.85, 0.99])
        attack_goals = [None, None, "policy_bypass", "data_exfiltration", "tool_hijack", "privilege_misuse"]

        # Threshold that catches all attacks
        metrics = compute_asr_metrics(y_true, y_scores, attack_goals, threshold=0.5)

        assert metrics["asr_before"] == 1.0
        assert metrics["asr_after"] == 0.0  # No attacks get through
        assert metrics["asr_reduction"] == 1.0

    def test_no_detection(self) -> None:
        """Test ASR when no attacks are detected."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.1, 0.15, 0.05, 0.09])  # All low scores
        attack_goals = [None, None, "policy_bypass", "data_exfiltration", "tool_hijack", "privilege_misuse"]

        # High threshold that misses all attacks
        metrics = compute_asr_metrics(y_true, y_scores, attack_goals, threshold=0.5)

        assert metrics["asr_before"] == 1.0
        assert metrics["asr_after"] == 1.0  # All attacks get through
        assert metrics["asr_reduction"] == 0.0

    def test_partial_detection(self) -> None:
        """Test ASR for partial attack detection."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.85, 0.2])  # Some detected, some not
        attack_goals = [None, None, "policy_bypass", "data_exfiltration", "tool_hijack", "privilege_misuse"]

        metrics = compute_asr_metrics(y_true, y_scores, attack_goals, threshold=0.5)

        assert metrics["asr_before"] == 1.0
        assert 0.0 < metrics["asr_after"] < 1.0  # Some attacks get through
        assert 0.0 < metrics["asr_reduction"] < 1.0


class TestBlockedBenignRate:
    """Tests for blocked benign rate (FPR at threshold)."""

    def test_no_false_positives(self) -> None:
        """Test blocked benign rate when no false positives."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        rate = compute_blocked_benign_rate(y_true, y_scores, threshold=0.5)
        assert rate == 0.0

    def test_all_benign_blocked(self) -> None:
        """Test blocked benign rate when all benign are blocked."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.9, 0.95, 0.85, 0.9, 0.95, 0.99])

        rate = compute_blocked_benign_rate(y_true, y_scores, threshold=0.5)
        assert rate == 1.0


class TestComputeAllMetrics:
    """Tests for the compute_all_metrics function."""

    def test_returns_all_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        attack_goals = [None, None, None, "policy_bypass", "data_exfiltration", "tool_hijack"]

        metrics = compute_all_metrics(y_true, y_scores, attack_goals)

        # Check all expected attributes
        assert hasattr(metrics, "roc_auc")
        assert hasattr(metrics, "pr_auc")
        assert hasattr(metrics, "fpr_at_tpr_90")
        assert hasattr(metrics, "fpr_at_tpr_95")
        assert hasattr(metrics, "threshold_at_tpr_90")
        assert hasattr(metrics, "threshold_at_tpr_95")
        assert hasattr(metrics, "asr_before")
        assert hasattr(metrics, "asr_after_90")
        assert hasattr(metrics, "asr_after_95")
        assert hasattr(metrics, "asr_reduction_90")
        assert hasattr(metrics, "asr_reduction_95")
        assert hasattr(metrics, "blocked_benign_rate_90")
        assert hasattr(metrics, "blocked_benign_rate_95")

    def test_to_dict(self) -> None:
        """Test that metrics can be converted to dictionary."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_scores)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "roc_auc" in metrics_dict
        assert "pr_auc" in metrics_dict
