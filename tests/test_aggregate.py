"""Tests for experiment aggregation script."""

import json
import tempfile
from pathlib import Path

import pytest

# Import the aggregation module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from aggregate_experiments import (
    discover_eval_outputs,
    parse_eval_json,
    deduplicate_results,
    validate_completeness,
    build_raw_dataframe,
    build_summary_dataframe,
    EvalResult,
)


@pytest.fixture
def sample_metrics_json():
    """Sample metrics JSON content."""
    return {
        "model_name": "test_model",
        "protocol": "S_random",
        "metrics": {
            "roc_auc": 0.95,
            "pr_auc": 0.92,
            "fpr_at_tpr_90": 0.05,
            "fpr_at_tpr_95": 0.10,
            "asr_reduction_90": 0.90,
            "asr_reduction_95": 0.85,
            "blocked_benign_rate_90": 0.05,
            "blocked_benign_rate_95": 0.10,
            "latency_p50_ms": 2.5,
            "latency_p95_ms": 5.0,
        },
        "budget_results": [
            {"budget": 0.01, "threshold": 0.9, "actual_fpr": 0.01, "tpr": 0.85, "asr_after": 0.15},
            {"budget": 0.03, "threshold": 0.8, "actual_fpr": 0.03, "tpr": 0.90, "asr_after": 0.10},
            {"budget": 0.05, "threshold": 0.7, "actual_fpr": 0.05, "tpr": 0.92, "asr_after": 0.08},
        ],
        "latency": {
            "latency_p50_ms": 2.5,
            "latency_p95_ms": 5.0,
        },
    }


@pytest.fixture
def temp_reports_dir(sample_metrics_json):
    """Create a temporary reports directory with sample eval outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir)
        
        # Create S_random protocol dir with metrics
        s_random_dir = reports_dir / "S_random"
        s_random_dir.mkdir(parents=True)
        
        for model in ["heuristic", "tfidf_lr", "transformer"]:
            metrics_path = s_random_dir / f"{model}_metrics.json"
            data = sample_metrics_json.copy()
            data["model_name"] = model
            with open(metrics_path, "w") as f:
                json.dump(data, f)
        
        yield reports_dir


class TestDiscoverEvalOutputs:
    """Tests for discover_eval_outputs function."""
    
    def test_discovers_protocol_metrics(self, temp_reports_dir):
        """Should discover metrics files in protocol directories."""
        outputs = discover_eval_outputs(temp_reports_dir)
        
        assert len(outputs) == 3
        models_found = {out[3] for out in outputs}
        assert models_found == {"heuristic", "tfidf_lr", "transformer"}
    
    def test_extracts_correct_metadata(self, temp_reports_dir):
        """Should extract correct seed, protocol, and model from path."""
        outputs = discover_eval_outputs(temp_reports_dir)
        
        for path, seed, protocol, model in outputs:
            assert seed == 0  # Default seed for protocol dir
            assert protocol == "S_random"
            assert model in ["heuristic", "tfidf_lr", "transformer"]


class TestParseEvalJson:
    """Tests for parse_eval_json function."""
    
    def test_parses_valid_json(self, temp_reports_dir, sample_metrics_json):
        """Should correctly parse valid metrics JSON."""
        metrics_path = temp_reports_dir / "S_random" / "heuristic_metrics.json"
        
        result = parse_eval_json(metrics_path, seed=0, protocol="S_random", model="heuristic")
        
        assert result is not None
        assert result.roc_auc == 0.95
        assert result.pr_auc == 0.92
        assert result.fpr_at_tpr_90 == 0.05
        assert result.latency_p50_ms == 2.5
    
    def test_parses_budget_results(self, temp_reports_dir):
        """Should parse budget results correctly."""
        metrics_path = temp_reports_dir / "S_random" / "heuristic_metrics.json"
        
        result = parse_eval_json(metrics_path, seed=0, protocol="S_random", model="heuristic")
        
        assert 1 in result.budget_results
        assert 3 in result.budget_results
        assert 5 in result.budget_results
        assert result.budget_results[1]["fpr"] == 0.01
    
    def test_returns_none_for_invalid_json(self, temp_reports_dir):
        """Should return None for non-existent file."""
        fake_path = temp_reports_dir / "nonexistent.json"
        
        result = parse_eval_json(fake_path, seed=0, protocol="S_random", model="fake")
        
        assert result is None


class TestDeduplicateResults:
    """Tests for deduplicate_results function."""
    
    def test_deduplicates_by_key(self):
        """Should keep only one result per (seed, protocol, model) tuple."""
        results = [
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="path1",
                roc_auc=0.90, pr_auc=0.85, fpr_at_tpr_90=0.1, fpr_at_tpr_95=0.2,
                asr_reduction_90=0.8, asr_reduction_95=0.7,
                blocked_benign_rate_90=0.1, blocked_benign_rate_95=0.2,
            ),
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="path2",
                roc_auc=0.92, pr_auc=0.88, fpr_at_tpr_90=0.08, fpr_at_tpr_95=0.15,
                asr_reduction_90=0.85, asr_reduction_95=0.75,
                blocked_benign_rate_90=0.08, blocked_benign_rate_95=0.15,
            ),
        ]
        
        deduped = deduplicate_results(results)
        
        assert len(deduped) == 1
        # Should keep the one with higher ROC-AUC
        assert deduped[0].roc_auc == 0.92


class TestValidateCompleteness:
    """Tests for validate_completeness function."""
    
    def test_warns_single_protocol(self):
        """Should warn when only one protocol is present."""
        results = [
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="",
                roc_auc=0.9, pr_auc=0.8, fpr_at_tpr_90=0.1, fpr_at_tpr_95=0.2,
                asr_reduction_90=0.8, asr_reduction_95=0.7,
                blocked_benign_rate_90=0.1, blocked_benign_rate_95=0.2,
            ),
        ]
        
        warnings = validate_completeness(results, expected_seeds=None, strict=False)
        
        assert any("Only one protocol" in w for w in warnings)
    
    def test_warns_missing_seeds(self):
        """Should warn when expected seeds are missing."""
        results = [
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="",
                roc_auc=0.9, pr_auc=0.8, fpr_at_tpr_90=0.1, fpr_at_tpr_95=0.2,
                asr_reduction_90=0.8, asr_reduction_95=0.7,
                blocked_benign_rate_90=0.1, blocked_benign_rate_95=0.2,
            ),
        ]
        
        warnings = validate_completeness(results, expected_seeds=[0, 1, 2], strict=False)
        
        assert any("Missing expected seeds" in w for w in warnings)
    
    def test_warns_missing_transformers(self):
        """Should warn when transformer models are missing."""
        results = [
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="",
                roc_auc=0.9, pr_auc=0.8, fpr_at_tpr_90=0.1, fpr_at_tpr_95=0.2,
                asr_reduction_90=0.8, asr_reduction_95=0.7,
                blocked_benign_rate_90=0.1, blocked_benign_rate_95=0.2,
            ),
        ]
        
        warnings = validate_completeness(results, expected_seeds=None, strict=False)
        
        assert any("Transformer models missing" in w for w in warnings)


class TestBuildDataFrames:
    """Tests for DataFrame building functions."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample EvalResults for testing."""
        return [
            EvalResult(
                seed=0, protocol="S_random", model="heuristic", source_path="",
                roc_auc=0.90, pr_auc=0.85, fpr_at_tpr_90=0.10, fpr_at_tpr_95=0.20,
                asr_reduction_90=0.80, asr_reduction_95=0.70,
                blocked_benign_rate_90=0.10, blocked_benign_rate_95=0.20,
                latency_p50_ms=1.5, latency_p95_ms=2.5,
            ),
            EvalResult(
                seed=1, protocol="S_random", model="heuristic", source_path="",
                roc_auc=0.92, pr_auc=0.87, fpr_at_tpr_90=0.08, fpr_at_tpr_95=0.18,
                asr_reduction_90=0.82, asr_reduction_95=0.72,
                blocked_benign_rate_90=0.08, blocked_benign_rate_95=0.18,
                latency_p50_ms=1.6, latency_p95_ms=2.6,
            ),
        ]
    
    def test_build_raw_dataframe(self, sample_results):
        """Should build correct raw dataframe with all columns."""
        df = build_raw_dataframe(sample_results)
        
        assert len(df) == 2
        assert "seed" in df.columns
        assert "protocol" in df.columns
        assert "model" in df.columns
        assert "roc_auc" in df.columns
        assert "latency_p50_ms" in df.columns
    
    def test_build_summary_dataframe(self, sample_results):
        """Should compute mean and std correctly."""
        raw_df = build_raw_dataframe(sample_results)
        summary_df = build_summary_dataframe(raw_df)
        
        assert len(summary_df) == 1  # One model
        assert summary_df.iloc[0]["n_seeds"] == 2
        assert summary_df.iloc[0]["roc_auc_mean"] == pytest.approx(0.91, rel=0.01)
        assert summary_df.iloc[0]["roc_auc_std"] > 0
    
    def test_summary_sorted_deterministically(self, sample_results):
        """Summary should be sorted by protocol and model."""
        raw_df = build_raw_dataframe(sample_results)
        summary_df = build_summary_dataframe(raw_df)
        
        # Running twice should produce same order
        summary_df2 = build_summary_dataframe(raw_df)
        assert list(summary_df["model"]) == list(summary_df2["model"])
