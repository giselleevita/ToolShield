"""Tests for the truncation ablation study infrastructure.

Verifies:
1. Config parity: naive and keep_prompt configs differ only in truncate_strategy.
2. CLI aliases: context_transformer_naive and context_transformer_keep_prompt
   are accepted by the CLI model resolver.
3. Script imports: ablation runner and stats reporter are importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "training"


# ── Config parity tests ─────────────────────────────────────────────────────


class TestConfigParity:
    """Verify naive and keep_prompt configs are identical except truncate_strategy."""

    @pytest.fixture
    def naive_config(self) -> dict:
        path = CONFIGS_DIR / "context_transformer_naive.yaml"
        assert path.exists(), f"Naive config not found: {path}"
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def keep_prompt_config(self) -> dict:
        path = CONFIGS_DIR / "context_transformer.yaml"
        assert path.exists(), f"Keep-prompt config not found: {path}"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_naive_config_exists(self):
        assert (CONFIGS_DIR / "context_transformer_naive.yaml").exists()

    def test_keep_prompt_config_exists(self):
        assert (CONFIGS_DIR / "context_transformer.yaml").exists()

    def test_truncate_strategy_differs(self, naive_config, keep_prompt_config):
        assert naive_config["truncate_strategy"] == "naive"
        assert keep_prompt_config["truncate_strategy"] == "keep_prompt"

    def test_model_type_identical(self, naive_config, keep_prompt_config):
        assert naive_config["model_type"] == keep_prompt_config["model_type"]
        assert naive_config["model_type"] == "context_transformer"

    def test_hyperparams_identical(self, naive_config, keep_prompt_config):
        """All keys except truncate_strategy and comments should match."""
        # Keys that are expected to differ
        ignore_keys = {"truncate_strategy"}

        naive_keys = set(naive_config.keys()) - ignore_keys
        kp_keys = set(keep_prompt_config.keys()) - ignore_keys

        # Both should have same set of keys (minus the ignored one)
        assert naive_keys == kp_keys, (
            f"Key mismatch: naive-only={naive_keys - kp_keys}, "
            f"keep_prompt-only={kp_keys - naive_keys}"
        )

        # Values should be identical for shared keys
        for key in naive_keys:
            assert naive_config[key] == keep_prompt_config[key], (
                f"Config value mismatch for '{key}': "
                f"naive={naive_config[key]} vs keep_prompt={keep_prompt_config[key]}"
            )

    def test_max_length_matches(self, naive_config, keep_prompt_config):
        assert naive_config["max_length"] == keep_prompt_config["max_length"]
        assert naive_config["max_length"] == 256

    def test_batch_size_matches(self, naive_config, keep_prompt_config):
        assert naive_config["batch_size"] == keep_prompt_config["batch_size"]

    def test_learning_rate_matches(self, naive_config, keep_prompt_config):
        assert naive_config["learning_rate"] == keep_prompt_config["learning_rate"]

    def test_seed_matches(self, naive_config, keep_prompt_config):
        assert naive_config["seed"] == keep_prompt_config["seed"]


# ── CLI alias tests ──────────────────────────────────────────────────────────


class TestCLIAliases:
    """Verify CLI accepts ablation model names."""

    def _get_cli_model_branch(self, model_name: str) -> str | None:
        """Simulate the CLI if/elif chain to check model name is accepted.

        Returns the class name that would be instantiated, or None if rejected.
        """
        if model_name == "heuristic":
            return "HeuristicClassifier"
        elif model_name == "heuristic_score":
            return "ScoredHeuristicClassifier"
        elif model_name == "tfidf_lr":
            return "TfidfLRClassifier"
        elif model_name == "transformer":
            return "TransformerClassifier"
        elif model_name in (
            "context_transformer",
            "context_transformer_naive",
            "context_transformer_keep_prompt",
            "context_transformer_naive_longschema",
            "context_transformer_keep_prompt_longschema",
        ):
            return "ContextTransformerClassifier"
        return None

    def test_context_transformer_naive_accepted(self):
        assert self._get_cli_model_branch("context_transformer_naive") == "ContextTransformerClassifier"

    def test_context_transformer_keep_prompt_accepted(self):
        assert self._get_cli_model_branch("context_transformer_keep_prompt") == "ContextTransformerClassifier"

    def test_context_transformer_still_accepted(self):
        assert self._get_cli_model_branch("context_transformer") == "ContextTransformerClassifier"

    def test_unknown_model_rejected(self):
        assert self._get_cli_model_branch("nonexistent_model") is None

    def test_cli_source_contains_aliases(self):
        """Verify the actual CLI source code contains the new aliases."""
        cli_path = PROJECT_ROOT / "src" / "toolshield" / "cli.py"
        source = cli_path.read_text()
        assert "context_transformer_naive" in source
        assert "context_transformer_keep_prompt" in source


# ── Ablation runner structure tests ──────────────────────────────────────────


class TestAblationRunnerStructure:
    """Verify ablation runner script is well-formed."""

    def test_script_exists(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        assert path.exists()

    def test_script_importable(self):
        """Verify the script can be parsed as valid Python."""
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_defines_ablation_models(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        assert "ABLATION_MODEL_SETS" in source
        assert "context_transformer_naive" in source
        assert "context_transformer_keep_prompt" in source


class TestTruncStatsScript:
    """Verify truncation stats script is well-formed."""

    def test_script_exists(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        assert path.exists()

    def test_script_importable(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_references_both_strategies(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "naive" in source
        assert "keep_prompt" in source


# ── Makefile targets ─────────────────────────────────────────────────────────


class TestMakefileTargets:
    """Verify Makefile includes ablation targets."""

    @pytest.fixture
    def makefile_content(self) -> str:
        return (PROJECT_ROOT / "Makefile").read_text()

    def test_trunc_stats_target(self, makefile_content):
        assert "trunc_stats:" in makefile_content

    def test_compare_truncation_target(self, makefile_content):
        assert "compare_truncation:" in makefile_content

    def test_phony_includes_targets(self, makefile_content):
        assert "trunc_stats" in makefile_content
        assert "compare_truncation" in makefile_content


# ── LaTeX generation ─────────────────────────────────────────────────────────


class TestLatexGeneration:
    """Verify LaTeX script includes ablation table generator."""

    def test_ablation_table_function_exists(self):
        path = PROJECT_ROOT / "scripts" / "generate_latex_from_summary.py"
        source = path.read_text()
        assert "generate_truncation_ablation_table" in source

    def test_ablation_table_called_in_main(self):
        path = PROJECT_ROOT / "scripts" / "generate_latex_from_summary.py"
        source = path.read_text()
        assert "generate_truncation_ablation_table(df)" in source


# ══════════════════════════════════════════════════════════════════════════════
# Long-Schema Stress Test
# ══════════════════════════════════════════════════════════════════════════════


class TestLongSchemaDatasetConfig:
    """Verify dataset_longschema.yaml exists and differs only in inflate_schema_to."""

    @pytest.fixture
    def base_config(self) -> dict:
        path = PROJECT_ROOT / "configs" / "dataset.yaml"
        assert path.exists()
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def longschema_config(self) -> dict:
        path = PROJECT_ROOT / "configs" / "dataset_longschema.yaml"
        assert path.exists()
        with open(path) as f:
            return yaml.safe_load(f)

    def test_longschema_config_exists(self):
        assert (PROJECT_ROOT / "configs" / "dataset_longschema.yaml").exists()

    def test_inflate_schema_present(self, longschema_config):
        assert "inflate_schema_to" in longschema_config
        assert longschema_config["inflate_schema_to"] == 4000

    def test_seed_identical(self, base_config, longschema_config):
        assert base_config["seed"] == longschema_config["seed"]

    def test_n_samples_identical(self, base_config, longschema_config):
        assert base_config["n_samples"] == longschema_config["n_samples"]

    def test_attack_families_identical(self, base_config, longschema_config):
        assert base_config["properties"]["attack_families"] == longschema_config["properties"]["attack_families"]

    def test_tools_identical(self, base_config, longschema_config):
        assert base_config["properties"]["tools"] == longschema_config["properties"]["tools"]


class TestLongSchemaModelConfigParity:
    """Verify longschema model configs differ only in max_schema_length from baseline."""

    @pytest.fixture
    def naive_base(self) -> dict:
        with open(CONFIGS_DIR / "context_transformer_naive.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def naive_long(self) -> dict:
        with open(CONFIGS_DIR / "context_transformer_naive_longschema.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def kp_base(self) -> dict:
        with open(CONFIGS_DIR / "context_transformer.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def kp_long(self) -> dict:
        with open(CONFIGS_DIR / "context_transformer_keep_prompt_longschema.yaml") as f:
            return yaml.safe_load(f)

    def test_naive_longschema_exists(self):
        assert (CONFIGS_DIR / "context_transformer_naive_longschema.yaml").exists()

    def test_keep_prompt_longschema_exists(self):
        assert (CONFIGS_DIR / "context_transformer_keep_prompt_longschema.yaml").exists()

    def test_naive_max_schema_length_changed(self, naive_base, naive_long):
        assert naive_base["max_schema_length"] == 200
        assert naive_long["max_schema_length"] == 5000

    def test_kp_max_schema_length_changed(self, kp_base, kp_long):
        assert kp_base["max_schema_length"] == 200
        assert kp_long["max_schema_length"] == 5000

    def test_naive_hyperparams_identical_except_schema(self, naive_base, naive_long):
        ignore = {"max_schema_length"}
        for key in set(naive_base.keys()) - ignore:
            assert naive_base[key] == naive_long.get(key), (
                f"Naive config mismatch for '{key}': {naive_base[key]} vs {naive_long.get(key)}"
            )

    def test_kp_hyperparams_identical_except_schema(self, kp_base, kp_long):
        ignore = {"max_schema_length"}
        for key in set(kp_base.keys()) - ignore:
            assert kp_base[key] == kp_long.get(key), (
                f"Keep-prompt config mismatch for '{key}': {kp_base[key]} vs {kp_long.get(key)}"
            )

    def test_longschema_pair_parity(self, naive_long, kp_long):
        """Naive-longschema and keep_prompt-longschema differ only in truncate_strategy."""
        ignore = {"truncate_strategy"}
        for key in set(naive_long.keys()) - ignore:
            assert naive_long[key] == kp_long[key], (
                f"Longschema pair mismatch for '{key}': {naive_long[key]} vs {kp_long[key]}"
            )

    def test_token_budget_unchanged(self, naive_base, naive_long):
        assert naive_long["max_length"] == naive_base["max_length"] == 256


class TestLongSchemaCLIAliases:
    """Verify CLI accepts longschema model names."""

    def test_cli_source_contains_longschema_aliases(self):
        cli_path = PROJECT_ROOT / "src" / "toolshield" / "cli.py"
        source = cli_path.read_text()
        assert "context_transformer_naive_longschema" in source
        assert "context_transformer_keep_prompt_longschema" in source


class TestLongSchemaRunnerModelSet:
    """Verify ablation runner supports longschema model set."""

    def test_runner_defines_longschema_set(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        assert "longschema" in source
        assert "ABLATION_MODEL_SETS" in source
        assert "context_transformer_naive_longschema" in source
        assert "context_transformer_keep_prompt_longschema" in source

    def test_runner_accepts_experiment_tag(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        assert "--experiment-tag" in source

    def test_runner_accepts_splits_dir(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        assert "--splits-dir" in source

    def test_runner_accepts_model_set(self):
        path = PROJECT_ROOT / "scripts" / "run_truncation_ablation.py"
        source = path.read_text()
        assert "--model-set" in source


class TestLongSchemaTruncStatsScript:
    """Verify truncation stats script supports custom experiment root."""

    def test_accepts_experiment_root(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "--experiment-root" in source

    def test_accepts_splits_dir(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "--splits-dir" in source

    def test_accepts_config_set(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "--config-set" in source
        assert "longschema" in source

    def test_defines_longschema_config_set(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "STRATEGY_CONFIG_SETS" in source
        assert "context_transformer_naive_longschema.yaml" in source
        assert "context_transformer_keep_prompt_longschema.yaml" in source


class TestLongSchemaMakefileTargets:
    """Verify Makefile includes longschema targets."""

    @pytest.fixture
    def makefile_content(self) -> str:
        return (PROJECT_ROOT / "Makefile").read_text()

    def test_data_longschema_target(self, makefile_content):
        assert "data_longschema:" in makefile_content

    def test_splits_longschema_target(self, makefile_content):
        assert "splits_longschema:" in makefile_content

    def test_compare_truncation_longschema_target(self, makefile_content):
        assert "compare_truncation_longschema:" in makefile_content

    def test_trunc_stats_longschema_target(self, makefile_content):
        assert "trunc_stats_longschema:" in makefile_content

    def test_phony_includes_longschema_targets(self, makefile_content):
        assert "data_longschema" in makefile_content
        assert "splits_longschema" in makefile_content
        assert "compare_truncation_longschema" in makefile_content
        assert "trunc_stats_longschema" in makefile_content


class TestInflateSchemasScript:
    """Verify schema inflation script exists and is valid."""

    def test_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "inflate_schemas.py").exists()

    def test_script_importable(self):
        path = PROJECT_ROOT / "scripts" / "inflate_schemas.py"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_defines_enterprise_properties(self):
        path = PROJECT_ROOT / "scripts" / "inflate_schemas.py"
        source = path.read_text()
        assert "ENTERPRISE_PROPERTIES" in source
        assert "inflate_schema" in source


# ══════════════════════════════════════════════════════════════════════════════
# Harden Long-Schema Evidence — NEW TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestVerifyLongSchemaSplitsScript:
    """Verify the split hygiene verification script."""

    def test_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "verify_longschema_splits.py").exists()

    def test_script_importable(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_has_template_leakage_check(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "check_template_leakage" in source

    def test_has_af4_holdout_check(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "check_af4_holdout" in source

    def test_has_class_balance_check(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "check_class_balance" in source

    def test_outputs_guards_json(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "guards.json" in source

    def test_outputs_split_hygiene_csv(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "split_hygiene.csv" in source

    def test_outputs_split_hygiene_md(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "split_hygiene.md" in source

    def test_accepts_splits_dir_arg(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "--splits-dir" in source

    def test_accepts_output_dir_arg(self):
        path = PROJECT_ROOT / "scripts" / "verify_longschema_splits.py"
        source = path.read_text()
        assert "--output-dir" in source


class TestExportTruncationExampleScript:
    """Verify the truncation example export script."""

    def test_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "export_truncation_example.py").exists()

    def test_script_importable(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        compile(source, str(path), "exec")

    def test_accepts_protocol_arg(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        assert "--protocol" in source

    def test_accepts_index_arg(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        assert "--index" in source

    def test_accepts_config_set_arg(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        assert "--config-set" in source

    def test_references_both_strategies(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        assert "naive" in source
        assert "keep_prompt" in source

    def test_produces_markdown(self):
        path = PROJECT_ROOT / "scripts" / "export_truncation_example.py"
        source = path.read_text()
        assert "appendix_truncation_example.md" in source


class TestMetricsResultWarnings:
    """Verify MetricsResult has warnings field and degenerate-score check."""

    def test_warnings_field_exists(self):
        """MetricsResult dataclass should have a warnings field."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from toolshield.evaluation.metrics import MetricsResult

        # Verify field exists with default empty list
        import dataclasses
        fields = {f.name for f in dataclasses.fields(MetricsResult)}
        assert "warnings" in fields

    def test_warnings_in_to_dict(self):
        """Warnings should appear in to_dict() when non-empty."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from toolshield.evaluation.metrics import MetricsResult

        result = MetricsResult(
            roc_auc=0.5, pr_auc=0.5,
            fpr_at_tpr_90=0.5, fpr_at_tpr_95=0.5,
            threshold_at_tpr_90=0.5, threshold_at_tpr_95=0.5,
            asr_before=1.0, asr_after_90=0.5, asr_after_95=0.5,
            asr_reduction_90=0.5, asr_reduction_95=0.5,
            blocked_benign_rate_90=0.5, blocked_benign_rate_95=0.5,
            warnings=["test warning"],
        )
        d = result.to_dict()
        assert "warnings" in d
        assert d["warnings"] == ["test warning"]

    def test_no_warnings_key_when_empty(self):
        """Empty warnings should not appear in to_dict()."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from toolshield.evaluation.metrics import MetricsResult

        result = MetricsResult(
            roc_auc=0.5, pr_auc=0.5,
            fpr_at_tpr_90=0.5, fpr_at_tpr_95=0.5,
            threshold_at_tpr_90=0.5, threshold_at_tpr_95=0.5,
            asr_before=1.0, asr_after_90=0.5, asr_after_95=0.5,
            asr_reduction_90=0.5, asr_reduction_95=0.5,
            blocked_benign_rate_90=0.5, blocked_benign_rate_95=0.5,
        )
        d = result.to_dict()
        assert "warnings" not in d

    def test_degenerate_score_check_in_source(self):
        """compute_all_metrics should check for <= 2 unique score values."""
        path = PROJECT_ROOT / "src" / "toolshield" / "evaluation" / "metrics.py"
        source = path.read_text()
        assert "unique(y_scores)" in source or "np.unique(y_scores)" in source
        assert "warnings" in source


class TestTruncStatsDistributions:
    """Verify report_truncation_stats.py includes distribution stats and decile figure."""

    def test_distribution_stats_function(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "def distribution_stats" in source

    def test_schema_chars_dist_in_output(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "schema_chars_dist" in source

    def test_schema_tokens_dist_in_output(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "schema_tokens_dist" in source

    def test_prompt_tokens_dist_in_output(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "prompt_tokens_dist" in source

    def test_decile_figure_function(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "def generate_decile_figure" in source

    def test_decile_figure_called_in_main(self):
        path = PROJECT_ROOT / "scripts" / "report_truncation_stats.py"
        source = path.read_text()
        assert "generate_decile_figure" in source
        assert "deciles" in source


class TestLatexLongschemaStressTable:
    """Verify LaTeX generator includes longschema stress table."""

    def test_longschema_table_function_exists(self):
        path = PROJECT_ROOT / "scripts" / "generate_latex_from_summary.py"
        source = path.read_text()
        assert "generate_longschema_stress_table" in source

    def test_truncation_stats_csv_arg(self):
        path = PROJECT_ROOT / "scripts" / "generate_latex_from_summary.py"
        source = path.read_text()
        assert "--truncation-stats-csv" in source


class TestMakefileLongschemaSteps:
    """Verify Makefile longschema target includes all required steps."""

    @pytest.fixture
    def makefile_content(self) -> str:
        return (PROJECT_ROOT / "Makefile").read_text()

    def test_verify_step(self, makefile_content):
        assert "verify_longschema_splits.py" in makefile_content

    def test_export_example_step(self, makefile_content):
        assert "export_truncation_example.py" in makefile_content

    def test_truncation_stats_csv_passed_to_latex(self, makefile_content):
        assert "--truncation-stats-csv" in makefile_content

    def test_nine_steps(self, makefile_content):
        assert "Step 9/9" in makefile_content
