.PHONY: install install-dev generate split-all split-random split-attack-holdout split-tool-holdout \
        train-all train-heuristic train-heuristic-score train-tfidf train-transformer train-context \
        eval-all eval-protocol eval-full test lint typecheck clean help \
        data splits train eval demo verify aggregate aggregate-strict \
        full_run full_run_quick full_run_fast \
        trunc_stats compare_truncation \
        data_longschema splits_longschema compare_truncation_longschema trunc_stats_longschema \
        verify_longschema_results

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DATA_DIR := data
OUTPUT_DIR := outputs
CONFIG_DIR := configs
# Set PYTHONPATH to handle spaces in directory names
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev]"

#------------------------------------------------------------------------------
# Data Generation
#------------------------------------------------------------------------------

generate: ## Generate the full dataset
	toolshield generate --config $(CONFIG_DIR)/dataset.yaml --output $(DATA_DIR)/

#------------------------------------------------------------------------------
# Split Generation
#------------------------------------------------------------------------------

split-all: split-random split-attack-holdout split-tool-holdout ## Generate all splits

split-random: ## Generate S_random split (stratified random, no template leakage)
	toolshield split --protocol S_random \
		--input $(DATA_DIR)/dataset.jsonl \
		--output $(DATA_DIR)/splits/S_random/

split-attack-holdout: ## Generate S_attack_holdout split (hold out AF4)
	toolshield split --protocol S_attack_holdout \
		--input $(DATA_DIR)/dataset.jsonl \
		--output $(DATA_DIR)/splits/S_attack_holdout/

split-tool-holdout: ## Generate S_tool_holdout split (hold out exportReport)
	toolshield split --protocol S_tool_holdout \
		--input $(DATA_DIR)/dataset.jsonl \
		--output $(DATA_DIR)/splits/S_tool_holdout/

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

train-all: train-heuristic train-heuristic-score train-tfidf train-transformer train-context ## Train all models

train-heuristic: ## Train heuristic baseline
	toolshield train --model heuristic \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/heuristic/ \
		--config $(CONFIG_DIR)/training/heuristic.yaml

train-heuristic-score: ## Train scored heuristic baseline
	toolshield train --model heuristic_score \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/heuristic_score/ \
		--config $(CONFIG_DIR)/training/heuristic_score.yaml

train-tfidf: ## Train TF-IDF + Logistic Regression
	toolshield train --model tfidf_lr \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/tfidf_lr/ \
		--config $(CONFIG_DIR)/training/tfidf_lr.yaml

train-transformer: ## Train transformer (text-only)
	toolshield train --model transformer \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/transformer/ \
		--config $(CONFIG_DIR)/training/transformer.yaml

train-context: ## Train context-augmented transformer
	toolshield train --model context_transformer \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/context_transformer/ \
		--config $(CONFIG_DIR)/training/context_transformer.yaml

#------------------------------------------------------------------------------
# Evaluation
#------------------------------------------------------------------------------

eval-all: ## Evaluate all models on S_random split
	@mkdir -p $(DATA_DIR)/reports/S_random
	@echo "Evaluating heuristic..."
	toolshield eval --model $(OUTPUT_DIR)/heuristic/ --test $(DATA_DIR)/splits/S_random/test.jsonl \
		--output $(DATA_DIR)/reports/S_random/heuristic_metrics.json
	@echo "Evaluating heuristic_score..."
	toolshield eval --model $(OUTPUT_DIR)/heuristic_score/ --test $(DATA_DIR)/splits/S_random/test.jsonl \
		--output $(DATA_DIR)/reports/S_random/heuristic_score_metrics.json
	@echo "Evaluating TF-IDF + LR..."
	toolshield eval --model $(OUTPUT_DIR)/tfidf_lr/ --test $(DATA_DIR)/splits/S_random/test.jsonl \
		--output $(DATA_DIR)/reports/S_random/tfidf_lr_metrics.json
	@echo "Evaluating transformer..."
	toolshield eval --model $(OUTPUT_DIR)/transformer/ --test $(DATA_DIR)/splits/S_random/test.jsonl \
		--output $(DATA_DIR)/reports/S_random/transformer_metrics.json
	@echo "Evaluating context transformer..."
	toolshield eval --model $(OUTPUT_DIR)/context_transformer/ --test $(DATA_DIR)/splits/S_random/test.jsonl \
		--output $(DATA_DIR)/reports/S_random/context_transformer_metrics.json
	@echo "Generating combined tables..."
	toolshield report --input-dir $(DATA_DIR)/reports/S_random/ --output $(DATA_DIR)/reports/S_random/

eval-protocol: ## Evaluate all models on a specific protocol (usage: make eval-protocol PROTOCOL=S_random)
	@mkdir -p $(DATA_DIR)/reports/$(PROTOCOL)
	@echo "Evaluating all models on $(PROTOCOL)..."
	@for model in heuristic heuristic_score tfidf_lr transformer context_transformer; do \
		echo "  Evaluating $$model..."; \
		toolshield eval --model $(OUTPUT_DIR)/$$model/ --test $(DATA_DIR)/splits/$(PROTOCOL)/test.jsonl \
			--output $(DATA_DIR)/reports/$(PROTOCOL)/$${model}_metrics.json 2>/dev/null || \
			echo "    Skipped $$model (not trained or test split missing)"; \
	done

eval-full: ## Evaluate all models on all protocols
	@echo "Evaluating on all protocols..."
	$(MAKE) eval-protocol PROTOCOL=S_random
	$(MAKE) eval-protocol PROTOCOL=S_attack_holdout
	$(MAKE) eval-protocol PROTOCOL=S_tool_holdout
	@echo "Aggregating results..."
	$(MAKE) aggregate

#------------------------------------------------------------------------------
# Aggregation
#------------------------------------------------------------------------------

aggregate: ## Aggregate all evaluation results into summary CSV/JSON
	@echo "Aggregating experiment results..."
	$(PYTHON) scripts/aggregate_experiments.py --verbose
	@echo "Results written to data/reports/experiments/"

aggregate-strict: ## Aggregate with strict validation (fail on warnings)
	$(PYTHON) scripts/aggregate_experiments.py --strict --verbose

#------------------------------------------------------------------------------
# Testing & Quality
#------------------------------------------------------------------------------

test: ## Run all tests
	pytest tests/ -v --cov=src/toolshield --cov-report=term-missing

test-splits: ## Run only split-related tests (template leakage checks)
	pytest tests/test_splits.py -v

lint: ## Run linter
	ruff check src/ tests/

lint-fix: ## Run linter with auto-fix
	ruff check src/ tests/ --fix

typecheck: ## Run type checker
	mypy src/

#------------------------------------------------------------------------------
# Utility
#------------------------------------------------------------------------------

clean: ## Remove generated files and caches
	rm -rf $(DATA_DIR)/dataset.jsonl
	rm -rf $(DATA_DIR)/manifest.json
	rm -rf $(DATA_DIR)/splits/
	rm -rf $(OUTPUT_DIR)/*/
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-models: ## Remove only model artifacts
	rm -rf $(OUTPUT_DIR)/*/

#------------------------------------------------------------------------------
# Aliases (user-friendly shortcuts)
#------------------------------------------------------------------------------

data: generate ## Alias: Generate dataset (same as 'generate')

splits: split-all ## Alias: Generate all splits (same as 'split-all')

train: train-heuristic train-heuristic-score train-tfidf train-transformer ## Train main models

eval: eval-all ## Alias: Evaluate all models (same as 'eval-all')

#------------------------------------------------------------------------------
# Demo & Verification
#------------------------------------------------------------------------------

demo: ## Launch FastAPI guard demo on localhost:8000
	$(PYTHON) scripts/run_demo.py

verify: ## Verify dataset determinism, split hygiene, and constraints
	$(PYTHON) scripts/verify_dataset.py

#------------------------------------------------------------------------------
# Full Experiment Suite (Thesis Command)
#------------------------------------------------------------------------------

full_run: ## Run full thesis experiments: 3 protocols × 5 models × 3 seeds
	@echo "Running full experiment suite..."
	$(PYTHON) scripts/run_full_experiments.py --seeds 0 1 2 --protocols all --models all

full_run_quick: ## Quick MVT run: 3 protocols × 5 models × 1 seed
	@echo "Running quick MVT experiment..."
	$(PYTHON) scripts/run_full_experiments.py --seeds 0 --protocols all --models all

full_run_fast: ## Fast run (no transformers): 3 protocols × 3 models × 1 seed
	@echo "Running fast experiment (skip transformers)..."
	$(PYTHON) scripts/run_full_experiments.py --seeds 0 --protocols all --models all --skip-models transformer context_transformer

#------------------------------------------------------------------------------
# Truncation Ablation Study
#------------------------------------------------------------------------------

trunc_stats: ## Compute truncation retention statistics + figure
	$(PYTHON) scripts/report_truncation_stats.py

compare_truncation: ## Run naive vs keep_prompt ablation, then aggregate + tables
	$(PYTHON) scripts/run_truncation_ablation.py --seeds 0 1 2 --protocols S_random S_attack_holdout
	$(PYTHON) scripts/aggregate_experiments.py --verbose
	$(PYTHON) scripts/report_truncation_stats.py
	$(PYTHON) scripts/generate_latex_from_summary.py > data/reports/latex_tables.txt

#------------------------------------------------------------------------------
# Long-Schema Stress Test (Enterprise Truncation Bias)
#------------------------------------------------------------------------------

data_longschema: ## Generate long-schema dataset (inflate schemas to ~4000 chars)
	@echo "Step 1: Generate base dataset..."
	toolshield generate --config $(CONFIG_DIR)/dataset.yaml --output $(DATA_DIR)/
	@echo "Step 2: Inflate schemas to enterprise length..."
	$(PYTHON) scripts/inflate_schemas.py \
		--input $(DATA_DIR)/dataset.jsonl \
		--output $(DATA_DIR)/dataset_longschema.jsonl \
		--target-chars 4000 --seed 1337

splits_longschema: ## Generate splits from long-schema dataset
	toolshield split --protocol S_random \
		--input $(DATA_DIR)/dataset_longschema.jsonl \
		--output $(DATA_DIR)/splits_longschema/S_random/
	toolshield split --protocol S_attack_holdout \
		--input $(DATA_DIR)/dataset_longschema.jsonl \
		--output $(DATA_DIR)/splits_longschema/S_attack_holdout/

trunc_stats_longschema: ## Compute truncation stats on long-schema data
	$(PYTHON) scripts/report_truncation_stats.py \
		--splits-dir $(DATA_DIR)/splits_longschema \
		--experiment-root $(DATA_DIR)/reports/experiments_longschema \
		--figures-dir $(DATA_DIR)/reports/figures \
		--config-set longschema \
		--figure-suffix _longschema

compare_truncation_longschema: ## Full long-schema ablation: data → splits → train → eval → stats → verify → tables
	@echo "═══════════════════════════════════════════════════════════"
	@echo " Long-Schema Stress Test: Enterprise Truncation Bias"
	@echo "═══════════════════════════════════════════════════════════"
	@echo "Step 1/9: Generate long-schema dataset..."
	$(MAKE) data_longschema
	@echo "Step 2/9: Generate splits from long-schema dataset..."
	$(MAKE) splits_longschema
	@echo "Step 3/9: Run truncation ablation (naive vs keep_prompt)..."
	$(PYTHON) scripts/run_truncation_ablation.py \
		--seeds 0 1 2 \
		--protocols S_random S_attack_holdout \
		--experiment-tag experiments_longschema \
		--splits-dir $(DATA_DIR)/splits_longschema \
		--model-set longschema
	@echo "Step 4/9: Aggregate experiment results..."
	$(PYTHON) scripts/aggregate_experiments.py --verbose \
		--reports-dir $(DATA_DIR)/reports/experiments_longschema \
		--output-dir $(DATA_DIR)/reports/experiments_longschema
	@echo "Step 5/9: Compute truncation statistics + figures..."
	$(MAKE) trunc_stats_longschema
	@echo "Step 6/9: Verify split hygiene..."
	$(PYTHON) scripts/verify_longschema_splits.py \
		--splits-dir $(DATA_DIR)/splits_longschema \
		--output-dir $(DATA_DIR)/reports/experiments_longschema
	@echo "Step 7/9: Export truncation example for appendix..."
	$(PYTHON) scripts/export_truncation_example.py \
		--protocol S_attack_holdout --index 0 \
		--splits-dir $(DATA_DIR)/splits_longschema \
		--config-set longschema \
		--out $(DATA_DIR)/reports/experiments_longschema/appendix_truncation_example.md
	@echo "Step 8/9: Generate LaTeX tables..."
	$(PYTHON) scripts/generate_latex_from_summary.py \
		--summary-csv $(DATA_DIR)/reports/experiments_longschema/summary.csv \
		--truncation-stats-csv $(DATA_DIR)/reports/experiments_longschema/truncation_stats.csv \
		> $(DATA_DIR)/reports/experiments_longschema/latex_tables.txt
	@echo "Step 9/9: Write run manifest..."
	$(PYTHON) scripts/write_run_manifest.py \
		--experiment-root $(DATA_DIR)/reports/experiments_longschema \
		--dataset-config $(CONFIG_DIR)/dataset_longschema.yaml
	@echo "═══════════════════════════════════════════════════════════"
	@echo " Long-Schema Stress Test COMPLETE"
	@echo " Results: $(DATA_DIR)/reports/experiments_longschema/"
	@echo "═══════════════════════════════════════════════════════════"

verify_longschema_results: ## Verify long-schema experiment bundle completeness and consistency
	$(PYTHON) scripts/verify_experiments_longschema_bundle.py \
		--root $(DATA_DIR)/reports/experiments_longschema \
		--expect-seeds 0 1 2 \
		--protocols S_random S_attack_holdout

#------------------------------------------------------------------------------
# Full Pipeline (Legacy)
#------------------------------------------------------------------------------

pipeline: generate split-all train-all eval-all ## Run full pipeline: generate -> split -> train -> eval

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "ToolShield - Prompt Injection Detection Research"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
