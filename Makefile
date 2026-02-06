.PHONY: install install-dev generate split-all split-random split-attack-holdout split-tool-holdout \
        train-all train-heuristic train-tfidf train-transformer train-context \
        eval-all test lint typecheck clean help \
        data splits train eval demo verify

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DATA_DIR := data
OUTPUT_DIR := outputs
CONFIG_DIR := configs

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

train-all: train-heuristic train-tfidf train-transformer train-context ## Train all models

train-heuristic: ## Train heuristic baseline
	toolshield train --model heuristic \
		--split $(DATA_DIR)/splits/S_random/ \
		--output $(OUTPUT_DIR)/heuristic/ \
		--config $(CONFIG_DIR)/training/heuristic.yaml

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

train: train-tfidf train-transformer ## Train main models (tfidf_lr + transformer)

eval: eval-all ## Alias: Evaluate all models (same as 'eval-all')

#------------------------------------------------------------------------------
# Demo & Verification
#------------------------------------------------------------------------------

demo: ## Launch FastAPI guard demo on localhost:8000
	$(PYTHON) scripts/run_demo.py

verify: ## Verify dataset determinism, split hygiene, and constraints
	$(PYTHON) scripts/verify_dataset.py

#------------------------------------------------------------------------------
# Full Pipeline
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
