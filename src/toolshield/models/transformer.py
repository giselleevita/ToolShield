"""Transformer text-only classifier for prompt injection detection.

Uses HuggingFace transformers (DistilRoBERTa or DistilBERT) with
a classification head for binary prompt injection detection.

Input: prompt text only (no context).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

from toolshield.data.schema import DatasetRecord
from toolshield.evaluation.metrics import compute_all_metrics, print_metrics
from toolshield.models.base import BaseClassifier


class PromptDataset(TorchDataset):
    """PyTorch Dataset for prompt classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        """Initialize the dataset.

        Args:
            texts: List of text strings.
            labels: Optional list of labels (None for inference).
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def compute_metrics_for_trainer(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction from Trainer.

    Returns:
        Dictionary of metric names to values.
    """
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    scores = probs[:, 1]  # Probability of attack class

    metrics = compute_all_metrics(labels, scores)

    return {
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "fpr_at_tpr_90": metrics.fpr_at_tpr_90,
        "fpr_at_tpr_95": metrics.fpr_at_tpr_95,
    }


class TransformerClassifier(BaseClassifier):
    """Transformer-based classifier using HuggingFace models.

    Uses DistilRoBERTa or DistilBERT as the base model with a
    classification head for binary prompt injection detection.

    Input: prompt text only (no context augmentation).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the transformer classifier.

        Args:
            config: Configuration dictionary with optional keys:
                - model_name: HuggingFace model name (default: "distilroberta-base")
                - max_length: Maximum sequence length (default: 512)
                - batch_size: Training batch size (default: 16)
                - learning_rate: Learning rate (default: 2e-5)
                - num_epochs: Number of training epochs (default: 3)
                - warmup_steps: Warmup steps (default: 100)
                - weight_decay: Weight decay (default: 0.01)
                - seed: Random seed (default: 42)
        """
        super().__init__(config)

        self.model_name = self.config.get("model_name", "distilroberta-base")
        self.max_length = self.config.get("max_length", 512)
        self.batch_size = self.config.get("batch_size", 16)
        self.learning_rate = self.config.get("learning_rate", 2e-5)
        self.num_epochs = self.config.get("num_epochs", 3)
        self.warmup_steps = self.config.get("warmup_steps", 100)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.seed = self.config.get("seed", 42)

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Will be initialized during training or loading
        self.tokenizer: Any = None
        self.model: Any = None

    def _get_texts(self, records: list[DatasetRecord]) -> list[str]:
        """Extract text inputs from records.

        Override in subclasses to change input format.

        Args:
            records: Dataset records.

        Returns:
            List of text strings.
        """
        return self.extract_prompts(records)

    def train(
        self,
        train_records: list[DatasetRecord],
        val_records: list[DatasetRecord] | None = None,
    ) -> dict[str, Any]:
        """Train the transformer classifier.

        Args:
            train_records: Training data records.
            val_records: Optional validation data records.

        Returns:
            Dictionary containing training metrics.
        """
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification",
        )

        # Prepare datasets
        train_texts = self._get_texts(train_records)
        train_labels = self.extract_labels(train_records).tolist()
        train_dataset = PromptDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )

        eval_dataset = None
        if val_records:
            val_texts = self._get_texts(val_records)
            val_labels = self.extract_labels(val_records).tolist()
            eval_dataset = PromptDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )

        # Training arguments
        # Use a temporary directory for training outputs
        output_dir = Path("./transformer_training_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="roc_auc" if eval_dataset else None,
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            seed=self.seed,
            report_to="none",  # Disable wandb/tensorboard
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_for_trainer if eval_dataset else None,
        )

        # Train
        train_result = trainer.train()
        self._is_trained = True

        result = {
            "train_samples": len(train_records),
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
        }

        # Validation metrics
        if val_records:
            val_scores = self.predict_scores(val_records)
            val_labels_arr = self.extract_labels(val_records)
            val_goals = self.extract_attack_goals(val_records)

            val_metrics = compute_all_metrics(val_labels_arr, val_scores, val_goals)
            result.update({
                "val_samples": len(val_records),
                "val_roc_auc": val_metrics.roc_auc,
                "val_pr_auc": val_metrics.pr_auc,
                "val_fpr_at_tpr_90": val_metrics.fpr_at_tpr_90,
                "val_fpr_at_tpr_95": val_metrics.fpr_at_tpr_95,
            })

            print("\nValidation Metrics:")
            print_metrics(val_metrics)

        return result

    def predict_scores(
        self,
        records: list[DatasetRecord],
    ) -> np.ndarray:
        """Predict attack scores for the provided records.

        Args:
            records: Records to predict on.

        Returns:
            Array of scores in [0, 1] where higher = more likely attack.

        Raises:
            RuntimeError: If model hasn't been trained/loaded.
        """
        if not self._is_trained or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be trained or loaded before prediction")

        texts = self._get_texts(records)
        dataset = PromptDataset(texts, None, self.tokenizer, self.max_length)

        # Set model to eval mode
        self.model.eval()
        device = next(self.model.parameters()).device

        all_scores = []
        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                input_ids = item["input_ids"].unsqueeze(0).to(device)
                attention_mask = item["attention_mask"].unsqueeze(0).to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                score = probs[0, 1].item()  # Probability of attack class
                all_scores.append(score)

        return np.array(all_scores)

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Directory to save model artifacts.

        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if not self._is_trained or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be trained before saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_to_save = {
            "model_type": "transformer",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)

        # Save model and tokenizer
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")

    @classmethod
    def load(cls, path: str | Path) -> "TransformerClassifier":
        """Load a model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded TransformerClassifier instance.
        """
        path = Path(path)

        # Load config
        with (path / "config.json").open("r") as f:
            config = json.load(f)

        # Remove model_type key if present
        config.pop("model_type", None)

        # Create instance
        instance = cls(config=config)

        # Load model and tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(path / "model")

        instance._is_trained = True
        return instance
