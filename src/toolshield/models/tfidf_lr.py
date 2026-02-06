"""TF-IDF + Logistic Regression classifier for prompt injection detection.

This classifier uses:
- TF-IDF vectorization with configurable parameters
- L2-regularized Logistic Regression
- Calibrated probability outputs
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from toolshield.data.schema import DatasetRecord
from toolshield.evaluation.metrics import compute_all_metrics, print_metrics
from toolshield.models.base import BaseClassifier


class TfidfLRClassifier(BaseClassifier):
    """TF-IDF + Logistic Regression classifier.

    Combines TF-IDF text vectorization with L2-regularized logistic
    regression for prompt injection detection.

    Attributes:
        vectorizer: TfidfVectorizer instance.
        classifier: LogisticRegression instance.
        pipeline: Sklearn Pipeline combining vectorizer and classifier.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the TF-IDF + LR classifier.

        Args:
            config: Configuration dictionary with optional keys:
                - max_features: Maximum vocabulary size (default: 10000)
                - ngram_range: Tuple of (min_n, max_n) (default: (1, 3))
                - C: Regularization strength (default: 1.0)
                - max_iter: Maximum iterations (default: 1000)
                - random_state: Random seed (default: 42)
        """
        super().__init__(config)

        # Vectorizer config
        self.max_features = self.config.get("max_features", 10000)
        self.ngram_range = tuple(self.config.get("ngram_range", [1, 3]))

        # Classifier config
        self.C = self.config.get("C", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
        self.random_state = self.config.get("random_state", 42)

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents="unicode",
            stop_words=None,  # Keep all words for injection detection
        )

        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
            class_weight="balanced",  # Handle imbalanced data
        )

        self.pipeline: Pipeline | None = None

    def train(
        self,
        train_records: list[DatasetRecord],
        val_records: list[DatasetRecord] | None = None,
    ) -> dict[str, Any]:
        """Train the classifier on the provided data.

        Args:
            train_records: Training data records.
            val_records: Optional validation data records.

        Returns:
            Dictionary containing training metrics.
        """
        # Extract data
        X_train = self.extract_prompts(train_records)
        y_train = self.extract_labels(train_records)

        # Create and fit pipeline
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier),
        ])

        self.pipeline.fit(X_train, y_train)
        self._is_trained = True

        # Compute training metrics
        train_scores = self.pipeline.predict_proba(X_train)[:, 1]
        train_metrics = compute_all_metrics(y_train, train_scores)

        result = {
            "train_samples": len(train_records),
            "train_roc_auc": train_metrics.roc_auc,
            "train_pr_auc": train_metrics.pr_auc,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
        }

        # Validation metrics if provided
        if val_records:
            X_val = self.extract_prompts(val_records)
            y_val = self.extract_labels(val_records)
            val_goals = self.extract_attack_goals(val_records)

            val_scores = self.pipeline.predict_proba(X_val)[:, 1]
            val_metrics = compute_all_metrics(y_val, val_scores, val_goals)

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
            RuntimeError: If model hasn't been trained.
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained before prediction")

        X = self.extract_prompts(records)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Directory to save model artifacts.

        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained before saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_to_save = {
            "model_type": "tfidf_lr",
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)

        # Save pipeline
        with (path / "pipeline.pkl").open("wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfLRClassifier":
        """Load a model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded TfidfLRClassifier instance.
        """
        path = Path(path)

        # Load config
        with (path / "config.json").open("r") as f:
            config = json.load(f)

        # Remove model_type key if present
        config.pop("model_type", None)

        # Create instance
        instance = cls(config=config)

        # Load pipeline
        with (path / "pipeline.pkl").open("rb") as f:
            instance.pipeline = pickle.load(f)

        instance._is_trained = True
        return instance

    def get_feature_importance(self, top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
        """Get most important features for each class.

        Args:
            top_k: Number of top features to return per class.

        Returns:
            Dictionary with 'benign' and 'attack' keys mapping to
            lists of (feature, importance) tuples.

        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained first")

        # Get feature names and coefficients
        feature_names = self.pipeline.named_steps["vectorizer"].get_feature_names_out()
        coefficients = self.pipeline.named_steps["classifier"].coef_[0]

        # Sort by coefficient (positive = attack, negative = benign)
        sorted_indices = np.argsort(coefficients)

        # Top features for benign (most negative coefficients)
        benign_indices = sorted_indices[:top_k]
        benign_features = [
            (feature_names[i], float(coefficients[i]))
            for i in benign_indices
        ]

        # Top features for attack (most positive coefficients)
        attack_indices = sorted_indices[-top_k:][::-1]
        attack_features = [
            (feature_names[i], float(coefficients[i]))
            for i in attack_indices
        ]

        return {
            "benign": benign_features,
            "attack": attack_features,
        }
