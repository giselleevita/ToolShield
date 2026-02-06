"""Base classifier interface for prompt injection detection.

All classifiers should inherit from BaseClassifier and implement
the required methods for training, prediction, and serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from toolshield.data.schema import DatasetRecord


class BaseClassifier(ABC):
    """Abstract base class for prompt injection classifiers.

    All classifier implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the classifier.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    @abstractmethod
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
            Dictionary containing training metrics/info.
        """
        pass

    @abstractmethod
    def predict_scores(
        self,
        records: list[DatasetRecord],
    ) -> np.ndarray:
        """Predict attack scores for the provided records.

        Args:
            records: Records to predict on.

        Returns:
            Array of scores in [0, 1] where higher = more likely attack.
        """
        pass

    def predict(
        self,
        records: list[DatasetRecord],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels for the provided records.

        Args:
            records: Records to predict on.
            threshold: Classification threshold.

        Returns:
            Array of binary predictions (0=benign, 1=attack).
        """
        scores = self.predict_scores(records)
        return (scores >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Directory to save model artifacts.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseClassifier":
        """Load a model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded classifier instance.
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """Get the model configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()

    @staticmethod
    def extract_prompts(records: list[DatasetRecord]) -> list[str]:
        """Extract prompt texts from records.

        Args:
            records: Dataset records.

        Returns:
            List of prompt strings.
        """
        return [r.prompt for r in records]

    @staticmethod
    def extract_labels(records: list[DatasetRecord]) -> np.ndarray:
        """Extract binary labels from records.

        Args:
            records: Dataset records.

        Returns:
            Array of binary labels.
        """
        return np.array([r.label_binary for r in records])

    @staticmethod
    def extract_attack_goals(records: list[DatasetRecord]) -> list[str | None]:
        """Extract attack goals from records.

        Args:
            records: Dataset records.

        Returns:
            List of attack goals (None for benign).
        """
        return [r.attack_goal for r in records]

    @staticmethod
    def format_context_input(record: DatasetRecord) -> str:
        """Format a record with full context for context-aware models.

        Args:
            record: A dataset record.

        Returns:
            Formatted string with role, tool, schema, description, and prompt.
        """
        import json

        role_str = ", ".join(record.role_sequence)
        schema_str = json.dumps(record.tool_schema, separators=(",", ":"))

        return (
            f"ROLE: {role_str} [SEP] "
            f"TOOL: {record.tool_name} [SEP] "
            f"SCHEMA: {schema_str} [SEP] "
            f"DESC: {record.tool_description} [SEP] "
            f"PROMPT: {record.prompt}"
        )
