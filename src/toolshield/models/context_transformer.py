"""Context-augmented transformer classifier for prompt injection detection.

Extends the base transformer classifier to include contextual information:
- Role sequence
- Tool name
- Tool schema
- Tool description
- Prompt text

Input format:
[CLS] ROLE: {role_sequence} [SEP] TOOL: {tool_name} [SEP]
SCHEMA: {tool_schema} [SEP] DESC: {tool_description} [SEP]
PROMPT: {prompt} [SEP]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolshield.data.schema import DatasetRecord
from toolshield.models.transformer import TransformerClassifier


class ContextTransformerClassifier(TransformerClassifier):
    """Context-augmented transformer classifier.

    Extends TransformerClassifier to include role, tool, schema,
    and description context in the input representation.

    Input format concatenates all context fields with the prompt.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the context-augmented transformer classifier.

        Args:
            config: Configuration dictionary with optional keys:
                All TransformerClassifier config options, plus:
                - include_schema: Whether to include tool schema (default: True)
                - include_description: Whether to include tool description (default: True)
                - max_schema_length: Max characters for schema (default: 200)
        """
        super().__init__(config)

        self.include_schema = self.config.get("include_schema", True)
        self.include_description = self.config.get("include_description", True)
        self.max_schema_length = self.config.get("max_schema_length", 200)

    def _format_record(self, record: DatasetRecord) -> str:
        """Format a record with full context.

        Args:
            record: A dataset record.

        Returns:
            Formatted string with context and prompt.
        """
        parts = []

        # Role sequence
        role_str = ", ".join(record.role_sequence)
        parts.append(f"ROLE: {role_str}")

        # Tool name
        parts.append(f"TOOL: {record.tool_name}")

        # Tool schema (optional, truncated)
        if self.include_schema:
            schema_str = json.dumps(record.tool_schema, separators=(",", ":"))
            if len(schema_str) > self.max_schema_length:
                schema_str = schema_str[: self.max_schema_length] + "..."
            parts.append(f"SCHEMA: {schema_str}")

        # Tool description (optional)
        if self.include_description:
            parts.append(f"DESC: {record.tool_description}")

        # Prompt (always included)
        parts.append(f"PROMPT: {record.prompt}")

        return " [SEP] ".join(parts)

    def _get_texts(self, records: list[DatasetRecord]) -> list[str]:
        """Extract context-augmented text inputs from records.

        Args:
            records: Dataset records.

        Returns:
            List of formatted text strings with context.
        """
        return [self._format_record(r) for r in records]

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

        # Save config (including context-specific options)
        config_to_save = {
            "model_type": "context_transformer",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "include_schema": self.include_schema,
            "include_description": self.include_description,
            "max_schema_length": self.max_schema_length,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)

        # Save model and tokenizer
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")

    @classmethod
    def load(cls, path: str | Path) -> "ContextTransformerClassifier":
        """Load a model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded ContextTransformerClassifier instance.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
