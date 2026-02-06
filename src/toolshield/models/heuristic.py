"""Heuristic baseline classifier using keyword rules and patterns.

This classifier implements a rule-based approach to prompt injection
detection using:
- Keyword matching for known injection patterns
- Pattern matching for structural indicators
- Allowlist for known-safe patterns

Outputs binary scores {0, 1}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from toolshield.data.schema import DatasetRecord
from toolshield.models.base import BaseClassifier


# Default suspicious keywords indicating potential injection
DEFAULT_SUSPICIOUS_KEYWORDS = [
    # Instruction override patterns
    "ignore",
    "ignore previous",
    "ignore all",
    "disregard",
    "forget",
    "override",
    "bypass",
    "skip",
    "instead",
    "actually",
    "new instructions",
    "new directive",
    "system override",
    "admin mode",
    "maintenance mode",
    "debug mode",
    "elevated privileges",
    # Jailbreak patterns
    "dan",
    "do anything now",
    "unrestricted",
    "no restrictions",
    "pretend",
    "roleplay",
    "hypothetical",
    "for testing",
    # Exfiltration patterns
    "send to",
    "send a copy",
    "forward to",
    "email to",
    "post to",
    "webhook",
    "external",
    "mirror",
    "transmit",
    # Tool hijacking patterns
    "instead of",
    "change to",
    "switch to",
    "redirect to",
    "call instead",
    "reset mfa",
    "resetusermfa",
    # Indirect injection markers
    "system note",
    "internal",
    "admin flag",
    "escalation",
    "auto-approve",
    "hidden",
]

# Regex patterns for structural indicators
DEFAULT_SUSPICIOUS_PATTERNS = [
    r"\[\[.*ADMIN.*\]\]",  # [[ADMIN: ...]]
    r"\{\{.*SYSTEM.*\}\}",  # {{SYSTEM: ...}}
    r"<<<.*>>>",  # <<<...>>>
    r"---END.*---",  # ---END OF SYSTEM PROMPT---
    r"BEGIN.*COMMAND",  # BEGIN SYSTEM COMMAND
    r"\[INTERNAL\]",  # [INTERNAL]
    r"\[PRIORITY\s*OVERRIDE\]",
    r"<!--.*-->",  # HTML comments with instructions
    r"auth_level\s*=\s*admin",
    r"restrictions\s*=\s*none",
]

# Allowlist patterns that indicate benign requests
DEFAULT_ALLOWLIST_PATTERNS = [
    r"^please\s+(look\s+up|retrieve|fetch|get|find|search|pull)",
    r"^i\s+need\s+(to\s+)?(retrieve|find|get|search|look\s+up)",
    r"^can\s+you\s+(please\s+)?(get|find|retrieve|search|look\s+up|pull)",
    r"ticket\s*#?\s*\w+-\d+",  # Ticket references like TKT-12345
    r"customer\s*(id|ID)\s*:?\s*\w+-?\d+",  # Customer ID references
    r"standard\s+(request|operation|procedure)",
    r"following\s+up",
    r"routine\s+request",
]


class HeuristicClassifier(BaseClassifier):
    """Rule-based heuristic classifier for prompt injection detection.

    Uses keyword matching, pattern detection, and allowlisting to
    classify prompts. Outputs binary scores {0, 1}.

    Attributes:
        suspicious_keywords: List of keywords indicating potential injection.
        suspicious_patterns: List of regex patterns for structural indicators.
        allowlist_patterns: List of regex patterns for known-safe requests.
        case_sensitive: Whether keyword matching is case-sensitive.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the heuristic classifier.

        Args:
            config: Configuration dictionary with optional keys:
                - suspicious_keywords: Custom keyword list
                - suspicious_patterns: Custom pattern list
                - allowlist_patterns: Custom allowlist patterns
                - case_sensitive: bool (default: False)
        """
        super().__init__(config)

        self.suspicious_keywords = self.config.get(
            "suspicious_keywords", DEFAULT_SUSPICIOUS_KEYWORDS
        )
        self.suspicious_patterns = self.config.get(
            "suspicious_patterns", DEFAULT_SUSPICIOUS_PATTERNS
        )
        self.allowlist_patterns = self.config.get(
            "allowlist_patterns", DEFAULT_ALLOWLIST_PATTERNS
        )
        self.case_sensitive = self.config.get("case_sensitive", False)

        # Compile patterns
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled_suspicious = [
            re.compile(p, flags) for p in self.suspicious_patterns
        ]
        self._compiled_allowlist = [
            re.compile(p, flags) for p in self.allowlist_patterns
        ]

        # Pre-process keywords
        if not self.case_sensitive:
            self.suspicious_keywords = [k.lower() for k in self.suspicious_keywords]

        self._is_trained = True  # No training needed

    def train(
        self,
        train_records: list[DatasetRecord],
        val_records: list[DatasetRecord] | None = None,
    ) -> dict[str, Any]:
        """Heuristic classifier doesn't require training.

        Args:
            train_records: Training data (unused).
            val_records: Validation data (unused).

        Returns:
            Empty training info dict.
        """
        self._is_trained = True
        return {"note": "Heuristic classifier requires no training"}

    def _check_keywords(self, text: str) -> tuple[bool, list[str]]:
        """Check for suspicious keywords in text.

        Args:
            text: Text to check.

        Returns:
            Tuple of (has_suspicious, matched_keywords).
        """
        check_text = text if self.case_sensitive else text.lower()
        matched = []

        for keyword in self.suspicious_keywords:
            if keyword in check_text:
                matched.append(keyword)

        return len(matched) > 0, matched

    def _check_patterns(self, text: str) -> tuple[bool, list[str]]:
        """Check for suspicious patterns in text.

        Args:
            text: Text to check.

        Returns:
            Tuple of (has_suspicious, matched_patterns).
        """
        matched = []

        for i, pattern in enumerate(self._compiled_suspicious):
            if pattern.search(text):
                matched.append(self.suspicious_patterns[i])

        return len(matched) > 0, matched

    def _check_allowlist(self, text: str) -> bool:
        """Check if text matches allowlist patterns.

        Args:
            text: Text to check.

        Returns:
            True if text matches any allowlist pattern.
        """
        for pattern in self._compiled_allowlist:
            if pattern.search(text):
                return True
        return False

    def _score_single(self, prompt: str) -> tuple[float, dict[str, Any]]:
        """Score a single prompt.

        Args:
            prompt: Prompt text.

        Returns:
            Tuple of (score, details) where score is 0 or 1.
        """
        details: dict[str, Any] = {
            "keywords_found": [],
            "patterns_found": [],
            "allowlisted": False,
        }

        # Check allowlist first
        if self._check_allowlist(prompt):
            details["allowlisted"] = True
            # Allowlisted prompts get reduced but not zero score
            # if they also have suspicious indicators

        # Check keywords
        has_keywords, matched_keywords = self._check_keywords(prompt)
        details["keywords_found"] = matched_keywords

        # Check patterns
        has_patterns, matched_patterns = self._check_patterns(prompt)
        details["patterns_found"] = matched_patterns

        # Scoring logic
        if details["allowlisted"] and not has_patterns:
            # Allowlisted and no structural patterns = benign
            return 0.0, details
        elif has_patterns:
            # Structural patterns are strong indicators
            return 1.0, details
        elif has_keywords and len(matched_keywords) >= 2:
            # Multiple keyword matches = suspicious
            return 1.0, details
        elif has_keywords:
            # Single keyword match - could be false positive
            # Be conservative and flag it
            return 1.0, details
        else:
            return 0.0, details

    def predict_scores(
        self,
        records: list[DatasetRecord],
    ) -> np.ndarray:
        """Predict attack scores for the provided records.

        Args:
            records: Records to predict on.

        Returns:
            Array of binary scores (0 or 1).
        """
        prompts = self.extract_prompts(records)
        scores = np.array([self._score_single(p)[0] for p in prompts])
        return scores

    def predict_with_details(
        self,
        records: list[DatasetRecord],
    ) -> list[tuple[float, dict[str, Any]]]:
        """Predict with detailed explanations.

        Args:
            records: Records to predict on.

        Returns:
            List of (score, details) tuples.
        """
        prompts = self.extract_prompts(records)
        return [self._score_single(p) for p in prompts]

    def save(self, path: str | Path) -> None:
        """Save the classifier configuration.

        Args:
            path: Directory to save configuration.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_to_save = {
            "model_type": "heuristic",
            "suspicious_keywords": self.suspicious_keywords,
            "suspicious_patterns": self.suspicious_patterns,
            "allowlist_patterns": self.allowlist_patterns,
            "case_sensitive": self.case_sensitive,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HeuristicClassifier":
        """Load a classifier from disk.

        Args:
            path: Directory containing configuration.

        Returns:
            Loaded HeuristicClassifier instance.
        """
        path = Path(path)

        with (path / "config.json").open("r") as f:
            config = json.load(f)

        # Remove model_type key if present
        config.pop("model_type", None)

        return cls(config=config)
