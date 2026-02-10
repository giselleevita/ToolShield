"""Scored heuristic classifier for prompt injection detection.

Unlike the binary heuristic, this classifier outputs continuous scores in [0,1]
which enables meaningful threshold selection under FPR budgets.

The scoring formula:
    raw = w_kw * keyword_hits + w_pat * pattern_hits - w_allow * allowlist_hits
    score = sigmoid(raw / temperature)

This produces a probability-like score suitable for ROC analysis and budget tuning.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from toolshield.data.schema import DatasetRecord
from toolshield.models.base import BaseClassifier


# Default suspicious keywords (holdout-safe, no tool names)
DEFAULT_SUSPICIOUS_KEYWORDS = [
    # Instruction override patterns
    "ignore previous",
    "ignore all",
    "disregard",
    "forget",
    "override",
    "bypass",
    "skip",
    "new instructions",
    "new directive",
    "system override",
    "admin mode",
    "maintenance mode",
    "debug mode",
    "elevated privileges",
    # Jailbreak patterns
    "do anything now",
    "unrestricted",
    "no restrictions",
    "pretend",
    "roleplay",
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
    # Tool hijacking patterns (generic, no tool names)
    "redirect to",
    "call instead",
    # Indirect injection markers
    "auto-approve",
    "hidden",
    "internal instruction",
    "system note",
]

# Regex patterns for structural indicators
DEFAULT_SUSPICIOUS_PATTERNS = [
    r"\[\[.*ADMIN.*\]\]",
    r"\{\{.*SYSTEM.*\}\}",
    r"<<<.*>>>",
    r"---END.*---",
    r"BEGIN.*COMMAND",
    r"\[INTERNAL\]",
    r"\[PRIORITY\s*OVERRIDE\]",
    r"<!--.*-->",
    r"auth_level\s*=\s*admin",
    r"restrictions\s*=\s*none",
]

# Allowlist patterns (narrow, safe)
DEFAULT_ALLOWLIST_PATTERNS = [
    r"ticket\s*#?\s*\w+-\d+",
    r"customer\s*(id|ID)\s*:?\s*\w+-?\d+",
    r"\b(explain|definition|what\s+is|how\s+does|mitigation|defense|research|paper)\b.*\b(prompt\s+injection|jailbreak|llm\s+security)\b",
]

# Default scoring weights
DEFAULT_WEIGHTS = {
    "w_keyword": 1.0,
    "w_pattern": 2.0,
    "w_allowlist": 1.0,
}

DEFAULT_TEMPERATURE = 3.0


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


class ScoredHeuristicClassifier(BaseClassifier):
    """Scored heuristic classifier for prompt injection detection.
    
    Unlike the binary HeuristicClassifier, this outputs continuous scores
    in [0,1] based on weighted feature counts passed through a sigmoid.
    
    This enables meaningful threshold selection under FPR budgets.
    
    Attributes:
        suspicious_keywords: List of keywords indicating potential injection.
        suspicious_patterns: List of regex patterns for structural indicators.
        allowlist_patterns: List of regex patterns for known-safe requests.
        weights: Dictionary with w_keyword, w_pattern, w_allowlist.
        temperature: Sigmoid temperature for score calibration.
        case_sensitive: Whether keyword matching is case-sensitive.
    """
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the scored heuristic classifier.
        
        Args:
            config: Configuration dictionary with optional keys:
                - suspicious_keywords: Custom keyword list
                - suspicious_patterns: Custom pattern list
                - allowlist_patterns: Custom allowlist patterns
                - weights: Dict with w_keyword, w_pattern, w_allowlist
                - temperature: Sigmoid temperature (default: 3.0)
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
        
        # Scoring weights
        weights = self.config.get("weights", DEFAULT_WEIGHTS)
        self.w_keyword = weights.get("w_keyword", DEFAULT_WEIGHTS["w_keyword"])
        self.w_pattern = weights.get("w_pattern", DEFAULT_WEIGHTS["w_pattern"])
        self.w_allowlist = weights.get("w_allowlist", DEFAULT_WEIGHTS["w_allowlist"])
        
        self.temperature = self.config.get("temperature", DEFAULT_TEMPERATURE)
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
        """Scored heuristic doesn't require training.
        
        Args:
            train_records: Training data (unused).
            val_records: Validation data (unused).
            
        Returns:
            Empty training info dict.
        """
        self._is_trained = True
        return {"note": "Scored heuristic classifier requires no training"}
    
    def _count_keyword_hits(self, text: str) -> tuple[int, list[str]]:
        """Count distinct keyword matches.
        
        Args:
            text: Text to check.
            
        Returns:
            Tuple of (hit_count, matched_keywords).
        """
        check_text = text if self.case_sensitive else text.lower()
        matched = []
        
        for keyword in self.suspicious_keywords:
            if keyword in check_text:
                matched.append(keyword)
        
        return len(matched), matched
    
    def _count_pattern_hits(self, text: str) -> tuple[int, list[str]]:
        """Count distinct pattern matches.
        
        Args:
            text: Text to check.
            
        Returns:
            Tuple of (hit_count, matched_patterns).
        """
        matched = []
        
        for i, pattern in enumerate(self._compiled_suspicious):
            if pattern.search(text):
                matched.append(self.suspicious_patterns[i])
        
        return len(matched), matched
    
    def _count_allowlist_hits(self, text: str) -> tuple[int, list[str]]:
        """Count distinct allowlist matches.
        
        Args:
            text: Text to check.
            
        Returns:
            Tuple of (hit_count, matched_patterns).
        """
        matched = []
        
        for i, pattern in enumerate(self._compiled_allowlist):
            if pattern.search(text):
                matched.append(self.allowlist_patterns[i])
        
        return len(matched), matched
    
    def _score_single(self, prompt: str) -> tuple[float, dict[str, Any]]:
        """Score a single prompt.
        
        Args:
            prompt: Prompt text.
            
        Returns:
            Tuple of (score, details) where score is in [0, 1].
        """
        # Count hits
        kw_count, kw_matched = self._count_keyword_hits(prompt)
        pat_count, pat_matched = self._count_pattern_hits(prompt)
        allow_count, allow_matched = self._count_allowlist_hits(prompt)
        
        # Compute raw score
        raw = (
            self.w_keyword * kw_count
            + self.w_pattern * pat_count
            - self.w_allowlist * allow_count
        )
        
        # Apply sigmoid with temperature
        score = sigmoid(raw / self.temperature)
        
        # Build explanation
        details = {
            "keyword_hits": kw_count,
            "keywords_matched": kw_matched,
            "pattern_hits": pat_count,
            "patterns_matched": pat_matched,
            "allowlist_hits": allow_count,
            "allowlist_matched": allow_matched,
            "raw_score": raw,
            "temperature": self.temperature,
            "score": score,
        }
        
        return score, details
    
    def predict_scores(
        self,
        records: list[DatasetRecord],
    ) -> np.ndarray:
        """Predict attack scores for the provided records.
        
        Args:
            records: Records to predict on.
            
        Returns:
            Array of scores in [0, 1].
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
            "model_type": "heuristic_score",
            "suspicious_keywords": self.suspicious_keywords,
            "suspicious_patterns": self.suspicious_patterns,
            "allowlist_patterns": self.allowlist_patterns,
            "weights": {
                "w_keyword": self.w_keyword,
                "w_pattern": self.w_pattern,
                "w_allowlist": self.w_allowlist,
            },
            "temperature": self.temperature,
            "case_sensitive": self.case_sensitive,
        }
        
        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "ScoredHeuristicClassifier":
        """Load a classifier from disk.
        
        Args:
            path: Directory containing configuration.
            
        Returns:
            Loaded ScoredHeuristicClassifier instance.
        """
        path = Path(path)
        
        with (path / "config.json").open("r") as f:
            config = json.load(f)
        
        # Remove model_type key if present
        config.pop("model_type", None)
        
        return cls(config=config)
