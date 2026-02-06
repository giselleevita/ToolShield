"""Pytest fixtures for ToolShield tests."""

from __future__ import annotations

import pytest

from toolshield.data.generate_dataset import generate_dataset
from toolshield.data.make_splits import (
    create_splits,
    split_s_attack_holdout,
    split_s_random,
    split_s_tool_holdout,
)
from toolshield.data.schema import DatasetRecord


@pytest.fixture(scope="module")
def small_dataset() -> list[DatasetRecord]:
    """Generate a small dataset for testing.

    Uses a fixed seed for reproducibility.
    Returns 200 samples (100 benign, 100 attack).
    """
    records, _ = generate_dataset(seed=1337, n_samples=200)
    return records


@pytest.fixture(scope="module")
def medium_dataset() -> list[DatasetRecord]:
    """Generate a medium-sized dataset for testing.

    Uses a fixed seed for reproducibility.
    Returns 500 samples.
    """
    records, _ = generate_dataset(seed=1337, n_samples=500)
    return records


@pytest.fixture(scope="module")
def s_random_splits(small_dataset: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Create S_random splits from the small dataset."""
    return split_s_random(small_dataset, seed=2026)


@pytest.fixture(scope="module")
def s_attack_holdout_splits(small_dataset: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Create S_attack_holdout splits from the small dataset."""
    return split_s_attack_holdout(small_dataset, seed=2026)


@pytest.fixture(scope="module")
def s_tool_holdout_splits(small_dataset: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Create S_tool_holdout splits from the small dataset."""
    return split_s_tool_holdout(small_dataset, seed=2026)
