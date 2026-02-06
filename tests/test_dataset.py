"""Tests for dataset generation."""

from __future__ import annotations

import pytest

from toolshield.data.generate_dataset import generate_dataset
from toolshield.data.schema import DatasetRecord
from toolshield.data.tools import TOOL_NAMES


class TestDatasetGeneration:
    """Tests for dataset generation."""

    def test_deterministic_generation(self) -> None:
        """Test that generation is deterministic with same seed."""
        records1, _ = generate_dataset(seed=1337, n_samples=100)
        records2, _ = generate_dataset(seed=1337, n_samples=100)

        assert len(records1) == len(records2)

        for r1, r2 in zip(records1, records2):
            assert r1.id == r2.id
            assert r1.prompt == r2.prompt
            assert r1.label_binary == r2.label_binary
            assert r1.template_id == r2.template_id

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different datasets."""
        records1, _ = generate_dataset(seed=1337, n_samples=100)
        records2, _ = generate_dataset(seed=42, n_samples=100)

        # IDs should be different
        ids1 = {r.id for r in records1}
        ids2 = {r.id for r in records2}
        assert ids1 != ids2

    def test_balanced_labels(self) -> None:
        """Test that dataset has balanced labels (50% benign, 50% attack)."""
        records, manifest = generate_dataset(seed=1337, n_samples=1000)

        benign_count = sum(1 for r in records if r.label_binary == 0)
        attack_count = sum(1 for r in records if r.label_binary == 1)

        assert benign_count == 500
        assert attack_count == 500

    def test_attack_family_distribution(self) -> None:
        """Test that attacks are evenly distributed among families."""
        records, _ = generate_dataset(seed=1337, n_samples=1000)

        attack_records = [r for r in records if r.label_binary == 1]
        family_counts: dict[str | None, int] = {}

        for r in attack_records:
            family_counts[r.attack_family] = family_counts.get(r.attack_family, 0) + 1

        # Should have all 4 attack families
        assert "AF1" in family_counts
        assert "AF2" in family_counts
        assert "AF3" in family_counts
        assert "AF4" in family_counts

        # Each family should have ~125 samples (500 attacks / 4 families)
        for family, count in family_counts.items():
            assert count >= 100, f"{family} has too few samples: {count}"
            assert count <= 150, f"{family} has too many samples: {count}"

    def test_all_tools_represented(self) -> None:
        """Test that all tools are represented in the dataset."""
        records, _ = generate_dataset(seed=1337, n_samples=1000)

        tools_used = {r.tool_name for r in records}

        for tool_name in TOOL_NAMES:
            assert tool_name in tools_used, f"Tool {tool_name} not in dataset"

    def test_record_fields_populated(self) -> None:
        """Test that all required fields are populated."""
        records, _ = generate_dataset(seed=1337, n_samples=100)

        for record in records:
            assert record.id
            assert record.language == "en"
            assert len(record.role_sequence) > 0
            assert record.tool_name in TOOL_NAMES
            assert isinstance(record.tool_schema, dict)
            assert record.tool_description
            assert record.prompt
            assert record.label_binary in [0, 1]
            assert record.template_id
            assert record.variant_id
            assert record.seed == 1337

    def test_benign_records_have_no_attack_info(self) -> None:
        """Test that benign records have no attack family/goal."""
        records, _ = generate_dataset(seed=1337, n_samples=100)

        for record in records:
            if record.label_binary == 0:
                assert record.attack_family is None
                assert record.attack_goal is None

    def test_attack_records_have_attack_info(self) -> None:
        """Test that attack records have attack family and goal."""
        records, _ = generate_dataset(seed=1337, n_samples=100)

        for record in records:
            if record.label_binary == 1:
                assert record.attack_family in ["AF1", "AF2", "AF3", "AF4"]
                assert record.attack_goal is not None

    def test_unique_ids(self) -> None:
        """Test that all record IDs are unique."""
        records, _ = generate_dataset(seed=1337, n_samples=1000)

        ids = [r.id for r in records]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_manifest_accuracy(self) -> None:
        """Test that manifest accurately reflects dataset."""
        records, manifest = generate_dataset(seed=1337, n_samples=1000)

        assert manifest.seed == 1337
        assert manifest.total_records == len(records)

        # Verify label counts
        actual_benign = sum(1 for r in records if r.label_binary == 0)
        actual_attack = sum(1 for r in records if r.label_binary == 1)
        assert int(manifest.label_counts.get("0", 0)) == actual_benign
        assert int(manifest.label_counts.get("1", 0)) == actual_attack
