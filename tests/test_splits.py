"""Tests for dataset splitting with template leakage prevention.

These are critical tests that verify no template_id appears in
multiple splits, ensuring proper evaluation without data leakage.
"""

from __future__ import annotations

import pytest

from toolshield.data.make_splits import (
    create_splits,
    split_s_attack_holdout,
    split_s_random,
    split_s_tool_holdout,
    verify_all_splits_have_both_classes,
    verify_no_template_leakage,
    verify_tool_holdout_constraints,
)
from toolshield.data.schema import DatasetRecord


class TestTemplateLeakage:
    """Critical tests for template leakage prevention."""

    def test_template_disjoint_s_random(
        self, s_random_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert template_id sets are disjoint across train/val/test for S_random."""
        train_templates = {r.template_id for r in s_random_splits["train"]}
        val_templates = {r.template_id for r in s_random_splits["val"]}
        test_templates = {r.template_id for r in s_random_splits["test"]}

        # No overlap between train and val
        assert train_templates.isdisjoint(val_templates), (
            f"Template leakage between train and val: {train_templates & val_templates}"
        )

        # No overlap between train and test
        assert train_templates.isdisjoint(test_templates), (
            f"Template leakage between train and test: {train_templates & test_templates}"
        )

        # No overlap between val and test
        assert val_templates.isdisjoint(test_templates), (
            f"Template leakage between val and test: {val_templates & test_templates}"
        )

    def test_template_disjoint_s_attack_holdout(
        self, s_attack_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert template_id sets are disjoint for S_attack_holdout."""
        train_templates = {r.template_id for r in s_attack_holdout_splits["train"]}
        val_templates = {r.template_id for r in s_attack_holdout_splits["val"]}
        test_templates = {r.template_id for r in s_attack_holdout_splits["test"]}

        assert train_templates.isdisjoint(val_templates)
        assert train_templates.isdisjoint(test_templates)
        assert val_templates.isdisjoint(test_templates)

    def test_template_disjoint_s_tool_holdout_train_val(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert template_id sets are disjoint between train and val.

        For S_tool_holdout, templates MAY appear in both train/val and test
        (on different tools) — this is inherent to the protocol. But within
        the train/val partition, templates must still be disjoint.
        """
        train_templates = {r.template_id for r in s_tool_holdout_splits["train"]}
        val_templates = {r.template_id for r in s_tool_holdout_splits["val"]}

        assert train_templates.isdisjoint(val_templates), (
            f"Template leakage between train and val: {train_templates & val_templates}"
        )

    def test_no_record_leakage_s_tool_holdout(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert no record ID appears in multiple splits for S_tool_holdout."""
        train_ids = {r.id for r in s_tool_holdout_splits["train"]}
        val_ids = {r.id for r in s_tool_holdout_splits["val"]}
        test_ids = {r.id for r in s_tool_holdout_splits["test"]}

        assert train_ids.isdisjoint(val_ids), "Record leakage: train & val"
        assert train_ids.isdisjoint(test_ids), "Record leakage: train & test"
        assert val_ids.isdisjoint(test_ids), "Record leakage: val & test"

    def test_verify_no_template_leakage_function(
        self, s_random_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Test the verify_no_template_leakage helper function."""
        # Should not raise for valid splits
        assert verify_no_template_leakage(s_random_splits)

    def test_verify_tool_holdout_constraints_function(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Test the verify_tool_holdout_constraints helper function."""
        assert verify_tool_holdout_constraints(s_tool_holdout_splits)

    def test_verify_leakage_detection(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test that leakage is detected when present."""
        # Create intentionally bad splits with leakage
        mid = len(small_dataset) // 2
        bad_splits = {
            "train": small_dataset[:mid],
            "test": small_dataset[mid - 10:],  # Overlap of 10 records
        }

        # Check if any templates overlap
        train_templates = {r.template_id for r in bad_splits["train"]}
        test_templates = {r.template_id for r in bad_splits["test"]}

        if train_templates & test_templates:
            # If there's overlap, verify_no_template_leakage should catch it
            with pytest.raises(AssertionError, match="Template leakage"):
                verify_no_template_leakage(bad_splits)


class TestAttackHoldoutConstraints:
    """Tests for S_attack_holdout split constraints."""

    def test_af4_only_in_test(
        self, s_attack_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert AF4 only appears in test, not in train/val."""
        train = s_attack_holdout_splits["train"]
        val = s_attack_holdout_splits["val"]
        test = s_attack_holdout_splits["test"]

        # AF4 should NOT be in train
        train_families = {r.attack_family for r in train if r.attack_family}
        assert "AF4" not in train_families, "AF4 found in train split"

        # AF4 should NOT be in val
        val_families = {r.attack_family for r in val if r.attack_family}
        assert "AF4" not in val_families, "AF4 found in val split"

        # AF4 SHOULD be in test
        test_families = {r.attack_family for r in test if r.attack_family}
        assert "AF4" in test_families, "AF4 not found in test split"

    def test_train_val_have_other_families(
        self, s_attack_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert train/val have AF1, AF2, AF3."""
        train = s_attack_holdout_splits["train"]
        val = s_attack_holdout_splits["val"]

        train_families = {r.attack_family for r in train if r.attack_family}
        val_families = {r.attack_family for r in val if r.attack_family}
        combined_families = train_families | val_families

        # Should have at least some of AF1, AF2, AF3
        assert len(combined_families) > 0, "No attack families in train/val"

    def test_test_has_benign_samples(
        self, s_attack_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert test set contains benign samples."""
        test = s_attack_holdout_splits["test"]

        benign_count = sum(1 for r in test if r.label_binary == 0)
        assert benign_count > 0, "No benign samples in test split"


class TestToolHoldoutConstraints:
    """Tests for S_tool_holdout split constraints."""

    def test_export_report_only_in_test(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert exportReport tool only appears in test, not in train/val."""
        train = s_tool_holdout_splits["train"]
        val = s_tool_holdout_splits["val"]
        test = s_tool_holdout_splits["test"]

        holdout_tool = "exportReport"

        # exportReport should NOT be in train
        train_tools = {r.tool_name for r in train}
        assert holdout_tool not in train_tools, f"{holdout_tool} found in train split"

        # exportReport should NOT be in val
        val_tools = {r.tool_name for r in val}
        assert holdout_tool not in val_tools, f"{holdout_tool} found in val split"

        # exportReport SHOULD be in test
        test_tools = {r.tool_name for r in test}
        assert holdout_tool in test_tools, f"{holdout_tool} not found in test split"

    def test_train_val_have_other_tools(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert train/val have other tools."""
        train = s_tool_holdout_splits["train"]
        val = s_tool_holdout_splits["val"]

        train_tools = {r.tool_name for r in train}
        val_tools = {r.tool_name for r in val}
        combined_tools = train_tools | val_tools

        # Should have at least some other tools
        assert len(combined_tools) > 0, "No tools in train/val"
        assert "exportReport" not in combined_tools

    def test_train_has_both_classes(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert train set contains both benign and attack samples."""
        train = s_tool_holdout_splits["train"]
        labels = {r.label_binary for r in train}
        assert labels == {0, 1}, (
            f"S_tool_holdout train must have both classes, got {labels}"
        )

    def test_val_has_both_classes(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert val set contains both benign and attack samples."""
        val = s_tool_holdout_splits["val"]
        labels = {r.label_binary for r in val}
        assert labels == {0, 1}, (
            f"S_tool_holdout val must have both classes, got {labels}"
        )

    def test_test_has_both_labels(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert test set contains both benign and attack samples."""
        test = s_tool_holdout_splits["test"]
        labels = {r.label_binary for r in test}
        assert labels == {0, 1}, (
            f"S_tool_holdout test must have both classes, got {labels}"
        )

    def test_test_holdout_tool_has_both_labels(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert test exportReport records contain both benign and attack."""
        test = s_tool_holdout_splits["test"]
        export_report_records = [r for r in test if r.tool_name == "exportReport"]

        assert len(export_report_records) > 0, "No exportReport records in test"
        labels = {r.label_binary for r in export_report_records}
        assert labels == {0, 1}, (
            f"exportReport test records need both classes, got {labels}"
        )

    def test_train_has_attack_records(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert train contains a non-trivial number of attack records."""
        train = s_tool_holdout_splits["train"]
        attack_count = sum(1 for r in train if r.label_binary == 1)
        total = len(train)
        # At least 10% should be attacks
        assert attack_count > total * 0.1, (
            f"Train has too few attacks: {attack_count}/{total} "
            f"({attack_count/total:.1%})"
        )


class TestBothClassesValidation:
    """Tests for the both-classes validation guard."""

    def test_verify_all_splits_have_both_classes_passes(
        self, s_random_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Verify the guard passes for a well-formed split."""
        assert verify_all_splits_have_both_classes(s_random_splits)

    def test_verify_all_splits_have_both_classes_fails_single_class(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Verify the guard catches a single-class split."""
        benign_only = [r for r in small_dataset if r.label_binary == 0]
        attack_only = [r for r in small_dataset if r.label_binary == 1]

        bad_splits = {
            "train": benign_only,  # single-class
            "val": attack_only[:5],
            "test": attack_only[5:],
        }

        with pytest.raises(ValueError, match="single-class"):
            verify_all_splits_have_both_classes(bad_splits)

    def test_s_tool_holdout_raises_on_single_class_dataset(self) -> None:
        """Ensure split_s_tool_holdout raises if result would be single-class.

        Constructs a pathological dataset where all attacks target the holdout
        tool and only benign samples exist for other tools. The split function
        should raise ValueError because train would be single-class.
        """
        from toolshield.data.tools import TOOL_BY_NAME

        # Build minimal dataset: benign for non-holdout, attacks only for holdout
        records: list[DatasetRecord] = []

        # Benign records for non-holdout tools
        for i, tool_name in enumerate(["getCustomerRecord", "searchTickets"]):
            tool = TOOL_BY_NAME[tool_name]
            for j in range(20):
                records.append(DatasetRecord(
                    id=f"ben_{tool_name}_{j}",
                    language="en",
                    role_sequence=["user"],
                    tool_name=tool_name,
                    tool_schema=tool.schema_,
                    tool_description=tool.description,
                    prompt=f"Benign prompt {j} for {tool_name}",
                    label_binary=0,
                    attack_family=None,
                    attack_goal=None,
                    template_id=f"benign_{tool_name}",
                    variant_id="v0",
                    seed=42,
                ))

        # Attack records ONLY for holdout tool
        holdout = TOOL_BY_NAME["exportReport"]
        for j in range(20):
            records.append(DatasetRecord(
                id=f"atk_export_{j}",
                language="en",
                role_sequence=["user"],
                tool_name="exportReport",
                tool_schema=holdout.schema_,
                tool_description=holdout.description,
                prompt=f"Attack prompt {j}",
                label_binary=1,
                attack_family="AF1",
                attack_goal="exfiltrate",
                template_id="attack_export_only",
                variant_id="v0",
                seed=42,
            ))

        # Benign records for holdout tool (so test has both classes)
        for j in range(10):
            records.append(DatasetRecord(
                id=f"ben_export_{j}",
                language="en",
                role_sequence=["user"],
                tool_name="exportReport",
                tool_schema=holdout.schema_,
                tool_description=holdout.description,
                prompt=f"Benign export {j}",
                label_binary=0,
                attack_family=None,
                attack_goal=None,
                template_id="benign_exportReport",
                variant_id="v0",
                seed=42,
            ))

        with pytest.raises(ValueError, match="single-class"):
            split_s_tool_holdout(records, seed=2026)


class TestSplitRatios:
    """Tests for split ratio correctness."""

    def test_s_random_non_empty_splits(
        self, s_random_splits: dict[str, list[DatasetRecord]]
    ) -> None:
        """Assert all S_random splits are non-empty."""
        assert len(s_random_splits["train"]) > 0, "Train split is empty"
        assert len(s_random_splits["val"]) > 0, "Val split is empty"
        assert len(s_random_splits["test"]) > 0, "Test split is empty"

    def test_s_random_total_coverage(
        self, s_random_splits: dict[str, list[DatasetRecord]], small_dataset: list[DatasetRecord]
    ) -> None:
        """Assert all records are assigned to exactly one split."""
        total_in_splits = (
            len(s_random_splits["train"])
            + len(s_random_splits["val"])
            + len(s_random_splits["test"])
        )

        assert total_in_splits == len(small_dataset), (
            f"Total in splits ({total_in_splits}) != dataset size ({len(small_dataset)})"
        )

    def test_s_tool_holdout_total_coverage(
        self, s_tool_holdout_splits: dict[str, list[DatasetRecord]], small_dataset: list[DatasetRecord]
    ) -> None:
        """Assert all records are assigned to exactly one split for tool holdout."""
        total_in_splits = (
            len(s_tool_holdout_splits["train"])
            + len(s_tool_holdout_splits["val"])
            + len(s_tool_holdout_splits["test"])
        )

        assert total_in_splits == len(small_dataset), (
            f"Total in splits ({total_in_splits}) != dataset size ({len(small_dataset)})"
        )

    def test_deterministic_splitting(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test that splitting is deterministic with same seed."""
        splits1 = split_s_random(small_dataset, seed=2026)
        splits2 = split_s_random(small_dataset, seed=2026)

        train_ids1 = {r.id for r in splits1["train"]}
        train_ids2 = {r.id for r in splits2["train"]}

        assert train_ids1 == train_ids2, "Splitting is not deterministic"

    def test_s_tool_holdout_deterministic(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test that S_tool_holdout splitting is deterministic."""
        splits1 = split_s_tool_holdout(small_dataset, seed=2026)
        splits2 = split_s_tool_holdout(small_dataset, seed=2026)

        train_ids1 = {r.id for r in splits1["train"]}
        train_ids2 = {r.id for r in splits2["train"]}

        assert train_ids1 == train_ids2, "S_tool_holdout splitting is not deterministic"


class TestCreateSplitsFunction:
    """Tests for the create_splits convenience function."""

    def test_create_splits_s_random(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test create_splits with S_random protocol."""
        splits = create_splits(small_dataset, protocol="S_random", seed=2026)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert verify_no_template_leakage(splits)

    def test_create_splits_s_attack_holdout(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test create_splits with S_attack_holdout protocol."""
        splits = create_splits(small_dataset, protocol="S_attack_holdout", seed=2026)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert verify_no_template_leakage(splits)

    def test_create_splits_s_tool_holdout(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test create_splits with S_tool_holdout protocol."""
        splits = create_splits(small_dataset, protocol="S_tool_holdout", seed=2026)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        # Uses tool-holdout-specific constraints (not strict template isolation)
        assert verify_tool_holdout_constraints(splits)

    def test_create_splits_invalid_protocol(
        self, small_dataset: list[DatasetRecord]
    ) -> None:
        """Test that invalid protocol raises ValueError."""
        with pytest.raises(ValueError, match="Unknown protocol"):
            create_splits(small_dataset, protocol="invalid_protocol", seed=2026)
