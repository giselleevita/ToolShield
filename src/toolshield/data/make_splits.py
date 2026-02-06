"""Dataset splitting for prompt injection detection.

This module implements three split protocols:
- S_random: Stratified random split with no template leakage
- S_attack_holdout: Hold out AF4 from training
- S_tool_holdout: Hold out exportReport tool from training

All protocols ensure no template_id appears in multiple splits.

Usage:
    from toolshield.data.make_splits import create_splits
    splits = create_splits(records, protocol="S_random", seed=2026)
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from toolshield.data.schema import DatasetRecord, SplitManifest
from toolshield.utils.io import load_jsonl, save_jsonl, save_manifest


class SplitProtocol(str, Enum):
    """Available split protocols."""

    S_RANDOM = "S_random"
    S_ATTACK_HOLDOUT = "S_attack_holdout"
    S_TOOL_HOLDOUT = "S_tool_holdout"


def _records_to_objects(records: list[dict[str, Any]]) -> list[DatasetRecord]:
    """Convert list of dicts to DatasetRecord objects."""
    return [DatasetRecord(**r) for r in records]


def _group_by_template(records: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Group records by template_id.

    Args:
        records: List of DatasetRecord objects.

    Returns:
        Dictionary mapping template_id to list of records.
    """
    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        groups[record.template_id].append(record)
    return dict(groups)


def _create_split_manifest(
    records: list[DatasetRecord],
    protocol: str,
    seed: int,
    split_name: str,
) -> SplitManifest:
    """Create a manifest for a split.

    Args:
        records: Records in this split.
        protocol: Split protocol name.
        seed: Split seed.
        split_name: Name of the split (train/val/test).

    Returns:
        SplitManifest object.
    """
    label_counts = Counter(r.label_binary for r in records)
    family_counts = Counter(r.attack_family for r in records)
    tool_counts = Counter(r.tool_name for r in records)
    template_ids = sorted(set(r.template_id for r in records))

    return SplitManifest(
        protocol=protocol,
        seed=seed,
        split_name=split_name,
        total_records=len(records),
        label_counts={str(k): v for k, v in sorted(label_counts.items())},
        family_counts={
            str(k) if k else "benign": v
            for k, v in sorted(family_counts.items(), key=lambda x: str(x[0]))
        },
        tool_counts=dict(sorted(tool_counts.items())),
        template_ids=template_ids,
    )


def _stratified_template_split(
    template_groups: dict[str, list[DatasetRecord]],
    rng: random.Random,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """Split template IDs with stratification by label and family.

    Args:
        template_groups: Mapping of template_id to records.
        rng: Seeded random number generator.
        train_ratio: Fraction of templates for training.
        val_ratio: Fraction of templates for validation.

    Returns:
        Tuple of (train_template_ids, val_template_ids, test_template_ids).
    """
    # Group templates by their "stratum" (label + attack_family combination)
    strata: dict[tuple[int, str | None], list[str]] = defaultdict(list)

    for template_id, records in template_groups.items():
        # Use first record to determine stratum (all should be same for a template)
        first = records[0]
        stratum = (first.label_binary, first.attack_family)
        strata[stratum].append(template_id)

    train_templates: list[str] = []
    val_templates: list[str] = []
    test_templates: list[str] = []

    for stratum, templates in strata.items():
        rng.shuffle(templates)
        n = len(templates)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        # Ensure we don't exceed template count
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))

        train_templates.extend(templates[:n_train])
        val_templates.extend(templates[n_train:n_train + n_val])
        test_templates.extend(templates[n_train + n_val:])

    return train_templates, val_templates, test_templates


def split_s_random(
    records: list[DatasetRecord],
    seed: int = 2026,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[DatasetRecord]]:
    """Create S_random split (stratified random, no template leakage).

    Stratifies by label_binary and attack_family while ensuring no
    template_id appears in multiple splits.

    Args:
        records: All dataset records.
        seed: Random seed for splitting.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to record lists.
    """
    rng = random.Random(seed)
    template_groups = _group_by_template(records)

    train_templates, val_templates, test_templates = _stratified_template_split(
        template_groups, rng, train_ratio, val_ratio
    )

    # Collect records for each split
    train_records = [r for tid in train_templates for r in template_groups[tid]]
    val_records = [r for tid in val_templates for r in template_groups[tid]]
    test_records = [r for tid in test_templates for r in template_groups[tid]]

    # Shuffle within each split
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def split_s_attack_holdout(
    records: list[DatasetRecord],
    seed: int = 2026,
    holdout_family: str = "AF4",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[DatasetRecord]]:
    """Create S_attack_holdout split (hold out a specific attack family).

    Train/val contain AF1, AF2, AF3 + benign (no AF4).
    Test contains AF4 + proportional benign subset.
    No template leakage across splits.

    Args:
        records: All dataset records.
        seed: Random seed for splitting.
        holdout_family: Attack family to hold out (default: AF4).
        train_ratio: Fraction for training (of non-holdout data).
        val_ratio: Fraction for validation (of non-holdout data).

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to record lists.
    """
    rng = random.Random(seed)
    template_groups = _group_by_template(records)

    # Separate templates by whether they contain holdout family
    holdout_templates: list[str] = []
    non_holdout_templates: list[str] = []

    for template_id, template_records in template_groups.items():
        if any(r.attack_family == holdout_family for r in template_records):
            holdout_templates.append(template_id)
        else:
            non_holdout_templates.append(template_id)

    # Split non-holdout templates into train/val, with some benign for test
    non_holdout_groups = {tid: template_groups[tid] for tid in non_holdout_templates}

    # Separate benign and attack templates
    benign_templates = [tid for tid in non_holdout_templates
                        if template_groups[tid][0].label_binary == 0]
    attack_templates = [tid for tid in non_holdout_templates
                        if template_groups[tid][0].label_binary == 1]

    # Shuffle
    rng.shuffle(benign_templates)
    rng.shuffle(attack_templates)

    # For benign: put some in test to accompany holdout attack
    n_benign_test = max(1, len(benign_templates) // 5)  # ~20% of benign to test

    # Split benign templates
    benign_test = benign_templates[:n_benign_test]
    benign_trainval = benign_templates[n_benign_test:]

    # Split trainval benign
    n_benign_train = int(len(benign_trainval) * train_ratio / (train_ratio + val_ratio))
    benign_train = benign_trainval[:n_benign_train]
    benign_val = benign_trainval[n_benign_train:]

    # Split attack templates (all non-holdout attacks go to train/val)
    n_attack_train = int(len(attack_templates) * train_ratio / (train_ratio + val_ratio))
    attack_train = attack_templates[:n_attack_train]
    attack_val = attack_templates[n_attack_train:]

    # Collect records
    train_templates = benign_train + attack_train
    val_templates = benign_val + attack_val
    test_templates = benign_test + holdout_templates

    train_records = [r for tid in train_templates for r in template_groups[tid]]
    val_records = [r for tid in val_templates for r in template_groups[tid]]
    test_records = [r for tid in test_templates for r in template_groups[tid]]

    # Shuffle within each split
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def split_s_tool_holdout(
    records: list[DatasetRecord],
    seed: int = 2026,
    holdout_tool: str = "exportReport",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[DatasetRecord]]:
    """Create S_tool_holdout split (hold out a specific tool).

    Train/val contain all tools except the holdout tool.
    Test contains all samples for the holdout tool (benign + attack).
    No template leakage across splits.

    Args:
        records: All dataset records.
        seed: Random seed for splitting.
        holdout_tool: Tool to hold out (default: exportReport).
        train_ratio: Fraction for training (of non-holdout data).
        val_ratio: Fraction for validation (of non-holdout data).

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to record lists.
    """
    rng = random.Random(seed)
    template_groups = _group_by_template(records)

    # Separate templates by whether they contain holdout tool
    holdout_templates: list[str] = []
    non_holdout_templates: list[str] = []

    for template_id, template_records in template_groups.items():
        if any(r.tool_name == holdout_tool for r in template_records):
            holdout_templates.append(template_id)
        else:
            non_holdout_templates.append(template_id)

    # Split non-holdout templates
    non_holdout_groups = {tid: template_groups[tid] for tid in non_holdout_templates}

    train_templates, val_templates, _ = _stratified_template_split(
        non_holdout_groups, rng, train_ratio, val_ratio
    )

    # All holdout tool samples go to test
    # Any templates not assigned yet (from non-holdout) also go to test
    assigned = set(train_templates) | set(val_templates)
    remaining_non_holdout = [tid for tid in non_holdout_templates if tid not in assigned]
    test_templates = holdout_templates + remaining_non_holdout

    # Collect records
    train_records = [r for tid in train_templates for r in template_groups[tid]]
    val_records = [r for tid in val_templates for r in template_groups[tid]]
    test_records = [r for tid in test_templates for r in template_groups[tid]]

    # Shuffle within each split
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def create_splits(
    records: list[DatasetRecord] | list[dict[str, Any]],
    protocol: str | SplitProtocol,
    seed: int = 2026,
) -> dict[str, list[DatasetRecord]]:
    """Create train/val/test splits using the specified protocol.

    Args:
        records: Dataset records (DatasetRecord objects or dicts).
        protocol: Split protocol name.
        seed: Random seed for splitting.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to record lists.

    Raises:
        ValueError: If unknown protocol is specified.
    """
    # Convert dicts to DatasetRecord if needed
    if records and isinstance(records[0], dict):
        records = _records_to_objects(records)  # type: ignore

    protocol_str = protocol.value if isinstance(protocol, SplitProtocol) else protocol

    if protocol_str == SplitProtocol.S_RANDOM.value:
        return split_s_random(records, seed=seed)  # type: ignore
    elif protocol_str == SplitProtocol.S_ATTACK_HOLDOUT.value:
        return split_s_attack_holdout(records, seed=seed)  # type: ignore
    elif protocol_str == SplitProtocol.S_TOOL_HOLDOUT.value:
        return split_s_tool_holdout(records, seed=seed)  # type: ignore
    else:
        raise ValueError(f"Unknown protocol: {protocol}. "
                        f"Available: {[p.value for p in SplitProtocol]}")


def verify_no_template_leakage(
    splits: dict[str, list[DatasetRecord]],
) -> bool:
    """Verify that no template_id appears in multiple splits.

    Args:
        splits: Dictionary mapping split names to record lists.

    Returns:
        True if no leakage, raises AssertionError otherwise.
    """
    split_names = list(splits.keys())

    for i, name1 in enumerate(split_names):
        templates1 = {r.template_id for r in splits[name1]}
        for name2 in split_names[i + 1:]:
            templates2 = {r.template_id for r in splits[name2]}
            overlap = templates1 & templates2
            assert len(overlap) == 0, (
                f"Template leakage between {name1} and {name2}: {overlap}"
            )

    return True


def save_splits(
    splits: dict[str, list[DatasetRecord]],
    output_dir: str | Path,
    protocol: str,
    seed: int,
) -> dict[str, Path]:
    """Save splits to JSONL files with manifests.

    Args:
        splits: Dictionary mapping split names to record lists.
        output_dir: Directory to save split files.
        protocol: Split protocol name.
        seed: Split seed.

    Returns:
        Dictionary mapping split names to output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for split_name, records in splits.items():
        # Save records
        split_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(records, split_path)
        paths[split_name] = split_path

        # Save manifest
        manifest = _create_split_manifest(records, protocol, seed, split_name)
        manifest_path = output_dir / f"{split_name}_manifest.json"
        save_manifest(manifest.model_dump(), manifest_path)

    # Save combined manifest
    combined_manifest = {
        "protocol": protocol,
        "seed": seed,
        "splits": {
            name: _create_split_manifest(records, protocol, seed, name).model_dump()
            for name, records in splits.items()
        },
    }
    save_manifest(combined_manifest, output_dir / "manifest.json")

    return paths


def create_and_save_splits(
    input_path: str | Path,
    output_dir: str | Path,
    protocol: str,
    seed: int = 2026,
) -> dict[str, Path]:
    """Load dataset, create splits, and save to files.

    Args:
        input_path: Path to input dataset.jsonl.
        output_dir: Directory to save split files.
        protocol: Split protocol name.
        seed: Split seed.

    Returns:
        Dictionary mapping split names to output paths.
    """
    # Load records
    records_raw = load_jsonl(input_path)
    records = _records_to_objects(records_raw)

    # Create splits
    splits = create_splits(records, protocol, seed)

    # Verify no leakage
    verify_no_template_leakage(splits)

    # Save
    return save_splits(splits, output_dir, protocol, seed)


def main() -> None:
    """CLI entry point for split generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset splits")
    parser.add_argument("--input", type=str, required=True, help="Input dataset.jsonl path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=[p.value for p in SplitProtocol],
        help="Split protocol",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")

    args = parser.parse_args()

    print(f"Creating {args.protocol} split with seed={args.seed}")
    paths = create_and_save_splits(
        input_path=args.input,
        output_dir=args.output,
        protocol=args.protocol,
        seed=args.seed,
    )
    print(f"Splits saved to: {args.output}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
