"""Dataset generation for prompt injection detection.

This module generates a balanced dataset of benign and attack prompts
for training prompt injection detection models. The generation is
fully deterministic given the seed.

Usage:
    from toolshield.data.generate_dataset import generate_dataset
    records, manifest = generate_dataset(seed=1337, n_samples=1000)
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from toolshield.data.schema import AttackFamily, DatasetManifest, DatasetRecord
from toolshield.data.templates.attack import ATTACK_TEMPLATES, AttackTemplate
from toolshield.data.templates.benign import BENIGN_TEMPLATES, BenignTemplate
from toolshield.data.tools import TOOL_BY_NAME, TOOL_NAMES
from toolshield.utils.io import save_jsonl, save_manifest


def _deterministic_uuid(
    seed: int, template_id: str, variant_id: str, tool_name: str, iteration: int
) -> str:
    """Generate a deterministic UUID-like ID from the generation parameters.

    Args:
        seed: The global seed.
        template_id: The template identifier.
        variant_id: The variant identifier.
        tool_name: The tool name.
        iteration: The iteration index within the template (ensures uniqueness).

    Returns:
        A deterministic hex string ID.
    """
    combined = f"{seed}:{template_id}:{variant_id}:{tool_name}:{iteration}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _generate_benign_records(
    rng: random.Random,
    seed: int,
    n_samples: int,
) -> list[DatasetRecord]:
    """Generate benign prompt records.

    Args:
        rng: Seeded random number generator.
        seed: The global seed for ID generation.
        n_samples: Number of benign samples to generate.

    Returns:
        List of DatasetRecord objects for benign prompts.
    """
    records: list[DatasetRecord] = []
    samples_per_template = max(1, n_samples // len(BENIGN_TEMPLATES))
    extra_samples = n_samples - (samples_per_template * len(BENIGN_TEMPLATES))

    for template in BENIGN_TEMPLATES:
        # Determine applicable tools
        if template.tool_name == "any":
            applicable_tools = TOOL_NAMES
        else:
            applicable_tools = [template.tool_name]

        # Generate samples for this template
        n_for_template = samples_per_template
        if extra_samples > 0:
            n_for_template += 1
            extra_samples -= 1

        for i in range(n_for_template):
            # Select tool
            tool_name = rng.choice(applicable_tools)
            tool = TOOL_BY_NAME[tool_name]

            # Select variant
            variant_idx = i % len(template.variants)
            variant_id = f"v{variant_idx}"
            variant_fn = template.variants[variant_idx]

            # Generate prompt
            prompt = variant_fn(rng, tool_name)

            # Create record
            record_id = _deterministic_uuid(
                seed, template.template_id, variant_id, tool_name, i
            )

            record = DatasetRecord(
                id=f"ben_{record_id}",
                language="en",
                role_sequence=template.role_sequence,
                tool_name=tool_name,
                tool_schema=tool.schema_,
                tool_description=tool.description,
                prompt=prompt,
                label_binary=0,
                attack_family=None,
                attack_goal=None,
                template_id=template.template_id,
                variant_id=variant_id,
                seed=seed,
            )
            records.append(record)

    return records[:n_samples]  # Ensure exact count


def _generate_attack_records(
    rng: random.Random,
    seed: int,
    n_samples: int,
) -> list[DatasetRecord]:
    """Generate attack prompt records.

    Distributes samples evenly across attack families (AF1-AF4).

    Args:
        rng: Seeded random number generator.
        seed: The global seed for ID generation.
        n_samples: Total number of attack samples to generate.

    Returns:
        List of DatasetRecord objects for attack prompts.
    """
    records: list[DatasetRecord] = []
    families = [AttackFamily.AF1.value, AttackFamily.AF2.value,
                AttackFamily.AF3.value, AttackFamily.AF4.value]
    samples_per_family = n_samples // len(families)
    extra_samples = n_samples - (samples_per_family * len(families))

    for family in families:
        # Get templates for this family
        family_templates = [t for t in ATTACK_TEMPLATES if t.attack_family == family]

        # Samples for this family
        n_for_family = samples_per_family
        if extra_samples > 0:
            n_for_family += 1
            extra_samples -= 1

        samples_per_template = max(1, n_for_family // len(family_templates))
        template_extra = n_for_family - (samples_per_template * len(family_templates))

        for template in family_templates:
            # Determine applicable tools
            if template.tool_name == "any":
                applicable_tools = TOOL_NAMES
            else:
                applicable_tools = [template.tool_name]

            n_for_template = samples_per_template
            if template_extra > 0:
                n_for_template += 1
                template_extra -= 1

            for i in range(n_for_template):
                # Select tool
                tool_name = rng.choice(applicable_tools)
                tool = TOOL_BY_NAME[tool_name]

                # Select variant
                variant_idx = i % len(template.variants)
                variant_id = f"v{variant_idx}"
                variant_fn = template.variants[variant_idx]

                # Generate prompt
                prompt = variant_fn(rng, tool_name)

                # Create record
                record_id = _deterministic_uuid(
                    seed, template.template_id, variant_id, tool_name, i
                )

                record = DatasetRecord(
                    id=f"atk_{record_id}",
                    language="en",
                    role_sequence=template.role_sequence,
                    tool_name=tool_name,
                    tool_schema=tool.schema_,
                    tool_description=tool.description,
                    prompt=prompt,
                    label_binary=1,
                    attack_family=template.attack_family,
                    attack_goal=template.attack_goal,
                    template_id=template.template_id,
                    variant_id=variant_id,
                    seed=seed,
                )
                records.append(record)

    return records[:n_samples]  # Ensure exact count


def generate_dataset(
    seed: int = 1337,
    n_samples: int = 1000,
) -> tuple[list[DatasetRecord], DatasetManifest]:
    """Generate a balanced dataset of benign and attack prompts.

    The dataset is 50% benign, 50% attack. Attack samples are evenly
    distributed among the four attack families (AF1-AF4).

    Args:
        seed: Random seed for reproducibility (default: 1337).
        n_samples: Total number of samples to generate (default: 1000).

    Returns:
        Tuple of (records, manifest) where records is the list of
        DatasetRecord objects and manifest contains statistics.
    """
    rng = random.Random(seed)

    # Generate balanced dataset
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign

    benign_records = _generate_benign_records(rng, seed, n_benign)
    attack_records = _generate_attack_records(rng, seed, n_attack)

    # Combine and shuffle
    all_records = benign_records + attack_records
    rng.shuffle(all_records)

    # Compute manifest statistics
    label_counts = Counter(r.label_binary for r in all_records)
    family_counts = Counter(r.attack_family for r in all_records)
    tool_counts = Counter(r.tool_name for r in all_records)

    # Template counts by category
    benign_templates = set(r.template_id for r in all_records if r.is_benign())
    attack_templates = set(r.template_id for r in all_records if r.is_attack())

    manifest = DatasetManifest(
        seed=seed,
        total_records=len(all_records),
        label_counts={str(k): v for k, v in sorted(label_counts.items())},
        family_counts={str(k) if k else "benign": v for k, v in sorted(family_counts.items(), key=lambda x: str(x[0]))},
        tool_counts=dict(sorted(tool_counts.items())),
        template_counts={
            "benign": len(benign_templates),
            "attack": len(attack_templates),
            "total": len(benign_templates) + len(attack_templates),
        },
    )

    return all_records, manifest


def generate_and_save(
    output_dir: str | Path,
    seed: int = 1337,
    n_samples: int = 1000,
) -> tuple[Path, Path]:
    """Generate dataset and save to files.

    Args:
        output_dir: Directory to save output files.
        seed: Random seed for reproducibility.
        n_samples: Total number of samples to generate.

    Returns:
        Tuple of (dataset_path, manifest_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, manifest = generate_dataset(seed=seed, n_samples=n_samples)

    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"

    save_jsonl(records, dataset_path)
    save_manifest(manifest.model_dump(), manifest_path)

    return dataset_path, manifest_path


def main() -> None:
    """CLI entry point for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate prompt injection dataset")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")

    args = parser.parse_args()

    print(f"Generating dataset with seed={args.seed}, n_samples={args.n_samples}")
    dataset_path, manifest_path = generate_and_save(
        output_dir=args.output,
        seed=args.seed,
        n_samples=args.n_samples,
    )
    print(f"Dataset saved to: {dataset_path}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
