"""I/O utilities for JSONL files and configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

    from toolshield.data.schema import DatasetRecord


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output path for the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line contains invalid JSON.
    """
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over records in a JSONL file without loading all into memory.

    Args:
        path: Path to the JSONL file.

    Yields:
        Dictionaries, one per line.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: list[dict[str, Any]] | list["DatasetRecord"], path: str | Path) -> int:
    """Save records to a JSONL file.

    Args:
        records: List of dictionaries or DatasetRecord objects.
        path: Output path for the JSONL file.

    Returns:
        Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # Handle both dicts and Pydantic models
            if hasattr(record, "model_dump"):
                data = record.model_dump()
            else:
                data = record
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    return count


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Save a manifest JSON file with pretty printing.

    Args:
        manifest: Manifest dictionary.
        path: Output path for the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Manifest dictionary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
