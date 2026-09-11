from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_explorer_contract() -> None:
    path = ROOT / "scripts" / "validate_explorer.py"
    spec = importlib.util.spec_from_file_location("validate_explorer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate()
