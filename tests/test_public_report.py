from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _validator():
    path = ROOT / "scripts/validate_public_report.py"
    spec = importlib.util.spec_from_file_location("validate_public_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_public_report_is_strict_and_complete():
    _validator().validate(ROOT / "docs/results.json")
