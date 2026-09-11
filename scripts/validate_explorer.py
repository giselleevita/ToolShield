#!/usr/bin/env python3
"""Fail closed when the static explorer and its evidence contract diverge."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "index.html"
JAVASCRIPT = ROOT / "docs" / "app.js"


def validate() -> None:
    html = HTML.read_text(encoding="utf-8")
    javascript = JAVASCRIPT.read_text(encoding="utf-8")
    element_ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', html))
    referenced_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', javascript))
    missing = sorted(referenced_ids - element_ids)
    if missing:
        raise ValueError(f"JavaScript references missing explorer elements: {', '.join(missing)}")

    script_sources = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
    external = [source for source in script_sources if "://" in source or source.startswith("//")]
    if external:
        raise ValueError(f"Explorer must not execute third-party scripts: {', '.join(external)}")

    if 'fetch("results.json")' not in javascript and "fetch('results.json')" not in javascript:
        raise ValueError("Explorer must load the committed machine-readable report")

    required_copy = ("static", "no model key", "limitations", "provenance")
    normalized = html.lower()
    absent_copy = [phrase for phrase in required_copy if phrase not in normalized]
    if absent_copy:
        raise ValueError(f"Explorer is missing required disclosure text: {', '.join(absent_copy)}")


if __name__ == "__main__":
    validate()
    print("Validated static explorer contract")
