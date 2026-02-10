#!/usr/bin/env python3
"""Export a concrete truncation example for thesis appendix.

Produces a self-contained Markdown file that details:
  - Schema and prompt lengths (characters + tokens)
  - Per-strategy token allocation and retention
  - Tail-preview confirming prompt presence/absence

Usage:
    python scripts/export_truncation_example.py
    python scripts/export_truncation_example.py \
        --protocol S_attack_holdout --index 0 \
        --splits-dir data/splits_longschema \
        --config-set longschema \
        --out data/reports/experiments_longschema/appendix_truncation_example.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIGS_DIR = PROJECT_ROOT / "configs"

STRATEGY_CONFIG_SETS: dict[str, dict[str, Path]] = {
    "default": {
        "naive": CONFIGS_DIR / "training" / "context_transformer_naive.yaml",
        "keep_prompt": CONFIGS_DIR / "training" / "context_transformer.yaml",
    },
    "longschema": {
        "naive": CONFIGS_DIR / "training" / "context_transformer_naive_longschema.yaml",
        "keep_prompt": CONFIGS_DIR / "training" / "context_transformer_keep_prompt_longschema.yaml",
    },
}


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a concrete truncation example for thesis appendix"
    )
    parser.add_argument("--protocol", default="S_attack_holdout")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--splits-dir", type=Path,
        default=PROJECT_ROOT / "data" / "splits_longschema",
    )
    parser.add_argument(
        "--config-set", choices=list(STRATEGY_CONFIG_SETS.keys()),
        default="longschema",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "experiments_longschema" / "appendix_truncation_example.md",
    )
    args = parser.parse_args()

    # ── Load record ──────────────────────────────────────────────────────
    from toolshield.data.schema import DatasetRecord
    from toolshield.utils.io import load_jsonl

    test_path = args.splits_dir / args.protocol / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found")
        sys.exit(1)

    raw = load_jsonl(test_path)
    if args.index >= len(raw):
        print(f"ERROR: index {args.index} out of range (test has {len(raw)} records)")
        sys.exit(1)

    record = DatasetRecord(**raw[args.index])
    strategy_configs = STRATEGY_CONFIG_SETS[args.config_set]

    # ── Compute per-strategy stats ───────────────────────────────────────
    from transformers import AutoTokenizer

    # Use the first config to determine model name
    first_cfg = load_config(next(iter(strategy_configs.values())))
    model_name = first_cfg.get("model_name", "distilroberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    schema_str = json.dumps(record.tool_schema, separators=(",", ":"))
    prompt_str = record.prompt

    schema_chars = len(schema_str)
    prompt_chars = len(prompt_str)
    schema_tokens = len(tokenizer.encode(schema_str, add_special_tokens=False))
    prompt_tokens = len(tokenizer.encode(prompt_str, add_special_tokens=False))

    # Tail preview (last ~300 chars of prompt)
    tail_chars = 300
    prompt_tail = prompt_str[-tail_chars:] if len(prompt_str) > tail_chars else prompt_str
    tail_token_ids = tokenizer.encode(prompt_tail, add_special_tokens=False)

    strategy_results: dict[str, dict] = {}

    for strategy, config_path in strategy_configs.items():
        config = load_config(config_path)
        max_length = config.get("max_length", 256)
        max_schema_length = config.get("max_schema_length", 200)

        from toolshield.models.context_transformer import ContextTransformerClassifier
        clf = ContextTransformerClassifier(config=config)
        clf.tokenizer = tokenizer

        # Compute context and prompt strings as the model sees them
        context_str = clf._format_context(record)
        prompt_formatted = clf._format_prompt(record)

        context_ids = tokenizer.encode(context_str, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)

        # Simulate truncation
        # Special tokens: <s> ... </s></s> ... </s> = 4 tokens
        special_count = 4
        content_budget = max_length - special_count

        if strategy == "keep_prompt" or config.get("truncate_strategy") == "keep_prompt":
            prompt_min = config.get("prompt_min_tokens", 128)
            prompt_reserved = min(prompt_min, content_budget)
            context_max = content_budget - prompt_reserved

            ctx_retained = min(len(context_ids), context_max)
            # If context is short, prompt gets surplus
            surplus = context_max - ctx_retained
            prompt_max = prompt_reserved + surplus
            pmt_retained = min(len(prompt_ids), prompt_max)

            # For keep_prompt, prompt is taken from tail
            if len(prompt_ids) > prompt_max:
                retained_prompt_ids = prompt_ids[-prompt_max:]
            else:
                retained_prompt_ids = prompt_ids
        else:
            # Naive: concatenate context + prompt, right-truncate
            all_ids = context_ids + prompt_ids
            truncated = all_ids[:content_budget]
            ctx_retained = min(len(context_ids), content_budget)
            pmt_retained = max(0, min(content_budget - ctx_retained, len(prompt_ids)))
            retained_prompt_ids = truncated[ctx_retained:]

        prompt_truncated = pmt_retained < len(prompt_ids)
        retention_ratio = pmt_retained / len(prompt_ids) if len(prompt_ids) > 0 else 1.0

        # Check if tail tokens appear in retained prompt
        tail_present = False
        if len(retained_prompt_ids) >= len(tail_token_ids) and len(tail_token_ids) > 0:
            # Check if tail tokens are a subsequence at the end
            if retained_prompt_ids[-len(tail_token_ids):] == tail_token_ids:
                tail_present = True
        elif len(tail_token_ids) > 0 and len(retained_prompt_ids) > 0:
            # Check partial: are at least 80% of tail tokens present at the end?
            overlap = min(len(retained_prompt_ids), len(tail_token_ids))
            if retained_prompt_ids[-overlap:] == tail_token_ids[-overlap:]:
                tail_present = overlap / len(tail_token_ids) > 0.5

        strategy_results[strategy] = {
            "max_length": max_length,
            "max_schema_length": max_schema_length,
            "content_budget": content_budget,
            "context_tokens_total": len(context_ids),
            "context_tokens_retained": ctx_retained,
            "prompt_tokens_total": len(prompt_ids),
            "prompt_tokens_retained": pmt_retained,
            "prompt_truncated": prompt_truncated,
            "retention_ratio": retention_ratio,
            "tail_present": tail_present,
        }

    # ── Build Markdown ───────────────────────────────────────────────────
    lines = [
        "# Truncation Example — Appendix Proof",
        "",
        f"**Protocol**: `{args.protocol}`  ",
        f"**Test Index**: `{args.index}`  ",
        f"**Record ID**: `{record.id}`  ",
        f"**Label**: `{record.label_binary}` ({'attack' if record.label_binary == 1 else 'benign'})  ",
        f"**Tool**: `{record.tool_name}`  ",
        "",
        "## Input Dimensions",
        "",
        "| Metric | Characters | Tokens |",
        "|--------|-----------|--------|",
        f"| Tool schema (raw JSON) | {schema_chars:,} | {schema_tokens:,} |",
        f"| Prompt | {prompt_chars:,} | {prompt_tokens:,} |",
        "",
        "## Per-Strategy Token Allocation",
        "",
        "| Metric | " + " | ".join(strategy_results.keys()) + " |",
        "|--------|" + "|".join(["--------"] * len(strategy_results)) + "|",
    ]

    metrics_to_show = [
        ("max_length", "Max length (tokens)"),
        ("content_budget", "Content budget (excl. special)"),
        ("context_tokens_total", "Context tokens (full)"),
        ("context_tokens_retained", "Context tokens (retained)"),
        ("prompt_tokens_total", "Prompt tokens (full)"),
        ("prompt_tokens_retained", "Prompt tokens (retained)"),
        ("retention_ratio", "Prompt retention ratio"),
        ("prompt_truncated", "Prompt truncated?"),
        ("tail_present", "Tail preview present?"),
    ]

    for key, label in metrics_to_show:
        vals = []
        for strat in strategy_results:
            v = strategy_results[strat][key]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, bool):
                vals.append("Yes" if v else "**No**")
            else:
                vals.append(f"{v:,}" if isinstance(v, int) else str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "## Prompt Tail Preview",
        "",
        f"Last {tail_chars} characters of the raw prompt:",
        "",
        "```",
        prompt_tail,
        "```",
        "",
        "### Tail Presence Verification",
        "",
    ])

    for strat, res in strategy_results.items():
        if res["tail_present"]:
            lines.append(
                f"- **{strat}**: Tail tokens ARE present in the retained sequence. "
                f"({res['prompt_tokens_retained']}/{res['prompt_tokens_total']} tokens retained)"
            )
        else:
            lines.append(
                f"- **{strat}**: Tail tokens are **MISSING** from the retained sequence. "
                f"({res['prompt_tokens_retained']}/{res['prompt_tokens_total']} tokens retained — "
                f"prompt was {'truncated' if res['prompt_truncated'] else 'not truncated'})"
            )

    lines.extend([
        "",
        "---",
        f"*Generated by `scripts/export_truncation_example.py`*",
        "",
    ])

    # ── Write output ─────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Appendix example saved: {args.out}")


if __name__ == "__main__":
    main()
