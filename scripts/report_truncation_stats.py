#!/usr/bin/env python3
"""Compute and report truncation retention statistics for the ablation study.

This script does NOT require trained models — it only needs the tokenizer
and config to simulate truncation behaviour on test-set records.

Outputs:
    {experiment_root}/truncation_stats.csv
    {experiment_root}/truncation_stats.json
    {figures_dir}/truncation_bias_prompt_retention_vs_schema.png

Usage:
    # Default (standard dataset + configs)
    python scripts/report_truncation_stats.py

    # Long-schema stress test
    python scripts/report_truncation_stats.py \\
        --splits-dir data/splits_longschema \\
        --experiment-root data/reports/experiments_longschema \\
        --config-set longschema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_PROTOCOLS = ["S_random", "S_attack_holdout"]

# Strategy → config YAML (keyed by config set name)
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_records(path: Path):
    """Load DatasetRecord objects from a JSONL file."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from toolshield.data.schema import DatasetRecord
    from toolshield.utils.io import load_jsonl

    raw = load_jsonl(path)
    return [DatasetRecord(**r) for r in raw]


def load_config(path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def make_classifier(config: dict):
    """Instantiate a ContextTransformerClassifier with tokenizer ready."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from toolshield.models.context_transformer import ContextTransformerClassifier
    from transformers import AutoTokenizer

    clf = ContextTransformerClassifier(config=config)
    model_name = config.get("model_name", "distilroberta-base")
    clf.tokenizer = AutoTokenizer.from_pretrained(model_name)
    return clf


# ── Distribution helper ───────────────────────────────────────────────────────

def distribution_stats(arr) -> dict[str, float]:
    """Compute min / mean / p95 / max for a numeric array."""
    a = np.array(arr)
    if len(a) == 0:
        return {"min": 0, "mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(a.min()),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "max": int(a.max()),
    }


def compute_schema_char_counts(records) -> list[int]:
    """Compute serialised JSON schema length in characters for each record."""
    import json as _json
    return [len(_json.dumps(r.tool_schema, separators=(",", ":"))) for r in records]


# ── Core ─────────────────────────────────────────────────────────────────────

def compute_stats_for_strategy(
    strategy: str, config_path: Path, records, protocol: str
) -> dict:
    """Compute truncation stats for one (strategy, protocol) pair."""
    config = load_config(config_path)
    clf = make_classifier(config)
    stats = clf.compute_truncation_stats(records)
    summary = stats.summary()

    # Distribution stats for schema and prompt tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model_name", "distilroberta-base")
    )
    schema_chars = compute_schema_char_counts(records)
    schema_tokens = compute_schema_token_counts(records, tokenizer)

    dist = {
        "schema_chars_dist": distribution_stats(schema_chars),
        "schema_tokens_dist": distribution_stats(schema_tokens),
        "prompt_tokens_dist": distribution_stats(stats.prompt_tokens_total),
    }

    # Merge distribution stats into summary for CSV flattening
    for dist_name, dist_vals in dist.items():
        prefix = dist_name.replace("_dist", "")
        for k, v in dist_vals.items():
            summary[f"{prefix}_{k}"] = v

    return {
        "protocol": protocol,
        "strategy": strategy,
        "summary": summary,
        "distributions": dist,
        "per_sample": {
            "prompt_tokens_total": stats.prompt_tokens_total,
            "prompt_tokens_retained": stats.prompt_tokens_retained,
            "context_tokens_total": stats.context_tokens_total,
            "context_tokens_retained": stats.context_tokens_retained,
            "prompt_retention_ratio": stats.prompt_retention_ratio,
        },
    }


def compute_schema_token_counts(records, tokenizer) -> list[int]:
    """Compute the tokenized schema length for each record."""
    import json as _json
    counts = []
    for r in records:
        schema_str = _json.dumps(r.tool_schema, separators=(",", ":"))
        ids = tokenizer.encode(schema_str, add_special_tokens=False)
        counts.append(len(ids))
    return counts


# ── Output ───────────────────────────────────────────────────────────────────

def build_summary_df(all_stats: list[dict]) -> pd.DataFrame:
    """Build a summary DataFrame from computed stats."""
    rows = []
    for s in all_stats:
        row = {
            "protocol": s["protocol"],
            "strategy": s["strategy"],
        }
        row.update(s["summary"])
        rows.append(row)
    return pd.DataFrame(rows)


def generate_figure(
    all_stats: list[dict],
    all_records: dict[str, list],
    output_path: Path,
) -> None:
    """Generate prompt retention vs schema length figure.

    Grouped bar chart: naive vs keep_prompt retention at schema-length quartiles.
    One subplot per protocol.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    protocols = sorted(set(s["protocol"] for s in all_stats))
    n_protocols = len(protocols)

    fig, axes = plt.subplots(1, n_protocols, figsize=(6 * n_protocols, 5), squeeze=False)

    # Style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        records = all_records[protocol]
        schema_counts = np.array(compute_schema_token_counts(records, tokenizer))

        # Quartile bins
        q25, q50, q75 = np.percentile(schema_counts, [25, 50, 75])
        bin_edges = [0, q25, q50, q75, schema_counts.max() + 1]
        bin_labels = [
            f"Q1 (≤{int(q25)})",
            f"Q2 ({int(q25)}-{int(q50)})",
            f"Q3 ({int(q50)}-{int(q75)})",
            f"Q4 (>{int(q75)})",
        ]
        bin_indices = np.digitize(schema_counts, bin_edges) - 1  # 0-based

        # Collect retention per strategy per quartile
        strategy_data: dict[str, list[float]] = {}
        for s in all_stats:
            if s["protocol"] != protocol:
                continue
            retention = np.array(s["per_sample"]["prompt_retention_ratio"])
            means = []
            for q in range(4):
                mask = bin_indices == q
                if mask.sum() > 0:
                    means.append(float(np.mean(retention[mask])))
                else:
                    means.append(0.0)
            strategy_data[s["strategy"]] = means

        # Bar chart
        x = np.arange(len(bin_labels))
        width = 0.35

        strategies = sorted(strategy_data.keys())
        colors = {"naive": "#e74c3c", "keep_prompt": "#2ecc71"}
        hatches = {"naive": "//", "keep_prompt": ""}

        for i, strat in enumerate(strategies):
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                strategy_data[strat],
                width,
                label=strat.replace("_", " "),
                color=colors.get(strat, f"C{i}"),
                hatch=hatches.get(strat, ""),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Schema Token Length (Quartile)")
        ax.set_ylabel("Mean Prompt Retention Ratio")
        ax.set_title(f"{protocol.replace('_', ' ')}")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle("Prompt Token Retention vs. Schema Length", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {output_path}")


def generate_decile_figure(
    all_stats: list[dict],
    all_records: dict[str, list],
    output_path: Path,
) -> None:
    """Generate prompt retention vs schema token length DECILES figure.

    Grouped bar chart: naive vs keep_prompt retention at 10 schema-length decile bins.
    One subplot per protocol.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    protocols = sorted(set(s["protocol"] for s in all_stats))
    n_protocols = len(protocols)

    fig, axes = plt.subplots(1, n_protocols, figsize=(8 * n_protocols, 5), squeeze=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        records = all_records[protocol]
        schema_counts = np.array(compute_schema_token_counts(records, tokenizer))

        # Compute decile edges (10th, 20th, ..., 100th percentiles)
        decile_pcts = list(range(10, 101, 10))
        edges = [0] + [float(np.percentile(schema_counts, p)) for p in decile_pcts]

        # Build labels and bin assignments
        bin_labels = []
        for i in range(10):
            lo = int(edges[i])
            hi = int(edges[i + 1])
            bin_labels.append(f"D{i+1}\n({lo}-{hi})")

        bin_indices = np.digitize(schema_counts, edges[1:], right=True)  # 0-9
        bin_indices = np.clip(bin_indices, 0, 9)

        # Collect retention per strategy per decile
        strategy_data: dict[str, list[float]] = {}
        for s in all_stats:
            if s["protocol"] != protocol:
                continue
            retention = np.array(s["per_sample"]["prompt_retention_ratio"])
            means = []
            for d in range(10):
                mask = bin_indices == d
                if mask.sum() > 0:
                    means.append(float(np.mean(retention[mask])))
                else:
                    means.append(0.0)
            strategy_data[s["strategy"]] = means

        # Bar chart
        x = np.arange(10)
        width = 0.38

        strategies = sorted(strategy_data.keys())
        colors = {"naive": "#e74c3c", "keep_prompt": "#2ecc71"}
        hatches = {"naive": "//", "keep_prompt": ""}

        for i, strat in enumerate(strategies):
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                strategy_data[strat],
                width,
                label=strat.replace("_", " "),
                color=colors.get(strat, f"C{i}"),
                hatch=hatches.get(strat, ""),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Schema Token Length (Decile)")
        ax.set_ylabel("Mean Prompt Retention Ratio")
        ax.set_title(f"{protocol.replace('_', ' ')}")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle("Prompt Token Retention vs. Schema Length (Deciles)", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Decile figure saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Report truncation retention stats")
    parser.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS)
    parser.add_argument(
        "--splits-dir", type=Path, default=DATA_DIR / "splits",
        help="Directory containing split protocol subdirs (default: data/splits)",
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=DATA_DIR / "reports" / "experiments",
        help="Output root for CSV/JSON (default: data/reports/experiments)",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DATA_DIR / "reports" / "figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--config-set", choices=list(STRATEGY_CONFIG_SETS.keys()), default="default",
        help="Which config set to use (default or longschema)",
    )
    parser.add_argument(
        "--figure-suffix", default="",
        help="Suffix for figure filename (e.g., '_longschema')",
    )
    args = parser.parse_args()

    splits_dir = args.splits_dir
    experiment_root = args.experiment_root
    figures_dir = args.figures_dir
    strategy_configs = STRATEGY_CONFIG_SETS[args.config_set]

    print("=" * 60)
    print("Truncation Stats Reporter")
    print("=" * 60)
    print(f"Protocols:   {args.protocols}")
    print(f"Config set:  {args.config_set}")
    print(f"Splits dir:  {splits_dir}")
    print(f"Output root: {experiment_root}")
    print(f"Strategies:  {list(strategy_configs.keys())}")
    print()

    all_stats: list[dict] = []
    all_records: dict[str, list] = {}

    for protocol in args.protocols:
        test_path = splits_dir / protocol / "test.jsonl"
        if not test_path.exists():
            print(f"  [WARN] {test_path} not found, skipping {protocol}")
            continue

        print(f"  Loading {protocol} test set...")
        records = load_records(test_path)
        all_records[protocol] = records
        print(f"  Loaded {len(records)} records")

        for strategy, config_path in strategy_configs.items():
            if not config_path.exists():
                print(f"  [WARN] Config {config_path} not found, skipping")
                continue
            print(f"  Computing stats: {protocol} / {strategy}...")
            result = compute_stats_for_strategy(strategy, config_path, records, protocol)
            all_stats.append(result)

            # Print quick summary
            s = result["summary"]
            print(f"    n_samples={s['n_samples']}, "
                  f"prompt_truncated={s['n_prompt_truncated']} "
                  f"({s['pct_prompt_truncated']*100:.1f}%), "
                  f"retention_mean={s['prompt_retention_ratio_mean']:.3f}")

    if not all_stats:
        print("No stats computed — check splits exist.")
        return

    # Save CSV
    experiment_root.mkdir(parents=True, exist_ok=True)
    df = build_summary_df(all_stats)
    csv_path = experiment_root / "truncation_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # Save JSON (with per-sample arrays and distribution stats)
    json_path = experiment_root / "truncation_stats.json"
    json_data = []
    for s in all_stats:
        entry = {
            "protocol": s["protocol"],
            "strategy": s["strategy"],
            "summary": s["summary"],
            "distributions": s.get("distributions", {}),
            "per_sample": {
                k: [float(v) for v in vals]
                for k, vals in s["per_sample"].items()
            },
        }
        json_data.append(entry)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # Generate quartile figure
    fig_name = f"truncation_bias_prompt_retention_vs_schema{args.figure_suffix}.png"
    fig_path = figures_dir / fig_name
    print(f"\n  Generating quartile figure...")
    generate_figure(all_stats, all_records, fig_path)

    # Generate decile figure
    decile_fig_name = f"truncation_bias_prompt_retention_vs_schema_deciles{args.figure_suffix}.png"
    decile_fig_path = figures_dir / decile_fig_name
    print(f"  Generating decile figure...")
    generate_decile_figure(all_stats, all_records, decile_fig_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
