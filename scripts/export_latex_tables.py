#!/usr/bin/env python3
"""Export evaluation results to LaTeX table format for thesis.

This script reads evaluation reports and generates LaTeX table rows
that can be directly pasted into the thesis document.

Usage:
    python scripts/export_latex_tables.py
    python scripts/export_latex_tables.py --input-dir data/reports/S_random/
    python scripts/export_latex_tables.py --all-protocols

Output formats:
    1. Main results table (model x metrics)
    2. Budget evaluation table (budget x metrics)
    3. Ablation table (if available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage for LaTeX.
    
    Args:
        value: Value between 0 and 1.
        decimals: Number of decimal places.
        
    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}\\%"


def format_float(value: float, decimals: int = 3) -> str:
    """Format a float for LaTeX.
    
    Args:
        value: Float value.
        decimals: Number of decimal places.
        
    Returns:
        Formatted string.
    """
    return f"{value:.{decimals}f}"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters.
    
    Args:
        text: Input text.
        
    Returns:
        LaTeX-safe string.
    """
    replacements = {
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "$": "\\$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_main_results_table(
    tables_csv: Path,
    caption: str = "Main Results",
) -> str:
    """Generate LaTeX table for main results.
    
    Expected columns: model, split, fpr_at_tpr_0_90, fpr_at_tpr_0_95, pr_auc, roc_auc
    
    Args:
        tables_csv: Path to tables.csv file.
        caption: Table caption.
        
    Returns:
        LaTeX table string.
    """
    if not tables_csv.exists():
        return f"% Table not found: {tables_csv}\n"
    
    df = pd.read_csv(tables_csv)
    
    lines = []
    lines.append("% Main Results Table")
    lines.append("% Copy the following into your thesis LaTeX file")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:main-results}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Model & Split & FPR@TPR$_{0.90}$ & FPR@TPR$_{0.95}$ & PR-AUC & ROC-AUC \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        model = escape_latex(str(row.get("model", "unknown")))
        split = escape_latex(str(row.get("split", "unknown")))
        fpr_90 = format_percentage(row.get("fpr_at_tpr_0_90", 0))
        fpr_95 = format_percentage(row.get("fpr_at_tpr_0_95", 0))
        pr_auc = format_float(row.get("pr_auc", 0))
        roc_auc = format_float(row.get("roc_auc", 0))
        
        lines.append(f"{model} & {split} & {fpr_90} & {fpr_95} & {pr_auc} & {roc_auc} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)


def generate_budget_table(
    budget_csv: Path,
    caption: str = "Budget-based Threshold Evaluation",
) -> str:
    """Generate LaTeX table for budget evaluation.
    
    Expected columns: model, split, budget, threshold, tpr, asr, blocked_benign_rate
    
    Args:
        budget_csv: Path to budget_tables.csv file.
        caption: Table caption.
        
    Returns:
        LaTeX table string.
    """
    if not budget_csv.exists():
        return f"% Budget table not found: {budget_csv}\n"
    
    df = pd.read_csv(budget_csv)
    
    if df.empty:
        return "% Budget table is empty\n"
    
    lines = []
    lines.append("% Budget Evaluation Table")
    lines.append("% Copy the following into your thesis LaTeX file")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:budget-eval}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Model & Budget & Threshold & TPR & ASR & Blocked Benign \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        model = escape_latex(str(row.get("model", "unknown")))
        budget = format_percentage(row.get("budget", 0))
        threshold = format_float(row.get("threshold", 0), 4)
        tpr = format_percentage(row.get("tpr", 0))
        asr = format_percentage(row.get("asr", 0))
        blocked = format_percentage(row.get("blocked_benign_rate", 0))
        
        lines.append(f"{model} & {budget} & {threshold} & {tpr} & {asr} & {blocked} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)


def generate_ablation_table(
    results_dir: Path,
    caption: str = "Context Ablation Study",
) -> str:
    """Generate LaTeX table for ablation study.
    
    Compares text-only vs context-augmented transformer.
    
    Args:
        results_dir: Directory containing evaluation results.
        caption: Table caption.
        
    Returns:
        LaTeX table string.
    """
    # Look for transformer and context_transformer results
    tables_csv = results_dir / "tables.csv"
    if not tables_csv.exists():
        return "% Ablation table requires tables.csv\n"
    
    df = pd.read_csv(tables_csv)
    
    # Filter for transformer variants
    transformer_rows = df[df["model"].str.contains("transformer", case=False)]
    
    if transformer_rows.empty:
        return "% No transformer results found for ablation\n"
    
    lines = []
    lines.append("% Context Ablation Table")
    lines.append("% Copy the following into your thesis LaTeX file")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Configuration & FPR@TPR$_{0.90}$ & FPR@TPR$_{0.95}$ & PR-AUC & ROC-AUC \\\\")
    lines.append("\\midrule")
    
    for _, row in transformer_rows.iterrows():
        model = row.get("model", "unknown")
        if "context" in str(model).lower():
            config = "Text + Context"
        else:
            config = "Text Only"
        
        fpr_90 = format_percentage(row.get("fpr_at_tpr_0_90", 0))
        fpr_95 = format_percentage(row.get("fpr_at_tpr_0_95", 0))
        pr_auc = format_float(row.get("pr_auc", 0))
        roc_auc = format_float(row.get("roc_auc", 0))
        
        lines.append(f"{config} & {fpr_90} & {fpr_95} & {pr_auc} & {roc_auc} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export evaluation results to LaTeX table format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/reports/S_random"),
        help="Input directory containing tables.csv and metrics.json"
    )
    parser.add_argument(
        "--all-protocols",
        action="store_true",
        help="Generate tables for all protocol directories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    output_lines = []
    output_lines.append("%" + "=" * 70)
    output_lines.append("% ToolShield Evaluation Results - LaTeX Tables")
    output_lines.append("% Generated by scripts/export_latex_tables.py")
    output_lines.append("%" + "=" * 70)
    output_lines.append("")
    output_lines.append("% Required packages:")
    output_lines.append("% \\usepackage{booktabs}")
    output_lines.append("")
    
    if args.all_protocols:
        # Find all protocol directories
        reports_dir = Path("data/reports")
        protocol_dirs = [d for d in reports_dir.iterdir() if d.is_dir()]
    else:
        protocol_dirs = [args.input_dir]
    
    for protocol_dir in protocol_dirs:
        if not protocol_dir.exists():
            output_lines.append(f"% Directory not found: {protocol_dir}")
            continue
        
        protocol_name = protocol_dir.name
        output_lines.append(f"%" + "-" * 70)
        output_lines.append(f"% Protocol: {protocol_name}")
        output_lines.append(f"%" + "-" * 70)
        output_lines.append("")
        
        # Main results table
        tables_csv = protocol_dir / "tables.csv"
        output_lines.append(generate_main_results_table(
            tables_csv,
            caption=f"Main Results - {protocol_name}"
        ))
        
        # Budget table
        budget_csv = protocol_dir / "budget_tables.csv"
        output_lines.append(generate_budget_table(
            budget_csv,
            caption=f"Budget Evaluation - {protocol_name}"
        ))
        
        # Ablation table
        output_lines.append(generate_ablation_table(
            protocol_dir,
            caption=f"Context Ablation - {protocol_name}"
        ))
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"LaTeX tables written to: {args.output}")
    else:
        print(output_text)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
