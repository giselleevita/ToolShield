#!/usr/bin/env python3
"""Generate LaTeX tables from experiment summary CSV.

Usage:
    python scripts/generate_latex_from_summary.py > data/reports/latex_tables.txt
"""

import pandas as pd
from pathlib import Path


def format_metric(mean: float, std: float, precision: int = 3) -> str:
    """Format metric as mean±std or just mean if std≈0."""
    if std > 0.001:
        return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"
    return f"{mean:.{precision}f}"


def format_percent(mean: float, std: float = 0.0) -> str:
    """Format as percentage."""
    if std > 0.001:
        return f"{mean*100:.1f}$\\pm${std*100:.1f}\\%"
    return f"{mean*100:.1f}\\%"


def generate_main_results_table(df: pd.DataFrame) -> str:
    """Generate main results table (Table 1)."""
    
    lines = [
        "% Table 1: Main Results - Model Performance Across Split Protocols",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Model performance across split protocols. Higher ROC-AUC and lower FPR@TPR are better.}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Protocol & Model & ROC-AUC & PR-AUC & FPR@TPR90 & FPR@TPR95 \\\\",
        "\\midrule",
    ]
    
    for protocol in df['protocol'].unique():
        protocol_df = df[df['protocol'] == protocol]
        for _, row in protocol_df.iterrows():
            model = row['model'].replace('_', '\\_')
            roc = format_metric(row['roc_auc_mean'], row['roc_auc_std'])
            pr = format_metric(row['pr_auc_mean'], row['pr_auc_std'])
            fpr90 = format_metric(row['fpr_at_tpr_90_mean'], row['fpr_at_tpr_90_std'])
            fpr95 = format_metric(row['fpr_at_tpr_95_mean'], row['fpr_at_tpr_95_std'])
            
            prot_display = protocol.replace('_', '\\_')
            lines.append(f"{prot_display} & {model} & {roc} & {pr} & {fpr90} & {fpr95} \\\\")
        lines.append("\\midrule")
    
    # Remove last midrule
    lines[-1] = "\\bottomrule"
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    
    return "\n".join(lines)


def generate_asr_table(df: pd.DataFrame) -> str:
    """Generate ASR reduction table (Table 2)."""
    
    lines = [
        "% Table 2: Attack Success Rate (ASR) Reduction",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{ASR reduction at different TPR operating points. Higher is better.}",
        "\\label{tab:asr-results}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Protocol & Model & ASR Red.@TPR90 & ASR Red.@TPR95 \\\\",
        "\\midrule",
    ]
    
    for protocol in df['protocol'].unique():
        protocol_df = df[df['protocol'] == protocol]
        for _, row in protocol_df.iterrows():
            model = row['model'].replace('_', '\\_')
            asr90 = format_percent(row['asr_reduction_90_mean'], row.get('asr_reduction_90_std', 0))
            asr95 = format_percent(row['asr_reduction_95_mean'], row.get('asr_reduction_95_std', 0))
            
            prot_display = protocol.replace('_', '\\_')
            lines.append(f"{prot_display} & {model} & {asr90} & {asr95} \\\\")
        lines.append("\\midrule")
    
    lines[-1] = "\\bottomrule"
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    
    return "\n".join(lines)


def generate_budget_table(df: pd.DataFrame) -> str:
    """Generate budget evaluation table (Table 3)."""
    
    lines = [
        "% Table 3: Budget-Based Evaluation (Threshold Selected on Validation)",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance at FPR budgets (1\\%, 3\\%, 5\\%). Threshold selected on validation set.}",
        "\\label{tab:budget-results}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Protocol & Model & TPR@1\\% & TPR@3\\% & TPR@5\\% \\\\",
        "\\midrule",
    ]
    
    for protocol in df['protocol'].unique():
        protocol_df = df[df['protocol'] == protocol]
        for _, row in protocol_df.iterrows():
            model = row['model'].replace('_', '\\_')
            
            tpr1 = format_percent(row.get('budget_1_tpr_mean', 0), row.get('budget_1_tpr_std', 0))
            tpr3 = format_percent(row.get('budget_3_tpr_mean', 0), row.get('budget_3_tpr_std', 0))
            tpr5 = format_percent(row.get('budget_5_tpr_mean', 0), row.get('budget_5_tpr_std', 0))
            
            prot_display = protocol.replace('_', '\\_')
            lines.append(f"{prot_display} & {model} & {tpr1} & {tpr3} & {tpr5} \\\\")
        lines.append("\\midrule")
    
    lines[-1] = "\\bottomrule"
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    
    return "\n".join(lines)


def generate_latency_table(df: pd.DataFrame) -> str:
    """Generate latency table (Table 4)."""
    
    lines = [
        "% Table 4: Inference Latency",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Inference latency per batch (100 samples). Lower is better.}",
        "\\label{tab:latency-results}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & P50 (ms) & P95 (ms) \\\\",
        "\\midrule",
    ]
    
    # Get unique models
    seen_models = set()
    for _, row in df.iterrows():
        model = row['model']
        if model in seen_models:
            continue
        seen_models.add(model)
        
        model_display = model.replace('_', '\\_')
        p50 = f"{row.get('latency_p50_ms_mean', 0):.2f}"
        p95 = f"{row.get('latency_p95_ms_mean', 0):.2f}"
        
        lines.append(f"{model_display} & {p50} & {p95} \\\\")
    
    lines.append("\\bottomrule")
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    
    return "\n".join(lines)


def generate_generalization_gap_table(df: pd.DataFrame) -> str:
    """Generate generalization gap comparison."""
    
    lines = [
        "% Table 5: Generalization Gap (S_random vs S_attack_holdout)",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Generalization gap: difference in FPR@TPR90 between random and attack-holdout splits.}",
        "\\label{tab:generalization-gap}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & S\\_random & S\\_attack\\_holdout & $\\Delta$ Gap \\\\",
        "\\midrule",
    ]
    
    random_df = df[df['protocol'] == 'S_random'].set_index('model')
    holdout_df = df[df['protocol'] == 'S_attack_holdout'].set_index('model')
    
    for model in random_df.index:
        if model in holdout_df.index:
            model_display = model.replace('_', '\\_')
            fpr_random = random_df.loc[model, 'fpr_at_tpr_90_mean']
            fpr_holdout = holdout_df.loc[model, 'fpr_at_tpr_90_mean']
            gap = fpr_holdout - fpr_random
            
            lines.append(f"{model_display} & {fpr_random:.3f} & {fpr_holdout:.3f} & {gap:+.3f} \\\\")
    
    lines.append("\\bottomrule")
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    
    return "\n".join(lines)


def generate_truncation_ablation_table(df: pd.DataFrame) -> str:
    """Generate truncation strategy ablation table (Table 6).

    Compares context_transformer_naive vs context_transformer_keep_prompt
    across protocols with KPIs: TPR@FPR=1%/3%/5%, ASR@1%/3%/5%,
    latency P50/P95.
    """
    ablation_models = ["context_transformer_naive", "context_transformer_keep_prompt"]
    adf = df[df["model"].isin(ablation_models)].copy()

    if adf.empty:
        return (
            "% Table 6: Truncation Ablation — no data found for "
            "context_transformer_naive / context_transformer_keep_prompt\n"
        )

    lines = [
        "% Table 6: Truncation Strategy Ablation (Naive vs Keep-Prompt)",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Truncation strategy ablation. "
        "TPR and ASR at fixed FPR budgets (1\\%, 3\\%, 5\\%), "
        "plus warm inference latency. Mean$\\pm$std over seeds.}",
        "\\label{tab:truncation-ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll ccc ccc cc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{TPR (\\%)} "
        "& \\multicolumn{3}{c}{ASR (\\%)} "
        "& \\multicolumn{2}{c}{Latency (ms)} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-10}",
        "Protocol & Strategy "
        "& @1\\% & @3\\% & @5\\% "
        "& @1\\% & @3\\% & @5\\% "
        "& P50 & P95 \\\\",
        "\\midrule",
    ]

    for protocol in sorted(adf["protocol"].unique()):
        pdf = adf[adf["protocol"] == protocol].sort_values("model")
        for _, row in pdf.iterrows():
            model = row["model"]
            strategy = "naive" if "naive" in model else "keep\\_prompt"
            prot_display = protocol.replace("_", "\\_")

            # TPR at budgets
            tpr1 = format_percent(
                row.get("budget_1_tpr_mean", 0), row.get("budget_1_tpr_std", 0)
            )
            tpr3 = format_percent(
                row.get("budget_3_tpr_mean", 0), row.get("budget_3_tpr_std", 0)
            )
            tpr5 = format_percent(
                row.get("budget_5_tpr_mean", 0), row.get("budget_5_tpr_std", 0)
            )

            # ASR at budgets
            asr1 = format_percent(
                row.get("budget_1_asr_mean", 0), row.get("budget_1_asr_std", 0)
            )
            asr3 = format_percent(
                row.get("budget_3_asr_mean", 0), row.get("budget_3_asr_std", 0)
            )
            asr5 = format_percent(
                row.get("budget_5_asr_mean", 0), row.get("budget_5_asr_std", 0)
            )

            # Latency
            p50 = f"{row.get('latency_p50_ms_mean', 0):.2f}"
            p95 = f"{row.get('latency_p95_ms_mean', 0):.2f}"

            lines.append(
                f"{prot_display} & {strategy} "
                f"& {tpr1} & {tpr3} & {tpr5} "
                f"& {asr1} & {asr3} & {asr5} "
                f"& {p50} & {p95} \\\\"
            )
        lines.append("\\midrule")

    # Replace last midrule with bottomrule
    lines[-1] = "\\bottomrule"

    lines.extend([
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)


def generate_longschema_stress_table(
    summary_df: pd.DataFrame,
    trunc_df: pd.DataFrame,
) -> str:
    """Generate the Enterprise Truncation Stress Test table (longschema).

    Merges summary metrics with truncation retention stats.
    Columns: Protocol, Strategy, Prompt Truncated %, Prompt Retention Mean,
             ROC-AUC, PR-AUC, FPR@TPR90, Latency P50/P95.
    """
    # Derive strategy name from model name
    def _strategy(model: str) -> str:
        if "naive" in model:
            return "naive"
        if "keep_prompt" in model:
            return "keep_prompt"
        return model

    sdf = summary_df.copy()
    sdf["strategy"] = sdf["model"].apply(_strategy)

    # Merge with truncation stats on (protocol, strategy)
    merged = sdf.merge(
        trunc_df[["protocol", "strategy", "pct_prompt_truncated", "prompt_retention_ratio_mean"]],
        on=["protocol", "strategy"],
        how="left",
    )

    if merged.empty:
        return (
            "% Table 7: Longschema Stress Test — no data after merge\n"
        )

    lines = [
        "% Table 7: Enterprise Truncation Stress Test (Long Schema)",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Enterprise truncation stress test (4000-char schemas). "
        "Prompt truncation and model performance under naive vs keep\\_prompt strategies.}",
        "\\label{tab:longschema-stress}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll cc cccc cc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Truncation} "
        "& \\multicolumn{4}{c}{Performance} "
        "& \\multicolumn{2}{c}{Latency (ms)} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-8} \\cmidrule(lr){9-10}",
        "Protocol & Strategy "
        "& Trunc.\\% & Ret. Mean "
        "& ROC-AUC & PR-AUC & FPR@90 & FPR@95 "
        "& P50 & P95 \\\\",
        "\\midrule",
    ]

    for protocol in sorted(merged["protocol"].unique()):
        pdf = merged[merged["protocol"] == protocol].sort_values("strategy")
        for _, row in pdf.iterrows():
            strat = row["strategy"].replace("_", "\\_")
            prot_display = protocol.replace("_", "\\_")

            trunc_pct = f"{row.get('pct_prompt_truncated', 0)*100:.1f}\\%"
            ret_mean = f"{row.get('prompt_retention_ratio_mean', 0):.3f}"

            roc = format_metric(row.get("roc_auc_mean", 0), row.get("roc_auc_std", 0))
            pr = format_metric(row.get("pr_auc_mean", 0), row.get("pr_auc_std", 0))
            fpr90 = format_metric(
                row.get("fpr_at_tpr_90_mean", 0), row.get("fpr_at_tpr_90_std", 0)
            )
            fpr95 = format_metric(
                row.get("fpr_at_tpr_95_mean", 0), row.get("fpr_at_tpr_95_std", 0)
            )

            p50 = f"{row.get('latency_p50_ms_mean', 0):.2f}"
            p95 = f"{row.get('latency_p95_ms_mean', 0):.2f}"

            lines.append(
                f"{prot_display} & {strat} "
                f"& {trunc_pct} & {ret_mean} "
                f"& {roc} & {pr} & {fpr90} & {fpr95} "
                f"& {p50} & {p95} \\\\"
            )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"

    lines.extend([
        "\\end{tabular}}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from summary CSV")
    parser.add_argument(
        "--summary-csv", type=Path,
        default=Path("data/reports/experiments/summary.csv"),
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--truncation-stats-csv", type=Path, default=None,
        help="Path to truncation_stats.csv (enables longschema stress table)",
    )
    args = parser.parse_args()

    summary_path = args.summary_csv

    if not summary_path.exists():
        print(f"% Error: {summary_path} not found")
        print("% Run experiments first: python scripts/run_experiments.py")
        return
    
    df = pd.read_csv(summary_path)

    output_parts = []
    
    output_parts.append("%======================================================================")
    output_parts.append("% ToolShield Evaluation Results - LaTeX Tables")
    output_parts.append("% Generated by scripts/generate_latex_from_summary.py")
    output_parts.append("%======================================================================")
    output_parts.append("")
    output_parts.append("% Required packages:")
    output_parts.append("% \\usepackage{booktabs}")
    output_parts.append("")
    
    output_parts.append(generate_main_results_table(df))
    output_parts.append(generate_asr_table(df))
    output_parts.append(generate_budget_table(df))
    output_parts.append(generate_latency_table(df))
    output_parts.append(generate_generalization_gap_table(df))
    output_parts.append(generate_truncation_ablation_table(df))

    # Longschema stress table (if truncation stats provided)
    if args.truncation_stats_csv and args.truncation_stats_csv.exists():
        trunc_df = pd.read_csv(args.truncation_stats_csv)
        output_parts.append(generate_longschema_stress_table(df, trunc_df))

    output_text = "\n".join(output_parts)
    print(output_text)


if __name__ == "__main__":
    main()
