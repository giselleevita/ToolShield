#!/usr/bin/env python3
"""Generate thesis plots from experiment results.

Creates two "killer plots":
1. Generalization Gap Plot - shows FPR increase under distribution shift
2. Budget Trade-off Curve - shows ASR vs FPR budget

Usage:
    python scripts/generate_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def plot_generalization_gap(df: pd.DataFrame, output_path: Path) -> None:
    """Create the generalization gap bar plot.
    
    Shows ΔFPR@TPR(0.90) between random and holdout splits for each model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get data
    random_df = df[df['protocol'] == 'S_random'].set_index('model')
    holdout_df = df[df['protocol'] == 'S_attack_holdout'].set_index('model')
    
    models = []
    gaps = []
    random_vals = []
    holdout_vals = []
    
    for model in random_df.index:
        if model in holdout_df.index:
            models.append(model.replace('_', ' ').title())
            fpr_random = random_df.loc[model, 'fpr_at_tpr_90_mean']
            fpr_holdout = holdout_df.loc[model, 'fpr_at_tpr_90_mean']
            random_vals.append(fpr_random)
            holdout_vals.append(fpr_holdout)
            gaps.append(fpr_holdout - fpr_random)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Grouped bars
    bars1 = ax.bar(x - width/2, random_vals, width, label='S_random', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, holdout_vals, width, label='S_attack_holdout', color='#e74c3c', alpha=0.8)
    
    # Add gap annotations
    for i, (r, h, g) in enumerate(zip(random_vals, holdout_vals, gaps)):
        if g > 0.01:
            ax.annotate(f'+{g:.1%}', 
                       xy=(i + width/2, h), 
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       color='#c0392b',
                       fontweight='bold')
    
    ax.set_ylabel('FPR @ TPR=0.90')
    ax.set_xlabel('Model')
    ax.set_title('Generalization Gap: Random vs Attack-Holdout Split')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    
    # Add a horizontal line at common FPR budgets
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='5% budget')
    ax.axhline(y=0.03, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='3% budget')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_budget_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    """Create the budget trade-off curve.
    
    Shows ASR (or TPR) vs FPR budget for each protocol.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    budgets = [1, 3, 5]
    budget_labels = ['1%', '3%', '5%']
    
    # Plot 1: TPR at budget by protocol (for tfidf_lr model)
    colors = {'S_random': '#3498db', 'S_attack_holdout': '#e74c3c'}
    markers = {'S_random': 'o', 'S_attack_holdout': 's'}
    
    for protocol in df['protocol'].unique():
        protocol_df = df[df['protocol'] == protocol]
        tfidf_row = protocol_df[protocol_df['model'] == 'tfidf_lr']
        
        if len(tfidf_row) > 0:
            row = tfidf_row.iloc[0]
            tpr_values = [
                row.get('budget_1_tpr_mean', 0),
                row.get('budget_3_tpr_mean', 0),
                row.get('budget_5_tpr_mean', 0),
            ]
            
            ax1.plot(budgets, tpr_values, 
                    marker=markers.get(protocol, 'o'),
                    label=protocol.replace('_', ' '),
                    color=colors.get(protocol, 'gray'),
                    linewidth=2,
                    markersize=8)
    
    ax1.set_xlabel('FPR Budget (%)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.set_title('TPR at FPR Budget (TF-IDF Model)')
    ax1.set_xticks(budgets)
    ax1.set_xticklabels(budget_labels)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ASR at budget by protocol
    for protocol in df['protocol'].unique():
        protocol_df = df[df['protocol'] == protocol]
        tfidf_row = protocol_df[protocol_df['model'] == 'tfidf_lr']
        
        if len(tfidf_row) > 0:
            row = tfidf_row.iloc[0]
            asr_values = [
                row.get('budget_1_asr_mean', 0),
                row.get('budget_3_asr_mean', 0),
                row.get('budget_5_asr_mean', 0),
            ]
            
            ax2.plot(budgets, asr_values,
                    marker=markers.get(protocol, 'o'),
                    label=protocol.replace('_', ' '),
                    color=colors.get(protocol, 'gray'),
                    linewidth=2,
                    markersize=8)
    
    ax2.set_xlabel('FPR Budget (%)')
    ax2.set_ylabel('Attack Success Rate (ASR)')
    ax2.set_title('ASR at FPR Budget (TF-IDF Model)')
    ax2.set_xticks(budgets)
    ax2.set_xticklabels(budget_labels)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for the key insight
    ax2.annotate('Lower is better\n(attacks blocked)',
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                fontsize=9,
                ha='right',
                va='top',
                style='italic',
                color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_model_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create a model comparison radar/spider chart alternative as bar chart."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['roc_auc_mean', 'pr_auc_mean']
    metric_labels = ['ROC-AUC', 'PR-AUC']
    
    # Filter to S_random for cleaner comparison
    df_random = df[df['protocol'] == 'S_random']
    
    models = df_random['model'].tolist()
    x = np.arange(len(metrics))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, model in enumerate(models):
        row = df_random[df_random['model'] == model].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model.replace('_', ' ').title(), color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on S_random Split')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    summary_path = Path("data/reports/experiments/summary.csv")
    output_dir = Path("data/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        print("Run experiments first: python scripts/run_experiments.py")
        return
    
    df = pd.read_csv(summary_path)
    
    print("Generating thesis plots...")
    print()
    
    # 1. Generalization Gap Plot
    plot_generalization_gap(df, output_dir / "generalization_gap.png")
    
    # 2. Budget Trade-off Curves
    plot_budget_tradeoff(df, output_dir / "budget_tradeoff.png")
    
    # 3. Model Comparison (bonus)
    plot_model_comparison(df, output_dir / "model_comparison.png")
    
    print()
    print("All plots generated!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
