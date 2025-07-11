"""
Fixed FastPlot implementation for Experiment 1: DAG vs Vanilla vs CPDAG comparison.
Addresses: contours, font issues, heatmap colors, and file organization.
Updated for new CSV structure with single file containing all conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import sys
import fastplot
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from viz_config import setup_plotting, get_significance_marker, FONT_SIZES, DPI, CAUSAL_COLORS, METRIC_CONFIG, DAG_TYPE_COLORS

# Use centralized metrics configuration
METRICS = list(METRIC_CONFIG.keys())

# --- Generalized Output Directory Creation ---
SCRIPT_DIR = Path(__file__).resolve().parent

def create_output_directories(base_dir):
    """Create organized output directories inside the base directory."""
    base_dir = SCRIPT_DIR / base_dir
    base_dir.mkdir(exist_ok=True)
    formats = ['png', 'pdf']
    for fmt in formats:
        out_dir = base_dir / fmt
        out_dir.mkdir(exist_ok=True)
    return formats

# --- Generalized Analysis Pipeline ---
def run_full_analysis(df, output_base_dir, label):
    print(f"\n{'='*80}\nPROCESSING: {label.upper()}\n{'='*80}")
    formats = create_output_directories(output_base_dir)
    statistical_results = perform_statistical_tests(df)
    create_fastplot_visualizations(df, statistical_results, output_base_dir, formats)
    generate_report(df, statistical_results, output_base_dir, label)

# --- Data Loading for Each Setting ---
def load_and_filter_results(csv_path, strategy_value):
    """Load CSV and create aligned correct_dag/vanilla_worst/vanilla/cpdag metrics for each train_size/repetition pair."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter data based on new CSV structure
    dag_data = df[(df['algorithm'] == 'dag') & (df['graph_type'] == 'correct')].copy()
    vanilla_worst_data = df[(df['algorithm'] == 'vanilla') & (df['column_order_strategy'] == 'worst')].copy()
    vanilla_data = df[(df['algorithm'] == 'vanilla') & (df['column_order_strategy'] == strategy_value)].copy()
    cpdag_data = df[(df['algorithm'] == 'cpdag') & (df['graph_type'] == 'cpdag')].copy()
    
    if dag_data.empty:
        print(f"[WARNING] No 'dag' data found in {csv_path}")
        return None
    if vanilla_worst_data.empty:
        print(f"[WARNING] No 'vanilla worst' data found in {csv_path}")
        return None
    if vanilla_data.empty:
        print(f"[WARNING] No 'vanilla' data for strategy '{strategy_value}' found in {csv_path}")
        return None
    if cpdag_data.empty:
        print(f"[WARNING] No 'cpdag' data found in {csv_path}")
        return None

    # Align the four dataframes by train_size and repetition
    rows = []
    for (train_size, repetition), vanilla_group in vanilla_data.groupby(['train_size', 'repetition']):
        dag_match = dag_data[(dag_data['train_size'] == train_size) & (dag_data['repetition'] == repetition)]
        vanilla_worst_match = vanilla_worst_data[(vanilla_worst_data['train_size'] == train_size) & (vanilla_worst_data['repetition'] == repetition)]
        cpdag_match = cpdag_data[(cpdag_data['train_size'] == train_size) & (cpdag_data['repetition'] == repetition)]
        if dag_match.empty:
            print(f"[WARNING] No matching 'dag' data for train_size={train_size}, repetition={repetition}")
            continue
        if vanilla_worst_match.empty:
            print(f"[WARNING] No matching 'vanilla worst' data for train_size={train_size}, repetition={repetition}")
            continue
        if cpdag_match.empty:
            print(f"[WARNING] No matching 'cpdag' data for train_size={train_size}, repetition={repetition}")
            continue
        row = {'train_size': train_size, 'repetition': repetition}
        for metric in METRICS:
            row[f'{metric}_dag'] = dag_match[metric].iloc[0]
            row[f'{metric}_vanilla_worst'] = vanilla_worst_match[metric].iloc[0]
            row[f'{metric}_vanilla'] = vanilla_group[metric].iloc[0]
            row[f'{metric}_cpdag'] = cpdag_match[metric].iloc[0]
        rows.append(row)
    if not rows:
        print(f"[ERROR] Could not align any data for strategy '{strategy_value}'. Please check CSV integrity.")
        return None
    df_aligned = pd.DataFrame(rows)
    print(f"[INFO] Aligned {len(df_aligned)} quadruplets of dag/vanilla_worst/vanilla/cpdag data for strategy '{strategy_value}'.")
    return df_aligned

# --- Visualization and Report Functions (patched to accept output_base_dir) ---
def create_fastplot_visualizations(df, statistical_results, output_base_dir, formats):
    setup_plotting()
    train_sizes = sorted(df['train_size'].unique())
    # Use CAUSAL_COLORS for consistency with standard seaborn palette
    colors = ['#FF6B6B', CAUSAL_COLORS['without_dag'], CAUSAL_COLORS['with_dag'], DAG_TYPE_COLORS['cpdag']]  # Vanilla Worst (red), Vanilla, DAG, CPDAG
    print(f"\n{'='*60}\nCREATING VISUALIZATIONS in ./{output_base_dir}\n{'='*60}")
    for metric in METRICS:
        print(f"\nProcessing {metric}...")
        slug = METRIC_CONFIG[metric]['slug']
        for fmt in formats:
            out_dir = SCRIPT_DIR / output_base_dir / fmt
            # Boxplot
            try:
                fastplot.plot(
                    data=None,
                    path=str(out_dir / f'comparison_boxplot_{slug}.{fmt}'),
                    mode='callback',
                    callback=lambda plt, metric=metric: plot_single_boxplot(plt, df, metric, train_sizes, colors, statistical_results),
                    style='serif',
                    figsize=(8.0, 4.5),
                    dpi=DPI
                )
                print(f"  ✓ Boxplot saved as {out_dir}/comparison_boxplot_{slug}.{fmt}")
            except Exception as e:
                print(f"  ✗ Error creating boxplot for {metric} in {fmt}: {e}")
            # Effect size
            try:
                fastplot.plot(
                    data=None,
                    path=str(out_dir / f'effect_size_{slug}.{fmt}'),
                    mode='callback',
                    callback=lambda plt, metric=metric: plot_single_effect_size(plt, df, metric, train_sizes, colors, statistical_results),
                    style='serif',
                    figsize=(8.0, 4.5),
                    dpi=DPI
                )
                print(f"  ✓ Effect size saved as {out_dir}/effect_size_{slug}.{fmt}")
            except Exception as e:
                print(f"  ✗ Error creating effect size for {metric} in {fmt}: {e}")
            # Heatmap
            try:
                fastplot.plot(
                    data=None,
                    path=str(out_dir / f'heatmap_{slug}.{fmt}'),
                    mode='callback',
                    callback=lambda plt, metric=metric: plot_single_heatmap(plt, df, metric, train_sizes),
                    style='serif',
                    figsize=(8.0, 4.0),
                    dpi=DPI
                )
                print(f"  ✓ Heatmap saved as {out_dir}/heatmap_{slug}.{fmt}")
            except Exception as e:
                print(f"  ✗ Error creating heatmap for {metric} in {fmt}: {e}")
    print(f"\n[SUCCESS] All plots saved in organized directories: {', '.join([str(SCRIPT_DIR / output_base_dir / f) for f in formats])}")
    print("[INFO] PDF files are vector graphics perfect for publications!")

def generate_report(df, statistical_results, output_base_dir, label):
    report_lines = [
        f"EXPERIMENT 1: {label.upper()}",
        "=" * 70,
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Results: {len(df)}",
        f"Training Sizes: {sorted(df['train_size'].unique())}",
        f"Repetitions per size: {df.groupby('train_size').size().iloc[0] if not df.empty else 'N/A'}",
        "\nSUMMARY OF FINDINGS (DAG vs Vanilla vs CPDAG)",
        "-" * 70,
    ]
    for metric in METRICS:
        dag_mean = df[f'{metric}_dag'].mean()
        vanilla_worst_mean = df[f'{metric}_vanilla_worst'].mean()
        vanilla_mean = df[f'{metric}_vanilla'].mean()
        cpdag_mean = df[f'{metric}_cpdag'].mean()
        
        # Calculate improvements
        vanilla_vs_worst = (vanilla_worst_mean - vanilla_mean) / vanilla_worst_mean * 100 if vanilla_worst_mean != 0 else float('inf')
        dag_vs_vanilla = (vanilla_mean - dag_mean) / vanilla_mean * 100 if vanilla_mean != 0 else float('inf')
        cpdag_vs_vanilla = (vanilla_mean - cpdag_mean) / vanilla_mean * 100 if vanilla_mean != 0 else float('inf')
        dag_vs_cpdag = (cpdag_mean - dag_mean) / cpdag_mean * 100 if cpdag_mean != 0 else float('inf')
        
        stats_res = statistical_results.get(metric, {})
        title = METRIC_CONFIG[metric]['title']
        report_lines.extend([
            f"\nMETRIC: {title.upper()}",
            f"  - Vanilla Worst (Mean): {vanilla_worst_mean:.4f}",
            f"  - Vanilla (Mean):       {vanilla_mean:.4f}",
            f"  - Correct DAG (Mean):   {dag_mean:.4f}",
            f"  - CPDAG (Mean):         {cpdag_mean:.4f}",
            f"  - Vanilla vs Vanilla Worst: {vanilla_vs_worst:+.1f}% improvement",
            f"  - Correct DAG vs Vanilla:   {dag_vs_vanilla:+.1f}% improvement",
            f"  - CPDAG vs Vanilla:         {cpdag_vs_vanilla:+.1f}% improvement",
            f"  - Correct DAG vs CPDAG:     {dag_vs_cpdag:+.1f}% improvement"
        ])
        
        # Add significance markers
        for comparison in ['vanilla_vs_worst', 'dag_vs_vanilla', 'cpdag_vs_vanilla', 'dag_vs_cpdag']:
            comp_stats = stats_res.get(comparison, {})
            p_val = comp_stats.get('p_value')
            if p_val is not None:
                sig_marker = get_significance_marker(p_val)
                report_lines.append(f"  - {comparison.replace('_', ' vs ').title()}: {sig_marker} (p={p_val:.6f}, d={comp_stats.get('cohens_d', 0.0):.3f})")
    
    report_lines.extend([
        "\n\nDETAILED STATISTICAL ANALYSIS BY TRAINING SIZE",
        "=" * 70
    ])
    for metric in METRICS:
        title = METRIC_CONFIG[metric]['title']
        report_lines.extend([
            f"\n{title.upper()} - Statistical Tests",
            "-" * 50,
            "Train Size  Vanilla vs Worst  DAG vs Vanilla  CPDAG vs Vanilla  DAG vs CPDAG",
            "-" * 65
        ])
        for size in sorted(df['train_size'].unique()):
            size_data = df[df['train_size'] == size]
            row_str = f"{size:<11}"
            for comparison in ['vanilla_vs_worst', 'dag_vs_vanilla', 'cpdag_vs_vanilla', 'dag_vs_cpdag']:
                stats_res = statistical_results[metric]['by_size'].get(size, {}).get(comparison, {})
                if stats_res:
                    p_val = stats_res.get('p_value')
                    sig_marker = get_significance_marker(p_val) if p_val is not None else 'N/A'
                    row_str += f" {sig_marker:<15}"
                else:
                    row_str += f" {'N/A':<15}"
            report_lines.append(row_str)
            
    report_lines.extend([
        "\n\n",
        "FILES GENERATED",
        "-" * 70,
        "Visualizations saved in subfolders (png/, pdf/):",
        "- comparison_boxplot_[metric]",
        "- effect_size_[metric]", 
        "- heatmap_[metric]",
        "",
        "Report:",
        "- experiment_1_report.txt (this file)",
        "",
        "Statistical Notes:",
        "- Significance level: α = 0.05",
        "- Paired t-tests were used to compare conditions.",
        "- Cohen's d for paired samples is used as the effect size measure.",
        "- All metrics are 'lower is better'."
    ])
    report_path = SCRIPT_DIR / output_base_dir / "experiment_1_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[SUCCESS] Detailed report saved to: {report_path}")

def plot_single_boxplot(plt, df, metric, train_sizes, colors, statistical_results):
    """Create a single boxplot with proper academic styling for four conditions."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    
    # Prepare data in new order: Vanilla Worst, Vanilla, CPDAG, DAG
    data_to_plot = []
    positions = []
    for j, size in enumerate(train_sizes):
        size_data = df[df['train_size'] == size]
        data_to_plot.append(size_data[f'{metric}_vanilla_worst'].values)
        positions.append(j * 4)
        data_to_plot.append(size_data[f'{metric}_vanilla'].values)
        positions.append(j * 4 + 1)
        data_to_plot.append(size_data[f'{metric}_cpdag'].values)
        positions.append(j * 4 + 2)
        data_to_plot.append(size_data[f'{metric}_dag'].values)
        positions.append(j * 4 + 3)
    
    # Create boxplots
    bp = ax.boxplot(
        data_to_plot, positions=positions, widths=0.7, patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6, 'markeredgecolor': 'black'},
        boxprops={'linewidth': 1.0, 'edgecolor': 'black'},
        whiskerprops={'linewidth': 1.0, 'color': 'black'},
        capprops={'linewidth': 1.0, 'color': 'black'}
    )
    
    # Update colors to match new order using standard seaborn palette
    new_colors = ['#FF6B6B', CAUSAL_COLORS['without_dag'], DAG_TYPE_COLORS['cpdag'], CAUSAL_COLORS['with_dag']]  # Vanilla Worst (red), Vanilla, CPDAG, DAG
    for patch, pos in zip(bp['boxes'], positions):
        patch.set_facecolor(new_colors[pos % 4])
        patch.set_alpha(0.8)
    
    # Add mean markers
    for data, pos in zip(data_to_plot, positions):
        ax.scatter(pos, np.mean(data), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
    
    # Add significance markers
    for j, size in enumerate(train_sizes):
        stats_res = statistical_results[metric]['by_size'].get(size, {})
        if stats_res:
            # Check DAG vs Vanilla
            dag_vs_vanilla = stats_res.get('dag_vs_vanilla', {})
            if dag_vs_vanilla.get('p_value', 1) < 0.05:
                y_max = max(np.max(data_to_plot[j * 4 + 1]), np.max(data_to_plot[j * 4 + 3]))
                ax.text(j * 4 + 2, y_max * 1.05, '*', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', color='black')
            
            # Check CPDAG vs Vanilla
            cpdag_vs_vanilla = stats_res.get('cpdag_vs_vanilla', {})
            if cpdag_vs_vanilla.get('p_value', 1) < 0.05:
                y_max = max(np.max(data_to_plot[j * 4 + 1]), np.max(data_to_plot[j * 4 + 2]))
                ax.text(j * 4 + 1.5, y_max * 1.05, '*', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', color='black')
    
    # Academic styling - clean borders
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('black')
    
    # Grid
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    
    # Ticks
    ax.tick_params(axis='x', which='major', pad=5, labelsize=10, direction='in', length=4)
    ax.tick_params(axis='y', which='major', pad=4, labelsize=10, direction='in', length=4)
    
    # Labels and title
    title = METRIC_CONFIG[metric]['title']
    ax.set_xticks([(j * 4 + j * 4 + 3) / 2 for j in range(len(train_sizes))])
    ax.set_xticklabels(train_sizes, rotation=0, ha='center')
    ax.set_title(f"{title}\n(Lower is Better)", pad=10, fontsize=11)
    ax.set_xlabel('Training Set Size', labelpad=6, fontsize=10)
    ax.set_ylabel(title, labelpad=6, fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=new_colors[0], alpha=0.8, edgecolor='black', label='Vanilla Worst'),
        Patch(facecolor=new_colors[1], alpha=0.8, edgecolor='black', label='Vanilla'),
        Patch(facecolor=new_colors[2], alpha=0.8, edgecolor='black', label='CPDAG'),
        Patch(facecolor=new_colors[3], alpha=0.8, edgecolor='black', label='Correct DAG'),
        plt.Line2D([0], [0], marker='D', color='w', markeredgecolor='black', 
                  markerfacecolor='white', markersize=6, label='Mean', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, 
             fancybox=False, shadow=False, framealpha=1.0, edgecolor='black')

def plot_single_effect_size(plt, df, metric, train_sizes, colors, statistical_results):
    """Create effect size plot with academic styling for four conditions."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    
    # Calculate differences: Vanilla - Vanilla Worst, DAG - Vanilla, CPDAG - Vanilla, DAG - CPDAG
    differences_vanilla_vs_worst = [
        df[df['train_size'] == size][f'{metric}_vanilla'].values - 
        df[df['train_size'] == size][f'{metric}_vanilla_worst'].values 
        for size in train_sizes
    ]
    
    differences_dag_vs_vanilla = [
        df[df['train_size'] == size][f'{metric}_dag'].values - 
        df[df['train_size'] == size][f'{metric}_vanilla'].values 
        for size in train_sizes
    ]
    
    differences_cpdag_vs_vanilla = [
        df[df['train_size'] == size][f'{metric}_cpdag'].values - 
        df[df['train_size'] == size][f'{metric}_vanilla'].values 
        for size in train_sizes
    ]
    
    differences_dag_vs_cpdag = [
        df[df['train_size'] == size][f'{metric}_dag'].values - 
        df[df['train_size'] == size][f'{metric}_cpdag'].values 
        for size in train_sizes
    ]
    
    # Create boxplots for each comparison
    all_differences = differences_vanilla_vs_worst + differences_dag_vs_vanilla + differences_cpdag_vs_vanilla + differences_dag_vs_cpdag
    positions = []
    for i in range(len(train_sizes)):
        positions.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
    
    bp = ax.boxplot(
        all_differences, positions=positions, widths=0.7, patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6, 'markeredgecolor': 'black'},
        boxprops={'linewidth': 1.0, 'edgecolor': 'black'},
        whiskerprops={'linewidth': 1.0, 'color': 'black'},
        capprops={'linewidth': 1.0, 'color': 'black'}
    )
    
    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % 4])
        patch.set_alpha(0.8)
    
    # Add mean markers and significance
    for j, size in enumerate(train_sizes):
        # Vanilla vs Vanilla Worst
        diff_vanilla_worst = differences_vanilla_vs_worst[j]
        ax.scatter(j * 4, np.mean(diff_vanilla_worst), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
        
        # DAG vs Vanilla
        diff_dag_vanilla = differences_dag_vs_vanilla[j]
        ax.scatter(j * 4 + 1, np.mean(diff_dag_vanilla), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
        
        stats_res = statistical_results[metric]['by_size'].get(size, {}).get('dag_vs_vanilla', {})
        if stats_res and stats_res.get('p_value', 1) < 0.05:
            y_max = np.max(diff_dag_vanilla)
            ax.text(j * 4 + 1, y_max * 1.07, '*', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='black')
        
        # CPDAG vs Vanilla
        diff_cpdag_vanilla = differences_cpdag_vs_vanilla[j]
        ax.scatter(j * 4 + 2, np.mean(diff_cpdag_vanilla), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
        
        stats_res = statistical_results[metric]['by_size'].get(size, {}).get('cpdag_vs_vanilla', {})
        if stats_res and stats_res.get('p_value', 1) < 0.05:
            y_max = np.max(diff_cpdag_vanilla)
            ax.text(j * 4 + 2, y_max * 1.07, '*', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='black')
        
        # DAG vs CPDAG
        diff_dag_cpdag = differences_dag_vs_cpdag[j]
        ax.scatter(j * 4 + 3, np.mean(diff_dag_cpdag), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
        
        stats_res = statistical_results[metric]['by_size'].get(size, {}).get('dag_vs_cpdag', {})
        if stats_res and stats_res.get('p_value', 1) < 0.05:
            y_max = np.max(diff_dag_cpdag)
            ax.text(j * 4 + 3, y_max * 1.07, '*', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='black')
    
    # Academic styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('black')
    
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    ax.tick_params(axis='x', which='major', pad=5, labelsize=10, direction='in', length=4)
    ax.tick_params(axis='y', which='major', pad=4, labelsize=10, direction='in', length=4)
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8, zorder=0)
    
    # Labels - Use LaTeX-safe Delta symbol
    title = METRIC_CONFIG[metric]['title']
    ax.set_title(f'Effect of Causal Information on {title}', pad=10, fontsize=11)
    ax.set_ylabel(r'$\Delta$ ' + f'{title}\n(Difference from Vanilla)', 
                 labelpad=6, fontsize=10)
    ax.set_xlabel('Training Set Size', labelpad=6, fontsize=10)
    
    # Set x-ticks to show training sizes
    ax.set_xticks([(j * 4 + j * 4 + 3) / 2 for j in range(len(train_sizes))])
    ax.set_xticklabels(train_sizes, rotation=0, ha='center')
    
    # Add legend for the four comparisons
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.8, edgecolor='black', label='Vanilla - Vanilla Worst'),
        Patch(facecolor=colors[1], alpha=0.8, edgecolor='black', label='DAG - Vanilla'),
        Patch(facecolor=colors[2], alpha=0.8, edgecolor='black', label='CPDAG - Vanilla'),
        Patch(facecolor=colors[3], alpha=0.8, edgecolor='black', label='DAG - CPDAG'),
        plt.Line2D([0], [0], marker='D', color='w', markeredgecolor='black', 
                   markerfacecolor='white', markersize=6, label='Mean', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, 
             fancybox=False, shadow=False, framealpha=1.0, edgecolor='black')

def plot_single_heatmap(plt, df, metric, train_sizes):
    """Create heatmap with medians for four conditions."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    
    # Prepare data using MEDIANS for four conditions
    heatmap_data = []
    for condition in ['Vanilla Worst', 'DAG', 'Vanilla', 'CPDAG']:
        if condition == 'Vanilla Worst':
            row = [
                df[(df['train_size'] == size)][f'{metric}_vanilla_worst'].median()
                for size in train_sizes
            ]
        else:
            row = [
                df[(df['train_size'] == size)][f'{metric}_{condition.lower()}'].median()
                for size in train_sizes
            ]
        heatmap_data.append(row)
    
    # Use RdYlBu_r: Red=high values (bad), Blue=low values (good)
    title = METRIC_CONFIG[metric]['title']
    sns.heatmap(
        heatmap_data, 
        xticklabels=train_sizes, 
        yticklabels=['Vanilla Worst', 'DAG', 'Vanilla', 'CPDAG'],
        cbar_kws={'label': title}, 
        ax=ax,
        annot=True, 
        fmt=".3f", 
        linewidths=0.5, 
        linecolor="black", 
        square=False, 
        cmap='RdYlBu_r',
        cbar=True,
        annot_kws={'size': 9}
    )
    
    # Academic styling
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('black')
    
    # Updated title to reflect median usage
    ax.set_title(f'{title} Heatmap (Medians)', pad=10, fontsize=11)
    ax.set_xlabel('Training Set Size', labelpad=6, fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', which='major', pad=5, labelsize=10)
    ax.tick_params(axis='y', which='major', pad=4, labelsize=10, rotation=0)

def perform_statistical_tests(df):
    """Perform paired t-tests for each metric and training size."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (PAIRED T-TESTS)")
    print("=" * 60)
    
    results = {}
    
    for metric in METRICS:
        print(f"\n{metric.upper()}:")
        print("-" * 50)
        results[metric] = {'by_size': {}}
        
        # Overall comparisons (with Bonferroni correction)
        vanilla_worst_overall = df[f'{metric}_vanilla_worst']
        dag_overall = df[f'{metric}_dag']
        vanilla_overall = df[f'{metric}_vanilla']
        cpdag_overall = df[f'{metric}_cpdag']
        
        # Store overall p-values for Bonferroni correction
        overall_p_vals = []
        overall_comparisons = []
        
        # Vanilla vs Vanilla Worst
        valid_pairs_vanilla_worst = ~np.isnan(vanilla_overall) & ~np.isnan(vanilla_worst_overall)
        if np.sum(valid_pairs_vanilla_worst) >= 2:
            t_stat, p_val = stats.ttest_rel(vanilla_overall[valid_pairs_vanilla_worst], vanilla_worst_overall[valid_pairs_vanilla_worst])
            diff = vanilla_overall[valid_pairs_vanilla_worst] - vanilla_worst_overall[valid_pairs_vanilla_worst]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            overall_p_vals.append(p_val)
            overall_comparisons.append({
                'key': 'vanilla_vs_worst',
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff)
            })
        
        # DAG vs Vanilla
        valid_pairs_dag_vanilla = ~np.isnan(dag_overall) & ~np.isnan(vanilla_overall)
        if np.sum(valid_pairs_dag_vanilla) >= 2:
            t_stat, p_val = stats.ttest_rel(dag_overall[valid_pairs_dag_vanilla], vanilla_overall[valid_pairs_dag_vanilla])
            diff = dag_overall[valid_pairs_dag_vanilla] - vanilla_overall[valid_pairs_dag_vanilla]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            overall_p_vals.append(p_val)
            overall_comparisons.append({
                'key': 'dag_vs_vanilla',
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff)
            })
        
        # CPDAG vs Vanilla
        valid_pairs_cpdag_vanilla = ~np.isnan(cpdag_overall) & ~np.isnan(vanilla_overall)
        if np.sum(valid_pairs_cpdag_vanilla) >= 2:
            t_stat, p_val = stats.ttest_rel(cpdag_overall[valid_pairs_cpdag_vanilla], vanilla_overall[valid_pairs_cpdag_vanilla])
            diff = cpdag_overall[valid_pairs_cpdag_vanilla] - vanilla_overall[valid_pairs_cpdag_vanilla]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            overall_p_vals.append(p_val)
            overall_comparisons.append({
                'key': 'cpdag_vs_vanilla',
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff)
            })
        
        # DAG vs CPDAG
        valid_pairs_dag_cpdag = ~np.isnan(dag_overall) & ~np.isnan(cpdag_overall)
        if np.sum(valid_pairs_dag_cpdag) >= 2:
            t_stat, p_val = stats.ttest_rel(dag_overall[valid_pairs_dag_cpdag], cpdag_overall[valid_pairs_dag_cpdag])
            diff = dag_overall[valid_pairs_dag_cpdag] - cpdag_overall[valid_pairs_dag_cpdag]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            overall_p_vals.append(p_val)
            overall_comparisons.append({
                'key': 'dag_vs_cpdag',
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff)
            })
        
        # Store overall comparisons (no correction for planned comparisons)
        for comp in overall_comparisons:
            results[metric][comp['key']] = comp
            
            # Print with uncorrected p-values  
            if comp['key'] == 'vanilla_vs_worst':
                print(f"Vanilla vs Vanilla Worst: t={comp['t_stat']:.3f}, p={comp['p_value']:.6f} {get_significance_marker(comp['p_value'])}, d={comp['cohens_d']:.3f}")
            elif comp['key'] == 'dag_vs_vanilla':
                print(f"DAG vs Vanilla: t={comp['t_stat']:.3f}, p={comp['p_value']:.6f} {get_significance_marker(comp['p_value'])}, d={comp['cohens_d']:.3f}")
            elif comp['key'] == 'cpdag_vs_vanilla':
                print(f"CPDAG vs Vanilla: t={comp['t_stat']:.3f}, p={comp['p_value']:.6f} {get_significance_marker(comp['p_value'])}, d={comp['cohens_d']:.3f}")
            elif comp['key'] == 'dag_vs_cpdag':
                print(f"DAG vs CPDAG: t={comp['t_stat']:.3f}, p={comp['p_value']:.6f} {get_significance_marker(comp['p_value'])}, d={comp['cohens_d']:.3f}")

        # By training size (no correction for planned comparisons)
        print("By training size:")
        for size in sorted(df['train_size'].unique()):
            size_data = df[df['train_size'] == size]
            results[metric]['by_size'][size] = {}
            
            # Get all data for this size
            vanilla_worst = size_data[f'{metric}_vanilla_worst']
            dag = size_data[f'{metric}_dag']
            vanilla = size_data[f'{metric}_vanilla']
            cpdag = size_data[f'{metric}_cpdag']
            
            # Vanilla vs Vanilla Worst
            valid_pairs_vanilla_worst = ~np.isnan(vanilla) & ~np.isnan(vanilla_worst)
            if np.sum(valid_pairs_vanilla_worst) >= 2:
                t_stat, p_val = stats.ttest_rel(vanilla[valid_pairs_vanilla_worst], vanilla_worst[valid_pairs_vanilla_worst])
                diff = vanilla[valid_pairs_vanilla_worst] - vanilla_worst[valid_pairs_vanilla_worst]
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
                
                results[metric]['by_size'][size]['vanilla_vs_worst'] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'mean_diff': np.mean(diff)
                }
            
            # DAG vs Vanilla
            valid_pairs_dag_vanilla = ~np.isnan(dag) & ~np.isnan(vanilla)
            if np.sum(valid_pairs_dag_vanilla) >= 2:
                t_stat, p_val = stats.ttest_rel(dag[valid_pairs_dag_vanilla], vanilla[valid_pairs_dag_vanilla])
                diff = dag[valid_pairs_dag_vanilla] - vanilla[valid_pairs_dag_vanilla]
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
                
                results[metric]['by_size'][size]['dag_vs_vanilla'] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'mean_diff': np.mean(diff)
                }
            
            # CPDAG vs Vanilla
            valid_pairs_cpdag_vanilla = ~np.isnan(cpdag) & ~np.isnan(vanilla)
            if np.sum(valid_pairs_cpdag_vanilla) >= 2:
                t_stat, p_val = stats.ttest_rel(cpdag[valid_pairs_cpdag_vanilla], vanilla[valid_pairs_cpdag_vanilla])
                diff = cpdag[valid_pairs_cpdag_vanilla] - vanilla[valid_pairs_cpdag_vanilla]
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
                
                results[metric]['by_size'][size]['cpdag_vs_vanilla'] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'mean_diff': np.mean(diff)
                }
            
            # DAG vs CPDAG
            valid_pairs_dag_cpdag = ~np.isnan(dag) & ~np.isnan(cpdag)
            if np.sum(valid_pairs_dag_cpdag) >= 2:
                t_stat, p_val = stats.ttest_rel(dag[valid_pairs_dag_cpdag], cpdag[valid_pairs_dag_cpdag])
                diff = dag[valid_pairs_dag_cpdag] - cpdag[valid_pairs_dag_cpdag]
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
                
                results[metric]['by_size'][size]['dag_vs_cpdag'] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'mean_diff': np.mean(diff)
                }
            
            # Print results for this size
            vanilla_vs_worst = results[metric]['by_size'][size].get('vanilla_vs_worst', {})
            dag_vs_vanilla = results[metric]['by_size'][size].get('dag_vs_vanilla', {})
            cpdag_vs_vanilla = results[metric]['by_size'][size].get('cpdag_vs_vanilla', {})
            dag_vs_cpdag = results[metric]['by_size'][size].get('dag_vs_cpdag', {})
            
            print(f"  Size {size}:")
            if vanilla_vs_worst:
                print(f"    Vanilla vs Vanilla Worst: t={vanilla_vs_worst['t_stat']:.3f}, p={vanilla_vs_worst['p_value']:.6f} {get_significance_marker(vanilla_vs_worst['p_value'])}, d={vanilla_vs_worst['cohens_d']:.3f}")
            if dag_vs_vanilla:
                print(f"    DAG vs Vanilla: t={dag_vs_vanilla['t_stat']:.3f}, p={dag_vs_vanilla['p_value']:.6f} {get_significance_marker(dag_vs_vanilla['p_value'])}, d={dag_vs_vanilla['cohens_d']:.3f}")
            if cpdag_vs_vanilla:
                print(f"    CPDAG vs Vanilla: t={cpdag_vs_vanilla['t_stat']:.3f}, p={cpdag_vs_vanilla['p_value']:.6f} {get_significance_marker(cpdag_vs_vanilla['p_value'])}, d={cpdag_vs_vanilla['cohens_d']:.3f}")
            if dag_vs_cpdag:
                print(f"    DAG vs CPDAG: t={dag_vs_cpdag['t_stat']:.3f}, p={dag_vs_cpdag['p_value']:.6f} {get_significance_marker(dag_vs_cpdag['p_value'])}, d={dag_vs_cpdag['cohens_d']:.3f}")
            
    return results

# --- Main ---
def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    settings = [
        {
            'csv': script_dir / 'results_experiment_1.csv',
            'strategy': 'original',
            'output': 'original_plots',
            'label': 'original'
        },
        {
            'csv': script_dir / 'results_experiment_1.csv',
            'strategy': 'topological',
            'output': 'topological_plots',
            'label': 'topological'
        }
    ]
    for setting in settings:
        df = load_and_filter_results(setting['csv'], setting['strategy'])
        if df is not None:
            run_full_analysis(df, setting['output'], setting['label'])
        else:
            print(f"[INFO] Skipping {setting['label']} due to missing or invalid data.")
    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()