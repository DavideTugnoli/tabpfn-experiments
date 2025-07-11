"""
Experiment 1 Analysis: By Training Size - DAG vs Vanilla vs CPDAG Comparison.
Creates separate analysis and plots for each training size instead of aggregating.
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from viz_config import setup_plotting, get_significance_marker, FONT_SIZES, DPI, CAUSAL_COLORS, METRIC_CONFIG, DAG_TYPE_COLORS

# Use centralized metrics configuration
METRICS = list(METRIC_CONFIG.keys())
SCRIPT_DIR = Path(__file__).resolve().parent

def create_output_directories(base_dir, train_size):
    """Create organized output directories for a specific training size."""
    base_dir = SCRIPT_DIR / base_dir
    size_dir = base_dir / f"training_size_{train_size}"
    size_dir.mkdir(parents=True, exist_ok=True)
    formats = ['png', 'pdf']
    for fmt in formats:
        out_dir = size_dir / fmt
        out_dir.mkdir(exist_ok=True)
    return formats

def load_and_filter_results(csv_path, strategy_value, train_size):
    """Load CSV and create aligned correct_dag/vanilla_worst/vanilla/cpdag metrics for a specific training size."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df = df[df['train_size'] == train_size]
    if df.empty:
        print(f"[WARNING] No data found for training size {train_size} in {csv_path}")
        return None
    
    # Filter data based on new CSV structure
    dag_data = df[(df['algorithm'] == 'dag') & (df['graph_type'] == 'correct')].copy()
    vanilla_worst_data = df[(df['algorithm'] == 'vanilla') & (df['column_order_strategy'] == 'worst')].copy()
    vanilla_data = df[(df['algorithm'] == 'vanilla') & (df['column_order_strategy'] == strategy_value)].copy()
    cpdag_data = df[(df['algorithm'] == 'cpdag') & (df['graph_type'] == 'cpdag')].copy()
    
    if dag_data.empty:
        print(f"[WARNING] No 'dag' data found for training size {train_size}")
        return None
    if vanilla_worst_data.empty:
        print(f"[WARNING] No 'vanilla worst' data found for training size {train_size}")
        return None
    if vanilla_data.empty:
        print(f"[WARNING] No 'vanilla' data for strategy '{strategy_value}' and training size {train_size}")
        return None
    if cpdag_data.empty:
        print(f"[WARNING] No 'cpdag' data found for training size {train_size}")
        return None

    rows = []
    for repetition, vanilla_group in vanilla_data.groupby('repetition'):
        dag_match = dag_data[dag_data['repetition'] == repetition]
        vanilla_worst_match = vanilla_worst_data[vanilla_worst_data['repetition'] == repetition]
        cpdag_match = cpdag_data[cpdag_data['repetition'] == repetition]
        if dag_match.empty:
            print(f"[WARNING] No matching 'dag' data for repetition={repetition}")
            continue
        if vanilla_worst_match.empty:
            print(f"[WARNING] No matching 'vanilla worst' data for repetition={repetition}")
            continue
        if cpdag_match.empty:
            print(f"[WARNING] No matching 'cpdag' data for repetition={repetition}")
            continue
        row = {'train_size': train_size, 'repetition': repetition}
        for metric in METRICS:
            row[f'{metric}_dag'] = dag_match[metric].iloc[0]
            row[f'{metric}_vanilla_worst'] = vanilla_worst_match[metric].iloc[0]
            row[f'{metric}_vanilla'] = vanilla_group[metric].iloc[0]
            row[f'{metric}_cpdag'] = cpdag_match[metric].iloc[0]
        rows.append(row)
    if not rows:
        print(f"[ERROR] Could not align any data for training size {train_size}")
        return None
    df_aligned = pd.DataFrame(rows)
    print(f"[INFO] Aligned {len(df_aligned)} quadruplets for training size {train_size}")
    return df_aligned

def perform_statistical_tests(df, train_size):
    """Perform paired t-tests for each metric for a specific training size."""
    print(f"\n{'='*60}\nSTATISTICAL TESTS FOR TRAINING SIZE {train_size}\n{'='*60}")
    
    results = {}
    
    for metric in METRICS:
        print(f"\n{metric.upper()}:")
        print("-" * 50)
        results[metric] = {}
        
        vanilla_worst_data = df[f'{metric}_vanilla_worst']
        dag_data = df[f'{metric}_dag']
        vanilla_data = df[f'{metric}_vanilla']
        cpdag_data = df[f'{metric}_cpdag']
        
        # Vanilla vs Vanilla Worst
        valid_pairs_vanilla_worst = ~np.isnan(vanilla_data) & ~np.isnan(vanilla_worst_data)
        if np.sum(valid_pairs_vanilla_worst) >= 2:
            t_stat, p_val = stats.ttest_rel(vanilla_data[valid_pairs_vanilla_worst], vanilla_worst_data[valid_pairs_vanilla_worst])
            diff = vanilla_data[valid_pairs_vanilla_worst] - vanilla_worst_data[valid_pairs_vanilla_worst]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            results[metric]['vanilla_vs_worst'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff),
                'vanilla_mean': np.mean(vanilla_data[valid_pairs_vanilla_worst]),
                'vanilla_worst_mean': np.mean(vanilla_worst_data[valid_pairs_vanilla_worst])
            }
            print(f"  Vanilla vs Vanilla Worst: t={t_stat:.3f}, p={p_val:.6f} {get_significance_marker(p_val)}, d={cohens_d:.3f}")
        
        # DAG vs Vanilla
        valid_pairs_dag_vanilla = ~np.isnan(dag_data) & ~np.isnan(vanilla_data)
        if np.sum(valid_pairs_dag_vanilla) >= 2:
            t_stat, p_val = stats.ttest_rel(dag_data[valid_pairs_dag_vanilla], vanilla_data[valid_pairs_dag_vanilla])
            diff = dag_data[valid_pairs_dag_vanilla] - vanilla_data[valid_pairs_dag_vanilla]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            results[metric]['dag_vs_vanilla'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff),
                'dag_mean': np.mean(dag_data[valid_pairs_dag_vanilla]),
                'vanilla_mean': np.mean(vanilla_data[valid_pairs_dag_vanilla])
            }
            print(f"  DAG vs Vanilla: t={t_stat:.3f}, p={p_val:.6f} {get_significance_marker(p_val)}, d={cohens_d:.3f}")
        
        # CPDAG vs Vanilla
        valid_pairs_cpdag_vanilla = ~np.isnan(cpdag_data) & ~np.isnan(vanilla_data)
        if np.sum(valid_pairs_cpdag_vanilla) >= 2:
            t_stat, p_val = stats.ttest_rel(cpdag_data[valid_pairs_cpdag_vanilla], vanilla_data[valid_pairs_cpdag_vanilla])
            diff = cpdag_data[valid_pairs_cpdag_vanilla] - vanilla_data[valid_pairs_cpdag_vanilla]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            results[metric]['cpdag_vs_vanilla'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff),
                'cpdag_mean': np.mean(cpdag_data[valid_pairs_cpdag_vanilla]),
                'vanilla_mean': np.mean(vanilla_data[valid_pairs_cpdag_vanilla])
            }
            print(f"  CPDAG vs Vanilla: t={t_stat:.3f}, p={p_val:.6f} {get_significance_marker(p_val)}, d={cohens_d:.3f}")
        
        # DAG vs CPDAG
        valid_pairs_dag_cpdag = ~np.isnan(dag_data) & ~np.isnan(cpdag_data)
        if np.sum(valid_pairs_dag_cpdag) >= 2:
            t_stat, p_val = stats.ttest_rel(dag_data[valid_pairs_dag_cpdag], cpdag_data[valid_pairs_dag_cpdag])
            diff = dag_data[valid_pairs_dag_cpdag] - cpdag_data[valid_pairs_dag_cpdag]
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
            
            results[metric]['dag_vs_cpdag'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'mean_diff': np.mean(diff),
                'dag_mean': np.mean(dag_data[valid_pairs_dag_cpdag]),
                'cpdag_mean': np.mean(cpdag_data[valid_pairs_dag_cpdag])
            }
            print(f"  DAG vs CPDAG: t={t_stat:.3f}, p={p_val:.6f} {get_significance_marker(p_val)}, d={cohens_d:.3f}")
            
    return results

def plot_single_boxplot(plt, df, metric, train_size, statistical_results):
    """Create a single boxplot for a specific training size with four conditions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data in new order: Vanilla Worst, Vanilla, CPDAG, DAG
    data_to_plot = [df[f'{metric}_vanilla_worst'].values, df[f'{metric}_vanilla'].values, df[f'{metric}_cpdag'].values, df[f'{metric}_dag'].values]
    colors = ['#FF6B6B', CAUSAL_COLORS['without_dag'], DAG_TYPE_COLORS['cpdag'], CAUSAL_COLORS['with_dag']]  # Vanilla Worst (red), Vanilla, CPDAG, DAG
    
    # Create boxplots
    bp = ax.boxplot(
        data_to_plot, widths=0.7, patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6, 'markeredgecolor': 'black'},
        boxprops={'linewidth': 1.0, 'edgecolor': 'black'},
        whiskerprops={'linewidth': 1.0, 'color': 'black'},
        capprops={'linewidth': 1.0, 'color': 'black'}
    )
    
    # Color the boxes in new order
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add mean markers
    for i, data in enumerate(data_to_plot):
        ax.scatter(i + 1, np.mean(data), marker='D', s=25, color='white', 
                  edgecolor='black', linewidth=1.0, zorder=5)
    
    # Add significance markers
    stats_res = statistical_results.get(metric, {})
    # Significance stars removed for cleaner presentation - significance reported in text reports only

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
    
    # Labels and title
    title = METRIC_CONFIG[metric]['title']
    ax.set_title(f"{title} - Training Size {train_size}\n(Lower is Better)", 
                pad=10, fontsize=FONT_SIZES['title'])
    ax.set_xlabel('Condition', labelpad=6, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(title, labelpad=6, fontsize=FONT_SIZES['label'])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Vanilla Worst', 'Vanilla', 'CPDAG', 'Correct DAG'])
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.8, edgecolor='black', label='Vanilla Worst'),
        Patch(facecolor=colors[1], alpha=0.8, edgecolor='black', label='Vanilla'),
        Patch(facecolor=colors[2], alpha=0.8, edgecolor='black', label='CPDAG'),
        Patch(facecolor=colors[3], alpha=0.8, edgecolor='black', label='Correct DAG'),
        plt.Line2D([0], [0], marker='D', color='w', markeredgecolor='black', 
                   markerfacecolor='white', markersize=6, label='Mean', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZES['legend'], 
             frameon=True, fancybox=False, shadow=False, framealpha=1.0, edgecolor='black')

def create_visualizations_for_size(df, statistical_results, train_size, output_base_dir, formats):
    """Create all visualizations for a specific training size."""
    setup_plotting()
    print(f"\n{'='*60}\nCREATING VISUALIZATIONS FOR TRAINING SIZE {train_size}\n{'='*60}")
    
    for metric in METRICS:
        print(f"\nProcessing {metric}...")
        slug = METRIC_CONFIG[metric]['slug']
        for fmt in formats:
            out_dir = SCRIPT_DIR / output_base_dir / f"training_size_{train_size}" / fmt
            try:
                fastplot.plot(
                    data=None,
                    path=str(out_dir / f'boxplot_{slug}_size_{train_size}.{fmt}'),
                    mode='callback',
                    callback=lambda plt, m=metric: plot_single_boxplot(plt, df, m, train_size, statistical_results),
                    style='serif',
                    figsize=(7, 5),
                    dpi=DPI
                )
                print(f"  ✓ Saved {out_dir}/boxplot_{slug}_size_{train_size}.{fmt}")
            except Exception as e:
                print(f"  ✗ Error creating plot for {metric} in {fmt}: {e}")

def generate_report(df, statistical_results, train_size, output_base_dir, label):
    """Generate report for a specific training size."""
    report_lines = [
        f"EXPERIMENT 1: {label.upper()} - TRAINING SIZE {train_size}",
        "=" * 70,
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Training Size: {train_size}",
        f"Number of Repetitions: {len(df)}",
        "\nSUMMARY OF FINDINGS (DAG vs Vanilla vs CPDAG)",
        "-" * 70,
    ]
    
    for metric in METRICS:
        stats_res = statistical_results.get(metric, {})
        title = METRIC_CONFIG[metric]['title']
        
        # Calculate means
        vanilla_worst_mean = df[f'{metric}_vanilla_worst'].mean()
        dag_mean = df[f'{metric}_dag'].mean()
        vanilla_mean = df[f'{metric}_vanilla'].mean()
        cpdag_mean = df[f'{metric}_cpdag'].mean()
        
        # Calculate improvements
        vanilla_vs_worst = (vanilla_worst_mean - vanilla_mean) / vanilla_worst_mean * 100 if vanilla_worst_mean != 0 else float('inf')
        dag_vs_vanilla = (vanilla_mean - dag_mean) / vanilla_mean * 100 if vanilla_mean != 0 else float('inf')
        cpdag_vs_vanilla = (vanilla_mean - cpdag_mean) / vanilla_mean * 100 if vanilla_mean != 0 else float('inf')
        dag_vs_cpdag = (cpdag_mean - dag_mean) / cpdag_mean * 100 if cpdag_mean != 0 else float('inf')
        
        report_lines.extend([
            f"\nMETRIC: {title.upper()}",
            f"  - Vanilla Worst (Mean): {vanilla_worst_mean:.4f}",
            f"  - Vanilla (Mean):       {vanilla_mean:.4f}",
            f"  - DAG (Mean):           {dag_mean:.4f}",
            f"  - CPDAG (Mean):         {cpdag_mean:.4f}",
            f"  - Vanilla vs Vanilla Worst: {vanilla_vs_worst:+.1f}% improvement",
            f"  - DAG vs Vanilla:           {dag_vs_vanilla:+.1f}% improvement",
            f"  - CPDAG vs Vanilla:         {cpdag_vs_vanilla:+.1f}% improvement",
            f"  - DAG vs CPDAG:             {dag_vs_cpdag:+.1f}% improvement"
        ])
        
        # Add significance markers
        for comparison in ['vanilla_vs_worst', 'dag_vs_vanilla', 'cpdag_vs_vanilla', 'dag_vs_cpdag']:
            comp_stats = stats_res.get(comparison, {})
            if comp_stats:
                p_val = comp_stats.get('p_value')
                if p_val is not None:
                    sig_marker = get_significance_marker(p_val)
                    report_lines.append(f"  - {comparison.replace('_', ' vs ').title()}: {sig_marker} (p={p_val:.6f}, d={comp_stats.get('cohens_d', 0.0):.3f})")
    
    report_lines.extend([
        "\n\nSTATISTICAL NOTES",
        "-" * 70,
        "- Paired t-tests used to compare DAG vs Vanilla, CPDAG vs Vanilla, and DAG vs CPDAG conditions",
        "- Cohen's d for paired samples used as effect size measure",
        "- All metrics are 'lower is better'",
        f"- Analysis performed for training size {train_size} only"
    ])
    
    report_path = SCRIPT_DIR / output_base_dir / f"training_size_{train_size}" / f"experiment_1_report_size_{train_size}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[SUCCESS] Report saved to: {report_path}")

def analyze_single_training_size(csv_path, strategy_value, output_base_dir, label, train_size):
    """Analyze a single training size."""
    print(f"\n{'='*80}\nPROCESSING: {label.upper()} - TRAINING SIZE {train_size}\n{'='*80}")
    
    df = load_and_filter_results(csv_path, strategy_value, train_size)
    if df is None:
        print(f"[INFO] Skipping training size {train_size} due to missing or invalid data.")
        return
    
    formats = create_output_directories(output_base_dir, train_size)
    statistical_results = perform_statistical_tests(df, train_size)
    create_visualizations_for_size(df, statistical_results, train_size, output_base_dir, formats)
    generate_report(df, statistical_results, train_size, output_base_dir, label)

def main():
    """Main execution function."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    settings = [
        {
            'csv': script_dir / 'results_experiment_1.csv',
            'strategy': 'original',
            'output': 'by_training_size/original',
            'label': 'original'
        },
        {
            'csv': script_dir / 'results_experiment_1.csv',
            'strategy': 'topological',
            'output': 'by_training_size/topological',
            'label': 'topological'
        }
    ]
    
    # Get all available training sizes from the first dataset
    df_sample = pd.read_csv(settings[0]['csv'])
    train_sizes = sorted(df_sample['train_size'].unique())
    print(f"[INFO] Found training sizes: {train_sizes}")
    
    for setting in settings:
        for train_size in train_sizes:
            analyze_single_training_size(
                setting['csv'], 
                setting['strategy'], 
                setting['output'], 
                setting['label'], 
                train_size
            )
    
    print("\n[ALL DONE] - Analysis completed for all training sizes")

if __name__ == "__main__":
    main()