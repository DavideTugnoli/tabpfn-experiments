"""
Fixed visualization configuration for macOS with proper font handling.
Uses RdYlBu_r colormap for better visual interpretation.
Includes STRATEGY_ORDER for consistent experiment 2 ordering.
Updated with standard seaborn palette for consistent colors across experiments.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import platform
import warnings
warnings.filterwarnings('ignore')

# Font and display settings
FONT_SIZES = {
    'title': 11,
    'label': 10,
    'legend': 9,
    'tick': 10,
    'significance': 10,
    'suptitle': 12
}

DPI = 300
FIG_SIZE_TRIPLE = (15, 4)

# --- Centralized Configuration for Metrics and Strategies ---

# Defines the display name and filename slug for each metric column in the CSV
METRIC_CONFIG = {
    'mean_corr_difference': {
        'title': 'Mean Correlation Difference',
        'slug': 'mean_corr_diff'
    },
    'max_corr_difference': {
        'title': 'Max Correlation Difference',
        'slug': 'max_corr_diff'
    },
    'propensity_metrics_avg_pMSE': {
        'title': 'Propensity MSE',
        'slug': 'propensity_mse'
    },
    'k_marginal_tvd': {
        'title': 'K-Marginal TVD',
        'slug': 'kmarginal'
    }
}

# Strategy ordering for Experiment 2 consistency
STRATEGY_ORDER = ['original', 'topological', 'worst', 'random']

# Standard seaborn palette for consistent colors across experiments
# Using seaborn's "Set2" palette which is colorblind-friendly and distinct
SEABORN_PALETTE = sns.color_palette("Set2", 8)

# Consistent color mapping for strategies (Experiment 2)
STRATEGY_COLORS = {
    'original': SEABORN_PALETTE[0],    # Blue
    'topological': SEABORN_PALETTE[1], # Orange  
    'worst': SEABORN_PALETTE[2],       # Green
    'random': SEABORN_PALETTE[3]       # Red
}

# Consistent color mapping for DAG types (Experiment 3)
# Using the same palette to ensure no color repetitions
DAG_TYPE_COLORS = {
    'vanilla': SEABORN_PALETTE[0],       # Blue - baseline (Vanilla)
    'dag': SEABORN_PALETTE[1],           # Orange - correct DAG
    'cpdag': SEABORN_PALETTE[2],         # Green - CPDAG
    'wrong_parents': SEABORN_PALETTE[3], # Red - problematic
    'missing_edges': SEABORN_PALETTE[4], # Purple - missing edges
    'extra_edges': SEABORN_PALETTE[5],   # Brown - extra edges
    'disconnected': SEABORN_PALETTE[6]   # Pink - disconnected
}

# Consistent color mapping for causal vs non-causal (Experiment 1) 
# Using the same palette for consistency
CAUSAL_COLORS = {
    'with_dag': SEABORN_PALETTE[1],     # Orange - with causal structure (DAG)
    'without_dag': SEABORN_PALETTE[0]   # Blue - without causal structure (Vanilla)
}

# Heatmap configuration with RdYlBu_r (Red=bad, Blue=good)
HEATMAP_CONFIG = {
    'annot': True,
    'fmt': '.3f',
    'linewidths': 0.5,
    'linecolor': 'black',
    'square': False,
    'cmap': 'RdYlBu_r',  # Red=high values (worse), Blue=low values (better)
    'cbar': True,
    'annot_kws': {'size': 9}
}

def setup_plotting():
    """Setup matplotlib for publication-quality plots with macOS compatibility and minimal academic style."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    import platform
    import warnings
    warnings.filterwarnings('ignore')

    # Reset to defaults first
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Detect system and configure fonts accordingly
    system = platform.system()

    if system == 'Darwin':  # macOS
        font_config = {
            'font.family': 'serif',
            'font.serif': [
                'Times New Roman',
                'Times',
                'DejaVu Serif',
                'serif'
            ],
            'font.sans-serif': [
                'Helvetica',
                'Arial',
                'DejaVu Sans',
                'sans-serif'
            ],
            'font.size': FONT_SIZES['tick'],
            'mathtext.fontset': 'dejavuserif',
            'mathtext.default': 'regular'
        }
    else:
        font_config = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': FONT_SIZES['tick'],
            'mathtext.fontset': 'dejavuserif',
            'mathtext.default': 'regular'
        }
    mpl.rcParams.update(font_config)

    # Minimal academic style
    academic_style = {
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'axes.axisbelow': True,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.labelsize': FONT_SIZES['label'],
        'axes.titlesize': FONT_SIZES['title'],
        'grid.color': 'gray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': False,  # No box around legend
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'legend.borderpad': 0.4,
        'legend.columnspacing': 2.0,
        'legend.handlelength': 2.0,
        'legend.handletextpad': 0.8,
        'legend.labelspacing': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        'patch.linewidth': 1.0,
        'patch.edgecolor': 'black',
        'errorbar.capsize': 3,
        'text.color': 'black',
    }
    mpl.rcParams.update(academic_style)

    # Seaborn style for grid only on y-axis
    sns.set_style("whitegrid", {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid.axis': 'y',
        'axes.grid': True
    })
    
    # Set the standard seaborn palette
    sns.set_palette(SEABORN_PALETTE)
    
    print(f"[INFO] Academic plotting style configured for {system}")
    print(f"[INFO] DPI set to {DPI} for high-quality output")
    print(f"[INFO] Heatmap colormap: {HEATMAP_CONFIG['cmap']} (Red=worse, Blue=better)")
    print(f"[INFO] Using standard seaborn 'Set2' palette for consistent colors")
    print(f"[INFO] Strategy order: {STRATEGY_ORDER}")

def get_significance_marker(p_value):
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def save_plot_info():
    """Save information about the plotting configuration."""
    info = {
        'font_sizes': FONT_SIZES,
        'dpi': DPI,
        'palette': 'Set2',
        'heatmap_cmap': HEATMAP_CONFIG['cmap'],
        'strategy_order': STRATEGY_ORDER,
        'dag_type_colors': {k: str(v) for k, v in DAG_TYPE_COLORS.items()},
        'causal_colors': {k: str(v) for k, v in CAUSAL_COLORS.items()},
        'strategy_colors': {k: str(v) for k, v in STRATEGY_COLORS.items()}
    }
    return info