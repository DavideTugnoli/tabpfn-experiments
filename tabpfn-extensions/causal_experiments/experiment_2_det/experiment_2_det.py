"""
Experiment 2 IMPROVED: Column ordering effects on vanilla TabPFN.

This experiment tests whether column ordering affects synthetic data quality
when TabPFN uses its implicit autoregressive mechanism (no DAG provided).

Usage:
    python experiment_2_det_improved.py                    # Run full experiment
    python experiment_2_det_improved.py --no-resume       # Start fresh
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import argparse
import json
import random
import hashlib
from collections import OrderedDict

# Environment setup
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
import numpy as np

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports - cleaned up
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, print_dag_info
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Centralized default config
DEFAULT_CONFIG = {
    # Experimental design
    'train_sizes': [20, 50, 100, 200, 500],
    'ordering_strategies': ['original', 'topological', 'worst', 'random'],
    'n_repetitions': 10,
    'test_size': 2000,
    
    # Model parameters
    'n_estimators': 3,
    'n_permutations': 3,
    
    # Data configuration
    'include_categorical': False,
    'random_seed_base': 42,
    
    # Evaluation metrics
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
}

# Preferred order for result columns
PREFERRED_ORDER = [
    'algorithm', 'train_size', 'repetition', 'seed', 'categorical', 
    'column_order_strategy', 'column_order'
]

SAVE_DATA_SAMPLES = True  # Set to True to save data_samples for debugging

def hash_array(arr: np.ndarray) -> str:
    """Generate hash for data integrity checking."""
    return hashlib.md5(arr.tobytes()).hexdigest()

def setup_deterministic_mode(seed: int) -> None:
    """Setup deterministic mode for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # For older PyTorch versions

def create_tabpfn_model(config: Dict[str, Any]) -> unsupervised.TabPFNUnsupervisedModel:
    """Create and configure TabPFN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    return unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

def clean_gpu_memory() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_data_integrity(X_train: np.ndarray, X_test: np.ndarray, 
                           train_size: int, repetition: int, 
                           hash_check_dict: Dict[Tuple[int, int], Tuple[str, str]]) -> None:
    """Validate data integrity using hashes."""
    train_hash = hash_array(X_train)
    test_hash = hash_array(X_test)
    
    key = (train_size, repetition)
    if key in hash_check_dict:
        prev_train_hash, prev_test_hash = hash_check_dict[key]
        if prev_train_hash != train_hash or prev_test_hash != test_hash:
            raise RuntimeError(
                f"Data integrity check failed for train_size={train_size}, "
                f"repetition={repetition}"
            )
    else:
        hash_check_dict[key] = (train_hash, test_hash)

def evaluate_synthetic_data_metrics(X_test: np.ndarray, X_synth: np.ndarray, 
                                   col_names: List[str], categorical_cols: List[int]) -> Dict[str, Any]:
    """Evaluate synthetic data quality metrics."""
    evaluator = FaithfulDataEvaluator()
    cat_col_names = [col_names[i] for i in categorical_cols] if categorical_cols else []
    
    return evaluator.evaluate(
        pd.DataFrame(X_test, columns=col_names),
        pd.DataFrame(X_synth, columns=col_names),
        categorical_columns=cat_col_names if cat_col_names else None,
        k_for_kmarginal=2
    )

def build_result_row(base_info: Dict[str, Any], metrics: Dict[str, Any], 
                    config: Dict[str, Any]) -> OrderedDict:
    """Build result row with consistent column ordering."""
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    
    # Build ordered row
    row = OrderedDict()
    for k in PREFERRED_ORDER:
        row[k] = base_info.get(k, '')
    for k in flat_metrics:
        row[k] = flat_metrics[k]
    
    return row

def save_data_samples(X_train: np.ndarray, X_test: np.ndarray, X_synth: np.ndarray,
                     col_names: List[str], data_samples_dir: Path, file_prefix: str) -> None:
    """Save data samples for debugging if enabled."""
    if SAVE_DATA_SAMPLES and data_samples_dir:
        pd.DataFrame(X_train, columns=col_names).head(10).to_csv(
            data_samples_dir / f"{file_prefix}_train.csv", index=False)
        pd.DataFrame(X_test, columns=col_names).head(10).to_csv(
            data_samples_dir / f"{file_prefix}_test.csv", index=False)
        pd.DataFrame(X_synth, columns=col_names).head(10).to_csv(
            data_samples_dir / f"{file_prefix}_synth.csv", index=False)

def validate_and_prepare_column_orderings(config: Dict[str, Any], correct_dag: Dict[int, List[int]]) -> Dict[str, List[int]]:
    """Validate and prepare all column orderings upfront."""
    available_orderings = get_ordering_strategies(correct_dag)
    orders = {}
    
    for strategy in config['ordering_strategies']:
        if strategy not in available_orderings:
            raise ValueError(
                f"Unknown ordering strategy: {strategy}. "
                f"Available: {list(available_orderings.keys())}"
            )
        orders[strategy] = available_orderings[strategy]
        print(f"Pre-calculated column order for {strategy}: {orders[strategy]}")
    
    return orders

def run_vanilla_tabpfn_with_ordering(X_train: np.ndarray, X_test: np.ndarray, 
                                    col_names: List[str], categorical_cols: List[int],
                                    column_order: List[int], order_strategy: str, 
                                    config: Dict[str, Any], seed: int, train_size: int, 
                                    repetition: int, data_samples_dir: Optional[Path] = None) -> OrderedDict:
    """Run vanilla TabPFN with specific column ordering."""
    # Reorder data and metadata
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, column_order
    )
    
    # Create and configure model
    model = create_tabpfn_model(config)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    
    # Train and generate
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], 
        n_permutations=config['n_permutations']
    )
    
    clean_gpu_memory()
    
    # Save data samples if requested
    if data_samples_dir:
        file_prefix = f"vanilla_{order_strategy}_size{train_size}_rep{repetition}"
        save_data_samples(X_train_reordered, X_test_reordered, X_synth, 
                         col_names_reordered, data_samples_dir, file_prefix)
    
    # Evaluate
    metrics = evaluate_synthetic_data_metrics(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': order_strategy,
        'column_order': str(column_order),
        'algorithm': 'vanilla',
    }
    
    return build_result_row(base_info, metrics, config)

def run_single_configuration(train_size: int, order_strategy: str, repetition: int, 
                           X_test: np.ndarray, col_names: List[str], categorical_cols: List[int],
                           pre_calculated_orders: Dict[str, List[int]], config: Dict[str, Any],
                           hash_check_dict: Dict[Tuple[int, int], Tuple[str, str]],
                           data_samples_dir: Optional[Path] = None) -> OrderedDict:
    """Run a single experimental configuration."""
    print(f"    Order: {order_strategy}, Rep: {repetition+1}/{config['n_repetitions']}")
    
    # Setup for this iteration
    seed = config['random_seed_base'] + repetition
    setup_deterministic_mode(seed)
    
    # Generate training data
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    validate_data_integrity(X_train_original, X_test, train_size, repetition, hash_check_dict)
    
    # Get pre-calculated column order
    column_order = pre_calculated_orders[order_strategy]
    
    # Run experiment
    return run_vanilla_tabpfn_with_ordering(
        X_train_original, X_test, col_names, categorical_cols,
        column_order, order_strategy, config, seed, train_size, repetition, data_samples_dir
    )

def run_experiment_2_improved(config: Optional[Dict[str, Any]] = None, 
                             output_dir: str = "experiment_2_results", 
                             resume: bool = True) -> pd.DataFrame:
    """
    Run Experiment 2: Column ordering effects on vanilla TabPFN.
    
    Tests whether column ordering affects synthetic data quality when TabPFN
    uses its implicit autoregressive mechanism (no DAG provided).
    
    Args:
        config: Optional configuration overrides
        output_dir: Output directory name
        resume: Whether to resume from checkpoint
        
    Returns:
        DataFrame with experimental results
    """
    # Setup configuration
    final_config = DEFAULT_CONFIG.copy()
    if config is not None:
        final_config.update(config)
    
    # Setup directories
    script_dir = Path(__file__).parent
    output_dir_path = script_dir / 'results'
    output_dir_path.mkdir(exist_ok=True)
    data_samples_dir = script_dir / 'data_samples'
    data_samples_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 2 IMPROVED - Output dir: {output_dir_path}")
    print(f"Config: {json.dumps(final_config, indent=2)}")
    
    # Get experimental setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(final_config['include_categorical'])
    X_test_original = generate_scm_data(final_config['test_size'], 123, final_config['include_categorical'])
    
    # Pre-calculate and validate all column orderings
    pre_calculated_orders = validate_and_prepare_column_orderings(final_config, correct_dag)
    
    # Setup checkpointing
    raw_results_file = output_dir_path / "raw_results.csv"
    final_results_file = output_dir_path / "experiment_2_improved_results.csv"
    
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir_path)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Calculate total work
    total_iterations = (len(final_config['train_sizes']) * 
                       len(final_config['ordering_strategies']) * 
                       final_config['n_repetitions'])
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    hash_check_dict = {}
    
    try:
        config_idx = 0
        
        for train_idx, train_size in enumerate(final_config['train_sizes']):
            for rep in range(final_config['n_repetitions']):
                for order_strategy in final_config['ordering_strategies']:
                    if config_idx < completed:
                        config_idx += 1
                        continue
                    
                    # Run single configuration
                    result = run_single_configuration(
                        train_size, order_strategy, rep, X_test_original,
                        col_names, categorical_cols, pre_calculated_orders, final_config,
                        hash_check_dict, data_samples_dir
                    )
                    
                    # Save progress
                    results_so_far.append(result)
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(raw_results_file, index=False)
                    save_checkpoint(results_so_far, train_idx, rep + 1, output_dir_path)
                    
                    completed += 1
                    config_idx += 1
                    
                    progress_pct = 100 * completed / total_iterations
                    print(f"    Progress: {completed}/{total_iterations} ({progress_pct:.1f}%)")
                    print(f"    Results saved to: {raw_results_file}")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    print("\nExperiment completed!")
    cleanup_checkpoint(output_dir_path)
    
    # Final results processing
    df_results = pd.DataFrame(results_so_far)
    
    # Ensure consistent column ordering
    metric_cols = [col for col in df_results.columns if col not in PREFERRED_ORDER]
    ordered_cols = [col for col in PREFERRED_ORDER if col in df_results.columns] + metric_cols
    df_results = df_results[ordered_cols]
    
    df_results.to_csv(final_results_file, index=False)
    print(f"Results saved to: {final_results_file}")
    print(f"Total results: {len(df_results)}")
    
    return df_results

def main():
    """Main CLI interface for Experiment 2 Improved."""
    parser = argparse.ArgumentParser(description='Run Experiment 2: Column ordering effects (Improved)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show experiment info
    dag, col_names, _ = get_dag_and_config(False)
    
    print("=" * 60)
    print("EXPERIMENT 2 IMPROVED: Column Ordering Effects on Vanilla TabPFN")
    print("=" * 60)
    print("\nResearch Question:")
    print("Does column ordering affect synthetic data quality when TabPFN")
    print("uses its implicit autoregressive mechanism (no DAG provided)?")
    print("\nSCM structure (for ordering strategies reference):")
    print_dag_info(dag, col_names)
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    output_dir = args.output or "experiment_2_improved_results"
    
    # Calculate total configurations
    total_configs = (len(config['train_sizes']) * 
                    len(config['ordering_strategies']) * 
                    config['n_repetitions'])
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  Ordering strategies: {config['ordering_strategies']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    
    # Run experiment
    results = run_experiment_2_improved(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Get actual metric columns from results
        metric_columns = [col for col in results.columns 
                         if col not in ['train_size', 'repetition', 'categorical', 'seed', 
                                       'column_order_strategy', 'column_order', 'algorithm']]
        
        # Best and worst orderings per metric
        for metric in metric_columns:
            if results[metric].dtype in ['float64', 'int64']:  # Only numeric metrics
                print(f"\n{metric.upper()}:")
                mean_by_order = results.groupby('column_order_strategy')[metric].mean()
                best_order = mean_by_order.idxmin()
                worst_order = mean_by_order.idxmax()
                
                print(f"  Best ordering: {best_order} ({mean_by_order[best_order]:.4f})")
                print(f"  Worst ordering: {worst_order} ({mean_by_order[worst_order]:.4f})")
                print(f"  Difference: {mean_by_order[worst_order] - mean_by_order[best_order]:.4f}")

if __name__ == "__main__":
    main()