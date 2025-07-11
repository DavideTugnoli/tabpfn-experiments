"""
Experiment 1 IMPROVED: Effect of DAG and training set size.

This experiment compares TabPFN synthetic data quality when provided with:
- No DAG (vanilla TabPFN with column reordering)
- Correct DAG knowledge
- CPDAG (equivalence class of DAGs)

Usage:
    python experiment_1_det_cpdag_improved.py                    # Fair comparison (topological order)
    python experiment_1_det_cpdag_improved.py --order original  # Original order (neutral)
    python experiment_1_det_cpdag_improved.py --order worst     # Worst case for vanilla
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import argparse
import json
import hashlib
from collections import OrderedDict

# Environment setup
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
import numpy as np

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config, get_cpdag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, print_dag_info, get_graph_edge_counts
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Centralized default config
DEFAULT_CONFIG = {
    # Experimental design
    'train_sizes': [20, 50, 100, 200, 500],
    'n_repetitions': 10,
    'test_size': 2000,
    
    # Model parameters
    'n_estimators': 3,
    'n_permutations': 3,
    
    # Data configuration
    'include_categorical': False,
    'random_seed_base': 42,
    'column_order_strategy': 'original',
    
    # Evaluation metrics
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    
    # Experimental cases
    'cases': [
        {'algorithm': 'vanilla', 'graph_type': None},
        {'algorithm': 'dag', 'graph_type': 'correct'},
        {'algorithm': 'cpdag', 'graph_type': 'cpdag'},
    ],
}

# Preferred order for result columns
PREFERRED_ORDER = [
    'algorithm', 'graph_type', 'graph_structure', 'dag_edges_directed', 
    'dag_edges_undirected', 'dag_nodes', 'train_size', 'repetition', 
    'seed', 'categorical', 'column_order_strategy', 'column_order'
]

SAVE_DATA_SAMPLES = True  # Set to True to save data_samples for debugging

def hash_array(arr: np.ndarray) -> str:
    """Generate hash for data integrity checking."""
    return hashlib.md5(arr.tobytes()).hexdigest()

def setup_deterministic_mode(seed: int) -> None:
    """Setup deterministic mode for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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

def evaluate_synthetic_data_metrics(X_test: np.ndarray, X_synth: np.ndarray, 
                                   col_names: List[str], categorical_cols: List[int],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
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

def run_with_dag_knowledge(X_train: np.ndarray, X_test: np.ndarray, dag: Dict[int, List[int]], 
                          col_names: List[str], categorical_cols: List[int], 
                          config: Dict[str, Any], seed: int, train_size: int, repetition: int, 
                          algorithm: str, graph_type: str, data_samples_dir: Optional[Path] = None) -> Tuple[OrderedDict, np.ndarray]:
    """Run experiment with DAG knowledge."""
    model = create_tabpfn_model(config)
    
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag=dag, 
        n_permutations=config['n_permutations']
    )
    
    clean_gpu_memory()
    
    # Save data samples if requested
    if data_samples_dir:
        file_prefix = f"{algorithm}_{graph_type}_size{train_size}_rep{repetition}"
        save_data_samples(X_train, X_test, X_synth, col_names, data_samples_dir, file_prefix)
    
    # Evaluate metrics
    metrics = evaluate_synthetic_data_metrics(X_test, X_synth, col_names, categorical_cols, config)
    edge_counts = get_graph_edge_counts(dag)
    
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
        'algorithm': algorithm,
        'graph_type': graph_type,
        'graph_structure': str(dag),
        'dag_edges_directed': edge_counts['directed'],
        'dag_edges_undirected': edge_counts['undirected'],
        'dag_nodes': len(dag),
        'dag_structure': str(dag),
    }
    
    return build_result_row(base_info, metrics, config), X_synth

def run_with_cpdag_knowledge(X_train: np.ndarray, X_test: np.ndarray, cpdag: np.ndarray,
                            col_names: List[str], categorical_cols: List[int], 
                            config: Dict[str, Any], seed: int, train_size: int, repetition: int, 
                            algorithm: str, data_samples_dir: Optional[Path] = None) -> Tuple[OrderedDict, np.ndarray]:
    """Run experiment with CPDAG knowledge."""
    model = create_tabpfn_model(config)
    
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], cpdag=cpdag,
        n_permutations=config['n_permutations']
    )
    
    clean_gpu_memory()
    
    # Save data samples if requested
    if data_samples_dir:
        file_prefix = f"{algorithm}_size{train_size}_rep{repetition}"
        save_data_samples(X_train, X_test, X_synth, col_names, data_samples_dir, file_prefix)
    
    # Evaluate metrics
    metrics = evaluate_synthetic_data_metrics(X_test, X_synth, col_names, categorical_cols, config)
    
    # Convert CPDAG to dict for edge counting
    if isinstance(cpdag, np.ndarray):
        cpdag_dict = model._parse_cpdag_adjacency_matrix(cpdag)
    else:
        cpdag_dict = cpdag
    
    edge_counts = get_graph_edge_counts(cpdag_dict)
    
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
        'algorithm': algorithm,
        'graph_type': 'cpdag',
        'graph_structure': str(cpdag_dict),
        'dag_edges_directed': edge_counts['directed'],
        'dag_edges_undirected': edge_counts['undirected'],
        'dag_nodes': len(cpdag_dict),
        'dag_structure': str(cpdag_dict),
    }
    
    return build_result_row(base_info, metrics, config), X_synth

def run_vanilla_tabpfn(X_train: np.ndarray, X_test: np.ndarray, col_names: List[str],
                      categorical_cols: List[int], column_order: List[int], 
                      config: Dict[str, Any], seed: int, train_size: int, repetition: int, 
                      algorithm: str, column_order_name: str, 
                      data_samples_dir: Optional[Path] = None) -> Tuple[OrderedDict, np.ndarray, List[str]]:
    """Run experiment without DAG knowledge (vanilla TabPFN with column reordering)."""
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, column_order
    )
    model = create_tabpfn_model(config)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], 
        n_permutations=config['n_permutations']
    )
    clean_gpu_memory()
    # Save data samples if requested
    if data_samples_dir:
        file_prefix = f"{algorithm}_{column_order_name}_size{train_size}_rep{repetition}"
        save_data_samples(X_train_reordered, X_test_reordered, X_synth, 
                         col_names_reordered, data_samples_dir, file_prefix)
    # Evaluate metrics
    metrics = evaluate_synthetic_data_metrics(X_test_reordered, X_synth, col_names_reordered, 
                                            categorical_cols_reordered, config)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': column_order_name,
        'column_order': str(column_order),
        'algorithm': algorithm,
        'graph_type': None,
        'graph_structure': '',
        'dag_edges_directed': 0,
        'dag_edges_undirected': 0,
        'dag_nodes': 0,
        'dag_structure': '',
    }
    return build_result_row(base_info, metrics, config), X_synth, col_names_reordered

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

def run_experiment_1_improved(config: Optional[Dict[str, Any]] = None, 
                             output_dir: str = "experiment_1_results", 
                             resume: bool = True, cases_filter: Optional[List[str]] = None) -> List[OrderedDict]:
    """
    Run Experiment 1: Effect of DAG and training set size.
    
    Compares TabPFN synthetic data quality across different levels of causal knowledge:
    - vanilla: No DAG provided (column reordering applied)
    - dag: True DAG structure provided  
    - cpdag: CPDAG (equivalence class) provided
    
    Args:
        config: Optional configuration overrides
        output_dir: Output directory name
        resume: Whether to resume from checkpoint
        cases_filter: List of algorithm names to run (e.g., ['vanilla', 'dag'])
        
    Returns:
        List of result dictionaries
    """
    # Setup configuration
    final_config = DEFAULT_CONFIG.copy()
    if config is not None:
        final_config.update(config)
    
    # Filter cases if specified
    if cases_filter:
        final_config['cases'] = [case for case in final_config['cases'] if case['algorithm'] in cases_filter]
    
    # Setup directories
    script_dir = Path(__file__).parent
    output_dir_path = script_dir / 'results'
    output_dir_path.mkdir(exist_ok=True)
    data_samples_dir = script_dir / 'data_samples'
    data_samples_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 1 IMPROVED - Output dir: {output_dir_path}")
    print(f"Config: {json.dumps(final_config, indent=2)}")
    
    # Get experimental setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(final_config['include_categorical'])
    cpdag, _, _ = get_cpdag_and_config(final_config['include_categorical'])
    X_test_original = generate_scm_data(final_config['test_size'], 123, final_config['include_categorical'])
    
    # Get column ordering for vanilla case
    available_orderings = get_ordering_strategies(correct_dag)
    column_order_name = final_config.get('column_order_strategy')
    
    if column_order_name not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {column_order_name}. "
                        f"Available: {list(available_orderings.keys())}")
    
    vanilla_column_order = available_orderings[column_order_name]
    print(f"Column order for vanilla case: {column_order_name} = {vanilla_column_order}")
    
    # Determine output file name based on cases being run
    if cases_filter and len(cases_filter) == 1:
        if cases_filter[0] == 'vanilla':
            strategy = final_config.get('column_order_strategy')
            raw_results_file = output_dir_path / f"raw_results_vanilla_{strategy}.csv"
        elif cases_filter[0] == 'dag':
            raw_results_file = output_dir_path / f"raw_results_dag_correct.csv"
        elif cases_filter[0] == 'cpdag':
            raw_results_file = output_dir_path / f"raw_results_cpdag.csv"
        else:
            strategy = final_config.get('column_order_strategy')
            raw_results_file = output_dir_path / f"raw_results_{strategy}.csv"
    else:
        strategy = final_config.get('column_order_strategy')
        raw_results_file = output_dir_path / f"raw_results_{strategy}.csv"
    
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir_path)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Calculate total work
    total_iterations = len(final_config['train_sizes']) * final_config['n_repetitions'] * len(final_config['cases'])
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    hash_check_dict = {}
    
    try:
        config_idx = 0
        
        for case in final_config['cases']:
            algorithm = case['algorithm']
            graph_type = case['graph_type']
            
            print(f"\n=== Running algorithm: {algorithm}" + 
                  (f" (graph: {graph_type})" if graph_type else "") + " ===")
            
            for train_idx, train_size in enumerate(final_config['train_sizes']):
                for rep in range(final_config['n_repetitions']):
                    if config_idx < completed:
                        config_idx += 1
                        continue
                    
                    # Setup for this iteration
                    seed = final_config['random_seed_base'] + rep
                    setup_deterministic_mode(seed)
                    
                    X_train_original = generate_scm_data(train_size, seed, final_config['include_categorical'])
                    validate_data_integrity(X_train_original, X_test_original, train_size, rep, hash_check_dict)
                    
                    # Run specific algorithm
                    if algorithm == 'vanilla':
                        result_row, X_synth, col_names_reordered = run_vanilla_tabpfn(
                            X_train_original, X_test_original, col_names, categorical_cols,
                            vanilla_column_order, final_config, seed, train_size, rep, 
                            algorithm, column_order_name, data_samples_dir
                        )
                    elif algorithm == 'dag':
                        result_row, X_synth = run_with_dag_knowledge(
                            X_train_original, X_test_original, correct_dag, col_names, categorical_cols,
                            final_config, seed, train_size, rep, algorithm, graph_type, data_samples_dir
                        )
                    elif algorithm == 'cpdag':
                        result_row, X_synth = run_with_cpdag_knowledge(
                            X_train_original, X_test_original, cpdag, col_names, categorical_cols,
                            final_config, seed, train_size, rep, algorithm, data_samples_dir
                        )
                    else:
                        raise ValueError(f"Unknown algorithm: {algorithm}")
                    
                    # Save progress
                    results_so_far.append(result_row)
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(raw_results_file, index=False)
                    save_checkpoint(results_so_far, train_idx, rep + 1, output_dir_path)
                    
                    completed += 1
                    config_idx += 1
                    
                    progress_pct = 100 * completed / total_iterations
                    print(f"    Progress ({algorithm}" + 
                          (f"/{graph_type}" if graph_type else "") + 
                          f"): {completed}/{total_iterations} ({progress_pct:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return results_so_far
    
    print("\nExperiment completed!")
    cleanup_checkpoint(output_dir_path)
    
    print(f"Total results: {len(results_so_far)}")
    
    return results_so_far

def main():
    """Main CLI interface for Experiment 1 Improved."""
    parser = argparse.ArgumentParser(description='Run Experiment 1 (Improved Version)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--order', type=str, default=None,
                       choices=['original', 'topological', 'worst', 'random', 'both'],
                       help='Column ordering strategy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Show experiment info
    dag, col_names, _ = get_dag_and_config(False)
    cpdag, _, _ = get_cpdag_and_config(False)
    
    print("=" * 60)
    print("EXPERIMENT 1 IMPROVED: DAG and Training Set Size Effects")
    print("=" * 60)
    print("\nCurrent SCM structure:")
    print_dag_info(dag, col_names)
    print(f"\nCPDAG structure:\n{cpdag}")
    
    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Determine which orderings to run for vanilla
    if args.order is None or args.order == 'both':
        vanilla_orderings = ['original', 'topological']
    else:
        vanilla_orderings = [args.order]
    
    all_results = []
    
    # Run DAG and CPDAG only once (they don't depend on column ordering)
    print(f"\n{'='*50}")
    print("Running DAG and CPDAG cases (independent of column ordering)")
    print(f"{'='*50}")
    
    # Run DAG case
    config_dag = DEFAULT_CONFIG.copy()
    config_dag['column_order_strategy'] = 'original'  # Doesn't matter for DAG
    results_dag = run_experiment_1_improved(
        config=config_dag,
        output_dir=str(output_dir),
        resume=not args.no_resume,
        cases_filter=['dag']
    )
    all_results.extend(results_dag)
    
    # Run CPDAG case
    config_cpdag = DEFAULT_CONFIG.copy()
    config_cpdag['column_order_strategy'] = 'original'  # Doesn't matter for CPDAG
    results_cpdag = run_experiment_1_improved(
        config=config_cpdag,
        output_dir=str(output_dir),
        resume=not args.no_resume,
        cases_filter=['cpdag']
    )
    all_results.extend(results_cpdag)
    
    # Run vanilla cases with different orderings
    for ordering in vanilla_orderings:
        print(f"\n{'='*50}")
        print(f"Running vanilla case with ordering: {ordering}")
        print(f"{'='*50}")
        
        config_vanilla = DEFAULT_CONFIG.copy()
        config_vanilla['column_order_strategy'] = ordering
        results_vanilla = run_experiment_1_improved(
            config=config_vanilla,
            output_dir=str(output_dir),
            resume=not args.no_resume,
            cases_filter=['vanilla']
        )
        all_results.extend(results_vanilla)
    
    # Save combined results
    df_final = pd.DataFrame(all_results)
    final_results_file = output_dir / "results_experiment_1_improved.csv"
    df_final.to_csv(final_results_file, index=False, na_rep='')
    print(f"\nAll experiments completed!")
    print(f"Combined results saved to: {final_results_file}")
    print(f"Total results: {len(df_final)}")

if __name__ == "__main__":
    main()