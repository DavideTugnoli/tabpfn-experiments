"""
Experiment 1: Effect of DAG and training set size.
Clean, generic, works with any SCM/DAG.

Usage:
    python experiment_1.py                    # Fair comparison (topological order)
    python experiment_1.py --order original  # Original order (neutral)
    python experiment_1.py --order worst     # Worst case for vanilla
"""
import sys
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import warnings
import argparse
import shutil
import hashlib
import json
from collections import OrderedDict

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config, get_cpdag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, reorder_data_and_dag, print_dag_info, create_dag_variations, dag_belongs_to_cpdag, get_graph_edge_counts
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'cases': [
        {'algorithm': 'vanilla', 'graph_type': None},
        {'algorithm': 'dag', 'graph_type': 'correct'},
        {'algorithm': 'cpdag', 'graph_type': 'cpdag'},
    ],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42,
    'column_order_strategy': 'original',
}

# Preferred order for result columns
PREFERRED_ORDER = [
    'algorithm', 'graph_type', 'graph_structure', 'dag_edges_directed', 'dag_edges_undirected', 'dag_nodes', 'train_size', 'repetition', 'seed', 'categorical', 'column_order_strategy', 'column_order'
]

SAVE_DATA_SAMPLES = True  # Set to True to save data_samples for debugging

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

# Utility: Evaluate metrics

def evaluate_metrics(X_test, X_synth, col_names, categorical_cols, k_for_kmarginal=2):
    evaluator = FaithfulDataEvaluator()
    cat_col_names = [col_names[i] for i in categorical_cols] if categorical_cols else []
    return evaluator.evaluate(
        pd.DataFrame(X_test, columns=col_names),
        pd.DataFrame(X_synth, columns=col_names),
        categorical_columns=cat_col_names if cat_col_names else None,
        k_for_kmarginal=k_for_kmarginal
    )

# Helper to build result row in correct order
def build_result_row(base_info, metrics, preferred_order, metric_cols):
    row = OrderedDict()
    for k in preferred_order:
        row[k] = base_info.get(k, '')
    for k in metric_cols:
        row[k] = metrics.get(k, '')
    return row

# Pipeline: With DAG (no reordering)

def run_with_dag(X_train, X_test, dag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, graph_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag=dag, n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test, X_synth, col_names, categorical_cols)
    dag_dict = dag
    edge_counts = get_graph_edge_counts(dag_dict)
    dag_nodes = len(dag_dict)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
        'algorithm': algorithm,
        'graph_type': graph_type,
        'graph_structure': str(dag_dict),
        'dag_edges_directed': edge_counts['directed'],
        'dag_edges_undirected': edge_counts['undirected'],
        'dag_nodes': dag_nodes,
        'dag_structure': str(dag_dict),
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row, X_synth

# Pipeline: With CPDAG (no reordering)

def run_with_cpdag(X_train, X_test, cpdag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], cpdag=cpdag, n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test, X_synth, col_names, categorical_cols)
    if isinstance(cpdag, np.ndarray):
        cpdag_dict = model._parse_cpdag_adjacency_matrix(cpdag)
    else:
        cpdag_dict = cpdag
    edge_counts = get_graph_edge_counts(cpdag_dict)
    dag_nodes = len(cpdag_dict)
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
        'dag_nodes': dag_nodes,
        'dag_structure': str(cpdag_dict),
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row, X_synth

# Pipeline: No DAG (with reordering)

def run_vanilla(X_train, X_test, col_names, categorical_cols, column_order, config, seed, train_size, repetition, algorithm, column_order_name=None):
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, column_order
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': column_order_name if column_order_name is not None else '',
        'column_order': str(column_order) if column_order is not None else '',
        'algorithm': algorithm,
        'graph_type': None,
        'graph_structure': '',
        'dag_edges_directed': 0,
        'dag_edges_undirected': 0,
        'dag_nodes': 0,
        'dag_structure': '',
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row, X_synth, col_names_reordered

# Main iteration orchestrator

def run_single_configuration(train_size, algorithm, graph_type, repetition, config, 
                           X_test, col_names, categorical_cols,
                           correct_dag, cpdag, vanilla_column_order, vanilla_order_name,
                           data_samples_dir=None, hash_check_dict=None):
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set PyTorch to deterministic mode for reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # For older PyTorch versions
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    train_hash = hash_array(X_train_original)
    test_hash = hash_array(X_test)
    if hash_check_dict is not None:
        key = (train_size, repetition)
        if key in hash_check_dict:
            prev_train_hash, prev_test_hash = hash_check_dict[key]
            if prev_train_hash != train_hash or prev_test_hash != test_hash:
                raise RuntimeError(f"[HASH ERROR] Train/Test data hash mismatch for train_size={train_size}, repetition={repetition}!\nPrev train hash: {prev_train_hash}\nCurrent train hash: {train_hash}\nPrev test hash: {prev_test_hash}\nCurrent test hash: {test_hash}")
        else:
            hash_check_dict[key] = (train_hash, test_hash)
    result_row, X_synth = None, None
    if algorithm == 'vanilla':
        result_row, X_synth, col_names_reordered = run_vanilla(
            X_train_original, X_test, col_names, categorical_cols,
            vanilla_column_order, config, seed, train_size, repetition, algorithm, vanilla_order_name
        )
        if SAVE_DATA_SAMPLES and data_samples_dir:
            X_train_reordered, _, _ = reorder_data_and_columns(X_train_original, col_names, categorical_cols, vanilla_column_order)
            X_test_reordered, _, _ = reorder_data_and_columns(X_test, col_names, categorical_cols, vanilla_column_order)
            file_prefix = f"{algorithm}_{vanilla_order_name}_size{train_size}_rep{repetition}"
            pd.DataFrame(X_train_reordered, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_train.csv", index=False)
            pd.DataFrame(X_test_reordered, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_test.csv", index=False)
            pd.DataFrame(X_synth, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_synth.csv", index=False)
    elif algorithm == 'dag':
        result_row, X_synth = run_with_dag(X_train_original, X_test, correct_dag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, graph_type)
        if SAVE_DATA_SAMPLES and data_samples_dir:
            file_prefix = f"{algorithm}_{graph_type}_size{train_size}_rep{repetition}"
            pd.DataFrame(X_train_original, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_train.csv", index=False)
            pd.DataFrame(X_test, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_test.csv", index=False)
            pd.DataFrame(X_synth, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_synth.csv", index=False)
    elif algorithm == 'cpdag':
        result_row, X_synth = run_with_cpdag(X_train_original, X_test, cpdag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm)
        if SAVE_DATA_SAMPLES and data_samples_dir:
            file_prefix = f"{algorithm}_size{train_size}_rep{repetition}"
            pd.DataFrame(X_train_original, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_train.csv", index=False)
            pd.DataFrame(X_test, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_test.csv", index=False)
            pd.DataFrame(X_synth, columns=col_names).head(10).to_csv(data_samples_dir / f"{file_prefix}_synth.csv", index=False)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return result_row

def run_experiment_1(config=None, output_dir="experiment_1_results", resume=True, cases_filter=None):
    """
    Main experiment function with column ordering control.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        resume: Whether to resume from checkpoint
        cases_filter: List of algorithm names to run (e.g., ['vanilla', 'dag'])
    """
    base_config = DEFAULT_CONFIG.copy()
    if config is not None:
        base_config.update(config)
    config = base_config
    
    # Filter cases if specified
    if cases_filter:
        config['cases'] = [case for case in config['cases'] if case['algorithm'] in cases_filter]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'results'
    output_dir.mkdir(exist_ok=True)
    data_samples_dir = Path(script_dir) / 'data_samples'
    data_samples_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 1 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    cpdag, _, _ = get_cpdag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    available_orderings = get_ordering_strategies(correct_dag)
    column_order_name = config.get('column_order_strategy')
    if column_order_name not in available_orderings:
        raise ValueError(f"Unknown ordering strategy: {column_order_name}. "
                        f"Available: {list(available_orderings.keys())}")
    vanilla_column_order = available_orderings[column_order_name]
    print(f"Pre-calculated column order for vanilla case: {column_order_name} = {vanilla_column_order}")
    
    # Determine output file name based on cases being run
    if cases_filter and len(cases_filter) == 1:
        if cases_filter[0] == 'vanilla':
            strategy = config.get('column_order_strategy')
            raw_results_file = output_dir / f"raw_results_vanilla_{strategy}.csv"
        elif cases_filter[0] == 'dag':
            raw_results_file = output_dir / f"raw_results_dag_correct.csv"
        elif cases_filter[0] == 'cpdag':
            raw_results_file = output_dir / f"raw_results_cpdag.csv"
        else:
            strategy = config.get('column_order_strategy')
            raw_results_file = output_dir / f"raw_results_{strategy}.csv"
    else:
        strategy = config.get('column_order_strategy')
        raw_results_file = output_dir / f"raw_results_{strategy}.csv"
    
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    total_iterations = len(config['train_sizes']) * config['n_repetitions'] * len(config['cases'])
    completed = len(results_so_far)
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        config_idx = 0
        cases = config['cases']
        for case in cases:
            algorithm = case['algorithm']
            graph_type = case['graph_type']
            print(f"\n=== Running algorithm: {algorithm}{' (graph: ' + str(graph_type) + ')' if graph_type else ''} ===")
            for train_idx, train_size in enumerate(config['train_sizes']):
                for rep in range(config['n_repetitions']):
                    if config_idx < completed:
                        config_idx += 1
                        continue
                    row = run_single_configuration(
                        train_size, algorithm, graph_type, rep, config, X_test_original, col_names, categorical_cols, correct_dag, cpdag, vanilla_column_order, column_order_name, data_samples_dir=data_samples_dir
                    )
                    results_so_far.append(row)
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(raw_results_file, index=False)
                    save_checkpoint(results_so_far, train_idx, rep + 1, output_dir)
                    completed += 1
                    config_idx += 1
                    print(f"    Progress ({algorithm}{'/' + str(graph_type) if graph_type else ''}): {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                    print(f"    Results saved to: {raw_results_file}")
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return results_so_far
    print("\nExperiment completed!")
    cleanup_checkpoint(output_dir)
    print(f"Total results: {len(results_so_far)}")
    return results_so_far

def test_problematic_combination():
    """Test the specific combination that causes TabPFN to crash: train_size=20, rep=18, vanilla original."""
    print("=" * 60)
    print("TESTING PROBLEMATIC COMBINATION (COPY VERSION)")
    print("=" * 60)
    print("Testing: train_size=20, rep=18, algorithm=vanilla, order=original")
    
    # Setup exact same configuration as main experiment
    config = DEFAULT_CONFIG.copy()
    config['column_order_strategy'] = 'original'
    
    # Get experimental setup
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    available_orderings = get_ordering_strategies(correct_dag)
    
    # Set exact problematic parameters
    train_size = 20
    rep = 18
    seed = config['random_seed_base'] + rep  # 42 + 18 = 60
    algorithm = 'vanilla'
    column_order_strategy = 'original'
    
    print(f"Parameters: train_size={train_size}, rep={rep}, seed={seed}")
    print(f"Algorithm: {algorithm}, column_order_strategy={column_order_strategy}")
    
    # Setup deterministic mode with exact same seed (same as run_single_configuration)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    # Generate exact same training data
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    
    print(f"Generated training data stats:")
    print(f"  Shape: {X_train_original.shape}, dtype: {X_train_original.dtype}")
    print(f"  Min: {np.min(X_train_original):.6f}, Max: {np.max(X_train_original):.6f}")
    print(f"  Mean: {np.mean(X_train_original):.6f}, Std: {np.std(X_train_original):.6f}")
    print(f"  Has inf: {np.any(np.isinf(X_train_original))}, Has nan: {np.any(np.isnan(X_train_original))}")
    
    try:
        # Get column order
        column_order = available_orderings[column_order_strategy]
        print(f"Column order: {column_order}")
        
        # Run the exact same function that crashes (using copy version functions)
        print("\nCalling run_vanilla...")
        result_row, X_synth, col_names_reordered = run_vanilla(
            X_train_original, X_test_original, col_names, categorical_cols,
            column_order, config, seed, train_size, rep, 
            algorithm, column_order_strategy
        )
        
        print("SUCCESS! No crash occurred in copy version.")
        print(f"Generated synthetic data shape: {X_synth.shape}")
        print(f"Synthetic data stats: min={np.min(X_synth):.6f}, max={np.max(X_synth):.6f}")
        
    except Exception as e:
        print(f"CRASH REPRODUCED IN COPY VERSION!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"This confirms the bug exists in both versions.")
        raise

def main():
    """Main CLI interface for Experiment 1."""
    parser = argparse.ArgumentParser(description='Run Experiment 1')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--order', type=str, default=None,
                       choices=['original', 'topological', 'worst', 'random', 'reverse', 'both'],
                       help='Column ordering strategy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--test-bug', action='store_true',
                       help='Test the specific problematic combination that causes TabPFN to crash')
    
    args = parser.parse_args()
    
    # Test problematic combination if requested
    if args.test_bug:
        test_problematic_combination()
        return

    # Show DAG and CPDAG info first
    dag, col_names, _ = get_dag_and_config(False)
    cpdag, _, _ = get_cpdag_and_config(False)
    print("Current SCM structure:")
    print_dag_info(dag, col_names)
    print("\nCPDAG structure (X1 -> X2 <- X3 - X4):")
    print(f"CPDAG adjacency matrix:\n{cpdag}")
    print()
    print("\nAlgorithms/Graphs to test:")
    print("-" * 40)
    print("1. vanilla: Algorithm (no graph provided, TabPFN vanilla)")
    print("2. dag (correct): Algorithm (true DAG provided)")
    print("3. cpdag: Algorithm (CPDAG provided, equivalence class of DAGs)")

    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'results'
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
    results_dag = run_experiment_1(
        config=config_dag,
        output_dir=output_dir,
        resume=not args.no_resume,
        cases_filter=['dag']
    )
    all_results.extend(results_dag)
    
    # Run CPDAG case
    config_cpdag = DEFAULT_CONFIG.copy()
    config_cpdag['column_order_strategy'] = 'original'  # Doesn't matter for CPDAG
    results_cpdag = run_experiment_1(
        config=config_cpdag,
        output_dir=output_dir,
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
        results_vanilla = run_experiment_1(
            config=config_vanilla,
            output_dir=output_dir,
            resume=not args.no_resume,
            cases_filter=['vanilla']
        )
        all_results.extend(results_vanilla)
    
    # Save combined results
    df_final = pd.DataFrame(all_results)
    final_results_file = output_dir / "results_experiment_1.csv"
    df_final.to_csv(final_results_file, index=False, na_rep='')
    print(f"\nAll experiments completed!")
    print(f"Combined results saved to: {final_results_file}")
    print(f"Total results: {len(df_final)}")

if __name__ == "__main__":
    main()