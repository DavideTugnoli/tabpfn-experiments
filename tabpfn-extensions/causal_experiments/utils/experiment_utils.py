import matplotlib.pyplot as plt
import sys
from io import StringIO
import numpy as np

def generate_synthetic_data_quiet(model, n_samples, dag=None, cpdag=None, n_permutations=3):
    """Generate synthetic data with TabPFN, suppressing output."""
    plt.ioff()
    plt.close('all')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        # Build kwargs dict with only the parameters that are not None
        kwargs = {
            'n_samples': n_samples,
            't': 1.0,
            'n_permutations': n_permutations,
        }
        if dag is not None:
            kwargs['dag'] = dag
        if cpdag is not None:
            kwargs['cpdag'] = cpdag
        
        # INTENSIVE DEBUG: Print everything before generation
        print(f"\n=== DEBUG generate_synthetic_data_quiet ===")
        print(f"n_samples: {n_samples}")
        print(f"n_permutations: {n_permutations}")
        print(f"dag: {dag}")
        print(f"cpdag type: {type(cpdag)}")
        if isinstance(cpdag, np.ndarray):
            print(f"cpdag shape: {cpdag.shape}")
            print(f"cpdag stats: min={np.min(cpdag)}, max={np.max(cpdag)}")
        
        # Check model's fitted data statistics
        if hasattr(model, 'tabpfn_clf') and hasattr(model.tabpfn_clf, 'X_'):
            X_fitted = model.tabpfn_clf.X_
            print(f"Model fitted data shape: {X_fitted.shape}")
            print(f"Model fitted data stats: min={np.min(X_fitted)}, max={np.max(X_fitted)}")
            print(f"Model fitted data has_inf: {np.any(np.isinf(X_fitted))}")
            print(f"Model fitted data has_nan: {np.any(np.isnan(X_fitted))}")
        
        try:
            print("About to call model.generate_synthetic_data...")
            X_synthetic = model.generate_synthetic_data(**kwargs).cpu().numpy()
            print(f"SUCCESS: Generated synthetic data shape: {X_synthetic.shape}")
            print(f"Generated data stats: min={np.min(X_synthetic)}, max={np.max(X_synthetic)}")
            print(f"Generated data has_inf: {np.any(np.isinf(X_synthetic))}")
            print(f"Generated data has_nan: {np.any(np.isnan(X_synthetic))}")
        except Exception as e:
            print(f"ERROR in generate_synthetic_data: {type(e).__name__}: {str(e)}")
            print(f"Full error traceback will follow...")
            raise
            
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        plt.close('all')
    
    return X_synthetic

def reorder_data_and_columns(X, col_names, categorical_cols, column_order):
    """
    Reorder data and column metadata according to column_order.
    Does NOT reorder the DAG since TabPFN doesn't care about column order when using a DAG.
    
    Args:
        X: Data array to reorder
        col_names: List of column names
        categorical_cols: List of categorical column indices
        column_order: New order of columns (list of indices)
    
    Returns:
        X_reordered: Reordered data array
        col_names_reordered: Reordered column names
        categorical_cols_reordered: Reordered categorical column indices
    """
    # Reorder data
    X_reordered = X[:, column_order]
    
    # Reorder column names
    col_names_reordered = [col_names[i] for i in column_order]
    
    # Reorder categorical column indices
    categorical_cols_reordered = None
    if categorical_cols:
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(column_order)}
        categorical_cols_reordered = [old_to_new[col] for col in categorical_cols if col in old_to_new]
    
    return X_reordered, col_names_reordered, categorical_cols_reordered 