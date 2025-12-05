"""Generate tables for analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from scipy import stats


def load_experiment_results(log_dir: str = "experiments/logs") -> Dict:
    """Load all experiment results from log directory.

    Args:
        log_dir: Directory containing experiment logs

    Returns:
        Dictionary of results grouped by experiment
    """
    log_path = Path(log_dir)
    results = {}

    for file in log_path.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        exp_name = metadata.get('experiment_name', file.stem)

        if exp_name not in results:
            results[exp_name] = []

        results[exp_name].append(data)

    return results


def generate_performance_table(
    results: Dict,
    save_path: str = "reports/tables/performance.csv"
) -> pd.DataFrame:
    """Generate table of final performance metrics.

    Args:
        results: Dictionary of experiment results
        save_path: Path to save table

    Returns:
        DataFrame with performance metrics
    """
    rows = []

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']
            metrics = result['metrics_history']

            if not metrics:
                continue

            # Get final metrics
            final_metrics = metrics[-1]

            dataset = metadata.get('dataset')
            if dataset is None and metadata.get('problem', '').lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'

            test_acc_history = [
                m.get('test_accuracy')
                for m in metrics
                if m.get('test_accuracy') is not None
            ]
            best_test_acc = max(test_acc_history) if test_acc_history else np.nan

            row = {
                'Experiment': exp_name,
                'Optimizer': metadata.get('optimizer', 'Unknown'),
                'Problem': metadata.get('problem', 'Unknown'),
                'Dataset': dataset or 'Unknown',
                'Learning Rate': metadata.get('learning_rate', np.nan),
                'Final Loss': final_metrics.get('train_loss', np.nan),
                'Final Train Acc': final_metrics.get('train_accuracy', np.nan),
                'Final Test Acc': final_metrics.get('test_accuracy', np.nan),
                'Best Test Acc': best_test_acc,
                'Total Iterations': len(metrics),
                'Total Time (s)': metadata.get('duration', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"Performance table saved to {save_path}")

    return df


def generate_convergence_table(
    results: Dict,
    thresholds: Dict[str, float] = None,
    save_path: str = "reports/tables/convergence.csv"
) -> pd.DataFrame:
    """Generate table of convergence statistics.

    Args:
        results: Dictionary of experiment results
        thresholds: Dictionary of convergence thresholds per problem
        save_path: Path to save table

    Returns:
        DataFrame with convergence statistics
    """
    if thresholds is None:
        thresholds = {'logistic': 0.5, 'mlp': 1.0}

    rows = []

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']
            metrics = result['metrics_history']

            if not metrics:
                continue

            problem = metadata.get('problem', 'Unknown')
            threshold = thresholds.get(problem.lower(), 0.5)

            # Find convergence iteration
            conv_iter = None
            conv_time = None
            for i, m in enumerate(metrics):
                if m.get('train_loss', float('inf')) <= threshold:
                    conv_iter = m['iteration']
                    conv_time = m.get('timestamp', 0) - metrics[0].get('timestamp', 0)
                    break

            converged = conv_iter is not None

            dataset = metadata.get('dataset')
            if dataset is None and metadata.get('problem', '').lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'

            row = {
                'Optimizer': metadata.get('optimizer', 'Unknown'),
                'Problem': problem,
                'Dataset': dataset or 'Unknown',
                'Threshold': threshold,
                'Convergence Iter': conv_iter if converged else np.nan,
                'Convergence Time (s)': conv_time if converged else np.nan,
                'Converged': 'Yes' if converged else 'No',
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Compute statistics per optimizer
    grouped = df.groupby(['Optimizer', 'Problem', 'Dataset']).agg({
        'Convergence Iter': ['mean', 'std'],
        'Convergence Time (s)': ['mean', 'std'],
        'Converged': lambda x: sum(x == 'Yes') / len(x)
    }).round(2)

    grouped.index.set_names(['Optimizer', 'Problem', 'Dataset'], inplace=True)
    grouped = grouped.reset_index()
    grouped.columns = [
        ' '.join(col).strip().replace('  ', ' ')
        if isinstance(col, tuple) else col
        for col in grouped.columns.to_list()
    ]
    grouped = grouped.rename(columns={
        'Convergence Iter mean': 'Convergence Iter Mean',
        'Convergence Iter std': 'Convergence Iter Std',
        'Convergence Time (s) mean': 'Convergence Time Mean (s)',
        'Convergence Time (s) std': 'Convergence Time Std (s)',
        'Converged <lambda>': 'Convergence Rate',
    })

    # Save to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(save_path, index=False)
    print(f"Convergence table saved to {save_path}")

    return grouped


def generate_computational_table(
    results: Dict,
    save_path: str = "reports/tables/computational.csv"
) -> pd.DataFrame:
    """Generate table of computational requirements.

    Args:
        results: Dictionary of experiment results
        save_path: Path to save table

    Returns:
        DataFrame with computational metrics
    """
    rows = []

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']
            metrics = result['metrics_history']

            if not metrics or len(metrics) < 2:
                continue

            # Compute average time per iteration
            times = [m.get('timestamp', 0) for m in metrics]
            time_diffs = np.diff(times)
            avg_time_per_iter = np.mean(time_diffs)

            dataset = metadata.get('dataset')
            if dataset is None and metadata.get('problem', '').lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'

            row = {
                'Optimizer': metadata.get('optimizer', 'Unknown'),
                'Problem': metadata.get('problem', 'Unknown'),
                'Dataset': dataset or 'Unknown',
                'Avg Time/Iter (ms)': avg_time_per_iter * 1000,
                'Total Time (s)': metadata.get('duration', np.nan),
                'Total Iterations': len(metrics),
                'Num Parameters': metadata.get('num_parameters', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Group by optimizer and problem
    grouped = df.groupby(['Optimizer', 'Problem', 'Dataset']).agg({
        'Avg Time/Iter (ms)': ['mean', 'std'],
        'Total Time (s)': ['mean', 'std'],
        'Total Iterations': 'mean',
    }).round(3)

    grouped.index.set_names(['Optimizer', 'Problem', 'Dataset'], inplace=True)
    grouped = grouped.reset_index()
    grouped.columns = [
        ' '.join(col).strip().replace('  ', ' ')
        if isinstance(col, tuple) else col
        for col in grouped.columns.to_list()
    ]
    grouped = grouped.rename(columns={
        'Avg Time/Iter (ms) mean': 'Avg Time/Iter Mean (ms)',
        'Avg Time/Iter (ms) std': 'Avg Time/Iter Std (ms)',
        'Total Time (s) mean': 'Total Time Mean (s)',
        'Total Time (s) std': 'Total Time Std (s)',
        'Total Iterations mean': 'Total Iterations Mean',
    })

    # Save to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(save_path, index=False)
    print(f"Computational table saved to {save_path}")

    return grouped


def _collect_final_metric(
    results: Dict,
    metric_name: str = 'test_accuracy'
) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """Aggregate final metric across runs for each optimizer/problem/dataset."""
    aggregated: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']
            metrics = result['metrics_history']

            if not metrics:
                continue

            final_metrics = metrics[-1]
            metric_value = final_metrics.get('test_accuracy')
            if metric_name == 'train_accuracy':
                metric_value = final_metrics.get('train_accuracy')
            elif metric_name == 'loss':
                metric_value = final_metrics.get('train_loss')

            if metric_value is None:
                continue

            problem = metadata.get('problem', 'Unknown')
            dataset = metadata.get('dataset')
            if dataset is None and problem.lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'

            key = (problem, dataset or 'Unknown')
            if key not in aggregated:
                aggregated[key] = {}

            optimizer = metadata.get('optimizer', 'Unknown')
            aggregated[key].setdefault(optimizer, []).append(metric_value)

    return aggregated


def generate_hyperparameter_table(
    results: Dict,
    save_path: str = "reports/tables/hyperparameters.csv"
) -> pd.DataFrame:
    """Generate table of best hyperparameters.

    Args:
        results: Dictionary of experiment results
        save_path: Path to save table

    Returns:
        DataFrame with hyperparameters
    """
    rows = []

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']

            dataset = metadata.get('dataset')
            if dataset is None and metadata.get('problem', '').lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'

            row = {
                'Optimizer': metadata.get('optimizer', 'Unknown'),
                'Problem': metadata.get('problem', 'Unknown'),
                'Dataset': dataset or 'Unknown',
                'Learning Rate': metadata.get('learning_rate', np.nan),
                'Batch Size': metadata.get('batch_size', np.nan),
                'Momentum': metadata.get('momentum', np.nan),
                'Beta1': metadata.get('beta1', np.nan),
                'Beta2': metadata.get('beta2', np.nan),
                'Weight Decay': metadata.get('weight_decay', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Remove columns that are all NaN
    df = df.dropna(axis=1, how='all')

    # Save to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"Hyperparameter table saved to {save_path}")

    return df


def generate_significance_table(
    results: Dict,
    save_path: str = "reports/tables/significance.csv",
    metric: str = 'test_accuracy'
) -> pd.DataFrame:
    """Generate table of statistical significance tests versus reference optimizer.

    Args:
        results: Dictionary of experiment results
        save_path: Path to save table
        metric: Metric to compare (default: final test accuracy)
    """
    aggregated = _collect_final_metric(results, metric_name=metric)
    rows = []

    for (problem, dataset), optimizer_values in aggregated.items():
        if not optimizer_values:
            continue

        # Prefer AdaMom as reference if present, else use best mean
        reference_optimizer = None
        if 'adamom' in {k.lower() for k in optimizer_values}:
            for name in optimizer_values:
                if name.lower() == 'adamom':
                    reference_optimizer = name
                    break
        if reference_optimizer is None:
            reference_optimizer = max(
                optimizer_values.items(),
                key=lambda item: np.mean(item[1])
            )[0]

        ref_values = optimizer_values[reference_optimizer]
        n_ref = len(ref_values)

        for optimizer, values in optimizer_values.items():
            if optimizer == reference_optimizer:
                continue

            n_opt = len(values)
            if n_ref < 2 or n_opt < 2:
                p_value = np.nan
            else:
                _, p_value = stats.ttest_ind(ref_values, values, equal_var=False)

            row = {
                'Problem': problem,
                'Dataset': dataset,
                'Metric': f'final_{metric}',
                'Reference Optimizer': reference_optimizer,
                'Reference Mean': np.mean(ref_values),
                'Reference Std': np.std(ref_values),
                'Reference Samples': n_ref,
                'Comparison Optimizer': optimizer,
                'Comparison Mean': np.mean(values),
                'Comparison Std': np.std(values),
                'Comparison Samples': n_opt,
                'Mean Difference': np.mean(ref_values) - np.mean(values),
                'p-value': p_value,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"Significance table saved to {save_path}")
    return df


def generate_all_tables(log_dir: str = "experiments/logs") -> None:
    """Generate all analysis tables.

    Args:
        log_dir: Directory containing experiment logs
    """
    print("Loading experiment results...")
    results = load_experiment_results(log_dir)

    print("\nGenerating performance table...")
    generate_performance_table(results)

    print("\nGenerating convergence table...")
    generate_convergence_table(results)

    print("\nGenerating computational table...")
    generate_computational_table(results)

    print("\nGenerating hyperparameter table...")
    generate_hyperparameter_table(results)

    print("\nGenerating statistical significance table...")
    generate_significance_table(results)

    print("\nAll tables generated successfully!")
