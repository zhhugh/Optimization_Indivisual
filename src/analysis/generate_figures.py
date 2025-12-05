"""Generate figures for analysis."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

from ..utils.visualization import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_optimization_trajectory,
    plot_learning_rate_sensitivity,
    plot_convergence_comparison,
    plot_gradient_norms,
)


def load_experiment_results(log_dir: str = "experiments/logs") -> Dict:
    """Load all experiment results from log directory."""
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


def organize_results_by_problem(results: Dict) -> Dict[tuple, Dict[str, Dict]]:
    """Organize results by problem and optimizer, averaging across seeds.

    Returns:
        {problem_name: {optimizer_name: metrics_dict with mean and std}}
    """
    # First collect all runs per problem-optimizer pair
    raw_data = {}

    for exp_name, exp_results in results.items():
        for result in exp_results:
            metadata = result['metadata']
            metrics_history = result['metrics_history']

            problem = metadata.get('problem', 'Unknown')
            dataset = metadata.get('dataset')
            if dataset is None and problem.lower() in {'logistic', 'mlp'}:
                dataset = 'mnist'
            dataset_key = (dataset or 'unknown').lower()
            optimizer = metadata.get('optimizer', 'Unknown')

            key = (problem, dataset_key)

            if key not in raw_data:
                raw_data[key] = {}
            if optimizer not in raw_data[key]:
                raw_data[key][optimizer] = []

            # Extract metrics for this run
            metrics = {
                'iterations': [m['iteration'] for m in metrics_history],
                'losses': [m.get('train_loss', np.nan) for m in metrics_history],
                'train_accuracies': [m.get('train_accuracy', np.nan) for m in metrics_history if 'train_accuracy' in m],
                'test_accuracies': [m.get('test_accuracy', np.nan) for m in metrics_history if 'test_accuracy' in m],
                'gradient_norms': [m.get('gradient_norm', np.nan) for m in metrics_history if 'gradient_norm' in m],
                'wall_times': [],
            }

            # Compute wall times
            if metrics_history:
                start_time = metrics_history[0].get('timestamp', 0)
                metrics['wall_times'] = [m.get('timestamp', 0) - start_time for m in metrics_history]

            raw_data[key][optimizer].append(metrics)

    # Now average across seeds for each problem-optimizer pair
    organized = {}
    for key, optimizers in raw_data.items():
        organized[key] = {}
        for optimizer, runs in optimizers.items():
            # Get common iteration points (use the run with most iterations)
            max_len = max(len(run['iterations']) for run in runs)
            common_iterations = runs[0]['iterations'][:max_len]

            # Average each metric across runs
            averaged = {
                'iterations': common_iterations,
                'losses_mean': [],
                'losses_std': [],
                'test_accuracies_mean': [],
                'test_accuracies_std': [],
                'gradient_norms_mean': [],
                'gradient_norms_std': [],
                'wall_times_mean': [],
            }

            # For each iteration point, average across seeds
            for i in range(len(common_iterations)):
                # Losses
                losses_at_i = [run['losses'][i] for run in runs if i < len(run['losses'])]
                averaged['losses_mean'].append(np.mean(losses_at_i))
                averaged['losses_std'].append(np.std(losses_at_i))

                # Test accuracies
                test_accs = [run['test_accuracies'][i] for run in runs if i < len(run['test_accuracies'])]
                if test_accs:
                    averaged['test_accuracies_mean'].append(np.mean(test_accs))
                    averaged['test_accuracies_std'].append(np.std(test_accs))

                # Gradient norms
                grad_norms = [run['gradient_norms'][i] for run in runs if i < len(run['gradient_norms'])]
                if grad_norms:
                    averaged['gradient_norms_mean'].append(np.mean(grad_norms))
                    averaged['gradient_norms_std'].append(np.std(grad_norms))

                # Wall times
                times = [run['wall_times'][i] for run in runs if i < len(run['wall_times'])]
                if times:
                    averaged['wall_times_mean'].append(np.mean(times))

            organized[key][optimizer] = averaged

    return organized


def generate_convergence_figures(
    results: Dict,
    save_dir: str = "reports/figures"
) -> None:
    """Generate convergence plots for each problem.

    Args:
        results: Dictionary of experiment results
        save_dir: Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    organized = organize_results_by_problem(results)

    for (problem, dataset_key), optimizer_results in organized.items():
        dataset_suffix = '' if dataset_key in {'mnist', 'unknown'} else f"_{dataset_key}"
        dataset_title = '' if dataset_key in {'mnist', 'unknown'} else f" ({dataset_key.replace('_', ' ').title()})"

        # Loss vs iterations
        plot_loss_curves(
            optimizer_results,
            save_path=str(save_path / f"{problem}{dataset_suffix}_loss_iterations.pdf"),
            title=f"Loss vs Iterations - {problem.upper()}{dataset_title}",
            use_log_scale=False
        )
        plt.close()

        # Loss vs iterations (log scale)
        plot_loss_curves(
            optimizer_results,
            save_path=str(save_path / f"{problem}{dataset_suffix}_loss_iterations_log.pdf"),
            title=f"Loss vs Iterations (Log Scale) - {problem.upper()}{dataset_title}",
            use_log_scale=True
        )
        plt.close()

        # Loss vs time
        plot_loss_curves(
            optimizer_results,
            save_path=str(save_path / f"{problem}{dataset_suffix}_loss_time.pdf"),
            title=f"Loss vs Time - {problem.upper()}{dataset_title}",
            use_time=True
        )
        plt.close()

        # Accuracy curves (if available)
        if any(('test_accuracies_mean' in m and m['test_accuracies_mean']) or
               ('test_accuracies' in m and m['test_accuracies'])
               for m in optimizer_results.values()):
            # Full range accuracy plot
            plot_accuracy_curves(
                optimizer_results,
                save_path=str(save_path / f"{problem}{dataset_suffix}_accuracy.pdf"),
                title=f"Test Accuracy vs Iterations - {problem.upper()}{dataset_title}",
                metric='test_accuracies'
            )
            plt.close()

            # Zoomed-in accuracy plot (focus on final performance)
            plot_accuracy_curves(
                optimizer_results,
                save_path=str(save_path / f"{problem}{dataset_suffix}_accuracy_zoomed.pdf"),
                title=f"Test Accuracy (Final Performance) - {problem.upper()}{dataset_title}",
                metric='test_accuracies',
                ylim=(0.75, 0.93),
                start_iter=50
            )
            plt.close()

        # Gradient norms (if available)
        if any(('gradient_norms_mean' in m and m['gradient_norms_mean']) or
               ('gradient_norms' in m and m['gradient_norms'])
               for m in optimizer_results.values()):
            plot_gradient_norms(
                optimizer_results,
                save_path=str(save_path / f"{problem}{dataset_suffix}_gradient_norms.pdf"),
            )
            plt.close()

    print(f"Convergence figures saved to {save_dir}")


def generate_trajectory_figures(
    results: Dict,
    save_dir: str = "reports/figures"
) -> None:
    """Generate trajectory plots for 2D toy problems.

    Args:
        results: Dictionary of experiment results
        save_dir: Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Look for 2D optimization trajectories
    for exp_name, exp_results in results.items():
        trajectories = {}

        for result in exp_results:
            metadata = result['metadata']
            metrics_history = result['metrics_history']

            # Check if this is a 2D problem
            if metadata.get('num_parameters') == 2 and 'param_trajectory' in metadata:
                optimizer = metadata.get('optimizer', 'Unknown')
                trajectory = np.array(metadata['param_trajectory'])
                trajectories[optimizer] = trajectory

        if trajectories:
            # Determine the loss function
            problem_type = metadata.get('problem', '').lower()

            if 'rosenbrock' in problem_type:
                from ..problems.toy_functions import Rosenbrock
                loss_fn = Rosenbrock().loss
                bounds = (-2, 2, -1, 3)
                title = "Optimization Trajectories - Rosenbrock Function"
            elif 'beale' in problem_type:
                from ..problems.toy_functions import Beale
                loss_fn = Beale().loss
                bounds = (-4.5, 4.5, -4.5, 4.5)
                title = "Optimization Trajectories - Beale Function"
            else:
                continue

            plot_optimization_trajectory(
                trajectories,
                loss_fn,
                bounds=bounds,
                save_path=str(save_path / f"{problem_type}_trajectories.pdf"),
                title=title
            )
            plt.close()

    print(f"Trajectory figures saved to {save_dir}")


def generate_all_figures(log_dir: str = "experiments/logs", save_dir: str = "reports/figures") -> None:
    """Generate all analysis figures.

    Args:
        log_dir: Directory containing experiment logs
        save_dir: Directory to save figures
    """
    print("Loading experiment results...")
    results = load_experiment_results(log_dir)

    if not results:
        print("No experiment results found!")
        return

    print(f"\nFound {len(results)} experiment groups")

    print("\nGenerating convergence figures...")
    generate_convergence_figures(results, save_dir)

    print("\nGenerating trajectory figures...")
    generate_trajectory_figures(results, save_dir)

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    generate_all_figures()
