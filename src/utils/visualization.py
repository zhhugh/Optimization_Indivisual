"""Visualization utilities for optimization results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

OPTIMIZER_COLORS = {
    'adamom': '#d7191c',
    'adam': '#2b83ba',
    'adamw': '#1a9641',
    'rmsprop': '#abdda4',
    'adagrad': '#fdae61',
    'momentum': '#808080',
    'nesterov': '#fee08b',
    'sgd': '#5e4fa2',
}


def _display_name(name: str) -> str:
    return 'AdaMom (Ours)' if name.lower() == 'adamom' else name.title()


def _format_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.9)
    ax.set_facecolor('white')


def plot_loss_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Loss vs Iterations",
    use_log_scale: bool = False,
    use_time: bool = False
) -> None:
    plt.figure(figsize=(5.4, 3.3))

    for name, metrics in results.items():
        base = name.lower()
        color = OPTIMIZER_COLORS.get(base)

        if use_time and 'wall_times_mean' in metrics:
            x = metrics['wall_times_mean']
            xlabel = 'Wall-clock time (s)'
        else:
            x = metrics['iterations']
            xlabel = 'Iterations'

        if 'losses_mean' in metrics and metrics['losses_mean']:
            y_mean = metrics['losses_mean']
            y_std = metrics.get('losses_std', [0] * len(y_mean))
            line = plt.plot(
                x, y_mean,
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0,
            )
            fill_color = color or line[0].get_color()
            plt.fill_between(
                x,
                np.array(y_mean) - np.array(y_std),
                np.array(y_mean) + np.array(y_std),
                alpha=0.18,
                color=fill_color
            )
        elif 'losses' in metrics:
            plt.plot(
                x,
                metrics['losses'],
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0
            )

    if use_log_scale:
        plt.yscale('log')

    plt.xlabel(xlabel)
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend(loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    ax = plt.gca()
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_accuracy_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Accuracy vs Iterations",
    metric: str = 'test_accuracies',
    ylim: Optional[Tuple[float, float]] = None,
    start_iter: int = 0
) -> None:
    plt.figure(figsize=(5.4, 3.3))

    for name, metrics in results.items():
        base = name.lower()
        color = OPTIMIZER_COLORS.get(base)
        metric_mean_key = f'{metric}_mean'
        metric_std_key = f'{metric}_std'

        if metric_mean_key in metrics and metrics[metric_mean_key]:
            x_full = metrics['iterations'][:len(metrics[metric_mean_key])]
            y_mean_full = metrics[metric_mean_key]
            y_std_full = metrics.get(metric_std_key, [0] * len(y_mean_full))

            if start_iter > 0:
                indices = [i for i, val in enumerate(x_full) if val >= start_iter]
                if not indices:
                    continue
                x = [x_full[i] for i in indices]
                y_mean = [y_mean_full[i] for i in indices]
                y_std = [y_std_full[i] for i in indices]
            else:
                x = x_full
                y_mean = y_mean_full
                y_std = y_std_full

            line = plt.plot(
                x, y_mean,
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0,
            )
            fill_color = color or line[0].get_color()
            plt.fill_between(
                x,
                np.array(y_mean) - np.array(y_std),
                np.array(y_mean) + np.array(y_std),
                alpha=0.18,
                color=fill_color
            )
        elif metric in metrics and metrics[metric]:
            x_full = metrics['iterations'][:len(metrics[metric])]
            y_full = metrics[metric]
            if start_iter > 0:
                indices = [i for i, val in enumerate(x_full) if val >= start_iter]
                if not indices:
                    continue
                x = [x_full[i] for i in indices]
                y = [y_full[i] for i in indices]
            else:
                x = x_full
                y = y_full
            plt.plot(
                x, y,
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0
            )

    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend(loc='lower right' if not ylim else 'best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')

    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([0.1, 1.0])

    ax = plt.gca()
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_optimization_trajectory(
    trajectories: Dict[str, np.ndarray],
    loss_function,
    bounds: Tuple[float, float, float, float] = (-2, 2, -1, 3),
    save_path: Optional[str] = None,
    title: str = "Optimization Trajectories"
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_function(np.array([X[i, j], Y[i, j]]))

    levels = np.logspace(np.log10(Z.min() + 1e-8), np.log10(Z.max()), 18)
    contour = ax.contour(
        X, Y, Z,
        levels=levels,
        colors='black',
        linewidths=0.6,
        alpha=1.0
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1e', colors='black')

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for (name, trajectory), color in zip(trajectories.items(), colors):
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            '-', color=color, linewidth=2.0, alpha=1.0,
            label=_display_name(name)
        )
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, markersize=5)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, markersize=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_gradient_norms(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    use_log_scale: bool = True
) -> None:
    plt.figure(figsize=(5.4, 3.3))

    for name, metrics in results.items():
        base = name.lower()
        color = OPTIMIZER_COLORS.get(base)

        if 'gradient_norms_mean' in metrics and metrics['gradient_norms_mean']:
            x = metrics['iterations'][:len(metrics['gradient_norms_mean'])]
            y_mean = metrics['gradient_norms_mean']
            y_std = metrics.get('gradient_norms_std', [0] * len(y_mean))
            line = plt.plot(
                x, y_mean,
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0
            )
            fill_color = color or line[0].get_color()
            plt.fill_between(
                x,
                np.array(y_mean) - np.array(y_std),
                np.array(y_mean) + np.array(y_std),
                alpha=0.18,
                color=fill_color
            )
        elif 'gradient_norms' in metrics and metrics['gradient_norms']:
            x = metrics['iterations'][:len(metrics['gradient_norms'])]
            plt.plot(
                x,
                metrics['gradient_norms'],
                label=_display_name(name),
                linewidth=2.0,
                color=color,
                alpha=1.0
            )

    if use_log_scale:
        plt.yscale('log')

    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Evolution')
    plt.legend(loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    ax = plt.gca()
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_learning_rate_sensitivity(
    results: Dict[str, Dict[float, Dict]],
    save_path: Optional[str] = None,
    metric: str = 'best_test_accuracy'
) -> None:
    plt.figure(figsize=(5.4, 3.2))

    for opt_name, lr_results in results.items():
        lrs = sorted(lr_results.keys())
        values = [lr_results[lr][metric] for lr in lrs]
        plt.plot(
            lrs, values, 'o-',
            label=_display_name(opt_name),
            linewidth=2.0,
            markersize=4,
            alpha=1.0
        )

    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title('Learning Rate Sensitivity')
    plt.legend(loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    ax = plt.gca()
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison(
    results: Dict[str, Dict],
    threshold: float,
    metric: str = 'loss',
    save_path: Optional[str] = None
) -> None:
    optimizers = []
    iterations = []
    for name, metrics in results.items():
        values = metrics['losses'] if metric == 'loss' else metrics.get('test_accuracies', [])
        if not values:
            continue
        if metric == 'loss':
            converged = [i for i, v in enumerate(values) if v <= threshold]
        else:
            converged = [i for i, v in enumerate(values) if v >= threshold]
        if converged:
            optimizers.append(_display_name(name))
            iterations.append(metrics['iterations'][converged[0]])

    plt.figure(figsize=(5.4, 3.0))
    x = np.arange(len(optimizers))
    plt.bar(x, iterations, color='#4c72b0', alpha=0.9)
    plt.xticks(x, optimizers, rotation=45, ha='right')
    plt.ylabel('Iterations to Convergence')
    plt.title(f'Convergence Speed (threshold={threshold})')
    plt.grid(True, axis='y', alpha=0.3)
    ax = plt.gca()
    _format_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
