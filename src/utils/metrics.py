"""Metrics tracking utilities."""

import numpy as np
from typing import Dict, List
import time


class MetricsTracker:
    """Track and compute various optimization metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.gradient_norms = []
        self.param_norms = []
        self.iterations = []
        self.wall_times = []
        self.start_time = None

    def start_timing(self) -> None:
        """Start timing the optimization."""
        self.start_time = time.time()

    def record(
        self,
        iteration: int,
        loss: float,
        train_accuracy: float = None,
        test_accuracy: float = None,
        gradient_norm: float = None,
        param_norm: float = None
    ) -> None:
        """Record metrics for an iteration.

        Args:
            iteration: Iteration number
            loss: Loss value
            train_accuracy: Training accuracy (optional)
            test_accuracy: Test accuracy (optional)
            gradient_norm: Gradient norm (optional)
            param_norm: Parameter norm (optional)
        """
        self.iterations.append(iteration)
        self.losses.append(loss)

        if train_accuracy is not None:
            self.train_accuracies.append(train_accuracy)

        if test_accuracy is not None:
            self.test_accuracies.append(test_accuracy)

        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

        if param_norm is not None:
            self.param_norms.append(param_norm)

        if self.start_time is not None:
            self.wall_times.append(time.time() - self.start_time)

    def get_convergence_iteration(self, threshold: float, metric: str = 'loss') -> int:
        """Get iteration when metric first reaches threshold.

        Args:
            threshold: Threshold value
            metric: Metric name ('loss', 'train_accuracy', 'test_accuracy')

        Returns:
            Iteration number, or -1 if threshold never reached
        """
        if metric == 'loss':
            values = self.losses
            compare = lambda x: x <= threshold
        elif metric == 'train_accuracy':
            values = self.train_accuracies
            compare = lambda x: x >= threshold
        elif metric == 'test_accuracy':
            values = self.test_accuracies
            compare = lambda x: x >= threshold
        else:
            raise ValueError(f"Unknown metric: {metric}")

        for i, val in enumerate(values):
            if compare(val):
                return self.iterations[i]

        return -1  # Threshold never reached

    def get_best_value(self, metric: str = 'loss') -> tuple:
        """Get best value and iteration for a metric.

        Args:
            metric: Metric name

        Returns:
            (best_value, iteration)
        """
        if metric == 'loss':
            values = self.losses
            best_idx = np.argmin(values)
        elif metric == 'train_accuracy':
            values = self.train_accuracies
            best_idx = np.argmax(values)
        elif metric == 'test_accuracy':
            values = self.test_accuracies
            best_idx = np.argmax(values)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return values[best_idx], self.iterations[best_idx]

    def get_statistics(self) -> Dict[str, float]:
        """Compute summary statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'final_loss': self.losses[-1] if self.losses else None,
            'best_loss': min(self.losses) if self.losses else None,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
            'best_train_accuracy': max(self.train_accuracies) if self.train_accuracies else None,
            'final_test_accuracy': self.test_accuracies[-1] if self.test_accuracies else None,
            'best_test_accuracy': max(self.test_accuracies) if self.test_accuracies else None,
            'total_iterations': len(self.iterations),
            'total_time': self.wall_times[-1] if self.wall_times else None,
            'avg_time_per_iteration': np.mean(np.diff(self.wall_times)) if len(self.wall_times) > 1 else None,
        }

        # Loss variance (stability metric)
        if len(self.losses) > 10:
            # Compute variance of losses in final 20% of iterations
            n_final = max(10, len(self.losses) // 5)
            stats['final_loss_variance'] = np.var(self.losses[-n_final:])

        # Gradient norm statistics
        if self.gradient_norms:
            stats['final_gradient_norm'] = self.gradient_norms[-1]
            stats['mean_gradient_norm'] = np.mean(self.gradient_norms)
            stats['max_gradient_norm'] = np.max(self.gradient_norms)

        return stats

    def to_dict(self) -> Dict[str, List]:
        """Convert all metrics to dictionary.

        Returns:
            Dictionary of metric lists
        """
        return {
            'iterations': self.iterations,
            'losses': self.losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'gradient_norms': self.gradient_norms,
            'param_norms': self.param_norms,
            'wall_times': self.wall_times,
        }
