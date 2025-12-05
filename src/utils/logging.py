"""Experiment logging utilities."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


class ExperimentLogger:
    """Logger for tracking experiment metrics and results.

    Logs metrics at each iteration and saves results to JSON.
    """

    def __init__(self, experiment_name: str, save_dir: str = "experiments/logs"):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': time.time(),
        }

    def log_iteration(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a single iteration.

        Args:
            iteration: Iteration number
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            **metrics
        }
        self.metrics_history.append(log_entry)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log experiment metadata.

        Args:
            metadata: Dictionary of metadata
        """
        self.metadata.update(metadata)

    def get_metric_history(self, metric_name: str) -> List[Any]:
        """Get history of a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric values
        """
        return [entry.get(metric_name) for entry in self.metrics_history]

    def save_results(self, filename: str = None) -> str:
        """Save results to JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{timestamp}.json"

        filepath = self.save_dir / filename

        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']

        results = {
            'metadata': self.metadata,
            'metrics_history': self.metrics_history,
        }

        # Convert numpy types to Python types for JSON serialization
        results = self._convert_to_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")
        return str(filepath)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """Load results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary of results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
