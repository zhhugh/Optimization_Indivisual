"""Base class for optimization problems."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems.

    All problems should inherit from this class and implement the required methods.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize problem.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for given parameters and data.

        Args:
            params: Model parameters
            X: Input data of shape (n_samples, input_dim)
            y: Target data of shape (n_samples,) or (n_samples, output_dim)

        Returns:
            Loss value
        """
        pass

    @abstractmethod
    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to parameters.

        Args:
            params: Model parameters
            X: Input data of shape (n_samples, input_dim)
            y: Target data of shape (n_samples,) or (n_samples, output_dim)

        Returns:
            Gradient of loss with respect to parameters
        """
        pass

    @abstractmethod
    def predict(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Make predictions for given parameters and data.

        Args:
            params: Model parameters
            X: Input data of shape (n_samples, input_dim)

        Returns:
            Predictions
        """
        pass

    def evaluate(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on data.

        Args:
            params: Model parameters
            X: Input data
            y: Target data

        Returns:
            Dictionary of evaluation metrics
        """
        loss = self.loss(params, X, y)
        predictions = self.predict(params, X)

        # Compute accuracy for classification
        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            # Binary classification
            pred_classes = (predictions.flatten() > 0.5).astype(int)
            true_classes = y.flatten().astype(int)
            accuracy = np.mean(pred_classes == true_classes)
        else:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            if y.ndim == 1:
                true_classes = y.astype(int)
            else:
                true_classes = np.argmax(y, axis=1)
            accuracy = np.mean(pred_classes == true_classes)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }

    @abstractmethod
    def initialize_params(self, seed: int = 42) -> np.ndarray:
        """Initialize parameters.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initialized parameters
        """
        pass

    @abstractmethod
    def num_params(self) -> int:
        """Return number of parameters in the model.

        Returns:
            Number of parameters
        """
        pass
