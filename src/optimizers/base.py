"""Base optimizer class."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class Optimizer(ABC):
    """Abstract base class for all optimizers.

    All optimizers should inherit from this class and implement the step method.
    """

    def __init__(self, lr: float = 0.01):
        """Initialize optimizer.

        Args:
            lr: Learning rate
        """
        self.lr = lr
        self.t = 0  # Iteration counter

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dictionary.

        Returns:
            Dictionary containing optimizer state
        """
        return {
            'lr': self.lr,
            't': self.t,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from dictionary.

        Args:
            state: Dictionary containing optimizer state
        """
        self.lr = state['lr']
        self.t = state['t']

    def reset(self) -> None:
        """Reset optimizer state."""
        self.t = 0
