"""Stochastic Gradient Descent optimizer."""

import numpy as np
from .base import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Implements vanilla SGD with optional weight decay.

    Update rule:
        θ_{t+1} = θ_t - η * ∇L(θ_t)

    Args:
        lr: Learning rate (default: 0.01)
        weight_decay: L2 penalty coefficient (default: 0.0)
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Add weight decay to gradients
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params

        # Update parameters
        params = params - self.lr * grads

        return params

    def state_dict(self):
        state = super().state_dict()
        state['weight_decay'] = self.weight_decay
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.weight_decay = state['weight_decay']
