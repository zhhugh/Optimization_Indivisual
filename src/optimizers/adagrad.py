"""Adagrad optimizer."""

import numpy as np
from typing import Dict, Any
from .base import Optimizer


class Adagrad(Optimizer):
    """Adagrad (Adaptive Gradient) optimizer.

    Adapts the learning rate for each parameter based on historical gradient information.
    Useful for sparse gradients but can be too aggressive in reducing learning rates.

    Update rule:
        G_t = G_{t-1} + ∇L(θ_t)²
        θ_{t+1} = θ_t - (η / √(G_t + ε)) * ∇L(θ_t)

    Reference:
        Duchi et al., "Adaptive Subgradient Methods for Online Learning
        and Stochastic Optimization", JMLR 2011

    Args:
        lr: Learning rate (default: 0.01)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 penalty coefficient (default: 0.0)
    """

    def __init__(self, lr: float = 0.01, eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr)
        self.eps = eps
        self.weight_decay = weight_decay
        self.sum_squared_grads = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Initialize accumulator on first step
        if self.sum_squared_grads is None:
            self.sum_squared_grads = np.zeros_like(params)

        # Add weight decay to gradients
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params

        # Accumulate squared gradients
        self.sum_squared_grads += grads ** 2

        # Compute adaptive learning rate
        adapted_lr = self.lr / (np.sqrt(self.sum_squared_grads) + self.eps)

        # Update parameters
        params = params - adapted_lr * grads

        return params

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'sum_squared_grads': self.sum_squared_grads,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.eps = state['eps']
        self.weight_decay = state['weight_decay']
        self.sum_squared_grads = state['sum_squared_grads']

    def reset(self) -> None:
        super().reset()
        self.sum_squared_grads = None
