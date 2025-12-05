"""RMSprop optimizer."""

import numpy as np
from typing import Dict, Any
from .base import Optimizer


class RMSprop(Optimizer):
    """RMSprop (Root Mean Square Propagation) optimizer.

    Addresses Adagrad's aggressive learning rate decay by using exponential
    moving average of squared gradients instead of cumulative sum.

    Update rule:
        E[g²]_t = β * E[g²]_{t-1} + (1-β) * ∇L(θ_t)²
        θ_{t+1} = θ_t - (η / √(E[g²]_t + ε)) * ∇L(θ_t)

    Reference:
        Tieleman & Hinton, "Lecture 6.5-rmsprop: Divide the gradient by
        a running average of its recent magnitude", COURSERA 2012

    Args:
        lr: Learning rate (default: 0.01)
        beta: Decay rate for moving average (default: 0.9)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 penalty coefficient (default: 0.0)
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.mean_squared_grads = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Initialize moving average on first step
        if self.mean_squared_grads is None:
            self.mean_squared_grads = np.zeros_like(params)

        # Add weight decay to gradients
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params

        # Update moving average of squared gradients
        self.mean_squared_grads = (
            self.beta * self.mean_squared_grads + (1 - self.beta) * grads ** 2
        )

        # Compute adaptive learning rate
        adapted_lr = self.lr / (np.sqrt(self.mean_squared_grads) + self.eps)

        # Update parameters
        params = params - adapted_lr * grads

        return params

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'beta': self.beta,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'mean_squared_grads': self.mean_squared_grads,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.beta = state['beta']
        self.eps = state['eps']
        self.weight_decay = state['weight_decay']
        self.mean_squared_grads = state['mean_squared_grads']

    def reset(self) -> None:
        super().reset()
        self.mean_squared_grads = None
