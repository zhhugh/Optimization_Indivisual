"""Nesterov Accelerated Gradient optimizer."""

import numpy as np
from typing import Dict, Any
from .base import Optimizer


class Nesterov(Optimizer):
    """Nesterov Accelerated Gradient (NAG) optimizer.

    Implements Nesterov momentum which computes gradient at the "look-ahead" position.

    Update rule:
        v_t = β * v_{t-1} + ∇L(θ_t - β * v_{t-1})
        θ_{t+1} = θ_t - η * v_t

    Simplified form used here:
        v_t = β * v_{t-1} + ∇L(θ_t)
        θ_{t+1} = θ_t - η * (β * v_t + ∇L(θ_t))

    Args:
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient β (default: 0.9)
        weight_decay: L2 penalty coefficient (default: 0.0)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # Add weight decay to gradients
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params

        # Update velocity
        self.velocity = self.momentum * self.velocity + grads

        # Nesterov update: look-ahead
        params = params - self.lr * (self.momentum * self.velocity + grads)

        return params

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocity': self.velocity,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.momentum = state['momentum']
        self.weight_decay = state['weight_decay']
        self.velocity = state['velocity']

    def reset(self) -> None:
        super().reset()
        self.velocity = None
