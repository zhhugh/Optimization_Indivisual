"""AdamW optimizer."""

import numpy as np
from typing import Dict, Any
from .base import Optimizer


class AdamW(Optimizer):
    """AdamW (Adam with decoupled Weight decay) optimizer.

    Fixes Adam's weight decay implementation by decoupling it from the
    gradient-based update. This leads to better generalization.

    Update rule:
        m_t = β₁ * m_{t-1} + (1-β₁) * ∇L(θ_t)
        v_t = β₂ * v_{t-1} + (1-β₂) * ∇L(θ_t)²
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)
        θ_{t+1} = θ_t - η * (m̂_t / (√v̂_t + ε) + λ * θ_t)

    Note the weight decay term λ * θ_t is applied directly to parameters,
    not added to gradients like in Adam.

    Reference:
        Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019

    Args:
        lr: Learning rate (default: 0.001)
        beta1: Decay rate for first moment (default: 0.9)
        beta2: Decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform a single optimization step.

        Args:
            params: Current parameters
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Update biased first moment estimate (no weight decay in gradients!)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters with decoupled weight decay
        # Key difference from Adam: weight decay is applied separately
        params = params - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params)

        return params

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': self.m,
            'v': self.v,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.weight_decay = state['weight_decay']
        self.m = state['m']
        self.v = state['v']

    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None
