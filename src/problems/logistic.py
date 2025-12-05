"""Logistic regression problem."""

import numpy as np
from typing import Dict
from .base import OptimizationProblem


class LogisticRegression(OptimizationProblem):
    """Logistic regression for binary or multi-class classification.

    Model: p(y=k|x) = softmax(W^T x + b)_k
    Loss: Cross-entropy loss with optional L2 regularization

    Args:
        input_dim: Dimension of input features
        output_dim: Number of classes (1 for binary, >1 for multi-class)
        reg: L2 regularization coefficient (default: 0.0)
    """

    def __init__(self, input_dim: int, output_dim: int = 1, reg: float = 0.0):
        super().__init__(input_dim, output_dim)
        self.reg = reg

        # For binary classification
        if output_dim == 1:
            self.n_params = input_dim + 1  # weights + bias
        else:
            self.n_params = (input_dim + 1) * output_dim

    def _unpack_params(self, params: np.ndarray) -> tuple:
        """Unpack flat parameter vector into weights and bias."""
        if self.output_dim == 1:
            W = params[:-1].reshape(-1, 1)
            b = params[-1]
        else:
            W = params[:-self.output_dim].reshape(self.input_dim, self.output_dim)
            b = params[-self.output_dim:]
        return W, b

    def _pack_params(self, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pack weights and bias into flat parameter vector."""
        return np.concatenate([W.flatten(), b.flatten()])

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax function."""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        W, b = self._unpack_params(params)
        n_samples = X.shape[0]

        # Compute predictions
        z = X @ W + b

        if self.output_dim == 1:
            # Binary classification
            probs = self._sigmoid(z)
            # Binary cross-entropy
            loss = -np.mean(y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15))
        else:
            # Multi-class classification
            probs = self._softmax(z)
            # One-hot encode y if needed
            if y.ndim == 1:
                y_onehot = np.zeros((n_samples, self.output_dim))
                y_onehot[np.arange(n_samples), y.astype(int)] = 1
            else:
                y_onehot = y
            # Cross-entropy
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))

        # Add L2 regularization
        if self.reg > 0:
            loss += 0.5 * self.reg * np.sum(W ** 2)

        return loss

    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of loss."""
        W, b = self._unpack_params(params)
        n_samples = X.shape[0]

        # Compute predictions
        z = X @ W + b

        if self.output_dim == 1:
            # Binary classification
            probs = self._sigmoid(z)
            error = probs - y.reshape(-1, 1)
        else:
            # Multi-class classification
            probs = self._softmax(z)
            # One-hot encode y if needed
            if y.ndim == 1:
                y_onehot = np.zeros((n_samples, self.output_dim))
                y_onehot[np.arange(n_samples), y.astype(int)] = 1
            else:
                y_onehot = y
            error = probs - y_onehot

        # Compute gradients
        grad_W = X.T @ error / n_samples
        grad_b = np.mean(error, axis=0)

        # Add regularization gradient
        if self.reg > 0:
            grad_W += self.reg * W

        return self._pack_params(grad_W, grad_b)

    def predict(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        W, b = self._unpack_params(params)
        z = X @ W + b

        if self.output_dim == 1:
            return self._sigmoid(z)
        else:
            return self._softmax(z)

    def initialize_params(self, seed: int = 42) -> np.ndarray:
        """Initialize parameters using Xavier initialization."""
        np.random.seed(seed)

        if self.output_dim == 1:
            scale = np.sqrt(2.0 / self.input_dim)
            W = np.random.randn(self.input_dim) * scale
            b = np.zeros(1)
        else:
            scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            W = np.random.randn(self.input_dim, self.output_dim) * scale
            b = np.zeros(self.output_dim)

        return self._pack_params(W, b)

    def num_params(self) -> int:
        """Return number of parameters."""
        return self.n_params
