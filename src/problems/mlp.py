"""Multi-layer perceptron problem."""

import numpy as np
from typing import List, Dict
from .base import OptimizationProblem


class MLP(OptimizationProblem):
    """Multi-layer perceptron (feedforward neural network).

    Architecture: input -> hidden_1 -> ... -> hidden_n -> output
    Activation: ReLU for hidden layers, softmax for output

    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes
        reg: L2 regularization coefficient (default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        reg: float = 0.0
    ):
        super().__init__(input_dim, output_dim)
        self.hidden_dims = hidden_dims
        self.reg = reg

        # Build layer dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.n_layers = len(self.layer_dims) - 1

        # Compute total number of parameters
        self.n_params = 0
        self.param_shapes = []
        for i in range(self.n_layers):
            W_shape = (self.layer_dims[i], self.layer_dims[i + 1])
            b_shape = (self.layer_dims[i + 1],)
            self.param_shapes.append((W_shape, b_shape))
            self.n_params += np.prod(W_shape) + np.prod(b_shape)

    def _unpack_params(self, params: np.ndarray) -> List[tuple]:
        """Unpack flat parameter vector into list of (W, b) tuples."""
        layers = []
        idx = 0
        for W_shape, b_shape in self.param_shapes:
            W_size = np.prod(W_shape)
            b_size = np.prod(b_shape)

            W = params[idx:idx + W_size].reshape(W_shape)
            idx += W_size

            b = params[idx:idx + b_size].reshape(b_shape)
            idx += b_size

            layers.append((W, b))

        return layers

    def _pack_params(self, layers: List[tuple]) -> np.ndarray:
        """Pack list of (W, b) tuples into flat parameter vector."""
        params = []
        for W, b in layers:
            params.append(W.flatten())
            params.append(b.flatten())
        return np.concatenate(params)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (z > 0).astype(float)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax function."""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, layers: List[tuple], X: np.ndarray) -> tuple:
        """Forward pass through the network.

        Returns:
            activations: List of activations for each layer
            pre_activations: List of pre-activations (before activation function)
        """
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(layers):
            z = activations[-1] @ W + b
            pre_activations.append(z)

            if i < len(layers) - 1:
                # Hidden layers: ReLU
                a = self._relu(z)
            else:
                # Output layer: softmax
                a = self._softmax(z)

            activations.append(a)

        return activations, pre_activations

    def loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        layers = self._unpack_params(params)
        n_samples = X.shape[0]

        # Forward pass
        activations, _ = self._forward(layers, X)
        probs = activations[-1]

        # One-hot encode y if needed
        if y.ndim == 1:
            y_onehot = np.zeros((n_samples, self.output_dim))
            y_onehot[np.arange(n_samples), y.astype(int)] = 1
        else:
            y_onehot = y

        # Cross-entropy loss
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))

        # Add L2 regularization
        if self.reg > 0:
            for W, _ in layers:
                loss += 0.5 * self.reg * np.sum(W ** 2)

        return loss

    def gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient using backpropagation."""
        layers = self._unpack_params(params)
        n_samples = X.shape[0]

        # Forward pass
        activations, pre_activations = self._forward(layers, X)
        probs = activations[-1]

        # One-hot encode y if needed
        if y.ndim == 1:
            y_onehot = np.zeros((n_samples, self.output_dim))
            y_onehot[np.arange(n_samples), y.astype(int)] = 1
        else:
            y_onehot = y

        # Backward pass
        gradients = []

        # Output layer gradient
        delta = probs - y_onehot

        for i in range(len(layers) - 1, -1, -1):
            W, b = layers[i]

            # Gradient with respect to W and b
            grad_W = activations[i].T @ delta / n_samples
            grad_b = np.mean(delta, axis=0)

            # Add regularization gradient
            if self.reg > 0:
                grad_W += self.reg * W

            gradients.insert(0, (grad_W, grad_b))

            # Backpropagate to previous layer
            if i > 0:
                delta = (delta @ W.T) * self._relu_derivative(pre_activations[i - 1])

        return self._pack_params(gradients)

    def predict(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        layers = self._unpack_params(params)
        activations, _ = self._forward(layers, X)
        return activations[-1]

    def initialize_params(self, seed: int = 42) -> np.ndarray:
        """Initialize parameters using He initialization for ReLU."""
        np.random.seed(seed)
        layers = []

        for i in range(self.n_layers):
            # He initialization for ReLU
            scale = np.sqrt(2.0 / self.layer_dims[i])
            W = np.random.randn(self.layer_dims[i], self.layer_dims[i + 1]) * scale
            b = np.zeros(self.layer_dims[i + 1])
            layers.append((W, b))

        return self._pack_params(layers)

    def num_params(self) -> int:
        """Return number of parameters."""
        return self.n_params
