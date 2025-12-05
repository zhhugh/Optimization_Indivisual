"""2D toy functions for visualization."""

import numpy as np
from .base import OptimizationProblem


class Rosenbrock(OptimizationProblem):
    """Rosenbrock function (banana function).

    A classic non-convex optimization test function.
    f(x, y) = (a - x)^2 + b(y - x^2)^2

    Global minimum at (a, a^2) with value 0.
    Default: a=1, b=100, minimum at (1, 1)

    Args:
        a: Parameter a (default: 1.0)
        b: Parameter b (default: 100.0)
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__(input_dim=2, output_dim=1)
        self.a = a
        self.b = b

    def loss(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> float:
        """Compute Rosenbrock function value."""
        x, y = params[0], params[1]
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def gradient(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """Compute gradient of Rosenbrock function."""
        x, y = params[0], params[1]
        dx = -2 * (self.a - x) - 4 * self.b * x * (y - x ** 2)
        dy = 2 * self.b * (y - x ** 2)
        return np.array([dx, dy])

    def predict(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Not applicable for toy functions."""
        raise NotImplementedError("Predict not applicable for toy functions")

    def evaluate(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> dict:
        """Evaluate function value."""
        return {'loss': self.loss(params)}

    def initialize_params(self, seed: int = 42) -> np.ndarray:
        """Initialize parameters randomly."""
        np.random.seed(seed)
        # Start away from minimum
        return np.random.randn(2) * 2 + np.array([-1.0, 1.0])

    def num_params(self) -> int:
        """Return number of parameters."""
        return 2


class Beale(OptimizationProblem):
    """Beale function.

    Another classic non-convex optimization test function.
    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Global minimum at (3, 0.5) with value 0.

    """

    def __init__(self):
        super().__init__(input_dim=2, output_dim=1)

    def loss(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> float:
        """Compute Beale function value."""
        x, y = params[0], params[1]
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y ** 2) ** 2
        term3 = (2.625 - x + x * y ** 3) ** 2
        return term1 + term2 + term3

    def gradient(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """Compute gradient of Beale function."""
        x, y = params[0], params[1]

        # Partial derivative with respect to x
        dx = (
            2 * (1.5 - x + x * y) * (-1 + y) +
            2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2) +
            2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3)
        )

        # Partial derivative with respect to y
        dy = (
            2 * (1.5 - x + x * y) * x +
            2 * (2.25 - x + x * y ** 2) * (2 * x * y) +
            2 * (2.625 - x + x * y ** 3) * (3 * x * y ** 2)
        )

        return np.array([dx, dy])

    def predict(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Not applicable for toy functions."""
        raise NotImplementedError("Predict not applicable for toy functions")

    def evaluate(self, params: np.ndarray, X: np.ndarray = None, y: np.ndarray = None) -> dict:
        """Evaluate function value."""
        return {'loss': self.loss(params)}

    def initialize_params(self, seed: int = 42) -> np.ndarray:
        """Initialize parameters randomly."""
        np.random.seed(seed)
        # Start away from minimum
        return np.random.randn(2) * 2 + np.array([0.0, 0.0])

    def num_params(self) -> int:
        """Return number of parameters."""
        return 2
