"""Unit tests for optimization problems."""

import numpy as np
import pytest
from src.problems import LogisticRegression, MLP, Rosenbrock, Beale


class TestLogisticRegression:
    def test_initialization(self):
        problem = LogisticRegression(input_dim=10, output_dim=2)
        assert problem.input_dim == 10
        assert problem.output_dim == 2

    def test_parameter_initialization(self):
        problem = LogisticRegression(input_dim=10, output_dim=2)
        params = problem.initialize_params(seed=42)
        assert params.shape[0] == problem.num_params()

    def test_forward_pass(self):
        problem = LogisticRegression(input_dim=10, output_dim=2)
        params = problem.initialize_params()
        X = np.random.randn(5, 10)
        predictions = problem.predict(params, X)
        assert predictions.shape == (5, 2)
        # Softmax outputs should sum to 1
        assert np.allclose(predictions.sum(axis=1), 1.0)

    def test_loss_computation(self):
        problem = LogisticRegression(input_dim=10, output_dim=2)
        params = problem.initialize_params()
        X = np.random.randn(5, 10)
        y = np.random.randint(0, 2, 5)
        loss = problem.loss(params, X, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_gradient_computation(self):
        problem = LogisticRegression(input_dim=10, output_dim=2)
        params = problem.initialize_params()
        X = np.random.randn(5, 10)
        y = np.random.randint(0, 2, 5)
        grads = problem.gradient(params, X, y)
        assert grads.shape == params.shape


class TestMLP:
    def test_initialization(self):
        problem = MLP(input_dim=10, hidden_dims=[20, 15], output_dim=5)
        assert problem.input_dim == 10
        assert problem.output_dim == 5
        assert problem.hidden_dims == [20, 15]

    def test_parameter_count(self):
        problem = MLP(input_dim=10, hidden_dims=[20], output_dim=5)
        # 10*20 + 20 (first layer) + 20*5 + 5 (output layer)
        expected = 10*20 + 20 + 20*5 + 5
        assert problem.num_params() == expected

    def test_forward_pass(self):
        problem = MLP(input_dim=10, hidden_dims=[20], output_dim=5)
        params = problem.initialize_params()
        X = np.random.randn(3, 10)
        predictions = problem.predict(params, X)
        assert predictions.shape == (3, 5)
        # Softmax outputs should sum to 1
        assert np.allclose(predictions.sum(axis=1), 1.0)

    def test_loss_and_gradient(self):
        problem = MLP(input_dim=10, hidden_dims=[20], output_dim=5)
        params = problem.initialize_params()
        X = np.random.randn(3, 10)
        y = np.random.randint(0, 5, 3)

        loss = problem.loss(params, X, y)
        grads = problem.gradient(params, X, y)

        assert isinstance(loss, float)
        assert loss > 0
        assert grads.shape == params.shape


class TestRosenbrock:
    def test_initialization(self):
        problem = Rosenbrock(a=1.0, b=100.0)
        assert problem.a == 1.0
        assert problem.b == 100.0
        assert problem.num_params() == 2

    def test_global_minimum(self):
        problem = Rosenbrock(a=1.0, b=100.0)
        # Global minimum is at (1, 1) with value 0
        params = np.array([1.0, 1.0])
        loss = problem.loss(params)
        assert abs(loss) < 1e-10

    def test_gradient_at_minimum(self):
        problem = Rosenbrock(a=1.0, b=100.0)
        params = np.array([1.0, 1.0])
        grads = problem.gradient(params)
        # Gradient should be close to zero at minimum
        assert np.allclose(grads, 0.0, atol=1e-10)

    def test_gradient_nonzero(self):
        problem = Rosenbrock(a=1.0, b=100.0)
        params = np.array([0.0, 0.0])
        grads = problem.gradient(params)
        # Gradient should be non-zero away from minimum
        assert not np.allclose(grads, 0.0)


class TestBeale:
    def test_initialization(self):
        problem = Beale()
        assert problem.num_params() == 2

    def test_global_minimum(self):
        problem = Beale()
        # Global minimum is at (3, 0.5) with value 0
        params = np.array([3.0, 0.5])
        loss = problem.loss(params)
        assert abs(loss) < 1e-10

    def test_gradient_shape(self):
        problem = Beale()
        params = np.array([0.0, 0.0])
        grads = problem.gradient(params)
        assert grads.shape == (2,)
