"""Unit tests for optimizers."""

import numpy as np
import pytest
from src.optimizers import SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, AdaMom


@pytest.fixture
def simple_params():
    """Simple parameter array for testing."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def simple_grads():
    """Simple gradient array for testing."""
    return np.array([0.1, 0.2, 0.3])


class TestSGD:
    def test_initialization(self):
        opt = SGD(lr=0.01)
        assert opt.lr == 0.01
        assert opt.t == 0

    def test_step_decreases_params(self, simple_params, simple_grads):
        opt = SGD(lr=0.1)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert np.all(new_params < simple_params)

    def test_iteration_counter(self, simple_params, simple_grads):
        opt = SGD(lr=0.01)
        assert opt.t == 0
        opt.step(simple_params, simple_grads)
        assert opt.t == 1


class TestMomentum:
    def test_initialization(self):
        opt = Momentum(lr=0.01, momentum=0.9)
        assert opt.lr == 0.01
        assert opt.momentum == 0.9
        assert opt.velocity is None

    def test_velocity_initialization(self, simple_params, simple_grads):
        opt = Momentum(lr=0.01)
        opt.step(simple_params.copy(), simple_grads)
        assert opt.velocity is not None
        assert opt.velocity.shape == simple_params.shape

    def test_step(self, simple_params, simple_grads):
        opt = Momentum(lr=0.1, momentum=0.9)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape
        assert np.all(new_params < simple_params)


class TestNesterov:
    def test_initialization(self):
        opt = Nesterov(lr=0.01, momentum=0.9)
        assert opt.lr == 0.01
        assert opt.momentum == 0.9

    def test_step(self, simple_params, simple_grads):
        opt = Nesterov(lr=0.1, momentum=0.9)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape


class TestAdagrad:
    def test_initialization(self):
        opt = Adagrad(lr=0.01)
        assert opt.lr == 0.01
        assert opt.sum_squared_grads is None

    def test_accumulator_initialization(self, simple_params, simple_grads):
        opt = Adagrad(lr=0.01)
        opt.step(simple_params.copy(), simple_grads)
        assert opt.sum_squared_grads is not None
        assert np.all(opt.sum_squared_grads > 0)

    def test_step(self, simple_params, simple_grads):
        opt = Adagrad(lr=0.1)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape


class TestRMSprop:
    def test_initialization(self):
        opt = RMSprop(lr=0.01, beta=0.9)
        assert opt.lr == 0.01
        assert opt.beta == 0.9

    def test_step(self, simple_params, simple_grads):
        opt = RMSprop(lr=0.01)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape


class TestAdam:
    def test_initialization(self):
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        assert opt.lr == 0.001
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999
        assert opt.m is None
        assert opt.v is None

    def test_moments_initialization(self, simple_params, simple_grads):
        opt = Adam(lr=0.001)
        opt.step(simple_params.copy(), simple_grads)
        assert opt.m is not None
        assert opt.v is not None

    def test_step(self, simple_params, simple_grads):
        opt = Adam(lr=0.001)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape

    def test_bias_correction(self, simple_params, simple_grads):
        opt = Adam(lr=0.001)
        # First step should apply bias correction
        opt.step(simple_params.copy(), simple_grads)
        assert opt.t == 1


class TestAdamW:
    def test_initialization(self):
        opt = AdamW(lr=0.001, weight_decay=0.01)
        assert opt.lr == 0.001
        assert opt.weight_decay == 0.01

    def test_step(self, simple_params, simple_grads):
        opt = AdamW(lr=0.001, weight_decay=0.01)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape


class TestAdaMom:
    """Test our novel AdaMom optimizer."""

    def test_initialization(self):
        opt = AdaMom(lr=0.01, momentum=0.9)
        assert opt.lr == 0.01
        assert opt.momentum == 0.9
        assert opt.sum_squared_grads is None
        assert opt.velocity is None

    def test_state_initialization(self, simple_params, simple_grads):
        opt = AdaMom(lr=0.01)
        opt.step(simple_params.copy(), simple_grads)
        assert opt.sum_squared_grads is not None
        assert opt.velocity is not None

    def test_step(self, simple_params, simple_grads):
        opt = AdaMom(lr=0.01, momentum=0.9)
        new_params = opt.step(simple_params.copy(), simple_grads)
        assert new_params.shape == simple_params.shape
        assert np.all(new_params < simple_params)  # Should decrease with positive grads

    def test_convergence(self):
        """Test that AdaMom can minimize a simple quadratic."""
        params = np.array([10.0])
        opt = AdaMom(lr=0.5, momentum=0.9)  # Higher LR for faster convergence

        for _ in range(200):  # More iterations for hybrid method
            grads = 2 * params  # gradient of x^2
            params = opt.step(params, grads)

        assert abs(params[0]) < 0.5  # Should converge close to 0

    def test_combines_adagrad_and_momentum(self):
        """Verify AdaMom actually combines both components."""
        opt = AdaMom(lr=0.1, momentum=0.9)
        params = np.array([5.0, 5.0])

        # First step
        grads1 = np.array([1.0, 2.0])
        params1 = opt.step(params.copy(), grads1)

        # Check that sum_squared_grads was updated (Adagrad component)
        assert opt.sum_squared_grads is not None
        assert np.allclose(opt.sum_squared_grads, grads1 ** 2)

        # Second step - velocity should incorporate previous step (momentum component)
        grads2 = np.array([1.0, 2.0])
        params2 = opt.step(params1, grads2)

        # Velocity should be non-zero (momentum component)
        assert np.any(opt.velocity != 0)


class TestOptimizersConvergence:
    """Test that optimizers can minimize simple quadratic functions."""

    def test_sgd_convergence(self):
        # Minimize f(x) = x^2
        params = np.array([10.0])
        opt = SGD(lr=0.1)

        for _ in range(100):
            grads = 2 * params  # gradient of x^2
            params = opt.step(params, grads)

        assert abs(params[0]) < 0.1  # Should be close to 0

    def test_adam_convergence(self):
        # Minimize f(x) = x^2
        params = np.array([10.0])
        opt = Adam(lr=0.2)

        for _ in range(100):
            grads = 2 * params
            params = opt.step(params, grads)

        assert abs(params[0]) < 0.1

    def test_adamom_convergence(self):
        """Test our novel AdaMom optimizer."""
        params = np.array([10.0])
        opt = AdaMom(lr=0.2, momentum=0.9)

        for _ in range(100):
            grads = 2 * params
            params = opt.step(params, grads)

        assert abs(params[0]) < 0.1
