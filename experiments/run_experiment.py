"""Main experiment runner."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import numpy as np
from tqdm import tqdm
import time

from src.optimizers import *
from src.problems import *
from src.utils import (
    ExperimentLogger,
    MetricsTracker,
    create_dataloaders,
    load_mnist,
    load_fashion_mnist,
)


def run_single_experiment(config: dict, seed: int = 42) -> dict:
    """Run a single experiment with given configuration.

    Args:
        config: Experiment configuration
        seed: Random seed

    Returns:
        Dictionary of results
    """
    np.random.seed(seed)

    # Create problem
    problem_name = config['problem']['name']
    problem_params = config['problem'].get('params', {})

    data_config = config['problem'].get('data', {})
    dataset_name = data_config.get('name', 'mnist').lower()
    dataset_max_samples = data_config.get('max_samples')

    if problem_name == 'logistic':
        if dataset_name in {'mnist', 'mnist_784'}:
            X_train, X_test, y_train, y_test = load_mnist(max_samples=dataset_max_samples)
        elif dataset_name in {'fashion_mnist', 'fashion'}:
            X_train, X_test, y_train, y_test = load_fashion_mnist(max_samples=dataset_max_samples)
        else:
            raise ValueError(f"Unknown dataset for logistic problem: {dataset_name}")
        problem = LogisticRegression(
            input_dim=X_train.shape[1],
            output_dim=10,
            **problem_params
        )
    elif problem_name == 'mlp':
        if dataset_name in {'mnist', 'mnist_784'}:
            X_train, X_test, y_train, y_test = load_mnist(max_samples=dataset_max_samples)
        elif dataset_name in {'fashion_mnist', 'fashion'}:
            X_train, X_test, y_train, y_test = load_fashion_mnist(max_samples=dataset_max_samples)
        else:
            raise ValueError(f"Unknown dataset for MLP problem: {dataset_name}")
        problem = MLP(
            input_dim=X_train.shape[1],
            hidden_dims=problem_params.get('hidden_dims', [128, 64]),
            output_dim=10,
            **{k: v for k, v in problem_params.items() if k != 'hidden_dims'}
        )
    elif problem_name == 'rosenbrock':
        problem = Rosenbrock(**problem_params)
        X_train = X_test = y_train = y_test = None
    elif problem_name == 'beale':
        problem = Beale()
        X_train = X_test = y_train = y_test = None
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    # Create optimizer
    optimizer_name = config['optimizer']['name']
    optimizer_params = config['optimizer'].get('params', {})
    lr = optimizer_params.pop('lr', 0.01)

    optimizer_class = {
        'sgd': SGD,
        'momentum': Momentum,
        'nesterov': Nesterov,
        'adagrad': Adagrad,
        'rmsprop': RMSprop,
        'adam': Adam,
        'adamw': AdamW,
        'adamom': AdaMom,
    }[optimizer_name.lower()]

    optimizer = optimizer_class(lr=lr, **optimizer_params)

    # Initialize parameters
    params = problem.initialize_params(seed=seed)

    # Training parameters
    max_iterations = config['training'].get('max_iterations', 1000)
    batch_size = config['training'].get('batch_size', 32)
    eval_interval = config['training'].get('eval_interval', 10)
    track_gradients = config['training'].get('track_gradients', False)

    # Create logger
    exp_name = f"{problem_name}_{optimizer_name}_lr{lr}_seed{seed}"
    logger = ExperimentLogger(exp_name)
    logger.log_metadata({
        'problem': problem_name,
        'dataset': dataset_name,
        'dataset_max_samples': dataset_max_samples,
        'optimizer': optimizer_name,
        'learning_rate': lr,
        'batch_size': batch_size,
        'seed': seed,
        'num_parameters': problem.num_params(),
        **optimizer_params,
        **problem_params,
    })

    # Create metrics tracker
    metrics = MetricsTracker()
    metrics.start_timing()

    # For 2D problems, track parameter trajectory
    param_trajectory = [] if problem.num_params() == 2 else None

    # Training loop
    print(f"\nRunning {exp_name}...")

    # Determine if we need data loaders
    use_batches = X_train is not None and batch_size < X_train.shape[0]

    for iteration in tqdm(range(max_iterations), desc="Training"):
        # Get batch
        if use_batches:
            # Sample random batch
            indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
        else:
            X_batch = X_train
            y_batch = y_train

        # Compute gradient
        grads = problem.gradient(params, X_batch, y_batch)

        # Track gradient norm
        grad_norm = np.linalg.norm(grads) if track_gradients else None

        # Update parameters
        params = optimizer.step(params, grads)

        # Track trajectory for 2D problems
        if param_trajectory is not None:
            param_trajectory.append(params.copy())

        # Evaluate periodically
        if iteration % eval_interval == 0 or iteration == max_iterations - 1:
            if X_train is not None:
                train_metrics = problem.evaluate(params, X_train, y_train)
                test_metrics = problem.evaluate(params, X_test, y_test)

                log_data = {
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'test_loss': test_metrics['loss'],
                    'test_accuracy': test_metrics['accuracy'],
                }

                metrics.record(
                    iteration,
                    loss=train_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    test_accuracy=test_metrics['accuracy'],
                    gradient_norm=grad_norm,
                )
            else:
                # For toy functions
                loss = problem.loss(params)
                log_data = {'train_loss': loss}
                metrics.record(iteration, loss=loss, gradient_norm=grad_norm)

            if track_gradients and grad_norm is not None:
                log_data['gradient_norm'] = grad_norm

            logger.log_iteration(iteration, log_data)

    # Save trajectory for 2D problems
    if param_trajectory is not None:
        logger.log_metadata({'param_trajectory': param_trajectory})

    # Save results
    results_path = logger.save_results()

    # Get summary statistics
    stats = metrics.get_statistics()

    return {
        'results_path': results_path,
        'stats': stats,
        'final_params': params,
    }


def run_experiment_suite(config_path: str) -> None:
    """Run a suite of experiments from configuration file.

    Args:
        config_path: Path to YAML configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get list of configurations to run
    base_config = config.get('base', {})
    experiments = config.get('experiments', [])
    n_seeds = config.get('n_seeds', 1)

    results = []

    for exp_config in experiments:
        # Merge with base config
        full_config = {
            'problem': {**base_config.get('problem', {}), **exp_config.get('problem', {})},
            'optimizer': {**base_config.get('optimizer', {}), **exp_config.get('optimizer', {})},
            'training': {**base_config.get('training', {}), **exp_config.get('training', {})},
        }

        # Run for multiple seeds
        for seed in range(n_seeds):
            try:
                result = run_single_experiment(full_config, seed=seed)
                results.append(result)
                print(f"✓ Completed: {result['results_path']}")
                print(f"  Final test accuracy: {result['stats'].get('final_test_accuracy', 'N/A')}")
            except Exception as e:
                print(f"✗ Failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Experiment suite completed: {len(results)} experiments run")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run optimization experiments')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment configuration YAML file')
    args = parser.parse_args()

    run_experiment_suite(args.config)
