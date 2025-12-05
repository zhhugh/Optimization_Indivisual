"""Data loading utilities."""

import numpy as np
from typing import Tuple, Iterator, Optional
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _limit_samples(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally truncate dataset to a fixed number of samples."""

    if max_samples is None or max_samples >= X.shape[0]:
        return X, y

    return X[:max_samples], y[:max_samples]


def load_mnist(
    normalize: bool = True,
    flatten: bool = True,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.

    Args:
        normalize: Whether to normalize pixel values to [0, 1]
        flatten: Whether to flatten images to 1D vectors

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        # Try to load MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist['data'], mnist['target']

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

    except Exception as e:
        print(f"Could not load MNIST from OpenML: {e}")
        print("Generating synthetic data for testing...")
        # Generate synthetic data for testing
        n_samples = max_samples or 70000
        X = np.random.randn(n_samples, 784).astype(np.float32)
        y = np.random.randint(0, 10, n_samples).astype(np.int64)

    # Optionally subsample
    X, y = _limit_samples(X, y, max_samples)

    # Normalize
    if normalize:
        X = X / 255.0

    # Reshape if not flattening
    if not flatten:
        X = X.reshape(-1, 28, 28)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST dataset.

    Args:
        normalize: Whether to normalize pixel values to [0, 1]
        flatten: Whether to flatten images to 1D vectors

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        # Try to load Fashion-MNIST
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        X, y = fashion_mnist['data'], fashion_mnist['target']

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

    except Exception as e:
        print(f"Could not load Fashion-MNIST from OpenML: {e}")
        print("Generating synthetic data for testing...")
        # Generate synthetic data for testing
        n_samples = max_samples or 70000
        X = np.random.randn(n_samples, 784).astype(np.float32)
        y = np.random.randint(0, 10, n_samples).astype(np.int64)

    # Optionally subsample
    X, y = _limit_samples(X, y, max_samples)

    # Normalize
    if normalize:
        X = X / 255.0

    # Reshape if not flattening
    if not flatten:
        X = X.reshape(-1, 28, 28)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


class MiniBatchIterator:
    """Iterator for creating mini-batches."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        """Initialize mini-batch iterator.

        Args:
            X: Input data
            y: Target data
            batch_size: Size of mini-batches
            shuffle: Whether to shuffle data before creating batches
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over mini-batches."""
        indices = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[MiniBatchIterator, MiniBatchIterator]:
    """Create data loaders for training and testing.

    Args:
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        batch_size: Size of mini-batches
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, test_loader
    """
    train_loader = MiniBatchIterator(X_train, y_train, batch_size, shuffle=shuffle_train)
    test_loader = MiniBatchIterator(X_test, y_test, batch_size, shuffle=False)

    return train_loader, test_loader
