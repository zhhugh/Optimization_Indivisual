# Optimization Methods for Deep Learning

This repository hosts the code for an individual research project that studies classic and modern gradient-based optimizers. All datasets, experiment logs, and generated reports are intentionally excluded from version control so the repo stays lightweight—run the commands below to reproduce every result locally.

## Key Features
- **Optimizers**: SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, and the proposed **AdaMom (Ours)**.
- **Problems**: Logistic regression (MNIST & Fashion-MNIST), two-layer MLP (MNIST), Rosenbrock trajectories.
- **Pipeline**: Deterministic experiment runner (`experiments/run_experiment.py`) + analysis scripts that aggregate logs into CSV tables and publication-ready figures.
- **Testing**: Unit tests for optimizers and problems ensure each update rule behaves as expected.

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run tests
```bash
pytest                    # unit tests
pytest --cov=src --cov-report=term-missing
```

### 3. Launch experiments
```bash
python experiments/run_experiment.py --config experiments/configs/logistic_mnist.yaml
python experiments/run_experiment.py --config experiments/configs/mlp_mnist.yaml
python experiments/run_experiment.py --config experiments/configs/fashion_mnist.yaml
python experiments/run_experiment.py --config experiments/configs/toy_2d.yaml
```
Each run writes JSON logs to `experiments/logs/` (ignored by git).

### 4. Generate tables & figures
```bash
python generate_all_analysis.py
```
Outputs land under `reports/tables/` and `reports/figures/` (also ignored). These include loss/accuracy curves, gradient norms, Rosenbrock trajectories, and CSV summaries (performance, convergence, compute, hyperparameters, significance tests).

## Repository Layout
```
src/            # Optimizers, problems, utils, and analysis modules
experiments/    # YAML configs + runner (logs generated locally)
tests/          # Unit tests mirroring src structure
generate_all_analysis.py
requirements.txt
README.md
```

## Notes on Datasets
- MNIST/Fashion-MNIST are fetched via scikit-learn’s OpenML helper. When offline, loaders fall back to synthetic Gaussian data with matching shapes so the pipeline still runs; logs record whether real or fallback data was used.
- Rosenbrock experiments are deterministic but randomize initial points to produce multiple trajectories.

## Code Quality
```bash
black src tests
ruff check src tests
```

## Citation
```
@misc{optimization-methods-2024,
  author = {Zhou Han},
  title  = {Optimization Methods for Deep Learning: An Empirical Comparison},
  year   = {2025},
  url    = {https://github.com/zhhugh/optimization-project}
}
```

MIT License — see `LICENSE` for details.
