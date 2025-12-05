#!/usr/bin/env python
import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("OPTIMIZATION RESEARCH PROJECT - ANALYSIS DEMONSTRATION")
print("=" * 70)

# 1. Show experiment logs
print("\n📁 EXPERIMENT LOGS")
print("-" * 70)
log_dir = Path("experiments/logs")
log_files = list(log_dir.glob("*.json"))
print(f"Total experiments completed: {len(log_files)}")

# Load and summarize a few experiments
print("\n📊 SAMPLE EXPERIMENT RESULTS")
print("-" * 70)

results_summary = []
for log_file in sorted(log_files)[:10]:  # Show first 10
    with open(log_file, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    metrics = data['metrics_history']

    if not metrics:
        continue

    final_metrics = metrics[-1]
    dataset = metadata.get('dataset')
    if dataset is None and metadata.get('problem', '').lower() in {'logistic', 'mlp'}:
        dataset = 'mnist'

    result = {
        'Optimizer': metadata.get('optimizer', 'unknown'),
        'LR': metadata.get('learning_rate', 0),
        'Seed': metadata.get('seed', 0),
        'Dataset': dataset or 'unknown',
        'Final Loss': final_metrics.get('train_loss', 0),
        'Final Test Acc': final_metrics.get('test_accuracy', 0),
        'Iterations': len(metrics),
    }
    results_summary.append(result)

if results_summary:
    df = pd.DataFrame(results_summary)
    df = df.drop_duplicates(subset=['Optimizer', 'LR', 'Seed'])
    print(df.to_string(index=False))

    # Group by optimizer
    print("\n📈 OPTIMIZER COMPARISON (Grouped by Optimizer)")
    print("-" * 70)
    grouped = df.groupby('Optimizer').agg({
        'Final Test Acc': ['mean', 'std', 'count'],
        'Final Loss': ['mean', 'std'],
    }).round(4)
    print(grouped)