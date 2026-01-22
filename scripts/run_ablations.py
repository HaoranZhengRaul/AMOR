#!/usr/bin/env python3
"""
Run ablation studies for AMOR v2.

Ablations:
1. Gate Methods: entropy vs learned_ste
2. Target Retrieval Rates: 0.1, 0.2, 0.3
3. Balance Loss Weights: 0.0, 0.1, 0.5
"""
import sys
sys.path.append('.')

import torch
import json
from datetime import datetime
from pathlib import Path

from src.data.retrieval_task import RetrievalTaskDataset
from src.models.amor_v2 import AMORv2
from src.training.supervised import train_amor_v2, evaluate, save_results
from torch.utils.data import DataLoader

# Config
device = 'cpu'
epochs = 15  # Reduced for faster ablations
batch_size = 32
lr = 5e-4
d_model = 64

print("="*60)
print("AMOR v2 - Ablation Studies")
print("="*60)

# Create datasets
print("Creating datasets...")
train_ds = RetrievalTaskDataset(size=2000, seq_length=128, seed=42)
val_ds = RetrievalTaskDataset(size=500, seq_length=128, seed=43)
val_loader = DataLoader(val_ds, batch_size=batch_size)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
print()

results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'device': device,
        'epochs': epochs,
        'd_model': d_model,
    },
    'ablations': {}
}

def run_ablation(name, gate_method, target_rate, balance_weight):
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"  gate_method={gate_method}, target_rate={target_rate}, balance_weight={balance_weight}")
    print(f"{'='*60}")

    model = AMORv2(
        vocab_size=train_ds.total_vocab,
        d_model=d_model,
        gate_method=gate_method,
        n_heads=4,
        n_layers=2,
        target_retrieval_rate=target_rate,
    )

    history = train_amor_v2(
        model, train_ds, val_ds,
        num_epochs=epochs,
        lr=lr,
        device=device,
        balance_loss_weight=balance_weight,
    )

    model.eval()
    metrics = evaluate(model, val_loader, device)

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Retrieval Acc: {metrics['retrieval_accuracy']:.2%}")
    print(f"  Gate F1: {metrics['gate_f1']:.2%}")
    print(f"  Gate Fires: {metrics['gate_fires']:.2%}")
    print(f"  FLOPs Saved: {metrics['flops_saved']:.2%}")

    return {
        'config': {
            'gate_method': gate_method,
            'target_rate': target_rate,
            'balance_weight': balance_weight,
        },
        'metrics': {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()},
        'history': history,
    }

# Ablation 1: Gate Methods
print("\n" + "="*70)
print("ABLATION 1: Gate Methods")
print("="*70)

results['ablations']['entropy_default'] = run_ablation(
    'entropy_default',
    gate_method='entropy',
    target_rate=0.2,
    balance_weight=0.1,
)

results['ablations']['learned_ste'] = run_ablation(
    'learned_ste',
    gate_method='learned_ste',
    target_rate=0.2,
    balance_weight=0.1,
)

# Ablation 2: Target Retrieval Rates
print("\n" + "="*70)
print("ABLATION 2: Target Retrieval Rates")
print("="*70)

for rate in [0.1, 0.2, 0.3]:
    results['ablations'][f'target_rate_{rate}'] = run_ablation(
        f'target_rate_{rate}',
        gate_method='entropy',
        target_rate=rate,
        balance_weight=0.1,
    )

# Ablation 3: Balance Loss Weights
print("\n" + "="*70)
print("ABLATION 3: Balance Loss Weights")
print("="*70)

for weight in [0.0, 0.1, 0.5]:
    results['ablations'][f'balance_weight_{weight}'] = run_ablation(
        f'balance_weight_{weight}',
        gate_method='entropy',
        target_rate=0.2,
        balance_weight=weight,
    )

# Summary Table
print("\n" + "="*70)
print("ABLATION SUMMARY")
print("="*70)
print(f"{'Ablation':<25} {'Accuracy':>10} {'Retr Acc':>10} {'Gate F1':>10} {'FLOPs Saved':>12}")
print("-"*70)

for name, data in results['ablations'].items():
    m = data['metrics']
    print(f"{name:<25} {m['accuracy']:>10.2%} {m['retrieval_accuracy']:>10.2%} {m['gate_f1']:>10.2%} {m['flops_saved']:>11.2%}")

print("-"*70)

# Save results
save_results(results, 'experiments/ablations/results.json')
print(f"\nResults saved to experiments/ablations/results.json")
