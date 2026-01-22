#!/usr/bin/env python3
"""
Noise Length Sweep: Systematically measure SSM state decay.

This script runs experiments across different noise lengths to understand
the relationship between noise length and retrieval accuracy.

Output: experiments/diagnostic_needlehaystack/noise_sweep_detailed.json
Figure: paper/figures/fig3_ssm_horizon.png (with error bars)
"""
import sys
sys.path.append('.')

import random
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.needle_haystack import NeedleHaystackTask
from src.models.amor_v2 import AMORv2
from src.training.supervised import train_moe_style, evaluate


def set_global_seed(seed):
    """Set seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_trial(noise_range, trial_idx, epochs=30, device='cpu'):
    """
    Run a single training trial with specified noise range.

    Returns:
        dict with retrieval_accuracy and other metrics
    """
    # Set seed for this specific trial (reproducible but different per trial)
    trial_seed = 42 + trial_idx * 1000  # Spread seeds to avoid correlation
    set_global_seed(trial_seed)

    print(f"\n  Trial {trial_idx + 1}: noise_range={noise_range}, seed={trial_seed}")

    # Create datasets with trial-specific seeds
    train_ds = NeedleHaystackTask(
        size=2000,
        seq_length=256,
        vocab_size=16,
        noise_length_range=noise_range,
        impossible_query_prob=0.2,
        seed=42 + trial_idx,
    )

    val_ds = NeedleHaystackTask(
        size=500,
        seq_length=256,
        vocab_size=16,
        noise_length_range=noise_range,
        impossible_query_prob=0.2,
        seed=1000 + trial_idx,
    )

    # Create AMOR Oracle model (perfect gating)
    model = AMORv2(
        vocab_size=train_ds.total_vocab,
        d_model=64,
        n_layers=2,
        gate_method='oracle',  # Use oracle for ground truth
        attention_top_k=3,
    )

    # Train (train_moe_style modifies model in-place, returns history)
    _ = train_moe_style(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=epochs,
        batch_size=32,
        lr=5e-4,
        balance_loss_weight=0.1,
        device=device,
    )

    # Evaluate
    model.eval()
    loader = DataLoader(val_ds, batch_size=32)

    total_correct = 0
    total_retrieval = 0
    total_retrieval_correct = 0
    total_local_correct = 0
    total_local = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            needs_retrieval = batch['needs_retrieval'].to(device)

            x = tokens[:, :-1]
            y = tokens[:, 1:]
            nr = needs_retrieval[:, 1:]

            outputs = model(x, ground_truth_gate=nr.float())
            preds = outputs['logits'].argmax(dim=-1)

            # Overall accuracy
            total_correct += (preds == y).sum().item()

            # Retrieval accuracy
            if nr.any():
                retrieval_correct = ((preds == y) & nr).sum().item()
                total_retrieval_correct += retrieval_correct
                total_retrieval += nr.sum().item()

            # Local accuracy
            local_mask = ~nr
            if local_mask.any():
                local_correct = ((preds == y) & local_mask).sum().item()
                total_local_correct += local_correct
                total_local += local_mask.sum().item()

    retrieval_acc = total_retrieval_correct / total_retrieval if total_retrieval > 0 else 0
    local_acc = total_local_correct / total_local if total_local > 0 else 0

    return {
        'retrieval_accuracy': retrieval_acc,
        'local_accuracy': local_acc,
        'n_retrieval_tokens': total_retrieval,
        'n_local_tokens': total_local,
    }


def run_sweep(noise_lengths, n_trials=5, epochs=30, device='cpu'):
    """
    Run full sweep across all noise lengths.

    Args:
        noise_lengths: list of noise length midpoints (e.g., [10, 20, 30, ...])
        n_trials: number of trials per condition
        epochs: training epochs per trial
        device: cuda or cpu

    Returns:
        dict with results for each noise length
    """
    results = {}

    for noise_mid in noise_lengths:
        # Create range around midpoint (±10)
        noise_range = (max(5, noise_mid - 10), noise_mid + 10)
        print(f"\n{'='*60}")
        print(f"Noise length: {noise_mid} (range {noise_range})")
        print(f"{'='*60}")

        trial_results = []
        for trial_idx in range(n_trials):
            result = run_single_trial(noise_range, trial_idx, epochs, device)
            trial_results.append(result)
            print(f"    -> Retrieval acc: {result['retrieval_accuracy']:.4f}")

        # Aggregate
        retrieval_accs = [r['retrieval_accuracy'] for r in trial_results]
        results[noise_mid] = {
            'noise_range': noise_range,
            'retrieval_accuracy_mean': float(np.mean(retrieval_accs)),
            'retrieval_accuracy_std': float(np.std(retrieval_accs)),
            'retrieval_accuracy_all': retrieval_accs,
            'n_trials': n_trials,
        }

        print(f"  -> Mean: {results[noise_mid]['retrieval_accuracy_mean']:.4f} "
              f"± {results[noise_mid]['retrieval_accuracy_std']:.4f}")

    return results


def plot_results(results, save_path):
    """Generate the SSM horizon curve with error bars."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by noise length
    noise_lengths = sorted(results.keys())
    means = [results[n]['retrieval_accuracy_mean'] for n in noise_lengths]
    stds = [results[n]['retrieval_accuracy_std'] for n in noise_lengths]

    # Plot line with error bars
    ax.errorbar(noise_lengths, means, yerr=stds, fmt='o-',
                linewidth=2, markersize=10, capsize=5,
                color='#3498db', markeredgecolor='black', ecolor='#7f8c8d')

    # Fill error region
    ax.fill_between(noise_lengths,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='#3498db')

    # Reference lines
    ax.axhline(0.5, color='#95a5a6', linestyle=':', linewidth=1.5, label='50% accuracy')

    # Find approximate horizon (where accuracy drops below 50%)
    below_50 = [n for n, m in zip(noise_lengths, means) if m < 0.5]
    if below_50:
        horizon = below_50[0]
        ax.axvline(horizon, color='#e74c3c', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Approx. horizon (~{horizon} tokens)')

    ax.set_xlabel('Noise Length (tokens)')
    ax.set_ylabel('Oracle Retrieval Accuracy')
    ax.set_title('SSM State Decay vs Noise Length')
    ax.set_ylim(0, 1)
    ax.set_xlim(min(noise_lengths) - 5, max(noise_lengths) + 5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path)
    print(f"\nSaved figure: {save_path}")

    return fig


def main():
    # Configuration
    noise_lengths = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]  # 10 conditions
    n_trials = 5  # Per user decision
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("SSM State Decay Sweep")
    print("="*60)
    print(f"Noise lengths: {noise_lengths}")
    print(f"Trials per condition: {n_trials}")
    print(f"Epochs per trial: {epochs}")
    print(f"Device: {device}")
    print(f"Total training runs: {len(noise_lengths) * n_trials}")
    print("="*60)

    # Run sweep
    results = run_sweep(noise_lengths, n_trials, epochs, device)

    # Save results
    output_dir = Path('experiments/diagnostic_needlehaystack')
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'noise_lengths': noise_lengths,
            'n_trials': n_trials,
            'epochs': epochs,
            'device': device,
        },
        'results': results,
    }

    output_path = output_dir / 'noise_sweep_detailed.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {output_path}")

    # Generate figure
    fig_path = Path('paper/figures/fig3_ssm_horizon.png')
    plot_results(results, fig_path)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for noise_mid in sorted(results.keys()):
        r = results[noise_mid]
        print(f"Noise {noise_mid:3d}: {r['retrieval_accuracy_mean']:.3f} ± {r['retrieval_accuracy_std']:.3f}")


if __name__ == '__main__':
    main()
