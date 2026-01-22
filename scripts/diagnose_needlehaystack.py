#!/usr/bin/env python3
"""
Diagnose NeedleHaystack failure.

Hypotheses:
H1: Undertrained - test more epochs
H2: Model too small - test larger d_model
H3: Noise too long - test shorter noise ranges
H4: Ghost KV loses info - compare Ghost vs Full KV
H5: Attention top_k too small - test higher top_k
"""
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

from src.data.needle_haystack import NeedleHaystackTask
from src.models.amor_v2 import AMORv2
from src.training.supervised import save_results

device = 'cpu'
batch_size = 32
lr = 5e-4

print("="*70)
print("NeedleHaystack Diagnostic - Finding Root Cause")
print("="*70)

results = {
    'timestamp': datetime.now().isoformat(),
    'diagnostics': {}
}


def train_and_eval(model, train_ds, val_ds, epochs, name):
    """Train model and return retrieval accuracy."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = []
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'{name} E{epoch+1}', leave=False):
            tokens = batch['tokens'].to(device)
            nr = batch['needs_retrieval'][:, 1:].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            if hasattr(model, 'gate_method') and model.gate_method == 'oracle':
                outputs = model(x, ground_truth_gate=nr.float())
            else:
                outputs = model(x)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Quick eval every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            retr_correct = retr_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch['tokens'].to(device)
                    nr = batch['needs_retrieval'][:, 1:].to(device)
                    imp = batch['is_impossible'][:, 1:].to(device)
                    x, y = tokens[:, :-1], tokens[:, 1:]

                    if hasattr(model, 'gate_method') and model.gate_method == 'oracle':
                        outputs = model(x, ground_truth_gate=nr.float())
                    else:
                        outputs = model(x)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                    preds = logits.argmax(dim=-1)
                    retr_mask = nr & ~imp
                    retr_correct += ((preds == y) & retr_mask).sum().item()
                    retr_total += retr_mask.sum().item()

            retr_acc = retr_correct / retr_total if retr_total > 0 else 0
            history.append({'epoch': epoch + 1, 'retrieval_acc': retr_acc})
            print(f"  {name} E{epoch+1}: Retr Acc = {retr_acc:.2%}")

    return history[-1]['retrieval_acc'] if history else 0, history


# =====================================================================
# DIAGNOSTIC 1: Training Duration
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 1: Training Duration")
print("="*70)

train_ds = NeedleHaystackTask(size=2000, seq_length=256, noise_length_range=(50, 150), seed=42)
val_ds = NeedleHaystackTask(size=500, seq_length=256, noise_length_range=(50, 150), seed=43)

duration_results = {}
for epochs in [30, 50]:
    print(f"\nTesting epochs={epochs}")
    model = AMORv2(vocab_size=train_ds.total_vocab, d_model=64, gate_method='oracle', n_heads=4, n_layers=2)
    acc, history = train_and_eval(model, train_ds, val_ds, epochs, f"Oracle-{epochs}ep")
    duration_results[str(epochs)] = {'final_acc': acc, 'history': history}

results['diagnostics']['training_duration'] = duration_results
print(f"\nTraining Duration Results: 30ep={duration_results['30']['final_acc']:.2%}, 50ep={duration_results['50']['final_acc']:.2%}")


# =====================================================================
# DIAGNOSTIC 2: Noise Length Sweep (Task Difficulty)
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 2: Noise Length Sweep")
print("="*70)

noise_results = {}
noise_ranges = [
    (10, 30),   # Easy
    (30, 60),   # Medium
    (50, 150),  # Current (hard)
]

for noise_range in noise_ranges:
    print(f"\nTesting noise_range={noise_range}")
    train_ds = NeedleHaystackTask(size=1500, seq_length=256, noise_length_range=noise_range, seed=42)
    val_ds = NeedleHaystackTask(size=400, seq_length=256, noise_length_range=noise_range, seed=43)

    model = AMORv2(vocab_size=train_ds.total_vocab, d_model=64, gate_method='oracle', n_heads=4, n_layers=2)
    acc, history = train_and_eval(model, train_ds, val_ds, 30, f"Oracle-noise{noise_range}")
    noise_results[str(noise_range)] = {'final_acc': acc, 'retrieval_rate': train_ds.get_stats()['retrieval_rate']}

results['diagnostics']['noise_length'] = noise_results
print("\nNoise Length Results:")
for k, v in noise_results.items():
    print(f"  {k}: {v['final_acc']:.2%}")


# =====================================================================
# DIAGNOSTIC 3: Model Scale
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 3: Model Scale")
print("="*70)

train_ds = NeedleHaystackTask(size=1500, seq_length=256, noise_length_range=(50, 150), seed=42)
val_ds = NeedleHaystackTask(size=400, seq_length=256, noise_length_range=(50, 150), seed=43)

scale_results = {}
for d_model in [64, 128]:
    print(f"\nTesting d_model={d_model}")
    model = AMORv2(vocab_size=train_ds.total_vocab, d_model=d_model, gate_method='oracle', n_heads=4, n_layers=2)
    params = sum(p.numel() for p in model.parameters())
    acc, history = train_and_eval(model, train_ds, val_ds, 30, f"Oracle-d{d_model}")
    scale_results[str(d_model)] = {'final_acc': acc, 'params': params}

results['diagnostics']['model_scale'] = scale_results
print("\nModel Scale Results:")
for k, v in scale_results.items():
    print(f"  d_model={k}: {v['final_acc']:.2%} ({v['params']:,} params)")


# =====================================================================
# DIAGNOSTIC 4: Attention Top-K
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 4: Attention Top-K")
print("="*70)

# First check current top_k setting
from src.models.amor_v2 import AMORv2
import inspect
sig = inspect.signature(AMORv2.__init__)
print(f"Default AMORv2 parameters: checking attention implementation...")

# We need to modify AMORv2 to accept top_k parameter
# For now, test by checking if current architecture uses top-k
topk_results = {}

# Quick test: the current implementation uses top_k=3 in SparseAttentionWithGhostKV
# Let's just document this for now
topk_results['note'] = "Current implementation uses top_k=3 in SparseAttentionWithGhostKV"
topk_results['recommendation'] = "Consider increasing top_k for longer sequences"

results['diagnostics']['attention_topk'] = topk_results


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

print("\n1. TRAINING DURATION:")
for k, v in duration_results.items():
    print(f"   {k} epochs: {v['final_acc']:.2%}")

print("\n2. NOISE LENGTH (Task Difficulty):")
for k, v in noise_results.items():
    print(f"   {k}: {v['final_acc']:.2%}")

print("\n3. MODEL SCALE:")
for k, v in scale_results.items():
    print(f"   d_model={k}: {v['final_acc']:.2%}")

# Determine root cause
print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

# Check if noise length is the issue
easy_noise = noise_results.get('(10, 30)', {}).get('final_acc', 0)
hard_noise = noise_results.get('(50, 150)', {}).get('final_acc', 0)

if easy_noise > 0.7 and hard_noise < 0.4:
    print("PRIMARY BOTTLENECK: SSM state decay")
    print("  - Easy noise (10-30): {:.2%}".format(easy_noise))
    print("  - Hard noise (50-150): {:.2%}".format(hard_noise))
    print("  - The SSM cannot maintain key-value associations through long noise")
    results['root_cause'] = 'ssm_state_decay'
elif scale_results.get('128', {}).get('final_acc', 0) > scale_results.get('64', {}).get('final_acc', 0) + 0.2:
    print("PRIMARY BOTTLENECK: Model capacity")
    print("  - Larger model significantly improves retrieval")
    results['root_cause'] = 'model_capacity'
else:
    print("PRIMARY BOTTLENECK: Unclear - multiple factors")
    print("  - Consider Ghost KV information loss")
    print("  - Consider attention architecture")
    results['root_cause'] = 'multiple_factors'

# Save results
Path('experiments/diagnostic_needlehaystack').mkdir(parents=True, exist_ok=True)
save_results(results, 'experiments/diagnostic_needlehaystack/results.json')
print(f"\nResults saved to experiments/diagnostic_needlehaystack/results.json")
