#!/usr/bin/env python3
"""
RUN THIS FIRST before any gate training!

This script verifies whether SSM prediction entropy correlates with retrieval need.
If the entropy gap is positive (>0.1), we can use entropy-based gating.
Otherwise, fall back to learned_ste.
"""
import sys
sys.path.append('.')

import torch
import json
from datetime import datetime
from src.data.retrieval_task import RetrievalTaskDataset
from src.data.needle_haystack import NeedleHaystackTask
from src.models.ssm import SimpleSSM
from src.analysis.entropy_verification import verify_entropy_signal
from src.training.supervised import train_ssm_baseline

# Config - use CPU for stability, smaller datasets for speed
device = 'cpu'  # MPS has NaN issues with GRU
print(f"Using device: {device}")

# Create smaller datasets for faster verification
print("\n1. Creating datasets...")
simple_ds = RetrievalTaskDataset(size=1000, seq_length=128, seed=42)
hard_ds = NeedleHaystackTask(size=1000, seq_length=256, seed=42)

print(f"   Simple task stats: {simple_ds.get_stats()}")
print(f"   Hard task stats: {hard_ds.get_stats()}")

# Train SSM baseline on simple task
print("\n2. Training SSM baseline on SIMPLE task...")
ssm = SimpleSSM(vocab_size=simple_ds.total_vocab, d_model=64, n_layers=2)
ssm = train_ssm_baseline(ssm, simple_ds, epochs=5, lr=5e-4, device=device)

# Verify entropy on simple task
print("\n3. Verifying entropy on SIMPLE task...")
results_simple = verify_entropy_signal(ssm, simple_ds, device=device)

# Train SSM baseline on hard task
print("\n4. Training SSM baseline on HARD task (Needle-in-Haystack)...")
ssm_hard = SimpleSSM(vocab_size=hard_ds.total_vocab, d_model=64, n_layers=2)
ssm_hard = train_ssm_baseline(ssm_hard, hard_ds, epochs=5, lr=5e-4, device=device)

# Verify entropy on hard task
print("\n5. Verifying entropy on HARD task...")
results_hard = verify_entropy_signal(ssm_hard, hard_ds, device=device)

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Simple task gap: {results_simple['gap']:.4f} -> {'Use entropy' if results_simple['use_entropy'] else 'Use learned_ste'}")
print(f"Hard task gap:   {results_hard['gap']:.4f} -> {'Use entropy' if results_hard['use_entropy'] else 'Use learned_ste'}")
print("="*50)

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'device': device,
    'simple_task': results_simple,
    'hard_task': results_hard,
    'recommendation': 'entropy' if (results_simple['use_entropy'] or results_hard['use_entropy']) else 'learned_ste'
}

with open('experiments/entropy_verification_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to experiments/entropy_verification_results.json")
print(f"\nRECOMMENDATION: Use gate_method='{results['recommendation']}'")
