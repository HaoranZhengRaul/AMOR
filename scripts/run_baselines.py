#!/usr/bin/env python3
"""
Run baseline experiments for AMOR v2 comparison.

Baselines:
1. SSM Only - Lower bound (can't retrieve from distant positions)
2. Full Attention - Upper bound (expensive but accurate)
3. AMOR v2 Oracle Gate - Ceiling (perfect gate decisions)
"""
import sys
sys.path.append('.')

import torch
import json
from datetime import datetime
from pathlib import Path

from src.data.retrieval_task import RetrievalTaskDataset
from src.models.ssm import SimpleSSM
from src.models.attention import FullAttentionBaseline
from src.models.amor_v2 import AMORv2
from src.training.supervised import (
    train_ssm_baseline,
    train_attention_baseline,
    train_amor_v2,
    evaluate,
    save_results
)
from torch.utils.data import DataLoader

# Config
device = 'cpu'  # M1 Air - CPU is stable
epochs = 15
batch_size = 32
lr = 5e-4
d_model = 64  # Smaller for faster training

print("="*60)
print("AMOR v2 - Baseline Experiments")
print("="*60)
print(f"Device: {device}")
print(f"Epochs: {epochs}")
print(f"d_model: {d_model}")
print()

# Create datasets
print("Creating datasets...")
train_ds = RetrievalTaskDataset(size=2000, seq_length=128, seed=42)
val_ds = RetrievalTaskDataset(size=500, seq_length=128, seed=43)
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
print(f"Retrieval rate: {train_ds.get_stats()['retrieval_rate']:.2%}")
print()

results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'device': device,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'd_model': d_model,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
    },
    'baselines': {}
}

# 1. SSM Only Baseline
print("="*60)
print("1. Training SSM Only Baseline...")
print("="*60)
ssm = SimpleSSM(vocab_size=train_ds.total_vocab, d_model=d_model, n_layers=2)
ssm = train_ssm_baseline(ssm, train_ds, epochs=epochs, lr=lr, device=device)

# Evaluate SSM
ssm.eval()
val_loader = DataLoader(val_ds, batch_size=batch_size)
ssm_correct = 0
ssm_total = 0
ssm_retrieval_correct = 0
ssm_retrieval_total = 0
ssm_local_correct = 0
ssm_local_total = 0

with torch.no_grad():
    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        nr = batch['needs_retrieval'][:, 1:].to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        logits = ssm(x)
        preds = logits.argmax(dim=-1)
        correct = (preds == y)

        ssm_correct += correct.sum().item()
        ssm_total += y.numel()
        ssm_retrieval_correct += (correct & nr).sum().item()
        ssm_retrieval_total += nr.sum().item()
        ssm_local_correct += (correct & ~nr).sum().item()
        ssm_local_total += (~nr).sum().item()

ssm_results = {
    'accuracy': ssm_correct / ssm_total,
    'retrieval_accuracy': ssm_retrieval_correct / ssm_retrieval_total if ssm_retrieval_total > 0 else 0,
    'local_accuracy': ssm_local_correct / ssm_local_total if ssm_local_total > 0 else 0,
    'params': sum(p.numel() for p in ssm.parameters()),
}
results['baselines']['ssm_only'] = ssm_results
print(f"\nSSM Only Results:")
print(f"  Accuracy: {ssm_results['accuracy']:.2%}")
print(f"  Retrieval Accuracy: {ssm_results['retrieval_accuracy']:.2%}")
print(f"  Local Accuracy: {ssm_results['local_accuracy']:.2%}")
print(f"  Parameters: {ssm_results['params']:,}")
print()

# 2. Full Attention Baseline
print("="*60)
print("2. Training Full Attention Baseline...")
print("="*60)
attn = FullAttentionBaseline(vocab_size=train_ds.total_vocab, d_model=d_model, n_heads=4, n_layers=2)
attn, attn_history = train_attention_baseline(attn, train_ds, epochs=epochs, lr=lr, device=device)

# Evaluate Attention
attn.eval()
attn_correct = 0
attn_total = 0
attn_retrieval_correct = 0
attn_retrieval_total = 0
attn_local_correct = 0
attn_local_total = 0

with torch.no_grad():
    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        nr = batch['needs_retrieval'][:, 1:].to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        logits = attn(x)
        preds = logits.argmax(dim=-1)
        correct = (preds == y)

        attn_correct += correct.sum().item()
        attn_total += y.numel()
        attn_retrieval_correct += (correct & nr).sum().item()
        attn_retrieval_total += nr.sum().item()
        attn_local_correct += (correct & ~nr).sum().item()
        attn_local_total += (~nr).sum().item()

attn_results = {
    'accuracy': attn_correct / attn_total,
    'retrieval_accuracy': attn_retrieval_correct / attn_retrieval_total if attn_retrieval_total > 0 else 0,
    'local_accuracy': attn_local_correct / attn_local_total if attn_local_total > 0 else 0,
    'params': sum(p.numel() for p in attn.parameters()),
    'history': attn_history,
}
results['baselines']['full_attention'] = attn_results
print(f"\nFull Attention Results:")
print(f"  Accuracy: {attn_results['accuracy']:.2%}")
print(f"  Retrieval Accuracy: {attn_results['retrieval_accuracy']:.2%}")
print(f"  Local Accuracy: {attn_results['local_accuracy']:.2%}")
print(f"  Parameters: {attn_results['params']:,}")
print()

# 3. AMOR v2 with Oracle Gate
print("="*60)
print("3. Training AMOR v2 with Oracle Gate...")
print("="*60)
amor_oracle = AMORv2(
    vocab_size=train_ds.total_vocab,
    d_model=d_model,
    gate_method='oracle',
    n_heads=4,
    n_layers=2,
)
oracle_history = train_amor_v2(
    amor_oracle, train_ds, val_ds,
    num_epochs=epochs,
    lr=lr,
    device=device,
    balance_loss_weight=0.0,  # No balance loss for oracle
)

# Evaluate Oracle
amor_oracle.eval()
oracle_metrics = evaluate(amor_oracle, val_loader, device)
oracle_results = {
    'accuracy': oracle_metrics['accuracy'],
    'retrieval_accuracy': oracle_metrics['retrieval_accuracy'],
    'local_accuracy': oracle_metrics['local_accuracy'],
    'gate_f1': 1.0,  # Perfect by definition
    'gate_fires': oracle_metrics['gate_fires'],
    'flops_saved': oracle_metrics['flops_saved'],
    'params': sum(p.numel() for p in amor_oracle.parameters()),
    'history': oracle_history,
}
results['baselines']['amor_oracle'] = oracle_results
print(f"\nAMOR v2 Oracle Results:")
print(f"  Accuracy: {oracle_results['accuracy']:.2%}")
print(f"  Retrieval Accuracy: {oracle_results['retrieval_accuracy']:.2%}")
print(f"  Local Accuracy: {oracle_results['local_accuracy']:.2%}")
print(f"  Gate F1: {oracle_results['gate_f1']:.2%} (perfect by definition)")
print(f"  Gate Fires: {oracle_results['gate_fires']:.2%}")
print(f"  FLOPs Saved: {oracle_results['flops_saved']:.2%}")
print(f"  Parameters: {oracle_results['params']:,}")
print()

# Summary Table
print("="*60)
print("BASELINE SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Accuracy':>10} {'Retr Acc':>10} {'Local Acc':>10} {'FLOPs Saved':>12}")
print("-"*60)
print(f"{'SSM Only':<20} {ssm_results['accuracy']:>10.2%} {ssm_results['retrieval_accuracy']:>10.2%} {ssm_results['local_accuracy']:>10.2%} {'100%':>12}")
print(f"{'Full Attention':<20} {attn_results['accuracy']:>10.2%} {attn_results['retrieval_accuracy']:>10.2%} {attn_results['local_accuracy']:>10.2%} {'0%':>12}")
print(f"{'AMOR v2 Oracle':<20} {oracle_results['accuracy']:>10.2%} {oracle_results['retrieval_accuracy']:>10.2%} {oracle_results['local_accuracy']:>10.2%} {oracle_results['flops_saved']:>11.2%}")
print("-"*60)

# Save results
save_results(results, 'experiments/baselines/results.json')
print(f"\nResults saved to experiments/baselines/results.json")

# Save models
torch.save(ssm.state_dict(), 'experiments/baselines/ssm_only.pt')
torch.save(attn.state_dict(), 'experiments/baselines/full_attention.pt')
torch.save(amor_oracle.state_dict(), 'experiments/baselines/amor_oracle.pt')
print("Models saved to experiments/baselines/")
