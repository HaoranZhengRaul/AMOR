#!/usr/bin/env python3
"""
NeedleHaystack with FULL SEEDING for reproducibility.

Seeds both data generation AND model initialization.
Run twice to verify consistency.
"""
import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from src.data.needle_haystack import NeedleHaystackTask
from src.models.ssm import SimpleSSM
from src.models.attention import FullAttentionBaseline
from src.models.amor_v2 import AMORv2
from src.training.supervised import save_results


def set_all_seeds(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Config
device = 'cpu'
epochs = 30
batch_size = 32
lr = 5e-4
d_model = 64
MASTER_SEED = 42  # Master seed for everything

print("="*70)
print("NeedleHaystack - FULLY SEEDED (reproducible)")
print("="*70)
print(f"Device: {device}, Epochs: {epochs}, d_model: {d_model}")
print(f"MASTER SEED: {MASTER_SEED}")
print()

# Set seeds BEFORE creating datasets
set_all_seeds(MASTER_SEED)

# Create datasets - NO IMPOSSIBLE QUERIES
print("Creating NeedleHaystack datasets (impossible_query_prob=0)...")
train_ds = NeedleHaystackTask(
    size=3000,
    seq_length=256,
    noise_length_range=(50, 150),
    impossible_query_prob=0.0,
    seed=MASTER_SEED  # Data seed
)

# Use different seed for validation but still deterministic
set_all_seeds(MASTER_SEED + 1)
val_ds = NeedleHaystackTask(
    size=500,
    seq_length=256,
    noise_length_range=(50, 150),
    impossible_query_prob=0.0,
    seed=MASTER_SEED + 1
)

stats = train_ds.get_stats()
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
print(f"Retrieval rate: {stats['retrieval_rate']:.2%}")
print(f"Impossible rate: {stats['impossible_rate']:.2%}")
print()

results = {
    'timestamp': datetime.now().isoformat(),
    'task': 'NeedleHaystack_Seeded',
    'master_seed': MASTER_SEED,
    'config': {
        'epochs': epochs,
        'd_model': d_model,
        'seq_length': 256,
        'noise_range': [50, 150],
        'impossible_prob': 0.0,
    },
    'models': {}
}


def train_model(model, name, train_ds, epochs, lr, device, is_amor=False, model_seed=None):
    """Train a model and return it."""
    if model_seed is not None:
        set_all_seeds(model_seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in tqdm(loader, desc=f'{name} Epoch {epoch+1}', leave=False):
            tokens = batch['tokens'].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            if is_amor:
                nr = batch['needs_retrieval'][:, 1:].to(device)
                if model.gate_method == 'oracle':
                    outputs = model(x, ground_truth_gate=nr.float())
                else:
                    outputs = model(x)
                logits = outputs['logits']
            else:
                logits = model(x)

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = total_correct / total_tokens
            print(f"  {name} Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, acc={acc:.4f}")

    return model


def evaluate_model(model, name, val_ds, device, is_amor=False):
    """Evaluation for clean comparison (no impossible queries)."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size)

    total_correct = 0
    total_tokens = 0
    retrieval_correct = 0
    retrieval_total = 0
    local_correct = 0
    local_total = 0

    gate_fires_total = 0
    gate_total = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            nr = batch['needs_retrieval'][:, 1:].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            if is_amor:
                if model.gate_method == 'oracle':
                    outputs = model(x, ground_truth_gate=nr.float())
                else:
                    outputs = model(x)
                logits = outputs['logits']
                gate = outputs['gate']
                gate_fires_total += gate.sum().item()
                gate_total += gate.numel()
            else:
                logits = model(x)

            preds = logits.argmax(dim=-1)
            correct = (preds == y)

            total_correct += correct.sum().item()
            total_tokens += y.numel()

            # Retrieval accuracy
            retrieval_correct += (correct & nr).sum().item()
            retrieval_total += nr.sum().item()

            # Local accuracy
            local_mask = ~nr
            local_correct += (correct & local_mask).sum().item()
            local_total += local_mask.sum().item()

    metrics = {
        'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'retrieval_accuracy': retrieval_correct / retrieval_total if retrieval_total > 0 else 0,
        'local_accuracy': local_correct / local_total if local_total > 0 else 0,
        'params': sum(p.numel() for p in model.parameters()),
    }

    if is_amor:
        metrics['gate_fires'] = gate_fires_total / gate_total if gate_total > 0 else 0

    return metrics


# =====================================================================
# 1. SSM Only (seed = MASTER_SEED + 100)
# =====================================================================
print("="*70)
print("1. SSM Only")
print("="*70)
set_all_seeds(MASTER_SEED + 100)
ssm = SimpleSSM(vocab_size=train_ds.total_vocab, d_model=d_model, n_layers=2)
print(f"Parameters: {sum(p.numel() for p in ssm.parameters()):,}")
ssm = train_model(ssm, "SSM", train_ds, epochs, lr, device, is_amor=False, model_seed=MASTER_SEED + 100)
ssm_metrics = evaluate_model(ssm, "SSM Only", val_ds, device, is_amor=False)
results['models']['ssm_only'] = ssm_metrics
print(f"\nSSM Only: Acc={ssm_metrics['accuracy']:.2%}, Retrieval={ssm_metrics['retrieval_accuracy']:.2%}")

# =====================================================================
# 2. Full Attention (seed = MASTER_SEED + 200)
# =====================================================================
print("\n" + "="*70)
print("2. Full Attention")
print("="*70)
set_all_seeds(MASTER_SEED + 200)
attn = FullAttentionBaseline(vocab_size=train_ds.total_vocab, d_model=d_model, n_heads=4, n_layers=2)
print(f"Parameters: {sum(p.numel() for p in attn.parameters()):,}")
attn = train_model(attn, "Attention", train_ds, epochs, lr, device, is_amor=False, model_seed=MASTER_SEED + 200)
attn_metrics = evaluate_model(attn, "Full Attention", val_ds, device, is_amor=False)
results['models']['full_attention'] = attn_metrics
print(f"\nFull Attention: Acc={attn_metrics['accuracy']:.2%}, Retrieval={attn_metrics['retrieval_accuracy']:.2%}")

# =====================================================================
# 3. AMOR Oracle (seed = MASTER_SEED + 300)
# =====================================================================
print("\n" + "="*70)
print("3. AMOR Oracle (ceiling)")
print("="*70)
set_all_seeds(MASTER_SEED + 300)
amor_oracle = AMORv2(
    vocab_size=train_ds.total_vocab,
    d_model=d_model,
    gate_method='oracle',
    n_heads=4,
    n_layers=2,
)
print(f"Parameters: {sum(p.numel() for p in amor_oracle.parameters()):,}")
amor_oracle = train_model(amor_oracle, "AMOR-Oracle", train_ds, epochs, lr, device, is_amor=True, model_seed=MASTER_SEED + 300)
oracle_metrics = evaluate_model(amor_oracle, "AMOR Oracle", val_ds, device, is_amor=True)
results['models']['amor_oracle'] = oracle_metrics
print(f"\nAMOR Oracle: Acc={oracle_metrics['accuracy']:.2%}, Retrieval={oracle_metrics['retrieval_accuracy']:.2%}, Gate fires={oracle_metrics['gate_fires']:.2%}")

# =====================================================================
# 4. AMOR Entropy (seed = MASTER_SEED + 400)
# =====================================================================
print("\n" + "="*70)
print("4. AMOR Entropy (our method)")
print("="*70)
set_all_seeds(MASTER_SEED + 400)
amor_entropy = AMORv2(
    vocab_size=train_ds.total_vocab,
    d_model=d_model,
    gate_method='entropy',
    n_heads=4,
    n_layers=2,
    target_retrieval_rate=0.2,
)
print(f"Parameters: {sum(p.numel() for p in amor_entropy.parameters()):,}")
amor_entropy = train_model(amor_entropy, "AMOR-Entropy", train_ds, epochs, lr, device, is_amor=True, model_seed=MASTER_SEED + 400)
entropy_metrics = evaluate_model(amor_entropy, "AMOR Entropy", val_ds, device, is_amor=True)
results['models']['amor_entropy'] = entropy_metrics
print(f"\nAMOR Entropy: Acc={entropy_metrics['accuracy']:.2%}, Retrieval={entropy_metrics['retrieval_accuracy']:.2%}, Gate fires={entropy_metrics['gate_fires']:.2%}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*70)
print("NEEDLEHAYSTACK SEEDED RESULTS (MASTER_SEED={})".format(MASTER_SEED))
print("="*70)
print(f"{'Model':<20} {'Params':>10} {'Accuracy':>10} {'Retrieval Acc':>14} {'Gate Fires':>12}")
print("-"*70)
print(f"{'SSM Only':<20} {ssm_metrics['params']:>10,} {ssm_metrics['accuracy']:>10.2%} {ssm_metrics['retrieval_accuracy']:>14.2%} {'---':>12}")
print(f"{'Full Attention':<20} {attn_metrics['params']:>10,} {attn_metrics['accuracy']:>10.2%} {attn_metrics['retrieval_accuracy']:>14.2%} {'---':>12}")
print(f"{'AMOR Oracle':<20} {oracle_metrics['params']:>10,} {oracle_metrics['accuracy']:>10.2%} {oracle_metrics['retrieval_accuracy']:>14.2%} {oracle_metrics['gate_fires']:>12.2%}")
print(f"{'AMOR Entropy':<20} {entropy_metrics['params']:>10,} {entropy_metrics['accuracy']:>10.2%} {entropy_metrics['retrieval_accuracy']:>14.2%} {entropy_metrics['gate_fires']:>12.2%}")
print("-"*70)

# Save results
Path('experiments/needlehaystack_seeded').mkdir(parents=True, exist_ok=True)
save_results(results, 'experiments/needlehaystack_seeded/results.json')
print(f"\nResults saved to experiments/needlehaystack_seeded/results.json")
print("\nRun this script again to verify reproducibility!")
