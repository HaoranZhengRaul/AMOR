#!/usr/bin/env python3
"""
Compare Ghost KV vs Raw Embedding KV attention for NeedleHaystack.

SEEDED VERSION: Full seeding for reproducibility (same as run_needlehaystack_seeded.py).
This ensures KV comparison results are consistent with baseline results.

Ghost KV: K,V projected from SSM hidden states (with temporal context)
Raw Embedding KV: K,V projected from raw embeddings (no temporal context)
"""
import sys
sys.path.append('.')

import torch
import torch.nn as nn
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
from src.models.gate import MetacognitiveGate
from src.models.amor_v2 import AMORv2
from src.training.supervised import save_results


def set_all_seeds(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = 'cpu'
batch_size = 32
lr = 5e-4
epochs = 30
MASTER_SEED = 42  # Same as run_needlehaystack_seeded.py


class RawEmbeddingKVAttention(nn.Module):
    """Attention with K,V from raw embeddings (not SSM hidden states)."""

    def __init__(self, d_model: int, n_heads: int = 4, top_k: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_query, h_kv, gate=None):
        batch, seq, _ = h_query.shape
        import math

        Q = self.q_proj(h_query)
        K = self.k_proj(h_kv)
        V = self.v_proj(h_kv)

        Q = Q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(seq, seq, device=h_query.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        if self.top_k < seq:
            topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
            hard_mask = torch.zeros_like(scores).scatter(-1, topk_idx, 1.0)
            scores = scores.masked_fill(hard_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        out = self.out_proj(out)

        if gate is not None:
            out = out * gate.unsqueeze(-1)

        return out


class AMORRawEmbeddingKV(nn.Module):
    """AMOR with Raw Embedding KV attention (K,V from embeddings, not SSM hidden)."""

    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, top_k=3, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.gate_method = 'oracle'

        self.embed = nn.Embedding(vocab_size, d_model)
        self.ssm = SimpleSSM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.gate = MetacognitiveGate(
            d_model=d_model,
            vocab_size=vocab_size,
            method='oracle',
        )
        self.attention = RawEmbeddingKVAttention(d_model, n_heads, top_k)
        self.combine = nn.Linear(d_model * 2, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, ground_truth_gate=None):
        batch, seq = x.shape
        h_embed = self.embed(x)
        ssm_logits, ssm_hidden = self.ssm(x, return_hidden=True)
        gate, gate_prob, entropy = self.gate(
            ssm_hidden,
            logits=ssm_logits,
            ground_truth=ground_truth_gate,
        )
        attn_out = self.attention(ssm_hidden, h_embed, gate=gate)
        combined = torch.cat([ssm_hidden, attn_out], dim=-1)
        h = self.combine(combined)
        logits = self.output(h)

        return {
            'logits': logits,
            'gate': gate,
            'gate_prob': gate_prob,
            'entropy': entropy,
        }


def train_and_eval(model, train_ds, val_ds, name, model_seed):
    """Train model and return retrieval accuracy + gate fire rate."""
    set_all_seeds(model_seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}', leave=False):
            tokens = batch['tokens'].to(device)
            nr = batch['needs_retrieval'][:, 1:].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            outputs = model(x, ground_truth_gate=nr.float())
            logits = outputs['logits']

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            retr_correct = retr_total = 0
            gate_fires = gate_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch['tokens'].to(device)
                    nr = batch['needs_retrieval'][:, 1:].to(device)
                    x, y = tokens[:, :-1], tokens[:, 1:]

                    outputs = model(x, ground_truth_gate=nr.float())
                    preds = outputs['logits'].argmax(dim=-1)
                    gate = outputs['gate']

                    retr_correct += ((preds == y) & nr).sum().item()
                    retr_total += nr.sum().item()

                    gate_fires += gate.sum().item()
                    gate_total += gate.numel()

            retr_acc = retr_correct / retr_total if retr_total > 0 else 0
            gate_rate = gate_fires / gate_total if gate_total > 0 else 0
            print(f"  {name} Epoch {epoch+1}: Retrieval Acc = {retr_acc:.2%}, Gate Fires = {gate_rate:.2%}")

    # Final eval
    model.eval()
    retr_correct = retr_total = 0
    gate_fires = gate_total = 0
    with torch.no_grad():
        for batch in val_loader:
            tokens = batch['tokens'].to(device)
            nr = batch['needs_retrieval'][:, 1:].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            outputs = model(x, ground_truth_gate=nr.float())
            preds = outputs['logits'].argmax(dim=-1)
            gate = outputs['gate']

            retr_correct += ((preds == y) & nr).sum().item()
            retr_total += nr.sum().item()

            gate_fires += gate.sum().item()
            gate_total += gate.numel()

    retr_acc = retr_correct / retr_total if retr_total > 0 else 0
    gate_rate = gate_fires / gate_total if gate_total > 0 else 0

    return retr_acc, gate_rate


print("="*70)
print("Ghost KV vs Raw Embedding KV Comparison (SEEDED for reproducibility)")
print("="*70)
print(f"Device: {device}, Epochs: {epochs}, d_model: 64")
print(f"MASTER SEED: {MASTER_SEED}")
print()

# Set seeds for data generation (same as run_needlehaystack_seeded.py)
set_all_seeds(MASTER_SEED)

print("Creating NeedleHaystack datasets (impossible_query_prob=0.0)...")
train_ds = NeedleHaystackTask(
    size=3000,
    seq_length=256,
    noise_length_range=(50, 150),
    impossible_query_prob=0.0,
    seed=MASTER_SEED
)

set_all_seeds(MASTER_SEED + 1)
val_ds = NeedleHaystackTask(
    size=500,
    seq_length=256,
    noise_length_range=(50, 150),
    impossible_query_prob=0.0,
    seed=MASTER_SEED + 1
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
print()

results = {
    'timestamp': datetime.now().isoformat(),
    'task': 'KV_Comparison_Seeded',
    'master_seed': MASTER_SEED,
    'config': {
        'epochs': epochs,
        'd_model': 64,
        'seq_length': 256,
        'noise_range': [50, 150],
        'impossible_query_prob': 0.0,
    },
    'models': {}
}

# Test Ghost KV (original AMOR) - use MASTER_SEED + 300 (same as Oracle in baseline)
print("\n" + "="*70)
print("1. Testing AMOR with Ghost KV (SSM hidden -> K,V)")
print("="*70)

# CRITICAL: Set seed BEFORE model creation for reproducible weight initialization
set_all_seeds(MASTER_SEED + 300)

ghost_kv_model = AMORv2(
    vocab_size=train_ds.total_vocab,
    d_model=64,
    gate_method='oracle',
    n_heads=4,
    n_layers=2,
)
ghost_kv_params = sum(p.numel() for p in ghost_kv_model.parameters())
print(f"Parameters: {ghost_kv_params:,}")
ghost_kv_acc, ghost_kv_gate = train_and_eval(ghost_kv_model, train_ds, val_ds, "Ghost-KV", MASTER_SEED + 300)

print(f"\nGhost KV: Retrieval Acc={ghost_kv_acc:.2%}, Gate fires={ghost_kv_gate:.2%}")

results['models']['ghost_kv'] = {
    'retrieval_accuracy': ghost_kv_acc,
    'gate_fires': ghost_kv_gate,
    'params': ghost_kv_params,
}

# Test Raw Embedding KV - use MASTER_SEED + 500 (different seed)
print("\n" + "="*70)
print("2. Testing AMOR with Raw Embedding KV (Embeddings -> K,V)")
print("="*70)

# CRITICAL: Set seed BEFORE model creation for reproducible weight initialization
set_all_seeds(MASTER_SEED + 500)

raw_embed_kv_model = AMORRawEmbeddingKV(
    vocab_size=train_ds.total_vocab,
    d_model=64,
    n_layers=2,
    n_heads=4,
    top_k=3,
)
raw_embed_kv_params = sum(p.numel() for p in raw_embed_kv_model.parameters())
print(f"Parameters: {raw_embed_kv_params:,}")
raw_embed_kv_acc, raw_embed_kv_gate = train_and_eval(raw_embed_kv_model, train_ds, val_ds, "Raw-Embed-KV", MASTER_SEED + 500)

print(f"\nRaw Embedding KV: Retrieval Acc={raw_embed_kv_acc:.2%}, Gate fires={raw_embed_kv_gate:.2%}")

results['models']['raw_embedding_kv'] = {
    'retrieval_accuracy': raw_embed_kv_acc,
    'gate_fires': raw_embed_kv_gate,
    'params': raw_embed_kv_params,
}

# Summary
print("\n" + "="*70)
print("KV METHOD COMPARISON SUMMARY (SEEDED)")
print("="*70)
print(f"\nGhost KV (SSM hidden -> K,V):     {ghost_kv_acc:.2%} retrieval, {ghost_kv_gate:.2%} gate fires ({ghost_kv_params:,} params)")
print(f"Raw Embedding KV (Embed -> K,V):  {raw_embed_kv_acc:.2%} retrieval, {raw_embed_kv_gate:.2%} gate fires ({raw_embed_kv_params:,} params)")

ratio = ghost_kv_acc / raw_embed_kv_acc if raw_embed_kv_acc > 0 else float('inf')
print(f"\nGhost KV is {ratio:.1f}x better than Raw Embedding KV")

# NOTE: Ghost KV should match Oracle from run_needlehaystack_seeded.py (35.18%)
# since they use the same seed (MASTER_SEED + 300)
print(f"\nExpected: Ghost KV should match Oracle baseline (~35.18%)")

if ghost_kv_acc > raw_embed_kv_acc + 0.1:
    print("\n--> Ghost KV provides CRITICAL advantage (SSM temporal context matters)")
    results['conclusion'] = 'ghost_kv_critical'
else:
    print("\n--> Unexpected: Ghost KV does not dominate")
    results['conclusion'] = 'unexpected'

# Save results
Path('experiments/diagnostic_needlehaystack').mkdir(parents=True, exist_ok=True)
save_results(results, 'experiments/diagnostic_needlehaystack/kv_comparison_seeded.json')
print(f"\nResults saved to experiments/diagnostic_needlehaystack/kv_comparison_seeded.json")
