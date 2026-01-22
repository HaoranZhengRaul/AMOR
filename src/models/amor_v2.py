"""
AMOR v2: Adaptive Metacognitive Output Router

Architecture:
    Input → SSM (System 1) → Metacognitive Gate → Sparse Attention (System 2) → Output
                                   ↓
                    "High entropy = I don't know = retrieve"

Key Innovation:
Routes based on PREDICTION ENTROPY (epistemic uncertainty), not learned opaque vectors.
This provides interpretability: we can see exactly why the model chose to retrieve.

Components:
- SSM (System 1): Handles local patterns efficiently with O(N) cost
- Metacognitive Gate: Fires when SSM is uncertain (high entropy)
- Sparse Attention (System 2): Retrieves from Ghost KV cache when needed

Graceful Degradation:
When SSM fails globally, entropy is uniformly high, so gate fires everywhere.
This is correct - if System 1 is confused, System 2 should help everywhere.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SimpleSSM
from .attention import SparseAttentionWithGhostKV
from .gate import MetacognitiveGate


class AMORv2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        gate_method: str = 'entropy',
        attention_top_k: int = 3,
        target_retrieval_rate: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.gate_method = gate_method

        # System 1: SSM for local patterns
        self.ssm = SimpleSSM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Metacognitive Gate: decides when to retrieve
        self.gate = MetacognitiveGate(
            d_model=d_model,
            vocab_size=vocab_size,
            method=gate_method,
            target_retrieval_rate=target_retrieval_rate,
        )

        # System 2: Sparse Attention with Ghost KV
        self.attention = SparseAttentionWithGhostKV(
            d_model=d_model,
            n_heads=n_heads,
            top_k=attention_top_k,
        )

        # Combination layer: merge SSM and attention outputs
        self.combine = nn.Linear(d_model * 2, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, ground_truth_gate=None):
        """
        Args:
            x: [batch, seq] token indices
            ground_truth_gate: [batch, seq] binary labels (for oracle training)

        Returns:
            dict with:
                logits: [batch, seq, vocab_size] predictions
                gate: [batch, seq] binary gate decisions
                gate_prob: [batch, seq] soft gate probabilities
                entropy: [batch, seq] normalized entropy (for entropy method)
                ssm_hidden: [batch, seq, d_model] SSM hidden states
                ssm_logits: [batch, seq, vocab_size] SSM raw predictions
        """
        # System 1: SSM processes sequence
        ssm_logits, ssm_hidden = self.ssm(x, return_hidden=True)

        # Metacognitive Gate: decide where to retrieve
        gate, gate_prob, entropy = self.gate(
            ssm_hidden,
            logits=ssm_logits,
            ground_truth=ground_truth_gate,
        )

        # System 2: Sparse attention where gate fires
        attn_out = self.attention(ssm_hidden, gate=gate)

        # Combine SSM and attention outputs
        combined = torch.cat([ssm_hidden, attn_out], dim=-1)
        h = self.combine(combined)

        # Final prediction
        logits = self.output(h)

        return {
            'logits': logits,
            'gate': gate,
            'gate_prob': gate_prob,
            'entropy': entropy,
            'ssm_hidden': ssm_hidden,
            'ssm_logits': ssm_logits,
        }

    def compute_metrics(self, outputs, targets, needs_retrieval, is_impossible=None):
        """
        Compute comprehensive metrics for evaluation.

        Args:
            outputs: dict from forward()
            targets: [batch, seq] ground truth tokens
            needs_retrieval: [batch, seq] binary mask of retrieval positions
            is_impossible: [batch, seq] binary mask of impossible queries (optional)

        Returns:
            dict with metrics
        """
        logits = outputs['logits']
        gate = outputs['gate']

        # Accuracy metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
        accuracy = correct.mean()

        retrieval_mask = needs_retrieval.float()
        local_mask = 1 - retrieval_mask

        retrieval_acc = (correct * retrieval_mask).sum() / (retrieval_mask.sum() + 1e-10)
        local_acc = (correct * local_mask).sum() / (local_mask.sum() + 1e-10)

        # Gate metrics
        gate_fires = gate.mean()
        gate_precision = ((gate * needs_retrieval.float()).sum() / (gate.sum() + 1e-10))
        gate_recall = ((gate * needs_retrieval.float()).sum() / (needs_retrieval.float().sum() + 1e-10))
        gate_f1 = 2 * gate_precision * gate_recall / (gate_precision + gate_recall + 1e-10)

        # FLOPs saved (approximation: 1 - gate_fires)
        flops_saved = 1 - gate_fires

        metrics = {
            'accuracy': accuracy,
            'retrieval_accuracy': retrieval_acc,
            'local_accuracy': local_acc,
            'gate_fires': gate_fires,
            'gate_precision': gate_precision,
            'gate_recall': gate_recall,
            'gate_f1': gate_f1,
            'flops_saved': flops_saved,
        }

        # Metacognitive test (for NeedleHaystack task)
        if is_impossible is not None:
            possible_retrieval = needs_retrieval & ~is_impossible
            impossible_positions = is_impossible

            gate_on_possible = gate[possible_retrieval].mean() if possible_retrieval.any() else torch.tensor(0.0)
            gate_on_impossible = gate[impossible_positions].mean() if impossible_positions.any() else torch.tensor(0.0)

            metrics['gate_on_possible'] = gate_on_possible
            metrics['gate_on_impossible'] = gate_on_impossible
            metrics['metacognitive_gap'] = gate_on_possible - gate_on_impossible

        return metrics

    def get_param_groups(self, lr=1e-3, gate_lr_mult=1.0):
        """
        Get parameter groups for optimizer.
        Allows different learning rates for different components.
        """
        return [
            {'params': self.ssm.parameters(), 'lr': lr},
            {'params': self.gate.parameters(), 'lr': lr * gate_lr_mult},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.combine.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]
