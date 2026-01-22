"""
Metacognitive Gate: Learns WHEN to trigger retrieval.

THE KEY INNOVATION: Route based on ENTROPY (epistemic uncertainty).

Methods:
- 'entropy': Gate based on normalized prediction entropy (PRIMARY - most novel)
- 'learned_ste': MoE-style with Straight-Through Estimator (backup)
- 'oracle': Ground truth (debugging)

Normalized Entropy:
We use entropy / max_entropy to get values in [0, 1].
This makes the threshold interpretable:
- 0.5 = "50% of max uncertainty"
- Works regardless of vocab size

Graceful Degradation:
When SSM fails globally (like on hard tasks), entropy is uniformly high,
so the gate fires everywhere. This is correct behavior - if the SSM is
confused about everything, we should retrieve for everything.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class MetacognitiveGate(nn.Module):

    def __init__(
        self,
        d_model: int,
        vocab_size: int = None,
        method: str = 'entropy',
        entropy_threshold: float = 0.5,
        target_retrieval_rate: float = 0.2,
        noise_std: float = 0.1,
    ):
        """
        Args:
            d_model: Hidden dimension size
            vocab_size: Vocab size for entropy normalization (required for entropy method)
            method: 'entropy', 'learned_ste', or 'oracle'
            entropy_threshold: Threshold for normalized entropy (0-1 scale)
            target_retrieval_rate: Target rate for balance loss
            noise_std: Noise for training stability (learned methods)
        """
        super().__init__()
        self.method = method
        self.target_rate = target_retrieval_rate
        self.noise_std = noise_std
        self.d_model = d_model

        if method == 'entropy':
            assert vocab_size is not None, "vocab_size required for entropy method"
            self.max_entropy = math.log(vocab_size)
            self.threshold = nn.Parameter(torch.tensor(entropy_threshold))
            self.entropy_scale = nn.Parameter(torch.tensor(5.0))

        elif method in ['learned_ste', 'learned_rl']:
            self.gate = nn.Linear(d_model, 1)

    def forward(self, h, logits=None, ground_truth=None):
        """
        Args:
            h: [batch, seq, d_model] hidden states from SSM
            logits: [batch, seq, vocab_size] SSM output logits (for entropy method)
            ground_truth: [batch, seq] binary labels (for oracle method)

        Returns:
            gate: [batch, seq] binary decisions (with STE gradient)
            gate_prob: [batch, seq] soft probabilities
            entropy: [batch, seq] normalized entropy values (for analysis)
        """
        batch, seq = h.shape[:2]

        if self.method == 'oracle':
            assert ground_truth is not None
            gate = ground_truth.float()
            gate_prob = gate
            entropy = None

        elif self.method == 'entropy':
            assert logits is not None, "Entropy method requires logits!"

            # Compute raw entropy
            probs = F.softmax(logits, dim=-1)
            raw_entropy = -(probs * (probs + 1e-10).log()).sum(-1)

            # Normalize to [0, 1] for scale-invariance
            entropy = raw_entropy / self.max_entropy

            # Compute gate probability using learned threshold and scale
            gate_prob = torch.sigmoid(
                self.entropy_scale * (entropy - self.threshold)
            )

            # Straight-Through Estimator: hard decision with soft gradient
            gate_hard = (gate_prob > 0.5).float()
            gate = gate_hard - gate_prob.detach() + gate_prob

        elif self.method == 'learned_ste':
            gate_logits = self.gate(h).squeeze(-1)

            # Add noise during training for exploration
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(gate_logits) * self.noise_std
                gate_logits = gate_logits + noise

            gate_prob = torch.sigmoid(gate_logits)

            # Straight-Through Estimator
            gate_hard = (gate_prob > 0.5).float()
            gate = gate_hard - gate_prob.detach() + gate_prob
            entropy = None

        elif self.method == 'learned_rl':
            gate_logits = self.gate(h).squeeze(-1)
            gate_prob = torch.sigmoid(gate_logits)

            if self.training:
                dist = Bernoulli(gate_prob)
                gate = dist.sample()
            else:
                gate = (gate_prob > 0.5).float()
            entropy = None

        return gate, gate_prob, entropy

    def balance_loss(self, gate_prob):
        """
        Auxiliary loss to prevent collapse to always-on or always-off.
        Encourages gate to fire at target_rate.
        """
        actual_rate = gate_prob.mean()
        return (actual_rate - self.target_rate) ** 2

    def entropy_analysis(self, entropy, needs_retrieval):
        """
        Analyze if entropy correlates with retrieval need.
        Useful for debugging and monitoring.
        """
        if entropy is None:
            return {}

        retrieval_mask = needs_retrieval.bool()
        local_mask = ~retrieval_mask

        entropy_at_retrieval = entropy[retrieval_mask].mean() if retrieval_mask.any() else torch.tensor(0.0)
        entropy_at_local = entropy[local_mask].mean() if local_mask.any() else torch.tensor(0.0)

        return {
            'entropy_at_retrieval': entropy_at_retrieval,
            'entropy_at_local': entropy_at_local,
            'entropy_gap': entropy_at_retrieval - entropy_at_local,
        }

    def get_log_prob(self, gate_prob, gate_action):
        """For REINFORCE (RL method only)"""
        log_prob = torch.where(
            gate_action > 0.5,
            (gate_prob + 1e-10).log(),
            (1 - gate_prob + 1e-10).log()
        )
        return log_prob
