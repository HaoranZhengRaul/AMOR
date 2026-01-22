"""
SSM + MLP baseline: Same parameter count as AMOR, but no attention.
Tests whether extra parameters alone explain the performance gap.

This is a diagnostic model to understand the learned_ste anomaly:
- learned_ste achieves 100% retrieval accuracy with gate never firing
- SSM Only gets only 68% retrieval accuracy
- Question: Is the gap due to extra parameters or architecture?
"""
import torch
import torch.nn as nn
from .ssm import SimpleSSM


class SSMWithMLP(nn.Module):
    """SSM with extra MLP to match AMOR's parameter count."""

    def __init__(self, vocab_size, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.ssm = SimpleSSM(vocab_size, d_model, n_layers, dropout)

        # Extra MLP to match AMOR's combine layer params
        # AMOR has: combine Linear(d_model * 2, d_model) + output Linear
        # We add: MLP with similar param count
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_hidden=False):
        ssm_logits, h = self.ssm(x, return_hidden=True)
        h = self.mlp(h)
        logits = self.output(h)

        if return_hidden:
            return logits, h
        return logits
