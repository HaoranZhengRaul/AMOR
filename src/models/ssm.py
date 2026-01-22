"""
SSM Module: Simple GRU-based state space model.
(Can swap for Mamba later, GRU is simpler to debug)
"""
import torch
import torch.nn as nn


class SimpleSSM(nn.Module):
    """
    GRU-based SSM as System 1.
    Maintains compressed state, handles local patterns.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            d_model, d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_hidden=False):
        """
        Args:
            x: [batch, seq] token indices
            return_hidden: if True, return hidden states for gate

        Returns:
            logits: [batch, seq, vocab_size]
            hidden: [batch, seq, d_model] if return_hidden
        """
        h = self.embed(x)
        h, _ = self.gru(h)
        logits = self.output(h)

        if return_hidden:
            return logits, h
        return logits
