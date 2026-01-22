"""
Ghost KV Cache: Generate cheap K,V from SSM hidden states.

This addresses the "KV Cache Paradox":
- We need K,V for attention, but computing full transformer KVs is expensive
- Solution: Project SSM hidden states to K,V (O(N) cost, not O(N²))

The SSM has already processed the sequence and built up a representation.
We project those hidden states to serve as Keys and Values for attention.
"""
import torch
import torch.nn as nn


class GhostKVCache(nn.Module):
    """
    Generate K,V from SSM hidden states.
    O(N) cost instead of O(N²) for full transformer KVs.

    The "ghost" name comes from the fact that these K,V pairs are
    derived from SSM states rather than computed fresh - they're
    echoes/projections of what the SSM already learned.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, h_ssm):
        """
        Args:
            h_ssm: [batch, seq, d_model] SSM hidden states
        Returns:
            K: [batch, seq, d_model]
            V: [batch, seq, d_model]
        """
        K = self.k_proj(h_ssm)
        V = self.v_proj(h_ssm)
        return K, V
