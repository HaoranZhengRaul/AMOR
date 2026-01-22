"""
Attention modules: Full attention baseline and sparse attention for System 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FullAttentionBaseline(nn.Module):
    """
    Standard transformer with full causal attention.
    Used as upper bound for accuracy.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch, seq = x.shape

        pos = torch.arange(seq, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()

        h = self.transformer(h, mask=mask)
        logits = self.output(h)

        return logits


class SparseAttentionWithGhostKV(nn.Module):
    """
    System 2: Sparse attention using Ghost KV cache.
    Only computes attention where gate fires.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        top_k: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Ghost KV generator
        from .ghost_kv import GhostKVCache
        self.ghost_kv = GhostKVCache(d_model, n_heads)

    def forward(self, h_ssm, gate=None):
        batch, seq, _ = h_ssm.shape

        K, V = self.ghost_kv(h_ssm)
        Q = self.q_proj(h_ssm)

        Q = Q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(seq, seq, device=h_ssm.device), diagonal=1
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
