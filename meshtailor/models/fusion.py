"""Cross-attention fusion (paper Sec. B.3.2).

Injects global shape tokens Z into per-vertex embeddings. 2 pre-norm
cross-attention layers with vertex embeddings as queries and shape tokens as
key/value.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q_attn = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q_attn(query)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.ffn(self.norm_q_ffn(query)))
        return query


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, vertices: torch.Tensor, shape_tokens: torch.Tensor) -> torch.Tensor:
        x = vertices
        for layer in self.layers:
            x = layer(x, shape_tokens)
        return x
