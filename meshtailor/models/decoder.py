"""MeshTailor Transformer decoder (paper Sec. B.3.3).

6 pre-norm decoder layers. Each layer: causal self-attention with RoPE
(implemented via F.scaled_dot_product_attention so PyTorch auto-selects the
Flash/mem-efficient backend under bf16), cross-attention to shape tokens Z
(nn.MultiheadAttention, need_weights=False -> SDPA fast path), and a FFN.
A learned chain-local positional embedding (resets after every [EOC]) is added
to the token embeddings before the stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q, k.

    q, k: (B, S, H, head_dim); position_ids: (B, S).
    """
    head_dim = q.shape[-1]
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    angles = position_ids.float()[..., None] * freqs  # (B, S, half)
    cos = angles.cos()[:, :, None, :]
    sin = angles.sin()[:, :, None, :]

    def rotate(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    return rotate(q), rotate(k)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, ffn_mult: int = 4,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        self.norm_self = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, d_model * 3)
        self.self_proj = nn.Linear(d_model, d_model)

        self.norm_cross_q = nn.LayerNorm(d_model)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor, z: torch.Tensor, position_ids: torch.Tensor,
                past_kv=None) -> tuple[torch.Tensor, tuple | None]:
        B, S, D = x.shape
        xn = self.norm_self(x)
        qkv = self.self_qkv(xn).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, S, H, head_dim)
        q, k = apply_rope(q, k, position_ids, self.rope_base)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, H, S, head_dim)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        sa = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past_kv is None),
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        sa = sa.transpose(1, 2).reshape(B, S, D)
        x = x + self.dropout(self.self_proj(sa))

        qn = self.norm_cross_q(x)
        zn = self.norm_cross_kv(z)
        ca, _ = self.cross_attn(qn, zn, zn, need_weights=False)
        x = x + self.dropout(ca)

        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, new_kv


class MeshTailorDecoder(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 6,
                 max_chain_pos: int = 512, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dropout=dropout, rope_base=rope_base)
             for _ in range(num_layers)]
        )
        self.chain_pe = nn.Embedding(max_chain_pos + 1, d_model)
        self.d_model = d_model

    def forward(self, token_emb: torch.Tensor, z: torch.Tensor,
                position_ids: torch.Tensor, chain_pos: torch.Tensor,
                past_kvs: list | None = None,
                collect_kv: bool = False) -> tuple[torch.Tensor, list | None]:
        x = token_emb + self.chain_pe(chain_pos)
        # collect_kv: build KV cache even in training (for scheduled-sampling
        # step-by-step decoding). Also skip grad checkpointing then — it is
        # incompatible with the KV-cache semantics, and S=1 makes it pointless.
        use_cache = collect_kv or not self.training
        new_kvs: list | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pk = past_kvs[i] if past_kvs is not None else None
            if self.training and not collect_kv:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, z, position_ids, pk, use_reentrant=False
                )[0]
            else:
                x, new_kv = layer(x, z, position_ids, pk)
                if new_kvs is not None:
                    new_kvs.append(new_kv)
        return x, new_kvs
