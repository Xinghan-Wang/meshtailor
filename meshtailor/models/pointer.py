"""Pointer layer (paper Sec. B.3.3, Eq. 2).

Computes logits over the candidate set U = {[EOC], [EOS]} ∪ V via dot-product
attention:  l_{t,u} = <q_t, W e_u> + m_{t,u}.  Candidate ids: [EOC]=0, [EOS]=1,
vertex v -> v+2. The neighbor mask m is built by the top-level model from the
1-ring adjacency and the (teacher-forcing) input tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PointerLayer(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q: torch.Tensor, candidates: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # q: (B, seq, d), candidates: (B, N+2, d), mask: (B, seq, N+2) or None
        We = self.W(candidates)  # (B, N+2, d)
        logits = torch.bmm(q, We.transpose(1, 2))  # (B, seq, N+2)
        if mask is not None:
            logits = logits + mask
        return logits
