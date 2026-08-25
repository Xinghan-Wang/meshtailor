"""Enc_G: mesh connectivity encoder via GraphSAGE (paper Sec. B.3.2).

Pipeline: vertex(pos|normal) -> Fourier feature -> MLP -> p_i (d_p=384)
         -> stack of SAGEConv layers (hidden [64,128,256,512], SiLU + LayerNorm)
         -> raw-coordinate fusion -> per-vertex embedding h_i in R^d (d=512).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from .fourier import FourierFeature


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        d_p: int = 384,
        hidden_widths: list[int] = (64, 128, 256, 512),
        d_model: int = 512,
        num_frequencies: int = 128,
        fourier_scale: float = 10.0,
    ):
        super().__init__()
        self.fourier = FourierFeature(in_dim, num_frequencies, fourier_scale)
        self.input_mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim + in_dim, d_p),
            nn.SiLU(),
            nn.Linear(d_p, d_p),
        )

        dims = [d_p] + list(hidden_widths)
        self.sage_layers = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(len(hidden_widths))]
        )
        self.sage_norms = nn.ModuleList([nn.LayerNorm(d) for d in hidden_widths])
        self.act = nn.SiLU()

        self.fusion = nn.Linear(hidden_widths[-1] + d_p, d_model)
        self.d_model = d_model

    def forward(self, vertices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        phi = self.fourier(vertices)
        p = self.input_mlp(torch.cat([phi, vertices], dim=-1))

        h = p
        for sage, norm in zip(self.sage_layers, self.sage_norms):
            h = sage(h, edge_index)
            h = self.act(h)
            h = norm(h)

        h_fused = self.fusion(torch.cat([h, p], dim=-1))
        return h_fused
