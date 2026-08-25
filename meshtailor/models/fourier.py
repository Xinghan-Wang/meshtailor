"""Fourier feature encoder for coordinate-based vertex features (paper Sec. B.3.2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeature(nn.Module):
    """Gaussian random Fourier features: phi(x) = [sin(2 pi B x), cos(2 pi B x)].

    The matrix B is fixed (Gaussian) at construction, matching the standard
    positional encoding used for coordinate inputs.
    """

    def __init__(self, in_dim: int = 6, num_frequencies: int = 128, scale: float = 10.0):
        super().__init__()
        B = torch.randn(num_frequencies, in_dim) * scale
        self.register_buffer("B", B)
        self.out_dim = num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B.t()
        return torch.cat([torch.sin(2.0 * torch.pi * proj),
                          torch.cos(2.0 * torch.pi * proj)], dim=-1)
