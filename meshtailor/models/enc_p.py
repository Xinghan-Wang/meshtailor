"""Enc_P: frozen Michelangelo point-cloud encoder wrapper (paper Sec. B.3.2).

Loads the pretrained aligned-shape VAE, freezes it, and projects its
per-token latent Z (B, 256, 64) to the model dimension (B, 256, d_model).

The decoder-only import chain (inference_utils -> graphics.primitives -> cv2)
is stubbed since encode() never uses mesh extraction; torch.load is patched to
weights_only=False for the PL checkpoint (numpy scalars, trusted source).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        michelangelo_root: str | None = None,
        ckpt_rel: str = "checkpoints/aligned_shape_latents/shapevae-256.ckpt",
        config_rel: str = "configs/aligned_shape_latents/shapevae-256.yaml",
        z_dim: int = 64,
        d_model: int = 512,
    ):
        super().__init__()
        if michelangelo_root is None:
            michelangelo_root = str(Path(__file__).resolve().parents[2] / "Michelangelo")
        self._load_michelangelo(michelangelo_root, ckpt_rel, config_rel)
        self.proj = nn.Linear(z_dim, d_model)
        self.d_model = d_model

    def _load_michelangelo(self, root: str, ckpt_rel: str, config_rel: str) -> None:
        stub = types.ModuleType("michelangelo.models.tsal.inference_utils")
        stub.extract_geometry = lambda *a, **k: None
        sys.modules["michelangelo.models.tsal.inference_utils"] = stub

        if not getattr(torch, "_meshtailor_load_patched", False):
            _orig_load = torch.load

            def _patched_load(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig_load(*a, **kw)

            torch.load = _patched_load
            torch._meshtailor_load_patched = True

        if root not in sys.path:
            sys.path.insert(0, root)
        from michelangelo.utils.misc import get_config_from_file, instantiate_from_config

        cwd = os.getcwd()
        os.chdir(root)
        try:
            cfg = get_config_from_file(config_rel)
            mcfg = cfg.model if hasattr(cfg, "model") else cfg
            model = instantiate_from_config(mcfg, ckpt_path=ckpt_rel)
        finally:
            os.chdir(cwd)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            model.set_shape_model_only()
        except Exception:
            pass
        self.enc_p = model

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        self.enc_p.eval()
        # Freeze only the pretrained Michelangelo encoder.  The projection
        # into MeshTailor's d_model is part of this model and must train.
        with torch.no_grad():
            z = self.enc_p.encode(surface, sample_posterior=False)
        return self.proj(z)
