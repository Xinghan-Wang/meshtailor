"""MeshTailor dataset (batched).

- __getitem__ returns one garment; optional sequence augmentation for train
  (loop start rotation + direction flip).
- collate_batch assembles a padded/offset batch for the batched model forward:
  vertices/edge_index use PyG-style big-graph cat+offset (no padding), the rest
  (surface, chains, adj) are kept per-item in lists; the model reconstructs
  padded views via batch_ptr.
"""
from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = PROJECT_ROOT / "meshtailor" / "data" / "split_seamless_128k.json"
DATA_DIR = PROJECT_ROOT / "processed_data_seamless_maximal"


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Pickleable Windows DataLoader worker seeding hook."""
    # DataLoader derives a fresh torch.initial_seed() for each epoch/worker
    # from its generator.  Use it so augmentation is reproducible without
    # repeating the same random flip stream every epoch.
    worker_seed = int(torch.initial_seed())
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def _angle_deficit(V, F):
    """Discrete Gaussian curvature proxy per vertex (angle deficit, numpy)."""
    V = V.numpy().astype(np.float64) if torch.is_tensor(V) else np.asarray(V, dtype=np.float64)
    F = F.numpy().astype(np.int64) if torch.is_tensor(F) else np.asarray(F, dtype=np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def ang(p, q, r):
        u = p - q
        v = r - q
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return np.arccos(np.clip((u * v).sum(1), -1.0, 1.0))

    out = np.zeros(len(V), dtype=np.float64)
    np.add.at(out, F[:, 0], ang(b, a, c))
    np.add.at(out, F[:, 1], ang(a, b, c))
    np.add.at(out, F[:, 2], ang(a, c, b))
    return 2.0 * np.pi - out


def canonicalize_chains(chains, V, F):
    """Geometric canonical chain starts (validated: recall 0.749 vs oracle
    0.740 on 150 test garments; tools/canonical_start_probe.py).

    - loop:       start = vertex with max |angle deficit| (tiebreak min index)
    - open chain: start = endpoint with larger |angle deficit|,
                  direction toward the other endpoint
    """
    K = _angle_deficit(V, F)
    out = []
    for c in chains:
        cc = [int(x) for x in c]
        if len(cc) < 2:
            out.append(cc)
            continue
        if cc[0] == cc[-1]:
            body = cc[:-1]
            s = int(np.argmax(np.abs(K[body])))
            rot = body[s:] + body[:s]
            out.append(rot + [rot[0]])
        else:
            if abs(K[cc[0]]) >= abs(K[cc[-1]]):
                out.append(cc)
            else:
                out.append(cc[::-1])
    return out


def augment_chains(chains: list[list[int]]) -> list[list[int]]:
    """Direction-flip augmentation for loops only. No start rotation: chain
    starts are canonicalized geometrically (curvature anchor) and rotating
    them would re-diffuse the p(start) training signal."""
    out: list[list[int]] = []
    for c in chains:
        cc = [int(x) for x in c]
        if len(cc) >= 3 and cc[0] == cc[-1] and random.random() < 0.5:
            inner = cc[:-1]
            # Reverse traversal while retaining the canonical anchor at
            # position zero.  A plain inner[::-1] changes the start vertex
            # and re-diffuses the canonical start target.
            rev = [inner[0]] + inner[:0:-1]
            out.append(rev + [rev[0]])
        else:
            out.append(cc)
    return out


class MeshTailorDataset(Dataset):
    def __init__(self, split: str = "train", split_file: Path = SPLIT_FILE, data_dir: Path = DATA_DIR,
                 augment: bool | None = None, gids: list[str] | None = None):
        self.split = split
        self.data_dir = Path(data_dir)
        self.augment = (split == "train") if augment is None else augment
        with open(split_file, encoding="utf-8") as f:
            split_gids = json.load(f)[split]
        self.gids = list(split_gids if gids is None else gids)

    def __len__(self) -> int:
        return len(self.gids)

    def __getitem__(self, idx: int) -> dict:
        gid = self.gids[idx]
        pt = torch.load(self.data_dir / f"{gid}.pt", weights_only=False)

        verts6 = torch.cat([pt["vertices"], pt["vertex_normals"]], dim=-1).float()  # (N, 6)
        edges = pt["edges"].long()
        edge_index = torch.cat([edges, edges.flip(1)], dim=0).t().contiguous()  # (2, 2E)

        n = pt["n_vertices"]
        adj = torch.zeros(n, n, dtype=torch.bool)
        adj[edges[:, 0], edges[:, 1]] = True
        adj[edges[:, 1], edges[:, 0]] = True

        chains = canonicalize_chains(pt["ordered_chains"], pt["vertices"], pt["faces"])
        if self.augment:
            chains = augment_chains(chains)

        return {
            "gid": gid,
            "vertices6": verts6,
            "edge_index": edge_index,
            "surface": pt["surface_points"].unsqueeze(0).float(),  # (1, 2048, 6)
            "chains": chains,
            "adj": adj,
            "n": n,
        }


def collate_batch(batch: list[dict]) -> dict:
    B = len(batch)
    n_list = [b["n"] for b in batch]
    offsets = torch.tensor([0] + n_list[:-1]).cumsum(0)

    vertices = torch.cat([b["vertices6"] for b in batch], dim=0)  # (total_N, 6)
    batch_ptr = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_list)])
    edge_index = torch.cat([b["edge_index"] + offsets[i] for i, b in enumerate(batch)], dim=1)
    surface = torch.stack([b["surface"][0] for b in batch])  # (B, 2048, 6)

    return {
        "vertices6": vertices,
        "edge_index": edge_index,
        "batch_ptr": batch_ptr,
        "n_list": n_list,
        "max_N": max(n_list),
        "surface": surface,
        "chains_list": [b["chains"] for b in batch],
        "adj_list": [b["adj"] for b in batch],
        "B": B,
    }


def make_loader(split: str = "train", batch_size: int = 8, shuffle: bool = True,
                num_workers: int = 0, data_dir: Path | str = DATA_DIR,
                split_file: Path | str = SPLIT_FILE, seed: int = 0,
                gids: list[str] | None = None) -> DataLoader:
    augment = split == "train"
    ds = MeshTailorDataset(split=split, augment=augment, data_dir=data_dir,
                           split_file=split_file, gids=gids)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_batch, generator=generator,
                      worker_init_fn=(partial(_seed_worker, base_seed=int(seed))
                                      if num_workers else None))
