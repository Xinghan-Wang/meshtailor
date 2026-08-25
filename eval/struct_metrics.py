"""Structural metrics: chart count and seam-length / 3D-area.

Reads generated seam.json under test_outputs/<gid>/ and the original .pt
for vertices/faces and ground-truth chains. Reports generated vs ground-truth.

Chart count = number of connected components of the face-dual graph after
removing seam edges (a seam edge separates the two faces sharing it).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]; v1 = vertices[faces[:, 1]]; v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def chart_count(faces: np.ndarray, seam_set: set[tuple[int, int]]) -> int:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for v0, v1 in ((a, b), (b, c), (c, a)):
            edge_to_faces[(min(v0, v1), max(v0, v1))].append(fi)
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for key, fl in edge_to_faces.items():
        if key in seam_set:
            continue
        for i in range(1, len(fl)):
            union(fl[0], fl[i])
    return len({find(i) for i in range(len(faces))})


def seam_length_over_area(vertices: np.ndarray, faces: np.ndarray,
                          seam_edges) -> tuple[float, float, float]:
    sl = 0.0
    for v0, v1 in seam_edges:
        sl += float(np.linalg.norm(vertices[v0] - vertices[v1]))
    area = float(face_areas(vertices, faces).sum())
    return sl, area, (sl / area if area > 0 else 0.0)


def chains_to_seam_set(chains) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    for c in chains:
        for k in range(len(c) - 1):
            a, b = int(c[k]), int(c[k + 1])
            s.add((min(a, b), max(a, b)))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_out", default=str(ROOT / "test_outputs"))
    ap.add_argument("--data_dir", default=str(ROOT / "processed_data_seamless_v13"))
    args = ap.parse_args()

    test_out = Path(args.test_out)
    data_dir = Path(args.data_dir)
    gids = sorted(p.name for p in test_out.iterdir() if p.is_dir())

    gen_charts, gt_charts = [], []
    gen_sl, gt_sl = [], []
    for gid in gids:
        seam_gen = [tuple(e) for e in json.loads((test_out / gid / "seam.json").read_text())]
        seam_gen_set = {(min(a, b), max(a, b)) for a, b in seam_gen}
        pt = torch.load(data_dir / f"{gid}.pt", weights_only=False)
        V = pt["vertices"].numpy().astype(np.float64)
        F = pt["faces"].numpy().astype(np.int64)
        gt_set = chains_to_seam_set(pt["ordered_chains"])

        gen_charts.append(chart_count(F, seam_gen_set))
        gt_charts.append(chart_count(F, gt_set))
        _, _, r_gen = seam_length_over_area(V, F, seam_gen_set)
        _, _, r_gt = seam_length_over_area(V, F, gt_set)
        gen_sl.append(r_gen)
        gt_sl.append(r_gt)

    n = len(gids)
    print(f"=== structural metrics on {n} garments ===")
    print(f"{'metric':<22} {'generated':>12} {'ground_truth':>14}")
    print(f"{'chart count (↓)':<22} {np.mean(gen_charts):>12.2f} {np.mean(gt_charts):>14.2f}")
    print(f"{'seam_len / area (↓)':<22} {np.mean(gen_sl):>12.4f} {np.mean(gt_sl):>14.4f}")


if __name__ == "__main__":
    main()
