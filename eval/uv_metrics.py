"""UV quality metrics (paper C.1) from an unwrapped mesh (uv.obj with v + vt).

Metrics:
  - area distortion: std of log(UV_area / 3D_area) across triangles (scale-invariant
    area stretch; lower = more uniform scaling).
  - chart count: number of UV islands (UV-space connected components).
  - compactness: 4*pi*area / perimeter^2 per UV island (1 = circle).
  - convexity: area / convex_hull_area per UV island (1 = convex).
  - boundary jaggedness: mean ||p_{i-1} - 2 p_i + p_{i+1}|| on resampled UV boundaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_uv_obj(path: Path):
    verts3d, uvs, faces = [], [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts3d.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "vt":
                uvs.append([float(t[1]), float(t[2])])
            elif t[0] == "f":
                face = []
                for tok in t[1:]:
                    v_idx, vt_idx = tok.split("/")[:2]
                    face.append((int(v_idx) - 1, int(vt_idx) - 1 if vt_idx else -1))
                faces.append(face)
    return np.array(verts3d, dtype=np.float64), np.array(uvs, dtype=np.float64), faces


def _tri_area_2d(p0, p1, p2):
    return 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _tri_area_3d(p0, p1, p2):
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def area_distortion(verts3d, uvs, faces):
    ratios = []
    for face in faces:
        (v0, t0), (v1, t1), (v2, t2) = face
        a3 = _tri_area_3d(verts3d[v0], verts3d[v1], verts3d[v2])
        if t0 < 0 or t1 < 0 or t2 < 0 or a3 < 1e-12:
            continue
        a2 = _tri_area_2d(uvs[t0], uvs[t1], uvs[t2])
        if a2 < 1e-12:
            continue
        ratios.append(np.log(a2 / a3))
    if not ratios:
        return float("nan")
    return float(np.std(ratios))


def uv_islands(uvs, faces):
    """Union-find over faces sharing a UV edge (by vt index)."""
    n = len(faces)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        vt = [face[k][1] for k in range(3)]
        for j in range(3):
            a, b = vt[j], vt[(j + 1) % 3]
            if a < 0 or b < 0:
                continue
            edge_to_faces[(min(a, b), max(a, b))].append(fi)
    for fl in edge_to_faces.values():
        for i in range(1, len(fl)):
            union(fl[0], fl[i])
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def island_polygons(uvs, faces, island):
    """Return 2D UV points and edge list for the island polygon boundary."""
    edges_count: dict[tuple[int, int], int] = defaultdict(int)
    for fi in island:
        vt = [faces[fi][k][1] for k in range(3)]
        for j in range(3):
            a, b = vt[j], vt[(j + 1) % 3]
            edges_count[(min(a, b), max(a, b))] += 1
    boundary_edges = [e for e, c in edges_count.items() if c == 1]
    if not boundary_edges:
        return None, None
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    start = boundary_edges[0][0]
    loop = [start]
    prev, cur = -1, start
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(boundary_edges) + 1:
            break
    return uvs[loop], loop


def compactness_convexity(uvs, faces, island):
    from scipy.spatial import ConvexHull
    poly2d, _ = island_polygons(uvs, faces, island)
    if poly2d is None or len(poly2d) < 3:
        return None, None
    # shoelace area + perimeter
    x, y = poly2d[:, 0], poly2d[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    peri = np.sum(np.linalg.norm(np.roll(poly2d, -1, axis=0) - poly2d, axis=1))
    compact = (4 * np.pi * area / peri ** 2) if peri > 0 else 0.0
    try:
        hull = ConvexHull(poly2d)
        hull_area = hull.volume  # 2D -> area
    except Exception:
        hull_area = area
    convex = (area / hull_area) if hull_area > 0 else 0.0
    return compact, convex


def boundary_jaggedness(uvs, faces, island, n_samples: int = 128):
    poly2d, _ = island_polygons(uvs, faces, island)
    if poly2d is None or len(poly2d) < 4:
        return None
    seg = np.linalg.norm(np.roll(poly2d, -1, axis=0) - poly2d, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return None
    x_ext = np.concatenate([poly2d[:, 0], poly2d[:1, 0]])
    y_ext = np.concatenate([poly2d[:, 1], poly2d[:1, 1]])
    s_uniform = np.linspace(0, total, n_samples, endpoint=False)
    px = np.interp(s_uniform, s, x_ext)
    py = np.interp(s_uniform, s, y_ext)
    kappa = np.linalg.norm(
        np.stack([np.roll(px, 1) - 2 * px + np.roll(px, -1),
                  np.roll(py, 1) - 2 * py + np.roll(py, -1)], axis=1),
        axis=1,
    )
    return float(np.mean(kappa))


def metrics_for_obj(path: Path) -> dict:
    verts3d, uvs, faces = parse_uv_obj(path)
    ad = area_distortion(verts3d, uvs, faces)
    islands = uv_islands(uvs, faces)
    comps, convs, jags = [], [], []
    for isl in islands:
        c, v = compactness_convexity(uvs, faces, isl)
        if c is not None:
            comps.append(c); convs.append(v)
        j = boundary_jaggedness(uvs, faces, isl)
        if j is not None:
            jags.append(j)
    return {
        "area_distortion": ad,
        "chart_count": len(islands),
        "compactness": float(np.mean(comps)) if comps else float("nan"),
        "convexity": float(np.mean(convs)) if convs else float("nan"),
        "boundary_jaggedness": float(np.mean(jags)) if jags else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv_obj", default="uv.obj")
    args = ap.parse_args()
    m = metrics_for_obj(Path(args.uv_obj))
    print(f"UV metrics for {args.uv_obj}")
    for k, v in m.items():
        print(f"  {k:<22} {v:.4f}")


if __name__ == "__main__":
    main()
