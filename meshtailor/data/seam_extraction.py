"""
Extract seam edges from a GarmentCodeData garment.

Method:
1. Load sim.ply with per-vertex UV (s, t)
2. Find duplicate vertices (same 3D position, different UV)
3. Seam edges are mesh edges where the two incident faces use
   different UV coordinates at the shared 3D positions.
4. Equivalently: a 3D edge that appears as multiple mesh edges
   (one from each panel) is a seam edge.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


def load_mesh_with_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a PLY mesh and return vertices, faces, and per-vertex UV.

    Returns
    -------
    vertices : (N, 3) float
    faces : (F, 3) int
    uv : (N, 2) float
    """
    mesh = trimesh.load(path, force="mesh", process=False)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Extract UV from PLY metadata
    raw = mesh.metadata.get("_ply_raw", {})
    vertex_data = raw.get("vertex", {}).get("data", None)

    if vertex_data is not None:
        s = vertex_data["s"]
        t = vertex_data["t"]
        uv = np.stack([s, t], axis=-1).astype(np.float32)
    elif hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    else:
        raise ValueError("Mesh has no UV coordinates")

    return vertices, faces, uv


def extract_seam_edges_from_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """Extract seam edges using UV coordinates.

    A seam edge is a 3D edge that appears as multiple mesh edges
    (because the vertices are duplicated: same 3D position, different UV).

    Parameters
    ----------
    vertices : (N, 3) float
    faces : (F, 3) int
    uv : (N, 2) float

    Returns
    -------
    seam_edges : (S, 2) int
        Seam edges as vertex index pairs (using the original vertex indices).
    """
    # Round positions to avoid floating point issues
    pos_rounded = np.round(vertices, decimals=5)

    # Build edge -> faces mapping
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(len(faces)):
        for j in range(3):
            v0 = int(faces[fi, j])
            v1 = int(faces[fi, (j + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            edge_to_faces.setdefault(key, []).append(fi)

    # Group mesh edges by 3D position
    edge_3d_positions: dict[frozenset, list[tuple[int, int]]] = {}
    for edge in edge_to_faces.keys():
        v0, v1 = edge
        pos0 = pos_rounded[v0].tobytes()
        pos1 = pos_rounded[v1].tobytes()
        key = frozenset([pos0, pos1])
        edge_3d_positions.setdefault(key, []).append(edge)

    # Seam edges: 3D edges with multiple mesh edges
    seam_edges = []
    for key, mesh_edges in edge_3d_positions.items():
        if len(mesh_edges) > 1:
            # This is a seam edge — add all mesh edges
            for edge in mesh_edges:
                seam_edges.append(edge)

    if len(seam_edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    return np.array(seam_edges, dtype=np.int64)


def build_seam_chains(
    seam_edges: np.ndarray,
) -> list[list[int]]:
    """Decompose an undirected seam-edge graph into deterministic chains.

    Every *unique* undirected edge is consumed at most once.  Paths are
    started at vertices whose degree is not two; after all such maximal open
    paths have been consumed, each remaining connected component is a pure
    degree-two component and is emitted as one closed loop.

    The returned representation uses ``[v0, ..., v0]`` for a loop.  Duplicate
    input edges are treated as one graph edge: the seam edge set, rather than
    the input row multiplicity, defines the decomposition.
    """
    if seam_edges is None:
        return []

    raw = np.asarray(seam_edges, dtype=np.int64)
    if raw.size == 0:
        return []
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"seam_edges must have shape (S, 2), got {raw.shape}")

    # Normalise once up front.  In particular, this makes duplicate rows and
    # reversed rows share the same used_edges key.
    edge_set: set[tuple[int, int]] = set()
    for a, b in raw.tolist():
        u, v = int(a), int(b)
        if u == v:
            raise ValueError(f"self-loop seam edge is not a mesh edge: {(u, v)}")
        edge_set.add((min(u, v), max(u, v)))
    if not edge_set:
        return []

    adj: dict[int, set[int]] = {}
    for u, v in sorted(edge_set):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    used_edges: set[tuple[int, int]] = set()

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (min(u, v), max(u, v))

    def is_unused(u: int, v: int) -> bool:
        return edge_key(u, v) not in used_edges

    def consume(u: int, v: int) -> None:
        key = edge_key(u, v)
        if key in used_edges:
            raise RuntimeError(f"seam edge consumed twice: {key}")
        used_edges.add(key)

    def canonical_open(path: list[int]) -> list[int]:
        # With vertex IDs as the only available identity, lexicographic
        # min(path, reverse(path)) is the stable canonical orientation.
        reverse = path[::-1]
        return path if tuple(path) <= tuple(reverse) else reverse

    def canonical_loop(path: list[int]) -> list[int]:
        # path already contains the start vertex once at both ends.
        body = path[:-1]
        if not body:
            return path
        candidates: list[tuple[int, ...]] = []
        for seq in (body, body[::-1]):
            for i in range(len(seq)):
                rotated = seq[i:] + seq[:i]
                candidates.append(tuple(rotated))
        best = list(min(candidates))
        return best + [best[0]]

    def trace_open(start: int, first: int) -> list[int]:
        path = [start]
        current = start
        next_vertex = first
        while True:
            consume(current, next_vertex)
            current, next_vertex = next_vertex, None
            path.append(current)

            # A maximal open path ends at a non-degree-two vertex.  At a
            # degree-two vertex there is exactly one possible continuation,
            # unless that edge was already consumed by a path from a branch.
            if len(adj[current]) != 2:
                break
            candidates = [n for n in sorted(adj[current]) if is_unused(current, n)]
            if not candidates:
                break
            next_vertex = candidates[0]
        return canonical_open(path)

    open_chains: list[list[int]] = []
    for start in sorted(adj):
        if len(adj[start]) == 2:
            continue
        for first in sorted(adj[start]):
            if is_unused(start, first):
                open_chains.append(trace_open(start, first))

    loop_chains: list[list[int]] = []
    while len(used_edges) < len(edge_set):
        remaining_vertices = sorted(
            v for v, neighbors in adj.items()
            if any(is_unused(v, n) for n in neighbors)
        )
        if not remaining_vertices:
            break
        start = remaining_vertices[0]
        first = next(n for n in sorted(adj[start]) if is_unused(start, n))
        path = [start]
        current = start
        while True:
            candidates = [n for n in sorted(adj[current]) if is_unused(current, n)]
            if not candidates:
                raise RuntimeError(
                    f"unclosed seam component while tracing from vertex {start}"
                )
            nxt = candidates[0]
            consume(current, nxt)
            path.append(nxt)
            current = nxt
            if current == start:
                break
        loop_chains.append(canonical_loop(path))

    # Sorting makes chain order independent of dictionary/set insertion order.
    open_chains.sort(key=lambda c: tuple(c))
    loop_chains.sort(key=lambda c: tuple(c))
    return open_chains + loop_chains


def _min_rotation_index(seq: list[int]) -> int:
    """Start index of the lexicographically minimal rotation of ``seq`` (O(L))."""
    n = len(seq)
    if n <= 1:
        return 0
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        a = seq[(i + k) % n]
        b = seq[(j + k) % n]
        if a == b:
            k += 1
        elif a > b:
            i += k + 1
            if i == j:
                i += 1
            k = 0
        else:
            j += k + 1
            if i == j:
                j += 1
            k = 0
    return min(i, j)


def build_seam_chains_maximal(
    seam_edges: np.ndarray,
    vertices: np.ndarray,
) -> list[list[int]]:
    """Decompose the seam graph into maximal chains that cross junctions.

    Paper B.1: "tracing maximal edge-connected paths on the seam subgraph,
    yielding both open chains and closed loop cuts".  At a junction
    (degree != 2) the incident edges are paired by tangent continuity:
    the most anti-parallel outgoing direction pairs are taken greedily
    with a deterministic lexicographic tie-break, odd degree leaves one
    dangling edge.  Walks follow pairings through junctions; a walk that
    returns to its start closes into a loop ``[v0, ..., v0]``.  A closed
    walk revisiting a junction (figure-eight under crossed pairing) is
    split into simple cycles.  Every seam edge is consumed exactly once.
    """
    if seam_edges is None:
        return []

    raw = np.asarray(seam_edges, dtype=np.int64)
    if raw.size == 0:
        return []
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"seam_edges must have shape (S, 2), got {raw.shape}")

    edge_set: set[tuple[int, int]] = set()
    for a, b in raw.tolist():
        u, v = int(a), int(b)
        if u == v:
            raise ValueError(f"self-loop seam edge is not a mesh edge: {(u, v)}")
        edge_set.add((min(u, v), max(u, v)))
    if not edge_set:
        return []

    pos = np.asarray(vertices, dtype=np.float64)

    adj: dict[int, set[int]] = {}
    for u, v in sorted(edge_set):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (min(u, v), max(u, v))

    pair_partner: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
    for v in sorted(adj):
        nbrs = sorted(adj[v])
        if len(nbrs) < 3:
            continue
        dirs: dict[int, np.ndarray] = {}
        for u in nbrs:
            vec = pos[u] - pos[v]
            norm = float(np.linalg.norm(vec))
            dirs[u] = vec / norm if norm > 1e-12 else np.zeros(3)
        candidates: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                score = -float(np.dot(dirs[nbrs[i]], dirs[nbrs[j]]))
                candidates.append((-score, edge_key(v, nbrs[i]), edge_key(v, nbrs[j])))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        matched: set[tuple[int, int]] = set()
        for _neg_score, ki, kj in candidates:
            if ki in matched or kj in matched:
                continue
            matched.add(ki)
            matched.add(kj)
            pair_partner[(v, ki)] = kj
            pair_partner[(v, kj)] = ki

    used_edges: set[tuple[int, int]] = set()

    def is_unused(u: int, v: int) -> bool:
        return edge_key(u, v) not in used_edges

    def consume(u: int, v: int) -> None:
        key = edge_key(u, v)
        if key in used_edges:
            raise RuntimeError(f"seam edge consumed twice: {key}")
        used_edges.add(key)

    def next_vertex(cur: int, prev: int) -> int | None:
        deg = len(adj[cur])
        if deg == 1:
            return None
        if deg == 2:
            for n in sorted(adj[cur]):
                if n != prev:
                    return n if is_unused(cur, n) else None
            return None
        partner = pair_partner.get((cur, edge_key(prev, cur)))
        if partner is None or partner in used_edges:
            return None
        return partner[0] if partner[1] == cur else partner[1]

    def trace(start: int, first: int) -> tuple[list[int], bool]:
        path = [start]
        cur = start
        nxt: int | None = first
        while True:
            consume(cur, nxt)
            prev, cur = cur, nxt
            path.append(cur)
            nxt = next_vertex(cur, prev)
            if nxt is None:
                break
        return path, cur == start

    def canonical_open(path: list[int]) -> list[int]:
        reverse = path[::-1]
        return path if tuple(path) <= tuple(reverse) else reverse

    def canonical_loop(path: list[int]) -> list[int]:
        body = path[:-1]
        k1 = _min_rotation_index(body)
        rot1 = body[k1:] + body[:k1]
        rev = body[::-1]
        k2 = _min_rotation_index(rev)
        rot2 = rev[k2:] + rev[:k2]
        best = rot1 if tuple(rot1) <= tuple(rot2) else rot2
        return best + [best[0]]

    open_chains: list[list[int]] = []
    loops: list[list[int]] = []

    def split_closed(body: list[int]) -> None:
        first_seen: dict[int, int] = {}
        for i, x in enumerate(body):
            j = first_seen.get(x)
            if j is None:
                first_seen[x] = i
                continue
            cycle = body[j : i + 1]
            if i - j < 3:
                raise RuntimeError(f"degenerate cycle in closed walk: {cycle}")
            loops.append(canonical_loop(cycle))
            remainder = [x] + body[i + 1 :] + body[:j]
            if len(remainder) >= 3:
                split_closed(remainder)
            elif len(remainder) == 2:
                raise RuntimeError(f"degenerate 2-cycle remainder: {remainder}")
            return
        loops.append(canonical_loop(body + [body[0]]))

    def emit_closed(path: list[int]) -> None:
        body = path[:-1]
        if len(body) >= 3 and len(set(body)) == len(body):
            loops.append(canonical_loop(path))
        else:
            split_closed(list(body))

    # Phase A: open chains start only at terminal edges (edges without a
    # partner at a non-degree-two vertex: degree-1 endpoints and dangling
    # junction edges).
    for v in sorted(adj):
        if len(adj[v]) == 2:
            continue
        for n in sorted(adj[v]):
            key = edge_key(v, n)
            if key in used_edges or (v, key) in pair_partner:
                continue
            path, closed = trace(v, n)
            if closed:
                emit_closed(path)
            else:
                open_chains.append(canonical_open(path))

    # Phase B: remaining edges belong to closed walks.
    while len(used_edges) < len(edge_set):
        v = min(u for u in adj if any(is_unused(u, w) for w in adj[u]))
        n = min(w for w in adj[v] if is_unused(v, w))
        path, closed = trace(v, n)
        if not closed:
            raise RuntimeError(f"expected a closed walk from vertex {v}")
        emit_closed(path)

    assert len(used_edges) == len(edge_set)

    open_chains.sort(key=lambda c: tuple(c))
    loops.sort(key=lambda c: tuple(c))
    return open_chains + loops


def extract_seams_from_garment(
    garment_dir: Path,
    gid: str,
) -> dict:
    """Extract seam information from a garment directory.

    Returns a dict with:
        - vertices: (N, 3) float
        - faces: (F, 3) int
        - uv: (N, 2) float
        - seam_edges: (S, 2) int
        - seam_chains: list of list of int
    """
    sim_path = garment_dir / f"{gid}_sim.ply"
    vertices, faces, uv = load_mesh_with_uv(sim_path)

    seam_edges = extract_seam_edges_from_uv(vertices, faces, uv)
    seam_chains = build_seam_chains(seam_edges)

    return {
        "vertices": vertices,
        "faces": faces,
        "uv": uv,
        "seam_edges": seam_edges,
        "seam_chains": seam_chains,
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "n_seam_edges": len(seam_edges),
        "n_chains": len(seam_chains),
    }


if __name__ == "__main__":
    # Test on the first garment under a given root directory
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    garment_dir = next(root.iterdir())
    gid = garment_dir.name

    result = extract_seams_from_garment(garment_dir, gid)

    print(f"Garment: {gid}")
    print(f"  Vertices:    {result['n_vertices']}")
    print(f"  Faces:       {result['n_faces']}")
    print(f"  Seam edges:  {result['n_seam_edges']}")
    print(f"  Chains:      {result['n_chains']}")

    print(f"\nChain lengths:")
    for i, chain in enumerate(result["seam_chains"]):
        is_loop = chain[0] == chain[-1] if len(chain) >= 2 else False
        print(f"  Chain {i}: length={len(chain)}, loop={is_loop}")
