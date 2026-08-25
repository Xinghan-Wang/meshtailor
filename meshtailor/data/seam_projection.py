"""Re-project seam chains from original mesh onto remeshed mesh."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial import cKDTree


def project_seam_chains(
    orig_chains: list[list[int]],
    orig_vertices: np.ndarray,
    remeshed_vertices: np.ndarray,
    remeshed_edges: np.ndarray,
) -> list[list[int]]:
    """
    Project original seam chains onto the remeshed mesh.

    For each original seam chain vertex, find the closest remeshed vertex.
    For consecutive projected vertices, walk along remeshed edges (BFS
    shortest path) so that the resulting chain is edge-aligned on the
    remeshed mesh.

    Parameters
    ----------
    orig_chains : list of list of int
        Original seam chains (vertex indices into ``orig_vertices``).
    orig_vertices : (N_orig, 3) float
        Original mesh vertex positions.
    remeshed_vertices : (N_new, 3) float
        Remeshed mesh vertex positions.
    remeshed_edges : (E_new, 2) int
        Unique undirected edges of the remeshed mesh.

    Returns
    -------
    remeshed_chains : list of list of int
        Seam chains on the remeshed mesh (vertex indices into
        ``remeshed_vertices``).
    """
    # Build KDTree on remeshed vertices
    tree = cKDTree(remeshed_vertices)

    # Build adjacency for remeshed mesh (from edges)
    n_remeshed = len(remeshed_vertices)
    adj: list[set[int]] = [set() for _ in range(n_remeshed)]
    for e in remeshed_edges:
        i, j = int(e[0]), int(e[1])
        adj[i].add(j)
        adj[j].add(i)
    adj_frozen = [frozenset(s) for s in adj]

    # BFS shortest path between two vertices
    def bfs_shortest_path(start: int, end: int) -> list[int]:
        if start == end:
            return [start]
        visited = {start}
        queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            for neighbor in adj_frozen[current]:
                if neighbor == end:
                    return path + [end]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return [start, end]  # fallback: direct jump

    remeshed_chains: list[list[int]] = []

    for orig_chain in orig_chains:
        if len(orig_chain) < 2:
            continue

        # Project each original vertex to closest remeshed vertex
        orig_pos = orig_vertices[orig_chain]  # (T, 3)
        _, projected_idx = tree.query(orig_pos, k=1)  # (T,)
        projected_idx = np.atleast_1d(projected_idx).astype(np.int64)

        # Build remeshed chain by walking between projected vertices
        remeshed_chain: list[int] = [int(projected_idx[0])]

        for t in range(1, len(projected_idx)):
            u = int(projected_idx[t - 1])
            v = int(projected_idx[t])

            if v == remeshed_chain[-1]:
                continue  # skip duplicate

            if v in adj_frozen[u]:
                # Direct 1-ring neighbor
                remeshed_chain.append(v)
            else:
                # Need shortest path
                path = bfs_shortest_path(u, v)
                # Skip first vertex (already in chain)
                remeshed_chain.extend(path[1:])

        if len(remeshed_chain) >= 2:
            remeshed_chains.append(remeshed_chain)

    return remeshed_chains


def chains_to_edges(chains: list[list[int]]) -> np.ndarray:
    """Convert vertex-walk chains to unique undirected edge pairs."""
    edges: set[tuple[int, int]] = set()
    for chain in chains:
        for i in range(len(chain) - 1):
            v0, v1 = int(chain[i]), int(chain[i + 1])
            edges.add((min(v0, v1), max(v0, v1)))
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)
