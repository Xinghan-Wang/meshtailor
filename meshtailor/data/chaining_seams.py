"""
ChainingSeams serialization (Algorithm 1).

Orders seam chains from global structural cuts down to local details
in a coarse-to-fine manner.

Ordering principles (by priority):
1. loops first: closed-loop cuts before open chains
2. largest patch first: target the largest current surface patch
3. area balance: split the patch into sub-patches of roughly equal area
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def chaining_seams(
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_chains: list[list[int]],
) -> list[list[int]]:
    """
    ChainingSeams serialization (Algorithm 1).

    Parameters
    ----------
    vertices : (N, 3) float
    faces : (F, 3) int
    seam_chains : list of list of int
        Seam chains on the mesh.

    Returns
    -------
    ordered_chains : list of list of int
        Ordered chain list according to ChainingSeams.
    """
    # Step 1: Separate closed loops from open chains
    loops = []
    open_chains = []
    for chain in seam_chains:
        if len(chain) >= 3 and chain[0] == chain[-1]:
            loops.append(chain)
        else:
            open_chains.append(chain)

    if not loops:
        # No loops to order; just sort open chains
        open_chains.sort(key=lambda c: -len(c))
        return open_chains

    # Step 2: Precompute auxiliary data
    face_areas = _compute_face_areas(vertices, faces)
    edge_to_faces = _build_edge_to_faces(faces)

    # Step 3: Initialize patches (whole mesh as one patch)
    patches: list[set[int]] = [set(range(len(faces)))]
    ordered_loops: list[list[int]] = []
    remaining_loops = list(loops)

    # Step 4: Main loop
    while patches and remaining_loops:
        # Find largest patch
        largest_idx = max(
            range(len(patches)),
            key=lambda i: sum(face_areas[f] for f in patches[i]),
        )
        largest_patch = patches[largest_idx]

        # Get patch vertices
        patch_vertices: set[int] = set()
        for fi in largest_patch:
            for v in faces[fi]:
                patch_vertices.add(int(v))

        # Find internal loops in largest patch
        internal_loops = [
            loop for loop in remaining_loops
            if all(int(v) in patch_vertices for v in loop)
        ]

        if not internal_loops:
            # No more loops to cut in this patch
            patches.pop(largest_idx)
            continue

        # Find loop with best area balance
        best_loop = None
        best_score = -1.0
        for loop in internal_loops:
            score = _area_balance_score(
                largest_patch, loop, edge_to_faces, face_areas, faces
            )
            if score > best_score:
                best_score = score
                best_loop = loop

        if best_loop is None or best_score <= 0:
            # Cannot split any loop; remove patch
            patches.pop(largest_idx)
            continue

        # Cut patch along best loop
        patch1, patch2 = _split_patch(
            largest_patch, best_loop, edge_to_faces, faces
        )

        # Update state
        ordered_loops.append(best_loop)
        remaining_loops.remove(best_loop)
        patches.pop(largest_idx)
        if patch1:
            patches.append(patch1)
        if patch2:
            patches.append(patch2)

    # Step 5: Append any remaining loops that couldn't be placed
    ordered_loops.extend(remaining_loops)

    # Step 6: Sort open chains by decreasing length
    open_chains.sort(key=lambda c: -len(c))

    # Step 7: Concatenate
    return ordered_loops + open_chains


def _compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _build_edge_to_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for j in range(3):
            v0, v1 = int(tri[j]), int(tri[(j + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            edge_to_faces[key].append(fi)
    return dict(edge_to_faces)


def _split_patch(
    patch_faces: set[int],
    loop: list[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    faces: np.ndarray,
) -> tuple[set[int], set[int]]:
    """
    Split a patch along a loop by removing loop edges from the
    face adjacency graph and finding connected components.

    Returns (face_set_1, face_set_2). Empty sets if split fails.
    """
    # Build loop edges
    loop_edges: set[tuple[int, int]] = set()
    for i in range(len(loop) - 1):
        u, v = int(loop[i]), int(loop[i + 1])
        loop_edges.add((min(u, v), max(u, v)))
    # If loop is closed (first == last), the range already covers all edges

    # Build face adjacency graph within patch
    adj: dict[int, set[int]] = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        patch_faces_in_list = [f for f in face_list if f in patch_faces]
        if len(patch_faces_in_list) >= 2:
            for i in range(len(patch_faces_in_list)):
                for j in range(i + 1, len(patch_faces_in_list)):
                    adj[patch_faces_in_list[i]].add(patch_faces_in_list[j])
                    adj[patch_faces_in_list[j]].add(patch_faces_in_list[i])

    # Remove edges that are part of the loop
    for edge in loop_edges:
        if edge in edge_to_faces:
            face_list = edge_to_faces[edge]
            patch_faces_in_list = [f for f in face_list if f in patch_faces]
            if len(patch_faces_in_list) >= 2:
                for i in range(len(patch_faces_in_list)):
                    for j in range(i + 1, len(patch_faces_in_list)):
                        adj[patch_faces_in_list[i]].discard(patch_faces_in_list[j])
                        adj[patch_faces_in_list[j]].discard(patch_faces_in_list[i])

    # Find connected components using BFS
    visited: set[int] = set()
    components: list[set[int]] = []

    for face in patch_faces:
        if face not in visited:
            component: set[int] = set()
            queue: deque[int] = deque([face])
            while queue:
                f = queue.popleft()
                if f in visited:
                    continue
                visited.add(f)
                component.add(f)
                for neighbor in adj[f]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

    # Need at least 2 components to split
    if len(components) < 2:
        return set(), set()

    # Return the two largest components
    components.sort(key=len, reverse=True)
    return components[0], components[1]


def _area_balance_score(
    patch_faces: set[int],
    loop: list[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_areas: np.ndarray,
    faces: np.ndarray,
) -> float:
    """
    Compute area-balance score for cutting patch along loop.

    r(L; P) = min(A_L^(1), A_L^(2)) / max(A_L^(1), A_L^(2)) in (0, 1]
    """
    patch1, patch2 = _split_patch(patch_faces, loop, edge_to_faces, faces)

    if not patch1 or not patch2:
        return 0.0

    area1 = sum(face_areas[f] for f in patch1)
    area2 = sum(face_areas[f] for f in patch2)

    if max(area1, area2) < 1e-10:
        return 0.0

    return float(min(area1, area2) / max(area1, area2))
