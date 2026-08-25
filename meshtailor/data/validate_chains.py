"""Validation helpers for seam-edge to chain serialization.

The validator intentionally works on plain arrays/lists so it can be used by
unit tests, preprocessing jobs, and a full ``.pt`` audit without importing the
training stack.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _edge_rows(edges: Any) -> list[tuple[int, int]]:
    """Convert numpy/torch/list edge data to normalized undirected rows."""
    if edges is None:
        return []
    if hasattr(edges, "detach"):
        edges = edges.detach().cpu().numpy()
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        return []
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"edges must have shape (S, 2), got {arr.shape}")
    return [(min(int(a), int(b)), max(int(a), int(b))) for a, b in arr.tolist()]


def chain_edge_rows(chains: list[list[int]]) -> list[tuple[int, int]]:
    """Return every transition in ``chains`` as an undirected edge row."""
    rows: list[tuple[int, int]] = []
    for chain in chains:
        rows.extend(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(chain, chain[1:])
        )
    return rows


def validate_chains(
    seam_edges: Any,
    ordered_chains: list[list[int]],
    mesh_edges: Any | None = None,
    *,
    allow_immediate_backtracking: bool = False,
) -> dict[str, Any]:
    """Validate chain topology and seam-edge coverage.

    Parameters
    ----------
    seam_edges:
        Expected seam-edge rows. Reversed and duplicate rows represent the
        same undirected edge.
    ordered_chains:
        Serialized vertex chains. A closed loop must be represented as
        ``[v0, ..., v0]``.
    mesh_edges:
        Optional complete mesh edge set. If omitted, seam edges are used as
        the admissible edge set. Supplying this catches chains that walk along
        an ordinary mesh edge instead of a seam edge.

    Returns
    -------
    dict
        Integer counters are zero on success. ``valid`` is true only when all
        required counters are zero.
    """
    raw_seams = _edge_rows(seam_edges)
    seam_set = set(raw_seams)
    mesh_set = set(_edge_rows(mesh_edges)) if mesh_edges is not None else seam_set
    chain_rows = chain_edge_rows(ordered_chains)
    chain_set = set(chain_rows)
    chain_counts = Counter(chain_rows)

    invalid_chain_steps = sum(edge not in mesh_set for edge in chain_rows)
    coverage_mismatch = len(seam_set.symmetric_difference(chain_set))
    duplicate_chain_edges = sum(n - 1 for n in chain_counts.values() if n > 1)
    immediate_backtracks = 0
    invalid_loop_closures = 0

    for chain in ordered_chains:
        values = [int(v) for v in chain]
        for a, _b, c in zip(values, values[1:], values[2:]):
            if a == c:
                immediate_backtracks += 1

        is_loop = len(values) >= 2 and values[0] == values[-1]
        if not is_loop:
            continue

        body = values[:-1]
        # A loop has one explicit closing transition, and no vertex may occur
        # twice in its body.  This also rejects A-B-A masquerading as a loop.
        if len(body) < 3 or len(set(body)) != len(body):
            invalid_loop_closures += 1

    return {
        "valid": (
            invalid_chain_steps == 0
            and coverage_mismatch == 0
            and duplicate_chain_edges == 0
            and (allow_immediate_backtracking or immediate_backtracks == 0)
            and invalid_loop_closures == 0
        ),
        "seam_edge_count": len(seam_set),
        "input_seam_edge_rows": len(raw_seams),
        "duplicate_input_edges": len(raw_seams) - len(seam_set),
        "chain_count": len(ordered_chains),
        "chain_edge_count": len(chain_rows),
        "invalid_chain_steps": int(invalid_chain_steps),
        "coverage_mismatch": int(coverage_mismatch),
        "duplicate_chain_edges": int(duplicate_chain_edges),
        "immediate_backtracks": int(immediate_backtracks),
        "invalid_loop_closures": int(invalid_loop_closures),
    }


def validate_sample(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a serialized MeshTailor sample dictionary."""
    if "ordered_chains" not in data:
        raise KeyError("sample has no ordered_chains field")
    return validate_chains(
        data.get("seam_edges"),
        data["ordered_chains"],
        data.get("edges"),
    )
