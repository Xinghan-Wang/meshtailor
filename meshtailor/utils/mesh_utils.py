"""Mesh utilities: loading, point-cloud sampling, 1-ring neighbor tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


@dataclass
class MeshData:
    """Lightweight container for the data needed by MeshTailor.

    Attributes
    ----------
    vertices : (N, 3) float
    faces : (F, 3) int
    normals : (N, 3) float, vertex normals (recomputed if missing)
    uv : (F, 3, 2) float, per-corner UV coordinates (None if not present)
    edges : (E, 2) int, sorted unique undirected edges
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv: Optional[np.ndarray]
    edges: np.ndarray

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def load_mesh(path: str) -> MeshData:
    """Load a mesh from disk using trimesh.

    Supports .obj, .glb, .gltf, .ply, .stl.  Preserves per-corner UV if
    available (trimesh stores it as ``visual.uv`` for TextureVisuals, but
    per-corner UV is only directly accessible for certain loaders).
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load a Trimesh from {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if mesh.visual is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        # trimesh TextureVisuals.uv is (V, 2); we need per-corner (F, 3, 2).
        # For .obj files with per-corner UV, trimesh does not expose it directly,
        # so we fall back to the per-vertex UV duplicated to faces.
        uv_per_vertex = np.asarray(mesh.visual.uv, dtype=np.float32)
        uv = uv_per_vertex[faces]  # (F, 3, 2)
    else:
        uv = None

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape[0] != vertices.shape[0]:
        normals = np.zeros_like(vertices, dtype=np.float32)

    edges = mesh.edges_unique  # (E, 2), sorted
    edges = np.asarray(edges, dtype=np.int64)

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uv=uv,
        edges=edges,
    )


def build_one_ringneighbors(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Build a 1-ring neighbor table.

    Returns a list of length ``n_vertices`` where each entry is a sorted
    ``np.ndarray`` of neighbor vertex indices.
    """
    adj = [set() for _ in range(n_vertices)]
    for tri in faces:
        i, j, k = tri
        adj[i].add(j); adj[i].add(k)
        adj[j].add(i); adj[j].add(k)
        adj[k].add(i); adj[k].add(j)
    return [np.array(sorted(s), dtype=np.int64) for s in adj]


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``n_points`` surface points, each storing (x, y, z, nx, ny, nz).

    Uses uniform area-weighted sampling via trimesh for efficiency.
    """
    if rng is None:
        rng = np.random.default_rng()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    pts = np.asarray(pts, dtype=np.float32)

    # Compute normals at sampled points via barycentric interpolation of
    # vertex normals.  For simplicity we use face normals here; if vertex
    # normals are preferred, replace with interpolation.
    fn = mesh.face_normals[face_idx]  # (n_points, 3)
    out = np.concatenate([pts, fn], axis=1)  # (n_points, 6)
    return out


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face triangle areas. Returns (F,) array."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def compute_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex area (1/3 of incident face areas). Returns (N,) array."""
    face_areas = compute_face_areas(vertices, faces)
    vertex_areas = np.zeros(vertices.shape[0], dtype=np.float32)
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], face_areas / 3.0)
    return vertex_areas
