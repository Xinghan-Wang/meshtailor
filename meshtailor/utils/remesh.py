"""PyMeshLab wrapper: merge duplicate vertices, repair topology, decimate.

Pipeline:
  1. Remove duplicate vertices (merge seam duplicates)
  2. Merge close vertices (within 0.01% of bbox diagonal)
  3. Repair non-manifold edges (split vertices)
  4. Repair non-manifold vertices
  5. Remove unreferenced vertices / null faces / duplicate faces
  6. Decimate to <= target_faces (quadric edge collapse)
  7. Final cleanup
"""
from __future__ import annotations

import numpy as np
import pymeshlab


def remesh_garment(
    input_ply: str,
    target_faces: int = 2000,
) -> dict:
    """
    Remesh a garment mesh by merging duplicate vertices, repairing
    topology, and decimating to <= target_faces.

    Steps:
        1. Remove duplicate vertices (merge seam duplicates)
        2. Merge close vertices (within 0.01% of bbox diagonal)
        3. Repair non-manifold edges (split vertices)
        4. Repair non-manifold vertices
        5. Remove unreferenced vertices
        6. Remove null faces
        7. Remove duplicate faces
        8. Final cleanup

    Returns
    -------
    dict with:
        - vertices: (N, 3) float
        - faces: (F, 3) int
        - n_vertices: int
        - n_faces: int
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply)
    # Step 1: merge duplicate vertices (seam dups)
    ms.meshing_remove_duplicate_vertices()
    # Step 2: merge near-duplicate vertices
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.01))
    # Step 3: repair non-manifold edges (split vertices)
    ms.meshing_repair_non_manifold_edges(method="Split Vertices")
    # Step 4: repair non-manifold vertices
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    # Step 5: remove unreferenced vertices
    ms.meshing_remove_unreferenced_vertices()
    # Step 6: remove null faces
    ms.meshing_remove_null_faces()
    # Step 7: remove duplicate faces
    ms.meshing_remove_duplicate_faces()
    # Step 8: Decimate to <= target_faces (quadric edge collapse)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preservetopology=True,
        preserveboundary=True,
        optimalplacement=True,
        preservenormal=True,
        planarquadric=True,
        autoclean=True,
    )
    # Step 9: final cleanup
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_null_faces()

    # Extract vertices and faces directly from memory
    m = ms.current_mesh()
    v_matrix = np.asarray(m.vertex_matrix(), dtype=np.float32)
    f_matrix = np.asarray(m.face_matrix(), dtype=np.int64)

    return {
        "vertices": v_matrix,
        "faces": f_matrix,
        "n_vertices": len(v_matrix),
        "n_faces": len(f_matrix),
    }


def remesh_garment_preserve_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    target_faces: int = 2000,
    extratcoordw: float = 10.0,
) -> dict:
    """Preserve-UV remesh via texture-aware quadric decimation.

    Decimates a mesh whose vertices carry per-vertex UV (s, t) and duplicate
    vertices at seam lines, in the 5D (x, y, z, u, v) space with
    ``preserveboundary=True`` so that seam edges (mesh boundaries due to the
    duplicate vertices) are kept intact. Duplicate seam vertices are PRESERVED
    in the output; merging them into a ``merged`` mesh is the caller's job.

    Note: ``extratcoordw`` has no observable effect here; seam preservation is
    driven entirely by ``preserveboundary``. It is kept for API symmetry. The
    final face count may exceed ``target_faces`` for seam-heavy meshes (the
    non-seam region is already fully decimated). Because duplicate seam
    vertices are kept at their exact positions, projecting seam chains from
    the original mesh onto this decimated mesh is essentially zero-error.

    Parameters
    ----------
    vertices : (N, 3) float
    faces : (F, 3) int
    uv : (N, 2) float   per-vertex UV
    target_faces : int  target face count (may be exceeded)
    extratcoordw : float  UV weight (kept for API symmetry)

    Returns
    -------
    dict with:
        - vertices: (N', 3) float64   (duplicate seam verts preserved)
        - faces:    (F', 3) int64
        - n_vertices: int
        - n_faces:    int
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float64)

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=vertices,
        face_matrix=faces,
        v_tex_coords_matrix=uv,
    )
    ms.add_mesh(pml_mesh, "garment")

    ms.compute_texcoord_transfer_vertex_to_wedge()

    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target_faces,
        extratcoordw=extratcoordw,
        preserveboundary=True,
        optimalplacement=True,
    )

    m = ms.current_mesh()
    v_matrix = np.asarray(m.vertex_matrix(), dtype=np.float64)
    f_matrix = np.asarray(m.face_matrix(), dtype=np.int64)

    return {
        "vertices": v_matrix,
        "faces": f_matrix,
        "n_vertices": len(v_matrix),
        "n_faces": len(f_matrix),
    }
