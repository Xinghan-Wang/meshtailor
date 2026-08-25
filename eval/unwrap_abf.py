"""ABF++ unwrap dispatcher: calls the WSL2 abf_unwrap binary (geogram's ABF++,
= MeshTailor paper [35]) to flatten a mesh along pre-marked seams.

Windows eval pipeline calls this; the actual unwrap runs in WSL2 over the
/mnt/<drive>/... view of the same files.

Configured via env vars (with defaults):
  ABF_BIN    - path in WSL to the abf_unwrap binary
  GEO_LIB    - path in WSL to libgeogram.so's directory
  ABF_WSL    - the wsl.exe launcher (default "wsl.exe")
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ABF_BIN = os.environ.get("ABF_BIN", "/root/abf_toolkit/tool/build/abf_unwrap")
GEO_LIB = os.environ.get("GEO_LIB", "/root/abf_toolkit/geogram/build/Release/lib")
ABF_WSL = os.environ.get("ABF_WSL", "wsl.exe")


def _wsl_path(p: str | Path) -> str:
    """D:\\foo\\bar -> /mnt/d/foo/bar"""
    p = str(Path(p)).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def unwrap_abf(mesh_obj, seam_json, uv_obj, timeout: int = 120) -> None:
    """Flatten mesh.obj along seam.json -> uv.obj using ABF++ (WSL subprocess).

    Raises RuntimeError on non-zero exit or timeout.
    """
    cmd_str = (
        f'export LD_LIBRARY_PATH={GEO_LIB}:$LD_LIBRARY_PATH; '
        f'"{ABF_BIN}" "{_wsl_path(mesh_obj)}" "{_wsl_path(seam_json)}" '
        f'"{_wsl_path(uv_obj)}"'
    )
    try:
        r = subprocess.run([ABF_WSL, "--", "bash", "-c", cmd_str],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"abf_unwrap timed out (> {timeout}s)")
    if r.returncode != 0:
        raise RuntimeError(
            f"abf_unwrap rc={r.returncode}: "
            f"{r.stderr.decode(errors='replace')[:400]}"
        )


def unwrap_blender(mesh_obj, seam_json, uv_obj, timeout: int = 120) -> None:
    """Original Blender ANGLE_BASED unwrap (kept for fallback / comparison)."""
    BLENDER = os.environ.get("BLENDER_BIN", r"D:\Blender\blender.exe")
    script = str(Path(__file__).resolve().parent / "unwrap_blender.py")
    subprocess.check_call([BLENDER, "--background", "--python", script,
                           "--", str(mesh_obj), str(seam_json), str(uv_obj)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=timeout)


def unwrap(mesh_obj, seam_json, uv_obj, method: str = "abf", timeout: int = 120) -> None:
    """Dispatch to the requested unwrap method ('abf' or 'blender')."""
    if method == "abf":
        unwrap_abf(mesh_obj, seam_json, uv_obj, timeout=timeout)
    elif method == "blender":
        unwrap_blender(mesh_obj, seam_json, uv_obj, timeout=timeout)
    else:
        raise ValueError(f"unknown unwrap method: {method!r}")
