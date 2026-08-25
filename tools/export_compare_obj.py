"""Export comparison OBJs: mesh (gray) + three-color seam tubes.

Green = GT∩pred (hit), blue = GT only (missed), red = pred only (spurious).
Points at the full 10k test outputs.

Output: viz_compare/<gid>.obj + viz_compare/compare.mtl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "gt_outputs_full"  # relabeling preserved the GT seam edge sets
P0_DIR = ROOT / "test_outputs_full"
OUT = ROOT / "viz_compare"
GIDS = [
    "rand_001CMHYSGE",  # recall 1.000 (perfect)
    "rand_6X7YOMWIIJ",  # recall 0.900 (p90 band)
    "rand_A8R81NEQXN",  # recall 0.500 (mid)
    "rand_4TWRJW8XYM",  # recall 0.100 (p10 band)
    "rand_013SQS1WZS",  # recall 0.961, dense (560 GT edges)
]
TUBE_R_FRAC = 0.006  # tube radius as fraction of bbox diagonal

MTL = """newmtl meshmat
Kd 0.82 0.82 0.86
Ka 0.1 0.1 0.1
newmtl hitgreen
Kd 0.05 0.85 0.15
Ka 0.05 0.3 0.1
newmtl gtblue
Kd 0.05 0.25 1.00
Ka 0.05 0.1 0.3
newmtl predred
Kd 1.00 0.08 0.05
Ka 0.3 0.05 0.05
"""


def seam_set(path: Path) -> set:
    return {tuple(e) for e in json.loads(path.read_text())}


def tube_stream(p0, p1, r, n1, n2):
    """8 verts / 4 quads (open tube) for one edge; yields (verts, faces-local)."""
    a = p0 + r * n1
    b = p0 + r * n2
    c = p0 - r * n1
    d = p0 - r * n2
    e = p1 + r * n1
    f = p1 + r * n2
    g = p1 - r * n1
    h = p1 - r * n2
    v = [a, b, c, d, e, f, g, h]
    q = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, q


def frame(d):
    d = d / (np.linalg.norm(d) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n1 = np.cross(d, ref)
    n1 /= np.linalg.norm(n1) + 1e-12
    n2 = np.cross(d, n1)
    return n1, n2


def export(gid: str) -> dict:
    gt_seam = seam_set(GT_DIR / gid / "seam.json")
    p0_seam = seam_set(P0_DIR / gid / "seam.json")

    # mesh vertices/faces from the inference export (same indexing as seams)
    verts, faces = [], []
    for line in open(P0_DIR / gid / "mesh.obj", encoding="utf-8"):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in t[1:]])
    V = np.asarray(verts)

    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    r = TUBE_R_FRAC * diag

    lines = ["mtllib compare.mtl"]
    for p in V:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    lines.append("usemtl meshmat")
    for f in faces:
        lines.append("f " + " ".join(str(i + 1) for i in f))

    state = {"next": len(V)}  # running OBJ vertex count (1-based index = next+1)

    def add_tubes(name_mat, edges):
        lines.append(f"usemtl {name_mat}")
        for (a, b) in sorted(edges):
            p0, p1 = V[a], V[b]
            n1, n2 = frame(p1 - p0)
            tv, quads = tube_stream(p0, p1, r, n1, n2)
            for p in tv:
                lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
            for q in quads:
                lines.append(
                    "f " + " ".join(str(state["next"] + i + 1) for i in q))
            state["next"] += 8

    add_tubes("hitgreen", gt_seam & p0_seam)   # overlap = correct
    add_tubes("gtblue", gt_seam - p0_seam)     # GT only = missed
    add_tubes("predred", p0_seam - gt_seam)    # pred only = spurious

    out = OUT / f"{gid}.obj"
    out.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")

    inter = len(gt_seam & p0_seam)
    return dict(gid=gid, gt=len(gt_seam), p0=len(p0_seam), recall=inter / max(len(gt_seam), 1),
                prec=inter / max(len(p0_seam), 1))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "compare.mtl").write_text(MTL)
    for gid in GIDS:
        try:
            info = export(gid)
            print(f"{gid}: GT {info['gt']} edges, predicted {info['p0']} edges, "
                  f"recall {info['recall']:.0%}, precision {info['prec']:.0%} "
                  f"-> {OUT / (gid + '.obj')}")
        except Exception as e:
            print(f"{gid}: FAILED {e!r}")


if __name__ == "__main__":
    main()
