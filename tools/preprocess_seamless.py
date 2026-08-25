"""Seamless (SeamAwareDecimater = MeshTailor paper ref [21]) preprocessing for GCD.

Replaces preprocess_100k.py's per-panel isotropic remesh with SEAM-PRESERVING
DECIMATION, producing the SAME 16-field .pt schema so downstream train/eval are
unchanged. Per garment:
  sim.ply -> weld 3D (unique V) + per-corner UV (TC,F,FT) -> .obj ->
  WSL `decimater` num-vertices N --strict 2 (preserves UV boundary = seams) ->
  decimated .obj -> extract seam edges (UV discontinuity) -> chains ->
  normalize + aux (normals, edges, 1-ring, surface pts, ChainingSeams) -> .pt

Batched for throughput: stage BATCH objs in a temp dir, ONE WSL session runs
`xargs -P PAR` to decimate them in parallel, then read back. Resumable (skips
gids whose .pt exists). ~3% of meshes crash the decimater (unordered_map::at);
those are retried at strict 1 / percent-vertices, then recorded as failed.
"""
from __future__ import annotations
import io, os, sys, re, json, tarfile, subprocess, time, argparse, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from meshtailor.data.seam_extraction import build_seam_chains
from meshtailor.data.chaining_seams import chaining_seams
from meshtailor.data.validate_chains import validate_chains
from meshtailor.utils.mesh_utils import build_one_ringneighbors, sample_surface_points

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DECIMATER = os.environ.get("SEAMLESS_DECIMATER", "/root/seam-decimater/build/decimater")
# keep the batch dir on a local (drvfs) disk; WSL UNC paths time out under heavy I/O
TMP = Path(os.environ.get("SEAMLESS_TMP", r"C:\Temp\seamless_batch"))


# ---------- path helpers ----------
def win_to_wsl(p) -> str:
    s = str(p).replace("\\", "/")
    low = s.lower()
    if low.startswith("//wsl.localhost/ubuntu"):
        return s[len("//wsl.localhost/ubuntu"):]          # -> /tmp/seamless_batch/...
    if low.startswith("//wsl$/ubuntu"):
        return s[len("//wsl$/ubuntu"):]
    if len(s) > 1 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def win_from_wsl(wsl_path: str) -> Path:
    m = re.match(r"/mnt/([a-z])/(.*)", wsl_path)
    if not m:
        return Path(wsl_path)
    return Path(f"{m.group(1).upper()}:\\") / m.group(2).replace("/", "\\")


# ---------- sim.ply / obj I/O (mirror preprocess_100k.load_sim) ----------
def load_sim(sim_bytes: bytes):
    mesh = trimesh.load(io.BytesIO(sim_bytes), file_type="ply", force="mesh", process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    raw = mesh.metadata.get("_ply_raw", {})
    vd = raw.get("vertex", {}).get("data", None)
    if vd is None:
        raise ValueError("sim.ply has no per-vertex UV (s, t)")
    uv = np.stack([vd["s"], vd["t"]], axis=-1).astype(np.float64)
    return V, F, uv


def to_obj_mesh(V_raw, F_raw, uv):
    """Weld 3D positions -> unique V; keep per-raw-vertex UV so seam positions
    (same xyz, different uv) become per-corner UV discontinuities the decimater
    preserves."""
    pr = np.round(V_raw, 5)
    uniq, inv = np.unique(pr, axis=0, return_inverse=True)
    inv = inv.astype(np.int64)
    V = uniq.astype(np.float64)
    F = inv[F_raw]      # 3D face indices into welded V
    FT = F_raw          # UV indices into TC (per raw vertex; differ at seams)
    TC = uv
    return V, F, TC, FT


def write_obj(path, V, TC, F, FT):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for t in TC:
            fh.write(f"vt {t[0]:.9g} {t[1]:.9g}\n")
        for i in range(len(F)):
            fv = F[i] + 1
            ft = FT[i] + 1
            fh.write(f"f {fv[0]}/{ft[0]} {fv[1]}/{ft[1]} {fv[2]}/{ft[2]}\n")


def read_obj(path):
    V = []; TC = []; F = []; FT = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                TC.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                fi = []; ti = []
                for p in line.split()[1:4]:
                    seg = p.split("/")
                    fi.append(int(seg[0]) - 1)
                    ti.append(int(seg[1]) - 1 if len(seg) > 1 and seg[1] else int(seg[0]) - 1)
                F.append(fi); FT.append(ti)
    return (np.array(V, dtype=np.float64),
            np.array(TC, dtype=np.float64) if TC else np.zeros((0, 2)),
            np.array(F, dtype=np.int64), np.array(FT, dtype=np.int64))


def seam_edges_from(F, TC, FT):
    """Edges shared by exactly 2 faces whose UV corners differ => seam."""
    if len(TC) == 0 or len(FT) == 0:
        return []
    uvc = TC[FT]
    e2f = defaultdict(list)
    for fi in range(len(F)):
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            e2f[(min(u, v), max(u, v))].append(fi)

    def uv_of(face, vert):
        for j in range(3):
            if int(F[face, j]) == vert:
                return uvc[face, j]

    seams = set()
    for (a, b), fs in e2f.items():
        if len(fs) != 2:
            continue
        fa, fb = fs
        ua1, ub1 = uv_of(fa, a), uv_of(fa, b)
        ua2, ub2 = uv_of(fb, a), uv_of(fb, b)
        if not (np.allclose(ua1, ua2, atol=1e-4) and np.allclose(ub1, ub2, atol=1e-4)):
            seams.add((a, b))
    return sorted(seams)


def normalize_mesh(vertices):
    center = vertices.mean(axis=0)
    vc = vertices - center
    scale = float(np.max(np.linalg.norm(vc, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return vc / scale, center, scale


# ---------- decimation ----------
def run_decimate_one(wsl_in, target, strict):
    """Single-garment decimation; returns wsl output path or None."""
    cmd = f'{DECIMATER} "{wsl_in}" num-vertices {target} --strict {strict}'
    r = subprocess.run(["wsl", "--", "bash", "-lc", cmd],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    wsl_out = f'{wsl_in[:-4]}-decimated_to_{target}_vertices.obj'
    return wsl_out if r.returncode == 0 else None


def decimate_batch_par(win_dir, target, strict, par):
    """Parallel-decimate every *.obj in win_dir via one WSL xargs session.
    Returns set of gids that produced an output."""
    wsl_dir = win_to_wsl(win_dir)
    script = (
        f'cd "{wsl_dir}" && '
        f"find . -maxdepth 1 -type f -name '*.obj' ! -name '*decimated_to*' "
        f"| xargs -P {par} -I{{}} {DECIMATER} {{}} num-vertices {target} --strict {strict} "
        f">/dev/null 2>&1; echo XDONE"
    )
    subprocess.run(["wsl", "--", "bash", "-lc", script],
                   capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    done = set()
    for p in win_dir.glob("*-decimated_to_*_vertices.obj"):
        m = re.match(r"(.+)-decimated_to_\d+_vertices\.obj$", p.name)
        if m:
            done.add(m.group(1))
    return done


def pack_pt(Vd, Fd, TCd, FTd, gid, output_path, n_surface=2048):
    seam_edges = np.array(seam_edges_from(Fd, TCd, FTd), dtype=np.int64)
    if len(seam_edges) == 0:
        raise ValueError(f"{gid}: no seam edges after decimation")
    seam_chains = build_seam_chains(seam_edges)
    Vn, center, scale = normalize_mesh(Vd)
    mesh = trimesh.Trimesh(vertices=Vn, faces=Fd, process=False)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    edges = np.array(mesh.edges_unique, dtype=np.int64)
    one_ring = build_one_ringneighbors(Fd, len(Vn))
    surface_points = sample_surface_points(Vn, Fd, n_surface).astype(np.float32)
    ordered_chains = chaining_seams(Vn, Fd, seam_chains)
    validation = validate_chains(seam_edges, ordered_chains, edges)
    if not validation["valid"]:
        raise ValueError(f"{gid}: invalid seam chains: {validation}")
    data = {
        "gid": gid,
        "vertices": torch.from_numpy(Vn.astype(np.float32)),
        "faces": torch.from_numpy(Fd.astype(np.int64)),
        "vertex_normals": torch.from_numpy(vertex_normals),
        "edges": torch.from_numpy(edges),
        "one_ring_neighbors": one_ring,
        "seam_edges": torch.from_numpy(seam_edges),
        "seam_chains": seam_chains,
        "ordered_chains": ordered_chains,
        "surface_points": torch.from_numpy(surface_points),
        "center": center,
        "scale": scale,
        "n_vertices": len(Vn),
        "n_faces": len(Fd),
        "n_seam_edges": len(seam_edges),
        "n_chains": len(seam_chains),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    return {"gid": gid, "n_vertices": len(Vn), "n_faces": len(Fd),
            "n_seam_edges": len(seam_edges), "n_chains": len(seam_chains), "status": "ok"}


# ---------- archive iteration ----------
def _gid_from_member(member_name: str) -> str:
    parts = member_name.replace("\\", "/").split("/")
    return parts[-2]


def iter_garments(tar_path, gid_set=None):
    """Yield (gid, sim_bytes) for each _sim.ply in an archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith("_sim.ply"):
                continue
            gid = _gid_from_member(member.name)
            if gid_set is not None and gid not in gid_set:
                continue
            try:
                sim_bytes = tf.extractfile(member).read()
            except Exception:
                continue
            yield gid, sim_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=ROOT / "GarmentCodeData" / "GarmentCodeData_v2")
    ap.add_argument("--output_dir", type=Path,
                    default=ROOT / "processed_data_seamless")
    ap.add_argument("--target", type=int, default=1000, help="decimater num-vertices")
    ap.add_argument("--strict", type=int, default=2)
    ap.add_argument("--batch", type=int, default=100, help="objs per WSL session")
    ap.add_argument("--par", type=int, default=0, help="xargs parallelism (0=detect)")
    ap.add_argument("--max_gids", type=int, default=0, help="stop after N garments (0=all)")
    ap.add_argument("--max_vertices", type=int, default=8000, help="drop garments above (model adj limit)")
    ap.add_argument("--batch_filter", type=str, default="", help="e.g. garments_5000_0")
    ap.add_argument("--max_archives", type=int, default=0, help="stop after N archives")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    # detect parallelism
    par = args.par
    if par <= 0:
        r = subprocess.run(["wsl", "--", "bash", "-lc", "nproc"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            par = max(2, min(16, int(r.stdout.strip())))   # cap 16: drvfs chokes at higher concurrency (26k run used 16)
        except Exception:
            par = 8

    tar_paths = sorted(args.data_root.glob("garments_5000_*/default_body/data.tar.gz"))
    if args.batch_filter:
        tar_paths = [p for p in tar_paths if args.batch_filter in str(p)]
    if args.max_archives:
        tar_paths = tar_paths[:args.max_archives]
    print(f"archives={len(tar_paths)} target={args.target} strict={args.strict} "
          f"batch={args.batch} par={par} out={args.output_dir}")

    stats = {"ok": 0, "failed": 0, "too_big": 0, "skipped": 0, "skipped_failed": 0}
    failed_gids = []
    failed_path = args.output_dir / "_failed.json"
    failed_set = set()
    if failed_path.exists():
        try:
            for g in json.loads(failed_path.read_text(encoding="utf-8")):
                failed_set.add(str(g).split(":")[0])
            print(f"loaded {len(failed_set)} prior-failed gids -> will SKIP them")
        except Exception:
            pass
    n_seen = 0
    t0 = time.time()

    def flush_batch(batch):
        """batch: list of (gid, V, F, TC, FT). Write objs, decimate, pack, clear."""
        if not batch:
            return
        # clear tmp
        for f in TMP.glob("*"):
            f.unlink(missing_ok=True)
        # ensure dir exists (safety net for any path)
        wsl_tmp = win_to_wsl(TMP)
        subprocess.run(["wsl", "--", "bash", "-lc", f'mkdir -p "{wsl_tmp}"'],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
        # write inputs
        present = {}
        for gid, V, F, TC, FT in batch:
            obj_in = TMP / f"{gid}.obj"
            write_obj(obj_in, V, TC, F, FT)
            present[gid] = obj_in
        # Cascade decimation strict 2 -> 1 -> 0. After each stage, delete inputs of
        # garments that now have an output, so the next (more permissive) stage re-runs
        # ONLY still-failed inputs and cannot overwrite successes.
        for s in (2, 1, 0):
            decimate_batch_par(TMP, args.target, s, par)
            for gid, _V, _F, _TC, _FT in batch:
                if (TMP / f"{gid}-decimated_to_{args.target}_vertices.obj").exists():
                    (TMP / f"{gid}.obj").unlink(missing_ok=True)
        # pack each
        for gid, V, F, TC, FT in batch:
            out_obj = TMP / f"{gid}-decimated_to_{args.target}_vertices.obj"
            if not out_obj.exists():
                # try fallback-output naming (percent/strict1 used different target token)
                cands = list(TMP.glob(f"{gid}-decimated_to_*_vertices.obj"))
                out_obj = cands[0] if cands else None
            if out_obj is None or not out_obj.exists():
                stats["failed"] += 1
                failed_gids.append(gid); failed_set.add(gid)
                continue
            try:
                Vd, TCd, Fd, FTd = read_obj(out_obj)
                out_pt = args.output_dir / f"{gid}.pt"
                r = pack_pt(Vd, Fd, TCd, FTd, gid, out_pt)
                if args.max_vertices and r["n_vertices"] > args.max_vertices:
                    out_pt.unlink(missing_ok=True)
                    stats["too_big"] += 1
                else:
                    stats["ok"] += 1
            except Exception as e:
                stats["failed"] += 1
                failed_gids.append(f"{gid}:{e!r}"); failed_set.add(gid)
        for f in TMP.glob("*"):
            f.unlink(missing_ok=True)

    batch = []
    BATCH = args.batch
    stop = False
    for tar_path in tar_paths:
        if stop:
            break
        for gid, sim_bytes in iter_garments(tar_path):
            if args.max_gids and n_seen >= args.max_gids:
                stop = True
                break
            n_seen += 1
            out_pt = args.output_dir / f"{gid}.pt"
            if out_pt.exists():
                stats["skipped"] += 1
                continue
            if gid in failed_set:
                stats["skipped_failed"] += 1
                continue
            try:
                V_raw, F_raw, uv = load_sim(sim_bytes)
                V, F, TC, FT = to_obj_mesh(V_raw, F_raw, uv)
            except Exception:
                stats["failed"] += 1
                failed_gids.append(f"{gid}:load"); failed_set.add(gid)
                continue
            batch.append((gid, V, F, TC, FT))
            if len(batch) >= BATCH:
                flush_batch(batch)
                batch = []
                failed_path.write_text(json.dumps(sorted(failed_set)), encoding="utf-8")
                el = time.time() - t0
                print(f"  [seen={n_seen} ok={stats['ok']} fail={stats['failed']} "
                      f"skip={stats['skipped']} skipf={stats['skipped_failed']} "
                      f"big={stats['too_big']}] {el/60:.1f}min", flush=True)
    flush_batch(batch)
    failed_path.write_text(json.dumps(sorted(failed_set)), encoding="utf-8")

    el = time.time() - t0
    print("\n" + "=" * 60)
    print(f"SUMMARY  ({el/60:.1f} min)")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    failed_path.write_text(json.dumps(sorted(failed_set)), encoding="utf-8")
    if failed_gids:
        print(f"  (failures persisted to _failed.json; unique {len(failed_set)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
