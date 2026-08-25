"""ABF++ calibration on GT seams: does our ABF++ unwrap reproduce the paper's
GT row (area 1.095, compact 0.592, convex 0.891, jagged 0.473, charts 11.4)?

Prepares mesh.obj + seam.json (from .pt ordered_chains) for N test garments,
runs the WSL abf_unwrap on each, aggregates metrics.
"""
import sys, json, subprocess, torch, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from uv_metrics import metrics_for_obj
from struct_metrics import chart_count

DATA = ROOT / "processed_data_seamless_maximal"
SPLIT = ROOT / "meshtailor" / "data" / "split_seamless_128k.json"
OUT = ROOT / "tools" / "abf" / "_calib"
WSL_BIN = "/root/abf_toolkit/tool/build/abf_unwrap"
GEOLIB = "/root/abf_toolkit/geogram/build/Release/lib"

PAPER_GT = {"area_distortion": 1.095, "compactness": 0.592, "convexity": 0.891,
            "jaggedness": 0.473, "chart_count": 11.449}


def export_obj(V, F, p):
    with open(p, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in F:
            f.write(f"f {int(t[0]) + 1} {int(t[1]) + 1} {int(t[2]) + 1}\n")


def chains_to_seam_edges(chains):
    s = set()
    for c in chains:
        for k in range(len(c) - 1):
            a, b = int(c[k]), int(c[k + 1])
            s.add((min(a, b), max(a, b)))
    return [list(e) for e in sorted(s)]


def wsl_path(p):
    p = str(Path(p)).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}"


def run_abf(mesh_obj, seam_json, uv_obj, timeout=120):
    cmd_str = (f'export LD_LIBRARY_PATH={GEOLIB}:$LD_LIBRARY_PATH; '
               f'"{WSL_BIN}" "{wsl_path(mesh_obj)}" "{wsl_path(seam_json)}" "{wsl_path(uv_obj)}"')
    r = subprocess.run(["wsl.exe", "--", "bash", "-c", cmd_str],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"abf rc={r.returncode}: {r.stderr.decode(errors='replace')[:300]}")


def main(N=30):
    OUT.mkdir(parents=True, exist_ok=True)
    test = json.load(open(SPLIT, encoding="utf-8"))["test"][:N]
    acc = {k: [] for k in ["area_distortion", "compactness", "convexity", "boundary_jaggedness"]}
    charts = []
    n_fail = 0
    for i, gid in enumerate(test):
        d = OUT / gid
        d.mkdir(parents=True, exist_ok=True)
        uv = d / "uv.obj"
        if uv.exists():
            uv.unlink()  # re-run fresh
        pt = torch.load(DATA / f"{gid}.pt", weights_only=False)
        V = pt["vertices"].numpy().astype(np.float64)
        F = pt["faces"].numpy().astype(np.int64)
        export_obj(V, F, d / "mesh.obj")
        seam = chains_to_seam_edges(pt["ordered_chains"])
        (d / "seam.json").write_text(json.dumps(seam))
        try:
            run_abf(d / "mesh.obj", d / "seam.json", uv)
        except Exception as e:
            n_fail += 1
            print(f"  [{gid}] ABF FAILED: {str(e)[:120]}")
            continue
        m = metrics_for_obj(uv)
        for k in acc:
            v = m.get(k)
            if v is not None and not np.isnan(v):
                acc[k].append(v)
        ss = {(min(a, b), max(a, b)) for a, b in seam}
        charts.append(chart_count(F, ss))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{N} done; last area={acc['area_distortion'][-1]:.3f}")
    n = len(acc["area_distortion"])
    print(f"\n=== ABF++ GT calibration (n={n} ok, {n_fail} failed) ===")
    print(f"{'metric':<22}{'ABF-GT':>10}{'paper-GT':>10}")
    for k, pk in [("area_distortion", "area_distortion"), ("compactness", "compactness"),
                  ("convexity", "convexity"), ("boundary_jaggedness", "jaggedness")]:
        val = float(np.mean(acc[k])) if acc[k] else float("nan")
        print(f"{k:<22}{val:>10.3f}{PAPER_GT[pk]:>10.3f}")
    print(f"{'chart_count':<22}{np.mean(charts):>10.2f}{PAPER_GT['chart_count']:>10.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30)
    a = ap.parse_args()
    main(a.n)
