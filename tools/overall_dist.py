"""Overall (area) distortion following PartUV (arXiv:2511.16659).

Per chart C: stretch(f) = (A_uv(f)/A_3d(f)) / mean_C(A_uv/A_3d)
Per garment: D = max_C mean_{f in C} max(stretch, 1/stretch)
Dataset level: macro average over garments.

Charts are the vt-edge connected components of the unwrapped mesh
(eval.uv_metrics.uv_islands). The uniform global scale applied by our ABF++
unwrap cancels inside the chart-wise mean normalization, so the existing
uv.obj files can be used as-is. Also reports std(log r) as a regression check
against the existing summary files.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from uv_metrics import parse_uv_obj, uv_islands  # noqa: E402

MIN_AREA = 1e-12


def garment_stats(uv_path: Path, ratio_cap: float = 0.0) -> dict:
    verts3d, uvs, faces = parse_uv_obj(uv_path)
    if not faces:
        return {"dist": float("nan"), "std_log": float("nan"),
                "charts_valid": 0, "charts_failed": 0, "charts_dropped": 0,
                "tris_used": 0}
    tris = np.array(faces, dtype=np.int64)  # (n,3) pairs of (v_idx, vt_idx)
    v_idx = tris[:, :, 0]
    t_idx = tris[:, :, 1]
    p3 = verts3d[v_idx]                      # (n,3,3)
    a3 = 0.5 * np.linalg.norm(np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0]), axis=1)
    p2 = uvs[t_idx]                          # (n,3,2)
    a2 = 0.5 * np.abs((p2[:, 1, 0] - p2[:, 0, 0]) * (p2[:, 2, 1] - p2[:, 0, 1])
                      - (p2[:, 2, 0] - p2[:, 0, 0]) * (p2[:, 1, 1] - p2[:, 0, 1]))
    vt_ok = t_idx.min(axis=1) >= 0
    valid = vt_ok & (a3 > MIN_AREA) & (a2 > MIN_AREA)

    chart_of = np.full(len(faces), -1, dtype=np.int64)
    charts = uv_islands(uvs, faces)
    for ci, island in enumerate(charts):
        chart_of[island] = ci

    chart_vals = []
    charts_failed = 0
    charts_dropped = 0
    for ci in range(len(charts)):
        m = valid & (chart_of == ci)
        if not m.any():
            charts_failed += 1
            continue
        r = a2[m] / a3[m]
        # Drop partially-collapsed charts (ABF++ sliver artifacts): a clean
        # chart's stretch ratios stay within ~1 order of magnitude of each
        # other, while collapsed charts span 5+. The ratio is scale-invariant.
        if ratio_cap > 0 and r.max() / r.min() > ratio_cap:
            charts_dropped += 1
            continue
        s = r / r.mean()
        chart_vals.append(float(np.maximum(s, 1.0 / s).mean()))

    r_all = a2[valid] / a3[valid]
    std_log = float(np.log(r_all).std()) if len(r_all) > 1 else float("nan")
    return {
        "dist": max(chart_vals) if chart_vals else float("nan"),
        "std_log": std_log,
        "charts_valid": len(chart_vals),
        "charts_failed": charts_failed,
        "charts_dropped": charts_dropped,
        "tris_used": int(valid.sum()),
    }


def _worker(args):
    gid, path, ratio_cap = args
    try:
        return gid, garment_stats(Path(path), ratio_cap)
    except Exception as e:
        return gid, {"dist": float("nan"), "std_log": float("nan"),
                     "charts_valid": 0, "charts_failed": 0, "charts_dropped": 0,
                     "tris_used": 0, "error": str(e)}


def run_dir(out_dir: Path, workers: int, limit: int, ratio_cap: float) -> dict:
    dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    if limit > 0:
        dirs = dirs[:limit]
    jobs = [(d.name, str(d / "uv.obj"), ratio_cap) for d in dirs if (d / "uv.obj").exists()]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gid, stats in ex.map(_worker, jobs, chunksize=64):
            results[gid] = stats
    dists = [s["dist"] for s in results.values() if np.isfinite(s["dist"])]
    stds = [s["std_log"] for s in results.values() if np.isfinite(s["std_log"])]
    clean = [s["dist"] for s in results.values()
             if np.isfinite(s["dist"]) and s["charts_dropped"] == 0]
    return {
        "n": len(results),
        "n_valid": len(dists),
        "overall_dist": float(np.mean(dists)) if dists else float("nan"),
        "median_dist": float(np.median(dists)) if dists else float("nan"),
        "overall_dist_clean_only": float(np.mean(clean)) if clean else float("nan"),
        "n_clean": len(clean),
        "std_log": float(np.mean(stds)) if stds else float("nan"),
        "charts_failed_total": sum(s["charts_failed"] for s in results.values()),
        "charts_dropped_total": sum(s["charts_dropped"] for s in results.values()),
        "garments_with_drops": sum(1 for s in results.values() if s["charts_dropped"] > 0),
        "per_garment": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path,
                    default=ROOT / "test_outputs_full")
    ap.add_argument("--gt_dir", type=Path, default=ROOT / "gt_outputs_full")
    ap.add_argument("--out_prefix", type=Path, default=ROOT / "checkpoints" / "overall_dist")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", choices=["model", "gt", ""], default="")
    ap.add_argument("--ratio_cap", type=float, default=100.0,
                    help="drop charts whose stretch-ratio spread exceeds this "
                         "(ABF++ collapse artifacts); 0 disables")
    args = ap.parse_args()

    summary = {"formula": "max_C mean_f max(s, 1/s), s = r / mean_C(r), r = A_uv/A_3d",
               "source": "PartUV (arXiv:2511.16659)",
               "ratio_cap": args.ratio_cap,
               "paper_reference": {"mesh_tailor_overall_dist": 1.097, "gt_overall_dist": 1.095}}
    for name, d in (("model", args.model_dir), ("gt", args.gt_dir)):
        if args.only and args.only != name:
            continue
        print(f"[overall-dist] {name}: {d} (limit={args.limit}, workers={args.workers}, "
              f"ratio_cap={args.ratio_cap})", flush=True)
        res = run_dir(d, args.workers, args.limit, args.ratio_cap)
        per = res.pop("per_garment")
        summary[name] = res
        out_json = args.out_prefix.parent / f"{args.out_prefix.name}_{name}.json"
        out_json.write_text(json.dumps(per, indent=None), encoding="utf-8")
        print(f"  n={res['n']} valid={res['n_valid']} mean={res['overall_dist']:.4f} "
              f"median={res['median_dist']:.4f} clean_only={res['overall_dist_clean_only']:.4f} "
              f"(n_clean={res['n_clean']}) std_log={res['std_log']:.4f} "
              f"dropped_charts={res['charts_dropped_total']} "
              f"garments_with_drops={res['garments_with_drops']}", flush=True)

    if "model" in summary and "gt" in summary:
        m, g = summary["model"]["overall_dist"], summary["gt"]["overall_dist"]
        summary["model_over_gt"] = m / g if g else float("nan")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = args.out_prefix.parent / f"{args.out_prefix.name}_summary.txt"
    txt.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[overall-dist] summary -> {txt}")


if __name__ == "__main__":
    main()
