"""Run the full test evaluation pipeline and compare to the paper.

Steps:
  1. inference.py -> test_outputs/<gid>/{mesh.obj, seam.json}
  2. blender unwrap each -> test_outputs/<gid>/uv.obj
  3. uv_metrics + struct_metrics
  4. aggregate, compare to paper Table (GarmentCodeData, Full model row)

Usage:
  python eval/run_eval.py --ckpt checkpoints/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uv_metrics import metrics_for_obj  # noqa: E402
from struct_metrics import chart_count, chains_to_seam_set, seam_length_over_area  # noqa: E402
from unwrap_abf import unwrap  # noqa: E402

BLENDER = r"D:\Blender\blender.exe"
UNWRAP_SCRIPT = str(ROOT / "eval" / "unwrap_blender.py")
INFERENCE_SCRIPT = str(ROOT / "meshtailor" / "inference.py")

PAPER = {  # GarmentCodeData, Full model (Sec 4.6 ablation table)
    "area_distortion": 1.097,
    "compactness": 0.591,
    "convexity": 0.887,
    "seam_len/area": 2.25,
    "jaggedness": 0.485,
}


def run_inference(ckpt, split, out_dir, limit, eos_penalty=0.0, visit_penalty=0.0, temperature=None,
                  data_dir=None, split_file=None, eoc_penalty=0.0, min_chain_len=0):
    cmd = [sys.executable, INFERENCE_SCRIPT, "--ckpt", ckpt, "--split", split,
           "--out_dir", str(out_dir), "--bf16"]
    if data_dir:
        cmd += ["--data_dir", str(data_dir)]
    if split_file:
        cmd += ["--split_file", str(split_file)]
    if limit:
        cmd += ["--limit", str(limit)]
    if eos_penalty:
        cmd += ["--eos_penalty", str(eos_penalty)]
    if visit_penalty:
        cmd += ["--visit_penalty", str(visit_penalty)]
    if eoc_penalty:
        cmd += ["--eoc_penalty", str(eoc_penalty)]
    if min_chain_len:
        cmd += ["--min_chain_len", str(min_chain_len)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)


def unwrap_all(out_dir, method="abf"):
    dirs = sorted(p for p in Path(out_dir).iterdir() if p.is_dir())
    n_failed = 0
    for i, gd in enumerate(dirs):
        uv = gd / "uv.obj"
        if uv.exists():
            continue
        mesh = gd / "mesh.obj"; seam = gd / "seam.json"
        if not mesh.exists() or not seam.exists():
            continue
        try:
            unwrap(mesh, seam, uv, method=method)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  [{gd.name}] unwrap failed: {str(e)[:140]}")
        if (i + 1) % 25 == 0:
            print(f"  unwrapped {i + 1}/{len(dirs)} ({n_failed} failed)")
    if n_failed:
        print(f"  total unwrap failures: {n_failed}")


def aggregate(out_dir, data_dir):
    uv_metrics_acc = {k: [] for k in ["area_distortion", "compactness", "convexity", "boundary_jaggedness"]}
    chart_counts, seam_ratios = [], []
    n = 0
    for gd in sorted(Path(out_dir).iterdir()):
        if not gd.is_dir():
            continue
        uv = gd / "uv.obj"; seam = gd / "seam.json"
        if not uv.exists():
            continue
        n += 1
        m = metrics_for_obj(uv)
        for k in uv_metrics_acc:
            v = m.get(k)
            if v is not None and not np.isnan(v):
                uv_metrics_acc[k].append(v)
        import torch
        pt = torch.load(Path(data_dir) / f"{gd.name}.pt", weights_only=False)
        seam_set = {tuple(e) for e in json.loads(seam.read_text())}
        seam_set = {(min(a, b), max(a, b)) for a, b in seam_set}
        V = pt["vertices"].numpy().astype(np.float64)
        F = pt["faces"].numpy().astype(np.int64)
        chart_counts.append(chart_count(F, seam_set))
        _, _, r = seam_length_over_area(V, F, seam_set)
        seam_ratios.append(r)
    return {
        "n": n,
        "area_distortion": float(np.mean(uv_metrics_acc["area_distortion"])) if uv_metrics_acc["area_distortion"] else float("nan"),
        "chart_count": float(np.mean(chart_counts)) if chart_counts else float("nan"),
        "compactness": float(np.mean(uv_metrics_acc["compactness"])) if uv_metrics_acc["compactness"] else float("nan"),
        "convexity": float(np.mean(uv_metrics_acc["convexity"])) if uv_metrics_acc["convexity"] else float("nan"),
        "seam_len/area": float(np.mean(seam_ratios)) if seam_ratios else float("nan"),
        "jaggedness": float(np.mean(uv_metrics_acc["boundary_jaggedness"])) if uv_metrics_acc["boundary_jaggedness"] else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best_paper100k.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default=str(ROOT / "test_outputs"))
    ap.add_argument("--data_dir", default=str(ROOT / "processed_data_seamless_maximal"))
    ap.add_argument("--split_file", default=str(ROOT / "meshtailor" / "data" / "split_seamless_128k.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eos_penalty", type=float, default=0.0)
    ap.add_argument("--visit_penalty", type=float, default=0.0)
    ap.add_argument("--eoc_penalty", type=float, default=0.0)
    ap.add_argument("--min_chain_len", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--skip_inference", action="store_true")
    ap.add_argument("--skip_unwrap", action="store_true")
    ap.add_argument("--unwrap", choices=["abf", "blender"], default="abf",
                    help="UV flatten method (abf=geogram ABF++ via WSL, blender=ANGLE_BASED)")
    args = ap.parse_args()

    if not args.skip_inference:
        run_inference(args.ckpt, args.split, args.out_dir, args.limit, args.eos_penalty,
                      args.visit_penalty, args.temperature, args.data_dir, args.split_file,
                      args.eoc_penalty, args.min_chain_len)
    if not args.skip_unwrap:
        print(f">>> UV unwrap ({args.unwrap}) ...")
        unwrap_all(args.out_dir, args.unwrap)
    print(">>> aggregating metrics ...")
    res = aggregate(args.out_dir, args.data_dir)

    print(f"\n=== Test evaluation on {res['n']} garments ===")
    print(f"{'metric':<20} {'ours':>12} {'paper(full)':>14}")
    for k in ["area_distortion", "compactness", "convexity", "seam_len/area", "jaggedness"]:
        paper = PAPER.get(k, "")
        print(f"{k:<20} {res[k]:>12.4f} {paper:>14}")
    print(f"{'chart_count':<20} {res['chart_count']:>12.2f} {'(lower better)':>14}")


if __name__ == "__main__":
    main()
