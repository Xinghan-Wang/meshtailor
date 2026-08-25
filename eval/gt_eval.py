"""GT seam eval: 用 .pt 的 ordered_chains (GT seam, 理论最优切法) 当 seam,
跑 unwrap + 6 metric, 诊断 v3 penalty6 与论文差距的瓶颈归属。

  - GT area ~ v3pen6 (2.14) → 瓶颈在 unwrap 算法(ANGLE_BASED) / metric 实现
  - GT area ~ 论文 (1.10)   → 瓶颈在模型 seam 质量

不依赖模型/权重, 只需 processed_data/*.pt + blender。

Usage:
  python eval/gt_eval.py --split test --out_dir test_outputs_gt
  python eval/gt_eval.py --limit 5            # 先跑 5 件验证 pipeline
  python eval/gt_eval.py --skip_prepare       # unwrap 已跑完, 只重新聚合
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import unwrap_all, aggregate, PAPER  # noqa: E402

DATA_DIR = ROOT / "processed_data_seamless_maximal"
SPLIT_FILE = ROOT / "meshtailor" / "data" / "split_seamless_128k.json"

# v3 penalty6 baseline (checkpoints/eval_v3_result.txt, test 250)
V3_PEN6 = {
    "area_distortion": 2.1444,
    "compactness": 0.3620,
    "convexity": 0.9281,
    "seam_len/area": 3.0632,
    "jaggedness": 0.0008,
    "chart_count": 26.33,
}

METRIC_ORDER = ["area_distortion", "compactness", "convexity", "seam_len/area", "jaggedness"]


def export_obj(vertices, faces, path: Path) -> None:
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def chains_to_seam_edges(chains: list[list[int]]) -> list[list[int]]:
    seam: set[tuple[int, int]] = set()
    for c in chains:
        for k in range(len(c) - 1):
            a, b = int(c[k]), int(c[k + 1])
            seam.add((min(a, b), max(a, b)))
    return [list(e) for e in sorted(seam)]


def prepare_gt_outputs(out_dir: Path, data_dir: Path, gids: list[str]) -> tuple[float, float]:
    """Per gid: write mesh.obj (v+f) + seam.json (ordered_chains -> seam edges).

    Format matches inference.py so unwrap_all / aggregate work unchanged.
    """
    n_chain_total, n_edge_total = 0, 0
    for i, gid in enumerate(gids):
        pt = torch.load(data_dir / f"{gid}.pt", weights_only=False)
        d = out_dir / gid
        d.mkdir(parents=True, exist_ok=True)
        export_obj(pt["vertices"].numpy(), pt["faces"].numpy(), d / "mesh.obj")
        chains = pt["ordered_chains"]
        seam = chains_to_seam_edges(chains)
        (d / "seam.json").write_text(json.dumps(seam))
        (d / "chains.json").write_text(json.dumps(chains))
        n_chain_total += len(chains)
        n_edge_total += len(seam)
        if (i + 1) % 50 == 0:
            print(f"  prepared {i + 1}/{len(gids)}")
    n = max(len(gids), 1)
    return n_chain_total / n, n_edge_total / n


def diagnose(res: dict) -> None:
    gt_a = res["area_distortion"]
    v3_a = V3_PEN6["area_distortion"]
    pa_a = PAPER["area_distortion"]
    print("\n=== Diagnosis (area_distortion) ===")
    if gt_a >= 1.8:
        print(f"GT area_distortion = {gt_a:.2f} ~ v3pen6 ({v3_a}), 远高于论文 ({pa_a})")
        print("→ 瓶颈在 unwrap 算法 (ANGLE_BASED) / metric 实现, 非模型 seam 质量")
        print("→ 改进方向: 换 ABF++ / 校准 metric, 而非改模型")
    elif gt_a <= 1.4:
        print(f"GT area_distortion = {gt_a:.2f} ~ 论文 ({pa_a}), 远低于 v3pen6 ({v3_a})")
        print("→ 瓶颈在模型 seam 质量 (GT 切法远优于模型生成)")
        print("→ 改进方向: 改模型/数据/loss (让模型 seam 布局接近 GT)")
    else:
        print(f"GT area_distortion = {gt_a:.2f} 介于 v3pen6 ({v3_a}) 与论文 ({pa_a}) 之间")
        print("→ 混合瓶颈: unwrap 与模型 seam 都有改进空间")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default=str(ROOT / "gt_outputs"))
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--split_file", default=str(SPLIT_FILE))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_unwrap", action="store_true")
    ap.add_argument("--unwrap", choices=["abf", "blender"], default="abf")
    args = ap.parse_args()

    with open(args.split_file, encoding="utf-8") as f:
        gids = json.load(f)[args.split]
    if args.limit:
        gids = gids[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    if not args.skip_prepare:
        print(f">>> preparing GT outputs for {len(gids)} garments ...")
        mc, me = prepare_gt_outputs(out_dir, data_dir, gids)
        print(f"  mean chains={mc:.1f}  mean seam_edges={me:.1f}")
    if not args.skip_unwrap:
        print(f">>> UV unwrap ({args.unwrap}) ...")
        unwrap_all(out_dir, args.unwrap)
    print(">>> aggregating metrics ...")
    res = aggregate(out_dir, data_dir)

    print(f"\n=== GT seam eval on {res['n']} garments ({args.split}) ===")
    print(f"{'metric':<20} {'GT':>10} {'v3pen6':>10} {'paper':>10}")
    for k in METRIC_ORDER:
        gt = res[k]
        v3 = V3_PEN6[k]
        pa = PAPER.get(k, float("nan"))
        print(f"{k:<20} {gt:>10.4f} {v3:>10.4f} {pa:>10.4f}")
    print(f"{'chart_count':<20} {res['chart_count']:>10.2f} {V3_PEN6['chart_count']:>10.2f} {'—':>10}")

    diagnose(res)


if __name__ == "__main__":
    main()
