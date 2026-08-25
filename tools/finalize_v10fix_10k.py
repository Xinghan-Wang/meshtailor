"""Summarize v10fix and GT metrics from the same 10k test outputs."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval.run_eval import aggregate  # noqa: E402


def parse_uv(path: Path):
    verts, uvs, faces = [], [], []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "vt":
                uvs.append([float(t[1]), float(t[2])])
            elif t[0] == "f":
                fc = []
                for tok in t[1:]:
                    vi, ti = tok.split("/")[:2]
                    fc.append((int(vi) - 1, int(ti) - 1 if ti else -1))
                faces.append(fc)
    return np.asarray(verts), np.asarray(uvs), faces


def dual_area(out_dir: Path) -> dict:
    vals = {"std_log": [], "mean_absr1": [], "rms_log": []}
    for fp in sorted(glob.glob(str(out_dir / "*" / "uv.obj"))):
        try:
            v3, uv, faces = parse_uv(Path(fp))
            ratios = []
            for (a, ta), (b, tb), (c, tc) in faces:
                if min(ta, tb, tc) < 0:
                    continue
                a3 = 0.5 * np.linalg.norm(np.cross(v3[b] - v3[a], v3[c] - v3[a]))
                a2 = 0.5 * abs((uv[tb, 0] - uv[ta, 0]) * (uv[tc, 1] - uv[ta, 1])
                               - (uv[tc, 0] - uv[ta, 0]) * (uv[tb, 1] - uv[ta, 1]))
                if a3 > 1e-12 and a2 > 1e-12:
                    ratios.append(a2 / a3)
            if not ratios:
                continue
            r = np.asarray(ratios)
            log_r = np.log(r)
            vals["std_log"].append(float(log_r.std()))
            vals["mean_absr1"].append(float(np.abs(r - 1).mean()))
            vals["rms_log"].append(float(np.sqrt((log_r ** 2).mean())))
        except Exception:
            continue
    return {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()} | {"n": len(vals["std_log"])}


def seam_set(path: Path) -> set[tuple[int, int]]:
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in json.loads(path.read_text(encoding="utf-8"))}


def edge_scores(model_dir: Path, gt_dir: Path) -> dict:
    recalls, precisions, hit_total = [], [], 0
    pred_total = gt_total = 0
    for gd in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        pred_file = model_dir / gd.name / "seam.json"
        gt_file = gd / "seam.json"
        if not pred_file.exists() or not gt_file.exists():
            continue
        pred, gt = seam_set(pred_file), seam_set(gt_file)
        hit = len(pred & gt)
        recalls.append(hit / max(len(gt), 1))
        precisions.append(hit / max(len(pred), 1))
        hit_total += hit
        pred_total += len(pred)
        gt_total += len(gt)
    return {
        "n": len(recalls),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "micro_recall": hit_total / max(gt_total, 1),
        "micro_precision": hit_total / max(pred_total, 1),
        "pred_edges": pred_total / max(len(recalls), 1),
        "gt_edges": gt_total / max(len(recalls), 1),
    }


def chain_stats(out_dir: Path) -> dict:
    chains, edges = [], []
    for fp in sorted(glob.glob(str(out_dir / "*" / "chains.json"))):
        c = json.loads(Path(fp).read_text(encoding="utf-8"))
        chains.append(len(c))
        edges.append(sum(max(len(x) - 1, 0) for x in c))
    return {"n": len(chains), "chains": float(np.mean(chains)) if chains else float("nan"),
            "edges": float(np.mean(edges)) if edges else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--gt_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--split_file", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    model = aggregate(str(args.model_dir), str(args.data_dir))
    gt = aggregate(str(args.gt_dir), str(args.data_dir))
    model_area = dual_area(args.model_dir)
    gt_area = dual_area(args.gt_dir)
    scores = edge_scores(args.model_dir, args.gt_dir)
    model_chain = chain_stats(args.model_dir)
    gt_chain = chain_stats(args.gt_dir)

    def ratio(a, b):
        return a / b if b else float("nan")

    lines = []
    lines.append("=== v10fix 10k full test (p0 + GT, same ABF++ pipeline) ===")
    lines.append("checkpoint: best_v10fix.pt; protocol: temp=0.1, seed=20260818, no penalties")
    lines.append(f"n: model={model['n']} gt={gt['n']} uv_model={model_area['n']} uv_gt={gt_area['n']}")
    lines.append("")
    lines.append(f"{'metric':<20}{'v10fix':>12}{'GT':>12}{'v10fix/GT':>12}")
    for key in ["area_distortion", "compactness", "convexity", "seam_len/area", "jaggedness", "chart_count"]:
        lines.append(f"{key:<20}{model[key]:>12.4f}{gt[key]:>12.4f}{ratio(model[key], gt[key]):>12.4f}")
    lines.append("")
    lines.append(f"{'area(std_log)':<20}{model_area['std_log']:>12.4f}{gt_area['std_log']:>12.4f}{ratio(model_area['std_log'], gt_area['std_log']):>12.4f}")
    lines.append(f"{'area(mean|r-1|)':<20}{model_area['mean_absr1']:>12.4f}{gt_area['mean_absr1']:>12.4f}{ratio(model_area['mean_absr1'], gt_area['mean_absr1']):>12.4f}")
    lines.append(f"{'area(rms_log)':<20}{model_area['rms_log']:>12.4f}{gt_area['rms_log']:>12.4f}{ratio(model_area['rms_log'], gt_area['rms_log']):>12.4f}")
    lines.append("")
    lines.append(f"generated chains: v10fix={model_chain['chains']:.4f} GT={gt_chain['chains']:.4f}")
    lines.append(f"generated edges : v10fix={model_chain['edges']:.4f} GT={gt_chain['edges']:.4f}")
    for key in ["macro_recall", "macro_precision", "micro_recall", "micro_precision"]:
        lines.append(f"{key}: {scores[key]:.6f}")
    lines.append(f"mean unique edges: v10fix={scores['pred_edges']:.4f} GT={scores['gt_edges']:.4f}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {args.summary}")


if __name__ == "__main__":
    main()
