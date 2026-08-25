"""Paired legacy-vs-paper structural evaluation on identical GIDs/seeds.

The two checkpoints are evaluated sequentially to avoid holding two full
models on the GPU.  Every checkpoint sees the same dataset item and
``seed + item_index``.  Sweep one inference argument at a time by invoking
this tool repeatedly and comparing the emitted JSON manifests.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.struct_metrics import chart_count
from meshtailor.data.dataset import MeshTailorDataset
from meshtailor.inference import load_model
from meshtailor.data.validate_chains import chain_edge_rows


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _graph_stats(edges: set[tuple[int, int]]) -> dict[str, int]:
    vertices = {v for edge in edges for v in edge}
    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    components = len({find(v) for v in vertices}) if vertices else 0
    return {
        "vertices": len(vertices),
        "components": components,
        "cycle_rank": len(edges) - len(vertices) + components,
    }


def _sample_metrics(
    pred_chains: list[list[int]],
    gt_chains: list[list[int]],
    faces: np.ndarray,
    trace: dict[str, Any],
) -> dict[str, Any]:
    pred_rows = chain_edge_rows(pred_chains)
    pred_edges = set(pred_rows)
    gt_edges = set(chain_edge_rows(gt_chains))
    hit = len(pred_edges & gt_edges)
    pred_loops = sum(1 for c in pred_chains if len(c) >= 2 and c[0] == c[-1])
    gt_loops = sum(1 for c in gt_chains if len(c) >= 2 and c[0] == c[-1])
    pred_graph = _graph_stats(pred_edges)
    gt_graph = _graph_stats(gt_edges)
    return {
        "generated_chains": len(pred_chains),
        "gt_chains": len(gt_chains),
        "raw_transitions": len(pred_rows),
        "unique_edges": len(pred_edges),
        "duplicate_transitions": len(pred_rows) - len(pred_edges),
        "duplicate_transition_ratio": (len(pred_rows) - len(pred_edges)) / max(len(pred_rows), 1),
        "gt_unique_edges": len(gt_edges),
        "edge_recall": hit / max(len(gt_edges), 1),
        "edge_precision": hit / max(len(pred_edges), 1),
        "closure_rate": pred_loops / max(len(pred_chains), 1),
        "gt_closure_rate": gt_loops / max(len(gt_chains), 1),
        "chart_count": chart_count(faces, pred_edges),
        "gt_chart_count": chart_count(faces, gt_edges),
        "component_count": pred_graph["components"],
        "cycle_rank": pred_graph["cycle_rank"],
        "gt_component_count": gt_graph["components"],
        "gt_cycle_rank": gt_graph["cycle_rank"],
        "eoc_tokens": int(trace["eoc_tokens"]),
        "eos_tokens": int(trace["eos_tokens"]),
        "terminated_by_eos": int(trace["terminated_by_eos"]),
        "decode_steps": len(trace["tokens"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))] if rows else []
    out: dict[str, Any] = {"samples": len(rows)}
    for key in numeric:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return out


def _run_one(
    label: str,
    ckpt: str,
    ds: MeshTailorDataset,
    n: int,
    args,
) -> dict[str, Any]:
    model = load_model(ckpt, device=args.device)
    rows: list[dict[str, Any]] = []
    for i in tqdm(range(n), desc=label):
        _seed(args.seed + i)
        item = ds[i]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            pred, trace = model.generate(
                item["vertices6"].to(args.device),
                item["edge_index"].to(args.device),
                item["surface"].to(args.device),
                item["adj"].to(args.device),
                max_len=args.max_len or None,
                temperature=args.temperature,
                eos_penalty=args.eos_penalty,
                visit_penalty=args.visit_penalty,
                eoc_penalty=args.eoc_penalty,
                min_chain_len=args.min_chain_len,
                return_trace=True,
            )
        pt = torch.load(ds.data_dir / f"{item['gid']}.pt", weights_only=False)
        sample = _sample_metrics(
            pred,
            item["chains"],
            pt["faces"].numpy().astype(np.int64),
            trace,
        )
        sample["gid"] = item["gid"]
        rows.append(sample)
    protocol = model.sequence_protocol
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(Path(ckpt).resolve()),
        "sequence_protocol": protocol,
        "summary": _aggregate(rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_ckpt", required=True)
    ap.add_argument("--paper_ckpt", required=True)
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "processed_data_seamless_maximal"))
    ap.add_argument("--split_file", default=str(Path(__file__).resolve().parents[1] / "meshtailor" / "data" / "split_seamless_128k.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--eos_penalty", type=float, default=0.0)
    ap.add_argument("--eoc_penalty", type=float, default=0.0)
    ap.add_argument("--visit_penalty", type=float, default=0.0)
    ap.add_argument("--min_chain_len", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    ds = MeshTailorDataset(
        split=args.split,
        data_dir=args.data_dir,
        split_file=args.split_file,
        augment=False,
    )
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    result = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "split": args.split,
        "gid_list": list(ds.gids[:n]),
        "seed": args.seed,
        "limit": n,
        "inference": {
            key: getattr(args, key)
            for key in (
                "temperature", "eos_penalty", "eoc_penalty", "visit_penalty",
                "min_chain_len", "max_len", "bf16",
            )
        },
        "legacy": _run_one("legacy", args.legacy_ckpt, ds, n, args),
        "paper": _run_one("paper", args.paper_ckpt, ds, n, args),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "limit": n,
        "legacy": result["legacy"]["summary"],
        "paper": result["paper"]["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
