"""Rebuild and audit seam chains in a copy of a processed-data directory.

The source directory is never modified.  The destination is populated only
after each sample passes topology validation; a non-zero failure count makes
the run exit unsuccessfully and records the failed GIDs in the manifest.

Example::

    python tools/repair_seam_chains.py \
        --src_dir processed_data_seamless \
        --out_dir processed_data_seamless_repaired \
        --split_file meshtailor/data/split_seamless_128k.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshtailor.data.chaining_seams import chaining_seams
from meshtailor.data.seam_extraction import build_seam_chains, build_seam_chains_maximal
from meshtailor.data.validate_chains import chain_edge_rows, validate_sample


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _edge_set(edges: Any) -> set[tuple[int, int]]:
    arr = _to_numpy(edges).astype(np.int64, copy=False)
    if arr.size == 0:
        return set()
    arr = arr.reshape(-1, 2)
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in arr.tolist()}


def _chain_metrics(chains: list[list[int]], seam_edges: Any) -> dict[str, float | int]:
    transitions = chain_edge_rows(chains)
    unique_transitions = set(transitions)
    n_chains = len(chains)
    n_loops = sum(
        1 for c in chains if len(c) >= 2 and int(c[0]) == int(c[-1])
    )
    return {
        "chain_count": n_chains,
        "seam_edge_count": len(_edge_set(seam_edges)),
        "seam_edge_rows": int(len(_to_numpy(seam_edges).reshape(-1, 2)))
        if _to_numpy(seam_edges).size
        else 0,
        "tau_len_legacy": 2 + sum(len(c) for c in chains) + n_chains,
        "tau_len_paper": 2 + sum(len(c) for c in chains) + max(n_chains - 1, 0),
        "loop_chain_ratio": n_loops / n_chains if n_chains else 0.0,
        "sample_loop_ratio": int(n_loops > 0),
        "duplicate_transitions": len(transitions) - len(unique_transitions),
    }


def _split_digest(split: dict[str, list[str]]) -> str:
    payload = json.dumps(split, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(rows)}
    for key in (
        "chain_count",
        "seam_edge_count",
        "seam_edge_rows",
        "tau_len_legacy",
        "tau_len_paper",
        "loop_chain_ratio",
        "sample_loop_ratio",
        "duplicate_transitions",
    ):
        vals = np.asarray([r[prefix + key] for r in rows], dtype=np.float64)
        if vals.size == 0:
            out[key] = {"count": 0}
            continue
        out[key] = {
            "count": int(vals.size),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max()),
        }
    return out


def _repair_one(src_path: Path, dst_path: Path, mode: str = "strict") -> dict[str, Any]:
    data = torch.load(src_path, weights_only=False)
    seam_edges = data["seam_edges"]
    old_chains = data.get("ordered_chains", data.get("seam_chains", []))
    vertices = _to_numpy(data["vertices"])
    faces = _to_numpy(data["faces"])
    fallback = False
    if mode == "maximal":
        new_seam_chains = build_seam_chains_maximal(_to_numpy(seam_edges), vertices)
    else:
        new_seam_chains = build_seam_chains(_to_numpy(seam_edges))
    new_ordered_chains = chaining_seams(vertices, faces, new_seam_chains)

    repaired = dict(data)
    repaired["seam_chains"] = new_seam_chains
    repaired["ordered_chains"] = new_ordered_chains
    repaired["n_seam_edges"] = len(_edge_set(seam_edges))
    repaired["n_chains"] = len(new_seam_chains)
    report = validate_sample(repaired)
    if not report["valid"] and mode == "maximal":
        fallback = True
        new_seam_chains = build_seam_chains(_to_numpy(seam_edges))
        new_ordered_chains = chaining_seams(vertices, faces, new_seam_chains)
        repaired["seam_chains"] = new_seam_chains
        repaired["ordered_chains"] = new_ordered_chains
        repaired["n_chains"] = len(new_seam_chains)
        report = validate_sample(repaired)
    if not report["valid"]:
        raise ValueError(json.dumps(report, sort_keys=True))

    old_edge_count = len(_edge_set(seam_edges))
    if repaired["n_seam_edges"] != old_edge_count:
        raise ValueError(f"seam edge count changed: {old_edge_count} -> {repaired['n_seam_edges']}")

    destination_tmp = dst_path.with_name(dst_path.name + ".tmp")
    torch.save(repaired, destination_tmp)
    destination_tmp.replace(dst_path)

    old_metrics = _chain_metrics(old_chains, seam_edges)
    new_metrics = _chain_metrics(new_ordered_chains, seam_edges)
    return {
        "gid": str(data.get("gid", src_path.stem)),
        "mode": mode,
        "fallback_to_strict": fallback,
        "old_chain_count": len(old_chains),
        "new_chain_count": len(new_ordered_chains),
        "old_seam_edge_count": old_metrics["seam_edge_count"],
        "new_seam_edge_count": new_metrics["seam_edge_count"],
        "old_duplicate_transitions": old_metrics["duplicate_transitions"],
        "new_duplicate_transitions": new_metrics["duplicate_transitions"],
        "old": old_metrics,
        "new": new_metrics,
        "validation": report,
    }


def _audit_existing(src_path: Path, dst_path: Path) -> dict[str, Any]:
    """Audit an already-written repaired sample without rewriting it."""
    source = torch.load(src_path, weights_only=False)
    repaired = torch.load(dst_path, weights_only=False)
    seam_edges = repaired["seam_edges"]
    old_chains = source.get("ordered_chains", source.get("seam_chains", []))
    new_chains = repaired["ordered_chains"]
    report = validate_sample(repaired)
    if not report["valid"]:
        raise ValueError(json.dumps(report, sort_keys=True))
    old_metrics = _chain_metrics(old_chains, seam_edges)
    new_metrics = _chain_metrics(new_chains, seam_edges)
    return {
        "gid": str(source.get("gid", src_path.stem)),
        "old_chain_count": len(old_chains),
        "new_chain_count": len(new_chains),
        "old_seam_edge_count": old_metrics["seam_edge_count"],
        "new_seam_edge_count": new_metrics["seam_edge_count"],
        "old_duplicate_transitions": old_metrics["duplicate_transitions"],
        "new_duplicate_transitions": new_metrics["duplicate_transitions"],
        "old": old_metrics,
        "new": new_metrics,
        "validation": report,
    }


def _process_task(task: tuple[Path, Path, str, bool]) -> dict[str, Any]:
    """Top-level worker (picklable under spawn). Never raises across processes."""
    src_path, dst_path, mode, resume = task
    try:
        if resume and dst_path.exists():
            row = _audit_existing(src_path, dst_path)
        else:
            row = _repair_one(src_path, dst_path, mode)
        return {"kind": "row", "gid": src_path.stem, "row": row}
    except Exception as exc:  # noqa: BLE001 - reported via manifest, not raised
        return {"kind": "error", "gid": src_path.stem, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--audit_per_split", type=int, default=1000)
    parser.add_argument("--resume", action="store_true",
                        help="audit existing destination .pt files without rewriting them")
    parser.add_argument("--mode", choices=("strict", "maximal"), default="strict",
                        help="chain decomposition: strict breaks at every junction, "
                             "maximal traces paper B.1 maximal chains across junctions")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel repair workers (spawn pool)")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N sorted GIDs (0 = all)")
    parser.add_argument("--max_fallback_rate", type=float, default=0.005,
                        help="abort when the maximal->strict fallback rate exceeds this")
    parser.add_argument("--fallback_min_samples", type=int, default=2000,
                        help="minimum completed samples before the fallback-rate check")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()
    split_file = args.split_file.resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"source directory does not exist: {src_dir}")
    if out_dir == src_dir:
        raise SystemExit("refusing in-place repair; choose a separate --out_dir")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise SystemExit(f"destination is non-empty; choose a new path or use --resume: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "repair_meta.json"
    split = json.loads(split_file.read_text(encoding="utf-8"))
    meta = {
        "mode": args.mode,
        "source_dir": str(src_dir),
        "split_file": str(split_file),
        "split_sha256": _split_digest(split),
    }
    if args.resume and meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev.get("mode") != args.mode:
            raise SystemExit(
                f"resume mode mismatch: destination was produced with mode="
                f"{prev.get('mode')!r}, requested {args.mode!r}"
            )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    split_gids = [gid for values in split.values() for gid in values]
    if len(split_gids) != len(set(split_gids)):
        raise SystemExit("split file contains duplicate GIDs")

    src_files = {p.stem: p for p in src_dir.glob("*.pt")}
    missing = [gid for gid in split_gids if gid not in src_files]
    if missing:
        raise SystemExit(f"split references {len(missing)} missing .pt files; first={missing[:5]}")

    sampled = {
        gid
        for values in split.values()
        for gid in values[: max(0, args.audit_per_split)]
    }
    split_sets = {name: set(gids) for name, gids in split.items()}
    gid_to_split = {
        gid: name for name, gids in split_sets.items() for gid in gids
    }
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    t0 = time.time()

    gids = sorted(src_files)
    if args.limit > 0:
        gids = gids[: args.limit]
    tasks = [(src_files[gid], out_dir / src_files[gid].name, args.mode, args.resume)
             for gid in gids]

    fallback_count = 0
    aborted = False
    done = 0

    def ingest(result: dict[str, Any]) -> None:
        nonlocal fallback_count, done
        done += 1
        if result["kind"] == "error":
            failures.append({"gid": result["gid"], "error": result["error"]})
            print(f"FAILED {result['gid']}: {result['error']}", flush=True)
            return
        row = result["row"]
        if row.get("fallback_to_strict"):
            fallback_count += 1
        gid = row["gid"]
        row["sampled"] = gid in sampled
        for prefix in ("old", "new"):
            for key, value in row[prefix].items():
                row[f"{prefix}_{key}"] = value
        all_rows.append(row)

    def fallback_exceeded() -> bool:
        return (
            args.mode == "maximal"
            and done >= args.fallback_min_samples
            and fallback_count > args.max_fallback_rate * done
        )

    def check_progress() -> None:
        if done % 1000 == 0:
            print(
                f"processed={done}/{len(tasks)} ok={len(all_rows)} "
                f"failed={len(failures)} fallback={fallback_count} "
                f"elapsed_min={(time.time() - t0) / 60:.1f}",
                flush=True,
            )

    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_process_task, tasks, chunksize=32):
                ingest(result)
                check_progress()
                if fallback_exceeded():
                    aborted = True
                    print(
                        f"ABORT: fallback rate {fallback_count}/{done} exceeds "
                        f"{args.max_fallback_rate}; terminating pool",
                        flush=True,
                    )
                    pool.terminate()
                    break
    else:
        for task in tasks:
            ingest(_process_task(task))
            check_progress()
            if fallback_exceeded():
                aborted = True
                print(
                    f"ABORT: fallback rate {fallback_count}/{done} exceeds "
                    f"{args.max_fallback_rate}",
                    flush=True,
                )
                break

    by_split: dict[str, dict[str, Any]] = {}
    rows_by_split = {name: [] for name in split}
    for row in all_rows:
        name = gid_to_split.get(row["gid"])
        if name is not None:
            rows_by_split[name].append(row)
    for name, gids in split.items():
        rows = rows_by_split[name]
        by_split[name] = {
            "sample_1000_old": _summary([r for r in rows if r["sampled"]], "old_"),
            "sample_1000_new": _summary([r for r in rows if r["sampled"]], "new_"),
            "full_old": _summary(rows, "old_"),
            "full_new": _summary(rows, "new_"),
        }

    validation_totals = {
        key: int(sum(int(r["validation"].get(key, 0)) for r in all_rows))
        for key in (
            "invalid_chain_steps",
            "coverage_mismatch",
            "duplicate_chain_edges",
            "immediate_backtracks",
            "invalid_loop_closures",
        )
    }

    if aborted:
        status = "aborted_high_fallback"
    elif not failures and len(all_rows) == len(tasks):
        status = "ok"
    else:
        status = "failed"
    summary = {
        "status": status,
        "mode": args.mode,
        "workers": args.workers,
        "limit": args.limit,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(all_rows) if all_rows else 0.0,
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "split_file": str(split_file),
        "split_counts": {name: len(gids) for name, gids in split.items()},
        "split_sha256": _split_digest(split),
        "source_pt_count": len(tasks),
        "written_pt_count": len(all_rows),
        "failed_count": len(failures),
        "failed": failures,
        "validation_totals": validation_totals,
        "full_old": _summary(all_rows, "old_"),
        "full_new": _summary(all_rows, "new_"),
        "by_split": by_split,
        "rows": all_rows,
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "seam_chain_repair_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "status", "mode", "source_pt_count", "written_pt_count", "failed_count",
        "fallback_count", "fallback_rate", "split_sha256", "elapsed_sec",
    )}, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else (2 if aborted else 1)


if __name__ == "__main__":
    raise SystemExit(main())
