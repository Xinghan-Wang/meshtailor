"""Inference: load best.pt, autoregressively generate seam chains on test split.

Outputs per garment: mesh.obj, seam.json (undirected edge pairs), chains.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from meshtailor.data.dataset import MeshTailorDataset
from meshtailor.models.model import MeshTailor


def load_model(ckpt_path: str, device: str = "cuda") -> MeshTailor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = MeshTailor(
        t_max=int(cfg.get("t_max", 2000)),
        dropout=float(cfg.get("dropout", 0.0)),
        eoc_weight=float(cfg.get("eoc_weight", 1.0)),
        eos_weight=float(cfg.get("eos_weight", 1.0)),
        sequence_protocol=cfg.get("sequence_protocol", "legacy"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    # Keep the exact checkpoint configuration available to the output
    # manifest; inference must not silently override its grammar.
    model.checkpoint_config = dict(cfg)
    model.eval()
    return model


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best_paper100k.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default=str(ROOT / "test_outputs"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--eos_penalty", type=float, default=0.0, help="subtract from [EOS] logit to delay stopping")
    ap.add_argument("--visit_penalty", type=float, default=0.0, help="penalize vertices used in prior chains (anti-overlap)")
    ap.add_argument("--eoc_penalty", type=float, default=0.0,
                    help="soft: subtract from [EOC] logit (2 = UV-area-priority alternative, chart oversoots)")
    ap.add_argument("--min_chain_len", type=int, default=0,
                    help="hard floor: forbid [EOC] until chain has >= this many vertices "
                         "(0 = off = p0 paper protocol, the validated default; 8 = demoted mcl8 variant)")
    ap.add_argument("--max_len", type=int, default=0,
                    help="decode token budget; 0 uses checkpoint/model t_max")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--data_dir", default=str(ROOT / "processed_data_seamless_maximal"))
    ap.add_argument("--split_file", default=str(Path(__file__).resolve().parent / "data" / "split_seamless_128k.json"))
    ap.add_argument("--skip_existing", action="store_true", help="skip garments already having seam.json in out_dir")
    args = ap.parse_args()

    device = "cuda"
    model = load_model(args.ckpt, device)
    ds = MeshTailorDataset(split=args.split, data_dir=args.data_dir, split_file=args.split_file)
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_config": getattr(model, "checkpoint_config", {}),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "split": args.split,
        "gid_list": list(ds.gids[:n]),
        "seed": args.seed,
        "sequence_protocol": model.sequence_protocol,
        "temperature": args.temperature,
        "eos_penalty": args.eos_penalty,
        "eoc_penalty": args.eoc_penalty,
        "visit_penalty": args.visit_penalty,
        "max_len": args.max_len or model.t_max,
        "min_chain_len": args.min_chain_len,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_chain_total, n_edge_total = 0, 0
    for i in tqdm(range(n), desc="inference"):
        item_seed = args.seed + i
        random.seed(item_seed)
        np.random.seed(item_seed % (2**32 - 1))
        torch.manual_seed(item_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(item_seed)
        batch = ds[i]
        gid = batch["gid"]
        d_dir = out / gid
        if args.skip_existing and (d_dir / "seam.json").exists():
            continue
        verts6 = batch["vertices6"].to(device)
        ei = batch["edge_index"].to(device)
        surf = batch["surface"].to(device)
        adj = batch["adj"].to(device)

        if args.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chains = model.generate(verts6, ei, surf, adj,
                                        temperature=args.temperature,
                                        max_len=(args.max_len or None),
                                        eos_penalty=args.eos_penalty,
                                        visit_penalty=args.visit_penalty,
                                        eoc_penalty=args.eoc_penalty,
                                        min_chain_len=args.min_chain_len)
        else:
            chains = model.generate(verts6, ei, surf, adj,
                                    temperature=args.temperature,
                                    max_len=(args.max_len or None),
                                    eos_penalty=args.eos_penalty,
                                    visit_penalty=args.visit_penalty,
                                    eoc_penalty=args.eoc_penalty,
                                    min_chain_len=args.min_chain_len)

        d_dir = out / gid
        d_dir.mkdir(parents=True, exist_ok=True)
        pt = torch.load(ds.data_dir / f"{gid}.pt", weights_only=False)
        export_obj(pt["vertices"].numpy(), pt["faces"].numpy(), d_dir / "mesh.obj")
        seam = chains_to_seam_edges(chains)
        (d_dir / "seam.json").write_text(json.dumps(seam))
        (d_dir / "chains.json").write_text(json.dumps(chains))

        n_chain_total += len(chains)
        n_edge_total += len(seam)

    print(f"generated {n} garments -> {out}")
    print(f"mean chains={n_chain_total / max(n, 1):.1f}  mean seam_edges={n_edge_total / max(n, 1):.1f}")


if __name__ == "__main__":
    main()
