"""MeshTailor training loop — batched + optimized.

Paper Sec 4.1 (AdamW lr=1e-4, effective batch 64, 30 epochs) but with:
  - per-step batch=8 + grad accum=8 (effective 64), to feed the GPU
  - bf16 autocast (Flash SDPA), num_workers for async loading
  - dropout for regularization, vectorized neighbor mask
Single-card; gradient accumulation reaches the effective batch size.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshtailor.data.dataset import make_loader
from meshtailor.models.model import MeshTailor


ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device):
    model.eval()
    losses = []
    sums = {"tok_acc": 0.0, "eoc_acc": 0.0, "eos_acc": 0.0}
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            losses.append(out["loss"].item())
            for k, v in out.get("stats", {}).items():
                if v == v:  # skip NaN (no EOC/EOS targets in batch)
                    sums[k] += v
    model.train()
    n = max(len(losses), 1)
    return sum(losses) / n, {k: v / n for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--t_max", type=int, default=2000)
    ap.add_argument("--eoc_weight", type=float, default=1.0, help="loss weight for [EOC] (downweight <1 to make chains longer)")
    ap.add_argument("--eos_weight", type=float, default=1.0, help="loss weight for [EOS]")
    ap.add_argument("--ss_prob", type=float, default=0.0, help="scheduled-sampling prob (>0: step-by-step training on model's own samples)")
    ap.add_argument("--sequence_protocol", choices=["legacy", "paper"], default="legacy",
                    help="token grammar: legacy has final EOC before EOS; paper ends final chain with EOS")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after this many non-improving validation epochs (0=off)")
    ap.add_argument("--ckpt_dir", type=str, default=str(ROOT / "checkpoints"))
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "processed_data_seamless_maximal"))
    ap.add_argument("--split_file", type=str,
                    default=str(ROOT / "meshtailor" / "data" / "split_seamless_128k.json"))
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--tag", type=str, default="v2", help="ckpt filename tag")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: cap opt steps per epoch (0=no cap)")
    ap.add_argument("--limit_train", type=int, default=0,
                    help="pilot only: use the first N train GIDs (0=full split)")
    ap.add_argument("--limit_val", type=int, default=0,
                    help="pilot only: use the first N val GIDs (0=full split)")
    args = ap.parse_args()

    split_payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    train_gids = (split_payload["train"][:args.limit_train]
                  if args.limit_train else split_payload["train"])
    val_gids = (split_payload["val"][:args.limit_val]
                if args.limit_val else split_payload["val"])
    run_split_payload = {"train": train_gids, "val": val_gids}

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ckpt_dir / f"{args.tag}_manifest.json"
    run_manifest = {
        "status": "running",
        "checkpoint": str((ckpt_dir / f"best_{args.tag}.pt").resolve()),
        "last_checkpoint": str((ckpt_dir / f"last_{args.tag}.pt").resolve()),
        "config": vars(args).copy(),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "split_counts": {k: len(v) for k, v in run_split_payload.items()},
        "gid_list": [gid for values in run_split_payload.values() for gid in values],
        "sequence_protocol": args.sequence_protocol,
    }
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    model = MeshTailor(t_max=args.t_max, dropout=args.dropout,
                       eoc_weight=args.eoc_weight, eos_weight=args.eos_weight,
                       sequence_protocol=args.sequence_protocol).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)

    train_loader = make_loader("train", batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, data_dir=args.data_dir,
                               split_file=args.split_file, seed=args.seed, gids=train_gids)
    val_loader = make_loader("val", batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, data_dir=args.data_dir,
                               split_file=args.split_file, seed=args.seed + 1, gids=val_gids)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"batch={args.batch_size} accum={args.accum} dropout={args.dropout} "
          f"workers={args.num_workers} eoc_weight={args.eoc_weight} eos_weight={args.eos_weight} "
          f"ss_prob={args.ss_prob} data={args.data_dir}")

    best_val = float("inf")
    bad_epochs = 0
    start_epoch = 0
    if args.resume:
        rpath = (ckpt_dir / f"last_{args.tag}.pt"
                 if args.resume == "last" else Path(args.resume))
        if rpath.exists():
            ckpt = torch.load(rpath, map_location=device, weights_only=False)
            ckpt_protocol = ckpt.get("config", {}).get("sequence_protocol", "legacy")
            if ckpt_protocol != args.sequence_protocol:
                raise ValueError(
                    f"cannot resume {ckpt_protocol!r} checkpoint with "
                    f"{args.sequence_protocol!r} config"
                )
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
            bad_epochs = ckpt.get("bad_epochs", 0)
            print(f"resumed from {rpath}: start_epoch={start_epoch} best_val={best_val:.4f} bad_epochs={bad_epochs}")
        else:
            print(f"resume requested but {rpath} not found; training from scratch")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running, n_batch, opt_step, n_oom = 0.0, 0, 0, 0
        stat_sums = {"tok_acc": 0.0, "eoc_acc": 0.0, "eos_acc": 0.0}
        t0 = time.time()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"ep{epoch}")
        for i, batch in pbar:
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.forward_scheduled(batch, ss_prob=args.ss_prob) if args.ss_prob > 0 else model(batch)
                (out["loss"] / args.accum).backward()
                running += out["loss"].item()
                for k, v in out.get("stats", {}).items():
                    if v == v:
                        stat_sums[k] += v
                n_batch += 1
            except torch.cuda.OutOfMemoryError:
                # SS retains the full S-step graph; a few huge batches (large
                # max_N or long sequences) can OOM. Skip them and carry on.
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                continue
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
            if n_batch % 5 == 0:
                pbar.set_postfix(loss=f"{running/n_batch:.3f}", step=opt_step, oom=n_oom)
            if args.max_steps and opt_step >= args.max_steps:
                break

        if n_batch % args.accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        train_loss = running / n_batch
        train_stats = {k: v / max(n_batch, 1) for k, v in stat_sums.items()}
        val_loss, val_stats = evaluate(model, val_loader, device)
        msg = (f"epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
               f"| tok_acc={train_stats['tok_acc']:.3f}/{val_stats['tok_acc']:.3f} "
               f"| eoc_acc={train_stats['eoc_acc']:.3f}/{val_stats['eoc_acc']:.3f} "
               f"| eos_acc={train_stats['eos_acc']:.3f}/{val_stats['eos_acc']:.3f} "
               f"| {time.time()-t0:.0f}s")
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        last = {"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "best_val": best_val,
                "bad_epochs": bad_epochs, "config": vars(args).copy()}
        torch.save(last, ckpt_dir / f"last_{args.tag}.pt")
        if improved:
            torch.save(last, ckpt_dir / f"best_{args.tag}.pt")
            msg += " | *best*"
        run_manifest.update({
            "status": "running",
            "epoch": epoch,
            "best_val": best_val,
            "last_val": val_loss,
        })
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(msg)
        if args.patience and bad_epochs >= args.patience:
            print(f"early stop: {bad_epochs} non-improving epochs (patience={args.patience})")
            run_manifest["status"] = "completed"
            run_manifest["stop_reason"] = "early_stopping"
            manifest_path.write_text(
                json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            break

    if run_manifest["status"] == "running":
        run_manifest["status"] = "completed"
        run_manifest["stop_reason"] = "epochs_complete"
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
