"""Oracle-start ablation (parameterized; GPU, ~10 min for 150 garments).

For each garment, decode twice with the same seed:
  free    — normal p0 protocol (start sampled from the model's p(start));
  oracle  — each chain's first vertex forced to the GT chain's canonical
            start (canonicalize_chains of the stored ordered_chains).
Everything after the start (routing, EOC/EOS, temperature 0.1) is identical.

  recall jumps  -> start-vertex selection is the bottleneck
  recall flat   -> edge routing / under-generation is the bottleneck

Example:
  python tools/oracle_start.py --ckpt checkpoints/v11rep_legacy/best.pt \
      --data_dir processed_data_seamless_repaired \
      --split_file meshtailor/data/split_seamless_26k.json \
      --out checkpoints/oracle_start_v11rep_legacy.txt
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meshtailor.data.dataset import MeshTailorDataset, canonicalize_chains  # noqa: E402
from meshtailor.inference import load_model  # noqa: E402
from meshtailor.models.model import EOC_ID, EOS_ID  # noqa: E402

TEMP = 0.1

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def decode_forced_starts(model, enc, adj, forced_starts, max_len, temperature=TEMP):
    """Clone of model.decode with chain-start forcing (start_k -> vertex id)."""
    h_tilde, z, We = enc
    device = h_tilde.device
    num_verts = h_tilde.shape[0]

    chains = []
    cur_chain = []
    prev_vertex = -1
    past_kvs = None
    cur_tok = EOC_ID
    pos = 0
    chain_idx = 0

    for step in range(max_len):
        input_ids = torch.tensor([cur_tok], device=device, dtype=torch.long)
        token_emb = model._embed_tokens(input_ids, h_tilde)
        position_ids = torch.tensor([[step]], device=device)
        chain_pos = torch.tensor([[min(pos, model.max_chain_pos)]], device=device)
        q, past_kvs = model.decoder(token_emb.unsqueeze(0), z,
                                    position_ids, chain_pos, past_kvs)
        q_last = q.squeeze(0)[0]
        logits = q_last @ We.t()
        mask_next = model._inference_mask(
            cur_tok, adj, num_verts, prev_vertex, device,
            allow_eos_after_eoc=(model.sequence_protocol == "legacy"),
        )
        logits = logits + mask_next

        # Oracle injection: at chain-start state, force the GT start vertex.
        if cur_tok == EOC_ID and chain_idx < len(forced_starts):
            v0 = int(forced_starts[chain_idx])
            next_tok = v0 + 2 if v0 < num_verts else int(
                torch.multinomial((logits / temperature).softmax(-1), 1).item())
        elif temperature <= 0:
            next_tok = int(logits.argmax().item())
        else:
            next_tok = int(torch.multinomial((logits / temperature).softmax(-1), 1).item())

        if next_tok == EOS_ID:
            break
        if next_tok == EOC_ID:
            if cur_chain:
                chains.append(cur_chain)
                cur_chain = []
                chain_idx += 1
            new_pos = pos + 1 if cur_tok != EOC_ID else 0
            prev_vertex = -1
        else:
            v = next_tok - 2
            cur_chain.append(v)
            new_pos = 0 if cur_tok == EOC_ID else pos + 1
            prev_vertex = (cur_tok - 2) if cur_tok >= 2 else -1
        cur_tok = next_tok
        pos = new_pos

    if cur_chain:
        chains.append(cur_chain)
    return chains


def edges_of(chains):
    s = set()
    for c in chains:
        for a, b in zip(c[:-1], c[1:]):
            s.add((min(a, b), max(a, b)))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--max_len", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda"
    model = load_model(args.ckpt, device)
    ds = MeshTailorDataset(split=args.split, data_dir=args.data_dir,
                           split_file=args.split_file, augment=False)
    gids = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split][:args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = open(args.out, "w", encoding="utf-8")

    def w(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    w(f"=== oracle-start ablation (ckpt={args.ckpt}, n={len(gids)}, temp={TEMP}, "
      f"protocol={model.sequence_protocol}) ===")
    rows = []
    t0 = time.time()
    for i, gid in enumerate(gids):
        pt = torch.load(Path(args.data_dir) / f"{gid}.pt", weights_only=False)
        gt_chains = canonicalize_chains(pt["ordered_chains"], pt["vertices"], pt["faces"])
        starts = [int(c[0]) for c in gt_chains if len(c) >= 1]
        b = ds[i]
        adj = b["adj"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model.encode_for_generate(b["vertices6"].to(device),
                                            b["edge_index"].to(device),
                                            b["surface"].to(device))
            _seed(args.seed + i)
            free_chains = model.decode(enc, adj, max_len=args.max_len, temperature=TEMP)
            _seed(args.seed + i)
            pred = decode_forced_starts(model, enc, adj, starts, args.max_len)
        G = edges_of(gt_chains)
        P = edges_of(pred)
        P0 = edges_of(free_chains)
        inter_gp = len(G & P)
        inter_g0 = len(G & P0)
        rows.append(dict(
            gid=gid, gt=len(G), p_oracle=len(P), p_free=len(P0),
            rec_o=inter_gp / max(len(G), 1), prec_o=inter_gp / max(len(P), 1),
            rec_f=inter_g0 / max(len(G), 1), prec_f=inter_g0 / max(len(P0), 1),
            chains_o=len(pred), chains_f=len(free_chains), chains_gt=len(gt_chains),
        ))
        if (i + 1) % 30 == 0:
            w(f"  {i+1}/{len(gids)} ({time.time()-t0:.0f}s)")

    R = {k: float(np.mean([r[k] for r in rows])) for k in
         ["rec_o", "prec_o", "rec_f", "prec_f", "chains_o", "chains_f",
          "chains_gt", "gt", "p_oracle", "p_free"]}
    w("")
    w(f"{'':<18}{'free':>10}{'oracle':>10}")
    w(f"{'recall':<18}{R['rec_f']:>10.3f}{R['rec_o']:>10.3f}")
    w(f"{'precision':<18}{R['prec_f']:>10.3f}{R['prec_o']:>10.3f}")
    w(f"{'edges':<18}{R['p_free']:>10.1f}{R['p_oracle']:>10.1f}")
    w(f"{'chains(gen)':<18}{R['chains_f']:>10.1f}{R['chains_o']:>10.1f}")
    w(f"{'GT edges':<18}{R['gt']:>10.1f}{'':>10}")
    w(f"GT chains mean = {R['chains_gt']:.1f}")
    dr = [r["rec_o"] - r["rec_f"] for r in rows]
    w(f"per-garment recall delta: mean {np.mean(dr):+.3f}  "
      f"p10 {np.percentile(dr,10):+.3f}  p90 {np.percentile(dr,90):+.3f}  "
      f"improved {sum(1 for d in dr if d > 0.02)}/{len(dr)}")
    payload = {"config": {k: str(v) for k, v in vars(args).items()},
               "summary": R, "rows": rows}
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    w(f"[done] {len(rows)} garments in {(time.time()-t0)/60:.1f} min -> {args.out}")
    log.close()


if __name__ == "__main__":
    main()
