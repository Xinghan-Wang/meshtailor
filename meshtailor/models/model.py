"""MeshTailor top-level model (paper Sec. 3) — batched.

Dual-stream encoder (Enc_G graph + Enc_P frozen point cloud) -> cross-attention
fusion -> autoregressive decoder -> pointer layer over candidate set
U = {[EOC], [EOS]} ∪ V. Trains with teacher-forced next-token CE (L_AR).

Candidate ids: [EOC]=0, [EOS]=1, vertex v -> v+2.

forward() is batched (B garments): Enc_G runs on the PyG-style big graph
(cat + offset), fusion does per-vertex cross-attention to each mesh's Z, and
the decoder/pointer operate on padded (B, max_N, ...) / (B, max_S, ...) tensors
with a loss mask. generate() stays single-mesh for autoregressive inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enc_g import GraphEncoder
from .enc_p import PointCloudEncoder
from .fusion import CrossAttentionFusion
from .decoder import MeshTailorDecoder
from .pointer import PointerLayer

EOC_ID = 0
EOS_ID = 1
NEG_INF = float("-inf")


class MeshTailor(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dec_layers: int = 6,
        fusion_layers: int = 2,
        max_chain_pos: int = 512,
        dropout: float = 0.0,
        t_max: int = 2000,
        num_frequencies: int = 128,
        eoc_weight: float = 1.0,
        eos_weight: float = 1.0,
        sequence_protocol: str = "legacy",
    ):
        super().__init__()
        if sequence_protocol not in {"legacy", "paper"}:
            raise ValueError(f"unknown sequence_protocol={sequence_protocol!r}")
        self.enc_g = GraphEncoder(d_model=d_model, num_frequencies=num_frequencies)
        self.enc_p = PointCloudEncoder(d_model=d_model)
        self.fusion = CrossAttentionFusion(d_model, num_heads, num_layers=fusion_layers, dropout=dropout)
        self.decoder = MeshTailorDecoder(d_model, num_heads, num_layers=dec_layers,
                                         max_chain_pos=max_chain_pos, dropout=dropout)
        self.pointer = PointerLayer(d_model)
        self.control_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.d_model = d_model
        self.max_chain_pos = max_chain_pos
        self.t_max = t_max
        self.eoc_weight = eoc_weight
        self.eos_weight = eos_weight
        self.sequence_protocol = sequence_protocol

    def chains_to_tau(self, ordered_chains: list[list[int]]) -> list[int]:
        # Inference starts from EOC and predicts the first chain start from
        # that state.  Make the same initial state an explicit training input
        # so the first vertex is included in the next-token loss.
        tau: list[int] = [EOC_ID]
        for i, chain in enumerate(ordered_chains):
            for v in chain:
                tau.append(int(v) + 2)
            # EOC separates consecutive chains.  The legacy target also
            # places a final EOC before EOS; the paper protocol ends the
            # final chain directly with EOS.
            if i < len(ordered_chains) - 1 or self.sequence_protocol == "legacy":
                tau.append(EOC_ID)
        tau.append(EOS_ID)
        return tau

    # ---------- batched helpers ----------
    def _scatter_to_padded(self, h, batch_ptr, n_list, max_N):
        B = len(n_list)
        d = h.shape[-1]
        out = h.new_zeros(B, max_N, d)
        mask = h.new_zeros(B, max_N, dtype=torch.bool)
        for i, n in enumerate(n_list):
            sel = (batch_ptr == i).nonzero(as_tuple=True)[0]
            out[i, :n] = h[sel]
            mask[i, :n] = True
        return out, mask

    def _embed_tokens_batch(self, input_ids, h_padded):
        B, S = input_ids.shape
        d = self.d_model
        is_vertex = input_ids >= 2
        v_idx = (input_ids - 2).clamp(min=0)
        gathered = torch.gather(h_padded, 1, v_idx.unsqueeze(-1).expand(-1, -1, d))
        ctrl = self.control_embed[input_ids.clamp(max=1)].to(h_padded.dtype)
        return torch.where(is_vertex.unsqueeze(-1), gathered, ctrl)

    def _chain_positions_batch(self, input_ids):
        B, S = input_ids.shape
        pos = torch.zeros_like(input_ids)
        pi = torch.zeros(B, dtype=torch.long, device=input_ids.device)
        for t in range(S):
            pos[:, t] = pi
            pi = torch.where(input_ids[:, t] == EOC_ID, torch.zeros_like(pi), pi + 1)
        return pos

    def _neighbor_mask_batch(self, input_ids, adj_pad, max_N, valid_vertices=None):
        B, S = input_ids.shape
        device = input_ids.device
        if valid_vertices is None:
            valid_vertices = torch.ones(B, max_N, dtype=torch.bool, device=device)
        mask = torch.full((B, S, max_N + 2), NEG_INF, device=device)
        is_vertex = input_ids >= 2
        is_eoc = input_ids == EOC_ID
        v_idx = (input_ids - 2).clamp(min=0)
        neigh = torch.gather(adj_pad, 1, v_idx.unsqueeze(-1).expand(-1, -1, max_N))  # (B,S,max_N)
        valid = valid_vertices.unsqueeze(1)
        mask_v = torch.where(neigh & valid, 0.0, NEG_INF)
        mask_e = torch.where(valid.expand(B, S, max_N), 0.0, NEG_INF)

        # The paper forbids immediate backtracking (A -> B -> A).  The
        # previous token is therefore the token immediately before the
        # current input row, not the current token itself.
        prev_ids = torch.full_like(input_ids, -1)
        if S > 1:
            prev_ids[:, 1:] = input_ids[:, :-1]
        prev_is_vertex = prev_ids >= 2
        prev_idx = (prev_ids - 2).clamp(min=0)
        cand_idx = torch.arange(max_N, device=device).view(1, 1, max_N)
        backtrack = (is_vertex.unsqueeze(-1) & prev_is_vertex.unsqueeze(-1)
                     & (cand_idx == prev_idx.unsqueeze(-1)))
        mask_v = mask_v.masked_fill(backtrack, NEG_INF)

        mask[..., 2:] = torch.where(is_vertex.unsqueeze(-1), mask_v, mask_e)
        mask[..., EOC_ID] = torch.where(is_vertex, 0.0, NEG_INF)
        allow_eos = is_vertex | (is_eoc & (self.sequence_protocol == "legacy"))
        mask[..., EOS_ID] = torch.where(allow_eos, 0.0, NEG_INF)
        return mask

    def forward(self, batch):
        device = self.control_embed.device
        vertices = batch["vertices6"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_ptr = batch["batch_ptr"].to(device)
        n_list = batch["n_list"]
        max_N = batch["max_N"]
        surface = batch["surface"].to(device)
        chains_list = batch["chains_list"]
        adj_list = batch["adj_list"]
        B = batch["B"]

        h = self.enc_g(vertices, edge_index)                       # (total_N, d)
        z = self.enc_p(surface)                                    # (B, 256, d)
        h_padded, valid_vertices = self._scatter_to_padded(h, batch_ptr, n_list, max_N)  # (B, max_N, d)
        h_tilde = self.fusion(h_padded, z)                         # (B, max_N, d)

        adj_pad = torch.zeros(B, max_N, max_N, dtype=torch.bool, device=device)
        for i, a in enumerate(adj_list):
            n = a.shape[0]
            adj_pad[i, :n, :n] = a.to(device)

        tau_list = [self.chains_to_tau(c)[: self.t_max + 1] for c in chains_list]
        lens = [len(t) for t in tau_list]
        max_seq = max(lens)
        tau_pad = torch.zeros(B, max_seq, dtype=torch.long, device=device)
        seq_mask = torch.zeros(B, max_seq, dtype=torch.bool, device=device)
        for i, t in enumerate(tau_list):
            tt = torch.tensor(t, dtype=torch.long, device=device)
            tau_pad[i, : len(t)] = tt
            seq_mask[i, : len(t)] = True

        input_ids = tau_pad[:, :-1]
        target = tau_pad[:, 1:]
        target_mask = seq_mask[:, 1:]
        S = input_ids.shape[1]

        # Keep training representation identical to autoregressive inference:
        # both token inputs and pointer candidates must use the fused mesh
        # features.  Using h_padded here silently bypasses fusion during the
        # standard teacher-forcing path.
        token_emb = self._embed_tokens_batch(input_ids, h_tilde)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        chain_pos = self._chain_positions_batch(input_ids).clamp(0, self.max_chain_pos)
        nm_mask = self._neighbor_mask_batch(input_ids, adj_pad, max_N, valid_vertices)

        # A tiny number of legacy seam annotations contain a literal A-B-A
        # subpath.  The paper mask forbids this transition, but hard-masking
        # the teacher target would make the whole batch CE infinite.  Keep
        # the prohibition everywhere else and unmask only the annotated
        # target so the exceptional sample remains trainable.
        prev_ids = torch.full_like(input_ids, -1)
        if S > 1:
            prev_ids[:, 1:] = input_ids[:, :-1]
        bt_target = ((input_ids >= 2) & (prev_ids >= 2) & (target >= 2)
                     & (target == prev_ids)) & target_mask
        if bt_target.any():
            flat_nm = nm_mask.reshape(-1, nm_mask.size(-1))
            flat_tgt = target.reshape(-1)
            rows = bt_target.reshape(-1).nonzero(as_tuple=True)[0]
            flat_nm[rows, flat_tgt[rows]] = 0.0

        control = self.control_embed.to(h_padded.dtype).unsqueeze(0).expand(B, 2, self.d_model)
        candidates = torch.cat([control, h_tilde], dim=1)          # (B, max_N+2, d)

        q, _ = self.decoder(token_emb, z, position_ids, chain_pos)    # (B, S, d)
        logits = self.pointer(q, candidates, nm_mask)              # (B, S, max_N+2)

        valid = target_mask.reshape(-1)
        logits_v = logits.reshape(-1, logits.size(-1))[valid]
        target_v = target.reshape(-1)[valid]
        if self.eoc_weight != 1.0 or self.eos_weight != 1.0:
            num_classes = logits.size(-1)
            w = torch.ones(num_classes, device=logits.device, dtype=torch.float32)
            w[EOC_ID] = self.eoc_weight
            w[EOS_ID] = self.eos_weight
            loss = F.cross_entropy(logits_v, target_v, weight=w)
        else:
            loss = F.cross_entropy(logits_v, target_v)
        with torch.no_grad():
            pred = logits_v.argmax(dim=-1)
            correct = pred == target_v
            is_eoc = target_v == EOC_ID
            is_eos = target_v == EOS_ID
            stats = {
                "tok_acc": correct.float().mean().item(),
                "eoc_acc": (correct[is_eoc].float().mean().item()
                            if is_eoc.any() else float("nan")),
                "eos_acc": (correct[is_eos].float().mean().item()
                            if is_eos.any() else float("nan")),
            }
        return {"loss": loss, "logits": logits, "target": target,
                "target_mask": target_mask, "stats": stats}

    def forward_scheduled(self, batch, ss_prob: float = 0.0):
        """Scheduled-sampling training forward: step-by-step autoregressive
        decoding WITH gradients. At each step the input token is, with prob
        ss_prob, the model's own masked sample (detached) instead of GT —
        closing the train/inference exposure gap. Loss is CE vs GT target,
        with eoc_weight/eos_weight applied (same as forward)."""
        device = self.control_embed.device
        vertices = batch["vertices6"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_ptr = batch["batch_ptr"].to(device)
        n_list = batch["n_list"]
        max_N = batch["max_N"]
        surface = batch["surface"].to(device)
        chains_list = batch["chains_list"]
        adj_list = batch["adj_list"]
        B = batch["B"]

        h = self.enc_g(vertices, edge_index)
        z = self.enc_p(surface)
        h_padded, valid_vertices = self._scatter_to_padded(h, batch_ptr, n_list, max_N)
        h_tilde = self.fusion(h_padded, z)

        adj_pad = torch.zeros(B, max_N, max_N, dtype=torch.bool, device=device)
        for i, a in enumerate(adj_list):
            n = a.shape[0]
            adj_pad[i, :n, :n] = a.to(device)

        tau_list = [self.chains_to_tau(c)[: self.t_max + 1] for c in chains_list]
        lens = [len(t) for t in tau_list]
        max_seq = max(lens)
        tau_pad = torch.zeros(B, max_seq, dtype=torch.long, device=device)
        seq_mask = torch.zeros(B, max_seq, dtype=torch.bool, device=device)
        for i, t in enumerate(tau_list):
            tt = torch.tensor(t, dtype=torch.long, device=device)
            tau_pad[i, : len(t)] = tt
            seq_mask[i, : len(t)] = True

        target = tau_pad[:, 1:]           # GT tokens to predict at each step
        target_mask = seq_mask[:, 1:]
        S = target.shape[1]
        num_classes = max_N + 2

        # candidates / We computed once
        control = self.control_embed.to(h_tilde.dtype).unsqueeze(0).expand(B, 2, self.d_model)
        candidates = torch.cat([control, h_tilde], dim=1)
        We = self.pointer.W(candidates)   # (B, max_N+2, d)

        if self.eoc_weight != 1.0 or self.eos_weight != 1.0:
            w = torch.ones(num_classes, device=device, dtype=torch.float32)
            w[EOC_ID] = self.eoc_weight
            w[EOS_ID] = self.eos_weight
        else:
            w = None

        cur_tok = tau_pad[:, 0].clone()         # first input = GT first token (matches forward)
        prev_tok = torch.full_like(cur_tok, EOC_ID)
        pi = torch.zeros(B, dtype=torch.long, device=device)   # chain position of cur_tok
        past_kvs = None
        total_loss = torch.zeros((), device=device)
        n_valid = 0
        rand_draws = torch.rand(S, device=device) if ss_prob > 0 else None

        for t in range(S):
            emb = self._embed_tokens_batch(cur_tok.unsqueeze(1), h_tilde).squeeze(1)  # (B, d)

            position_ids = torch.full((B, 1), t, dtype=torch.long, device=device)
            chain_pos_in = pi.unsqueeze(-1).clamp(0, self.max_chain_pos)
            q, past_kvs = self.decoder(emb.unsqueeze(1), z, position_ids, chain_pos_in,
                                       past_kvs, collect_kv=True)
            q_last = q.squeeze(1)                                         # (B, d)
            logits = torch.bmm(q_last.unsqueeze(1), We.transpose(1, 2)).squeeze(1)  # (B, N+2)

            # neighbor mask for the current (possibly sampled) input token
            nm_input = torch.stack([prev_tok, cur_tok], dim=1)
            nm = self._neighbor_mask_batch(nm_input, adj_pad, max_N,
                                           valid_vertices)[:, 1]
            logits = logits + nm

            # CE loss vs GT target. Under SS the input may have drifted, making
            # the GT target unreachable (masked to -inf); skip loss there (can't
            # blame the model for an impossible target from a drifted state).
            tgt = target[:, t]
            tgt_idx = torch.arange(B, device=device)
            bt_target = (cur_tok >= 2) & (prev_tok >= 2) & (tgt >= 2) & (tgt == prev_tok)
            if bt_target.any():
                nm[bt_target, tgt[bt_target]] = 0.0
            reachable = torch.isfinite(nm[tgt_idx, tgt])
            m = target_mask[:, t] & reachable
            if m.any():
                loss_t = F.cross_entropy(logits[m], tgt[m], weight=w, reduction="sum")
                total_loss = total_loss + loss_t
                n_valid += int(m.sum().item())

            # choose next input token: model sample (prob ss_prob) or GT.
            # Clamp -inf (masked) to a large negative so softmax is never NaN
            # (a finished garment's cur_tok can be EOS -> all-masked row).
            with torch.no_grad():
                if ss_prob > 0 and rand_draws[t] < ss_prob:
                    probs = logits.clamp(min=-30.0).softmax(-1)
                    nxt = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    nxt = tau_pad[:, t + 1] if (t + 1) < max_seq else tau_pad[:, -1]
            prev_tok = cur_tok
            cur_tok = nxt.detach()
            pi = torch.where(cur_tok == EOC_ID, torch.zeros_like(pi), pi + 1)

        loss = total_loss / max(n_valid, 1)
        return {"loss": loss}

    # ---------- single-mesh autoregressive inference (unchanged) ----------
    def _embed_tokens(self, input_ids, h_tilde):
        S = input_ids.shape[0]
        emb = torch.zeros(S, self.d_model, device=h_tilde.device, dtype=h_tilde.dtype)
        is_vertex = input_ids >= 2
        if is_vertex.any():
            emb[is_vertex] = h_tilde[input_ids[is_vertex] - 2]
        is_control = ~is_vertex
        if is_control.any():
            emb[is_control] = self.control_embed[input_ids[is_control]].to(emb.dtype)
        return emb

    def _chain_positions(self, input_ids):
        S = input_ids.shape[0]
        pos = torch.zeros(S, dtype=torch.long, device=input_ids.device)
        pi = 0
        for t in range(S):
            pos[t] = pi
            pi = 0 if int(input_ids[t]) == EOC_ID else min(pi + 1, self.max_chain_pos)
        return pos

    def _neighbor_mask(self, input_ids, adj, num_verts):
        S = input_ids.shape[0]
        mask = torch.full((S, num_verts + 2), NEG_INF, device=input_ids.device)
        for t in range(S):
            tok = int(input_ids[t])
            if tok == EOC_ID:
                mask[t, 2:2 + num_verts] = 0
                if self.sequence_protocol == "legacy":
                    mask[t, EOS_ID] = 0
            elif tok == EOS_ID:
                return mask
            else:
                v = tok - 2
                neighbors = adj[v].nonzero(as_tuple=True)[0]
                mask[t, neighbors + 2] = 0
                mask[t, EOC_ID] = 0
                mask[t, EOS_ID] = 0
                if t > 0:
                    prev = int(input_ids[t - 1])
                    if prev >= 2:
                        mask[t, prev] = NEG_INF
        return mask

    @torch.no_grad()
    def encode_for_generate(self, vertices6, edge_index, surface):
        """Run encoders + fusion once; reusable across multiple decodes (best-of-N)."""
        h = self.enc_g(vertices6, edge_index)
        z = self.enc_p(surface)
        h_tilde = self.fusion(h.unsqueeze(0), z).squeeze(0)
        candidates = torch.cat([self.control_embed.to(h_tilde.dtype), h_tilde], dim=0)
        We = self.pointer.W(candidates)
        return h_tilde, z, We

    @torch.no_grad()
    def decode(self, enc, adj, max_len: int | None = None, temperature: float = 0.1,
               eos_penalty: float = 0.0, visit_penalty: float = 0.0, eoc_penalty: float = 0.0,
               min_chain_len: int = 0, return_trace: bool = False):
        """Autoregressive seam generation with KV cache (paper B.3.4 / B.3.5),
        from precomputed encoder outputs `enc` = (h_tilde, z, We).

        eoc_penalty: subtract from [EOC] logit to discourage ending a chain early
            (the v5 model over-produces short len-2 chains).
        min_chain_len: forbid [EOC] entirely until the current chain has at least
            this many vertices (hard minimum chain length)."""
        h_tilde, z, We = enc
        device = h_tilde.device
        num_verts = h_tilde.shape[0]
        if max_len is None:
            max_len = self.t_max

        chains: list[list[int]] = []
        cur_chain: list[int] = []
        prev_vertex = -1
        past_kvs = None
        cur_tok = EOC_ID
        pos = 0  # chain-local position of cur_tok
        visited_count = torch.zeros(num_verts, device=device) if visit_penalty > 0 else None
        trace = {"tokens": [], "eoc_tokens": 0, "eos_tokens": 0,
                 "terminated_by_eos": False}

        for step in range(max_len):
            input_ids = torch.tensor([cur_tok], device=device, dtype=torch.long)
            token_emb = self._embed_tokens(input_ids, h_tilde)            # (1, d)
            position_ids = torch.tensor([[step]], device=device)          # (1, 1)
            chain_pos = torch.tensor([[min(pos, self.max_chain_pos)]], device=device)
            q, past_kvs = self.decoder(token_emb.unsqueeze(0), z,
                                       position_ids, chain_pos, past_kvs)
            q_last = q.squeeze(0)[0]
            logits = q_last @ We.t()
            mask_next = self._inference_mask(
                cur_tok, adj, num_verts, prev_vertex, device,
                allow_eos_after_eoc=(self.sequence_protocol == "legacy"),
            )
            logits = logits + mask_next
            if visit_penalty > 0:
                logits[2:2 + num_verts] -= visited_count * visit_penalty
            if eos_penalty > 0:
                logits[EOS_ID] -= eos_penalty
            # discourage / forbid ending the chain too early (v5 over-produces len-2 chains)
            if len(cur_chain) < min_chain_len:
                logits[EOC_ID] = NEG_INF       # hard minimum chain length
            elif eoc_penalty > 0:
                logits[EOC_ID] -= eoc_penalty  # soft penalty
            if temperature <= 0:
                next_tok = int(logits.argmax().item())
            else:
                probs = (logits / temperature).softmax(-1)
                next_tok = int(torch.multinomial(probs, 1).item())

            trace["tokens"].append(next_tok)

            if next_tok == EOS_ID:
                trace["eos_tokens"] += 1
                trace["terminated_by_eos"] = True
                break
            if next_tok == EOC_ID:
                trace["eoc_tokens"] += 1
                if cur_chain:
                    chains.append(cur_chain)
                    if visit_penalty > 0:
                        visited_count[torch.tensor(cur_chain, device=device)] += 1
                    cur_chain = []
                new_pos = pos + 1 if cur_tok != EOC_ID else 0
                prev_vertex = -1
            else:
                v = next_tok - 2
                cur_chain.append(v)
                new_pos = 0 if cur_tok == EOC_ID else pos + 1
                # Keep the vertex before the current token so the next
                # mask blocks A in the A -> B -> A pattern.
                prev_vertex = (cur_tok - 2) if cur_tok >= 2 else -1
            cur_tok = next_tok
            pos = new_pos

        if cur_chain:
            chains.append(cur_chain)
        return (chains, trace) if return_trace else chains

    def generate(self, vertices6, edge_index, surface, adj,
                 max_len: int | None = None, temperature: float = 0.1,
                 eos_penalty: float = 0.0, visit_penalty: float = 0.0,
                 eoc_penalty: float = 0.0, min_chain_len: int = 0,
                 return_trace: bool = False):
        """Convenience wrapper: encode once then decode."""
        enc = self.encode_for_generate(vertices6, edge_index, surface)
        return self.decode(enc, adj, max_len=max_len, temperature=temperature,
                           eos_penalty=eos_penalty, visit_penalty=visit_penalty,
                           eoc_penalty=eoc_penalty, min_chain_len=min_chain_len,
                           return_trace=return_trace)

    def _inference_mask(self, cur_tok, adj, num_verts, prev_vertex, device,
                        allow_eos_after_eoc: bool = True):
        mask = torch.full((num_verts + 2,), NEG_INF, device=device)
        if cur_tok == EOC_ID:
            mask[2:2 + num_verts] = 0
            if allow_eos_after_eoc:
                mask[EOS_ID] = 0
        elif cur_tok == EOS_ID:
            return mask
        else:
            v = cur_tok - 2
            neighbors = adj[v].nonzero(as_tuple=True)[0]
            mask[neighbors + 2] = 0
            mask[EOC_ID] = 0
            mask[EOS_ID] = 0
            if prev_vertex >= 0:
                mask[prev_vertex + 2] = NEG_INF
        return mask
