"""Probe: do sliding-window (S) layers pile attention mass on the oldest in-window key?

`dev/attention-ideas.md` lists attention sinks as the one untried idea that fits this
architecture: 3 of 4 layers run a 512-token window, every row starts with BOS, and past
position 512 an S layer can no longer see it. The failure mode reported for windowed
attention (StreamingLLM) is that the softmax still needs somewhere to dump probability, so
mass piles onto whatever key is oldest in the window -- a moving, meaningless sink.

This probe measures whether that actually happens here, before anything is built. It patches
`flash_attn.flash_attn_func`, which is the single seam every layer's attention goes through,
recomputes the softmax exactly in fp32 for a subsample of query positions, and reports where
the mass lands. The real kernel still produces the layer's output, so the forward pass is
unchanged.

Read the output as:

    oldest   mass on the single oldest in-window key (for an L layer that key is position 0)
    oldest8  mass on the 8 oldest in-window keys
    pos0     mass on absolute position 0 (the row's first BOS)
    bos      mass on every BOS in the window (rows pack ~3 documents, so BOS recurs)
    bosN     BOS tokens in the window, mean count / fraction of queries with at least one
    bos|in   mass on BOS *conditioned* on a BOS being in the window -- what the layer does with
             a sink when it has one, as opposed to how often it has one
    H yes/no attention entropy (nats) for queries that do / do not have a BOS in window. A layer
             starved of a sink should look less resolved (higher H) when it has none
    |v|bos   norm of the value vector at BOS positions over the mean value norm. A true sink is a
             no-op: mass lands on it without moving the output, so its value norm collapses. Near
             1.0 instead means BOS is carrying real content and the mass on it is not a dump
    recent64 mass on the 64 most recent keys, the locality baseline
    unif     1/(keys in window), what every number above would be under uniform attention

Verdict: the pathology is present if S layers put many multiples of `unif` on `oldest`, and
the effect concentrates in a few heads (printed as the per-head max). If `oldest` sits near
`unif` while `bos`/`recent64` carry the mass, S layers are not starved of a sink and the idea
should be dropped.

Examples:

    python -m scripts.probe_attention_sinks --model-tag d12 --step 150
    python -m scripts.probe_attention_sinks --model-tag d12 --step 150 --batches 4
    python -m scripts.probe_attention_sinks --model-tag d12 --step 1260 --source base --split train
"""
import argparse
from collections import defaultdict

import torch

import nanochat.flash_attention as fa
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


class SinkRecorder:
    """Wraps flash_attn_func: real kernel for the output, exact fp32 softmax for the stats."""

    def __init__(self, n_layer, num_queries, min_pos, bos_id):
        self.n_layer = n_layer
        self.num_queries = num_queries
        self.min_pos = min_pos
        self.bos_id = bos_id
        self.call = 0
        self.ids = None            # (B, T) tokens of the batch currently in flight
        self.sums = defaultdict(float)   # (layer, metric) -> running sum
        self.head_sums = {}        # layer -> (H,) running sum of oldest-key mass
        self.profiles = {}         # layer -> (w+1,) running sum of mass by relative offset
        self.count = 0             # batches folded in
        self.windows = {}

    def __call__(self, q, k, v, causal=False, window_size=(-1, -1)):
        layer = self.call % self.n_layer
        self.call += 1
        out = self._real(q, k, v, causal=causal, window_size=window_size)
        with torch.no_grad():
            self._record(layer, q, k, v, window_size[0])
        return out

    def _record(self, layer, q, k, v, window):
        B, T, H, D = q.shape
        w = T - 1 if window < 0 or window >= T else window
        self.windows[layer] = w
        # Query positions to score: evenly spaced, all with the window fully slid past BOS
        # where the layer's window allows it.
        lo_pos = max(self.min_pos, 0)
        t = torch.linspace(lo_pos, T - 1, self.num_queries, device=q.device).long().unique()
        nq = t.numel()

        qs = q[:, t].float().transpose(1, 2)                     # (B, H, nq, D)
        ks = k.float().transpose(1, 2)                           # (B, Hkv, T, D)
        if ks.size(1) != H:                                      # GQA
            ks = ks.repeat_interleave(H // ks.size(1), dim=1)
        scores = (qs @ ks.transpose(-1, -2)) * (D ** -0.5)       # (B, H, nq, T)
        col = torch.arange(T, device=q.device)
        keep = (col <= t[:, None]) & (t[:, None] - col <= w)
        scores = scores.masked_fill(~keep, float("-inf"))
        p = torch.softmax(scores, dim=-1)

        lo = (t - w).clamp_min(0)                                # oldest in-window key
        idx = lo.view(1, 1, nq, 1).expand(B, H, nq, 1)
        oldest = p.gather(-1, idx).squeeze(-1)                   # (B, H, nq)
        off8 = (lo[:, None] + torch.arange(8, device=q.device)).clamp_max(T - 1)
        oldest8 = p.gather(-1, off8.view(1, 1, nq, 8).expand(B, H, nq, 8)).sum(-1)
        rec = (t[:, None] - torch.arange(64, device=q.device)).clamp_min(0)
        recent64 = p.gather(-1, rec.view(1, 1, nq, 64).expand(B, H, nq, 64)).sum(-1)
        bos = (p * (self.ids == self.bos_id).view(B, 1, 1, T)).sum(-1)
        pos0 = p[..., 0]
        keys = (t - lo + 1).float()                              # keys visible per query
        # How often a real sink is available in-window at all: packing puts a BOS every ~640
        # tokens, so an S layer usually still has one even past position 512.
        cum = (self.ids == self.bos_id).float().cumsum(1)         # (B, T)
        hi = cum.gather(1, t.view(1, nq).expand(B, nq))
        below = cum.gather(1, (lo - 1).clamp_min(0).view(1, nq).expand(B, nq))
        n_bos = hi - torch.where((lo > 0).view(1, nq), below, torch.zeros_like(below))
        top = p.max(-1)
        top_offset = (t.view(1, 1, nq) - top.indices).float()

        # Sink signature: a key that only absorbs probability has little to contribute.
        vn = v.float().norm(dim=-1)                               # (B, T, Hkv)
        bmask = (self.ids == self.bos_id).unsqueeze(-1)            # (B, T, 1)
        add_vb = (vn * bmask).sum().item()
        add_vn = vn.sum().item()
        n_vb = bmask.expand_as(vn).sum().item()

        has = (n_bos >= 1).view(B, 1, nq).expand(B, H, nq)
        ent = -(p.clamp_min(1e-12).log() * p).sum(-1)

        n = B * H * nq
        add = self.sums
        add[(layer, "oldest")] += oldest.sum().item()
        add[(layer, "oldest8")] += oldest8.sum().item()
        add[(layer, "pos0")] += pos0.sum().item()
        add[(layer, "bos")] += bos.sum().item()
        add[(layer, "recent64")] += recent64.sum().item()
        add[(layer, "unif")] += (1.0 / keys).sum().item() * B * H
        add[(layer, "topmass")] += top.values.sum().item()
        add[(layer, "topoff")] += top_offset.sum().item()
        add[(layer, "n")] += n
        add[(layer, "bosN")] += n_bos.sum().item()
        add[(layer, "bosAny")] += (n_bos >= 1).float().sum().item()
        add[(layer, "nq")] += B * nq
        add[(layer, "bosCond")] += (bos * has).sum().item()
        add[(layer, "nCond")] += has.sum().item()
        add[(layer, "entYes")] += (ent * has).sum().item()
        add[(layer, "entNo")] += (ent * ~has).sum().item()
        add[(layer, "nNo")] += (~has).sum().item()
        add[(layer, "vBos")] += add_vb
        add[(layer, "nVBos")] += n_vb
        add[(layer, "vAll")] += add_vn
        add[(layer, "nVAll")] += vn.numel()
        h = oldest.mean(dim=(0, 2))                              # (H,)
        self.head_sums[layer] = self.head_sums.get(layer, 0) + h
        # mass by relative offset t-kv, so a sink at the window edge shows as a spike at d=w
        d = (t[:, None] - torch.arange(w + 1, device=q.device))
        prof = p.gather(-1, d.clamp_min(0).view(1, 1, nq, w + 1).expand(B, H, nq, w + 1))
        prof = prof * (d >= 0).view(1, 1, nq, w + 1)
        self.profiles[layer] = self.profiles.get(layer, 0) + prof.sum(dim=(0, 1, 2)) / (B * H * nq)

    def install(self):
        self._real = fa.flash_attn.flash_attn_func
        fa.flash_attn.flash_attn_func = self

    def remove(self):
        fa.flash_attn.flash_attn_func = self._real

    def report(self, window_pattern):
        s, n_layer = self.sums, self.n_layer
        print0(f"\nquery positions >= {self.min_pos}, {self.count} batches, "
               f"{int(s[(0, 'n')])} (batch,head,query) samples per layer")
        print0(f"{'layer':>5} {'win':>5} {'unif':>8} {'oldest':>8} {'x unif':>7} "
               f"{'oldest8':>8} {'pos0':>8} {'bos':>8} {'bosN':>11} {'bos|in':>7} "
               f"{'H yes/no':>11} {'|v|bos':>7} {'recent64':>9} {'top@off':>8} {'maxhead':>8}")
        for i in range(n_layer):
            n = s[(i, "n")]
            u = s[(i, "unif")] / n
            old = s[(i, "oldest")] / n
            head = (self.head_sums[i] / self.count).max().item()
            kind = window_pattern[i % len(window_pattern)] if i < n_layer - 1 else "L"
            print0(f"{i:>4}{kind} {self.windows[i]:>5} {u:>8.5f} {old:>8.5f} {old / u:>7.1f} "
                   f"{s[(i, 'oldest8')] / n:>8.5f} {s[(i, 'pos0')] / n:>8.5f} "
                   f"{s[(i, 'bos')] / n:>8.5f} "
                   f"{s[(i, 'bosN')] / s[(i, 'nq')]:>5.2f}/{s[(i, 'bosAny')] / s[(i, 'nq')]:>5.2f} "
                   f"{s[(i, 'bosCond')] / max(s[(i, 'nCond')], 1):>7.3f} "
                   f"{s[(i, 'entYes')] / max(s[(i, 'nCond')], 1):>5.2f}/"
                   f"{s[(i, 'entNo')] / max(s[(i, 'nNo')], 1):>5.2f} "
                   f"{(s[(i, 'vBos')] / s[(i, 'nVBos')]) / (s[(i, 'vAll')] / s[(i, 'nVAll')]):>7.2f} "
                   f"{s[(i, 'recent64')] / n:>9.5f} "
                   f"{s[(i, 'topmass')] / n:>4.2f}@{s[(i, 'topoff')] / n:>3.0f} {head:>8.5f}")
        print0("\nmass by relative offset (d = query - key), top 6 offsets per layer:")
        for i in range(n_layer):
            prof = self.profiles[i] / self.count
            vals, offs = prof.topk(6)
            edge = prof[-1].item()
            pairs = " ".join(f"d={o.item()}:{v.item():.4f}" for v, o in zip(vals, offs))
            print0(f"{i:>4}  {pairs}   [edge d={self.windows[i]}: {edge:.5f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=str, default="base", help="base|sft|rl")
    ap.add_argument("--model-tag", type=str, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--split", type=str, default="val", help="train|val")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--skip-batches", type=int, default=24,
                    help="batches to discard first. The best-fit packer starts with an empty "
                         "buffer, so early rows are packed from few candidates and hold one long "
                         "document each; BOS-in-window climbs from ~7%% to ~65%% as it fills")
    ap.add_argument("--device-batch-size", type=int, default=8)
    ap.add_argument("--num-queries", type=int, default=128, help="query positions scored per layer")
    ap.add_argument("--min-pos", type=int, default=-1,
                    help="lowest query position to score; default = the S-layer window (seq_len//4 if all-L)")
    ap.add_argument("--device-type", type=str, default="")
    args = ap.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    cfg = meta["model_config"]
    seq_len, n_layer = cfg["sequence_len"], cfg["n_layer"]
    windows = model.window_sizes
    short = min(w for w, _ in windows)
    # an all-L model has no short window, so fall back to a quarter of the sequence
    min_pos = min(short, seq_len // 4) if args.min_pos < 0 else args.min_pos
    print0(f"model {cfg['n_layer']}L window_pattern {cfg['window_pattern']} "
           f"seq_len {seq_len} val_bpb {meta['val_bpb']} step {meta['step']}")

    rec = SinkRecorder(n_layer, args.num_queries, min_pos, tokenizer.get_bos_token_id())
    rec.install()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, seq_len, args.split, device=device)
    try:
        for _ in range(args.skip_batches):
            next(loader)
        for _ in range(args.batches):
            x, y = next(loader)
            rec.ids = x
            rec.call = 0
            with torch.no_grad():
                model(x)
            rec.count += 1
    finally:
        rec.remove()
    rec.report(cfg["window_pattern"].upper())


if __name__ == "__main__":
    main()
