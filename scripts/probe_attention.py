"""Price the three ways this repo can realise causal attention, at one shape.

nanochat reaches attention through F.scaled_dot_product_attention, which has no window
parameter. So for a sliding window `_sdpa_attention` builds an explicit [Tq, Tk] bool mask --
and passing any mask drops SDPA off its flash backend onto the far slower math/mem-efficient
one. That is why the README's shortrun tables first recorded `--window-pattern L` as *faster*
than the SSSL default despite SSSL doing less work.

aten::_flash_attention_forward takes window_size_left and dispatches to the same pytorch_flash
kernels SDPA's fast path already uses, which is what `flash_attn_func` now does. Three arms:

    flash full causal    what --window-pattern L runs
    flash windowed       what --window-pattern SSSL runs on an S layer
    SDPA + explicit mask what an S layer used to run

Forward and forward+backward are timed separately because the backward is ~2.5x the forward
and does not shrink with the window as fast (it carries fixed dot_do_o and convert_dq passes).

Examples:

    python -m scripts.probe_attention                       # d12 training shape
    python -m scripts.probe_attention --heads 8 --window 512 # d16
    python -m scripts.probe_attention --window 1024
"""
import argparse

import torch
import torch.nn.functional as F


def fa(q, k, v, window, seq, scale):
    """(B, T, H, D) in and out. window < 0 means full causal."""
    return torch.ops.aten._flash_attention_forward(
        q, k, v, None, None, seq, seq, 0.0, True, False, scale=scale,
        window_size_left=(None if window < 0 else window),
        window_size_right=(None if window < 0 else 0))


def sdpa_masked(q, k, v, window):
    """The explicit-mask emulation, in SDPA's (B, H, T, D) layout."""
    seq = q.size(1)
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    row = torch.arange(seq, device=q.device).unsqueeze(1)
    col = torch.arange(seq, device=q.device).unsqueeze(0)
    mask = col <= row
    if window >= 0:
        mask = mask & ((row - col) <= window)
    return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask).transpose(1, 2)


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8, help="device batch size")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=6, help="6 at d12, 8 at d16")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=512, help="left window of an S layer (seq//4)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--soak", type=int, default=100)
    args = ap.parse_args()

    b, t, h, d = args.batch, args.seq, args.heads, args.head_dim
    scale = d ** -0.5
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    # Correctness first: a fast wrong kernel is not a result.
    for window in (-1, args.window):
        out = fa(q, k, v, window, t, scale)[0]
        ref = sdpa_masked(q, k, v, window)
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / ref.float().abs().max().item()
        label = "full causal" if window < 0 else f"window {window}"
        print(f"{label:>16}: max abs err {err:.3e} vs masked SDPA (rel {rel:.3e})")

    qg, kg, vg = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    dout = torch.randn_like(q)

    def fwd_bwd(window):
        out, lse, rng, unused, _ = fa(qg, kg, vg, window, t, scale)
        torch.ops.aten._flash_attention_backward(
            dout, qg, kg, vg, out, lse, None, None, t, t, 0.0, True, rng, unused, scale=scale,
            window_size_left=(None if window < 0 else window),
            window_size_right=(None if window < 0 else 0))

    for _ in range(args.soak):
        fa(q, k, v, -1, t, scale)
    torch.cuda.synchronize()

    full_f = timeit(lambda: fa(q, k, v, -1, t, scale), args.iters)
    win_f = timeit(lambda: fa(q, k, v, args.window, t, scale), args.iters)
    full_fb = timeit(lambda: fwd_bwd(-1), args.iters)
    win_fb = timeit(lambda: fwd_bwd(args.window), args.iters)
    mask_f = timeit(lambda: sdpa_masked(q, k, v, args.window), max(5, args.iters // 4), warmup=2)

    print(f"\nB={b} T={t} H={h} D={d}, window {args.window}")
    print(f"{'ms':>8} {'vs full':>8}  arm")
    print(f"{full_f:8.3f} {1.0:8.3f}  flash fwd, full causal")
    print(f"{win_f:8.3f} {win_f/full_f:8.3f}  flash fwd, windowed")
    print(f"{full_fb:8.3f} {1.0:8.3f}  flash fwd+bwd, full causal")
    print(f"{win_fb:8.3f} {win_fb/full_fb:8.3f}  flash fwd+bwd, windowed")
    print(f"{mask_f:8.3f} {mask_f/full_f:8.3f}  SDPA fwd, explicit mask (the old windowed path)")


if __name__ == "__main__":
    main()
