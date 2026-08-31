"""Autotune a cuBLASLt algorithm for every NVFP4 GEMM this model launches, against `_scaled_mm`.

    .venv/bin/python -m scripts.probe_fp4_gemm --depth 20 --tokens 8192 --grad-accum 16

Queue item A1 in `dev/nvfp4-quartet.md`. The fp4 GEMM row is 990.1 ms/step (27.2%) at d20/dbs4
and ncu puts the kernel at 66.3% SM / 19.9% DRAM, so the question this answers is whether any of
that 34 points of unused compute roofline is reachable by *algorithm selection* -- cuBLASLt has
more than one block-scaled fp4 kernel, and `torch._scaled_mm` takes the heuristic's first pick
with a 1 MiB workspace.

The inventory is the real one: every distinct Linear shape from a meta-device GPT, times the
three GEMMs each of them runs, times `grad_accum` micro-steps -- which reproduces the 5,808
launches/step the profile counted. Operands are produced by the actual Quartet quantizer, so the
block scales carry the cutlass 128x4 swizzle the GEMM indexes them through.

**Per `dev/perf-log.md`, a fixed-tensor probe is not an end-to-end throughput claim.** Both arms
here are timed back to back on the same warm card at the same shape, which makes the per-shape
ratio trustworthy; the projected step saving at the bottom is an upper bound on what a training
A/B would show, because a step is not all GEMM.
"""
import argparse

import torch

from nanochat.gpt import GPT, GPTConfig, Linear

TORCH_WORKSPACE = 1024 * 1024   # what torch queries the fp4 heuristic with, per its own trace


def linear_inventory(depth, aspect_ratio, head_dim, vocab_size, seq_len):
    """Every distinct (in_features, out_features) the model's Linear layers use, with counts."""
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                       n_head=model_dim // head_dim, n_kv_head=model_dim // head_dim,
                       n_embd=model_dim)
    with torch.device("meta"):
        model = GPT(config)
    shapes = {}
    for _, mod in model.named_modules():
        if isinstance(mod, Linear):
            key = (mod.in_features, mod.out_features)
            shapes[key] = shapes.get(key, 0) + 1
    return sorted(shapes.items())


def ab_time(fns, rounds=5, iters=20):
    """Median us/call for each fn, alternating the order every round.

    The first version of this probe timed `_scaled_mm` once, after the plan's autotune -- i.e.
    always at the hottest point of the shape's soak, which read one shape 15% in the plan's
    favour that a fresh process puts at 3%. Same reason the autotune itself scores round-robin.
    """
    times = [[] for _ in fns]
    for r in range(rounds):
        order = range(len(fns)) if r % 2 == 0 else reversed(range(len(fns)))
        for i in order:
            for _ in range(3):
                fns[i]()
            t0, t1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            t0.record()
            for _ in range(iters):
                fns[i]()
            t1.record()
            t1.synchronize()
            times[i].append(t0.elapsed_time(t1) * 1000.0 / iters)
    return [sorted(t)[len(t) // 2] for t in times]


def gemms(in_f, out_f, tokens):
    """The three (m, n, k) an NVFP4Linear launches. All three are TN: k is the contiguous dim.

    fwd   out[T, N]  = x[T, K] . w[N, K]^T
    dgrad grad_x[T, K] = e[T, N] . w_ht[K, N]^T
    wgrad grad_w[N, K] = e_t[N, T] . x_t[K, T]^T
    """
    return [("fwd", tokens, out_f, in_f),
            ("dgrad", tokens, in_f, out_f),
            ("wgrad", out_f, in_f, tokens)]


def make_operands(m, n, k, mode):
    """Real NVFP4 operands for a [m, k] x [n, k]^T product, from the Quartet quantizer."""
    from nanochat.sm120.quartet.quant import quant_fp4

    a = quant_fp4(torch.randn(m, k, device="cuda", dtype=torch.bfloat16), scale_override=1.0, mode=mode)
    b = quant_fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16), scale_override=1.0, mode=mode)
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=20, help="model depth, to derive the GEMM shapes")
    p.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    p.add_argument("--head-dim", type=int, default=128, help="target head dimension")
    p.add_argument("--vocab-size", type=int, default=32768, help="tokenizer vocab size")
    p.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
    p.add_argument("--tokens", type=int, default=8192, help="rows in the GEMM (device_batch_size * seq_len)")
    p.add_argument("--grad-accum", type=int, default=16, help="micro-steps per step, to weight the projection")
    p.add_argument("--roles", default="fwd,dgrad,wgrad", help="comma-separated subset of the three GEMMs")
    p.add_argument("--shape", help="probe one m,n,k instead of the inventory, listing every candidate")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs CUDA")

    from nanochat.sm120 import fp4_gemm
    from nanochat.sm120.quartet.ext import unavailable_reason
    from nanochat.sm120.quartet.quant import NVFP4QuantMode

    reason = unavailable_reason()
    if reason is not None:
        raise SystemExit(f"NVFP4 kernels unavailable: {reason}")

    ext = fp4_gemm._ext()
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}  "
          f"cuBLASLt {ext.cublaslt_version()}")
    print(f"d{args.depth}, {args.tokens} tokens/micro-step, grad-accum {args.grad_accum}\n")

    if args.shape:
        m, n, k = (int(v) for v in args.shape.split(","))
        a, b = make_operands(m, n, k, NVFP4QuantMode.FOUR_SIX)
        for ws in (TORCH_WORKSPACE, ext.WORKSPACE_BYTES):
            print(f"\ncandidates at a {ws >> 20} MiB workspace budget "
                  f"({'what torch asks for' if ws == TORCH_WORKSPACE else 'what this plan asks for'}):")
            rows = ext.Fp4Gemm(m, n, k, ws).probe(a.fp4, b.fp4, a.micro_scales, b.micro_scales,
                                                  fp4_gemm._MAX_ALGOS, fp4_gemm._ITERS)
            print(f"  {'#':>2} {'algo':>5} {'tile':>5} {'stages':>7} {'splitK':>7} {'reduce':>7} "
                  f"{'cluster':>8} {'ws req':>9} {'waves':>6} {'us':>8}")
            for r in rows:
                print(f"  {int(r['index']):>2} {int(r['algo_id']):>5} {int(r['tile']):>5} "
                      f"{int(r['stages']):>7} {int(r['splitk']):>7} {int(r['reduction']):>7} "
                      f"{int(r['cluster']):>8} {int(r['ws_required']):>9} {r['waves']:>6.1f} "
                      f"{r['us']:>8.1f}")
        print(f"\n_scaled_mm: {fp4_gemm._time_scaled_mm(a.fp4, b.fp4, a.micro_scales, b.micro_scales):.1f} us")
        return

    roles = args.roles.split(",")
    inventory = linear_inventory(args.depth, args.aspect_ratio, args.head_dim,
                                 args.vocab_size, args.max_seq_len)

    header = (f"{'shape':>16} {'role':>6} {'m':>6} {'n':>6} {'k':>6} {'n/step':>7} "
              f"{'_scaled_mm':>11} {'1MiB':>8} {'32MiB':>8} {'vs_ref':>7} {'algo':>5} "
              f"{'tile':>5} {'splitK':>7}")
    print(header)
    print("-" * len(header))

    ref_total = small_total = pin_total = 0.0
    for (in_f, out_f), count in inventory:
        if in_f % 128 or out_f % 128:
            continue  # not NVFP4-convertible, stays bf16
        for role, m, n, k in gemms(in_f, out_f, args.tokens):
            if role not in roles:
                continue
            launches = count * args.grad_accum
            a, b = make_operands(m, n, k, NVFP4QuantMode.FOUR_SIX)
            operands = (a.fp4, b.fp4, a.micro_scales, b.micro_scales)
            budget = (fp4_gemm._MAX_ALGOS, fp4_gemm._ITERS, fp4_gemm._FINALISTS, fp4_gemm._ROUNDS)
            try:
                # Two plans, because the interesting axis turned out to be the workspace the
                # heuristic is queried with, not the algorithm: torch asks for 1 MiB, which is
                # under what the split-K variants need.
                small = ext.Fp4Gemm(m, n, k, TORCH_WORKSPACE)
                small.autotune(*operands, *budget)
                plan = ext.Fp4Gemm(m, n, k)
                info = plan.autotune(*operands, *budget)
                ref = fp4_gemm._fallback(*operands)
                got = plan.run(*operands)
                err = float((got.float() - ref.float()).abs().max()
                            / ref.abs().max().clamp(min=1e-6))
                # Alternating A/B/A rather than the autotune's own numbers: those were taken
                # during the soak, and `_scaled_mm` has to be timed in the same thermal state.
                ref_us, small_us, pin_us = ab_time([lambda: fp4_gemm._fallback(*operands),
                                                    lambda: small.run(*operands),
                                                    lambda: plan.run(*operands)])
            except Exception as e:
                print(f"{f'{in_f}->{out_f}':>16} {role:>6} {m:>6} {n:>6} {k:>6} {launches:>7} "
                      f"FAILED: {type(e).__name__}: {e}")
                del a, b
                continue
            flag = "" if err < 0.05 else f"  !! rel err {err:.3g} vs _scaled_mm"
            print(f"{f'{in_f}->{out_f}':>16} {role:>6} {m:>6} {n:>6} {k:>6} {launches:>7} "
                  f"{ref_us:>10.1f}u {small_us:>7.1f}u {pin_us:>7.1f}u "
                  f"{ref_us / pin_us:>6.2f}x {int(info['algo_id']):>5} "
                  f"{int(info['tile']):>5} {int(info['splitk']):>7}{flag}")
            ref_total += ref_us * launches
            small_total += small_us * launches
            pin_total += pin_us * launches
            del a, b, plan, small, ref, got
            torch.cuda.empty_cache()

    print("-" * len(header))
    print(f"projected GEMM row: {ref_total / 1000:.1f} ms/step on _scaled_mm -> "
          f"{small_total / 1000:.1f} at 1 MiB -> {pin_total / 1000:.1f} pinned at 32 MiB "
          f"({ref_total / max(pin_total, 1e-9):.3f}x, {(ref_total - pin_total) / 1000:.1f} ms saved)")
    print("A step is not all GEMM: divide the saving by the step time for the throughput ceiling.")


if __name__ == "__main__":
    main()
