"""Microbenchmark: bf16 vs FP8 vs NVFP4 linear layers at nanochat shapes.

Run: python -m scripts.nvfp4_bench [--m 16384] [--compile] [--depths 12,20,24]
"""

import argparse

import torch
from triton.testing import do_bench

from nanochat.gpt import Linear
from nanochat.fp8 import Float8Linear
from nanochat.nvfp4 import NVFP4Linear


def shapes_for_depth(depth):
    d = ((depth * 64 + 127) // 128) * 128
    return [("attn", d, d), ("mlp_fc", d, 4 * d), ("mlp_proj", 4 * d, d), ("lm_head", d, 32768)]


def bench_linear(cls, K, N, M, compiled, fwd_only):
    if compiled:
        torch._dynamo.reset()  # each layer is its own compile unit
    lin = cls(K, N, bias=False, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        lin.weight.normal_(std=K ** -0.5)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=not fwd_only)
    fn = lin
    if compiled:
        fn = torch.compile(lambda t: lin(t), fullgraph=True, dynamic=False)
    if fwd_only:
        with torch.no_grad():
            return do_bench(lambda: fn(x))
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    def step():
        y = fn(x)
        torch.autograd.grad(y, (x, lin.weight), dy)

    step()  # warm up compile/autotune outside timing
    return do_bench(step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=16384, help="tokens per microbatch (B*T)")
    p.add_argument("--depths", type=str, default="12,20,24")
    p.add_argument("--compile", action="store_true", help="torch.compile each layer")
    args = p.parse_args()

    impls = [("bf16", Linear), ("fp8", Float8Linear), ("nvfp4", NVFP4Linear)]
    for mode, flops_mult in [("fwd", 2), ("fwd+bwd", 6)]:
        print(f"\n=== {mode} | M={args.m} | compile={args.compile} ===")
        print(f"{'shape':<24} " + " ".join(f"{n:>14}" for n, _ in impls) + "   speedup vs bf16")
        for depth in [int(d) for d in args.depths.split(",")]:
            for name, K, N in shapes_for_depth(depth):
                times = []
                for _, cls in impls:
                    t = bench_linear(cls, K, N, args.m, args.compile, mode == "fwd")
                    times.append(t)
                fl = flops_mult * args.m * K * N
                cells = " ".join(f"{t*1e3:7.0f}us{fl/t/1e9:5.0f}T" for t in times)
                sp = " ".join(f"{times[0]/t:.2f}x" for t in times[1:])
                print(f"d{depth} {name:<20} {cells}   {sp}")


if __name__ == "__main__":
    main()
