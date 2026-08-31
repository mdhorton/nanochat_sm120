"""Microbenchmark the NVFP4 linear against bf16 (and fp8) over this model's real GEMM shapes.

    python -m scripts.bench_nvfp4 --depth 12 --tokens 16384

Shapes are taken from an actual GPT built on the meta device, so the inventory is the one
training sees rather than a guessed list.

**Read this before quoting a number from it.** Per `dev/perf-log.md`'s measurement protocol, a
fixed-tensor probe like this one is *not* reliable for end-to-end A/B timing on this box: the
card power-throttles, arms run at different thermal states, and a sub-2% delta is not evidence.
What it is good for is **kernel ranking** -- which shapes NVFP4 helps at and by roughly how
much. A real throughput claim needs a matched-step-index `base_train.py` comparison.

The per-shape speedup here is also an upper bound on what the model gains: it prices the GEMMs
only, and a training step is not all GEMM.

Layers are `torch.compile`d by default, because training compiles the model and the quantized
arms live or die on whether their cast glue fuses into the surrounding ops. Measured eager
(`--eager`), *both* fp8 and NVFP4 come out several times slower than bf16 at these shapes --
that is the unfused glue, not the tensor cores, and it is not the regime that ships.

**`--backward` got the answer wrong once already.** It scored NVFP4 at 0.63x of bf16 over the
d12 inventory, while a matched `base_train.py` batch measured NVFP4 *faster* than both bf16 and
fp8 at the same depth. A probe that inverts the ranking is worse than no probe, so treat
`--backward` as unproven and settle throughput questions with `base_train.py` arms instead.
Forward-only rankings did match. See `dev/nvfp4-quartet.md`.
"""
import argparse
import time

import torch

from nanochat.gpt import GPT, GPTConfig, Linear


def bench(fn, iters=50, warmup=10):
    """Median-of-iters wall time in ms. Median, not mean: a stray context switch is one-sided."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def linear_inventory(depth, aspect_ratio, head_dim, vocab_size, seq_len):
    """Every distinct (in_features, out_features) the model's Linear layers use, with counts."""
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                       n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim)
    with torch.device("meta"):
        model = GPT(config)
    shapes = {}
    for name, mod in model.named_modules():
        if isinstance(mod, Linear):
            key = (mod.in_features, mod.out_features)
            shapes.setdefault(key, [0, name])
            shapes[key][0] += 1
    return [(k, v[0], v[1]) for k, v in shapes.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=12, help="model depth, to derive the GEMM shapes")
    p.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    p.add_argument("--head-dim", type=int, default=128, help="target head dimension")
    p.add_argument("--vocab-size", type=int, default=32768, help="tokenizer vocab size (this repo's tokenizer)")
    p.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
    p.add_argument("--tokens", type=int, default=16384, help="rows in the GEMM (device_batch_size * seq_len)")
    p.add_argument("--iters", type=int, default=50, help="timed iterations per arm")
    p.add_argument("--fp8", action="store_true", help="also bench the fp8 linear")
    p.add_argument("--backward", action="store_true", help="time forward+backward (see the module docstring: this mode did NOT reproduce the end-to-end ranking)")
    p.add_argument("--eager", action="store_true", help="skip torch.compile (training compiles, so this is not the training regime)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")

    from nanochat.sm120.nvfp4 import ALIGN, NVFP4Linear
    from nanochat.sm120.quartet.ext import unavailable_reason

    reason = unavailable_reason()
    if reason is not None:
        raise SystemExit(f"NVFP4 kernels unavailable: {reason}")

    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"tokens per GEMM: {args.tokens}, {'forward+backward' if args.backward else 'forward only'}\n")

    inventory = linear_inventory(args.depth, args.aspect_ratio, args.head_dim,
                                 args.vocab_size, args.max_seq_len)

    header = f"{'shape (in->out)':>20} {'n':>3} {'bf16 ms':>9} {'nvfp4 ms':>9} {'speedup':>8}"
    if args.fp8:
        header += f" {'fp8 ms':>8} {'vs fp8':>8}"
    print(header)
    print("-" * len(header))

    totals = {"bf16": 0.0, "nvfp4": 0.0, "fp8": 0.0}
    for (in_f, out_f), count, example in sorted(inventory, key=lambda s: -s[1]):
        x = torch.randn(args.tokens, in_f, device="cuda", dtype=torch.bfloat16,
                        requires_grad=args.backward)

        def make(cls, **kw):
            layer = cls(in_f, out_f, bias=False, device="cuda", dtype=torch.float32, **kw)
            fwd = layer if args.eager else torch.compile(layer)
            if not args.backward:
                return lambda: fwd(x)
            def step():
                fwd(x).sum().backward()
                layer.zero_grad(set_to_none=True)
                x.grad = None
            return step

        t_bf16 = bench(make(Linear), iters=args.iters)
        row = f"{f'{in_f}->{out_f}':>20} {count:>3} {t_bf16:>9.3f}"
        totals["bf16"] += t_bf16 * count

        if in_f % ALIGN == 0 and out_f % ALIGN == 0:
            t_fp4 = bench(make(NVFP4Linear), iters=args.iters)
            row += f" {t_fp4:>9.3f} {t_bf16 / t_fp4:>7.2f}x"
            totals["nvfp4"] += t_fp4 * count
        else:
            row += f" {'skipped':>9} {'-':>8}"
            totals["nvfp4"] += t_bf16 * count   # it would run in bf16

        if args.fp8:
            from nanochat.fp8 import Float8Linear
            # base_train's own fp8 filter: _scaled_mm needs dims divisible by 16, and the
            # tiny gate layers are skipped there too.
            if in_f % 16 == 0 and out_f % 16 == 0 and min(in_f, out_f) >= 128:
                t_fp8 = bench(make(Float8Linear), iters=args.iters)
                row += f" {t_fp8:>8.3f} {t_bf16 / t_fp8:>7.2f}x"
                totals["fp8"] += t_fp8 * count
            else:
                row += f" {'skipped':>8} {'-':>8}"
                totals["fp8"] += t_bf16 * count
        print(row)

    print("-" * len(header))
    print(f"{'model total':>20} {'':>3} {totals['bf16']:>9.3f} {totals['nvfp4']:>9.3f} "
          f"{totals['bf16'] / totals['nvfp4']:>7.2f}x"
          + (f" {totals['fp8']:>8.3f} {totals['bf16'] / totals['fp8']:>7.2f}x" if args.fp8 else ""))
    print("\nGEMM inventory only -- a training step is not all GEMM, and this probe cannot")
    print("resolve small deltas on a throttling card. See dev/perf-log.md before quoting it.")



if __name__ == "__main__":
    main()
