"""
Micro-benchmark of the attention paths available on this GPU (fwd+bwd, bf16).

    python -m dev.bench_attention                       # d20-ish shape at T=2048
    python -m dev.bench_attention --seq-len 8192 --batch-size 2

Compares, per layer:
  - SDPA causal            : what full-context ("L") layers do today without FA3
  - SDPA + dense mask      : what sliding-window ("S") layers do today without FA3
  - flex causal            : FlexAttention on plain causal (expected to LOSE to SDPA)
  - flex sliding window    : FlexAttention with a sliding-window BlockMask (the Step A win)
"""
import time
import argparse

import torch
import torch.nn.functional as F

from nanochat.flex_attn import flex_attn_func, BlockMaskCache

parser = argparse.ArgumentParser(description="Benchmark attention implementations")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--n-head", type=int, default=10)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--window-div", type=int, default=4, help="short window = seq_len // this")
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

assert torch.cuda.is_available(), "this benchmark needs a CUDA GPU"
B, T, H, D = args.batch_size, args.seq_len, args.n_head, args.head_dim
window = T // args.window_div
device = "cuda"

q, k, v = [torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
grad_out = torch.randn_like(q)


def bench(fn, warmup=5):
    for _ in range(warmup):
        fn().backward(grad_out)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fn().backward(grad_out)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / args.iters * 1e3


def sdpa(mask=None, is_causal=False):
    def run():
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, is_causal=is_causal)
        return y.transpose(1, 2)
    return run


flex = torch.compile(flex_attn_func, dynamic=False)
bm_causal = BlockMaskCache([(T - 1, 0)], T, device).build()[0]
bm_slide = BlockMaskCache([(window, 0)], T, device).build()[0]

row = torch.arange(T, device=device).unsqueeze(1)
col = torch.arange(T, device=device).unsqueeze(0)
dense = (col <= row) & ((row - col) <= window)  # the mask nanochat builds today for S layers

results = {
    "SDPA causal (L layers today)": bench(sdpa(is_causal=True)),
    f"SDPA dense mask win={window} (S layers today)": bench(sdpa(mask=dense)),
    "flex causal": bench(lambda: flex(q, k, v, bm_causal)),
    f"flex sliding win={window}": bench(lambda: flex(q, k, v, bm_slide)),
}

print(f"\n{torch.cuda.get_device_name(0)} | B={B} H={H} T={T} D={D} bf16, fwd+bwd, {args.iters} iters")
baseline = results["SDPA causal (L layers today)"]
for name, ms in results.items():
    print(f"  {name:46s} {ms:7.2f} ms  {baseline / ms:5.2f}x")
print(f"\nblock sparsity: causal {bm_causal.sparsity():.1f}% empty, sliding {bm_slide.sparsity():.1f}% empty")
