"""Does a Hadamard rotation help the NVFP4 *forward*? Queue B0 in dev/nvfp4-quartet.md.

    NANOCHAT_BASE_DIR=/remote/.nanochat-cache .venv/bin/python -m scripts.probe_fwd_rht

B0 proposed rotating both forward operands so the rotations cancel across the GEMM --
`(X H)(H^t W^t) = X W^t` -- on the argument that flattening outliers helps the 16-element block
scales. This measures that directly: load a checkpoint, capture (x, W) for every convertible
Linear on one real batch, and compare the effective bits of the fp4 forward GEMM with and
without the rotation, against the fp32 product of the same operands.

Two rotation families, because they answer different questions:

  kernel   `transform_rht128` + `quant_fp4` (what `--nvfp4-fwd-rht four-six` would run) and
           `rht128_quant_eden` (what B0's row proposed reusing). Both are 128-wide.
  width    the same rotation in torch at 16/32/64/128, which is the granularity sweep. NVFP4
           scales per 16 elements, so 16 is the width that flattens *within* a block instead
           of smearing across eight of them.
  control  `quant_fp4` in EDEN mode with no rotation, so `kernel-eden` splits into what the
           stochastic scale costs and what the rotation costs.

This is a forward-error probe, not a bpb claim -- but a lever that loses effective bits on
every layer here cannot be recovering the -0.0132 that *Numerics: C8* priced.
"""
import argparse
import math
import statistics

import torch

from nanochat.checkpoint_manager import build_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import Linear
from nanochat.sm120.nvfp4 import fp4_mm, is_nvfp4_convertible
from nanochat.sm120.quartet.quant import NVFP4QuantMode, new_seed, quant_fp4, rht128_quant_eden
from nanochat.sm120.quartet.rht import hadamard_matrix, transform_rht128

FOUR_SIX = NVFP4QuantMode.FOUR_SIX


def eff_bits(res, ref):
    """Relative MSE as bits, the metric the rest of this stack is ranked in."""
    return -math.log2(((res.float() - ref.float()).pow(2).mean() / ref.float().pow(2).mean()).item()) / 2


def quantized_gemm(x, w):
    """The fp4 forward as NVFP4Linear runs it: 4/6 both operands, per-tensor scale outside."""
    xq = quant_fp4(x, scale_override=1.0, mode=FOUR_SIX)
    wq = quant_fp4(w, scale_override=1.0, mode=FOUR_SIX)
    return fp4_mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales, xq.tensor_scale * wq.tensor_scale)


def rotate_torch(t, n, h):
    """`t h^t` blockwise along the last dim, in torch, so the output stays natural-order."""
    rows, cols = t.shape
    return (t.float().view(rows, cols // n, n) @ h.T).reshape(rows, cols).to(torch.bfloat16).contiguous()


def capture(model, x, y):
    """One forward, keeping each convertible Linear's 2-D bf16 input."""
    saved, handles = {}, []

    def hook(name):
        def fwd(mod, inputs, out):
            t = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.bfloat16)
            saved[name] = t[: t.shape[0] // 128 * 128].contiguous()
        return fwd

    for name, mod in model.named_modules():
        if isinstance(mod, Linear) and is_nvfp4_convertible(mod):
            handles.append(mod.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(x, y)
    for h in handles:
        h.remove()
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="/remote/.nanochat-cache/base_checkpoints/c2-fp8-s42")
    p.add_argument("--step", type=int, default=2520)
    p.add_argument("--device-batch-size", type=int, default=4, help="rows of sequence_len tokens; 4 = 8192 tokens at d12")
    p.add_argument("--widths", default="16,32,64,128", help="torch-side rotation widths to sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-layer", action="store_true", help="print every layer, not just the summary and the worst")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    hads = {n: hadamard_matrix(n, torch.float32, "cuda") for n in widths}
    h128 = hadamard_matrix(128, torch.bfloat16, "cuda")
    h16 = h128[:16, :].contiguous()

    model, tokenizer, meta = build_model(args.checkpoint_dir, args.step, torch.device("cuda"), phase="train")
    model.train()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, meta["model_config"]["sequence_len"], split="val", device="cuda")
    x, y = next(loader)
    saved = capture(model, x, y)
    print(f"checkpoint {args.checkpoint_dir} step {args.step}: d{model.config.n_layer}, "
          f"{len(saved)} convertible Linears, batch {tuple(x.shape)}")

    cols = ["kernel-46", "kernel-eden", "eden-unrot"] + [f"torch-{n}" for n in widths]
    deltas = {c: [] for c in cols}
    rows = []
    weights = dict(model.named_modules())
    for name, xx in saved.items():
        w = weights[name].weight.detach().to(torch.bfloat16).contiguous()
        ref = xx.float() @ w.float().T
        base = eff_bits(quantized_gemm(xx, w), ref)

        got = {}
        # kernel path: the bf16 rotation NVFP4Linear would run, then today's 4/6 quantizer
        got["kernel-46"] = eff_bits(quantized_gemm(transform_rht128(h=h16, x=xx),
                                                   transform_rht128(h=h16, x=w)), ref)
        # kernel path: rotation fused into the EDEN quantizer, which is what B0's row proposed
        xe = rht128_quant_eden(x=xx, h=h16, scale_override=1.0)
        we = rht128_quant_eden(x=w, h=h16, scale_override=1.0)
        got["kernel-eden"] = eff_bits(fp4_mm(xe.fp4, we.fp4, xe.micro_scales, we.micro_scales,
                                             xe.tensor_scale * we.tensor_scale), ref)
        # control: EDEN's stochastic block scale without the rotation
        xu = quant_fp4(xx, scale_override=1.0, mode=NVFP4QuantMode.EDEN, seed=new_seed())
        wu = quant_fp4(w, scale_override=1.0, mode=NVFP4QuantMode.EDEN, seed=new_seed())
        got["eden-unrot"] = eff_bits(fp4_mm(xu.fp4, wu.fp4, xu.micro_scales, wu.micro_scales,
                                            xu.tensor_scale * wu.tensor_scale), ref)
        for n in widths:
            got[f"torch-{n}"] = eff_bits(quantized_gemm(rotate_torch(xx, n, hads[n]),
                                                        rotate_torch(w, n, hads[n])), ref)

        for c in cols:
            deltas[c].append(got[c] - base)
        ratio = (xx.float().abs().amax() / xx.float().pow(2).mean().sqrt()).item()
        rows.append((name, ratio, base, [got[c] - base for c in cols]))

    head = f"{'layer':<28}{'amax/rms':>9}{'plain':>8}" + "".join(f"{c:>13}" for c in cols)
    show = rows if args.per_layer else sorted(rows, key=lambda r: r[3][0])[:8]
    if not args.per_layer:
        print("\n== the eight layers the 128-wide kernel rotation hurts most (delta in bits, + is better)")
    else:
        print("\n== every layer (delta in bits against the unrotated forward, + is better)")
    print(head)
    for name, ratio, base, ds in show:
        print(f"{name:<28}{ratio:>9.1f}{base:>8.3f}" + "".join(f"{d:>+13.3f}" for d in ds))

    print(f"\n== mean over {len(rows)} layers")
    for c in cols:
        m = statistics.mean(deltas[c])
        worse = sum(1 for d in deltas[c] if d < 0)
        print(f"  {c:<12} {m:+.4f} bits   ({worse}/{len(rows)} layers worse)")


if __name__ == "__main__":
    main()
