"""Rank the NVFP4 numerics levers on a trained checkpoint, in minutes, before any bpb run.

    NANOCHAT_BASE_DIR=/remote/.nanochat-cache .venv/bin/python -m scripts.probe_nvfp4_numerics

Loads a base checkpoint (default: the d12 fp8 arm from dev/nvfp4-quartet.md C2) and one real
validation batch, runs the bf16 model as the reference, then each variant of the levers in
`nanochat/sm120/nvfp4_numerics.py` on the same batch. Three views, coarse to fine:

  model      loss delta against bf16 (the forward's noise, which is what the bpb deficit is
             made of) and the projection of every parameter gradient onto the bf16 gradient.
  layer      for each Linear of a few blocks plus lm_head, the captured (x, E) replayed through
             one NVFP4Linear per variant: effective bits of out / grad_in / grad_weight.
  ladder     the gradient averaged over N draws. An unbiased estimator gains a bit per 4x; a
             plateau is bias, which is the one thing averaging over a run cannot fix.

**This is an estimator probe, not a bpb claim.** It measures one batch of one checkpoint,
eager, with `_scaled_mm` for the GEMMs (bit-identical to the cuBLASLt launcher at every model
shape, per fp4_gemm.py). Its job is to say which levers are worth a horizon run, and to catch
a variant whose gradient points somewhere else -- the kernel-family mistake reads as a cosine
of ~0.1 here and as a divergence at step 200 in training.
"""
import argparse
import copy
import json
import math
import re
import time

import torch

from nanochat.checkpoint_manager import build_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import Linear
from nanochat.sm120 import nvfp4_numerics as nn4
from nanochat.sm120.nvfp4 import NVFP4Linear, convert_to_nvfp4_training, enable_weight_caches

# Selection variants keep layers in bf16; numerics variants change the estimator. `te-style` is
# the whole Transformer Engine recipe as far as this stack can express it.
VARIANTS = {
    "baseline": {},
    "fwd-only": {"disable_backward_quant": True},
    "exclude-lm_head": {"exclude": frozenset({"lm_head"})},
    "lm_head-fp8": {"exclude": frozenset({"lm_head"}), "exclude_precision": "fp8"},
    "lm_head-fp8fwd": {"exclude": frozenset({"lm_head"}), "exclude_precision": "fp8-fwd"},
    "fp8fwd+last1": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (0, 1), "exclude_precision": "fp8-fwd"},
    "fp8fwd+last2": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (0, 2), "exclude_precision": "fp8-fwd"},
    "fp8fwd+first1": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (1, 0), "exclude_precision": "fp8-fwd"},
    "fp8fwd+1+1": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (1, 1), "exclude_precision": "fp8-fwd"},
    "fp8fwd+2+4": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (2, 4), "exclude_precision": "fp8-fwd"},
    "fp8fwd+bwd-bf16": {"exclude": frozenset({"lm_head"}), "exclude_precision": "fp8-fwd", "bwd_source": "bf16"},
    "bf16-blocks-2+4": {"bf16_blocks": (2, 4)},
    "bwd-bf16": {"bwd_source": "bf16"},
    "rht-wgrad": {"rht": "wgrad"},
    "rht-none": {"rht": "none", "bwd_source": "bf16"},
    "weight-2d": {"weight_2d": True, "rht": "wgrad"},
    "lm_head+2d": {"exclude": frozenset({"lm_head"}), "weight_2d": True, "rht": "wgrad"},
    "lm_head+blocks": {"exclude": frozenset({"lm_head"}), "bf16_blocks": (2, 4)},
    "te-style-1d": {"rht": "wgrad", "bwd_source": "bf16",
                    "exclude": frozenset({"lm_head"}), "bf16_blocks": (2, 4)},
    "te-style": {"weight_2d": True, "rht": "wgrad", "bwd_source": "bf16",
                 "exclude": frozenset({"lm_head"}), "bf16_blocks": (2, 4)},
    "te-style-fp8": {"rht": "wgrad", "bwd_source": "bf16", "exclude_precision": "fp8",
                     "exclude": frozenset({"lm_head"}), "bf16_blocks": (2, 4)},
}
NUMERICS_KEYS = ("bwd_source", "rht", "weight_2d")
_BLOCK_RE = re.compile(r"^transformer\.h\.(\d+)\.(.*)$")


def rel_mse(res, ref):
    res, ref = res.float(), ref.float()
    return ((res - ref).pow(2).mean() / ref.pow(2).mean()).item()


def eff_bits(res, ref):
    e = rel_mse(res, ref)
    return float("inf") if e == 0 else -math.log2(e) / 2


def cosine(res, ref):
    res, ref = res.float().flatten(), ref.float().flatten()
    return ((res @ ref) / (ref @ ref)).item()


def split(spec):
    """A variant dict -> (NumericsConfig, disable_backward_quant)."""
    fields = {k: v for k, v in spec.items() if k != "disable_backward_quant"}
    return nn4.NumericsConfig(**fields).validate(), spec.get("disable_backward_quant", False)


def layer_key(name):
    m = _BLOCK_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, name)   # lm_head sorts last


def param_group(name):
    if name.startswith("lm_head"):
        return "lm_head"
    if _BLOCK_RE.match(name) and name.endswith(".weight"):
        return "matrices"
    return "other"


def reference_pass(model, x, y, capture):
    """bf16 forward/backward on the batch; returns loss, grads, and (x, E) for the captured Linears."""
    saved, handles = {}, []

    def hook(name):
        def fwd(mod, inputs, out):
            saved[name] = {"x": inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.bfloat16)}
            out.register_hook(lambda g: saved[name].__setitem__("e", g.detach().reshape(-1, g.shape[-1]).to(torch.bfloat16)))
        return fwd

    for name, mod in model.named_modules():
        if name in capture:
            handles.append(mod.register_forward_hook(hook(name)))
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    for h in handles:
        h.remove()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return loss.item(), grads, saved


def variant_pass(ref_model, spec, x, y, seed):
    """Convert a copy of the model per `spec`, run the same batch, return (loss, grads)."""
    cfg, fwd_only = split(spec)
    model = copy.deepcopy(ref_model)
    convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, model.config.n_layer),
                              numerics=cfg, disable_backward_quant=fwd_only)
    if cfg.exclude_precision != "bf16":
        nn4.convert_excluded_to_fp8(model, cfg, model.config.n_layer)
    enable_weight_caches(model)
    torch.manual_seed(seed)
    loss = model(x, y)
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    n_fp4 = sum(1 for m in model.modules() if type(m) is NVFP4Linear)   # fp8-fwd layers are not fp4 forwards
    del model
    torch.cuda.empty_cache()
    return loss.item(), grads, n_fp4


def grad_summary(grads, ref):
    """Cosine and effective bits of the flattened gradient, overall and per group."""
    out = {}
    for group in ("all", "matrices", "lm_head", "other"):
        names = [n for n in ref if group == "all" or param_group(n) == group]
        if not names or any(n not in grads for n in names):
            continue
        g = torch.cat([grads[n].flatten() for n in names])
        r = torch.cat([ref[n].flatten() for n in names])
        out[group] = (cosine(g, r), eff_bits(g, r))
    return out


def replay(linear, cfg, x, e, seed, draws=1):
    """One captured (x, E) through a fresh NVFP4Linear; grads averaged over `draws` backwards."""
    layer = NVFP4Linear.from_float(copy.deepcopy(linear), numerics=cfg)
    layer.refresh_weight_cache()
    torch.manual_seed(seed)
    out = None
    acc_x = acc_w = None
    for _ in range(draws):
        layer.weight.grad = None
        xx = x.clone().requires_grad_()
        out = layer(xx)
        out.backward(e)
        acc_x = xx.grad.float() if acc_x is None else acc_x + xx.grad.float()
        acc_w = layer.weight.grad.float() if acc_w is None else acc_w + layer.weight.grad.float()
    return out.detach(), acc_x / draws, acc_w / draws


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default="/remote/.nanochat-cache/base_checkpoints/c2-fp8-s42")
    p.add_argument("--step", type=int, default=2520)
    p.add_argument("--device-batch-size", type=int, default=4, help="rows of sequence_len tokens; 4 = 8192 tokens at d12")
    p.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of: " + ", ".join(VARIANTS))
    p.add_argument("--blocks", default="0,5,11", help="blocks whose Linears get the per-layer replay")
    p.add_argument("--draws", default="1,4,16,64", help="averaging ladder sizes")
    p.add_argument("--ladder-layers", default="mlp.c_fc,mlp.c_proj", help="Linears (within the middle --blocks entry) for the ladder")
    p.add_argument("--lt-gemm", action="store_true", help="route GEMMs through the cuBLASLt launcher (autotunes per shape; numerics identical)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="", help="write all numbers here")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    assert not unknown, f"unknown variants {unknown}; choose from {list(VARIANTS)}"
    if args.lt_gemm:
        from nanochat.sm120 import fp4_gemm
        fp4_gemm.preload()
        fp4_gemm.configure(True)

    t0 = time.time()
    model, tokenizer, meta = build_model(args.checkpoint_dir, args.step, device, phase="train")
    model.train()
    seq_len = meta["model_config"]["sequence_len"]
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, seq_len, split="val", device="cuda")
    x, y = next(loader)
    print(f"checkpoint {args.checkpoint_dir} step {args.step}: d{model.config.n_layer}, n_embd {model.config.n_embd}, "
          f"batch {tuple(x.shape)} ({x.numel()} tokens), loaded in {time.time() - t0:.1f}s")

    blocks = [int(b) for b in args.blocks.split(",")]
    capture = {f"transformer.h.{b}.{s}" for b in blocks
               for s in ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")} | {"lm_head"}
    loss_ref, grads_ref, saved = reference_pass(model, x, y, capture)
    print(f"bf16 reference: loss {loss_ref:.5f}, {len(grads_ref)} gradients, {len(saved)} layers captured")
    results = {"reference": {"loss": loss_ref}, "model": {}, "layer": {}, "ladder": {}}

    # -- model view --------------------------------------------------------------------------
    print("\n== model: loss delta vs bf16, and gradient projection onto the bf16 gradient")
    print(f"{'variant':<18}{'fp4 layers':>11}{'loss':>10}{'dloss':>10} | {'cos all':>8}{'bits':>6} | {'cos matrices':>13}{'bits':>6} | {'cos lm_head':>12}{'bits':>6}")
    for name in variants:
        loss, grads, n_fp4 = variant_pass(model, VARIANTS[name], x, y, args.seed)
        summ = grad_summary(grads, grads_ref)
        results["model"][name] = {"loss": loss, "dloss": loss - loss_ref, "fp4_layers": n_fp4,
                                  **{f"{g}_{k}": v for g, (c, b) in summ.items() for k, v in (("cos", c), ("bits", b))}}
        cells = " | ".join(f"{summ[g][0]:>{w}.4f}{summ[g][1]:>6.2f}" if g in summ else " " * (w + 6)
                           for g, w in (("all", 8), ("matrices", 13), ("lm_head", 12)))
        print(f"{name:<18}{n_fp4:>11}{loss:>10.5f}{loss - loss_ref:>+10.5f} | {cells}")

    # -- layer view --------------------------------------------------------------------------
    numerics_variants = [v for v in variants if not any(VARIANTS[v].get(k) for k in ("exclude", "bf16_blocks", "exclude_precision"))]
    print("\n== layer: effective bits of out / grad_in / grad_weight vs the fp32 product of the captured operands")
    header = f"{'layer':<24}" + "".join(f"{v:>22}" for v in numerics_variants)
    print(header)
    linears = dict(model.named_modules())
    for name in sorted(saved, key=layer_key):
        xs, es = saved[name]["x"], saved[name]["e"]
        w = linears[name].weight.detach().float()
        ref_out, ref_gx, ref_gw = xs.float() @ w.T, es.float() @ w, es.float().T @ xs.float()
        row, cells = {}, []
        for v in numerics_variants:
            cfg, fwd_only = split(VARIANTS[v])
            if fwd_only:
                cells.append(f"{'-':>22}")
                continue
            out, gx, gw = replay(linears[name], cfg, xs, es, args.seed)
            bits = (eff_bits(out, ref_out), eff_bits(gx, ref_gx), eff_bits(gw, ref_gw))
            row[v] = bits
            cells.append(f"{bits[0]:>7.2f}{bits[1]:>7.2f}{bits[2]:>8.2f}")
        results["layer"][name] = row
        print(f"{name:<24}" + "".join(cells))
    print("(each cell: fwd / dgrad / wgrad bits; +1 bit = 4x lower relative MSE)")

    # -- ladder ------------------------------------------------------------------------------
    draws = [int(d) for d in args.draws.split(",")]
    mid = blocks[len(blocks) // 2]
    ladder_layers = [f"transformer.h.{mid}.{s}" for s in args.ladder_layers.split(",")] + ["lm_head"]
    print(f"\n== ladder: gradient averaged over N draws ({draws}); an unbiased estimator gains +1 bit per 4x")
    for name in ladder_layers:
        if name not in saved:
            continue
        xs, es = saved[name]["x"], saved[name]["e"]
        w = linears[name].weight.detach().float()
        ref_gx, ref_gw = es.float() @ w, es.float().T @ xs.float()
        print(f"{name}:")
        print(f"  {'variant':<16}" + "".join(f"{'dgrad@' + str(d):>10}" for d in draws) + "   |" + "".join(f"{'wgrad@' + str(d):>10}" for d in draws) + "   per-4x")
        for v in numerics_variants:
            cfg, fwd_only = split(VARIANTS[v])
            if fwd_only:
                continue
            bx, bw = [], []
            for d in draws:
                _, gx, gw = replay(linears[name], cfg, xs, es, args.seed, draws=d)
                bx.append(eff_bits(gx, ref_gx))
                bw.append(eff_bits(gw, ref_gw))
            span = math.log(draws[-1] / draws[0], 4) if draws[-1] > draws[0] else 1.0
            slope = ((bx[-1] - bx[0]) / span, (bw[-1] - bw[0]) / span)
            flag = "" if min(slope) >= 0.5 else "  <- plateau (biased)"
            results["ladder"][f"{name}/{v}"] = {"dgrad": bx, "wgrad": bw, "slope": slope}
            print(f"  {v:<16}" + "".join(f"{b:>10.2f}" for b in bx) + "   |" + "".join(f"{b:>10.2f}" for b in bw)
                  + f"   {slope[0]:.2f}/{slope[1]:.2f}{flag}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
