"""Numerics levers for the NVFP4 path, ported from NVIDIA's NVFP4 pretraining recipe.

`--nvfp4` loses +0.0137 bpb to `--fp8` at d12 ratio 12 (dev/nvfp4-quartet.md, C2). Transformer
Engine's recipe (arXiv 2509.25149, `NVFP4BlockScaling`) differs from the Quartet-II stack here in
four places, and each is a flag below so the probe can price them separately:

  --nvfp4-exclude / --nvfp4-bf16-blocks   NVIDIA keeps the output head and the first/last
                                          blocks in BF16. This repo quantizes lm_head.
  --nvfp4-exclude-precision fp8|fp8-fwd   run the excluded layers as Float8Linear, or with an
                                          fp8 forward and the NVFP4 backward (nvfp4_fp8fwd.py):
                                          the forward is where the noise costs bpb, the backward
                                          is where lm_head's FLOPs are.
  --nvfp4-bwd-source bf16                 TE quantizes both orientations from the high-precision
                                          input "to avoid double quantization errors". Here the
                                          backward re-quantizes the *saved fp4* operands.
  --nvfp4-rht wgrad                       TE rotates wgrad operands only; NVIDIA's ablation says
                                          RHT on dgrad degrades quality. Here all four backward
                                          operands are rotated.
  --nvfp4-weight-2d                       TE scales weights per 16x16 block so fprop's W and
                                          dgrad's W^T are the same numbers. Here W^T is a fresh
                                          quantization along N.

Every flag needs --nvfp4 and defaults to today's behaviour.

**One kernel family per GEMM.** The Quartet kernels do not all write the contraction dim in
natural order: `rht128_quant_eden` / `rht128_requant` store each rotated 128-block in a fixed
permuted group order (`had128_utils.cuh`), `quant_had_eden` / `dequant_tp_had_eden` use another,
and `quant_fp4` plus the torch quantizer below are natural. Two operands from different families
give a silently wrong product, not an error. So a rotated GEMM takes both operands from the
rht128 kernels, and an unrotated one takes both from `quant_fp4` or the 2D cache.
"""
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.sm120.quartet.quant import (
    NVFP4Quant,
    NVFP4QuantMode,
    new_seed,
    quant_fp4,
    rht128_quant_eden,
    rht128_requant,
)
from nanochat.sm120.quartet.reference import E2M1_LEVELS, e2m1_codes

RHT_CHOICES = ("all", "wgrad", "none")
BWD_SOURCE_CHOICES = ("fp4", "bf16")
EXCLUDE_PRECISION_CHOICES = ("bf16", "fp8", "fp8-fwd")
_BLOCK_RE = re.compile(r"^transformer\.h\.(\d+)\.")


@dataclass(frozen=True)
class NumericsConfig:
    exclude: frozenset = frozenset()      # Linear names kept in bf16, suffix-matched on the fqn
    bf16_blocks: tuple = (0, 0)           # (first N, last M) transformer blocks kept in bf16
    exclude_precision: str = "bf16"       # what the excluded layers run in: bf16 or fp8
    bwd_source: str = "fp4"               # what the backward requantizes W^T and X^T from
    rht: str = "all"                      # which backward GEMMs get the random Hadamard
    weight_2d: bool = False               # 16x16 weight block scales, shared by fprop and dgrad

    def validate(self):
        if self.rht not in RHT_CHOICES:
            raise ValueError(f"--nvfp4-rht must be one of {RHT_CHOICES}, got {self.rht!r}")
        if self.bwd_source not in BWD_SOURCE_CHOICES:
            raise ValueError(f"--nvfp4-bwd-source must be one of {BWD_SOURCE_CHOICES}, got {self.bwd_source!r}")
        if self.weight_2d and self.rht == "all":
            raise ValueError("--nvfp4-weight-2d needs --nvfp4-rht wgrad or none: the 2D weight is "
                             "shared with the forward, so dgrad cannot rotate it")
        if self.rht == "none" and self.bwd_source != "bf16":
            raise ValueError("--nvfp4-rht none needs --nvfp4-bwd-source bf16: an unrotated X^T "
                             "cannot be produced from the saved fp4 activation")
        if len(self.bf16_blocks) != 2 or min(self.bf16_blocks) < 0:
            raise ValueError(f"--nvfp4-bf16-blocks wants FIRST,LAST >= 0, got {self.bf16_blocks}")
        if self.exclude_precision not in EXCLUDE_PRECISION_CHOICES:
            raise ValueError(f"--nvfp4-exclude-precision must be one of {EXCLUDE_PRECISION_CHOICES}, "
                             f"got {self.exclude_precision!r}")
        if self.exclude_precision != "bf16" and not self.excludes_anything:
            raise ValueError(f"--nvfp4-exclude-precision {self.exclude_precision} needs --nvfp4-exclude "
                             "or --nvfp4-bf16-blocks: nothing is excluded")
        return self

    @property
    def excludes_anything(self):
        return bool(self.exclude) or any(self.bf16_blocks)

    @property
    def rotates_dgrad(self):
        return self.rht == "all"

    @property
    def rotates_wgrad(self):
        return self.rht != "none"


DEFAULT = NumericsConfig()

_DEFAULTS = {
    "nvfp4_exclude": "",
    "nvfp4_bf16_blocks": "0,0",
    "nvfp4_exclude_precision": "bf16",
    "nvfp4_bwd_source": "fp4",
    "nvfp4_rht": "all",
    "nvfp4_weight_2d": False,
}


def add_args(group):
    """The numerics flags. Registered from recipe.add_args so they sit with the rest of the stack."""
    group.add_argument("--nvfp4-exclude", type=str, default=_DEFAULTS["nvfp4_exclude"],
                       help="comma-separated Linear names to keep in bf16 under --nvfp4, suffix-matched "
                            "against the module fqn (e.g. 'lm_head', 'mlp.c_proj'). NVIDIA's NVFP4 "
                            "recipe keeps the output head in high precision")
    group.add_argument("--nvfp4-bf16-blocks", type=str, default=_DEFAULTS["nvfp4_bf16_blocks"],
                       help="FIRST,LAST: keep every Linear of the first FIRST and last LAST transformer "
                            "blocks out of NVFP4, in the --nvfp4-exclude-precision (NVIDIA: first 2 and last "
                            "4-8 of 64; the d12 probe says the last block matters and the first does not)")
    group.add_argument("--nvfp4-exclude-precision", type=str, default=_DEFAULTS["nvfp4_exclude_precision"],
                       choices=EXCLUDE_PRECISION_CHOICES,
                       help="what --nvfp4-exclude / --nvfp4-bf16-blocks layers run in. 'fp8' converts them "
                            "to Float8Linear (tensorwise). 'fp8-fwd' keeps the NVFP4 backward (from the bf16 "
                            "operands) under a tensorwise-fp8 forward: at the d12 lm_head shape 17.9 ms per "
                            "micro-step against 15.6 all-fp4, 22.1 fp8, 24.8 bf16")
    group.add_argument("--nvfp4-bwd-source", type=str, default=_DEFAULTS["nvfp4_bwd_source"],
                       choices=BWD_SOURCE_CHOICES,
                       help="where the backward's W^T and X^T come from: 'fp4' re-quantizes the saved "
                            "forward operands (Quartet-II), 'bf16' quantizes the high-precision tensors "
                            "(Transformer Engine). bf16 saves the activation for backward: +2 B/elem")
    group.add_argument("--nvfp4-rht", type=str, default=_DEFAULTS["nvfp4_rht"], choices=RHT_CHOICES,
                       help="which backward GEMMs get the random Hadamard: 'all' (Quartet-II), 'wgrad' "
                            "(Transformer Engine; dgrad operands quantize unrotated with EDEN), 'none'")
    group.add_argument("--nvfp4-weight-2d", action="store_true", default=_DEFAULTS["nvfp4_weight_2d"],
                       help="quantize weights with one E4M3 scale per 16x16 block (Transformer Engine) so "
                            "the forward and dgrad use the same quantized weight. Needs the weight cache "
                            "and --nvfp4-rht wgrad/none. Higher forward error, consistent gradient")


def orphan_flags(args):
    """The numerics flags given a non-default value, for base_train's without-`--nvfp4` error."""
    return [name for name, default in _DEFAULTS.items() if getattr(args, name) != default]


def from_args(args):
    """Parse and validate the numerics flags. Call only when --nvfp4 is on."""
    exclude = frozenset(n.strip() for n in args.nvfp4_exclude.split(",") if n.strip())
    try:
        first, last = (int(v) for v in args.nvfp4_bf16_blocks.split(","))
    except ValueError:
        raise ValueError(f"--nvfp4-bf16-blocks wants FIRST,LAST, got {args.nvfp4_bf16_blocks!r}") from None
    cfg = NumericsConfig(exclude=exclude, bf16_blocks=(first, last),
                         exclude_precision=args.nvfp4_exclude_precision,
                         bwd_source=args.nvfp4_bwd_source, rht=args.nvfp4_rht,
                         weight_2d=args.nvfp4_weight_2d).validate()
    if cfg.weight_2d and not args.nvfp4_weight_cache:
        raise ValueError("--nvfp4-weight-2d needs --nvfp4-weight-cache: the 2D forms are built once "
                         "per optimizer step, in torch")
    return cfg


def module_filter(cfg, n_layer):
    """The `module_filter_fn` for convert_to_nvfp4_training: which Linears to leave in bf16."""
    first, last = cfg.bf16_blocks

    def keep(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if any(fqn == name or fqn.endswith("." + name) for name in cfg.exclude):
            return False
        m = _BLOCK_RE.match(fqn)
        if m is not None:
            i = int(m.group(1))
            if i < first or i >= n_layer - last:
                return False
        return True

    return keep


def convert_excluded_to_fp8(model, cfg, n_layer):
    """--nvfp4-exclude-precision fp8 / fp8-fwd: convert the Linears the filter kept out of NVFP4.

    Runs after convert_to_nvfp4_training and before torch.compile. Only layers the numerics
    filter excluded are touched -- the ones NVFP4 cannot take (features not 128-aligned) stay
    bf16, as they would under --fp8's own minimum-width rule. Returns the converted names.
    """
    from nanochat import sm120
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    from nanochat.sm120.nvfp4 import NVFP4Linear, is_nvfp4_convertible

    keep_fp4 = module_filter(cfg, n_layer)

    def excluded(mod, fqn):
        return (isinstance(mod, nn.Linear) and not isinstance(mod, NVFP4Linear)
                and is_nvfp4_convertible(mod) and not keep_fp4(mod, fqn))

    names = [n for n, m in model.named_modules() if excluded(m, n)]
    if cfg.exclude_precision == "fp8-fwd":
        from nanochat.sm120.nvfp4_fp8fwd import FP8FwdNVFP4Linear

        def _convert(mod, prefix=""):
            for name, child in mod.named_children():
                fqn = f"{prefix}.{name}" if prefix else name
                _convert(child, fqn)
                if excluded(child, fqn):
                    setattr(mod, name, FP8FwdNVFP4Linear.from_float(child, numerics=cfg))

        _convert(model)
        return names
    sm120.install_fp8_backend()   # before conversion: Float8Linear.__init__ registers its buffers there
    convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"),
                               module_filter_fn=excluded)
    return names


def describe(model, cfg):
    """One startup line: the numerics settings and every convertible Linear the filter left out."""
    from nanochat.sm120.nvfp4 import NVFP4Linear, is_nvfp4_convertible

    def out_of_fp4(m):
        return "FP8Fwd" in type(m).__name__ or not isinstance(m, NVFP4Linear)

    kept = [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and out_of_fp4(m) and is_nvfp4_convertible(m)]
    kinds = {type(m).__name__ for n, m in model.named_modules() if n in kept}
    kept_in = "fp8-fwd" if any("FP8Fwd" in k for k in kinds) else "fp8" if any("Float8" in k for k in kinds) else "bf16"
    blocks = sorted({int(m.group(1)) for n in kept if (m := _BLOCK_RE.match(n))})
    others = [n for n in kept if not _BLOCK_RE.match(n)]
    ranges, start = [], None
    for i, b in enumerate(blocks):
        if start is None:
            start = b
        if i + 1 == len(blocks) or blocks[i + 1] != b + 1:
            ranges.append(f"{start}-{b}" if b != start else str(b))
            start = None
    kept_note = ", ".join(others + ([f"blocks {','.join(ranges)}"] if ranges else [])) or "none"
    return (f"numerics: bwd-source {cfg.bwd_source}, rht {cfg.rht}, weight-2d "
            f"{'on' if cfg.weight_2d else 'off'}; kept in {kept_in}: {kept_note} ({len(kept)} Linears)")


# ---------------------------------------------------------------------------------------------
# 2D weight quantization, in torch
# ---------------------------------------------------------------------------------------------

def to_blocked(scales):
    """[rows, cols/16] block scales -> the cutlass 128x4 swizzle. Inverse of `nvfp4._unblock`."""
    rows, n = scales.shape
    assert rows % 128 == 0 and n % 4 == 0
    r = scales.reshape(rows // 128, 128, n // 4, 4).permute(0, 2, 1, 3)
    r = r.reshape(-1, 4, 32, 4).permute(0, 2, 1, 3)
    return r.reshape(rows, n).contiguous()


def _pack(codes):
    """4-bit codes [R, C] -> packed uint8 [R, C/2]; the even column is the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


@torch.no_grad()
def quantize_weight_2d(w, *, four_over_six=True, scale_override=1.0):
    """bf16 [N, K] -> row form for fprop, transposed form for dgrad, one scale per 16x16 block.

    Same two-level scheme as `quartet.reference.quantize_reference` (per-tensor scale at
    amax/(6*256), E4M3 block scale, 4/6 candidate choice by squared error) with the block widened
    from 1x16 to 16x16, which is what makes the transposed form the *same* quantization rather
    than a second one. The block scale is replicated across the 16 rows of its block for the
    GEMM, which reads one scale per row per 16 along K.

    Returns (fp4 [N, K/2], scales [N, K/16] swizzled, tensor_scale, fp4_t [K, N/2], scales_t
    [K, N/16] swizzled). Pure torch, so it belongs in the once-per-step weight cache refresh,
    not on a per-micro-step path.
    """
    n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0, "both dims must be 128-aligned (is_nvfp4_convertible)"
    x = w.to(torch.float32)
    amax = x.abs().amax()
    scales_max = 256.0 if four_over_six else 448.0
    tensor_scale = amax * scale_override / (6.0 * scales_max)
    tensor_scale = torch.where(amax == 0, torch.ones_like(tensor_scale), tensor_scale)

    blocks = x.view(n // 16, 16, k // 16, 16)                       # [nb, 16, kb, 16]
    bmax = blocks.abs().amax(dim=(1, 3), keepdim=True)              # [nb, 1, kb, 1]
    grid = torch.tensor(E2M1_LEVELS + [-v for v in E2M1_LEVELS], dtype=torch.float32, device=w.device)

    def candidate(val_max):
        s = ((bmax / val_max) / tensor_scale).to(torch.float8_e4m3fn).to(torch.float32)
        s = torch.where(s == 0, torch.ones_like(s), s)
        step = s * tensor_scale
        codes = e2m1_codes(blocks / step)
        err = (blocks - grid[codes.to(torch.long)] * step).pow(2).sum(dim=(1, 3), keepdim=True)
        return codes, s, err

    codes, s, err = candidate(6.0 / scale_override)
    if four_over_six:
        codes4, s4, err4 = candidate(4.0 / scale_override)
        take4 = err4 < err
        codes = torch.where(take4, codes4, codes)
        s = torch.where(take4, s4, s)

    fp4 = _pack(codes.reshape(n, k))
    scales = to_blocked(s.expand(n // 16, 16, k // 16, 1).reshape(n, k // 16)).to(torch.float8_e4m3fn)
    fp4_t = _pack(codes.permute(2, 3, 0, 1).reshape(k, n))
    scales_t = to_blocked(s.permute(2, 3, 0, 1).expand(k // 16, 16, n // 16, 1).reshape(k, n // 16))
    return fp4, scales, tensor_scale, fp4_t, scales_t.to(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------------------------
# Backward operands
# ---------------------------------------------------------------------------------------------

def _eden(x, so):
    """Unrotated EDEN quantization: natural-order family, stochastic scale, unbiased."""
    return quant_fp4(x, scale_override=so, mode=NVFP4QuantMode.EDEN, seed=new_seed())


def dgrad_operands(grad_output, weight, w_fp4, w_ms, w_ts, w_t, w_ts_t, h16, so, scratch_amax, *,
                   rht, bwd_source):
    """(E, W^T) for `grad_input = E @ W`, both from one kernel family.

    rht=all keeps Quartet-II's rotated pair; otherwise E is EDEN-quantized unrotated and W^T is
    the 2D cache when present (weight consistent with the forward, deterministic) or an EDEN
    quantization of the bf16 weight's transpose.
    """
    if rht == "all":
        e = rht128_quant_eden(x=grad_output, h=h16, scale_override=so, scratch_amax=scratch_amax)
        if bwd_source == "fp4":
            wt = rht128_requant(x=w_fp4, x_group_scales=w_ms, x_tensor_scale=w_ts, h=h16,
                                scale_override=so, scratch_amax=scratch_amax)
        else:
            wt = rht128_quant_eden(x=weight, h=h16, scale_override=so, transpose=True,
                                   scratch_amax=scratch_amax)
        return e, wt
    e = _eden(grad_output, so)
    if w_t is not None:
        wt = NVFP4Quant(w_t, w_ts_t, w_ts)
    else:
        wt = _eden(weight.T.contiguous(), so)
    return e, wt


def wgrad_operands(grad_output, x_bf16, x_fp4, x_ms, x_ts, h16, so, scratch_amax, *, rht, bwd_source):
    """(E^T, X^T) for `grad_weight = E^T @ X`, both from one kernel family."""
    if rht != "none":
        et = rht128_quant_eden(x=grad_output, h=h16, scale_override=so, transpose=True,
                               scratch_amax=scratch_amax)
        if bwd_source == "fp4":
            xt = rht128_requant(x=x_fp4, x_group_scales=x_ms, x_tensor_scale=x_ts, h=h16,
                                scale_override=so, scratch_amax=scratch_amax)
        else:
            xt = rht128_quant_eden(x=x_bf16, h=h16, scale_override=so, transpose=True,
                                   scratch_amax=scratch_amax)
        return et, xt
    return _eden(grad_output.T.contiguous(), so), _eden(x_bf16.T.contiguous(), so)
