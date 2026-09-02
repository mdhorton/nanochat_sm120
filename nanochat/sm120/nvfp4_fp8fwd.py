"""An fp8 forward with an NVFP4 backward: `--nvfp4-exclude-precision fp8-fwd`.

The numerics probe (dev/nvfp4-transformer-engine.md) says the forward is where fp4 noise costs
bpb and the backward barely matters; the lm_head timing says the backward is where two thirds
of its FLOPs are. Taking lm_head out of fp4 entirely gave up ~6% of tok/s at d12 whether the
replacement was bf16 or fp8, because an fp8 backward has to amax-scan, cast and transpose the
[tokens, vocab] gradient. This layer keeps the Quartet-II backward -- rotated, EDEN, from the
bf16 operands, so still unbiased -- and runs only the forward GEMM in tensorwise fp8. Measured at
the d12 lm_head shape, compiled: 17.9 ms per micro-step against 15.6 all-fp4, 22.1 fp8, 24.8 bf16.

The fp8 weight is cached per optimizer step like the fp4 one (`refresh_weight_cache`), so the
forward is one small activation cast and one `_scaled_mm`. Reuses NVFP4Linear's buffers, the
weight-cache hooks and the fused wgrad accumulator unchanged.
"""
import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.fp8 import _to_fp8
from nanochat.sm120 import fp4_gemm, nvfp4_numerics
from nanochat.sm120.nvfp4 import BACKWARD_SCALE_OVERRIDE, NVFP4Linear, _pad_rows, fp4_mm
from nanochat.sm120.quartet.rht import rerotate_hadamard

FP8 = torch.float8_e4m3fn


class _FP8FwdMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w_fp8, w_inv, had, scratch_amax, main_grad):
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
        x_fp8, x_inv = _to_fp8(x, FP8)
        if w_fp8 is None:
            w_fp8, w_inv = _to_fp8(weight, FP8)
        ctx.save_for_backward(x, weight, had, scratch_amax, main_grad)
        # B column-major, as _scaled_mm wants: a contiguous [N, K] transposed is [K, N] col-major.
        return torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_inv, scale_b=w_inv, out_dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, had, scratch_amax, main_grad = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        h16 = rerotate_hadamard(had)[:16, :]
        so = BACKWARD_SCALE_OVERRIDE
        # The rotated, bf16-source Quartet backward (nvfp4_numerics: rht=all, bwd_source=bf16).
        e_ht, wt_ht = nvfp4_numerics.dgrad_operands(
            grad_output, weight, None, None, None, None, None, h16, so, scratch_amax,
            rht="all", bwd_source="bf16")
        grad_input = fp4_mm(e_ht.fp4, wt_ht.fp4, e_ht.micro_scales, wt_ht.micro_scales,
                            e_ht.tensor_scale * wt_ht.tensor_scale)
        et_ht, xt_ht = nvfp4_numerics.wgrad_operands(
            grad_output, x, None, None, None, h16, so, scratch_amax, rht="all", bwd_source="bf16")
        alpha = et_ht.tensor_scale * xt_ht.tensor_scale
        if main_grad is not None:
            fp4_gemm.mm_accum(et_ht.fp4, xt_ht.fp4, et_ht.micro_scales, xt_ht.micro_scales, alpha, main_grad)
            grad_weight = None
        else:
            grad_weight = fp4_mm(et_ht.fp4, xt_ht.fp4, et_ht.micro_scales, xt_ht.micro_scales, alpha)
        return grad_input, grad_weight, None, None, None, None, None


class FP8FwdNVFP4Linear(NVFP4Linear):
    """Tensorwise-fp8 forward GEMM, Quartet-II NVFP4 backward from the bf16 operands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("fp8_w", "fp8_w_inv"):
            self.register_buffer(name, None, persistent=False)

    @torch.no_grad()
    def refresh_weight_cache(self):
        """The fp8 weight, once per optimizer step. Same ordering rule as the fp4 cache."""
        w_fp8, w_inv = _to_fp8(self.weight.detach(), FP8)
        if self.fp8_w is None:
            self.fp8_w = torch.empty_like(w_fp8)
            self.fp8_w_inv = torch.empty_like(w_inv)
        self.fp8_w.copy_(w_fp8)
        self.fp8_w_inv.copy_(w_inv)

    def forward(self, x):
        x = x.to(COMPUTE_DTYPE)
        orig_shape = x.shape
        x_2d, pad = _pad_rows(x.reshape(-1, orig_shape[-1]).contiguous())
        out = _FP8FwdMatmul.apply(x_2d, self.weight.to(torch.bfloat16), self.fp8_w, self.fp8_w_inv,
                                  self.had, self.scratch_amax, self.fp4_main_grad)
        if pad:
            out = out[: x_2d.shape[0] - pad]
        out = out.reshape(*orig_shape[:-1], out.shape[-1])
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out
