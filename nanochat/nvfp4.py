"""NVFP4 (Quartet II recipe) training linear for nanochat — sibling of fp8.py.

Forward: input and weight quantized to NVFP4 (RTN + 4/6 group-scale search),
one torch._scaled_mm. Only the 4-bit tensors are saved for backward.
Backward: saved tensors are dequantized, transposed, rotated with a fresh
random-sign Hadamard (128-block) along the contraction dim and re-quantized
with MS-EDEN + stochastically rounded E4M3 scales; both backward GEMMs run in
NVFP4. See nvfp4_kernels.py and https://arxiv.org/abs/2601.22813.

Limitations:
- requires bf16 activations, CUDA capability >= 10 (Blackwell), dims % 128
- checkpoints are identical to bf16 models (buffers non-persistent); downstream
  scripts (base_eval, chat_sft, chat_cli, engine) currently load them into
  plain bf16 Linears, i.e. bf16 inference of an NVFP4-trained model.
"""

import torch

from nanochat.gpt import Linear
from nanochat.nvfp4_kernels import (
    HAD_DIM, fp4_forward_quant, fp4_rht_eden_quant, fp4_rht_eden_quant_t,
    fp4_rht_eden_requant_t, fp4_dequant, nvfp4_mm, hadamard_matrix,
)


def _rerotate(had):
    # fresh random column signs each call; H stays orthonormal
    signs = torch.randint(0, 2, (HAD_DIM,), device=had.device, dtype=had.dtype) * 2 - 1
    return (had * signs[None, :]).t().contiguous()  # transposed: kernels compute x @ hmat


class _NVFP4Matmul(torch.autograd.Function):
    """out = x @ w.T with NVFP4 forward and (optionally) NVFP4 Quartet II backward."""

    @staticmethod
    def forward(ctx, x, w, had, four_over_six, bwd_quant):
        x_c, x_sc, x_s = fp4_forward_quant(x, four_over_six)
        w_c, w_sc, w_s = fp4_forward_quant(w, four_over_six)
        ctx.save_for_backward(x_c, x_sc, x_s, w_c, w_sc, w_s, had)
        ctx.bwd_quant = bwd_quant
        return nvfp4_mm(x_c, x_sc, x_s, w_c, w_sc, w_s)

    @staticmethod
    def backward(ctx, g):
        x_c, x_sc, x_s, w_c, w_sc, w_s, had = ctx.saved_tensors
        g = g.contiguous()
        if not ctx.bwd_quant:
            x_dq = fp4_dequant(x_c, x_sc, x_s)  # (M, K)
            w_dq = fp4_dequant(w_c, w_sc, w_s)  # (N, K)
        if not ctx.bwd_quant:
            return g @ w_dq, g.t() @ x_dq, None, None, None
        M, N = g.shape
        K = x_c.shape[1] * 2
        assert M % HAD_DIM == 0 and N % HAD_DIM == 0 and K % HAD_DIM == 0, \
            f"NVFP4 backward needs dims % {HAD_DIM}, got M={M} N={N} K={K}"
        # grad_x = g @ w = (g H1ᵀ)(wᵀ H1ᵀ)ᵀ, contraction over N
        h1 = _rerotate(had)
        a = fp4_rht_eden_quant(g, h1)
        b = fp4_rht_eden_requant_t(w_c, w_sc, w_s, h1)
        grad_x = nvfp4_mm(*a, *b)
        # grad_w = gᵀ @ x = (gᵀ H2ᵀ)(xᵀ H2ᵀ)ᵀ, contraction over M
        h2 = _rerotate(had)
        c = fp4_rht_eden_quant_t(g, h2)
        d = fp4_rht_eden_requant_t(x_c, x_sc, x_s, h2)
        grad_w = nvfp4_mm(*c, *d)
        return grad_x, grad_w, None, None, None


class NVFP4Linear(Linear):
    """Drop-in nanochat Linear with NVFP4 compute. Master weight stays fp32."""

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None,
                 *, four_over_six=True, bwd_quant=True):
        assert not bias, "NVFP4Linear supports bias=False only"
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.four_over_six = four_over_six
        self.bwd_quant = bwd_quant
        had = None
        if self.weight.device.type == "cuda":
            had = hadamard_matrix(HAD_DIM, self.weight.device)
        self.register_buffer("had", had, persistent=False)

    def forward(self, x):
        assert x.dtype == torch.bfloat16, f"NVFP4Linear needs bf16 input, got {x.dtype}"
        w = self.weight.to(x.dtype)
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = _NVFP4Matmul.apply(x2, w, self.had, self.four_over_six, self.bwd_quant)
        return out.view(*x.shape[:-1], self.out_features)

    @classmethod
    def from_float(cls, mod, four_over_six=True, bwd_quant=True):
        """Wrap an existing Linear, sharing its weight (no copy)."""
        with torch.device("meta"):
            new = cls(mod.in_features, mod.out_features, bias=False,
                      four_over_six=four_over_six, bwd_quant=bwd_quant)
        new.weight = mod.weight
        new.had = hadamard_matrix(HAD_DIM, mod.weight.device)
        return new


def nvfp4_module_filter(mod, fqn, quantize_lm_head=False):
    """Default filter: big Linears with dims % 128; lm_head opt-in."""
    if not isinstance(mod, torch.nn.Linear) or mod.bias is not None:
        return False
    if mod.in_features % 128 != 0 or mod.out_features % 128 != 0:
        return False
    if min(mod.in_features, mod.out_features) < 128:
        return False
    if fqn == "lm_head" and not quantize_lm_head:
        return False
    return True


def convert_to_nvfp4_training(module, *, module_filter_fn=None, quantize_lm_head=False,
                              four_over_six=True, bwd_quant=True):
    """Replace matching Linears with NVFP4Linear (shared weights). Returns count."""
    major, _ = torch.cuda.get_device_capability()
    assert major >= 10, f"NVFP4 needs Blackwell (sm100+), got sm{major}x"
    if module_filter_fn is None:
        module_filter_fn = lambda m, fqn: nvfp4_module_filter(m, fqn, quantize_lm_head)
    count = 0

    def _convert(mod, prefix=""):
        nonlocal count
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, torch.nn.Linear) and not isinstance(child, NVFP4Linear):
                if module_filter_fn(child, fqn):
                    setattr(mod, name, NVFP4Linear.from_float(
                        child, four_over_six=four_over_six, bwd_quant=bwd_quant))
                    count += 1

    _convert(module)
    return count
