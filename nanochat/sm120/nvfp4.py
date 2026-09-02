"""NVFP4 training on sm_120, from the Quartet-II method (arXiv:2601.22813).

Both GEMM operands are NVFP4 in the forward *and* the backward, which is the part that
distinguishes this from a forward-only fp4 arm. NVFP4 is Blackwell-only -- Hopper's tensor
cores stop at fp8 -- and issues block-scaled fp4 at twice the fp8 rate.

What the method does, and why each piece is there:

  forward    Weights and activations quantize with **4/6 rounding**: each group of 16 is
             quantized twice, once with the block scale placed so the group max lands on 6 and
             once on 4, and the lower-error result is kept. E2M1's grid is coarse and uneven at
             the top ({..., 3, 4, 6}), so which of the two the max snaps to is worth a real
             amount of error.
  backward   The delicate half. Gradients are heavy-tailed, and quantizing them directly is
             biased in a way that compounds over a run. Each backward operand is first hit
             with a **random Hadamard transform** -- a fresh sign pattern every step
             (`rerotate_hadamard`), which spreads outliers across the 16-element groups instead
             of letting one dominate its block scale -- and then quantized with **EDEN**, which
             chooses the block scale stochastically so the quantized tensor is an *unbiased*
             estimate. Bias is what a short run cannot average away; variance it can.

The saved forward operands are reused: `rht128_requant` dequantizes, transposes, rotates and
requantizes in a single pass, so the backward never materializes a bf16 copy of x or W.

Two things this port changes from upstream `quartet2/linear.py`:

  the GEMM   upstream calls `flashinfer.mm_fp4`. `torch._scaled_mm` already does block-scaled
             NVFP4 here and the kernels emit block scales in exactly the cutlass 128x4 swizzle
             it indexes, so the substitution is a drop-in and drops the dependency.
  the base   `NVFP4Linear` subclasses nanochat's `Linear`, not `nn.Linear`. That class is the
             structural marker `GPT.num_matmul_params()` counts to derive FLOPs/token; getting
             this wrong under-reported FLOPs/token by 4.5x on every --fp8 run once already
             (see the `Float8Linear` docstring).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Linear as _NanochatLinear
from nanochat.sm120 import fp4_gemm, nvfp4_numerics, nvfp4_state
from nanochat.sm120.quartet.quant import (
    NVFP4QuantMode,
    new_seed,
    quant_fp4,
    rht128_quant_eden,
    rht128_requant,
)
from nanochat.sm120.quartet.rht import hadamard_matrix, rerotate_hadamard

FP4 = torch.float4_e2m1fn_x2
GROUP = 16          # elements per NVFP4 block scale
ALIGN = 128         # every kernel here wants both dims a multiple of this

# Upstream's constants. The forward is quantized at nominal scale; the backward deliberately
# overshoots the fp4 range by 17/16 and then pulls back by 0.93 -- clipping a little of the
# heavy tail costs less error than spending the top of the grid on it.
FORWARD_SCALE_OVERRIDE = 1.0
BACKWARD_SCALE_OVERRIDE = (17 / 16) * 0.93


def fp4_mm(a_fp4, b_fp4, a_scales, b_scales, alpha):
    """`a @ b.T` with both operands NVFP4. `a` is [M, K/2] packed, `b` is [N, K/2] packed.

    The GEMM applies only the E4M3 block scales, so the two per-tensor scales have to be applied
    to the result. On the `_scaled_mm` path **that multiply stays outside any opaque op on
    purpose**: as its own kernel it is a full pass over an [M, N] output where the GEMM is
    compute-bound, so the hope is that Inductor folds it into the consumer. `_scaled_mm`'s own
    `scale_result` cannot take it -- the block-scaled path rejects it.

    `--nvfp4-epilogue-alpha` deletes that pass instead of hoping: the cuBLASLt launcher takes
    alpha as a device pointer and applies it inside the GEMM epilogue, which also rounds to bf16
    once rather than twice (measurably the accurate direction). It needs `--nvfp4-lt-gemm`.
    """
    if fp4_gemm.epilogue_alpha():
        return fp4_gemm.mm(a_fp4, b_fp4, a_scales, b_scales, alpha)
    if fp4_gemm.enabled():
        out = fp4_gemm.mm(a_fp4, b_fp4, a_scales, b_scales)
    else:
        out = torch._scaled_mm(
            a_fp4.view(FP4), b_fp4.view(FP4).t(),
            scale_a=a_scales, scale_b=b_scales, out_dtype=torch.bfloat16,
        )
    return out * alpha


def _unblock(scales, rows, cols):
    """Undo the 128x4 scale swizzle: flat swizzled E4M3 -> [rows, cols/16]."""
    n_row_blocks, n_col_blocks = rows // ALIGN, (cols // GROUP) // 4
    r = scales.reshape(-1, 32, 4, 4).permute(0, 2, 1, 3)
    r = r.reshape(-1, n_col_blocks, ALIGN, 4).permute(0, 2, 1, 3)
    return r.reshape(n_row_blocks * ALIGN, n_col_blocks * 4)


_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequantize(fp4, micro_scales, tensor_scale):
    """NVFP4 -> bf16. Reference path, used by `disable_backward_quant` and by the tests."""
    codes = fp4.view(torch.uint8).to(torch.int32)
    unpacked = torch.stack([codes & 0xF, (codes >> 4) & 0xF], dim=-1).flatten(start_dim=-2)
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=fp4.device)
    values = grid[unpacked]
    scales = _unblock(micro_scales.to(torch.float32), fp4.shape[0], fp4.shape[1] * 2)
    out = (values.unflatten(dim=-1, sizes=(-1, GROUP)) * scales[..., None]).flatten(start_dim=-2)
    return (out * tensor_scale).to(torch.bfloat16)


def _pad_rows(x):
    """Pad dim 0 up to a multiple of 128, the only dimension that is not structural.

    The kernels want both dims aligned. A Linear's in/out features are fixed and checked once
    at conversion time (`is_nvfp4_convertible`), but the token count is whatever the caller
    passes -- 16384 in training, arbitrary during eval and sampling. Zero rows are exact here:
    they do not move a per-tensor amax, their block scales are discarded with them, and in the
    wgrad they contribute a zero term to a sum over M.
    """
    rows = x.shape[0]
    pad = -rows % ALIGN
    if pad == 0:
        return x, 0
    return F.pad(x, (0, 0, 0, pad)), pad


class _NVFP4Matmul(torch.autograd.Function):
    """out = x @ W.T, NVFP4 on all three GEMMs. x is 2-D and row-aligned by the caller.

    With `main_grad` (a WgradAccumStore view) the wgrad accumulates into it inside the GEMM
    epilogue and **no gradient is returned for the weight at all** -- so anything downstream
    reading `.grad` has to be reading that buffer instead. It is stashed through
    `save_for_backward` like the weight cache, and mutated in the backward that unpacked it;
    both are safe only because the store is zeroed after `optimizer.step()` and never between
    a forward and its backward.
    """

    @staticmethod
    def forward(ctx, x, weight, w_fp4, w_ms, w_ts, had, scratch_amax, main_grad, mode,
                disable_backward_quant, scale_state=None, w_t=None, w_ts_t=None, rht="all",
                bwd_source="fp4"):
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

        # --nvfp4-scaling delayed: the activation's amax comes from a history, so the
        # vector_norm pre-pass never runs and the next reading is taken off the block scales.
        amax_in, amax_out = (None, None) if scale_state is None else scale_state
        xq = quant_fp4(x, scale_override=FORWARD_SCALE_OVERRIDE, mode=mode, amax=amax_in)
        if amax_in is not None:
            nvfp4_state.record_amax(xq.micro_scales, amax_in, amax_out)
        if w_fp4 is None:
            wq = quant_fp4(weight, scale_override=FORWARD_SCALE_OVERRIDE, mode=mode)
            w_fp4, w_ms, w_ts = wq.fp4, wq.micro_scales, wq.tensor_scale
        # else: the cache holds the same quantization of the same weight (see WeightCache).
        # `weight` is still an input because it is the slot backward returns grad_weight into --
        # unless `main_grad` took it, in which case that slot returns None.

        # The numerics levers (nvfp4_numerics) decide what else the backward needs: the bf16
        # weight and activation when it quantizes from source, the 2D transposed weight when
        # dgrad shares the forward's quantization. Saved only when read, so the default path
        # holds exactly what it always did.
        needs_weight = bwd_source == "bf16" or rht != "all"
        ctx.save_for_backward(xq.fp4, xq.micro_scales, xq.tensor_scale,
                              w_fp4, w_ms, w_ts, had, scratch_amax, main_grad,
                              weight if needs_weight else None,
                              x if bwd_source == "bf16" else None, w_t, w_ts_t)
        ctx.disable_backward_quant = disable_backward_quant
        ctx.rht = rht
        ctx.bwd_source = bwd_source
        return fp4_mm(xq.fp4, w_fp4, xq.micro_scales, w_ms, xq.tensor_scale * w_ts)

    @staticmethod
    def backward(ctx, grad_output):
        (x_fp4, x_ms, x_ts, w_fp4, w_ms, w_ts, had, scratch_amax,
         main_grad, weight, x_bf16, w_t, w_ts_t) = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        nones = (None,) * 13

        if ctx.disable_backward_quant:
            # bf16 backward against the dequantized forward operands. For tests and ablations:
            # it isolates forward quantization error from the gradient estimator.
            xr = dequantize(x_fp4, x_ms, x_ts)
            wr = dequantize(w_fp4, w_ms, w_ts)
            grad_weight = grad_output.T @ xr
            if main_grad is not None:
                main_grad.add_(grad_weight.float())
                grad_weight = None
            return (grad_output @ wr, grad_weight) + nones

        # A fresh sign pattern per backward: the Hadamard is a fixed rotation, the randomness is
        # what keeps outliers from landing in the same group twice.
        if ctx.rht != "none":
            had = rerotate_hadamard(had)
        h16 = had[:16, :]  # the kernels regenerate the rest by Sylvester's construction
        so = BACKWARD_SCALE_OVERRIDE

        # dgrad = E @ W. By default both operands are rotated and requantized from the saved
        # fp4; --nvfp4-rht / --nvfp4-bwd-source / --nvfp4-weight-2d change the recipe there.
        e_ht, wt_ht = nvfp4_numerics.dgrad_operands(
            grad_output, weight, w_fp4, w_ms, w_ts, w_t, w_ts_t, h16, so, scratch_amax,
            rht=ctx.rht, bwd_source=ctx.bwd_source)
        grad_input = fp4_mm(e_ht.fp4, wt_ht.fp4, e_ht.micro_scales, wt_ht.micro_scales,
                            e_ht.tensor_scale * wt_ht.tensor_scale)

        # wgrad = E.T @ X, likewise
        et_ht, xt_ht = nvfp4_numerics.wgrad_operands(
            grad_output, x_bf16, x_fp4, x_ms, x_ts, h16, so, scratch_amax,
            rht=ctx.rht, bwd_source=ctx.bwd_source)
        alpha = et_ht.tensor_scale * xt_ht.tensor_scale
        if main_grad is not None:
            # Gradient accumulation inside the GEMM epilogue (WgradAccumStore): the contribution
            # lands in the fp32 buffer directly, so nothing is returned for `weight` and
            # autograd's own cast-and-add into `.grad` never runs. The optimizer reads the
            # buffer because attach() handed it over as `.grad` before step().
            fp4_gemm.mm_accum(et_ht.fp4, xt_ht.fp4, et_ht.micro_scales, xt_ht.micro_scales,
                              alpha, main_grad)
            grad_weight = None
        else:
            grad_weight = fp4_mm(et_ht.fp4, xt_ht.fp4, et_ht.micro_scales, xt_ht.micro_scales,
                                 alpha)

        return (grad_input, grad_weight) + nones


class NVFP4Linear(_NanochatLinear):
    """Drop-in Linear that runs all three matmuls in NVFP4.

    Weights stay fp32 masters; only the matmuls are quantized. In and out features must both be
    multiples of 128 (`is_nvfp4_convertible`) -- the token dimension is padded as needed.
    """

    def __init__(self, *args, four_over_six=True, disable_backward_quant=False, numerics=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = NVFP4QuantMode.FOUR_SIX if four_over_six else NVFP4QuantMode.RNE
        self.disable_backward_quant = disable_backward_quant
        # The numerics levers (nvfp4_numerics.NumericsConfig); frozen, so safe as an attribute.
        self.numerics = (numerics or nvfp4_numerics.DEFAULT).validate()
        self.weight_cache_enabled = False
        # Buffers, not plain attributes: a Python-side tensor cache captures a FakeTensor on the
        # first trace and the next compile dies with "Mixing fake modes NYI". Non-persistent so
        # they stay out of checkpoints -- both are derived, not learned.
        had = None if self.weight.device.type == "meta" else hadamard_matrix(128, torch.bfloat16, self.weight.device)
        self.register_buffer("had", had, persistent=False)
        self.register_buffer(
            "scratch_amax",
            torch.empty((), dtype=torch.uint32, device=self.weight.device) if self.weight.device.type != "meta" else None,
            persistent=False,
        )
        # The forward weight-quantization cache, filled by refresh_weight_cache(). While these
        # are None the forward quantizes the weight itself, so an unmanaged module still works.
        # fp4_wt / fp4_wts are the transposed form of a --nvfp4-weight-2d quantization, which
        # dgrad consumes directly; None otherwise.
        for name in ("fp4_w", "fp4_ws", "fp4_wg", "fp4_wt", "fp4_wts"):
            self.register_buffer(name, None, persistent=False)
        # Views into the model-wide DelayedScaleState (--nvfp4-scaling delayed). Registered
        # here rather than assigned later for the same reason as the cache above: a scale has
        # to reach the compiled graph as a buffer, not as Python state.
        for name in ("fp4_scale_in", "fp4_inv_in", "fp4_amax_in"):
            self.register_buffer(name, None, persistent=False)
        # This layer's fp32 gradient accumulator, a view into WgradAccumStore's flat buffer.
        # While it is None the backward returns a gradient the normal way.
        self.register_buffer("fp4_main_grad", None, persistent=False)

    def _apply(self, fn):
        # from_float() builds the shell on meta, so the real buffers are created on the move.
        was_meta = self.weight.device.type == "meta"
        super()._apply(fn)
        if was_meta and self.weight.device.type != "meta":
            self.had = hadamard_matrix(128, torch.bfloat16, self.weight.device)
            self.scratch_amax = torch.empty((), dtype=torch.uint32, device=self.weight.device)
        return self

    @torch.no_grad()
    def refresh_weight_cache(self):
        """Re-quantize the weight into the cache. Call once per optimizer step, after `step()`.

        The weight is an fp32 master that nothing inside the grad-accumulation loop moves, so
        quantizing it on every micro-step is (accum - 1)/accum redundant. Worth 2-8% depending
        on shape; see dev/nvfp4-quartet.md.

        **Ordering matters.** The forward hands these buffers to `save_for_backward`, and this
        writes into them in place, so it must run after every backward that consumed them --
        i.e. after `optimizer.step()`, never between a forward and its backward.

        Only the *forward* weight quantization is cacheable. The backward's `rht128_requant(w)`
        is not: it consumes a Hadamard that `rerotate_hadamard` re-randomizes every backward,
        which is the estimator working as intended. With `--nvfp4-weight-2d` the dgrad reads the
        cache too (its transposed form), so the ordering rule above covers the backward as well.
        """
        w = self.weight.detach().to(torch.bfloat16)
        if self.numerics.weight_2d:
            fp4, ws, wg, fp4_t, wts = nvfp4_numerics.quantize_weight_2d(
                w, four_over_six=self.mode is NVFP4QuantMode.FOUR_SIX,
                scale_override=FORWARD_SCALE_OVERRIDE)
        else:
            fp4, ws, wg = quant_fp4(w, scale_override=FORWARD_SCALE_OVERRIDE, mode=self.mode)
            fp4_t = wts = None
        if self.fp4_w is None:
            # First call: allocate. Buffers, not attributes -- see __init__.
            self.fp4_w = torch.empty_like(fp4)
            self.fp4_ws = torch.empty_like(ws)
            self.fp4_wg = torch.empty_like(wg)
            if fp4_t is not None:
                self.fp4_wt = torch.empty_like(fp4_t)
                self.fp4_wts = torch.empty_like(wts)
        self.fp4_w.copy_(fp4)
        self.fp4_ws.copy_(ws)
        self.fp4_wg.copy_(wg)
        if fp4_t is not None:
            self.fp4_wt.copy_(fp4_t)
            self.fp4_wts.copy_(wts)

    def forward(self, x):
        x = x.to(COMPUTE_DTYPE)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        x_2d, pad = _pad_rows(x_2d.contiguous())

        amax_in = nvfp4_state.assumed_amax(self)
        # None unless --nvfp4-scaling delayed attached a history. Skipped without grad, so an
        # eval or sampling pass through this module cannot poison it.
        scale_state = None if amax_in is None or not torch.is_grad_enabled() else (
            amax_in, self.fp4_amax_in)
        if self.numerics.weight_2d:
            assert self.fp4_wt is not None, "--nvfp4-weight-2d needs the weight cache primed"
        out = _NVFP4Matmul.apply(x_2d, self.weight.to(torch.bfloat16),
                                 self.fp4_w, self.fp4_ws, self.fp4_wg,
                                 self.had, self.scratch_amax, self.fp4_main_grad,
                                 self.mode, self.disable_backward_quant, scale_state,
                                 self.fp4_wt, self.fp4_wts, self.numerics.rht,
                                 self.numerics.bwd_source)
        if pad:
            out = out[: x_2d.shape[0] - pad]
        out = out.reshape(*orig_shape[:-1], out.shape[-1])
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    @classmethod
    def from_float(cls, mod, **kwargs):
        """Build from an existing Linear, sharing weight and bias (no copies)."""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, **kwargs)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.had = hadamard_matrix(128, torch.bfloat16, mod.weight.device)
        new_mod.scratch_amax = torch.empty((), dtype=torch.uint32, device=mod.weight.device)
        return new_mod


class WgradAccumStore:
    """The fp32 gradient accumulators the wgrad GEMM writes into directly (`--nvfp4-fuse-wgrad`).

    Without it each micro-step's wgrad is written to bf16, widened to fp32 and added into
    `.grad` by autograd's `AccumulateGrad` -- 12 bytes of traffic per element per micro-step,
    which `dev/nvfp4-quartet.md` measures as the 135.7 ms/step `CUDAFunctor_add<float>` row.
    Folding the accumulation into the GEMM epilogue (beta=1, C = D = this buffer, via
    `fp4_gemm.mm_accum`) does it in 8, and drops the per-contribution bf16 rounding: the term
    joins the sum out of the fp32 accumulator rather than after a round trip through bf16.

    One flat fp32 tensor with a contiguous view per layer, registered as a non-persistent buffer
    so the compiled graph reads it as a plain input. `attach()` hands the views over as `p.grad`
    before the optimizer reads them -- the same fp32 dtype it always read, so `MuonAdamW` and
    its reduce/gather paths need nothing. `zero_()` must run after each optimizer step: the
    epilogue always accumulates, so "the first micro-step assigns" becomes "the buffer starts at
    zero".

    Memory is close to neutral -- this replaces the fp32 `.grad` autograd would have allocated
    for the same parameters -- but it is *resident* where `.grad` was freed by
    `zero_grad(set_to_none=True)` each step.
    """

    def __init__(self, modules):
        self.modules = list(modules)
        device = self.modules[0].weight.device
        total = sum(m.weight.numel() for m in self.modules)
        self.flat = torch.zeros(total, device=device, dtype=torch.float32)
        off = 0
        for m in self.modules:
            n = m.weight.numel()
            m.fp4_main_grad = self.flat[off:off + n].view_as(m.weight)
            off += n

    @torch.no_grad()
    def attach(self):
        """Point every weight's `.grad` at its accumulator, before optimizer.step() reads it."""
        for m in self.modules:
            m.weight.grad = m.fp4_main_grad

    @torch.no_grad()
    def zero_(self):
        self.flat.zero_()


def enable_wgrad_accum(model):
    """Fuse fp32 wgrad accumulation into every NVFP4Linear's epilogue. Returns the store.

    Must run before `torch.compile`: it registers the buffers the compiled backward writes.
    Requires the cuBLASLt launcher -- without it `mm_accum` would fall back to a `_scaled_mm`
    plus an eager add, which is the accumulation this exists to delete plus a kernel, i.e. a
    flag that silently does the opposite of what it says.
    """
    if not fp4_gemm.enabled():
        raise ValueError("fused wgrad accumulation needs the cuBLASLt launcher "
                         "(--nvfp4-lt-gemm): beta=1 is an epilogue _scaled_mm cannot express")
    modules = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
    if not modules:
        return None
    return WgradAccumStore(modules)


def refresh_weight_caches(model):
    """Refresh every NVFP4Linear's weight cache. Call once per optimizer step, after `step()`.

    Returns the number of modules refreshed (0 if none are cached, so callers can no-op).
    """
    mods = [m for m in model.modules() if isinstance(m, NVFP4Linear) and m.weight_cache_enabled]
    for m in mods:
        m.refresh_weight_cache()
    return len(mods)


def enable_weight_caches(model):
    """Turn on the weight cache for every NVFP4Linear and prime it. Returns the count."""
    mods = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
    for m in mods:
        m.weight_cache_enabled = True
        m.refresh_weight_cache()
    return len(mods)


def is_nvfp4_convertible(mod):
    """Both features a multiple of 128, which is what every kernel in the path requires."""
    return mod.in_features % ALIGN == 0 and mod.out_features % ALIGN == 0


def convert_to_nvfp4_training(module, *, module_filter_fn=None, four_over_six=True,
                              disable_backward_quant=False, numerics=None):
    """Replace eligible nn.Linear layers with NVFP4Linear, in place. Returns the module.

    Mirrors `fp8.convert_to_float8_training`: post-order walk, weights and biases shared with
    the modules they replace. Layers whose features are not 128-aligned are left alone -- in
    nanochat that is `ve_gate` and `smear_gate`, which are negligible in FLOPs. `numerics` is
    an `nvfp4_numerics.NumericsConfig`; its layer selection arrives as `module_filter_fn`.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, NVFP4Linear):
                if not is_nvfp4_convertible(child):
                    continue
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, NVFP4Linear.from_float(
                        child, four_over_six=four_over_six,
                        disable_backward_quant=disable_backward_quant, numerics=numerics))

    _convert(module)
    return module
