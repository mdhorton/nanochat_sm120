"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    amax computation with the preceding layer's activation function).
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
# Subclass nanochat's Linear, not nn.Linear: GPT.num_matmul_params() counts
# isinstance(m, Linear) to derive FLOPs/token, so a Float8Linear that is not one silently
# drops out of the count (see the class docstring below). gpt.py does not import fp8, so
# this direction is safe.
from nanochat.gpt import Linear as _NanochatLinear

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# =============================================================================
# Backend seam
# =============================================================================
# The three GEMMs below are all of FP8 training, but *how* each operand is quantized and which
# kernel runs each matmul is where accelerator-specific work lives — delayed scaling, cached
# weight casts, fused gradient accumulation, tuned cuBLASLt algorithms. Rather than branch on
# each of those here, this module calls a backend and ships a default that does exactly what
# the docstring above describes. nanochat/sm120 installs its own; see dev/LOG_sm120.md.
#
# Per-layer tensors deliberately do NOT live on the backend object. A scale has to reach the
# compiled graph as a plain buffer input: Inductor's tuned_scaled_mm lowering cannot take an
# in-graph select as scale_a, and a Python-side tensor cache captures a FakeTensor on the first
# trace so the second compile dies with "Mixing fake modes NYI". They travel instead as an
# opaque `state` bundle, built by layer_state() and passed straight back, which this module
# never inspects.


class Float8Backend:
    """The substitution points for the three FP8 GEMMs. Default = plain dynamic scaling."""

    def init_layer(self, layer):
        """Register any per-layer buffers this backend needs. Must run before torch.compile."""

    def layer_state(self, layer):
        """The opaque per-layer bundle handed back to the methods below, or None."""
        return None

    def quantize(self, x, fp8_dtype, role, state):
        """Quantize one operand. role is 'in', 'w' or 'go'."""
        return _to_fp8(x, fp8_dtype)

    def weight_operands(self, weight, state):
        """(fp8 weight, inverse scale, column-major copy or None to build one in backward)."""
        w_fp8, w_inv = _to_fp8(weight, torch.float8_e4m3fn)
        return w_fp8, w_inv, None

    def mm_fwd(self, in_fp8, w_t, in_inv, w_inv, out_dtype, state=None):
        """`state` is this layer's opaque bundle, ignored here. It exists for backends that
        carry extra per-layer operands for this GEMM."""
        # use_fast_accum=True accumulates the dot products in lower precision.
        # Slightly less accurate but measurably faster. Standard practice for
        # the forward pass; we use False in backward for more precise gradients.
        return torch._scaled_mm(in_fp8, w_t, scale_a=in_inv, scale_b=w_inv,
                                out_dtype=out_dtype, use_fast_accum=True)

    def mm_dgrad(self, go_fp8, w_col, go_inv, w_inv, out_dtype):
        return torch._scaled_mm(go_fp8, w_col, scale_a=go_inv, scale_b=w_inv,
                                out_dtype=out_dtype, use_fast_accum=False)

    def wgrad(self, go_fp8, in_fp8, go_inv, in_inv, out_dtype, state):
        """grad_weight, or None if this backend accumulated it somewhere else."""
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()   # [N, B] row-major
        in_col = _to_col_major(in_fp8)   # [B, K] column-major
        return torch._scaled_mm(go_T, in_col, scale_a=go_inv, scale_b=in_inv,
                                out_dtype=out_dtype, use_fast_accum=False)


_backend = Float8Backend()


def set_backend(backend):
    """Install the FP8 backend, or None to restore the default. Before torch.compile.

    Also before convert_to_float8_training: Float8Linear.__init__ asks the backend to register
    its buffers, so a backend installed after conversion never sees the layers.
    """
    global _backend
    _backend = Float8Backend() if backend is None else backend


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight, state=None):
        backend = _backend
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = backend.quantize(input_2d, torch.float8_e4m3fn, "in", state)
        # `weight` stays an input even when the backend returns a cast it made earlier: it is
        # the slot backward returns grad_weight into.
        weight_fp8, weight_inv, w_col = backend.weight_operands(weight, state)
        ctx.save_for_backward(input_fp8, weight_fp8)
        # The inverse scales are stashed on ctx rather than saved: a backend may hand back
        # views into buffers its own backward mutates, which share a version counter with
        # them, and save_for_backward would reject those on the next backward. They need no
        # grad tracking, so ctx is the right place for them and for the opaque state.
        ctx.inv_scales = (input_inv, weight_inv)
        ctx.w_col = w_col
        ctx.state = state

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        return backend.mm_fwd(input_fp8, weight_fp8.t(), input_inv, weight_inv, input_2d.dtype,
                              state=state)

    @staticmethod
    def backward(ctx, grad_output):
        backend = _backend
        in_fp8, w_fp8 = ctx.saved_tensors
        in_inv, w_inv = ctx.inv_scales

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = backend.quantize(grad_output, torch.float8_e5m2, "go", ctx.state)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8) if ctx.w_col is None else ctx.w_col
        grad_input = backend.mm_dgrad(go_fp8, w_col, go_inv, w_inv, grad_output.dtype)

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # None when the backend accumulated it itself, in which case autograd's own
        # cast-and-add accumulation into .grad never runs.
        grad_weight = backend.wgrad(go_fp8, in_fp8, go_inv, in_inv, grad_output.dtype, ctx.state)

        return grad_input, grad_weight, None


class Float8Linear(_NanochatLinear):
    """Drop-in Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.

    Subclasses nanochat's Linear (gpt.py) rather than nn.Linear directly. That class is the
    structural marker GPT.num_matmul_params() counts to derive FLOPs/token; subclassing
    nn.Linear instead dropped every converted layer from the count and under-reported
    FLOPs/token by 7.7x at d12 and 10.5x at d20. That fed the bf16_mfu display (cosmetic),
    wandb's flops_so_far, and --target-flops, where it produced a ~7.7x too long run.
    We override forward() anyway, so nothing of the parent's behaviour is inherited.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _backend.init_layer(self)

    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight, _backend.layer_state(self))
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
