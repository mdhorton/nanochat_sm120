"""Per-layer FP8 scaling state for the sm120 stack.

The fp8 half of delayed scaling: the three roles, their constants, and the readback that reports
each tensor's amax for a later step. The history itself is in delayed_scale.py, shared with the
NVFP4 path.

Attached to every Float8Linear by `enable_delayed_scaling` and stepped once per optimizer step
by the recipe. It registers buffers, so it must run before torch.compile. They are
non-persistent: checkpoints stay interchangeable with a bf16 run, and a resumed run re-warms
the history rather than restoring it.

Ported from sm120_nanochat@refactor, which also carries a weight-cast cache and a fused fp32
wgrad accumulator against the same layers -- see TODO.md.
"""
import torch

from nanochat.fp8 import EPS, Float8Linear
from nanochat.sm120.delayed_scale import _AMAX, _INIT_AMAX, _INV, _SCALE, DelayedScaleState  # noqa: F401

# --- static-scale spike (measurement only, produces wrong gradients) ---------------------
# Removes the amax reduction that gates every cast, to measure the ceiling on delayed
# scaling before building the real thing. See dev/perf-log.md, "Single-pass fp8 scaling".
# Assumed amax per dtype; both failure directions are safe for a timing run (too small
# saturates at the clamp, too large underflows to zero), neither produces inf/NaN.
_STATIC_SPIKE = False
_SPIKE_AMAX = {torch.float8_e4m3fn: 8.0, torch.float8_e5m2: 1.0}


def set_static_spike(enabled):
    global _STATIC_SPIKE
    _STATIC_SPIKE = enabled


def _spike_scale(x, fp8_dtype):
    # Derived from x every call, never cached: a module-level tensor cache captures a
    # FakeTensor on the first trace and the next compile dies with "Mixing fake modes NYI".
    # Persistent scale state has to be a registered buffer (a graph input), not Python state.
    return x.new_full((), torch.finfo(fp8_dtype).max / _SPIKE_AMAX[fp8_dtype], dtype=torch.float32)

# --- delayed scaling ---------------------------------------------------------------------
# The history itself lives in delayed_scale.py, which is precision-agnostic; this file supplies
# the fp8 half -- the roles, their constants, and the readback.
#
# Worth +10.7% at d12, which is 97% of what deleting the amax outright would give
# (dev/perf-log.md experiment 10). The load-bearing detail from there: the next-step amax
# reduces over the *fp8 output*, not the source. Inductor will not fuse that reduction into the
# cast, so reducing over the source pays the same bytes as dynamic scaling and nets exactly
# nothing; over the 1-byte output it recovers ~93% of the ceiling. amax(source) == amax(q)/scale
# whenever nothing clipped.
_ROLE_NAMES = ("in", "w", "go")              # the three quantized tensors of a Linear
_IN, _W, _GO = 0, 1, 2                       # their slots in DelayedScaleState's [role, layer]
_ROLE_DTYPE = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)

# Saturation has to stay distinguishable from a tensor sitting exactly at its historical amax,
# since both read back FMAX. So scales target one fp8 ulp *below* FMAX: e4m3fn 448 -> 416, e5m2
# 57344 -> 49152, the largest representable below the max. Without this the search fires on
# every exactly-scaled tensor and oscillates the scale 2x with the history's period, costing a
# full mantissa bit; the headroom itself costs under 0.11 bits.
_SCALE_TARGET = {torch.float8_e4m3fn: 416.0, torch.float8_e5m2: 49152.0}


@torch.no_grad()
def to_fp8(x, fp8_dtype, state=None):
    """Quantize a tensor to FP8, with delayed or spiked scaling.

    Same contract as nanochat.fp8._to_fp8, which handles the dynamic case this falls back to.
    `state` is this role's (scale, inv_scale, amax) scalars; None selects dynamic scaling,
    where the amax reduction gates the cast and the tensor is read twice.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    if state is not None:
        # Delayed: the scale is already known, so nothing needs reading before the cast
        scale, inv_scale = state[_SCALE], state[_INV]
    elif _STATIC_SPIKE:
        # No reduction, so the cast fuses into its producer -- the whole point of the spike
        scale = _spike_scale(x, fp8_dtype)
        inv_scale = scale.reciprocal()
    else:
        # Dynamic, identical to upstream: the amax reduction gates the cast.
        amax = x.float().abs().max()
        scale = (fp8_max / amax.double().clamp(min=EPS)).float()
        inv_scale = scale.reciprocal()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    if state is not None:
        _record_amax(x_fp8, scale, state[_AMAX])
    return x_fp8, inv_scale


def _record_amax(x_fp8, scale, amax_out):
    """Report this tensor's amax for a later step, reducing over the fp8 output.

    Reading the 1-byte result rather than the source is what makes delayed scaling pay; see
    the note above. The reading is exact whenever nothing clipped, and DelayedScaleState
    handles the two cases where it is not.
    """
    raw = x_fp8.float().abs().max()
    # max over the grad-accumulation micro-steps; DelayedScaleState.update zeroes it after
    amax_out.copy_(torch.maximum(amax_out, raw / scale))


def enable_delayed_scaling(model, history_len=16, margin=2.0, allreduce=False):
    """Attach delayed scaling to every Float8Linear in model. Returns None if there are none.

    Must run before torch.compile: it registers the buffers the compiled graph reads.
    """
    modules = [m for m in model.modules() if isinstance(m, Float8Linear)]
    if not modules:
        return None
    return DelayedScaleState(
        modules,
        roles=_ROLE_NAMES,
        targets=[_SCALE_TARGET[d] for d in _ROLE_DTYPE],
        maxes=[torch.finfo(d).max for d in _ROLE_DTYPE],
        prefix="fp8",
        seed_role="w",   # the weight's amax is already known; the other two start at _INIT_AMAX
        history_len=history_len, margin=margin, allreduce=allreduce,
    )
