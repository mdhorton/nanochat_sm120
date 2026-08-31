"""Per-layer FP8 scaling state for the sm120 stack.

  DelayedScaleState  quantize with a scale derived from previous steps' amaxes, so the amax
                     reduction no longer gates the cast   (+10.7% at d12, --fp8-scaling delayed)

Attached to every Float8Linear by `enable_delayed_scaling` and stepped once per optimizer step
by the recipe. It registers buffers, so it must run before torch.compile. They are
non-persistent: checkpoints stay interchangeable with a bf16 run, and a resumed run re-warms
the history rather than restoring it.

Ported from sm120_nanochat@refactor, which also carries a weight-cast cache and a fused fp32
wgrad accumulator against the same layers -- see TODO.md.
"""
import torch
import torch.distributed as dist

from nanochat.fp8 import EPS, Float8Linear

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
# Dynamic scaling cannot begin the cast until it knows the amax, so every quantized tensor is
# read twice. Delayed scaling quantizes with a scale derived from *previous* steps' amaxes,
# held in a buffer: the cast becomes pure pointwise and the reduction leaves the critical path.
# Worth +10.7% at d12, which is 97% of what deleting the amax outright would give
# (dev/perf-log.md experiment 10).
#
# Two details are load-bearing, both learned there:
#   - The next-step amax reduces over the *fp8 output*, not the source. Inductor will not fuse
#     that reduction into the cast, so reducing over the source pays the same bytes as dynamic
#     scaling and nets exactly nothing; over the 1-byte output it recovers ~93% of the ceiling.
#     amax(source) == amax(q) / scale whenever nothing clipped.
#   - Scale state must be a registered buffer, i.e. a graph input, never a Python-side cache,
#     or the second compile dies with "Mixing fake modes NYI".
#
# Layout: two tensors per model, [2 fields, 3 roles, L] of scales and [3 roles, L] of amaxes,
# so the once-per-step update touches all L layers in a handful of kernels instead of 3 per
# layer. Each layer is handed nine 0-d *views* into them, one buffer each, rather than one
# indexed tensor: a scale must reach the graph as a plain input, because Inductor's
# tuned_scaled_mm lowering cannot take an in-graph select as scale_a.
_ROLE_NAMES = ("in", "w", "go")              # the three quantized tensors of a Linear
_IN, _W, _GO = 0, 1, 2
_ROLE_DTYPE = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)
_SCALE, _INV, _AMAX = 0, 1, 2                # fields of a role's (scale, inv_scale, amax)

# Step of the two-way search DelayedScaleState.update runs when a reading is uninformative
# (the cast saturated, or it underflowed to nothing).
_AMAX_SEARCH = 2.0
# The upward half needs saturation to be distinguishable from a tensor sitting exactly at its
# historical amax, since both read back FMAX. So scales target one fp8 ulp *below* FMAX:
# e4m3fn 448 -> 416, e5m2 57344 -> 49152, the largest representable below the max. Without this
# the search fires on every exactly-scaled tensor and oscillates the scale 2x with the
# history's period, costing a full mantissa bit; the headroom itself costs under 0.11 bits.
_SCALE_TARGET = {torch.float8_e4m3fn: 416.0, torch.float8_e5m2: 49152.0}
# Assumed amax before any measurement exists. Deliberately high: overestimating only costs
# precision on step 0 (small values underflow) and still reads back an accurate amax, so the
# scale is right from step 1, whereas underestimating saturates and needs several search steps.
# Weights do not use this — their scale is seeded exactly, from the real weights.
_INIT_AMAX = 1e3


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


class DelayedScaleState:
    """Model-wide scale and amax history for delayed fp8 scaling.

    One of these per model, created after convert_to_float8_training and before torch.compile.
    Call update() once per optimizer step, after the last backward.

    margin is headroom against the amax growing between steps, and is close to free: fp8 is
    floating point, so aiming below FMAX costs range at the bottom (e4m3 spans ~17.8 binades)
    rather than mantissa bits, whereas clipping costs real error and trips the amax search.
    """

    def __init__(self, modules, history_len=16, margin=2.0, allreduce=False):
        assert history_len >= 1 and margin >= 1.0
        device = modules[0].weight.device
        num = len(modules)
        self.history_len = history_len
        self.margin = margin
        self.allreduce = allreduce
        self.idx = 0
        self.filled = 0  # history slots holding a real measurement; the rest are not yet valid
        self.scales = torch.ones(2, 3, num, device=device, dtype=torch.float32)
        self.amax = torch.zeros(3, num, device=device, dtype=torch.float32)
        self.hist = torch.zeros(history_len, 3, num, device=device, dtype=torch.float32)
        self.target = torch.tensor([[_SCALE_TARGET[d]] for d in _ROLE_DTYPE],
                                   device=device, dtype=torch.float32)  # [3 roles, 1]
        self.fp8_max = torch.tensor([[torch.finfo(d).max] for d in _ROLE_DTYPE],
                                    device=device, dtype=torch.float32)  # [3 roles, 1]
        # Seed the *scales*, deliberately not the history: the history is a max, so a seeded
        # slot would outlive its usefulness by history_len steps and hold every activation at
        # the assumed amax for the whole window. Weights are already known, so their scale is
        # exact from step 0; activations and gradients correct on the first update.
        seed = torch.full((3, num), _INIT_AMAX, device=device, dtype=torch.float32)
        with torch.no_grad():
            for i, mod in enumerate(modules):
                # An all-zero weight is a real initialisation here (gpt.py zero-inits both
                # c_proj), and its amax says nothing about what it will be one step later, so
                # fall back to the assumption rather than seeding a scale of zero.
                amax = mod.weight.float().abs().max()
                if amax > 0:
                    seed[_W, i] = amax
        self._set_scales(seed)
        for i, mod in enumerate(modules):
            for r, role in enumerate(_ROLE_NAMES):
                setattr(mod, f"fp8_scale_{role}", self.scales[_SCALE, r, i])
                setattr(mod, f"fp8_inv_{role}", self.scales[_INV, r, i])
                setattr(mod, f"fp8_amax_{role}", self.amax[r, i])

    @torch.no_grad()
    def _set_scales(self, h):
        """Point every scale at amax h ([3 roles, num]), less the margin and ulp headroom."""
        scale = self.scales[_SCALE]
        # Where a role has no measurement at all, keep whatever scale it already had
        scale.copy_(torch.where(h > 0, self.target / (h * self.margin).clamp(min=EPS), scale))
        self.scales[_INV].copy_(scale.reciprocal())

    @torch.no_grad()
    def update(self):
        """Fold this step's amaxes into the history and recompute every scale.

        The hot path reports amax(q)/scale, which is exact whenever nothing clipped. The two
        cases where it is not are handled here, because both need a scale to compare against.
        """
        if self.allreduce:
            dist.all_reduce(self.amax, op=dist.ReduceOp.MAX)  # contiguous, NCCL takes it directly
        scale = self.scales[_SCALE]
        raw = self.amax * scale                             # back to fp8 units
        implied = self.target / (scale * self.margin)       # the amax this scale is aimed at
        # A saturated cast pins amax(q) at FMAX, which is a floor on the true amax rather than
        # a reading of it, so search up past it -- otherwise a clipped tensor reports the same
        # value forever and the scale never recovers. A cast that underflowed to nothing
        # reports 0, which bounds the amax from above instead; searching down means shrinking
        # the whole window, since the history is a max and would otherwise pin the scale.
        blind = raw <= 0
        self.hist.mul_(torch.where(blind, 1.0 / _AMAX_SEARCH, 1.0))
        est = torch.where(raw >= self.fp8_max, self.amax * _AMAX_SEARCH, self.amax)
        self.hist[self.idx].copy_(torch.where(blind, implied / _AMAX_SEARCH, est))
        self.idx = (self.idx + 1) % self.history_len
        self.filled = min(self.filled + 1, self.history_len)
        self.amax.zero_()
        self._set_scales(self.hist[:self.filled].amax(dim=0))


def enable_delayed_scaling(model, history_len=16, margin=2.0, allreduce=False):
    """Attach delayed scaling to every Float8Linear in model. Returns None if there are none.

    Must run before torch.compile: it registers the buffers the compiled graph reads.
    """
    modules = [m for m in model.modules() if isinstance(m, Float8Linear)]
    if not modules:
        return None
    return DelayedScaleState(modules, history_len=history_len, margin=margin, allreduce=allreduce)
