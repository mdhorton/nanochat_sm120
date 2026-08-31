"""Delayed (amax-history) scaling for the NVFP4 forward. Queue B1 in dev/nvfp4-quartet.md.

`quant_fp4` computes `torch.linalg.vector_norm(x, inf)` before every activation quantize, because
NVFP4's per-tensor scale must be known before the kernel runs and so cannot fuse into it. That
pre-pass is ~148 ms/step (4.1%) at d12 -- an upper bound, since Inductor folds the reduction into
its producer and those kernels compute other things too. This supplies the amax from a history
instead, which the kernel already accepts as a device pointer (`quart/quant.py:99`, `amax=`), so
there is no CUDA change here.

**Scope: the forward activation only.** The weight is already a once-per-optimizer-step quantity
via `--nvfp4-weight-cache`, and the four backward quantizes compute their scale device-side with
no `amax` input at all -- giving them one is queue B4, and it means editing vendored Quartet CUDA.

**The reading comes from the block scales, not the tensor.** The kernel writes one e4m3 scale per
16 elements, and from `round_four_six.cu`'s `quantize()`:

    scale       = amax_assumed * inv_scales_max * inv_val_max
    block_scale = e4m3( group_abs_max * inv_val * 6 / (amax_assumed * inv_scales_max) )

so `amax = max(block_scales) * amax_assumed * inv_scales_max`, a reduction over N/16 one-byte
values against the N two-byte values it replaces -- 32x fewer bytes, which is what makes this pay
at all (the same reason fp8 reads its amax back off the 1-byte output rather than the source).

Two consequences of that algebra, both load-bearing:

  - **The reading is biased high by up to 1.5x.** 4/6 picks per group between inv_val in {1/6,
    1/4} by minimum error, and that choice does not cancel: a group that picked 1/4 reports
    1.5x its true group max. Over-estimating is the safe direction (see below), and the margin
    already spans it.
  - **A wrong amax costs almost nothing, until it saturates.** The fp4 codes are `x / (block_scale
    * scale)` and `block_scale * scale ~= group_abs_max * inv_val_max` -- the assumed amax cancels.
    It survives only in *which* e4m3 bucket each block scale rounds into, and e4m3 rounding is
    relative, so a uniform shift is free. The whole risk is one-sided: too low and block scales
    clip at 448. Hence a generous margin and a high initial assumption are both cheap here, in a
    way they are not for fp8, where the single scale is the entire quantization.

RNE mode is not supported: its `inv_scales_max` is 1/448, so a group at the assumed amax already
reads e4m3's max and the saturation search would fire every step. `--nvfp4-rne` is an ablation
rather than a recommendation, so it errors instead of carrying a second set of constants.
"""
import torch

from nanochat.sm120.delayed_scale import DelayedScaleState
from nanochat.sm120.quartet.quant import NVFP4QuantMode

_ROLE = "in"            # the activation, and nothing else
_PREFIX = "fp4"         # buffers land as fp4_scale_in / fp4_inv_in / fp4_amax_in
_E4M3_MAX = 448.0       # what a saturated block scale pins at
_E4M3_MAX_BYTE = 126    # its encoding; 127 is NaN

# The kernel's `inv_scales_max`: a group sitting at the tensor amax lands its block scale on
# 1/inv_scales_max. The headroom to 448 is where saturation is detected, and it is 448/384 =
# 1.167x rather than 1.75x for a group that picked the 1/4 candidate -- the figure to use if
# the margin is ever lowered towards 1.
_INV_SCALES_MAX = 1.0 / 256.0    # four_six_fp4_kernel<6.f, 4.f>; RNE's <6.f> uses 1/448

# Above ~256x the true amax, block scales start flushing to e4m3 zero and a whole group of 16
# dequantizes to zero silently; below it the error is flat for two decades. fp8's 1e3 default
# sits at that knee for O(1-10) post-norm activations, so seed lower. Overestimating is still
# the right direction -- the readback is invariant to the assumption, so the first update is
# exact from any seed that did not saturate.
_INIT_AMAX = 100.0


def assumed_amax(module):
    """The amax to hand `quant_fp4`, or None when delayed scaling is off for this module.

    It rides in the state's inverse-scale slot: the history holds real amaxes, and this role's
    `target` is 1.0, so `scale = 1 / (amax * margin)` and its reciprocal is the amax itself.
    """
    return getattr(module, f"{_PREFIX}_inv_{_ROLE}", None)


@torch.no_grad()
def record_amax(micro_scales, amax_in, amax_out):
    """Report this tensor's amax for a later step, reducing over the e4m3 block scales.

    The max runs in the *byte* domain: e4m3 is monotone in its unsigned encoding for
    non-negative values, and block scales are magnitudes, so an integer max over the raw bytes
    is exact. That keeps the reduction at one byte per 16 elements without depending on
    Inductor to fuse a widening cast into it.

    The clamp is not cosmetic. 127 is e4m3's NaN, and the history is a running max, so a single
    NaN block scale would pin every layer's scale at NaN for the rest of the run with no way
    back. Clamping to 126 (448, the largest finite) turns an unrecoverable state into an
    ordinary saturation reading, which update()'s search already handles.
    """
    byte_max = micro_scales.view(torch.uint8).amax().clamp(max=_E4M3_MAX_BYTE)
    reading = byte_max.view(torch.float8_e4m3fn).float() * amax_in * _INV_SCALES_MAX
    # max over the grad-accumulation micro-steps; DelayedScaleState.update zeroes it after
    amax_out.copy_(torch.maximum(amax_out, reading))


def enable_delayed_scaling(model, history_len=16, margin=2.0, allreduce=False):
    """Attach delayed scaling to every NVFP4Linear in model. Returns None if there are none.

    Must run before torch.compile: it registers the buffers the compiled graph reads.
    """
    from nanochat.sm120.nvfp4 import NVFP4Linear

    modules = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
    if not modules:
        return None
    if any(m.mode is not NVFP4QuantMode.FOUR_SIX for m in modules):
        # Rejected as scope, not as physics: RNE's inv_scales_max is 1/448, so its saturation
        # threshold sits at margin 1.0 where 4/6's sits at 1.75, and it needs a stricter margin
        # assertion plus a per-mode constant. Its readback is actually the *cleaner* of the two
        # (exact, where 4/6 carries the 1.0-1.5x candidate bias). --nvfp4-rne is an ablation
        # that takes accuracy away, so there is no caller to justify carrying that.
        raise ValueError("delayed NVFP4 scaling is only wired for 4/6 rounding; drop --nvfp4-rne")
    return DelayedScaleState(
        modules,
        roles=(_ROLE,),
        # target 1.0 makes the inverse-scale slot hold the amax itself, which is what the kernel
        # takes; a saturated block scale reads back 448 * inv_scales_max in those units.
        targets=[1.0],
        maxes=[_E4M3_MAX * _INV_SCALES_MAX],
        prefix=_PREFIX,
        history_len=history_len, margin=margin, allreduce=allreduce, init_amax=_INIT_AMAX,
    )
