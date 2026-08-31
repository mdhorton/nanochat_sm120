"""Amax-history scaling state, shared by the fp8 and NVFP4 paths.

Dynamic scaling cannot begin the cast until it knows the amax, so every quantized tensor is read
twice. Delayed scaling quantizes with a scale derived from *previous* steps' amaxes, held in a
buffer: the cast becomes pure pointwise and the reduction leaves the critical path.

Nothing here is precision-specific. A caller supplies its roles, the amax each role's scale should
aim at (`targets`) and the value a saturated reading pins at (`maxes`), and reads the amax back
from whatever cheap by-product its own kernel leaves behind:

  fp8   the 1-byte quantized output  (fp8_state._record_amax)
  nvfp4 the per-16 e4m3 block scales (nvfp4_state)

Two details are load-bearing, both from dev/perf-log.md experiment 10:
  - The reading must come from a *cheaper* tensor than the source. Reducing over the source pays
    the same bytes as dynamic scaling and nets exactly nothing.
  - Scale state must be a registered buffer, i.e. a graph input, never a Python-side cache, or
    the second compile dies with "Mixing fake modes NYI".

Layout: two tensors per model, [2 fields, R roles, L] of scales and [R roles, L] of amaxes, so the
once-per-step update touches all L layers in a handful of kernels instead of R per layer. Each
layer is handed 3R 0-d *views* into them, one buffer each, rather than one indexed tensor: a scale
must reach the graph as a plain input, because Inductor's tuned_scaled_mm lowering cannot take an
in-graph select as scale_a.
"""
import torch
import torch.distributed as dist

from nanochat.fp8 import EPS

_SCALE, _INV, _AMAX = 0, 1, 2                # fields of a role's (scale, inv_scale, amax)

# Step of the two-way search update() runs when a reading is uninformative (the cast saturated,
# or it underflowed to nothing).
_AMAX_SEARCH = 2.0
# Assumed amax before any measurement exists. Deliberately high: overestimating only costs
# precision on step 0 (small values underflow) and still reads back an accurate amax, so the
# scale is right from step 1, whereas underestimating saturates and needs several search steps.
_INIT_AMAX = 1e3


class DelayedScaleState:
    """Model-wide scale and amax history. One per model, before torch.compile.

    Call update() once per optimizer step, after the last backward.

    `roles` names the quantized tensors of a layer. `targets[r]` is the reading a correctly
    scaled role lands on, and `maxes[r]` the reading a saturated one pins at; they differ so
    saturation stays distinguishable from a tensor sitting exactly at its historical amax --
    without that headroom the search fires on every exactly-scaled tensor and oscillates the
    scale with the history's period.

    `prefix` names the buffers on each module, `{prefix}_{field}_{role}`. `seed_role` is the one
    role whose amax is already known at construction (fp8's weight); everything else starts at
    `init_amax`, which a caller lowers when its format has a ceiling as well as a floor.

    margin is headroom against the amax growing between steps, and is close to free for a
    floating-point format: aiming below the max costs range at the bottom rather than mantissa
    bits, whereas clipping costs real error and trips the search.
    """

    def __init__(self, modules, roles, targets, maxes, prefix,
                 history_len=16, margin=2.0, allreduce=False, seed_role=None,
                 init_amax=_INIT_AMAX):
        assert history_len >= 1 and margin >= 1.0
        assert len(roles) == len(targets) == len(maxes)
        device = modules[0].weight.device
        num, nrole = len(modules), len(roles)
        self.roles = tuple(roles)
        self.prefix = prefix
        self.history_len = history_len
        self.margin = margin
        self.allreduce = allreduce
        self.idx = 0
        self.filled = 0  # history slots holding a real measurement; the rest are not yet valid
        self.scales = torch.ones(2, nrole, num, device=device, dtype=torch.float32)
        self.amax = torch.zeros(nrole, num, device=device, dtype=torch.float32)
        self.hist = torch.zeros(history_len, nrole, num, device=device, dtype=torch.float32)
        self.target = torch.tensor([[t] for t in targets],
                                   device=device, dtype=torch.float32)  # [R roles, 1]
        self.max = torch.tensor([[m] for m in maxes],
                                device=device, dtype=torch.float32)     # [R roles, 1]
        # Seed the *scales*, deliberately not the history: the history is a max, so a seeded slot
        # would outlive its usefulness by history_len steps and hold every role at the assumed
        # amax for the whole window.
        seed = torch.full((nrole, num), init_amax, device=device, dtype=torch.float32)
        if seed_role is not None:
            r = self.roles.index(seed_role)
            with torch.no_grad():
                for i, mod in enumerate(modules):
                    # An all-zero weight is a real initialisation here (gpt.py zero-inits both
                    # c_proj), and its amax says nothing about what it will be one step later, so
                    # fall back to the assumption rather than seeding a scale of zero.
                    amax = mod.weight.float().abs().max()
                    if amax > 0:
                        seed[r, i] = amax
        self._set_scales(seed)
        for i, mod in enumerate(modules):
            for r, role in enumerate(self.roles):
                setattr(mod, f"{prefix}_scale_{role}", self.scales[_SCALE, r, i])
                setattr(mod, f"{prefix}_inv_{role}", self.scales[_INV, r, i])
                setattr(mod, f"{prefix}_amax_{role}", self.amax[r, i])

    @torch.no_grad()
    def _set_scales(self, h):
        """Point every scale at amax h ([R roles, num]), less the margin and target headroom."""
        scale = self.scales[_SCALE]
        # Where a role has no measurement at all, keep whatever scale it already had
        scale.copy_(torch.where(h > 0, self.target / (h * self.margin).clamp(min=EPS), scale))
        self.scales[_INV].copy_(scale.reciprocal())

    @torch.no_grad()
    def update(self):
        """Fold this step's amaxes into the history and recompute every scale.

        The hot path reports the amax in source units, which is exact whenever nothing clipped.
        The two cases where it is not are handled here, because both need a scale to compare
        against.
        """
        if self.allreduce:
            dist.all_reduce(self.amax, op=dist.ReduceOp.MAX)  # contiguous, NCCL takes it directly
        scale = self.scales[_SCALE]
        raw = self.amax * scale                             # back to reading units
        implied = self.target / (scale * self.margin)       # the amax this scale is aimed at
        # A saturated cast pins the reading at its max, which is a floor on the true amax rather
        # than a reading of it, so search up past it -- otherwise a clipped tensor reports the
        # same value forever and the scale never recovers. A cast that underflowed to nothing
        # reports 0, which bounds the amax from above instead; searching down means shrinking the
        # whole window, since the history is a max and would otherwise pin the scale.
        blind = raw <= 0
        self.hist.mul_(torch.where(blind, 1.0 / _AMAX_SEARCH, 1.0))
        est = torch.where(raw >= self.max, self.amax * _AMAX_SEARCH, self.amax)
        self.hist[self.idx].copy_(torch.where(blind, implied / _AMAX_SEARCH, est))
        self.idx = (self.idx + 1) % self.history_len
        self.filled = min(self.filled + 1, self.history_len)
        self.amax.zero_()
        self._set_scales(self.hist[:self.filled].amax(dim=0))
