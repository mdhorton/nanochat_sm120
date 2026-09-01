"""The sm120 FP8 backend: what nanochat/fp8.py calls instead of its stock primitives.

Implements nanochat.fp8.Float8Backend, adding two things to the upstream path:

  delayed / spiked scaling  quantize() reads this role's scale from DelayedScaleState instead
                            of reducing the tensor first  (--fp8-scaling, +10.7% at d12)
  natural-layout wgrad      wgrad() reads both operands as they sit, so the go_T / in_col
                            transpose copies never enter the graph  (--wgrad-nt, +8.4% at d12)

fwd and dgrad are inherited unchanged, and so is the TN wgrad, so with nothing enabled this
backend runs upstream's own path. sm120_nanochat@refactor also substitutes tuned cuBLASLt
algorithms and cached weight casts at these same seams -- see TODO.md.
"""
import torch

from nanochat.fp8 import Float8Backend
from nanochat.sm120 import fp8_pinned
from nanochat.sm120.fp8_state import _ROLE_NAMES, to_fp8

# quantize() is called with a role name; these pick that role's slice of the layer's state.
_ROLE_SLOT = {"in": 0, "w": 1, "go": 2}


class SM120Backend(Float8Backend):
    """Delayed fp8 scaling. See dev/LOG_sm120.md for what it is worth."""

    def init_layer(self, layer):
        """Register the buffers enable_delayed_scaling may later fill.

        All None by default and all non-persistent, so a layer that never gets an enable_*
        call behaves exactly like the stock backend, and checkpoints stay interchangeable with
        a bf16 run. Dynamo guards on which of these is None, so turning the feature on costs
        one recompile and nothing per step.
        """
        # Nine 0-d views into the model-wide DelayedScaleState. Each is a separate buffer
        # rather than one indexed tensor: a scale must reach the graph as a plain input,
        # because Inductor's tuned_scaled_mm lowering cannot take an in-graph select.
        for role in _ROLE_NAMES:
            for field in ("scale", "inv", "amax"):
                layer.register_buffer(f"fp8_{field}_{role}", None, persistent=False)

    def layer_state(self, layer):
        """The opaque bundle nanochat/fp8.py threads through to the calls below.

        None whenever delayed scaling is off, which keeps the stock guard and the stock graph
        -- and is also the state --fp8-scaling static-spike runs in, since the spike derives
        its scale from the tensor rather than from here.
        """
        if layer.fp8_scale_in is None:
            return None
        return (
            (layer.fp8_scale_in, layer.fp8_inv_in, layer.fp8_amax_in),
            (layer.fp8_scale_w, layer.fp8_inv_w, layer.fp8_amax_w),
            (layer.fp8_scale_go, layer.fp8_inv_go, layer.fp8_amax_go),
        )

    @staticmethod
    def _scale(state, role):
        return None if state is None else state[_ROLE_SLOT[role]]

    def quantize(self, x, fp8_dtype, role, state):
        return to_fp8(x, fp8_dtype, self._scale(state, role))

    def weight_operands(self, weight, state):
        w_fp8, w_inv = to_fp8(weight, torch.float8_e4m3fn, self._scale(state, "w"))
        return w_fp8, w_inv, None

    def wgrad(self, go_fp8, in_fp8, go_inv, in_inv, out_dtype, state):
        # Natural-layout (NT) wgrad: sm120's cuBLASLt reads both operands as they sit, so the
        # go_T / in_col transpose copies super() builds never materialize -- they were 4.6% of
        # a training step as pure-copy kernels. The predicate is read at trace time, so with
        # the flag off the compiled graph is upstream's, branch and all.
        if fp8_pinned.wgrad_nt():
            return fp8_pinned.mm_wgrad_nt(go_fp8, in_fp8, go_inv, in_inv, out_dtype)
        return super().wgrad(go_fp8, in_fp8, go_inv, in_inv, out_dtype, state)
