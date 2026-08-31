"""Windowed flash attention: the fast path SDPA cannot express.

F.scaled_dot_product_attention has no window parameter, so nanochat/flash_attention.py's
_sdpa_attention emulates a sliding window with an explicit [Tq, Tk] mask -- and passing any
mask drops SDPA off its flash backend. Measured on sm120 at the d12 training shape that costs
4.6x a full-causal flash forward, which is why the README's shortrun tables first recorded
`--window-pattern L` as *faster* than the SSSL default despite SSSL doing 58% of the work.

aten::_flash_attention_forward takes window_size_left directly and dispatches to the same
pytorch_flash kernels SDPA's fast path already uses. It is not autograd-aware, so the
forward/backward pair is wrapped below.

Importing this module installs `windowed_flash` as flash_attention._windowed_impl. The
dependency only points this way: flash_attention.py knows nothing about nanochat.sm120.
"""
import torch

import nanochat.flash_attention as _fa

# Override for testing: set to 'flash', 'mask', or None (auto). See _windowed_ok.
_override_windowed = None


def _windowed_ok(q, k, window):
    """Only the training shape on a CUDA half dtype; everything else keeps the old path.

    Tq != Tk is chunked prefill, where the window is not aligned to the query positions, and
    window >= Tq is full context -- which SDPA already sends to the same kernels via is_causal.
    """
    if _override_windowed == "mask":
        return False
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.size(1) != k.size(1):
        return False
    return 0 <= window < q.size(1)


@torch._dynamo.allow_in_graph
class _WindowedFlashAttn(torch.autograd.Function):
    """aten::_flash_attention_forward/backward as one autograd op, with a left window.

    `allow_in_graph` so torch.compile treats this as a single node and AOTAutograd traces the
    pair, the same arrangement nanochat/fp8.py uses for _Float8Matmul.
    """

    @staticmethod
    def forward(ctx, q, k, v, window):
        seq = q.size(1)
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            q, k, v, None, None, seq, seq, 0.0, True, False,
            window_size_left=window, window_size_right=0)
        ctx.save_for_backward(q, k, v, out, lse, rng, unused)
        ctx.window = window
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, rng, unused = ctx.saved_tensors
        seq = q.size(1)
        # FA2's backward reads grad_out with the same strides as out.
        grad_out = grad_out.contiguous()
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out, q, k, v, out, lse, None, None, seq, seq, 0.0, True, rng, unused,
            window_size_left=ctx.window, window_size_right=0)
        return dq, dk, dv, None


def windowed_flash(q, k, v, window):
    """The _windowed_impl contract: the windowed result, or None for "not applicable".

    Returning None rather than raising keeps the predicate here, so flash_attention.py needs no
    knowledge of which shapes and dtypes the aten op accepts.
    """
    if not _windowed_ok(q, k, window):
        return None
    return _WindowedFlashAttn.apply(q, k, v, window)


_fa._windowed_impl = windowed_flash
