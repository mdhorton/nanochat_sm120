"""
Test the FlexAttention training path against the existing SDPA path.

Run: python -m pytest tests/test_flex_attention.py -v -s

flex_attention only has a usable implementation under torch.compile, so every test here compiles the
call. That is also how it runs in training (inside the compiled GPT), so this matches production.
"""
import torch
import pytest

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn
from nanochat.flex_attn import flex_attn_func, BlockMaskCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")

DEVICE = "cuda"
DTYPE = torch.bfloat16
B, T, H, D = 2, 256, 4, 64
WINDOW = 128


def make_qkv(dtype, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(B, T, H, D, device=DEVICE, dtype=dtype, requires_grad=True) for _ in range(3)]


def fwd_bwd(fn, qkv, grad_out):
    """Run fn, backprop grad_out, return (output, dq, dk, dv) as fp32."""
    for t in qkv:
        t.grad = None
    y = fn(*qkv)
    y.backward(grad_out)
    return [t.float() for t in (y.detach(), qkv[0].grad, qkv[1].grad, qkv[2].grad)]


def dense_window_mask(window):
    """Independent reference mask: causal, plus a left window if window >= 0."""
    row = torch.arange(T, device=DEVICE).unsqueeze(1)
    col = torch.arange(T, device=DEVICE).unsqueeze(0)
    mask = col <= row
    if window >= 0:
        mask = mask & ((row - col) <= window)
    return mask


def sdpa_run(window):
    """The existing SDPA fallback path, forced on regardless of hardware."""
    def run(q, k, v):
        fa_module.set_impl('sdpa')
        try:
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))
        finally:
            fa_module.set_impl(None)
    return run


def fp32_reference(window):
    """Plain fp32 SDPA with an explicitly built mask. Independent of the code under test."""
    mask = dense_window_mask(window)

    def run(q, k, v):
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask).transpose(1, 2)
    return run


def block_masks_for(window_sizes, seq_len=T):
    return BlockMaskCache(window_sizes, seq_len, DEVICE).build()


def assert_no_worse_than_sdpa(window, block_mask, label, seed):
    """
    flex and SDPA are two different bf16 kernels, so they will not agree bit-for-bit and comparing
    them to each other just measures bf16 noise. Anchor both to an fp32 reference instead and
    require flex to be no less accurate than the path it replaces.
    """
    grad_out = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, generator=None)
    flex = torch.compile(flex_attn_func, dynamic=False)

    qkv32 = make_qkv(torch.float32, seed=seed)
    ref = fwd_bwd(fp32_reference(window), qkv32, grad_out.float())

    qkv = make_qkv(DTYPE, seed=seed)
    got_flex = fwd_bwd(lambda q, k, v: flex(q, k, v, block_mask), qkv, grad_out)
    got_sdpa = fwd_bwd(sdpa_run(window), qkv, grad_out)

    for name, r, f, s in zip(("out", "dq", "dk", "dv"), ref, got_flex, got_sdpa):
        scale = r.abs().mean().item()
        err_flex = (f - r).abs().mean().item()
        err_sdpa = (s - r).abs().mean().item()
        print(f"{label}/{name}: flex_err={err_flex:.6f} sdpa_err={err_sdpa:.6f} scale={scale:.4f}")
        assert err_flex <= 1.5 * err_sdpa + 1e-6, \
            f"{label}/{name}: flex ({err_flex:.6f}) is less accurate than SDPA ({err_sdpa:.6f})"
        assert err_flex < 0.02 * max(scale, 1e-3), \
            f"{label}/{name}: flex error {err_flex:.6f} is large relative to magnitude {scale:.6f}"


class TestFlexVsSDPA:

    def test_sliding_window(self):
        """flex with a sliding-window BlockMask reproduces windowed causal attention, fwd and bwd."""
        block_mask = block_masks_for([(WINDOW, 0)])[0]
        assert block_mask is not None, "a window shorter than seq_len must produce a BlockMask"
        assert_no_worse_than_sdpa(WINDOW, block_mask, "sliding_window", seed=0)

    def test_full_context(self):
        """A window covering the whole sequence must behave like plain causal attention."""
        # window = T-1 spans every causal position, so this is causal attention expressed as a window
        block_mask = block_masks_for([(T - 1, 0)])[0]
        assert_no_worse_than_sdpa(T - 1, block_mask, "full_context", seed=1)


class TestBlockMaskCache:

    def test_full_context_layers_get_none(self):
        """Full-context layers must fall through to SDPA/FA3: flex loses on dense causal."""
        masks = block_masks_for([(WINDOW, 0), (T, 0), (WINDOW, 0), (-1, 0)])
        assert masks[0] is not None
        assert masks[1] is None, "window == seq_len should not get a BlockMask"
        assert masks[2] is not None
        assert masks[3] is None, "window == -1 (full context) should not get a BlockMask"
        assert masks[0] is masks[2], "layers sharing a window must share one BlockMask object"

    def test_num_masked_layers(self):
        cache = BlockMaskCache([(WINDOW, 0), (T, 0), (WINDOW, 0)], T, DEVICE)
        assert cache.num_masked_layers() == 2

    def test_build_is_memoized(self):
        """Sliding-window masks are batch-independent, so they are built once for the whole run."""
        cache = BlockMaskCache([(WINDOW, 0)], T, DEVICE)
        assert cache.build() is cache.build()

    def test_block_sparsity(self):
        """The whole point: at a training-realistic shape a short window skips most 128x128 blocks."""
        seq_len = 2048  # T=256 is too small to show block sparsity: only 2x2 blocks exist
        mask = block_masks_for([(seq_len // 4, 0)], seq_len=seq_len)[0]
        assert mask.sparsity() > 65.0, f"expected a mostly-empty mask, got {mask.sparsity():.1f}%"


class TestNoRecompiles:

    def test_repeated_calls_do_not_recompile(self):
        """A recompile per step would silently cancel the entire speedup."""
        import torch._dynamo as dynamo
        qkv = make_qkv(DTYPE, seed=2)
        grad_out = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        block_mask = block_masks_for([(WINDOW, 0)])[0]
        flex = torch.compile(flex_attn_func, dynamic=False)

        fwd_bwd(lambda q, k, v: flex(q, k, v, block_mask), qkv, grad_out)
        graphs_after_first = dynamo.utils.counters['stats'].get('unique_graphs')
        for _ in range(3):
            fwd_bwd(lambda q, k, v: flex(q, k, v, block_mask), qkv, grad_out)
        assert dynamo.utils.counters['stats'].get('unique_graphs') == graphs_after_first


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
