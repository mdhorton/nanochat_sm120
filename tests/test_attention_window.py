"""Sliding-window attention on the flash kernels must match the mask path it replaces.

`flash_attn_func` routes a causal sliding window through aten::_flash_attention_forward instead
of building an explicit mask for SDPA (see nanochat/flash_attention.py). That is a different
kernel, so every test here compares the two arms directly rather than checking a shape.

Run: python -m pytest tests/test_attention_window.py -v -s
"""
import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn
from nanochat.sm120 import attention as win_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DTYPE = torch.bfloat16
# bf16 through two different kernels; the mask arm accumulates in a different order.
ATOL, RTOL = 2e-2, 2e-2


def set_windowed(mode):
    """'flash', 'mask', or None for auto."""
    win_module._override_windowed = mode


@pytest.fixture(autouse=True)
def _reset():
    fa_module._override_impl = 'sdpa'   # this file is about the non-FA3 path
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    # The fast path is opt-in, so importing the module no longer installs it. Without this every
    # both_arms() comparison would be the mask path against itself and would pass vacuously.
    win_module.install()
    assert win_module.is_installed(), "the flash arm is not installed; comparisons would be vacuous"
    yield
    # Uninstall rather than leak: there is no conftest.py, so a process-wide install here would
    # otherwise reach test_attention_fallback.py's 'sdpa' leg depending on collection order.
    win_module.uninstall()
    fa_module._override_impl = None
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    set_windowed(None)


def qkv(b, t, h, d, h_kv=None, grad=False):
    make = lambda n: torch.randn(b, t, n, d, device="cuda", dtype=DTYPE, requires_grad=grad)
    return make(h), make(h_kv or h), make(h_kv or h)


def both_arms(fn):
    """Run fn under the flash window and under the mask emulation."""
    torch.manual_seed(0)
    set_windowed('flash')
    got = fn()
    torch.manual_seed(0)
    set_windowed('mask')
    ref = fn()
    return got, ref


def assert_close(got, ref, name):
    diff = (got.float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
        f"{name}: max {diff.max().item():.4g} mean {diff.mean().item():.4g} (scale {scale:.4g})")


# =============================================================================
# The flash window and the mask emulation must agree
# =============================================================================
@pytest.mark.parametrize("window", [8, 64, 127])
@pytest.mark.parametrize("shape", [(2, 128, 4, 32), (1, 256, 8, 64), (2, 512, 6, 128)])
def test_forward_matches_mask(window, shape):
    q, k, v = qkv(*shape)
    got, ref = both_arms(lambda: flash_attn.flash_attn_func(
        q, k, v, causal=True, window_size=(window, 0)))
    assert_close(got, ref, f"fwd window={window} shape={shape}")


def test_backward_matches_mask():
    b, t, h, d, window = 2, 256, 4, 64, 32
    data = [x.detach() for x in qkv(b, t, h, d)]
    dout = torch.randn(b, t, h, d, device="cuda", dtype=DTYPE)

    def run():
        q, k, v = (x.clone().requires_grad_(True) for x in data)
        flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0)).backward(dout)
        return torch.stack([q.grad, k.grad, v.grad])

    got, ref = both_arms(run)
    for i, name in enumerate("qkv"):
        assert_close(got[i], ref[i], f"grad_{name}")


def test_gqa_matches_mask():
    """Fewer KV heads than Q heads -- flash handles GQA natively, SDPA needs enable_gqa."""
    q, k, v = qkv(2, 256, 8, 64, h_kv=2)
    got, ref = both_arms(lambda: flash_attn.flash_attn_func(
        q, k, v, causal=True, window_size=(32, 0)))
    assert_close(got, ref, "gqa fwd")


# =============================================================================
# The window must actually be applied, and only where it should be
# =============================================================================
def test_window_actually_masks():
    """A key outside the window must not influence the output.

    Guards against the window being silently dropped, which would leave every test above
    passing against a full-causal reference that is also wrong.
    """
    b, t, h, d, window = 1, 128, 2, 32, 16
    q, k, v = qkv(b, t, h, d)
    set_windowed('flash')
    base = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

    # perturb a key/value far outside the last query's window
    v2 = v.clone()
    v2[:, 0, :, :] += 10.0
    out = flash_attn.flash_attn_func(q, k, v2, causal=True, window_size=(window, 0))

    last = (out[:, -1] - base[:, -1]).abs().max().item()   # position 0 is outside its window
    first = (out[:, 0] - base[:, 0]).abs().max().item()    # position 0 attends to itself
    assert last == 0.0, f"window not applied: perturbing key 0 moved query {t-1} by {last:.4g}"
    assert first > 0.1, f"reference perturbation had no effect at all ({first:.4g})"


def test_full_context_keeps_the_sdpa_path():
    """window >= T is full context and must not be diverted; SDPA's is_causal path is equivalent."""
    q, k, v = qkv(2, 128, 4, 32)
    assert not win_module._windowed_ok(q, k, 128)
    assert not win_module._windowed_ok(q, k, -1)
    assert win_module._windowed_ok(q, k, 127)


def test_fp32_and_cpu_fall_back():
    """FA2 is half-precision only; the guard must reject everything else."""
    q = torch.randn(2, 128, 4, 32, device="cuda", dtype=torch.float32)
    assert not win_module._windowed_ok(q, q, 32)
    c = torch.randn(2, 128, 4, 32, dtype=torch.float32)
    assert not win_module._windowed_ok(c, c, 32)


# =============================================================================
# The install is the opt-in, and nothing may perform it behind your back
# =============================================================================
def test_uninstalled_takes_the_mask_path():
    """With the fast path off, flash_attn_func must fall back to SDPA's mask emulation."""
    b, t, h, d, window = 2, 128, 4, 32, 32
    q, k, v = qkv(b, t, h, d)

    win_module.install()
    set_windowed('flash')
    with_flash = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

    win_module.uninstall()
    assert not win_module.is_installed()
    assert fa_module._windowed_impl is None
    # _override_windowed is deliberately left at 'flash': with nothing installed it is unreachable,
    # so this proves the *install* is the gate rather than the override.
    uninstalled = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

    assert_close(uninstalled, with_flash, "uninstalled vs flash")


def test_importing_a_submodule_does_not_install():
    """Importing any sm120 submodule runs the package __init__; that must not opt you in.

    nvfp4 is the case that matters: `--nvfp4` imports nanochat.sm120.nvfp4, and a module-scope
    install would silently switch the attention implementation with it.
    """
    import importlib

    win_module.uninstall()
    importlib.import_module("nanochat.sm120.nvfp4")
    assert not win_module.is_installed(), "importing nanochat.sm120.nvfp4 installed windowed flash"
    assert fa_module._windowed_impl is None


# =============================================================================
# It has to survive torch.compile, which is how training runs it
# =============================================================================
def test_compiled_matches_eager():
    b, t, h, d, window = 2, 256, 4, 64, 32
    data = [x.detach() for x in qkv(b, t, h, d)]
    dout = torch.randn(b, t, h, d, device="cuda", dtype=DTYPE)
    set_windowed('flash')

    def step(fn, q, k, v):
        out = fn(q, k, v)
        out.backward(dout)
        return out.detach(), q.grad, k.grad, v.grad

    fn = lambda q, k, v: flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))
    eager = step(fn, *(x.clone().requires_grad_(True) for x in data))
    comp = step(torch.compile(fn, dynamic=False),
                *(x.clone().requires_grad_(True) for x in data))
    for name, a, b_ in zip(("out", "grad_q", "grad_k", "grad_v"), eager, comp):
        assert_close(b_, a, f"compiled {name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
