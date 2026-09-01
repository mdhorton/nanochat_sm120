"""Test the pinned cuBLASLt algorithms (nanochat.sm120.fp8_pinned, --pin-gemm).

Two risks, and they are different in kind. The mode gate (`_pins`) decides which GEMMs are
routed at all, and getting it wrong is silent -- a mode that pins nothing still prints its
enablement line and reads ~2.5% low. The plan itself is a substituted algorithm, so it must
match `_scaled_mm` on the same operands to within reduction-order noise, and the `fast_accum`
axis must follow the role rather than the plan cache.

Requires a GPU and JIT-builds csrc/pinned_gemm.cu on first use (~30 s).

python -m pytest tests/test_fp8_pin_gemm.py -v -s
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="fp8 tests require CUDA")

if cuda_available:
    from nanochat.fp8 import _to_col_major, _to_fp8
    from nanochat.sm120 import fp8_pinned

DEVICE = "cuda"
E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2

# d12 layer dims (in_features, out_features) with a 2048-token micro-batch. attn is square,
# which is what --pin-gemm attn selects on.
TOKENS = 2048
LAYERS = {"attn": (768, 768), "cfc": (768, 3072), "cproj": (3072, 768)}


@pytest.fixture(autouse=True)
def reset_mode():
    """Each test sets the mode itself; leaking it would silently skip or add a pinned path."""
    yield
    fp8_pinned.configure("off")


def fwd_operands(k_in, n_out):
    """in [T, K] row-major and w.t() [K, N] column-major, exactly as _Float8Matmul builds them."""
    a, ai = _to_fp8(torch.randn(TOKENS, k_in, device=DEVICE, dtype=torch.bfloat16), E4M3)
    w, wi = _to_fp8(torch.randn(n_out, k_in, device=DEVICE, dtype=torch.bfloat16) * 0.02, E4M3)
    return a, w.t(), ai, wi


def wgrad_operands(k_in, n_out):
    """The TN wgrad operands: go.T [N, T] row-major and in [T, K] column-major."""
    go, gi = _to_fp8(torch.randn(TOKENS, n_out, device=DEVICE, dtype=torch.bfloat16), E5M2)
    x, xi = _to_fp8(torch.randn(TOKENS, k_in, device=DEVICE, dtype=torch.bfloat16), E4M3)
    return go.t().contiguous(), _to_col_major(x), gi, xi


def rel_err(got, ref):
    return float((got.float() - ref.float()).abs().max() / ref.abs().max().clamp(min=1e-6))


def test_pins_gate_by_mode():
    """The mode gate, which decides what is routed at all. 768x768 is the square attn wgrad."""
    cases = {
        "off":   {("fwd", 2048, 768): False, ("wgrad", 768, 768): False,
                  ("wgrad", 3072, 768): False, ("dgrad", 2048, 3072): False},
        "attn":  {("fwd", 2048, 768): False, ("wgrad", 768, 768): True,
                  ("wgrad", 3072, 768): False, ("dgrad", 2048, 3072): False},
        "wgrad": {("fwd", 2048, 768): False, ("wgrad", 768, 768): True,
                  ("wgrad", 3072, 768): True, ("dgrad", 2048, 3072): False},
        "all":   {("fwd", 2048, 768): True, ("wgrad", 768, 768): True,
                  ("wgrad", 3072, 768): True, ("dgrad", 2048, 3072): True},
    }
    for mode, expected in cases.items():
        fp8_pinned.configure(mode)
        for (role, m, n), want in expected.items():
            assert fp8_pinned._pins(role, m, n) is want, (mode, role, m, n)


def test_pins_rejects_an_output_too_big_to_verify():
    """Verification holds the output twice, so lm_head fwd (16384x32768 bf16, 1.07 GB) costs
    far more peak memory than a pin can win. Everything else at d12/d16 is under 100 MB."""
    fp8_pinned.configure("all")
    assert not fp8_pinned._pins("fwd", 16384, 32768)
    assert fp8_pinned._pins("fwd", 2048, 3072)


def test_enabled_tracks_either_flag():
    """report_once gates on this: --pin-gemm alone must not read as a no-op run."""
    assert not fp8_pinned.enabled()
    fp8_pinned.configure("attn")
    assert fp8_pinned.enabled()
    fp8_pinned.configure("off")
    assert not fp8_pinned.enabled()


def test_mode_off_is_bitwise_scaled_mm():
    """With --pin-gemm off nothing may change: same kernel, same operands, same bits."""
    fp8_pinned.configure("off")
    a, b, ai, bi = fwd_operands(*LAYERS["cfc"])
    got = fp8_pinned.mm(a, b, ai, bi, torch.bfloat16, "fwd")
    ref = torch._scaled_mm(a, b, scale_a=ai, scale_b=bi, out_dtype=torch.bfloat16,
                           use_fast_accum=True)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_pinned_fwd_matches_scaled_mm(layer):
    fp8_pinned.configure("all")
    a, b, ai, bi = fwd_operands(*LAYERS[layer])
    got = fp8_pinned.mm(a, b, ai, bi, torch.bfloat16, "fwd")
    ref = torch._scaled_mm(a, b, scale_a=ai, scale_b=bi, out_dtype=torch.bfloat16,
                           use_fast_accum=True)
    assert float(ref.abs().max()) > 0        # a vacuous comparison proves nothing
    assert rel_err(got, ref) < 0.05


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_pinned_wgrad_matches_scaled_mm(layer):
    """The wgrad shapes, where the donor measured the heuristic losing 15-42%. attn is square,
    so a transposed result would still have the right shape -- hence a value comparison."""
    fp8_pinned.configure("wgrad")
    k_in, n_out = LAYERS[layer]
    a, b, ai, bi = wgrad_operands(k_in, n_out)
    got = fp8_pinned.mm(a, b, ai, bi, torch.bfloat16, "wgrad")
    ref = torch._scaled_mm(a, b, scale_a=ai, scale_b=bi, out_dtype=torch.bfloat16,
                           use_fast_accum=False)
    assert got.shape == ref.shape == (n_out, k_in)
    assert float(ref.abs().max()) > 0
    assert rel_err(got, ref) < 0.05


def test_fast_accum_follows_the_role_not_the_cache():
    """A plan is tuned for one fast_accum setting, so the two must be separate cache entries:
    fwd fast-accumulates, dgrad and wgrad do not. Same shape, deliberately."""
    fp8_pinned.configure("all")
    a, b, ai, bi = fwd_operands(768, 768)
    fwd = fp8_pinned.mm(a, b, ai, bi, torch.bfloat16, "fwd")
    dgrad = fp8_pinned.mm(a, b, ai, bi, torch.bfloat16, "dgrad")
    m, k = a.shape
    n = b.shape[1]
    assert (m, n, k, a.dtype, b.dtype, True) in fp8_pinned._PLANS
    assert (m, n, k, a.dtype, b.dtype, False) in fp8_pinned._PLANS
    assert rel_err(fwd, dgrad) < 0.05        # different accumulation, same product


def test_falls_back_for_a_non_bf16_out():
    """The pinned kernel only produces bf16; anything else must take _scaled_mm, not silently
    return bf16."""
    fp8_pinned.configure("all")
    a, b, ai, bi = fwd_operands(*LAYERS["cproj"])
    got = fp8_pinned.mm(a, b, ai, bi, torch.float32, "fwd")
    ref = torch._scaled_mm(a, b, scale_a=ai, scale_b=bi, out_dtype=torch.float32,
                           use_fast_accum=True)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)


def test_through_matmul_eager_and_compiled():
    """_Float8Matmul end to end with the mode on: both grads match the unpinned run, eager and
    compiled. Non-square layer, which catches an operand swap.

    The SM120 backend has to be installed for this to be worth anything -- the stock
    Float8Backend never reaches fp8_pinned, so without it every comparison here would be
    unpinned against unpinned and pass on a no-op. `_PLANS` is checked at the end for that.
    """
    import nanochat.fp8 as fp8mod
    from nanochat.fp8 import _Float8Matmul
    from nanochat.sm120 import install_fp8_backend

    saved_backend = fp8mod._backend
    install_fp8_backend()
    torch.manual_seed(0)
    k_in, n_out = LAYERS["cfc"]
    x0 = torch.randn(TOKENS, k_in, device=DEVICE, dtype=torch.bfloat16)
    w0 = torch.randn(n_out, k_in, device=DEVICE, dtype=torch.bfloat16) * 0.02
    r = torch.randn(TOKENS, n_out, device=DEVICE, dtype=torch.bfloat16)

    def grads(mode, compiled=False):
        fp8_pinned.configure(mode)
        x = x0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        fn = lambda a, b: _Float8Matmul.apply(a, b)  # noqa: E731
        if compiled:
            fn = torch.compile(fn)
        (fn(x, w) * r).sum().backward()
        fp8_pinned.configure("off")
        return x.grad, w.grad

    try:
        gx_off, gw_off = grads("off")
        assert float(gw_off.abs().max()) > 0
        before = len(fp8_pinned._PLANS)
        for compiled in (False, True):
            gx, gw = grads("all", compiled=compiled)
            assert rel_err(gx, gx_off) < 0.05, compiled
            assert rel_err(gw, gw_off) < 0.05, compiled
        # fwd, dgrad and wgrad shapes, so at least one plan is new unless nothing was routed.
        assert len(fp8_pinned._PLANS) > before
    finally:
        fp8mod.set_backend(saved_backend)
