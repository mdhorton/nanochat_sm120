"""Test the natural-layout (NT) fp8 wgrad (nanochat.sm120.fp8_pinned, --wgrad-nt).

The NT plan reads `go` [k, m] and `x` [k, n] as they sit in memory instead of through the
transpose copies the TN form needs. The risk is entirely one of layout: an operand silently
interpreted as TN produces a valid-looking product of the wrong matrices, and the attention
wgrad is square (m == n), so the result still has the right shape and passes every check except
a value comparison. So every test here is a value comparison against the
transpose-copies-plus-_scaled_mm path the flag replaces.

Requires a GPU and JIT-builds csrc/pinned_gemm.cu on first use (~30 s).

python -m pytest tests/test_fp8_wgrad_nt.py -v -s
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

# Layer dims (in_features, out_features) at d12, with a 2048-token micro-batch. The wgrad sees
# (m, n, k) = (out_features, in_features, tokens) -- derived rather than written out, because
# getting that triple backwards is exactly the failure these tests exist to catch.
TOKENS = 2048
LAYERS = {"attn": (768, 768), "cfc": (768, 3072), "cproj": (3072, 768)}


def wgrad_mnk(k_in, n_out):
    """dW[N,K] = dy[T,N].T @ x[T,K]."""
    return n_out, k_in, TOKENS


@pytest.fixture(autouse=True)
def reset_flag():
    """Each test sets the flag itself; leaking it into the next would silently skip the NT path."""
    yield
    fp8_pinned.configure_wgrad_nt(False)


def nt_operands(m, n, k):
    """Natural-layout wgrad operands: go [k, m] e5m2 and x [k, n] e4m3, both row-major."""
    go, gi = _to_fp8(torch.randn(k, m, device=DEVICE, dtype=torch.bfloat16), E5M2)
    x, xi = _to_fp8(torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16), E4M3)
    return go, x, gi, xi


def tn_reference(go, x, gi, xi, out_dtype=torch.bfloat16):
    return torch._scaled_mm(go.t().contiguous(), _to_col_major(x), scale_a=gi, scale_b=xi,
                            out_dtype=out_dtype, use_fast_accum=False)


def rel_err(got, ref):
    return float((got.float() - ref.float()).abs().max() / ref.abs().max().clamp(min=1e-6))


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_wgrad_nt_matches_tn(layer):
    """The NT wgrad against the transpose copies + _scaled_mm it replaces, every layer shape."""
    fp8_pinned.configure_wgrad_nt(True)
    m, n, k = wgrad_mnk(*LAYERS[layer])
    go, x, gi, xi = nt_operands(m, n, k)
    ref = tn_reference(go, x, gi, xi)
    got = fp8_pinned.mm_wgrad_nt(go, x, gi, xi, torch.bfloat16)
    assert got.shape == ref.shape == (m, n)
    assert float(ref.abs().max()) > 0        # a vacuous comparison proves nothing
    assert rel_err(got, ref) < 0.05


def test_wgrad_nt_square_would_catch_a_transpose():
    """Teeth check: the square attn wgrad, where a swapped result still has the right shape."""
    fp8_pinned.configure_wgrad_nt(True)
    m, n, k = wgrad_mnk(*LAYERS["attn"])
    assert m == n
    go, x, gi, xi = nt_operands(m, n, k)
    ref = tn_reference(go, x, gi, xi)
    got = fp8_pinned.mm_wgrad_nt(go, x, gi, xi, torch.bfloat16)
    assert rel_err(got, ref) < 0.05
    assert rel_err(ref.t().contiguous(), ref) > 0.05


def test_wgrad_nt_falls_back_for_a_non_bf16_out():
    """The pinned kernel only produces bf16; anything else must take the TN path, not silently
    return bf16."""
    fp8_pinned.configure_wgrad_nt(True)
    m, n, k = wgrad_mnk(*LAYERS["cproj"])
    go, x, gi, xi = nt_operands(m, n, k)
    got = fp8_pinned.mm_wgrad_nt(go, x, gi, xi, torch.float32)
    ref = tn_reference(go, x, gi, xi, out_dtype=torch.float32)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)


def test_wgrad_nt_through_matmul_eager_and_compiled():
    """_Float8Matmul with the flag on: same weight grad without the copies, and the dgrad --
    which the flag must not touch -- stays bitwise identical. Non-square layer, catches a swap."""
    from nanochat.fp8 import _Float8Matmul
    torch.manual_seed(0)
    k_in, n_out = LAYERS["cfc"]
    x0 = torch.randn(TOKENS, k_in, device=DEVICE, dtype=torch.bfloat16)
    w0 = torch.randn(n_out, k_in, device=DEVICE, dtype=torch.bfloat16) * 0.02
    r = torch.randn(TOKENS, n_out, device=DEVICE, dtype=torch.bfloat16)

    def grads(nt, compiled=False):
        fp8_pinned.configure_wgrad_nt(nt)
        x = x0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        fn = lambda a, b: _Float8Matmul.apply(a, b)  # noqa: E731
        if compiled:
            fn = torch.compile(fn)
        (fn(x, w) * r).sum().backward()
        fp8_pinned.configure_wgrad_nt(False)
        return x.grad, w.grad

    gx_tn, gw_tn = grads(nt=False)
    gx_nt, gw_nt = grads(nt=True)
    assert float(gw_tn.abs().max()) > 0
    assert torch.equal(gx_tn, gx_nt)         # dgrad path untouched by the flag
    assert rel_err(gw_nt, gw_tn) < 0.05

    gx_c, gw_c = grads(nt=True, compiled=True)
    assert float(gw_c.abs().max()) > 0
    assert torch.equal(gx_c, gx_nt)
    assert rel_err(gw_c, gw_tn) < 0.05


def test_the_extension_links_cublaslt_13():
    """Half of what the NT wgrad is worth is that the extension links the *system* cuBLASLt:
    under 12.x every fp8 GEMM lands on sm89_xmma_* (Ada kernels on a Blackwell card) instead of
    nvjet_sm120_*. A build that picked torch's toolkit runs, and reads low, and says nothing."""
    import subprocess

    so = fp8_pinned._ext().__file__
    needed = subprocess.check_output(["objdump", "-p", so], text=True)
    libs = [ln.split()[-1] for ln in needed.splitlines() if "NEEDED" in ln]
    assert any(lib.startswith("libcublasLt.so.13") for lib in libs), libs
