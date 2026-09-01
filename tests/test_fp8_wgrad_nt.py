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
    from nanochat.sm120 import fp8_pinned, gemm_cache

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
    gemm_cache.reset()


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


# --- plan cache (--gemm-plan-cache) and exhaustive autotune -------------------------------
#
# These call _build_wgrad_nt_plan directly rather than going through mm_wgrad_nt: it bypasses
# the _PLANS memo, so a shape can be built twice in one process, which is the whole point.
# The shape is small and NON-SQUARE, so a transpose bug cannot hide behind m == n.

CACHE_MNK = (256, 512, 512)


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    """Point the cache at a private file. Patches gemm_cache.path rather than NANOCHAT_BASE_DIR,
    which also names the JIT build directory -- moving that would rebuild the extension."""
    import json

    f = tmp_path / "plans.json"
    monkeypatch.setattr(gemm_cache, "path", lambda: str(f))

    class Handle:
        path = f

        def read(self):
            return json.loads(f.read_text())

        def write(self, doc):
            f.write_text(json.dumps(doc))

        def plans(self):
            return self.read()["plans"]

        def only(self):
            plans = self.plans()
            assert len(plans) == 1, f"expected exactly one entry, got {list(plans)}"
            return next(iter(plans.values()))

    yield Handle()


def build(cache="off", exhaustive=False, max_candidates=512, mnk=CACHE_MNK):
    """Build one plan and return (plan, its log line).

    Warnings -- a stale entry, a slow re-time -- are logged ahead of the plan's own line, so the
    line of interest is the last one. Tests that care about a warning read fp8_pinned._LOG, which
    stays intact until the next build.
    """
    fp8_pinned.configure_plan_cache(cache, exhaustive=exhaustive, max_candidates=max_candidates)
    go, x, gi, xi = nt_operands(*mnk)
    fp8_pinned._LOG.clear()
    plan = fp8_pinned._build_wgrad_nt_plan(go, x, gi, xi)
    assert plan is not None, "the plan failed verification outright"
    assert fp8_pinned._LOG, "a built plan must log how it was obtained"
    return plan, fp8_pinned._LOG[-1]


def test_cache_round_trip(cache_file):
    """Tune, persist, restore -- and the restored config must match key for key.

    The equality is what catches a dropped axis: inner_shape and cluster_shape are uint16_t and
    reading them at the wrong width returns -1 silently, which would restore a *different*
    kernel that still passes every numeric check below.
    """
    plan, line = build(cache="use")
    assert "TUNED heuristic" in line
    stored = cache_file.only()
    assert stored["tier"] == "heuristic"
    assert stored["algo"] == {k: int(v) for k, v in plan.config().items()}

    plan2, line2 = build(cache="use")
    assert "CACHED heuristic" in line2
    assert "TUNED" not in line2
    assert plan2.config() == plan.config(), "restore must reproduce the config exactly"
    # The re-timing must be comparable to the stored one, i.e. measured the same way. Timing
    # run() instead of the bare launch loop would trip this on every small shape.
    assert not any("re-times" in ln for ln in fp8_pinned._LOG), fp8_pinned._LOG

    # and it still computes the right thing
    go, x, gi, xi = nt_operands(*CACHE_MNK)
    assert rel_err(plan2.run(go, x, gi, xi), tn_reference(go, x, gi, xi)) < 0.05


def test_tier_upgrade_retunes(cache_file):
    """A heuristic entry must not satisfy an exhaustive request -- the point of the tier."""
    build(cache="use")
    first = cache_file.only()
    assert first["tier"] == "heuristic"

    _, line = build(cache="use", exhaustive=True, max_candidates=128)
    assert "TUNED exhaustive" in line, "an exhaustive request replayed a heuristic-tier plan"
    upgraded = cache_file.only()
    assert upgraded["tier"] == "exhaustive"
    assert upgraded["budget"]["max_candidates"] == 128

    # ...and the upgraded entry now serves a plain request, since upgrades are monotone.
    _, line = build(cache="use")
    assert "CACHED exhaustive" in line


def test_stale_entry_self_heals_on_bad_algo_id(cache_file):
    """A retired algo id (the shape a cuBLAS bump takes) must warn, re-tune and overwrite."""
    build(cache="use")
    doc = cache_file.read()
    key = next(iter(doc["plans"]))
    doc["plans"][key]["algo"]["algo_id"] = 9999
    cache_file.write(doc)

    _, line = build(cache="use")
    assert "TUNED" in line, "a stale entry must fall through to a fresh tune"
    assert any("STALE" in ln for ln in fp8_pinned._LOG), "the self-heal must be visible"
    assert cache_file.only()["algo"]["algo_id"] != 9999, "the bad entry must be replaced"


def test_stale_entry_self_heals_on_unsupported_config(cache_file):
    """AlgoInit accepts the id but AlgoCheck rejects the config -- a distinct failure gate."""
    build(cache="use")
    doc = cache_file.read()
    key = next(iter(doc["plans"]))
    doc["plans"][key]["algo"]["tile"] = 3       # a real tile id this algo will not support
    cache_file.write(doc)

    _, line = build(cache="use")
    assert "TUNED" in line
    assert any("STALE" in ln for ln in fp8_pinned._LOG)
    assert cache_file.only()["algo"]["tile"] != 3


def test_default_writes_nothing(cache_file):
    """No flags: no file, no CACHED line. The default-behaviour-unchanged guarantee."""
    _, line = build(cache="off")
    assert not cache_file.path.exists(), "the cache must be inert unless asked for"
    assert "TUNED heuristic" in line and "CACHED" not in line


def test_refresh_retunes_but_persists(cache_file):
    build(cache="use")
    _, line = build(cache="refresh")
    assert "TUNED" in line, "refresh must ignore what is stored"
    assert cache_file.only()["tier"] == "heuristic", "refresh must still write"


def test_exhaustive_enumerates_a_superset():
    """Exhaustive must genuinely search wider, and never land below cuBLAS's own pick."""
    go, x, gi, xi = nt_operands(*CACHE_MNK)
    m, n, k = CACHE_MNK
    plan = fp8_pinned._ext().PinnedGemm(m, n, k, go.dtype, x.dtype, False, accum=False, nt=True)
    info = plan.autotune(go, x, gi, xi, 64, 20, 8, 5, 256)

    assert info["ids"] > 0
    assert info["enumerated"] > info["heuristic_count"]
    assert info["kept"] > info["heuristic_count"], "enumeration added no candidates"
    # The heuristic's own pick is always seeded, so the winner can only tie or beat it.
    assert info["us"] <= info["heuristic_us"] * 1.05


def test_exhaustive_is_bounded():
    """The cap is the mitigation for a cross product that is ~10^5 wide before it."""
    go, x, gi, xi = nt_operands(*CACHE_MNK)
    m, n, k = CACHE_MNK
    plan = fp8_pinned._ext().PinnedGemm(m, n, k, go.dtype, x.dtype, False, accum=False, nt=True)
    info = plan.autotune(go, x, gi, xi, 64, 20, 8, 5, 128)
    assert info["kept"] <= 128
    assert info["tune_ms"] < 30_000, f"tuning took {info['tune_ms'] / 1000:.1f} s"


def test_exhaustive_off_by_default_is_heuristic_only():
    """The ninth argument defaulting to 0 is what keeps the pre-cache path byte-identical."""
    go, x, gi, xi = nt_operands(*CACHE_MNK)
    m, n, k = CACHE_MNK
    plan = fp8_pinned._ext().PinnedGemm(m, n, k, go.dtype, x.dtype, False, accum=False, nt=True)
    info = plan.autotune(go, x, gi, xi, 64, 20, 8, 5)
    assert info["kept"] == info["heuristic_count"]
    assert info["enumerated"] == 0, "the capability API must not be touched at all"
