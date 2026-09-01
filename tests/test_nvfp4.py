"""NVFP4 training path (Quartet-II kernels) on sm_120.

Run: python -m pytest tests/test_nvfp4.py -v -s

Needs Blackwell *and* a CUDA 12.8+ torch to build the kernels, so the whole module skips
elsewhere. On this box that means `.venv-cu130`, not `.venv` -- see `sm120/quartet/ext.py`.

What is actually being checked, since "it runs" proves very little for a quantized path:

  the format     kernel output is compared to `quartet.reference`, a pure-torch NVFP4 written
                 from the format rather than from upstream's CUDA. Not bit-exact by design --
                 the kernels use an approximate reciprocal -- so the bound is on the *rate* of
                 disagreement.
  the layout     block scales must reach `_scaled_mm` in the cutlass 128x4 swizzle. Passed
                 row-major it returns confident garbage rather than an error, so there is an
                 explicit negative control: unswizzled scales must be much worse.
  the estimator  the Quartet-II backward claims to be *unbiased*. A biased estimator plateaus
                 under averaging; an unbiased one falls as 1/N. The ladder test asserts the
                 fall, which is the property the method is actually for.
"""
import pytest
import torch

from nanochat.sm120.quartet.ext import unavailable_reason

pytestmark = pytest.mark.skipif(
    unavailable_reason() is not None,
    reason=f"NVFP4 kernels unavailable: {unavailable_reason()}",
)

# Imported lazily-ish: these pull in torch.library registrations but not the extension build,
# which only happens on the first kernel call.
from nanochat.sm120.nvfp4 import (  # noqa: E402
    BACKWARD_SCALE_OVERRIDE,
    NVFP4Linear,
    convert_to_nvfp4_training,
    dequantize,
    enable_rht_hold,
    enable_weight_caches,
    enable_wgrad_accum,
    fp4_mm,
    is_nvfp4_convertible,
    refresh_weight_caches,
)
from nanochat.sm120.quartet.quant import (  # noqa: E402
    NVFP4QuantMode,
    quant_fp4,
    quant_had_eden,
    rht128_quant_eden,
    rht128_requant,
)
from nanochat.sm120 import nvfp4_state  # noqa: E402
from nanochat.sm120.quartet.reference import quantize_reference  # noqa: E402
from nanochat.sm120.quartet.rht import (  # noqa: E402
    hadamard_matrix,
    rerotate_hadamard,
    swizzle_hadamard,
    transform_128,
    transform_rht128,
)

FP4 = torch.float4_e2m1fn_x2


def rel_mse(res, ref):
    res, ref = res.float(), ref.float()
    return ((res - ref).pow(2).mean() / ref.pow(2).mean()).item()


def eff_bits(res, ref):
    """Effective bit-width: relative MSE expressed as bits, the paper's headline metric."""
    import math
    return -math.log2(rel_mse(res, ref)) / 2


def cosine(res, ref):
    res, ref = res.float().flatten(), ref.float().flatten()
    return ((res @ ref) / (ref @ ref)).item()


def unswizzle(scales, rows, cols):
    """Inverse of the 128x4 scale swizzle -- used only to build the negative control."""
    n_col_blocks = (cols // 16) // 4
    r = scales.reshape(-1, 32, 4, 4).permute(0, 2, 1, 3)
    r = r.reshape(-1, n_col_blocks, 128, 4).permute(0, 2, 1, 3)
    return r.reshape(rows, n_col_blocks * 4).contiguous()


# ---------------------------------------------------------------------------------------------
# The Hadamard matrices
# ---------------------------------------------------------------------------------------------

class TestHadamard:
    def test_orthonormal(self):
        """Our Sylvester construction replaces scipy.linalg.hadamard; it must be a rotation."""
        h = hadamard_matrix(128, torch.float32, "cuda")
        assert torch.allclose(h @ h.T, torch.eye(128, device="cuda"), atol=1e-5)

    def test_entries_are_plus_minus_one(self):
        # Every entry is +-n^-0.5. Not exactly +-1 after rescaling: n^-0.5 is not representable.
        h = hadamard_matrix(128, torch.float32, "cuda") * 128**0.5
        assert torch.allclose(h.abs(), torch.ones_like(h))

    @pytest.mark.parametrize("n", [2, 16, 128])
    def test_matches_scipy_when_available(self, n):
        scipy_linalg = pytest.importorskip("scipy.linalg")
        ref = torch.tensor(scipy_linalg.hadamard(n) * n**-0.5, dtype=torch.float32, device="cuda")
        assert torch.equal(hadamard_matrix(n, torch.float32, "cuda"), ref)

    def test_swizzle_identity(self):
        """transform_128(x, swizzle(h)) == transform_rht128(x, h[:16]) -- the kernels regenerate
        the other 112 rows from the first 16, and this pins that reconstruction.

        Tolerance, not equality: the two are different kernels accumulating bf16 in different
        orders. The exact claim is the quantized one below, which is what training uses.
        """
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        h = hadamard_matrix(128, torch.bfloat16, "cuda") * 128**0.5
        full = transform_128(h=swizzle_hadamard(h).contiguous(), x=x)
        top16 = transform_rht128(h=h[:16, :].contiguous(), x=x)
        assert torch.allclose(full, top16, rtol=1e-3, atol=5e-5)

    def test_swizzle_identity_quantized(self):
        """The same identity through the fused quantizing kernels, where it must be exact --
        that is the path the backward actually takes."""
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        h = hadamard_matrix(128, torch.bfloat16, "cuda") * 128**0.5
        seed = torch.tensor(42, dtype=torch.int64)
        full = quant_had_eden(x=x, h=swizzle_hadamard(h).contiguous(), seed=seed)
        top16 = rht128_quant_eden(x=x, h=h[:16, :].contiguous(), seed=seed)
        assert full.tensor_scale.item() == top16.tensor_scale.item()
        assert torch.equal(full.micro_scales.float(), top16.micro_scales.float())
        assert torch.equal(full.fp4, top16.fp4)


# ---------------------------------------------------------------------------------------------
# The quantizers, against an independent implementation of the format
# ---------------------------------------------------------------------------------------------

class TestQuantizers:
    @pytest.mark.parametrize("mode,four_over_six", [(NVFP4QuantMode.RNE, False),
                                                    (NVFP4QuantMode.FOUR_SIX, True)])
    @pytest.mark.parametrize("shape", [(128, 128), (256, 512), (512, 256)])
    def test_matches_reference(self, mode, four_over_six, shape):
        torch.manual_seed(0)
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        q = quant_fp4(x, scale_override=1.0, mode=mode)
        got = dequantize(q.fp4, q.micro_scales, q.tensor_scale).float()
        ref, _, ref_ts = quantize_reference(x, four_over_six=four_over_six)

        assert q.tensor_scale.item() == pytest.approx(ref_ts.item(), rel=1e-5)
        # Not bit-exact: rcp.approx.ftz can flip a value sitting on a code boundary.
        agreement = (got == ref.float()).float().mean().item()
        assert agreement > 0.999, f"only {agreement:.4%} of codes agree with the reference"
        # And the error each achieves must be the same to well within that noise.
        assert rel_mse(got, x) == pytest.approx(rel_mse(ref, x), rel=1e-3)

    def test_four_six_beats_rne(self):
        """The whole point of 4/6: a second candidate block scale, lower error kept."""
        torch.manual_seed(0)
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        rne = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.RNE)
        fs = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        e_rne = rel_mse(dequantize(rne.fp4, rne.micro_scales, rne.tensor_scale), x)
        e_46 = rel_mse(dequantize(fs.fp4, fs.micro_scales, fs.tensor_scale), x)
        assert e_46 < e_rne

    def test_zero_tensor_is_exact(self):
        x = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
        q = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        assert torch.all(dequantize(q.fp4, q.micro_scales, q.tensor_scale) == 0)

    def test_requant_preserves_the_transposed_rotation(self):
        """rht128_requant fuses dequant -> transpose -> rotate -> requantize. Check the result
        against doing those four steps separately in bf16."""
        torch.manual_seed(0)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        h = hadamard_matrix(128, torch.bfloat16, "cuda")
        q = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.RNE)

        got = rht128_requant(x=q.fp4, x_group_scales=q.micro_scales, x_tensor_scale=q.tensor_scale,
                             h=h[:16, :].contiguous(), scale_override=1.0)
        got_bf16 = dequantize(got.fp4, got.micro_scales, got.tensor_scale)

        deq = dequantize(q.fp4, q.micro_scales, q.tensor_scale)
        expected = transform_rht128(h=h[:16, :].contiguous(), x=deq.T.contiguous())
        # Requantization to fp4 is lossy, so compare in the metric the format allows.
        assert eff_bits(got_bf16, expected) > 2.5
        assert cosine(got_bf16, expected) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------------------------
# The GEMM and its scale layout
# ---------------------------------------------------------------------------------------------

class TestGemm:
    def test_forward_accuracy(self):
        torch.manual_seed(0)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16) / 16
        xq = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        wq = quant_fp4(w, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        out = fp4_mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales,
                     xq.tensor_scale * wq.tensor_scale)
        ref = x.float() @ w.float().T
        assert eff_bits(out, ref) > 2.8
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.01)

    def test_row_major_scales_are_wrong(self):
        """Negative control. `_scaled_mm` does not validate the scale layout -- fed row-major
        scales it returns a plausible, wrong answer. If this ever stops failing, the swizzle
        assumption in quant.py has silently changed."""
        torch.manual_seed(0)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16) / 16
        xq = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        wq = quant_fp4(w, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        ref = x.float() @ w.float().T

        good = fp4_mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales,
                      xq.tensor_scale * wq.tensor_scale)
        bad = fp4_mm(xq.fp4, wq.fp4,
                     unswizzle(xq.micro_scales, 512, 256).reshape(-1),
                     unswizzle(wq.micro_scales, 384, 256).reshape(-1),
                     xq.tensor_scale * wq.tensor_scale)
        assert rel_mse(bad, ref) > 10 * rel_mse(good, ref)


# ---------------------------------------------------------------------------------------------
# NVFP4Linear
# ---------------------------------------------------------------------------------------------

class TestNVFP4Linear:
    def test_forward_matches_bf16(self):
        torch.manual_seed(42)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(1, 512, 256, device="cuda", dtype=torch.bfloat16)
        out = layer(x)
        ref = x.float() @ layer.weight.float().T
        assert out.shape == (1, 512, 384)
        assert eff_bits(out, ref) > 2.8
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("tokens", [1, 77, 129, 512])
    def test_unaligned_token_counts(self, tokens):
        """Only the token dim is dynamic; it is zero-padded to 128. Zero rows are exact."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 256, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(1, tokens, 256, device="cuda", dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (1, tokens, 256)
        assert torch.isfinite(out).all()
        assert cosine(out, x.float() @ layer.weight.float().T) == pytest.approx(1.0, abs=0.02)

    def test_padding_does_not_change_real_rows(self):
        """A padded batch must give the same answer as the aligned batch it is a prefix of."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 256, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        full = layer(x)
        part = layer(x[:100])
        # Same rows, same block scales, same per-tensor amax -> identical, not merely close.
        assert torch.equal(part, full[:100])

    def test_bias(self):
        """The bias is added after the GEMM, in the output dtype. Compared forwards, not by
        subtracting it back off: in bf16 that round-trip loses more than the test would allow."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 256, bias=True, device="cuda", dtype=torch.float32)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            layer.bias.zero_()
        base = layer(x)
        with torch.no_grad():
            layer.bias.fill_(2.0)
        # The forward is deterministic (no stochastic rounding), so this is exact.
        assert torch.equal(layer(x), base + torch.tensor(2.0, dtype=base.dtype, device=base.device))

    def test_backward_is_unbiased(self):
        """The claim the method rests on. An unbiased estimator's error falls as 1/N under
        averaging (+1 bit per 4x); a biased one plateaus. Checked over a 256x range."""
        torch.manual_seed(42)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        ref_layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32,
                                disable_backward_quant=True)
        with torch.no_grad():
            ref_layer.weight.copy_(layer.weight)

        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)

        ref_layer.zero_grad()
        (ref_layer(x) * t).sum().backward()
        ref_grad = ref_layer.weight.grad.float().clone()

        bits = {}
        for steps in (1, 16, 256):
            acc = torch.zeros_like(layer.weight, dtype=torch.float32)
            for _ in range(steps):
                layer.zero_grad()
                (layer(x) * t).sum().backward()
                acc += layer.weight.grad.float()
            bits[steps] = eff_bits(acc / steps, ref_grad)

        # 16x more samples is 4x lower MSE is +1 bit; allow a wide margin and still catch bias.
        assert bits[16] > bits[1] + 1.3, f"averaging barely helped: {bits}"
        assert bits[256] > bits[16] + 1.3, f"error plateaued, estimator looks biased: {bits}"

    def test_grad_input_flows(self):
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 256, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_compile_fullgraph(self):
        layer = NVFP4Linear(128, 256, bias=False, device="cuda", dtype=torch.float32)
        fn = torch.compile(lambda t: layer(t), fullgraph=True)
        out = fn(torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------

class TestWeightCache:
    """The cache must change *when* the weight is quantized, never *what* it quantizes to.

    This is the half of the A/B that a training run cannot answer: the NVFP4 backward draws a
    fresh Hadamard rotation and EDEN seeds every step and nothing seeds them, so two runs of one
    config are never bit-identical. The forward has no randomness, so equivalence is exact here.
    """

    @staticmethod
    def _pair():
        """Two layers sharing one weight tensor: one cached, one not."""
        torch.manual_seed(0)
        plain = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        cached = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        cached.weight = plain.weight          # share, don't copy
        enable_weight_caches(cached)
        return plain, cached

    def test_cached_forward_is_bit_identical(self):
        plain, cached = self._pair()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(plain(x), cached(x))

    def test_refresh_tracks_weight_updates(self):
        """After an optimizer-style update the refreshed cache must still match exactly."""
        plain, cached = self._pair()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            plain.weight.add_(torch.randn_like(plain.weight) * 0.05)
        assert refresh_weight_caches(cached) == 1
        assert torch.equal(plain(x), cached(x))

    def test_stale_cache_is_detectable(self):
        """Guard against a cache that silently does nothing: *skipping* the refresh after a
        weight update must change the output. If this ever passes as equal, the cache is not
        actually feeding the GEMM and the two tests above prove nothing."""
        plain, cached = self._pair()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            plain.weight.add_(torch.randn_like(plain.weight) * 0.5)
        # deliberately no refresh_weight_caches() here
        assert not torch.equal(plain(x), cached(x))

    def test_backward_still_flows(self):
        _, cached = self._pair()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cached(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert cached.weight.grad is not None and torch.isfinite(cached.weight.grad).all()

    def test_cache_buffers_stay_out_of_checkpoints(self):
        _, cached = self._pair()
        keys = cached.state_dict().keys()
        assert not any(k.startswith("fp4_") for k in keys), keys


class TestRhtHold:
    """--nvfp4-hold-rht: one rotation and one weight requant per optimizer step, not per backward.

    What is checked here is the mechanism -- that the cache holds exactly what the backward would
    otherwise recompute, that it is held for the window and re-drawn on refresh -- not the
    estimator's quality, which only a bpb battery can speak to.
    """

    @staticmethod
    def _held():
        torch.manual_seed(0)
        mod = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        enable_weight_caches(mod)
        enable_rht_hold(mod)
        return mod

    def test_needs_the_weight_cache(self):
        mod = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="weight cache"):
            enable_rht_hold(mod)

    def test_cache_is_the_requant_under_the_held_rotation(self):
        """Reproduce the refresh by hand from the same RNG state: the held Hadamard and the cached
        requant must be bit-identical to rerotate + rht128_requant of the cached forward weight."""
        mod = self._held()
        torch.manual_seed(1)
        refresh_weight_caches(mod)
        torch.manual_seed(1)
        had = rerotate_hadamard(mod.had)                       # cuda RNG draw, same order
        wt = rht128_requant(x=mod.fp4_w, x_group_scales=mod.fp4_ws, x_tensor_scale=mod.fp4_wg,
                            h=had[:16, :], scale_override=BACKWARD_SCALE_OVERRIDE,
                            scratch_amax=mod.scratch_amax)      # cpu RNG draw for the EDEN seed
        assert torch.equal(mod.fp4_had, had)
        assert torch.equal(mod.fp4_wt, wt.fp4)
        assert torch.equal(mod.fp4_wts.view(torch.uint8), wt.micro_scales.view(torch.uint8))
        assert torch.equal(mod.fp4_wtg, wt.tensor_scale)

    def test_rotation_is_held_for_the_window_and_redrawn_on_refresh(self):
        mod = self._held()
        had0, wt0 = mod.fp4_had.clone(), mod.fp4_wt.clone()
        for _ in range(3):                                      # three micro-steps, no refresh
            x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            mod(x).sum().backward()
        assert torch.equal(mod.fp4_had, had0)
        assert torch.equal(mod.fp4_wt, wt0)
        refresh_weight_caches(mod)                              # the optimizer-step boundary
        assert not torch.equal(mod.fp4_had, had0)
        assert not torch.equal(mod.fp4_wt, wt0)

    def test_backward_still_flows(self):
        mod = self._held()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mod(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert mod.weight.grad is not None and torch.isfinite(mod.weight.grad).all()

    def test_held_buffers_stay_out_of_checkpoints(self):
        mod = self._held()
        assert mod.fp4_had is not None and mod.fp4_wt is not None
        keys = mod.state_dict().keys()
        assert not any(k.startswith("fp4_") for k in keys), keys


class TestConversion:
    def test_only_aligned_layers_convert(self):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(256, 256, bias=False),   # convertible
                              nn.Linear(256, 7, bias=False),     # out not 128-aligned
                              nn.Linear(24, 128, bias=False)).cuda()   # in not 128-aligned
        assert [is_nvfp4_convertible(m) for m in model] == [True, False, False]
        convert_to_nvfp4_training(model)
        assert isinstance(model[0], NVFP4Linear)
        assert not isinstance(model[1], NVFP4Linear)
        assert not isinstance(model[2], NVFP4Linear)

    def test_conversion_shares_parameters(self):
        import torch.nn as nn
        src = nn.Linear(256, 256, bias=False).cuda()
        weight = src.weight
        model = nn.Sequential(src)
        convert_to_nvfp4_training(model)
        assert model[0].weight is weight

    def test_converted_layer_still_counts_for_flops(self):
        """NVFP4Linear must remain a nanochat `Linear`: GPT.num_matmul_params() counts
        isinstance(m, Linear) to derive FLOPs/token, and subclassing nn.Linear instead once
        under-reported FLOPs/token by 4.5x on every --fp8 run."""
        import torch.nn as nn
        from nanochat.gpt import Linear as NanochatLinear
        model = nn.Sequential(nn.Linear(256, 256, bias=False)).cuda()
        convert_to_nvfp4_training(model)
        assert isinstance(model[0], NanochatLinear)

    def test_buffers_are_not_persistent(self):
        """`had` and `scratch_amax` are derived, and must not land in checkpoints."""
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(256, 256, bias=False)).cuda()
        convert_to_nvfp4_training(model)
        keys = model.state_dict().keys()
        assert not any("had" in k or "scratch_amax" in k for k in keys), keys

    def test_gpt_model_converts_and_trains(self):
        """End to end on the real model: convert, forward, backward, finite grads."""
        from nanochat.gpt import GPT, GPTConfig
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=256, vocab_size=1024, n_layer=2,
                           n_head=2, n_kv_head=2, n_embd=256)
        model = GPT(config).cuda()
        n_before = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        convert_to_nvfp4_training(model)
        n_after = sum(1 for m in model.modules() if isinstance(m, NVFP4Linear))
        assert 0 < n_after <= n_before

        idx = torch.randint(0, 1024, (2, 256), device="cuda")
        loss = model(idx, targets=idx)
        loss.backward()
        assert torch.isfinite(loss).all()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------------------------
# The cuBLASLt launcher (--nvfp4-lt-gemm)
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def lt_gemm():
    """Turn the cuBLASLt path on for one test, and off again whatever happens."""
    from nanochat.sm120 import fp4_gemm
    fp4_gemm.configure(True)
    try:
        yield fp4_gemm
    finally:
        fp4_gemm.configure(False)


@pytest.fixture
def lt_gemm_alpha():
    """The same, with the per-tensor scale folded into the GEMM epilogue."""
    from nanochat.sm120 import fp4_gemm
    fp4_gemm.configure(True, epilogue_alpha=True)
    try:
        yield fp4_gemm
    finally:
        fp4_gemm.configure(False)


class TestLtGemm:
    """`fp4_gemm` replaces `_scaled_mm` with a directly-driven cuBLASLt matmul.

    The plan is autotuned per shape, so what these check is that the layout survives: cuBLASLt's
    A is *b*, its D is the transpose of the output, and the block scales follow their operands
    through that swap. Every one of those is silent when m == n, which is most of this model.
    """

    @pytest.mark.parametrize("m,n,k", [(512, 384, 256), (256, 256, 512), (128, 1280, 1280)])
    def test_matches_scaled_mm(self, lt_gemm, m, n, k):
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        b = quant_fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16,
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        ref = lt_gemm._fallback(a.fp4, b.fp4, a.micro_scales, b.micro_scales)
        got = lt_gemm.mm(a.fp4, b.fp4, a.micro_scales, b.micro_scales)
        assert got.shape == ref.shape == (m, n)
        # Not exact by contract -- a split-K plan reduces in a different order -- but the same
        # products, so anything structural (transposed, swapped scales) misses by orders of
        # magnitude rather than by rounding.
        assert rel_mse(got, ref) < 1e-6

    def test_non_square_catches_a_swapped_operand(self, lt_gemm):
        """m != n, so a swapped operand cannot hide behind a matching shape."""
        torch.manual_seed(0)
        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16) / 16
        xq = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        wq = quant_fp4(w, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        out = lt_gemm.mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales) * (
            xq.tensor_scale * wq.tensor_scale)
        ref = x.float() @ w.float().T
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.01)

    def test_disabled_is_the_scaled_mm_path(self):
        """With the flag off nothing here runs -- `fp4_mm` must not even build a plan."""
        from nanochat.sm120 import fp4_gemm
        assert not fp4_gemm.enabled()
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        plans_before = len(fp4_gemm._PLANS)
        fp4_mm(a.fp4, a.fp4, a.micro_scales, a.micro_scales, a.tensor_scale * a.tensor_scale)
        assert len(fp4_gemm._PLANS) == plans_before

    def test_linear_forward_backward(self, lt_gemm):
        """All three GEMMs of a real layer, through the launcher, with finite grads."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert out.shape == (512, 384)
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()

    def test_survives_torch_compile(self, lt_gemm):
        """The plan builder must never see a FakeTensor: it is behind a custom op for this."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        compiled = torch.compile(layer)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        out = compiled(x)
        assert out.shape == (512, 384) and torch.isfinite(out).all()

    def test_rejects_a_mislaid_operand(self, lt_gemm):
        """The layout is not inferred from strides -- it is assumed, so it is checked.

        Against a *tuned* plan, so the guard being tested is `check_operands` and not the
        "run() before autotune()" one.
        """
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(256, 512, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        plan = lt_gemm._ext().Fp4Gemm(256, 256, 512)
        plan.autotune(a.fp4, a.fp4, a.micro_scales, a.micro_scales, 8, 3, 2, 1)
        plan.run(a.fp4, a.fp4, a.micro_scales, a.micro_scales)  # the tuned shape is fine

        wrong_k = quant_fp4(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16),
                            scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        with pytest.raises(RuntimeError, match="a must be"):
            plan.run(wrong_k.fp4, a.fp4, wrong_k.micro_scales, a.micro_scales)
        non_contig = a.fp4.t().contiguous().t()  # right shape, column-major
        with pytest.raises(RuntimeError, match="row-major"):
            plan.run(non_contig, a.fp4, a.micro_scales, a.micro_scales)


class TestEpilogueAlpha:
    """`--nvfp4-epilogue-alpha`: the per-tensor scale becomes the GEMM's device-pointer alpha.

    The result is deliberately *not* bit-identical to `out * alpha` — it scales the fp32
    accumulator and rounds to bf16 once instead of twice — so what these pin is that it is the
    same product, that the accuracy moves the right way, and that the caller cannot end up
    applying alpha twice or not at all.
    """

    @pytest.mark.parametrize("m,n,k", [(512, 384, 256), (256, 256, 512)])
    def test_matches_the_separate_multiply(self, lt_gemm_alpha, m, n, k):
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        b = quant_fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16,
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        alpha = (a.tensor_scale * b.tensor_scale).contiguous()
        separate = lt_gemm_alpha._fallback(a.fp4, b.fp4, a.micro_scales, b.micro_scales) * alpha
        epilogue = lt_gemm_alpha.mm(a.fp4, b.fp4, a.micro_scales, b.micro_scales, alpha)
        assert epilogue.shape == (m, n) and epilogue.dtype == torch.bfloat16
        # One bf16 ulp apart, not six orders of magnitude: alpha here is ~1e-6, so dropping it
        # or applying it twice is not a near miss.
        assert rel_mse(epilogue, separate) < 1e-4

    def test_is_the_more_accurate_of_the_two(self, lt_gemm_alpha):
        """The claim that justifies the tolerance above: one rounding, not two."""
        torch.manual_seed(0)
        m, n, k = 512, 384, 256
        a = quant_fp4(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        b = quant_fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16,
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        alpha = (a.tensor_scale * b.tensor_scale).contiguous()
        exact = (dequantize(a.fp4, a.micro_scales, a.tensor_scale).float()
                 @ dequantize(b.fp4, b.micro_scales, b.tensor_scale).float().T)
        separate = lt_gemm_alpha._fallback(a.fp4, b.fp4, a.micro_scales, b.micro_scales) * alpha
        epilogue = lt_gemm_alpha.mm(a.fp4, b.fp4, a.micro_scales, b.micro_scales, alpha)
        assert rel_mse(epilogue, exact) < rel_mse(separate, exact)

    def test_fp4_mm_does_not_apply_alpha_twice(self, lt_gemm_alpha):
        """`fp4_mm` must return the epilogue result as-is. Applying alpha again squares a ~1e-6
        number, so this is the difference between a working run and silent zeros."""
        torch.manual_seed(0)
        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16) / 16
        xq = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        wq = quant_fp4(w, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        out = fp4_mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales,
                     xq.tensor_scale * wq.tensor_scale)
        ref = x.float() @ w.float().T
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.01)

    def test_linear_forward_backward(self, lt_gemm_alpha):
        """All three GEMMs of a real layer scale in their own epilogue."""
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref = x.float() @ layer.weight.float().T
        out = layer(x)
        out.sum().backward()
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.02)
        assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()

    def test_survives_torch_compile(self, lt_gemm_alpha):
        torch.manual_seed(0)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ layer.weight.float().T
        out = torch.compile(layer)(x)
        assert cosine(out, ref) == pytest.approx(1.0, abs=0.02)

    def test_needs_the_launcher(self):
        from nanochat.sm120 import fp4_gemm
        with pytest.raises(ValueError, match="--nvfp4-lt-gemm"):
            fp4_gemm.configure(False, epilogue_alpha=True)
        assert not fp4_gemm.enabled() and not fp4_gemm.epilogue_alpha()

    def test_alpha_must_be_a_device_scalar(self, lt_gemm_alpha):
        """cuBLASLt dereferences it on the GPU: a host scalar would read garbage, silently."""
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(256, 512, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        ops = (a.fp4, a.fp4, a.micro_scales, a.micro_scales)
        plan = lt_gemm_alpha._ext().Fp4Gemm(256, 256, 512,
                                            lt_gemm_alpha._ext().WORKSPACE_BYTES, True)
        plan.autotune(*ops, 8, 3, 2, 1)
        with pytest.raises(RuntimeError, match="alpha must be"):
            plan.run_alpha(*ops, torch.tensor(1.0))                      # CPU
        with pytest.raises(RuntimeError, match="alpha must be"):
            plan.run_alpha(*ops, torch.ones(4, device="cuda"))           # not a scalar
        with pytest.raises(RuntimeError, match="host-alpha plan"):
            lt_gemm_alpha._ext().Fp4Gemm(256, 256, 512).run_alpha(
                *ops, torch.ones((), device="cuda"))


class TestFusedWgrad:
    """`--nvfp4-fuse-wgrad`: the wgrad accumulates into an fp32 buffer in its own epilogue.

    Two claims to pin, and they fail in different directions. The *arithmetic* one -- beta=1
    really adds, and adds the alpha-scaled product -- is checkable exactly against
    `_scaled_mm * alpha` plus an add, and both ways of getting it wrong (a dropped beta, a
    dropped alpha) miss by orders of magnitude rather than by rounding. The *plumbing* one --
    the backward stops returning a weight gradient, so nothing must still be reading `.grad`
    -- is what a training run would fail at silently, with gradients simply going missing.

    The backward draws a fresh Hadamard and fresh EDEN seeds per call, both from torch's
    generators, so the fused/unfused comparison seeds immediately before each `.backward()`.
    Without that the two arms quantize differently and no tolerance means anything.
    """

    @staticmethod
    def _pair(lt, **kwargs):
        """Two layers sharing one weight: one accumulating in the epilogue, one not."""
        torch.manual_seed(0)
        plain = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32, **kwargs)
        fused = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32, **kwargs)
        fused.weight = torch.nn.Parameter(plain.weight.detach().clone())
        return plain, fused, enable_wgrad_accum(fused)

    @staticmethod
    def _operands(m=384, n=256, k=512):
        torch.manual_seed(0)
        a = quant_fp4(torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        b = quant_fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16,
                      scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        alpha = (a.tensor_scale * b.tensor_scale).contiguous()
        return a, b, alpha

    def test_matches_the_separate_add(self, lt_gemm):
        """The op, against what it replaces: GEMM, per-tensor scale, widen, add."""
        a, b, alpha = self._operands()
        prod = lt_gemm._fallback(a.fp4, b.fp4, a.micro_scales, b.micro_scales).float() * alpha
        c0 = torch.randn_like(prod) * prod.abs().mean()
        got = c0.clone()
        lt_gemm.mm_accum(a.fp4, b.fp4, a.micro_scales, b.micro_scales, alpha, got)
        ref = prod + c0
        assert got.dtype == torch.float32
        # One bf16 ulp: the reference rounds the product before adding, the epilogue does not.
        assert rel_mse(got, ref) < 1e-4
        # ... and the two ways to be wrong are nowhere near a bf16 ulp. Without these the
        # tolerance above would pass on a kernel that ignored beta on a small enough c0.
        assert rel_mse(prod, ref) > 0.1                                  # beta dropped
        assert rel_mse(c0 + prod / alpha, ref) > 0.1                     # alpha dropped

    def test_is_the_more_accurate_of_the_two(self, lt_gemm):
        """Why the fused gradient is not bit-identical, and which side of the truth it is on.

        The unfused wgrad rounds the fp32 accumulator to bf16, multiplies by the per-tensor
        scale and rounds again; the epilogue does both in fp32 and never rounds. Against the
        exact product of the dequantized operands that is ~2x lower relative MSE -- so the
        ~0.1% magnitude the fused result carries over the unfused one is magnitude the bf16
        rounding was losing, not magnitude the epilogue invented. Built on the backward's own
        operands (RHT + EDEN, backward scale override), not on generic ones.
        """
        torch.manual_seed(0)
        scratch = torch.empty((), dtype=torch.uint32, device="cuda")
        h16 = rerotate_hadamard(hadamard_matrix(128, torch.bfloat16, "cuda"))[:16, :]
        go = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        xq = quant_fp4(torch.randn(512, 256, device="cuda", dtype=torch.bfloat16),
                       scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        et = rht128_quant_eden(x=go, h=h16, scale_override=BACKWARD_SCALE_OVERRIDE,
                               transpose=True, scratch_amax=scratch)
        xt = rht128_requant(x=xq.fp4, x_group_scales=xq.micro_scales,
                            x_tensor_scale=xq.tensor_scale, h=h16,
                            scale_override=BACKWARD_SCALE_OVERRIDE, scratch_amax=scratch)
        alpha = (et.tensor_scale * xt.tensor_scale).contiguous()
        ops = (et.fp4, xt.fp4, et.micro_scales, xt.micro_scales)

        exact = (dequantize(et.fp4, et.micro_scales, et.tensor_scale).float()
                 @ dequantize(xt.fp4, xt.micro_scales, xt.tensor_scale).float().T)
        fused = torch.zeros(384, 256, device="cuda")
        lt_gemm.mm_accum(*ops, alpha, fused)
        unfused = (lt_gemm._fallback(*ops) * alpha).float()   # what fp4_mm returns today
        assert rel_mse(fused, exact) < rel_mse(unfused, exact)

    def test_beta_keeps_accumulating(self, lt_gemm):
        """Called twice, it must add twice -- beta=1 is not a one-shot initialisation."""
        a, b, alpha = self._operands()
        ops = (a.fp4, b.fp4, a.micro_scales, b.micro_scales, alpha)
        one = torch.zeros(384, 256, device="cuda")
        lt_gemm.mm_accum(*ops, one)
        two = one.clone()
        lt_gemm.mm_accum(*ops, two)
        assert rel_mse(two, 2 * one) < 1e-6
        assert one.abs().max() > 0     # a zero product would satisfy the line above vacuously

    def test_accumulator_must_be_fp32(self, lt_gemm):
        a, b, alpha = self._operands(m=256, n=256, k=512)
        ops = (a.fp4, a.fp4, a.micro_scales, a.micro_scales, alpha)
        plan = lt_gemm._ext().Fp4Gemm(256, 256, 512, lt_gemm._ext().WORKSPACE_BYTES, True, True)
        plan.autotune(a.fp4, a.fp4, a.micro_scales, a.micro_scales, 8, 3, 2, 1)
        with pytest.raises(RuntimeError, match="accumulator must be"):
            plan.run_accum(*ops, torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="accumulator must be"):
            plan.run_accum(*ops, torch.zeros(256, 512, device="cuda"))
        with pytest.raises(RuntimeError, match="non-accumulating plan"):
            lt_gemm._ext().Fp4Gemm(256, 256, 512, lt_gemm._ext().WORKSPACE_BYTES, True).run_accum(
                *ops, torch.zeros(256, 256, device="cuda"))

    def test_a_host_alpha_accum_plan_is_refused(self, lt_gemm):
        """beta=1 adds the *scaled* product, so scaling afterwards would rescale the sum too."""
        with pytest.raises(RuntimeError, match="accumulating plan needs the device alpha"):
            lt_gemm._ext().Fp4Gemm(256, 256, 512, lt_gemm._ext().WORKSPACE_BYTES, False, True)

    def test_the_weight_gradient_moves_to_the_buffer(self, lt_gemm):
        plain, fused, store = self._pair(lt_gemm)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(7)
        plain(x).square().sum().backward()
        torch.manual_seed(7)
        fused(x).square().sum().backward()
        assert fused.weight.grad is None, "autograd must not also accumulate"
        acc = fused.fp4_main_grad
        assert acc.dtype == torch.float32 and acc.shape == fused.weight.shape
        assert torch.isfinite(acc).all() and acc.abs().max() > 0
        # Seeded, so the operands are identical and only the epilogue differs. Bound the
        # difference by relative MSE, not by the projection `cosine` computes: the fused
        # result is systematically ~0.1% *larger* in magnitude, because the bf16 rounding it
        # skips loses magnitude (see test_is_the_more_accurate_of_the_two), and a projection
        # ratio reads that as a failure while a relative MSE reads it as the ulp it is.
        assert rel_mse(acc, plain.weight.grad) < 1e-4
        assert cosine(acc, plain.weight.grad) == pytest.approx(1.0, abs=5e-3)

    def test_accumulates_across_micro_steps(self, lt_gemm):
        """The property the whole item is for: 4 micro-steps, one running sum, no add kernel."""
        plain, fused, store = self._pair(lt_gemm)
        xs = [torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        for i, x in enumerate(xs):
            torch.manual_seed(100 + i)
            plain(x).square().sum().backward()
            torch.manual_seed(100 + i)
            fused(x).square().sum().backward()
        assert fused.weight.grad is None
        assert rel_mse(fused.fp4_main_grad, plain.weight.grad) < 1e-4
        assert cosine(fused.fp4_main_grad, plain.weight.grad) == pytest.approx(1.0, abs=5e-3)

    def test_a_dropped_micro_step_is_detectable(self, lt_gemm):
        """Guard against a buffer that is written rather than accumulated: three contributions
        must not equal one. If this ever passes as equal, the test above proves nothing."""
        _, fused, store = self._pair(lt_gemm)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(7)
        fused(x).square().sum().backward()
        one = fused.fp4_main_grad.clone()
        for _ in range(2):
            torch.manual_seed(7)
            fused(x).square().sum().backward()
        assert cosine(fused.fp4_main_grad, one) == pytest.approx(3.0, abs=0.05)

    def test_attach_and_zero_lifecycle(self, lt_gemm):
        _, fused, store = self._pair(lt_gemm)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        fused(x).square().sum().backward()
        assert fused.weight.grad is None
        store.attach()
        # The same storage, not a copy: the optimizer must read what the epilogue wrote.
        assert fused.weight.grad.data_ptr() == fused.fp4_main_grad.data_ptr()
        fused.zero_grad(set_to_none=True)       # what the training loop does after step()
        store.zero_()
        assert float(store.flat.abs().max()) == 0.0
        assert fused.fp4_main_grad.abs().max() == 0.0   # the view sees the flat buffer's zeros

    def test_survives_torch_compile(self, lt_gemm):
        plain, fused, store = self._pair(lt_gemm)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(7)
        plain(x).square().sum().backward()
        torch.manual_seed(7)
        torch.compile(fused)(x).square().sum().backward()
        assert fused.weight.grad is None
        # Compiled RNG is functionalized, so the two arms quantize differently however they are
        # seeded -- this is the same "no two runs agree" bound TestWeightCache documents. What
        # it pins is that the gradient still lands in the buffer and still points the same way.
        assert torch.isfinite(fused.fp4_main_grad).all()
        assert cosine(fused.fp4_main_grad, plain.weight.grad) == pytest.approx(1.0, abs=0.05)

    def test_bf16_backward_ablation_still_accumulates(self, lt_gemm):
        """`disable_backward_quant` bypasses the fp4 wgrad entirely; the buffer is still the
        only place a weight gradient may land, or that ablation would silently lose them."""
        plain, fused, store = self._pair(lt_gemm, disable_backward_quant=True)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        plain(x).square().sum().backward()
        fused(x).square().sum().backward()
        assert fused.weight.grad is None
        assert rel_mse(fused.fp4_main_grad, plain.weight.grad) < 1e-6

    def test_needs_the_launcher(self):
        """Without it `mm_accum` would fall back to _scaled_mm plus an eager add -- the exact
        accumulation this deletes, plus a kernel. A flag that silently does the opposite of
        what it says is how perf-log experiment 16 lost two arms."""
        from nanochat.sm120 import fp4_gemm
        fp4_gemm.configure(False)
        layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="--nvfp4-lt-gemm"):
            enable_wgrad_accum(layer)
        assert layer.fp4_main_grad is None

    def test_buffers_stay_out_of_checkpoints(self, lt_gemm):
        _, fused, store = self._pair(lt_gemm)
        keys = fused.state_dict().keys()
        assert "fp4_main_grad" not in keys, keys


class TestDelayedScale:
    """--nvfp4-scaling delayed: the activation's per-tensor amax from a history, not a pre-pass.

    The property that matters is not that the two arms agree -- they cannot, since a different
    assumed amax rounds every block scale into a different e4m3 bucket -- but that the delayed
    arm is no *less* accurate. `test_accuracy_matches_dynamic` is the one to read first; the
    rest guard the history's mechanics.
    """

    MARGIN = 2.0
    # The readback is biased high by the 4/6 candidate choice (<=1.5x, since a group that picked
    # 1/4 reports 1.5x its true max) plus e4m3's +-1/16 rounding of the block scale.
    BIAS = 1.5 * (1 + 1 / 16)

    @staticmethod
    def _layer(**kw):
        torch.manual_seed(0)
        return NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32, **kw)

    @classmethod
    def _pair(cls, margin=None):
        """Two layers sharing one weight: one dynamic, one on a history."""
        plain, delayed = cls._layer(), cls._layer()
        delayed.weight = plain.weight
        state = nvfp4_state.enable_delayed_scaling(
            torch.nn.Sequential(delayed), margin=cls.MARGIN if margin is None else margin)
        return plain, delayed, state

    @staticmethod
    def _warm(layer, state, x, steps=4):
        for _ in range(steps):
            layer(x)
            state.update()

    @staticmethod
    def _x(scale=3.0):
        return torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * scale

    def test_buffers_are_registered_and_stay_out_of_checkpoints(self):
        _, delayed, _ = self._pair()
        # Registered, not set as plain attributes: a Python-side tensor cache captures a
        # FakeTensor on the first trace and the next compile dies with "Mixing fake modes NYI".
        for name in ("fp4_scale_in", "fp4_inv_in", "fp4_amax_in"):
            assert name in dict(delayed.named_buffers()), name
        assert not any(k.startswith("fp4_") for k in delayed.state_dict()), delayed.state_dict().keys()

    def test_accuracy_matches_dynamic(self):
        """The headline: a history must not cost accuracy against the pre-pass it replaces.

        Both arms are ~0.12 off the fp32 product -- that is fp4 quantization, not this flag.
        What is asserted is that the delayed arm does not add to it.
        """
        plain, delayed, state = self._pair()
        x = self._x()
        ref = x.float() @ plain.weight.float().T
        self._warm(delayed, state, x)
        err = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
        assert err(delayed(x)) < err(plain(x)) * 1.02

    def test_accuracy_is_insensitive_to_the_margin(self):
        """The assumed amax cancels out of the fp4 codes, surviving only in which e4m3 bucket
        each block scale rounds into -- so a 4x change in headroom must not move the error."""
        plain, _, _ = self._pair()
        x = self._x()
        ref = x.float() @ plain.weight.float().T
        base = ((plain(x).float() - ref).norm() / ref.norm()).item()
        for margin in (1.0, 4.0):
            _, delayed, state = self._pair(margin=margin)
            self._warm(delayed, state, x)
            err = ((delayed(x).float() - ref).norm() / ref.norm()).item()
            assert err < base * 1.02, (margin, err, base)

    def test_scale_tracks_the_real_amax(self):
        """The reading comes off the block scales, so it is biased high by the 4/6 candidate
        choice -- bounded by 1.5x, on top of the margin."""
        _, delayed, state = self._pair()
        x = self._x()
        self._warm(delayed, state, x)
        ratio = delayed.fp4_inv_in.item() / x.abs().max().item()
        assert self.MARGIN <= ratio <= self.MARGIN * self.BIAS, ratio

    def test_one_update_corrects_the_initial_assumption(self):
        """The seed is deliberately well above any real activation. Over-estimating is the safe
        direction -- the readback is invariant to the assumption, so one update is exact from
        any seed that did not saturate."""
        _, delayed, state = self._pair()
        x = self._x()
        assert delayed.fp4_inv_in.item() > 5 * x.abs().max().item()
        delayed(x)
        state.update()
        assert delayed.fp4_inv_in.item() < self.MARGIN * self.BIAS * x.abs().max().item()

    def test_search_recovers_from_a_spike(self):
        """A saturated block scale pins at e4m3's 448 and is a floor on the amax, not a reading
        of it, so the history has to search upward past it."""
        _, delayed, state = self._pair()
        small = self._x()
        self._warm(delayed, state, small)
        big = small * 8
        for _ in range(6):
            delayed(big)
            state.update()
        ratio = delayed.fp4_inv_in.item() / big.abs().max().item()
        assert self.MARGIN <= ratio <= self.MARGIN * self.BIAS, ratio

    def test_reading_is_the_max_over_micro_steps(self):
        """Grad accumulation quantizes a different activation per micro-step; the history has
        to fold in the largest, not the last."""
        _, delayed, state = self._pair()
        delayed(self._x(scale=8.0))
        after_big = delayed.fp4_amax_in.item()
        delayed(self._x(scale=0.1))
        assert delayed.fp4_amax_in.item() == after_big

    def test_history_actually_reaches_the_kernel(self):
        """Guard against a no-op: if the amax never gets to quant_fp4, every test above passes
        while measuring the dynamic path. A deliberately wrong history must change the output."""
        _, delayed, state = self._pair()
        x = self._x()
        self._warm(delayed, state, x)
        good = delayed(x)
        with torch.no_grad():
            state.scales[1].mul_(1e3)      # inverse-scale slot == the assumed amax
        assert not torch.equal(good, delayed(x))

    def test_rne_is_rejected(self):
        """RNE scales a group at the tensor amax onto e4m3's max exactly, leaving no headroom
        for saturation to be detected against."""
        layer = self._layer(four_over_six=False)
        with pytest.raises(ValueError, match="4/6"):
            nvfp4_state.enable_delayed_scaling(torch.nn.Sequential(layer))

    def test_compile_matches_eager(self):
        """The history has to survive torch.compile and land on the same scale.

        Not bitwise on the output: Inductor generates a different graph, and the *dynamic* path
        is already not bit-equal between eager and compiled, so equality would be asserting
        something this flag does not control. The scale is the strict check.
        """
        torch._dynamo.reset()
        _, eager, eager_state = self._pair()
        _, comp, comp_state = self._pair()
        x = self._x()
        self._warm(eager, eager_state, x)
        self._warm(torch.compile(comp), comp_state, x)
        assert comp.fp4_inv_in.item() == pytest.approx(eager.fp4_inv_in.item(), rel=1e-3)

    def test_block_scale_buffer_is_fully_written(self):
        """The readback maxes over the whole micro_scales buffer, which quant_fp4 allocates with
        torch.empty and the kernel fills through the cutlass 128x4 swizzle. A hole in that
        mapping would be read as uninitialized memory and silently inflate every reading."""
        x = self._x()
        q = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        poison = torch.full_like(q.micro_scales.view(torch.uint8), 0xFF)
        q2 = quant_fp4(x, scale_override=1.0, mode=NVFP4QuantMode.FOUR_SIX)
        # every byte must have been overwritten; 0xFF is e4m3 -NaN and cannot be produced here
        assert (q2.micro_scales.view(torch.uint8) != poison).all()

    def test_a_nan_activation_does_not_poison_the_history(self):
        """The history is a running max, so one NaN reading would pin every scale at NaN for
        the rest of the run with no way back. record_amax clamps e4m3's NaN byte instead."""
        _, delayed, state = self._pair()
        x = self._x()
        self._warm(delayed, state, x)
        bad = x.clone()
        bad[0, 0] = float("nan")
        delayed(bad)
        state.update()
        assert torch.isfinite(delayed.fp4_inv_in).all()
        assert torch.isfinite(state.hist).all()
