"""The NVFP4 numerics levers (nanochat/sm120/nvfp4_numerics.py).

Run: python -m pytest tests/test_nvfp4_numerics.py -v

What is being checked:

  the flags       parse, validate, and join base_train's without-`--nvfp4` error.
  the filter      --nvfp4-exclude / --nvfp4-bf16-blocks leave exactly the named Linears alone.
  the families    every new operand pairing feeds a GEMM from one kernel family. Mixing two
                  gives a silently wrong product, so each variant's gradients are projected onto
                  the bf16 gradient -- the test that catches a permuted operand immediately.
  the estimator   the bf16-source and unrotated paths stay unbiased (error falls under
                  averaging); the 2D weight is deterministic, so dgrad's error plateaus there.
  the 2D weight   row and transposed forms are the same numbers, survive the swizzle, multiply
                  correctly through the GEMM, and agree with Transformer Engine's torch oracle.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from nanochat.sm120.quartet.ext import unavailable_reason

pytestmark = pytest.mark.skipif(
    unavailable_reason() is not None,
    reason=f"NVFP4 kernels unavailable: {unavailable_reason()}",
)

from nanochat.gpt import GPT, GPTConfig, Linear  # noqa: E402
from nanochat.sm120 import nvfp4_numerics as nn4  # noqa: E402
from nanochat.sm120 import recipe  # noqa: E402
from nanochat.sm120.nvfp4 import (  # noqa: E402
    NVFP4Linear,
    _unblock,
    convert_to_nvfp4_training,
    dequantize,
    enable_weight_caches,
    enable_wgrad_accum,
    fp4_mm,
)
from nanochat.sm120.quartet.quant import quant_fp4  # noqa: E402

TE_ROOT = Path("/remote/projects/pycharm/TransformerEngine/transformer_engine/pytorch/custom_recipes")


def rel_mse(res, ref):
    res, ref = res.float(), ref.float()
    return ((res - ref).pow(2).mean() / ref.pow(2).mean()).item()


def eff_bits(res, ref):
    import math
    return -math.log2(rel_mse(res, ref)) / 2


def cosine(res, ref):
    res, ref = res.float().flatten(), ref.float().flatten()
    return ((res @ ref) / (ref @ ref)).item()


def parse(*argv, weight_cache=True):
    parser = argparse.ArgumentParser()
    recipe.add_args(parser)
    args = parser.parse_args(list(argv))
    args.nvfp4_weight_cache = weight_cache   # base_train resolves this from the stack flags
    return args


# ---------------------------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------------------------

class TestFlags:
    def test_defaults_are_todays_behaviour(self):
        args = parse()
        assert nn4.orphan_flags(args) == []
        assert nn4.from_args(args) == nn4.DEFAULT
        assert nn4.DEFAULT.rht == "all" and nn4.DEFAULT.bwd_source == "fp4" and not nn4.DEFAULT.weight_2d

    def test_orphans_are_exactly_the_flags_given(self):
        args = parse("--nvfp4-rht", "wgrad", "--nvfp4-weight-2d", "--nvfp4-exclude", "lm_head")
        assert nn4.orphan_flags(args) == ["nvfp4_exclude", "nvfp4_rht", "nvfp4_weight_2d"]

    def test_lists_parse(self):
        cfg = nn4.from_args(parse("--nvfp4-exclude", "lm_head, mlp.c_proj", "--nvfp4-bf16-blocks", "2,4"))
        assert cfg.exclude == frozenset({"lm_head", "mlp.c_proj"})
        assert cfg.bf16_blocks == (2, 4)

    def test_weight_2d_cannot_rotate_dgrad(self):
        with pytest.raises(ValueError, match="rht wgrad or none"):
            nn4.from_args(parse("--nvfp4-weight-2d"))

    def test_rht_none_needs_bf16_source(self):
        with pytest.raises(ValueError, match="bwd-source bf16"):
            nn4.from_args(parse("--nvfp4-rht", "none"))

    def test_weight_2d_needs_the_cache(self):
        with pytest.raises(ValueError, match="weight-cache"):
            nn4.from_args(parse("--nvfp4-weight-2d", "--nvfp4-rht", "wgrad", weight_cache=False))

    def test_bad_blocks(self):
        with pytest.raises(ValueError, match="FIRST,LAST"):
            nn4.from_args(parse("--nvfp4-bf16-blocks", "1"))

    def test_fp8_precision_needs_an_exclusion(self):
        with pytest.raises(ValueError, match="nothing is excluded"):
            nn4.from_args(parse("--nvfp4-exclude-precision", "fp8"))
        cfg = nn4.from_args(parse("--nvfp4-exclude-precision", "fp8", "--nvfp4-exclude", "lm_head"))
        assert cfg.exclude_precision == "fp8"
        assert nn4.orphan_flags(parse("--nvfp4-exclude-precision", "fp8")) == ["nvfp4_exclude_precision"]


# ---------------------------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------------------------

def small_gpt():
    config = GPTConfig(sequence_len=256, vocab_size=1024, n_layer=4, n_head=2, n_kv_head=2, n_embd=256)
    with torch.device("cuda"):
        return GPT(config)


def nvfp4_names(model):
    return sorted(n for n, m in model.named_modules() if isinstance(m, NVFP4Linear))


class TestLayerSelection:
    def test_exclude_and_bf16_blocks(self):
        model = small_gpt()
        cfg = nn4.NumericsConfig(exclude=frozenset({"lm_head"}), bf16_blocks=(1, 1))
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, 4), numerics=cfg)
        names = nvfp4_names(model)
        assert len(names) == 12 and all(n.startswith(("transformer.h.1.", "transformer.h.2.")) for n in names)
        assert type(model.lm_head) is Linear
        assert type(model.transformer.h[0].mlp.c_fc) is Linear
        assert type(model.transformer.h[3].attn.c_q) is Linear
        assert type(model.smear_gate) is Linear        # never eligible: 24 -> 1
        line = nn4.describe(model, cfg)
        assert "lm_head" in line and "blocks 0,3" in line and "13 Linears" in line

    def test_suffix_match(self):
        model = small_gpt()
        cfg = nn4.NumericsConfig(exclude=frozenset({"c_proj"}))
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, 4), numerics=cfg)
        assert not any(n.endswith("c_proj") for n in nvfp4_names(model))
        model = small_gpt()
        cfg = nn4.NumericsConfig(exclude=frozenset({"mlp.c_proj"}))
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, 4), numerics=cfg)
        names = nvfp4_names(model)
        assert any(n.endswith("attn.c_proj") for n in names) and not any(n.endswith("mlp.c_proj") for n in names)

    def test_excluded_layers_as_fp8(self):
        """--nvfp4-exclude-precision fp8: the excluded Linears become Float8Linear, nothing else moves."""
        from nanochat.fp8 import Float8Linear
        model = small_gpt()
        cfg = nn4.NumericsConfig(exclude=frozenset({"lm_head"}), bf16_blocks=(1, 0), exclude_precision="fp8")
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, 4), numerics=cfg)
        names = nn4.convert_excluded_to_fp8(model, cfg, 4)
        assert "lm_head" in names and len(names) == 7
        assert type(model.lm_head) is Float8Linear and type(model.transformer.h[0].mlp.c_fc) is Float8Linear
        assert type(model.transformer.h[1].mlp.c_fc) is NVFP4Linear
        assert type(model.smear_gate) is Linear                # too small for either
        assert len(nvfp4_names(model)) == 18
        line = nn4.describe(model, cfg)
        assert "kept in fp8: lm_head, blocks 0 (7 Linears)" in line
        # And it trains: a forward/backward through the mixed model.
        idx = torch.randint(0, 1024, (2, 64), device="cuda")
        loss = model(idx, idx)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.lm_head.weight.grad).all()

    def test_excluded_layers_as_fp8_fwd(self):
        """--nvfp4-exclude-precision fp8-fwd: fp8 forward, NVFP4 backward, still counted as NVFP4."""
        from nanochat.sm120.nvfp4_fp8fwd import FP8FwdNVFP4Linear
        model = small_gpt()
        cfg = nn4.NumericsConfig(exclude=frozenset({"lm_head"}), exclude_precision="fp8-fwd")
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(cfg, 4), numerics=cfg)
        assert nn4.convert_excluded_to_fp8(model, cfg, 4) == ["lm_head"]
        assert type(model.lm_head) is FP8FwdNVFP4Linear
        assert "kept in fp8-fwd: lm_head (1 Linears)" in nn4.describe(model, cfg)
        assert enable_weight_caches(model) == 25 and model.lm_head.fp8_w is not None
        idx = torch.randint(0, 1024, (2, 64), device="cuda")
        loss = model(idx, idx)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.lm_head.weight.grad).all()

    def test_default_converts_everything_eligible(self):
        model = small_gpt()
        convert_to_nvfp4_training(model, module_filter_fn=nn4.module_filter(nn4.DEFAULT, 4))
        assert len(nvfp4_names(model)) == 4 * 6 + 1
        assert "kept in bf16: none" in nn4.describe(model, nn4.DEFAULT)


# ---------------------------------------------------------------------------------------------
# 2D weight quantization
# ---------------------------------------------------------------------------------------------

class TestWeight2D:
    def test_both_forms_are_the_same_numbers(self):
        torch.manual_seed(0)
        w = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
        fp4, ws, ts, fp4_t, wts = nn4.quantize_weight_2d(w)
        assert fp4.shape == (384, 128) and ws.shape == (384, 16) and fp4_t.shape == (256, 192) and wts.shape == (256, 24)
        assert torch.equal(dequantize(fp4, ws, ts), dequantize(fp4_t, wts, ts).T)

    def test_swizzle_round_trips_kernel_scales(self):
        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        q = quant_fp4(x, scale_override=1.0)
        back = nn4.to_blocked(_unblock(q.micro_scales, 256, 512))
        assert torch.equal(back.view(torch.uint8), q.micro_scales.view(torch.uint8))

    @pytest.mark.parametrize("n,k", [(384, 256), (256, 512)])
    def test_gemm_reads_both_forms(self, n, k):
        """Proves the packing and the swizzle: the GEMM output is the dequantized product."""
        torch.manual_seed(1)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        fp4, ws, ts, fp4_t, wts = nn4.quantize_weight_2d(w)
        w_deq = dequantize(fp4, ws, ts).float()
        x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)
        xq = quant_fp4(x, scale_override=1.0)
        out = fp4_mm(xq.fp4, fp4, xq.micro_scales, ws, xq.tensor_scale * ts)
        assert rel_mse(out, dequantize(xq.fp4, xq.micro_scales, xq.tensor_scale).float() @ w_deq.T) < 1e-4
        e = torch.randn(512, n, device="cuda", dtype=torch.bfloat16)
        eq = quant_fp4(e, scale_override=1.0)
        out_t = fp4_mm(eq.fp4, fp4_t, eq.micro_scales, wts, eq.tensor_scale * ts)
        assert rel_mse(out_t, dequantize(eq.fp4, eq.micro_scales, eq.tensor_scale).float() @ w_deq) < 1e-4

    def test_four_six_helps_in_2d_too(self):
        torch.manual_seed(2)
        w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        e46 = rel_mse(dequantize(*nn4.quantize_weight_2d(w)[:3]), w)
        rne = rel_mse(dequantize(*nn4.quantize_weight_2d(w, four_over_six=False)[:3]), w)
        assert e46 < rne

    def test_costs_forward_error_against_1d(self):
        """The consistency tradeoff: a 256-element block is coarser than 16. Recorded, not hidden."""
        torch.manual_seed(3)
        w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        e2d = rel_mse(dequantize(*nn4.quantize_weight_2d(w)[:3]), w)
        q = quant_fp4(w, scale_override=1.0)
        e1d = rel_mse(dequantize(q.fp4, q.micro_scales, q.tensor_scale), w)
        print(f"\n2D/1D weight rel-MSE: {e2d:.4e} / {e1d:.4e} = {e2d / e1d:.2f}x")
        assert e2d >= e1d

    def test_zero_weight(self):
        w = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
        fp4, ws, ts, fp4_t, wts = nn4.quantize_weight_2d(w)
        assert torch.equal(dequantize(fp4, ws, ts), w) and torch.isfinite(ts)

    def test_matches_transformer_engine_reference(self):
        """TE's pure-torch NVFP4QuantizerRef at (16, 16) tiles, RNE, is the oracle."""
        te = _load_te_reference()
        if te is None:
            pytest.skip("TransformerEngine checkout not found")
        torch.manual_seed(4)
        w = torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)
        ref = te.NVFP4QuantizerRef(dtype=te.reference_utils.Fp4Formats.E2M1, quant_tile_shape=(16, 16),
                                   rowwise=True, columnwise=False)
        q_te, s_te, _, _, _, _ = ref._quantize(w)
        fp4, ws, ts, _, _ = nn4.quantize_weight_2d(w, four_over_six=False)
        codes_ok = (fp4 == q_te).float().mean().item()
        s_ours = _unblock(ws, 256, 384).view(torch.uint8)
        scales_ok = (s_ours == s_te.view(torch.uint8)).float().mean().item()
        # Not bit-exact by construction: TE multiplies by a reciprocal where this divides, so a
        # value within an ulp of a code boundary may land either side. Bound the disagreement.
        assert codes_ok > 0.999, codes_ok
        assert scales_ok > 0.99, scales_ok


def _load_te_reference():
    """Import TE's reference_nvfp4.py from the sibling checkout with its package stubbed out.

    The module needs only `reference_utils` (standalone), a `gemm` namespace for two type
    annotations, and two base classes from `quantized_tensor`; none of that needs TE built.
    """
    path = TE_ROOT / "reference_nvfp4.py"
    if not path.exists() or "transformer_engine" in sys.modules:
        return None if not path.exists() else sys.modules.get("transformer_engine.pytorch.custom_recipes.reference_nvfp4")

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    module("transformer_engine")
    module("transformer_engine.pytorch")
    module("transformer_engine.pytorch.custom_recipes")

    class Quantizer:
        def __init__(self, *, rowwise, columnwise):
            self.rowwise_usage, self.columnwise_usage = rowwise, columnwise

    module("transformer_engine.pytorch.quantized_tensor", QuantizedTensorStorage=object, Quantizer=Quantizer)
    module("transformer_engine.pytorch.custom_recipes.gemm", MMParams=object, GEMMType=types.SimpleNamespace(FPROP=0, WGRAD=1))

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("transformer_engine.pytorch.custom_recipes.reference_utils", TE_ROOT / "reference_utils.py")
    return load("transformer_engine.pytorch.custom_recipes.reference_nvfp4", path)


# ---------------------------------------------------------------------------------------------
# The backward under each lever
# ---------------------------------------------------------------------------------------------

VARIANTS = {
    "bwd-bf16": dict(bwd_source="bf16"),
    "rht-wgrad": dict(rht="wgrad"),
    "rht-wgrad-bf16": dict(rht="wgrad", bwd_source="bf16"),
    "rht-none": dict(rht="none", bwd_source="bf16"),
    "weight-2d": dict(weight_2d=True, rht="wgrad"),
    "weight-2d-none": dict(weight_2d=True, rht="none", bwd_source="bf16"),
}


def make_layer(cfg, in_features=256, out_features=384, seed=0, **kwargs):
    torch.manual_seed(seed)
    layer = NVFP4Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.float32,
                        numerics=nn4.NumericsConfig(**cfg), **kwargs)
    layer.refresh_weight_cache()
    return layer


class TestBackwardVariants:
    @pytest.mark.parametrize("name", list(VARIANTS))
    def test_gradients_point_at_the_bf16_gradient(self, name):
        """The kernel-family gate: a permuted operand against a natural one reads ~0.1 here."""
        layer = make_layer(VARIANTS[name])
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        (layer(x) * t).sum().backward()
        w = layer.weight.detach().float()
        assert cosine(x.grad, t.float() @ w) == pytest.approx(1.0, abs=0.05)
        assert cosine(layer.weight.grad, t.float().T @ x.detach().float()) == pytest.approx(1.0, abs=0.05)
        assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()

    def test_default_config_is_unchanged(self):
        """Same seeds, same numbers: the levers off must be the Quartet-II backward exactly."""
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        grads = []
        for numerics in (None, nn4.DEFAULT):
            torch.manual_seed(11)
            layer = NVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32, numerics=numerics)
            torch.manual_seed(12)
            (layer(x) * t).sum().backward()
            grads.append(layer.weight.grad.clone())
        assert torch.equal(grads[0], grads[1])

    def _ladder(self, layer, x, t, ref, which):
        bits = {}
        for steps in (1, 16, 256):
            acc = None
            for _ in range(steps):
                layer.zero_grad()
                xx = x.clone().requires_grad_()
                (layer(xx) * t).sum().backward()
                g = layer.weight.grad.float() if which == "w" else xx.grad.float()
                acc = g if acc is None else acc + g
            bits[steps] = eff_bits(acc / steps, ref)
        return bits

    def _wgrad_ref(self, layer, x, t):
        if layer.numerics.bwd_source == "bf16":
            return t.float().T @ x.float()
        # X^T is re-quantized from the saved fp4, so the target is E^T against the dequantized
        # activation -- what the fp4-source estimator is unbiased for.
        xq = quant_fp4(x, scale_override=1.0, mode=layer.mode)
        return t.float().T @ dequantize(xq.fp4, xq.micro_scales, xq.tensor_scale).float()

    @pytest.mark.parametrize("name", ["bwd-bf16", "rht-wgrad", "weight-2d"])
    def test_rotated_wgrad_stays_unbiased(self, name):
        """Port of test_backward_is_unbiased: +1 bit per 4x samples, or the estimator is biased."""
        layer = make_layer(VARIANTS[name], seed=42)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        bits = self._ladder(layer, x, t, self._wgrad_ref(layer, x, t), "w")
        assert bits[16] > bits[1] + 1.3, f"averaging barely helped: {bits}"
        assert bits[256] > bits[16] + 1.3, f"error plateaued, estimator looks biased: {bits}"

    @pytest.mark.parametrize("name,which", [("rht-wgrad", "x"), ("rht-none", "w"), ("rht-none", "x")])
    def test_unrotated_eden_is_biased(self, name, which):
        """The finding this test pins: EDEN's unbiasedness comes from the *random rotation*.

        The values are rounded to nearest in every kernel; only the block scale is stochastic.
        With the sign flips gone, averaging removes the scale noise and stops at the rounding
        bias. TE drops the rotation from dgrad *and* rounds gradient values stochastically; this
        stack can only do the first, so --nvfp4-rht wgrad/none trade unbiasedness for the
        rotation's own quantization error. The probe's ladder shows the same plateau.
        """
        layer = make_layer(VARIANTS[name], seed=42)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        ref = self._wgrad_ref(layer, x, t) if which == "w" else t.float() @ layer.weight.detach().float()
        bits = self._ladder(layer, x, t, ref, which)
        assert bits[256] < bits[16] + 0.5, f"expected a plateau, got {bits}"

    def test_weight_2d_dgrad_plateaus_at_the_weight_error(self):
        """The tradeoff on record: dgrad uses the deterministic forward weight, so averaging
        removes E's error and stops at W_q's. grad_weight is still unbiased (test above)."""
        layer = make_layer(VARIANTS["weight-2d"], seed=43)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        bits = self._ladder(layer, x, t, t.float() @ layer.weight.detach().float(), "x")
        floor = eff_bits(dequantize(layer.fp4_w, layer.fp4_ws, layer.fp4_wg), layer.weight.detach())
        assert bits[256] < bits[16] + 0.5, f"expected a plateau near the weight error: {bits}"
        assert abs(bits[256] - floor) < 0.5, f"plateau {bits[256]:.2f} bits vs weight error {floor:.2f}"

    @pytest.mark.parametrize("name", ["bwd-bf16", "rht-wgrad", "rht-none", "weight-2d"])
    def test_compile_fullgraph(self, name):
        layer = make_layer(VARIANTS[name])
        fn = torch.compile(lambda t: layer(t).float().square().sum(), fullgraph=True)
        x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        fn(x).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()
        assert cosine(layer.weight.grad, 2 * (x.detach().float() @ layer.weight.detach().float().T).flatten(0, 1).T @ x.detach().float().flatten(0, 1)) == pytest.approx(1.0, abs=0.1)

    def test_fused_wgrad_lands_in_the_buffer(self):
        from nanochat.sm120 import fp4_gemm
        fp4_gemm.configure(True)
        try:
            layer = make_layer(VARIANTS["weight-2d-none"])
            store = enable_wgrad_accum(layer)
            x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
            t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
            (layer(x) * t).sum().backward()
            assert layer.weight.grad is None
            assert cosine(layer.fp4_main_grad, t.float().T @ x.float()) == pytest.approx(1.0, abs=0.05)
            store.zero_()
        finally:
            fp4_gemm.configure(False)


class TestFP8Forward:
    """FP8FwdNVFP4Linear: the forward is fp8-exact-ish, the backward is the rotated bf16-source estimator."""

    def _layer(self, seed=0):
        from nanochat.sm120.nvfp4_fp8fwd import FP8FwdNVFP4Linear
        torch.manual_seed(seed)
        layer = FP8FwdNVFP4Linear(256, 384, bias=False, device="cuda", dtype=torch.float32)
        layer.refresh_weight_cache()
        return layer

    def test_forward_is_fp8(self):
        layer = self._layer()
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ layer.weight.detach().float().T
        got = layer(x)
        assert rel_mse(got, ref) < 4e-3 and eff_bits(got, ref) > 4      # fp8 (~4.8 bits), not fp4 (~3.3)
        assert layer.fp8_w.dtype == torch.float8_e4m3fn

    def test_backward_matches_and_is_unbiased(self):
        layer = self._layer(42)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
        ref_w = t.float().T @ x.float()
        bits = {}
        for steps in (1, 16, 256):
            acc = None
            for _ in range(steps):
                layer.weight.grad = None
                xx = x.clone().requires_grad_()
                (layer(xx) * t).sum().backward()
                acc = layer.weight.grad.float() if acc is None else acc + layer.weight.grad.float()
            bits[steps] = eff_bits(acc / steps, ref_w)
        assert cosine(xx.grad, t.float() @ layer.weight.detach().float()) == pytest.approx(1.0, abs=0.05)
        assert bits[16] > bits[1] + 1.3 and bits[256] > bits[16] + 1.3, bits

    def test_cache_and_unaligned_rows(self):
        layer = self._layer()
        x = torch.randn(77, 256, device="cuda", dtype=torch.bfloat16)
        assert layer(x).shape == (77, 384)
        assert all(not k.startswith(("fp8_", "fp4_")) for k in layer.state_dict())

    def test_compile_fullgraph(self):
        layer = self._layer()
        fn = torch.compile(lambda t: layer(t).float().square().sum(), fullgraph=True)
        x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        fn(x).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()

    def test_fused_wgrad(self):
        from nanochat.sm120 import fp4_gemm
        fp4_gemm.configure(True)
        try:
            layer = self._layer()
            store = enable_wgrad_accum(layer)
            x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
            t = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
            (layer(x) * t).sum().backward()
            assert layer.weight.grad is None
            assert cosine(layer.fp4_main_grad, t.float().T @ x.float()) == pytest.approx(1.0, abs=0.05)
            store.zero_()
        finally:
            fp4_gemm.configure(False)


class TestWeightCache2D:
    def test_refresh_tracks_the_weight(self):
        layer = make_layer(VARIANTS["weight-2d"])
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        before = layer(x)
        with torch.no_grad():
            layer.weight.mul_(2.0)
        assert torch.equal(layer(x), before), "stale by design until refresh"
        layer.refresh_weight_cache()
        assert cosine(layer(x), 2 * before) == pytest.approx(1.0, abs=1e-2)

    def test_buffers_stay_out_of_checkpoints(self):
        layer = make_layer(VARIANTS["weight-2d"])
        assert layer.fp4_wt is not None and layer.fp4_wts is not None
        assert all(not k.startswith("fp4_") for k in layer.state_dict())

    def test_1d_cache_leaves_the_transposed_form_empty(self):
        layer = make_layer({})
        assert layer.fp4_w is not None and layer.fp4_wt is None and layer.fp4_wts is None
