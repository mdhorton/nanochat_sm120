"""
Test delayed FP8 scaling (nanochat.fp8, --fp8-scaling delayed).

Delayed scaling replaces the amax reduction that gates every cast with a scale read from a
buffer, and records the next step's amax from the fp8 output instead. The properties that
matter are that the scales track the real amaxes, that the two self-correcting search paths
work, and that none of it changes under torch.compile.

Requires a GPU (fp8 needs sm89+, and the compiled tests dominate the runtime of this file).

python -m pytest tests/test_fp8_delayed.py -v
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="fp8 tests require CUDA")

if cuda_available:
    import nanochat.sm120 as sm120
    from nanochat import fp8
    from nanochat.fp8 import Float8Linear, convert_to_float8_training
    from nanochat.gpt import Linear
    from nanochat.sm120 import fp8_state
    from nanochat.sm120.fp8_state import enable_delayed_scaling

    sm120.install_fp8_backend()   # delayed scaling is an sm120 backend feature

DEVICE = "cuda"
E4M3_MAX = 448.0


class Net(torch.nn.Module):
    """Two fp8-eligible Linears with an elementwise op between them, as in a real block."""

    def __init__(self, d_in=256, d_hidden=512):
        super().__init__()
        self.a = Linear(d_in, d_hidden, bias=False)
        self.b = Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)).square())


def build(delayed, seed=0, **kwargs):
    torch.manual_seed(seed)
    model = Net().to(DEVICE).to(torch.bfloat16)
    convert_to_float8_training(model)
    state = enable_delayed_scaling(model, **kwargs) if delayed else None
    return model, state


def batch(step, mult=1.0, d_in=256):
    torch.manual_seed(1000 + step)
    return (torch.randn(128, d_in, device=DEVICE, dtype=torch.bfloat16) * mult).requires_grad_()


def run(model, state, steps, compiled=False, mult=1.0):
    """Train for `steps` micro-steps, returning the final gradients."""
    fn = torch.compile(model) if compiled else model
    grads = None
    for step in range(steps):
        fn(batch(step, mult)).square().sum().backward()
        grads = [p.grad.float().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        if state is not None:
            state.update()
    return grads


def implied_amax(module, role, fp8_max=E4M3_MAX):
    """The amax the layer's current scale is aimed at, undoing target/margin headroom."""
    return fp8_max / getattr(module, f"fp8_scale_{role}").item()


def test_dynamic_is_unchanged_by_the_delayed_code_path():
    """A model without delayed scaling must take the original path, buffers unset."""
    model, state = build(delayed=False)
    assert state is None
    for mod in model.modules():
        if isinstance(mod, Float8Linear):
            assert mod.fp8_scale_in is None and mod.fp8_amax_in is None
    grads = run(model, None, steps=2)
    assert all(torch.isfinite(g).all() for g in grads)


def test_scales_track_the_real_amaxes():
    """After a few steps every scale should aim at its tensor's true amax, times the headroom
    the margin and the one-ulp scale target deliberately leave."""
    model, state = build(delayed=True, margin=2.0)
    run(model, state, steps=4)
    headroom = 2.0 * (E4M3_MAX / fp8_state._SCALE_TARGET[torch.float8_e4m3fn])
    for mod in [model.a, model.b]:
        true = mod.weight.float().abs().max().item()
        assert implied_amax(mod, "w") == pytest.approx(true * headroom, rel=0.1)
    # the input to the first layer is the batch itself, so its amax is known exactly
    true_in = batch(3).float().abs().max().item()
    assert implied_amax(model.a, "in") == pytest.approx(true_in * headroom, rel=0.15)


def test_no_spurious_saturation_search_on_a_steady_tensor():
    """A tensor sitting at its historical amax reads back FMAX-ish; the one-ulp scale target
    is what keeps that from being mistaken for clipping and doubling the scale every step."""
    model, state = build(delayed=True, margin=2.0)
    run(model, state, steps=3)
    before = implied_amax(model.a, "w")
    run(model, state, steps=6)
    after = implied_amax(model.a, "w")
    # weights barely move here (no optimizer), so the estimate must not drift by 2x
    assert after == pytest.approx(before, rel=0.25)


def test_amax_search_recovers_from_a_spike():
    """A jump larger than the margin clips, which pins amax(q) at FMAX -- a floor, not a
    reading. The search has to grow past it rather than trust it."""
    model, state = build(delayed=True)
    run(model, state, steps=4)
    before = model.a.fp8_scale_in.item()
    grads = run(model, state, steps=4, mult=100.0)
    after = model.a.fp8_scale_in.item()
    assert all(torch.isfinite(g).all() for g in grads), "a spike must not produce inf/nan"
    assert after < before / 20.0, f"scale failed to follow a 100x spike: {before} -> {after}"


def test_amax_search_recovers_from_total_underflow():
    """The mirror case: if the scale is far too small everything flushes to zero and amax(q)
    reads 0, which carries no information. The search has to walk back down."""
    model, state = build(delayed=True)
    with torch.no_grad():  # force a hopeless scale, as _INIT_AMAX would on a tiny tensor
        state.scales[fp8_state._SCALE].fill_(1e-12)
        state.scales[fp8_state._INV].fill_(1e12)
    before = model.a.fp8_scale_in.item()
    run(model, state, steps=12)
    after = model.a.fp8_scale_in.item()
    assert after > before * 100.0, f"scale failed to search back up: {before} -> {after}"


def test_history_seed_does_not_outlive_the_first_update():
    """The _INIT_AMAX seed goes into the scales, never into the history. In the history it
    would be a max over history_len steps and hold every activation at the assumed amax."""
    model, state = build(delayed=True, history_len=16)
    assert state.filled == 0
    run(model, state, steps=1)
    assert state.filled == 1
    # one real measurement, so the seed must be entirely gone from the input scale
    assert implied_amax(model.a, "in") < fp8_state._INIT_AMAX / 10.0


def test_history_window_holds_the_max():
    """The scale is the max over the window, so a spike stays priced in until it ages out."""
    model, state = build(delayed=True, history_len=4)
    run(model, state, steps=4)
    run(model, state, steps=1, mult=50.0)
    spiked = model.a.fp8_scale_in.item()
    run(model, state, steps=3)  # still inside the 4-step window
    assert model.a.fp8_scale_in.item() == pytest.approx(spiked, rel=1e-6)
    run(model, state, steps=1)  # now it has aged out
    assert model.a.fp8_scale_in.item() > spiked * 5.0


def test_amax_is_the_max_over_grad_accum_micro_steps():
    """update() runs once per optimizer step, so the in-graph write accumulates a max across
    the micro-steps rather than letting the last one win."""
    model, state = build(delayed=True)
    model(batch(0, mult=10.0)).square().sum().backward()
    big = state.amax[fp8_state._IN, 0].item()
    model(batch(1, mult=1.0)).square().sum().backward()  # a smaller micro-batch must not lower it
    assert state.amax[fp8_state._IN, 0].item() == pytest.approx(big)


def test_zero_initialised_weight_does_not_poison_the_scale():
    """gpt.py zero-inits both c_proj (24 layers at d12), so a seeded weight amax of 0 is a
    real case, not a pathological one. Seeding a scale from it gives 1/0 = inf and the whole
    model comes out NaN on step 0."""
    model, state = build(delayed=True)
    with torch.no_grad():
        model.a.weight.zero_()
        model.b.weight.zero_()
    state = enable_delayed_scaling(model)
    assert (state.scales > 0).all(), "a zero weight must not seed a zero scale"
    assert torch.isfinite(state.scales).all(), "a zero weight must not seed an infinite scale"
    grads = run(model, state, steps=3)
    assert all(torch.isfinite(g).all() for g in grads)


def test_buffers_stay_out_of_the_state_dict():
    """Checkpoints must remain interchangeable with a bf16 or dynamically scaled run."""
    plain, _ = build(delayed=False)
    delayed, _ = build(delayed=True)
    assert set(plain.state_dict().keys()) == set(delayed.state_dict().keys())
    assert not any("fp8_" in k for k in delayed.state_dict())


@pytest.mark.slow
def test_compile_matches_eager():
    """The whole point is that the buffers survive torch.compile, including the write the
    backward makes -- and that the compiled scales end up where the eager ones do."""
    torch._dynamo.reset()
    eager_model, eager_state = build(delayed=True)
    eager_grads = run(eager_model, eager_state, steps=4, compiled=False)

    torch._dynamo.reset()
    comp_model, comp_state = build(delayed=True)
    comp_grads = run(comp_model, comp_state, steps=4, compiled=True)

    for name in ("in", "w", "go"):
        assert getattr(comp_model.a, f"fp8_scale_{name}").item() == pytest.approx(
            getattr(eager_model.a, f"fp8_scale_{name}").item(), rel=1e-3
        ), f"{name} scale diverged between eager and compiled"
    # not bitwise: Inductor generates a different graph, so the fp8 rounding path differs by
    # up to an ulp (the module docstring says as much). The scales above are the strict check.
    for eag, comp in zip(eager_grads, comp_grads):
        cos = torch.nn.functional.cosine_similarity(eag.flatten(), comp.flatten(), dim=0)
        assert cos.item() > 0.999, f"gradient cosine {cos.item():.6f}"


@pytest.mark.slow
def test_gradients_stay_close_to_dynamic_scaling():
    """Delayed scaling is an approximation of dynamic scaling, not of bf16: what matters is
    that using a one-step-stale scale does not materially move the gradient direction."""
    for compiled in (False, True):
        torch._dynamo.reset()
        dyn_model, _ = build(delayed=False)
        dyn_grads = run(dyn_model, None, steps=6, compiled=compiled)

        torch._dynamo.reset()
        del_model, del_state = build(delayed=True)
        del_grads = run(del_model, del_state, steps=6, compiled=compiled)

        for dyn, dly in zip(dyn_grads, del_grads):
            cos = torch.nn.functional.cosine_similarity(dyn.flatten(), dly.flatten(), dim=0)
            assert cos.item() > 0.98, f"compiled={compiled}: gradient cosine {cos.item():.4f}"
