"""Tests for NVFP4 (Quartet II) training kernels and module."""

import pytest
import torch

cuda_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
pytestmark = pytest.mark.skipif(not cuda_ok, reason="needs Blackwell (sm100+) GPU")

if cuda_ok:
    from nanochat.nvfp4_kernels import (
        GROUP, HAD_DIM, fp4_forward_quant, fp4_rht_eden_quant, fp4_dequant,
        nvfp4_mm, hadamard_matrix,
    )
    from nanochat.nvfp4 import NVFP4Linear, _rerotate, convert_to_nvfp4_training
    from nanochat.nvfp4_kernels import _rht_eden_op, _rht_eden_t_op, _rht_eden_rq_t_op

DEV = "cuda"


def _ref_rtn_quant(x, four_over_six, scale_override=1.0):
    """Pure-torch mirror of the RTN kernel (same fp32 ops -> bit-exact)."""
    val_max = 6.0 / scale_override
    scales_max = 256.0 if four_over_six else 448.0
    xf = x.float()
    amax = xf.abs().amax()
    s_dec = torch.where(amax == 0, torch.ones_like(amax), amax / (scales_max * val_max))
    xg = xf.view(-1, GROUP)

    def cand(mult):
        rq = (xg.abs().amax(1, keepdim=True) / val_max * mult / s_dec).to(torch.float8_e4m3fn)
        r = rq.float()
        r = torch.where(r == 0, torch.ones_like(r), r)
        xs = xg / (r * s_dec)
        a = xs.abs()
        idx = (torch.zeros_like(a, dtype=torch.int32)
               + (a >= 0.25).int() + (a >= 0.75).int() + (a >= 1.25).int() + (a >= 1.75).int()
               + (a >= 2.5).int() + (a >= 3.5).int() + (a >= 5).int())
        val = torch.where(idx == 7, 6.0, torch.where(idx == 6, 4.0, torch.where(idx == 5, 3.0, idx * 0.5)))
        val = val * torch.where(xs < 0, -1.0, 1.0)
        deq = val * r * s_dec
        return idx, rq, deq

    i6, r6q, d6 = cand(1.0)
    if four_over_six:
        i4, r4q, d4 = cand(1.5)
        use6 = ((xg - d6) ** 2).sum(1, keepdim=True) <= ((xg - d4) ** 2).sum(1, keepdim=True)
        idx = torch.where(use6, i6, i4)
        sq = torch.where(use6.squeeze(1), r6q.squeeze(1), r4q.squeeze(1))
    else:
        idx, sq = i6, r6q.squeeze(1)
    code = (idx + 8 * (xg < 0).int()).view(x.shape[0], -1)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, sq.view(x.shape[0], -1).to(torch.float8_e4m3fn), s_dec


@pytest.mark.parametrize("four_over_six", [False, True])
def test_rtn_quant_matches_reference(four_over_six):
    torch.manual_seed(0)
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16) * 3
    codes, scales, s_dec = fp4_forward_quant(x, four_over_six)
    rc, rs, rsd = _ref_rtn_quant(x, four_over_six)
    assert torch.equal(codes, rc)
    assert torch.equal(scales.view(torch.uint8), rs.view(torch.uint8))
    assert torch.allclose(s_dec, rsd)


def test_rtn_zero_input():
    z = torch.zeros(128, 128, device=DEV, dtype=torch.bfloat16)
    codes, scales, s_dec = fp4_forward_quant(z, True)
    assert fp4_dequant(codes, scales, s_dec).abs().max() == 0


def test_rtn_roundtrip_quality():
    torch.manual_seed(1)
    x = torch.randn(256, 768, device=DEV, dtype=torch.bfloat16)
    codes, scales, s_dec = fp4_dequant_args = fp4_forward_quant(x, True)
    xd = fp4_dequant(*fp4_dequant_args)
    mrqe = ((xd.float() - x.float()) ** 2).mean() / (x.float() ** 2).mean()
    assert (-mrqe.log2() / 2) > 3.0  # ~3.5 effective bits expected


def test_mm_matches_dequant_reference():
    torch.manual_seed(2)
    x = torch.randn(256, 768, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(512, 768, device=DEV, dtype=torch.bfloat16) / 27
    a = fp4_forward_quant(x, True)
    b = fp4_forward_quant(w, True)
    out = nvfp4_mm(*a, *b)
    ref = fp4_dequant(*a).float() @ fp4_dequant(*b).float().t()
    rel = (out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel < 0.01


def test_eden_unbiased():
    torch.manual_seed(3)
    g = torch.randn(256, 512, device=DEV, dtype=torch.bfloat16)
    had = hadamard_matrix(HAD_DIM, DEV)
    trials, acc = 32, torch.zeros(256, 512, device=DEV)
    for _ in range(trials):
        h = _rerotate(had)
        deq = fp4_dequant(*fp4_rht_eden_quant(g, h)).float()
        acc += (deq.view(256, -1, HAD_DIM) @ h.float().t()).view(256, 512)
    err1 = ((deq.view(256, -1, HAD_DIM) @ h.float().t()).view(256, 512) - g.float()).pow(2).mean()
    errN = (acc / trials - g.float()).pow(2).mean()
    assert errN < err1 / 6  # unbiased -> error shrinks ~1/trials


def test_linear_grads_to_fp32_master():
    torch.manual_seed(4)
    lin = NVFP4Linear(768, 512, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        lin.weight.copy_(torch.randn_like(lin.weight) / 27)
    x = torch.randn(2, 128, 768, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    assert y.shape == (2, 128, 512) and y.dtype == torch.bfloat16
    y.float().pow(2).mean().backward()
    assert lin.weight.grad is not None and lin.weight.grad.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16

    ref = torch.nn.Linear(768, 512, bias=False, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        ref.weight.copy_(lin.weight.to(torch.bfloat16))
    xr = x.detach().clone().requires_grad_(True)
    ref(xr).float().pow(2).mean().backward()
    cos_w = torch.nn.functional.cosine_similarity(
        lin.weight.grad.flatten(), ref.weight.grad.float().flatten(), dim=0)
    cos_x = torch.nn.functional.cosine_similarity(
        x.grad.float().flatten(), xr.grad.float().flatten(), dim=0)
    assert cos_w > 0.95 and cos_x > 0.95


def test_eval_odd_rows():
    lin = NVFP4Linear(768, 512, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        y5 = lin(torch.randn(5, 768, device=DEV, dtype=torch.bfloat16))
        y1 = lin(torch.randn(1, 1, 768, device=DEV, dtype=torch.bfloat16))
    assert y5.shape == (5, 512) and y1.shape == (1, 1, 512)


def test_no_bwd_quant():
    lin = NVFP4Linear(256, 256, device=DEV, dtype=torch.float32, bwd_quant=False)
    x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    lin(x).sum().backward()
    assert lin.weight.grad is not None and x.grad is not None


def test_convert_d12_gpt():
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=12,
                       n_head=6, n_kv_head=6, n_embd=768, window_pattern="L")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=DEV)
    model.init_weights()
    keys_before = set(model.state_dict().keys())
    params_before = model.num_matmul_params()
    n = convert_to_nvfp4_training(model)
    assert n == 72  # 12 layers x (c_q, c_k, c_v, attn.c_proj, c_fc, mlp.c_proj)
    assert not isinstance(model.lm_head, NVFP4Linear)
    assert not isinstance(model.smear_gate, NVFP4Linear)
    for block in model.transformer.h:
        if hasattr(block.attn, "ve_gate") and block.attn.ve_gate is not None:
            assert not isinstance(block.attn.ve_gate, NVFP4Linear)
    assert model.num_matmul_params() == params_before
    assert set(model.state_dict().keys()) == keys_before  # had/non-persistent
    idx = torch.randint(0, 32768, (1, 128), device=DEV)
    with torch.no_grad():
        logits = model(idx)
    assert torch.isfinite(logits.float()).all()  # zero-init c_proj -> amax==0 path


@pytest.mark.slow
def test_compile_fullgraph():
    torch._dynamo.reset()
    lin = NVFP4Linear(256, 256, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        lin.weight.copy_(torch.randn_like(lin.weight) / 16)
    x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    f = torch.compile(lambda t: lin(t).float().pow(2).mean(), fullgraph=True, dynamic=False)
    f(x).backward()
    assert lin.weight.grad is not None


def test_eden_t_matches_flat():
    torch.manual_seed(5)
    x = torch.randn(384, 256, device=DEV, dtype=torch.bfloat16)
    h = _rerotate(hadamard_matrix(HAD_DIM, DEV))
    ct, st, at = _rht_eden_t_op(x, h, 6.0)          # quantizes x^T along dim0 of x
    cf, sf, af = _rht_eden_op(x.t().contiguous(), h, 6.0)
    assert torch.equal(at, af)
    assert torch.equal(ct, cf)
    # reduction layout differs by 1 ulp in the EDEN sums; feeds stochastic rounding anyway
    assert torch.allclose(st, sf, rtol=1e-5, atol=1e-7)


def test_requant_t_matches_composition():
    torch.manual_seed(6)
    x = torch.randn(256, 384, device=DEV, dtype=torch.bfloat16)
    codes, scales, s_dec = fp4_forward_quant(x, True)
    h = _rerotate(hadamard_matrix(HAD_DIM, DEV))
    cq, sq, aq = _rht_eden_rq_t_op(codes, scales, s_dec, h, 6.0)
    ref_in = fp4_dequant(codes, scales, s_dec).t().contiguous()
    cf, sf, af = _rht_eden_op(ref_in, h, 6.0)
    assert torch.equal(aq, af)
    assert torch.equal(cq, cf)
    assert torch.allclose(sq, sf, rtol=1e-5, atol=1e-7)
