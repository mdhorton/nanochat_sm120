"""Pure-torch NVFP4 quantization, as the oracle the kernel tests measure against.

Written from the format and the scheme, not from upstream's CUDA, so a test that compares the
two is a real cross-check rather than a restatement. Slow and allocation-heavy on purpose --
nothing here is on a training path.

It cannot be bit-exact: the kernels divide with `rcp.approx.ftz.f32`, so a value sitting within
an ulp of an E2M1 code boundary can land either side of it. The tests allow for that by
bounding the *rate* of code disagreement rather than demanding none.
"""
import torch

# E2M1's eight magnitudes, and the midpoints round-to-nearest-even compares against.
E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_MIDPOINTS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
# Ties go to even codes: 0.25 -> 0 (0.0), 0.75 -> 2 (1.0), 1.25 -> 2, 1.75 -> 4 (2.0), etc.
_TIE_TO_UPPER = [False, True, False, True, False, True, False]

E4M3_MAX = 448.0


def e2m1_codes(x):
    """Round-to-nearest-even onto E2M1's grid, as 4-bit codes: magnitude index 0..7, sign in bit 3."""
    a = x.abs()
    code = torch.zeros_like(a, dtype=torch.uint8)
    for i, (m, tie_up) in enumerate(zip(_MIDPOINTS, _TIE_TO_UPPER)):
        code = torch.where(a > m if not tie_up else a >= m, torch.full_like(code, i + 1), code)
    return torch.where(x < 0, code | 8, code)


def round_e2m1(x):
    """Round-to-nearest-even onto E2M1's grid. Returns dequantized magnitudes with sign."""
    code = e2m1_codes(x).to(torch.long)
    levels = torch.tensor(E2M1_LEVELS, dtype=x.dtype, device=x.device)
    return levels[code & 7] * torch.where(code & 8 != 0, -1.0, 1.0).to(x.dtype)


def _quantize_with_divisor(x_groups, group_amax, tensor_scale, val_max):
    """One 4/6 candidate: place each group's max on `val_max`, then quantize against it.

    Returns (dequantized values, the stored E4M3 block scale as fp32).
    """
    s_group = group_amax / val_max
    s_e4m3 = (s_group / tensor_scale).to(torch.float8_e4m3fn).to(torch.float32)
    s_e4m3 = torch.where(s_e4m3 == 0, torch.ones_like(s_e4m3), s_e4m3)
    step = s_e4m3 * tensor_scale
    return round_e2m1(x_groups / step) * step, s_e4m3


def quantize_reference(x, *, four_over_six=True, scale_override=1.0, group_size=16):
    """bf16 [M, K] -> (dequantized bf16, block scales [M, K/16], per-tensor scale).

    Two-level scaling, both levels load-bearing. The per-tensor scale puts the largest block
    scale at the top of E4M3's range; without it the stored block scales for tensors of the
    magnitude a model actually produces fall under E4M3's smallest normal and quantize to zero.
    The `256 vs 448` split is upstream's: 4/6 keeps headroom because its second candidate
    scales groups up by 6/4.
    """
    x32 = x.to(torch.float32)
    amax = x32.abs().max()
    scales_max = 256.0 if four_over_six else 448.0
    tensor_scale = amax * scale_override / (6.0 * scales_max)
    tensor_scale = torch.where(amax == 0, torch.ones_like(tensor_scale), tensor_scale)

    groups = x32.unflatten(-1, (-1, group_size))
    group_amax = groups.abs().amax(dim=-1, keepdim=True)

    best, best_scale = _quantize_with_divisor(groups, group_amax, tensor_scale, 6.0 / scale_override)
    if four_over_six:
        alt, alt_scale = _quantize_with_divisor(groups, group_amax, tensor_scale, 4.0 / scale_override)
        err_six = (groups - best).pow(2).sum(dim=-1, keepdim=True)
        err_four = (groups - alt).pow(2).sum(dim=-1, keepdim=True)
        take_alt = err_four < err_six
        best = torch.where(take_alt, alt, best)
        best_scale = torch.where(take_alt, alt_scale, best_scale)

    return best.flatten(-2).to(torch.bfloat16), best_scale.squeeze(-1), tensor_scale
