"""NVFP4 quantize / rotate / requantize wrappers over the vendored Quartet-II kernels.

Ported from upstream `quartet2/quant.py`. The only change is where the kernels come from:
upstream imports a prebuilt nanobind module, this calls `ext.load()` (JIT, cached).

All of these produce an :class:`NVFP4Quant` -- packed E2M1 codes, per-16 E4M3 block scales, and
one fp32 per-tensor scale. **The block scales come out already in the cutlass 128x4 swizzle**
(`cvt_quant_to_fp4_get_sf_out_offset` in csrc/utils.cuh), which is exactly the layout
`torch._scaled_mm` indexes them through, so they feed the GEMM with no rearrangement. That
layout is not optional and not checked: passed row-major scales, `_scaled_mm` returns confident
garbage rather than an error.

The per-tensor scale is *not* applied by the GEMM -- see `nanochat.sm120.nvfp4.fp4_mm`.
"""
from collections import namedtuple
from enum import Enum

import torch

from .ext import load


# torch-compile compatible wrapping
@torch.library.custom_op("quartet2::four_six", mutates_args=("o", "s", "t"))
def _four_six_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float) -> None:
    load().four_six_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override)


@torch.library.custom_op("quartet2::rtn", mutates_args=("o", "s", "t"))
def _rtn_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float) -> None:
    load().rtn_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override)


@torch.library.custom_op("quartet2::eden", mutates_args=("o", "s", "t"))
def _eden_fp4_op(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, amax: torch.Tensor, scale_override: float, seed: torch.Tensor) -> None:
    load().eden_fp4(o, s.view(torch.uint8), t, x.detach(), amax, scale_override, seed)


@torch.library.custom_op("quartet2::group_transform_eden", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _group_transform_128_eden(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                              scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                              h: torch.Tensor, x: torch.Tensor, seed: torch.Tensor, fp4_max: float, fp8_max: float, transpose: bool) -> None:
    load().group_transform_128_eden(
        o, s.view(torch.uint8), t, scratch_scales, scratch_max, h, x.detach(), seed,
        fp4_max=fp4_max, fp8_max=fp8_max, transpose=transpose,
    )


@torch.library.custom_op("quartet2::rht128_eden", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _transform_rht128_eden(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                           scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                           h: torch.Tensor, x: torch.Tensor, seed: torch.Tensor, fp4_max: float, fp8_max: float, transpose: bool) -> None:
    load().rht128_eden(
        o, s.view(torch.uint8), t, scratch_scales, scratch_max, h, x.detach(), seed,
        fp4_max=fp4_max, fp8_max=fp8_max, transpose=transpose,
    )


@torch.library.custom_op("quartet2::dequant_tp_had_quant", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _dequant_tp_had_quant(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                          scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                          h: torch.Tensor, x: torch.Tensor, x_group_scales: torch.Tensor, x_tensor_scale: torch.Tensor,
                          seed: torch.Tensor, fp4_max: float, fp8_max: float) -> None:
    load().dequant_tp_had_quant(
        o, s, t, scratch_scales, scratch_max, h, x, x_group_scales, x_tensor_scale, seed,
        fp4_max=fp4_max, fp8_max=fp8_max,
    )


@torch.library.custom_op("quartet2::had128_requant", mutates_args=("o", "s", "t", "scratch_scales", "scratch_max"))
def _rht128_requant(o: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                    scratch_scales: torch.Tensor, scratch_max: torch.Tensor,
                    h: torch.Tensor, x: torch.Tensor, x_group_scales: torch.Tensor, x_tensor_scale: torch.Tensor,
                    seed: torch.Tensor, fp4_max: float, fp8_max: float) -> None:
    load().rht128_requant(
        o, s, t, scratch_scales, scratch_max, h, x, x_group_scales, x_tensor_scale, seed,
        fp4_max=fp4_max, fp8_max=fp8_max,
    )


NVFP4Quant = namedtuple("NVFP4Quant", ["fp4", "micro_scales", "tensor_scale"])


class NVFP4QuantMode(Enum):
    RNE = 0
    FOUR_SIX = 1
    EDEN = 2


def new_seed():
    """A 0-d CPU int64 seed tensor, the form the kernels take.

    A tensor rather than a Python int on purpose: under `torch.compile` an int would be baked
    into the graph as a constant, so every replay would reuse one stochastic-rounding seed.
    """
    return torch.randint(0, 2**32, (), dtype=torch.int64)


def quant_fp4(x, *, scale_override: float, amax: torch.Tensor = None, mode=NVFP4QuantMode.RNE, seed: torch.Tensor = None) -> NVFP4Quant:
    q = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty((x.shape[0], x.shape[1] // 16), device=x.device, dtype=torch.float8_e4m3fn)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.shape[0] % 128 == 0
    assert x.shape[1] % 128 == 0

    if amax is None:
        # inf-norm rather than max(abs(x)), which materializes a full |x| temporary. Bit-
        # identical (torch.equal, including zeros/negatives/tiny/huge) and 2.9x faster **in
        # eager** -- 1.038 -> 0.363 ms at [16384, 3072], 1.832 -> 0.632 at [16384, 5120].
        #
        # **Worth ~0% in training**: measured +0.09% against a 0.07% repeat spread at d12/dbs8,
        # because Inductor already fuses the max(abs(...)) form in the compiled path. Kept
        # because it is free and helps the eager callers (tests, probes), not because it speeds
        # up a run. Do not quote the 2.9x as a training number.
        #
        # The reduction itself is still a full pass over x that the quantize kernel repeats.
        # Removing it needs an amax *history* (TransformerEngine's delayed scaling, perf-log
        # experiment 10) -- NVFP4's per-tensor scale must be known before quantizing, so it
        # cannot fuse into the kernel. That is a numerics change; this is not.
        amax = torch.linalg.vector_norm(x, ord=float("inf")).to(torch.float32)
    else:
        assert amax.dtype == torch.float32
    global_scale = torch.empty((), device=x.device, dtype=torch.float32)
    if mode == NVFP4QuantMode.FOUR_SIX:
        _four_six_fp4_op(q, s, global_scale, x, amax, scale_override)
    elif mode == NVFP4QuantMode.RNE:
        _rtn_fp4_op(q, s, global_scale, x, amax, scale_override)
    elif mode == NVFP4QuantMode.EDEN:
        assert seed is not None
        _eden_fp4_op(q, s, global_scale, x, amax, scale_override, seed)
    else:
        raise ValueError(f"Unknown NVFP4QuantMode: {mode}")

    return NVFP4Quant(q, s, global_scale)


def _alloc_outputs(rows, cols, device, out, group_scales, tensor_scale, scratch_scales, scratch_amax, scratch_div):
    """The five output/scratch buffers the fused transform kernels write, allocated if absent."""
    if out is None:
        out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    else:
        out = out.reshape(-1, out.shape[-1])

    if group_scales is None:
        group_scales = torch.empty((rows, cols // 16), device=device, dtype=torch.float8_e4m3fn)
    else:
        group_scales = group_scales.reshape(-1, group_scales.shape[-1])

    if tensor_scale is None:
        tensor_scale = torch.empty((), device=device, dtype=torch.float32)
    if scratch_scales is None:
        scratch_scales = torch.empty((rows * cols // scratch_div), device=device, dtype=torch.bfloat16)
    if scratch_amax is None:
        scratch_amax = torch.empty((), device=device, dtype=torch.uint32)
    return out, group_scales, tensor_scale, scratch_scales, scratch_amax


def quant_had_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
        transpose: bool = False):
    """Transform `x` as `x h^t` with a full 128x128 `h`, then quantize the result to NVFP4.

    :param h: 128x128 transform matrix
    :param x: bf16 input, both dims a multiple of 128
    :param seed: seed for deterministic stochastic rounding
    :param scale_override: modulates the fp4 scaling factor
    :param transpose: transpose `x` before transforming
    :return: :class:`NVFP4Quant`
    """
    if seed is None:
        seed = new_seed()

    assert x.dim() == 2
    rows, cols = x.shape
    if transpose:
        rows, cols = cols, rows

    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()
    assert h.shape == (128, 128)

    out, group_scales, tensor_scale, scratch_scales, scratch_amax = _alloc_outputs(
        rows, cols, x.device, out, group_scales, tensor_scale, scratch_scales, scratch_amax, scratch_div=16)

    _group_transform_128_eden(
        out, group_scales, tensor_scale, scratch_scales, scratch_amax, h, x, seed,
        fp4_max=6.0 / scale_override, fp8_max=256.0, transpose=transpose,
    )
    return NVFP4Quant(out, group_scales.reshape(rows, cols // 16), tensor_scale)


def rht128_quant_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
        transpose: bool = False):
    """Transform `x` as `x h^t` and quantize to NVFP4, with EDEN stochastic rounding.

    :param h: the top 16x128 slice of a 128x128 Hadamard. The rest is regenerated in-kernel by
        Sylvester's construction, with rows reordered as in :func:`.rht.swizzle_hadamard`.
    :param x: bf16 input, both dims a multiple of 128
    :param seed: seed for deterministic stochastic rounding
    :param scale_override: modulates the fp4 scaling factor
    :param transpose: transpose `x` before transforming
    :return: :class:`NVFP4Quant`
    """
    if seed is None:
        seed = new_seed()

    assert x.dim() == 2
    rows, cols = x.shape
    if transpose:
        rows, cols = cols, rows

    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()

    out, group_scales, tensor_scale, scratch_scales, scratch_amax = _alloc_outputs(
        rows, cols, x.device, out, group_scales, tensor_scale, scratch_scales, scratch_amax, scratch_div=16)

    _transform_rht128_eden(
        out, group_scales, tensor_scale, scratch_scales, scratch_amax, h, x, seed,
        fp4_max=6.0 / scale_override, fp8_max=256.0, transpose=transpose,
    )
    return NVFP4Quant(out, group_scales.reshape(rows, cols // 16), tensor_scale)


def _check_requant_input(x, x_group_scales, x_tensor_scale, h):
    assert x.dim() == 2
    assert x.dtype == torch.uint8
    assert x.is_cuda
    assert x.is_contiguous()
    assert x_group_scales.dtype == torch.float8_e4m3fn
    assert x_group_scales.shape[0] == x.shape[0]
    assert x_group_scales.shape[1] == x.shape[1] // 8
    assert x_group_scales.is_cuda
    assert x_group_scales.is_contiguous()
    assert x_tensor_scale.dim() == 0
    assert x_tensor_scale.dtype == torch.float32
    assert h.dtype == torch.bfloat16
    assert h.is_cuda
    assert h.is_contiguous()


def dequant_tp_had_eden(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        x_group_scales: torch.Tensor,
        x_tensor_scale: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
):
    """Transform an already-NVFP4 `x` as `x^t h^t` with a full 128x128 `h`, and requantize.

    Note: the output is swizzled -- each group of 16 is written in the order
    `[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]`. A matrix product against another
    operand with the same swizzle is correct; otherwise rearrange `h` so the swizzle matches the
    output you want.

    :return: :class:`NVFP4Quant`
    """
    if seed is None:
        seed = new_seed()

    _check_requant_input(x, x_group_scales, x_tensor_scale, h)
    assert h.shape == (128, 128)
    rows, cols = 2 * x.shape[1], x.shape[0]  # transposed, and 2 values per packed byte

    out, group_scales, tensor_scale, scratch_scales, scratch_amax = _alloc_outputs(
        rows, cols, x.device, out, group_scales, tensor_scale, scratch_scales, scratch_amax, scratch_div=16)

    _dequant_tp_had_quant(
        out, group_scales.view(torch.uint8), tensor_scale, scratch_scales, scratch_amax,
        h, x, x_group_scales.view(torch.uint8), x_tensor_scale, seed,
        fp4_max=6.0 / scale_override, fp8_max=256.0,
    )
    return NVFP4Quant(out, group_scales.reshape(rows, cols // 16), tensor_scale)


def rht128_requant(
        *,
        out: torch.Tensor = None,
        group_scales: torch.Tensor = None,
        tensor_scale: torch.Tensor = None,
        scratch_scales: torch.Tensor = None,
        scratch_amax: torch.Tensor = None,
        h: torch.Tensor,
        x: torch.Tensor,
        x_group_scales: torch.Tensor,
        x_tensor_scale: torch.Tensor,
        seed: torch.Tensor = None,
        scale_override: float = 1.0,
):
    """Transform an already-NVFP4 `x` as `x^t h^t` and requantize, with EDEN rounding.

    This is what lets the backward reuse the forward's saved fp4 operands: the dequantize,
    transpose, rotation and requantize all happen in one pass, without ever materializing bf16.

    :param h: the top 16x128 slice of a 128x128 Hadamard (see :func:`rht128_quant_eden`)
    :return: :class:`NVFP4Quant`
    """
    if seed is None:
        seed = new_seed()

    _check_requant_input(x, x_group_scales, x_tensor_scale, h)
    rows, cols = 2 * x.shape[1], x.shape[0]  # transposed, and 2 values per packed byte

    out, group_scales, tensor_scale, scratch_scales, scratch_amax = _alloc_outputs(
        rows, cols, x.device, out, group_scales, tensor_scale, scratch_scales, scratch_amax, scratch_div=16)

    _rht128_requant(
        out, group_scales.view(torch.uint8), tensor_scale, scratch_scales, scratch_amax,
        h, x, x_group_scales.view(torch.uint8), x_tensor_scale, seed,
        fp4_max=6.0 / scale_override, fp8_max=256.0,
    )
    return NVFP4Quant(out, group_scales.reshape(rows, cols // 16), tensor_scale)
