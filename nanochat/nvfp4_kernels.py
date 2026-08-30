"""NVFP4 quantization kernels for Quartet II style training on sm120+.

Quantizers are Triton ports of the Quartet II reference kernels
(https://github.com/IST-DASLab/Quartet-II, Apache-2.0):
  - forward: RTN with optional 4/6 group-scale search
    (kernels/python/quartet2/pseudoquant.py: rtn_1x16s_fp4_kernel)
  - backward: random-Hadamard transform + MS-EDEN with stochastically
    rounded E4M3 group scales
    (src/models/quantization/schemes/quartet_2.py: eden_1x16s_fp4_kernel),
    restructured like their fused CUDA kernels (rht128_eden.cu): codes are
    computed against the raw bf16 group scale inside the kernel (plus an
    atomic global amax), and the scale is converted to E4M3 with stochastic
    rounding afterwards in traceable torch ops.

Backward needs both operands of each GEMM quantized along the contraction
dim, so there are three EDEN variants:
  - _rht_eden_kernel:      bf16 (R, C), quantize along C (rows as-is)
  - _rht_eden_t_kernel:    bf16 (R, C), quantize x^T along R (fused transpose)
  - _rht_eden_rq_t_kernel: packed fp4 (R, C), dequantize + quantize x^T along R
    (Quartet II's rht128_requant equivalent: only 4-bit activations are saved)

Outputs are packed NVFP4: uint8 codes (2 x E2M1 per byte, low nibble first),
one float8_e4m3fn scale per 16 elements, one fp32 per-tensor scale. The GEMM
is torch._scaled_mm (cuBLASLt 1x16 block scaling, scales in the cuBLAS 128x4
blocked layout via to_blocked).

Divisions in the quantizers use IEEE div_rn so results match a pure-torch
reference bitwise (Triton's default fdiv is approximate).
"""

import torch
import triton
import triton.language as tl

GROUP = 16                 # NVFP4 scaling group
HAD_DIM = 128              # Hadamard rotation block (backward)
FWD_SCALE_OVERRIDE = 1.0
BWD_SCALE_OVERRIDE = (17 / 16) * 0.93

_RTN_BLOCK = 8192
_EDEN_ROWS = 64            # rows per program: (64, HAD) x (HAD, HAD) dot
_T_COLS = 64               # transposed variants: (HAD, 64) tiles

_FP4_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _rtn_idx(a):
    # |x| -> E2M1 magnitude index (values 0,.5,1,1.5,2,3,4,6)
    return tl.where(a >= 5, 7,
           tl.where(a >= 3.5, 6,
           tl.where(a >= 2.5, 5,
           tl.where(a >= 1.75, 4,
           tl.where(a >= 1.25, 3,
           tl.where(a >= 0.75, 2,
           tl.where(a >= 0.25, 1, 0)))))))


@triton.jit
def _idx_val(i):
    return tl.where(i == 7, 6.0, tl.where(i == 6, 4.0, tl.where(i == 5, 3.0, i * 0.5)))


@triton.jit
def _eden_math(rot, val_max: tl.constexpr, ROWS: tl.constexpr, HAD: tl.constexpr,
               G: tl.constexpr):
    """rot: (ROWS, HAD) fp32 rotated values -> (codes int32 (ROWS, HAD) w/ sign bit,
    corrected group scales fp32 (ROWS, HAD // G))."""
    xg = tl.reshape(rot, (ROWS * HAD // G, G))
    gmax = tl.max(tl.abs(xg), axis=1, keep_dims=True)
    s_b = tl.math.div_rn(gmax, val_max)
    xs = tl.math.div_rn(xg, tl.where(s_b == 0, 1.0, s_b))
    idx = _rtn_idx(tl.abs(xs))
    val = _idx_val(idx) * tl.where(xs < 0, -1.0, 1.0)
    # MS-EDEN correction per Hadamard block
    xs_h = tl.reshape(xs, (ROWS, HAD))
    val_h = tl.reshape(val, (ROWS, HAD))
    num = tl.sum(xs_h * xs_h, axis=1, keep_dims=True)
    den = tl.sum(xs_h * val_h, axis=1, keep_dims=True)
    corr = tl.where(den == 0, 1.0, tl.math.div_rn(num, den))
    s_corr = tl.reshape(s_b, (ROWS, HAD // G)) * corr
    code = tl.reshape(idx + 8 * (xs < 0), (ROWS, HAD))
    return code, s_corr


@triton.jit
def _pack(code, ROWS: tl.constexpr, HAD: tl.constexpr):
    # (ROWS, HAD) int32 codes -> (ROWS, HAD // 2) packed uint8, low nibble first
    lo, hi = tl.split(tl.reshape(code, (ROWS * HAD // 2, 2)))
    return tl.reshape((lo | (hi << 4)).to(tl.uint8), (ROWS, HAD // 2))


@triton.jit
def _rtn_quant_kernel(x_ptr, s_dec_ptr, codes_ptr, scales_ptr, n,
                      val_max: tl.constexpr, FOUR_SIX: tl.constexpr, BLOCK: tl.constexpr,
                      G: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    xg = tl.reshape(x, (BLOCK // G, G))
    s_dec = tl.load(s_dec_ptr)

    gmax = tl.max(tl.abs(xg), axis=1, keep_dims=True)
    s_b = tl.math.div_rn(gmax, val_max)

    # candidate 1: scale for max -> 6
    r6q = tl.math.div_rn(s_b, s_dec).to(tl.float8e4nv)
    r6 = r6q.to(tl.float32)
    r6 = tl.where(r6 == 0, 1.0, r6)
    xs6 = tl.math.div_rn(xg, r6 * s_dec)
    i6 = _rtn_idx(tl.abs(xs6))
    d6 = _idx_val(i6) * tl.where(xs6 < 0, -1.0, 1.0) * r6 * s_dec

    if FOUR_SIX:
        # candidate 2: scale for max -> 4; keep whichever has lower group MSE
        r4q = tl.math.div_rn(s_b * 1.5, s_dec).to(tl.float8e4nv)
        r4 = r4q.to(tl.float32)
        r4 = tl.where(r4 == 0, 1.0, r4)
        xs4 = tl.math.div_rn(xg, r4 * s_dec)
        i4 = _rtn_idx(tl.abs(xs4))
        d4 = _idx_val(i4) * tl.where(xs4 < 0, -1.0, 1.0) * r4 * s_dec
        use6 = tl.sum((xg - d6) * (xg - d6), axis=1, keep_dims=True) <= \
               tl.sum((xg - d4) * (xg - d4), axis=1, keep_dims=True)
        idx = tl.where(use6, i6, i4)
        sbits = tl.where(tl.reshape(use6, (BLOCK // G,)),
                         r6q.to(tl.uint8, bitcast=True).reshape(BLOCK // G),
                         r4q.to(tl.uint8, bitcast=True).reshape(BLOCK // G))
    else:
        idx = i6
        sbits = r6q.to(tl.uint8, bitcast=True).reshape(BLOCK // G)

    code = tl.reshape(idx, (BLOCK,)) + 8 * (x < 0)
    lo, hi = tl.split(tl.reshape(code, (BLOCK // 2, 2)))
    coffs = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(codes_ptr + coffs, (lo | (hi << 4)).to(tl.uint8), mask=coffs < n // 2)
    soffs = pid * (BLOCK // G) + tl.arange(0, BLOCK // G)
    tl.store(scales_ptr + soffs, sbits, mask=soffs < n // G)


@triton.jit
def _rht_eden_kernel(x_ptr, h_ptr, codes_ptr, scales_ptr, amax_ptr, n,
                     val_max: tl.constexpr, HAD: tl.constexpr, BLOCK: tl.constexpr,
                     G: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    h = tl.load(h_ptr + tl.arange(0, HAD * HAD)).reshape(HAD, HAD)
    rot = tl.dot(tl.reshape(x, (BLOCK // HAD, HAD)), h)  # fp32 accum
    tl.atomic_max(amax_ptr, tl.max(tl.abs(rot)), sem="relaxed")

    code, s_corr = _eden_math(rot, val_max, BLOCK // HAD, HAD, G)

    lo, hi = tl.split(tl.reshape(code, (BLOCK // 2, 2)))
    coffs = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(codes_ptr + coffs, (lo | (hi << 4)).to(tl.uint8), mask=coffs < n // 2)
    soffs = pid * (BLOCK // G) + tl.arange(0, BLOCK // G)
    tl.store(scales_ptr + soffs, tl.reshape(s_corr, (BLOCK // G,)), mask=soffs < n // G)


@triton.jit
def _store_t(code, s_corr, codes_ptr, scales_ptr, r0, c0, R,
             BC: tl.constexpr, HAD: tl.constexpr, G: tl.constexpr):
    # outputs are (C, R/2) codes and (C, R/16) scales; this program owns
    # out rows [c0, c0+BC) and out cols [r0/2, ...) / [r0/16, ...)
    packed = _pack(code, BC, HAD)
    orow = c0 + tl.arange(0, BC)
    tl.store(codes_ptr + orow[:, None] * (R // 2) + (r0 // 2 + tl.arange(0, HAD // 2))[None, :], packed)
    tl.store(scales_ptr + orow[:, None] * (R // G) + (r0 // G + tl.arange(0, HAD // G))[None, :], s_corr)


@triton.jit
def _rht_eden_t_kernel(x_ptr, h_ptr, codes_ptr, scales_ptr, amax_ptr, R, C,
                       val_max: tl.constexpr, HAD: tl.constexpr, BC: tl.constexpr,
                       G: tl.constexpr):
    # quantize x^T (C, R) with rotation+groups along R; x is (R, C) bf16 row-major
    c0 = tl.program_id(0) * BC
    r0 = tl.program_id(1) * HAD
    ptrs = x_ptr + (r0 + tl.arange(0, HAD))[:, None] * C + (c0 + tl.arange(0, BC))[None, :]
    tile = tl.load(ptrs)  # (HAD, BC)
    h = tl.load(h_ptr + tl.arange(0, HAD * HAD)).reshape(HAD, HAD)
    rot = tl.dot(tl.trans(tile), h)  # (BC, HAD) fp32; same dot shape as flat kernel
    tl.atomic_max(amax_ptr, tl.max(tl.abs(rot)), sem="relaxed")
    code, s_corr = _eden_math(rot, val_max, BC, HAD, G)
    _store_t(code, s_corr, codes_ptr, scales_ptr, r0, c0, R, BC, HAD, G)


@triton.jit
def _rht_eden_rq_t_kernel(codes_in, scales_in, s_dec_ptr, h_ptr,
                          codes_ptr, scales_ptr, amax_ptr, R, C,
                          val_max: tl.constexpr, HAD: tl.constexpr, BC: tl.constexpr,
                          G: tl.constexpr):
    # like _rht_eden_t_kernel, but input is packed fp4 (R, C/2)+(R, C/16) - dequant in-kernel
    c0 = tl.program_id(0) * BC
    r0 = tl.program_id(1) * HAD
    rows = (r0 + tl.arange(0, HAD))[:, None]
    b = tl.load(codes_in + rows * (C // 2) + (c0 // 2 + tl.arange(0, BC // 2))[None, :]).to(tl.int32)
    code_in = tl.interleave(b & 0xF, (b >> 4) & 0xF)  # (HAD, BC)
    val = _idx_val(code_in & 7) * tl.where(code_in >= 8, -1.0, 1.0)
    sc = tl.load(scales_in + rows * (C // G) + (c0 // G + tl.arange(0, BC // G))[None, :]).to(tl.float32)
    s_dec = tl.load(s_dec_ptr)
    x = tl.reshape(tl.reshape(val, (HAD, BC // G, G)) * sc[:, :, None], (HAD, BC)) * s_dec
    tile = x.to(tl.bfloat16)  # match torch fp4_dequant output dtype bitwise
    h = tl.load(h_ptr + tl.arange(0, HAD * HAD)).reshape(HAD, HAD)
    rot = tl.dot(tl.trans(tile), h)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(rot)), sem="relaxed")
    code, s_corr = _eden_math(rot, val_max, BC, HAD, G)
    _store_t(code, s_corr, codes_ptr, scales_ptr, r0, c0, R, BC, HAD, G)


# --- custom ops (opaque to Dynamo/Inductor; everything else stays traceable) ---

def _alloc_t(x, R, C):
    return (x.new_empty((C, R // 2), dtype=torch.uint8),
            x.new_empty((C, R // GROUP), dtype=torch.float32),
            x.new_zeros((), dtype=torch.float32))


@torch.library.custom_op("nanochat::nvfp4_rtn_quant", mutates_args=())
def _rtn_quant_op(x: torch.Tensor, s_dec: torch.Tensor, val_max: float,
                  four_over_six: bool) -> tuple[torch.Tensor, torch.Tensor]:
    R, C = x.shape
    codes = torch.empty((R, C // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((R, C // GROUP), dtype=torch.uint8, device=x.device)
    n = x.numel()
    _rtn_quant_kernel[(triton.cdiv(n, _RTN_BLOCK),)](
        x, s_dec, codes, scales, n, val_max, four_over_six, BLOCK=_RTN_BLOCK, G=GROUP,
        num_warps=8)
    return codes, scales


@_rtn_quant_op.register_fake
def _(x, s_dec, val_max, four_over_six):
    R, C = x.shape
    return (x.new_empty((R, C // 2), dtype=torch.uint8),
            x.new_empty((R, C // GROUP), dtype=torch.uint8))


@torch.library.custom_op("nanochat::nvfp4_rht_eden_quant", mutates_args=())
def _rht_eden_op(x: torch.Tensor, hmat: torch.Tensor,
                 val_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, C = x.shape
    codes = torch.empty((R, C // 2), dtype=torch.uint8, device=x.device)
    s_corr = torch.empty((R, C // GROUP), dtype=torch.float32, device=x.device)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    n, BLOCK = x.numel(), _EDEN_ROWS * HAD_DIM
    _rht_eden_kernel[(triton.cdiv(n, BLOCK),)](
        x, hmat, codes, s_corr, amax, n, val_max, HAD_DIM, BLOCK=BLOCK, G=GROUP, num_warps=8)
    return codes, s_corr, amax


@_rht_eden_op.register_fake
def _(x, hmat, val_max):
    R, C = x.shape
    return (x.new_empty((R, C // 2), dtype=torch.uint8),
            x.new_empty((R, C // GROUP), dtype=torch.float32),
            x.new_empty((), dtype=torch.float32))


@torch.library.custom_op("nanochat::nvfp4_rht_eden_quant_t", mutates_args=())
def _rht_eden_t_op(x: torch.Tensor, hmat: torch.Tensor,
                   val_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, C = x.shape
    assert R % HAD_DIM == 0 and C % _T_COLS == 0, (R, C)
    codes, s_corr, amax = _alloc_t(x, R, C)
    _rht_eden_t_kernel[(C // _T_COLS, R // HAD_DIM)](
        x, hmat, codes, s_corr, amax, R, C, val_max, HAD_DIM, BC=_T_COLS, G=GROUP,
        num_warps=8)
    return codes, s_corr, amax


@_rht_eden_t_op.register_fake
def _(x, hmat, val_max):
    return _alloc_t(x, *x.shape)


@torch.library.custom_op("nanochat::nvfp4_rht_eden_requant_t", mutates_args=())
def _rht_eden_rq_t_op(codes_in: torch.Tensor, scales_in: torch.Tensor,
                      s_dec_in: torch.Tensor, hmat: torch.Tensor,
                      val_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, C = codes_in.shape[0], codes_in.shape[1] * 2
    assert R % HAD_DIM == 0 and C % _T_COLS == 0, (R, C)
    codes, s_corr, amax = _alloc_t(codes_in, R, C)
    _rht_eden_rq_t_kernel[(C // _T_COLS, R // HAD_DIM)](
        codes_in, scales_in, s_dec_in, hmat, codes, s_corr, amax, R, C,
        val_max, HAD_DIM, BC=_T_COLS, G=GROUP, num_warps=8)
    return codes, s_corr, amax


@_rht_eden_rq_t_op.register_fake
def _(codes_in, scales_in, s_dec_in, hmat, val_max):
    return _alloc_t(codes_in, codes_in.shape[0], codes_in.shape[1] * 2)


# --- traceable composition ---

def _stochastic_round_e4m3(ratio, rand):
    """Round positive fp32 ratios to E4M3, stochastically between neighbours (unbiased)."""
    ratio = ratio.clamp(max=448.0)
    q = ratio.to(torch.float8_e4m3fn)
    bits = q.view(torch.uint8)
    curr = q.to(torch.float32)
    prev = (bits - 1).view(torch.float8_e4m3fn).to(torch.float32)
    nxt = (bits + 1).view(torch.float8_e4m3fn).to(torch.float32)
    up = torch.where(curr > ratio, curr, nxt)
    down = torch.where(curr > ratio, prev, curr)
    span = up - down
    prob = torch.where(span > 0, (ratio - down) / span, torch.zeros_like(ratio))
    out = torch.where(rand < prob, up, down)
    out = torch.where(ratio <= 0, torch.ones_like(out), out)  # all-zero group -> 1.0
    return out.to(torch.float8_e4m3fn)


def _finish_eden(codes, s_corr, amax, val_max):
    s_dec = torch.where(amax == 0, torch.ones_like(amax), amax / (255.99 * val_max))
    scales = _stochastic_round_e4m3(s_corr / s_dec, torch.rand_like(s_corr))
    return codes, scales, s_dec


def fp4_forward_quant(x, four_over_six=True, scale_override=FWD_SCALE_OVERRIDE):
    """RTN (optionally 4/6) NVFP4 quantization of a 2D bf16 tensor, groups along dim 1."""
    val_max = 6.0 / scale_override
    scales_max = 256.0 if four_over_six else 448.0
    amax = x.abs().amax().to(torch.float32)
    s_dec = torch.where(amax == 0, torch.ones_like(amax), amax / (scales_max * val_max))
    codes, sbits = _rtn_quant_op(x, s_dec, val_max, four_over_six)
    return codes, sbits.view(torch.float8_e4m3fn), s_dec


def fp4_rht_eden_quant(x, hmat, scale_override=BWD_SCALE_OVERRIDE):
    """Hadamard-rotate x (per 128 cols, x @ hmat blockwise) and MS-EDEN quantize."""
    val_max = 6.0 / scale_override
    return _finish_eden(*_rht_eden_op(x, hmat, val_max), val_max)


def fp4_rht_eden_quant_t(x, hmat, scale_override=BWD_SCALE_OVERRIDE):
    """Same as fp4_rht_eden_quant(x.t().contiguous(), hmat) with the transpose fused."""
    val_max = 6.0 / scale_override
    return _finish_eden(*_rht_eden_t_op(x, hmat, val_max), val_max)


def fp4_rht_eden_requant_t(codes, scales, s_dec, hmat, scale_override=BWD_SCALE_OVERRIDE):
    """fp4_rht_eden_quant_t(fp4_dequant(codes, scales, s_dec), hmat) with dequant fused."""
    val_max = 6.0 / scale_override
    return _finish_eden(*_rht_eden_rq_t_op(codes, scales, s_dec, hmat, val_max), val_max)


def fp4_dequant(codes, scales, s_dec):
    """Unpack NVFP4 back to bf16 (no-bwd-quant path and tests)."""
    lut = torch.tensor(_FP4_VALS + [-v for v in _FP4_VALS], dtype=torch.float32,
                       device=codes.device)
    idx = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(-2).long()
    vals = lut[idx]
    R, C = vals.shape
    out = vals.view(R, C // GROUP, GROUP) * scales.to(torch.float32).unsqueeze(-1)
    return (out.view(R, C) * s_dec).to(torch.bfloat16)


def to_blocked(scales):
    """(R, G) E4M3 group scales -> flat cuBLAS 128x4 blocked layout (pads R to 128, G to 4)."""
    R, G = scales.shape
    nr, nc = (R + 127) // 128, (G + 3) // 4
    if (R, G) != (nr * 128, nc * 4):
        padded = scales.new_zeros((nr * 128, nc * 4))
        padded[:R, :G] = scales
        scales = padded
    blocks = scales.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def nvfp4_mm(a_codes, a_scales, a_s, b_codes, b_scales, b_s):
    """(M, C) x (L, C) -> (M, L) bf16 via cuBLASLt NVFP4; operands packed+quantized along C."""
    out = torch._scaled_mm(
        a_codes.view(torch.float4_e2m1fn_x2),
        b_codes.view(torch.float4_e2m1fn_x2).t(),
        to_blocked(a_scales),
        to_blocked(b_scales),
        out_dtype=torch.bfloat16,
    )
    # global scales applied here; fuses into the consumer's pointwise kernel under Inductor
    return (out.to(torch.float32) * (a_s * b_s)).to(torch.bfloat16)


def hadamard_matrix(n, device, dtype=torch.bfloat16):
    """Normalized Sylvester Hadamard matrix (n a power of two)."""
    assert n & (n - 1) == 0
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * n ** -0.5).to(dtype)
