# What NVFP4 kernels does Transformer Engine use?

Researched 2026-09-02. Sources at the bottom.

## Answer

TE does not write its own FP4 GEMM — it calls **cuBLASLt**. What TE owns is everything
wrapped around the matmul.

**The GEMM:** cuBLASLt block-scaled matmul, TN layout only. TE prepares the operands,
including *swizzling* the block scale factors into the layout cuBLASLt requires (same as
MXFP8). The per-tensor FP32 correction is applied after the cuBLAS call.

**TE's own CUDA kernels:**

| piece               | detail                                                                                               |
|---------------------|------------------------------------------------------------------------------------------------------|
| cast/quantize       | BF16 -> E2M1 (1 sign, 2 exp, 1 mantissa, +/-6 range)                                                 |
| scaling             | per-16-element E4M3 block scale under a per-tensor FP32 global scale                                 |
| 2D weight scaling   | 16x16 blocks for weights; `disable_2d_quantization=True` to opt out                                  |
| RHT                 | 16x16 Hadamard on columnwise quantization of inputs and grads, **wgrad GEMM only**, BF16 inputs only |
| stochastic rounding | **gradients only**, Blackwell native SR instructions; `disable_stochastic_rounding=True` to opt out  |
| scale swizzle       | layout transform required before the GEMM                                                            |

Recipe entry point: `transformer_engine.common.recipe.NVFP4BlockScaling()`.

## Why this repo cannot use TE

NVFP4 *training* in TE requires **SM 10.0 / 10.3** (B200/GB200-class). Inference is SM 10.0+.
**sm_120 is on neither list**, so TE's NVFP4 training path does not run on RTX PRO Blackwell.
Hence this fork's `torch._scaled_mm` + custom cuBLASLt launcher route.

## Against what this fork built

- `--nvfp4-lt-gemm` puts us on the same cuBLASLt backend TE uses.
- `--nvfp4-fuse-wgrad` is our version of TE's `fuse_wgrad_accumulation` (nvfp4-quartet.md:326).
- `--nvfp4-scaling delayed` mirrors TE's amax history (nvfp4-quartet.md:306).
- We use a **128x128 Hadamard** (`nanochat/sm120/nvfp4.py:226`) where TE uses 16x16, and apply
  it more broadly than TE's wgrad-columnwise-only placement.
- **TE has two things we do not:** stochastic rounding on gradients, and 2D 16x16 weight
  scaling. Both are convergence aids, and both target the deficit measured at d24 — worth
  considering if we revisit the numerics rather than the token ratio.

## Sources

- [TE NVFP4 feature docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- [TE FP8/FP4 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html)
- [Row-scaled NVFP4 fprop PR #2931](https://github.com/NVIDIA/TransformerEngine/pull/2931)
- [TE releases](https://github.com/NVIDIA/TransformerEngine/releases)
