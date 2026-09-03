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

## Ported (branch `nvfp4-numerics`, 2026-09-02)

TE's source is now a sibling checkout (`/remote/projects/pycharm/TransformerEngine`, reference
only). Its recipe, read from `common/recipe/__init__.py`: forward input = RHT columnwise only, no
SR, 1D; weight = 2D 16x16, no RHT, no SR; gradient = RHT + SR, 1D; and both orientations are
quantized "from the high precision input to avoid double quantization errors".

Four levers, all Python, all opt-in behind `--nvfp4`, in `nanochat/sm120/nvfp4_numerics.py`:

| flag                                   | TE feature                                     | mechanism here                                                                   |
|----------------------------------------|------------------------------------------------|----------------------------------------------------------------------------------|
| `--nvfp4-exclude lm_head`              | output head in high precision                  | conversion filter; suffix match on the fqn                                       |
| `--nvfp4-bf16-blocks FIRST,LAST`       | first 2 / last 4-8 blocks in BF16              | same filter on `transformer.h.{i}.`                                              |
| `--nvfp4-exclude-precision fp8`        | (this repo's addition)                         | excluded layers become tensorwise `Float8Linear`                                 |
| `--nvfp4-exclude-precision fp8-fwd`    | (this repo's addition)                         | excluded layers keep the NVFP4 backward (bf16 source) under an fp8 forward (`nvfp4_fp8fwd.py`) |
| `--nvfp4-bwd-source bf16`              | columnwise operands from the bf16 source       | `rht128_quant_eden(x, transpose=True)` instead of `rht128_requant(x_fp4)`; saves x |
| `--nvfp4-rht wgrad` / `none`           | RHT on wgrad only                              | unrotated pairs via `quant_fp4(EDEN)`                                            |
| `--nvfp4-weight-2d`                    | 2D weight scaling, same W_q in fprop and dgrad | torch quantizer in the weight-cache refresh, emits row + transposed forms        |

Not ported: value-level stochastic rounding (`cvt.rs.satfinite.e2m1x4`, sm_100a only; TE has
a software emulation next to it in `common/util/ptx.cuh`) and the 16x16 Hadamard (ours is 128).

**One kernel family per GEMM.** `rht128_*` kernels rotate each 128-block by
`swizzle_hadamard(h)`, a row-permuted Hadamard; `quant_fp4` and the 2D cache are unrotated. Mixing
them is a silently wrong product (cosine ~0.1). Every variant's gradients are projected onto the bf16 gradient in
`tests/test_nvfp4_numerics.py` for that reason.

**Finding: EDEN's unbiasedness is the random rotation.** Values are rounded to nearest in every
Quartet kernel; only the block scale is stochastic. Drop the sign flips and averaging plateaus
at the rounding bias (pinned by `test_unrotated_eden_is_biased`). So `--nvfp4-rht wgrad` gives
a *biased* dgrad, and `none` biases both -- TE gets away with an unrotated dgrad only because it
rounds gradient values stochastically. A value-SR kernel is the follow-up that would make the
TE placement honest here.

Probe: `NANOCHAT_BASE_DIR=/remote/.nanochat-cache .venv/bin/python -m scripts.probe_nvfp4_numerics`
(one batch of the d12 C2 fp8 checkpoint, eager, minutes). Results below.

### Probe results, 2026-09-02 (d12 C2 fp8 checkpoint, step 2520, one 8192-token val batch)

Model view: loss on the batch against bf16 (forward noise), and the projection of the full
parameter gradient onto the bf16 gradient (`cos`; `bits` = -log2(rel-MSE)/2).

| variant           | fp4 layers | dloss    | cos all | bits  |
|-------------------|-----------:|---------:|--------:|------:|
| baseline          |         73 | +0.0768  |  0.840  |  0.04 |
| fwd-only          |         73 | +0.0768  |  0.889  |  0.18 |
| exclude-lm_head   |         72 | +0.0419  |  0.979  |  0.92 |
| lm_head-fp8       |         72 | +0.0422  |  0.933  |  0.85 |
| lm_head-fp8fwd    |         73 | +0.0422  |  0.884  |  0.63 |
| bf16-blocks-2+4   |         37 | +0.0475  |  0.882  |  0.34 |
| bwd-bf16          |         73 | +0.0768  |  0.851  |  0.04 |
| rht-wgrad         |         73 | +0.0768  |  0.862  |  0.04 |
| rht-none          |         73 | +0.0768  |  0.866  |  0.04 |
| weight-2d         |         73 | +0.1168  |  0.911  | -0.34 |
| lm_head+2d        |         72 | +0.0597  |  0.934  |  0.36 |
| lm_head+blocks    |         36 | +0.0162  |  1.011  |  1.71 |
| te-style-1d       |         36 | +0.0162  |  1.014  |  1.73 |
| te-style (+2d)    |         36 | +0.0260  |  1.014  |  1.56 |
| te-style-fp8      |         36 | +0.0147  |  0.960  |  1.52 |

- **lm_head is the noise.** Keeping it in bf16 removes 45% of the loss delta and takes the
  gradient from 0.84 to 0.98 cosine. It is one of 73 layers, and at d12 its fp4 GEMM was 0.97x
  fp8 anyway. Half the blocks in bf16 buys less than lm_head alone.
- **Taking lm_head out of fp4 costs 6% tok/s whether it lands in bf16 or fp8** (234k -> 220k at
  d12/2 GPU, Mark's shortruns; bpb 1.6815 -> 1.6787, inside 50-step noise). The lm_head shape,
  fwd+bwd per micro-step, compiled: all-fp4 15.6 ms, fp8 22.1, bf16 24.8. The fp8 *backward* is
  as slow as bf16's -- the [tokens, vocab] gradient is amax-scanned, cast and transposed -- and
  the backward is two thirds of lm_head's FLOPs. fp8 with NT wgrad + delayed scaling: 18.4.
- **`fp8-fwd` is the shape of the fix**: fp8 forward (the noise the probe cares about is the
  forward's) over the rotated bf16-source NVFP4 backward (unbiased, fused kernels): 17.9 ms, i.e.
  +2.3 ms/micro-step over all-fp4, ~1.6% of a d12 step. Same loss delta as fp8 (+0.0422); the
  whole-model gradient cosine sits between all-fp4 and fp8 (0.88) because lm_head's fp4 dgrad
  feeds the trunk -- unbiased noise, which is what a run averages out. tok/s and bpb unmeasured.
- **The backward levers do not move the model-level numbers.** As predicted by the forward-only
  arm's identical deficit: at this checkpoint the forward is the cost.
- **2D weights are a net loss here.** Forward error rises (256-element blocks), and although the
  gradient direction improves against a *1D* baseline (0.84 -> 0.91), inside the combined recipe
  it is worse than 1D (1.56 vs 1.73 bits). NVIDIA's consistency argument is about long runs; the
  probe cannot see that, so it stays available but not recommended.

Ladder view, `transformer.h.5.mlp.c_proj`, gradient averaged over 1/4/16/64 draws, bits against
the *bf16* product (an unbiased estimator gains +1 bit per 4x):

| variant   | dgrad@1 | dgrad@64 | wgrad@1 | wgrad@64 | bits per 4x   |
|-----------|--------:|---------:|--------:|---------:|---------------|
| baseline  |   2.74  |   3.62   |   3.10  |   4.73   | 0.29 / 0.54   |
| bwd-bf16  |   2.99  |   5.97   |   3.16  |   5.85   | **1.00 / 0.90** |
| rht-wgrad |   2.97  |   3.03   |   3.10  |   4.73   | 0.02 / 0.54   |
| rht-none  |   2.97  |   3.03   |   3.26  |   3.39   | 0.02 / 0.04   |
| weight-2d |   2.86  |   2.88   |   3.10  |   4.73   | 0.01 / 0.54   |

- **The default backward is biased against the true gradient.** It re-quantizes the *saved fp4*
  operands, so it is unbiased for the dequantized forward's gradient and plateaus at that
  operand's forward error against bf16. `--nvfp4-bwd-source bf16` is the only variant that
  climbs at the full rate on both GEMMs -- TE's "avoid double quantization" is exactly this.
  Cost: the bf16 activation is saved for backward (+2 B/elem).
- **Unrotated EDEN plateaus immediately** (rht-wgrad dgrad, rht-none both), see the finding above.
  lm_head's wgrad also plateaus under bwd-bf16 (0.28 per 4x): the softmax gradient is the
  heaviest tail in the model and the backward's 7% clipping (`BACKWARD_SCALE_OVERRIDE`) is a
  candidate cause; not chased.

**Candidates for a horizon run**, in order: `--nvfp4 --nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd`,
then `+ --nvfp4-bwd-source bf16` (unbiased backward, memory cost), then `+ --nvfp4-bf16-blocks 2,4`
(quality vs the fp4 speed it gives back). `--nvfp4-weight-2d` and `--nvfp4-rht wgrad/none` are
ablations, not recommendations, until a value-SR kernel exists. **bpb is unmeasured for all of
these.**

### Edge blocks on top of the fp8-fwd lm_head (probe, same batch)

| variant                       | dloss    | cos all |
|-------------------------------|---------:|--------:|
| lm_head fp8-fwd               | +0.0422  |  0.884  |
| + last block (`0,1`)          | +0.0291  |  0.919  |
| + last 2 blocks (`0,2`)       | +0.0261  |  0.916  |
| + first block (`1,0`)         | +0.0435  |  0.896  |
| + first 2, last 4 (`2,4`)     | +0.0147  |  0.922  |
| + `--nvfp4-bwd-source bf16`   | +0.0422  |  0.892  |

The last block carries most of what the trunk contributes; the first block nothing (NVIDIA's
first-2 rule does not show at d12). `--nvfp4-bf16-blocks 0,1` with `fp8-fwd` is 6 more layers
with an fp8 forward, ~2% of the model's forward FLOPs. Mark's shortruns: all-fp4 234k tok/s,
lm_head fp8-fwd 229k, bpb better than all-fp4 (50-step, inside noise).
