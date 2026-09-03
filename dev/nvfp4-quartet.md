# NVFP4 training on sm_120, via Quartet-II

Integration of [Quartet-II](https://github.com/IST-DASLab/Quartet-II) (arXiv:2601.22813, ICML

2026) — NVFP4 for **both** the forward and the backward, where this fork's own fp4 attempt was
      forward-only and shelved.

```bash
.venv/bin/python -m scripts.base_train --depth 12 --nvfp4 ...
```

`--nvfp4` is mutually exclusive with `--fp8`; both replace the Linear layers.

**`--nvfp4` turns on the whole measured stack** (2026-08-18). The four flags below all default
*on* under it and each takes a `--no-` form to put back — `--no-nvfp4-fuse-wgrad` and so on —
which is how A/B arms are built and how a box whose cuBLASLt extension will not build still runs
the NVFP4 path. Passing any of them without `--nvfp4` is an error rather than a silent no-op, and
the startup log prints a `disabled:` line whenever a piece is opted out, so an arm's log always
records what it actually ran.

| flag                     | default               | what                                                                                                                                                                                                                           |
|--------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--nvfp4`                | off                   | NVFP4 forward and backward, plus the four below                                                                                                                                                                                |
| `--nvfp4-weight-cache`   | **on with `--nvfp4`** | quantize weights once per optimizer step. **+1.8-2.0%, +437 MiB — worth it**                                                                                                                                                   |
| `--nvfp4-lt-gemm`        | **on with `--nvfp4`** | drive the fp4 GEMMs through cuBLASLt directly instead of `_scaled_mm`. **~0% on its own** — it exists as the seam the two epilogue items need (queue A1)                                                                       |
| `--nvfp4-epilogue-alpha` | **on with `--nvfp4`** | apply the per-tensor scale inside the GEMM epilogue. **+0.35%**; also one bf16 rounding fewer, which is 0.1% of the GEMM's error and therefore not a reason on its own. Follows `--nvfp4-lt-gemm` (queue A2)                   |
| `--nvfp4-fuse-wgrad`     | **on with `--nvfp4`** | accumulate the weight gradient inside the wgrad epilogue (`beta=1` into an fp32 buffer) instead of by a separate cast-and-add. **+4.07% and −166 MiB — the largest item on this branch.** Follows `--nvfp4-lt-gemm` (queue A3) |
| `--nvfp4-rne`            | off                   | plain round-to-nearest forward instead of 4/6. **Deliberately not in the stack**: it takes accuracy away, so it is an ablation, not a recommendation                                                                           |

`--no-nvfp4-lt-gemm` takes the two epilogue items with it, since neither is expressible through
`_scaled_mm`; asking for one of them *and* dropping the launcher is a contradiction and raises.
The extension is built at startup rather than on the first GEMM, so a toolchain that cannot build
it fails before training with `--no-nvfp4-lt-gemm` named in the message.

**Every arm recorded below predates that change and cannot be reproduced by copying its flags.**
An entry that labels an arm `--nvfp4-lt-gemm` meant *that flag and nothing else* on top of
`--nvfp4`; today the same command line also brings the epilogue alpha and the fused wgrad. To
re-run one, spell out the `--no-` forms for everything the arm did not have — e.g. A3's reference
arm is now `--nvfp4 --no-nvfp4-fuse-wgrad`.

## Result

**+29.4% over `--fp8`, +59.8% over bf16**, at d20 / device-batch 4 / grad-accum 16 / 1 GPU,
`SSSL`, 100 steps, tok/s at step 99 — perf-log.md's batched-arm protocol, reference repeated.
Measured 2026-08-18 as one batch (queue D1 + D3), logs in `dev-ignore/d1d3-arms/`:

| arm                                                  | tok/s @ step 99 | MFU        | peak MiB |
|------------------------------------------------------|-----------------|------------|----------|
| `--fp8`                                              | 29,040          | 37.84%     | 16,136   |
| **`--nvfp4`**                                        | **37,532**      | **48.91%** | 16,811   |
| `--nvfp4 --no-nvfp4-weight-cache --no-nvfp4-lt-gemm` | 35,073          | 45.70%     | 16,796   |
| bf16                                                 | 23,493          | 30.61%     | 17,629   |
| `--fp8` (repeat)                                     | 28,949          | 37.72%     | 16,136   |

The reference repeat reproduces to **0.31%**, so every delta here clears the 2% bar. This is the
first table in this file measured *after* the windowed-flash port; all five arms log
`✓ windowed flash attention`, bf16 included — worth checking, because an arm that took the SDPA
mask branch instead would be measuring the mask rather than the precision.

Three numbers fall out that this file did not have:

- **The stack is +7.01%** as a single arm (37,532 vs 35,073) — against ~+6.3% from chaining its
  marginals, so for once they under-counted. Converges with the unbatched 2-step pair that read
  +7.1%. Queue D3.
- **Plain NVFP4, stack fully off, is already +21.0% over `--fp8`.** Most of the win is the fp4
  GEMMs themselves; the stack is real but secondary.
- **bf16 is the heaviest arm**, 818 MiB above `--nvfp4`. Neither quantization is a memory
  optimization — both keep fp32 masters — but at this shape the activation saving is not nothing.

~~+7.0% over `--fp8`, ~+29% over bf16~~ was **d12 / dbs 4 / 8 steps** (57,710 → 61,887 → 57,913,
repeat 0.35%), taken before the windowed-flash port when `SSSL` put 3 of 4 layers on a
materialized SDPA mask. That is a *different comparison*, not a superseded one — **the d12 row is
still unmeasured on the current tree.** What the table above replaces is the d20 figure of 1.25x
from *Where the time goes* — and that one was **already post-window-fix**, its nvfp4 arm being
`--nvfp4 --nvfp4-weight-cache` before A1-A3 landed. 1.25x x the stack's +4.4% predicts 1.305x
against 1.294x measured, so the move is the epilogue items, not the window.

For a kernel-level decomposition of the NVFP4 step — which is where the remaining work is ranked
from — see *Where the time goes* below.

**This is a throughput result only** — for the quality side see *Numerics: C1* and *C2*, which
measure a **+0.0137 bpb deficit at ratio 12** that this speedup roughly, but not quite, pays for.
Val bpb at step 100 reads fp8 1.648856, nvfp4 1.664012,
bf16 1.659533 — nvfp4 **+0.0151 against fp8**. That is *below the resolution floor* (one seed,
100 steps; perf-log.md experiment 18 puts it at ~0.03–0.04), and the arms confirm it themselves:
fp8 comes out **0.0107 better than bf16**, which cannot be real. So it is not evidence. But it is
the same sign and magnitude as the +0.0147 that killed `--fp4-fwd` at ratio 12, and the two NVFP4
arms agree to 0.0002 — **enter C1 expecting to find something.** The quality question is open, and
it is the one that decided against the previous fp4 arm.

## Where the speed comes from, and where it does not

Amortized GPU-only timings, M=16,384 tokens, the three GEMMs of a layer plus all quantization:

| shape           | bf16 GEMMs | fp4 total | speedup | + `--nvfp4-weight-cache` |
|-----------------|------------|-----------|---------|--------------------------|
| 768→768 (d12)   | 0.713 ms   | 1.020     | 0.70x   | 0.76x                    |
| 768→3072        | 2.945      | 3.119     | 0.94x   | 0.97x                    |
| 3072→768        | 2.728      | 2.206     | 1.24x   | 1.29x                    |
| 1280→1280 (d20) | 2.157      | 1.840     | 1.17x   | 1.23x                    |
| 1280→5120       | 7.961      | 5.828     | 1.37x   | 1.39x                    |
| 5120→1280       | 7.416      | 4.223     | 1.76x   | 1.79x                    |
| 1536→1536 (d24) | 2.721      | 2.265     | 1.20x   | 1.25x                    |
| 1536→6144       | 11.774     | 7.491     | 1.57x   | 1.59x                    |
| 6144→1536       | 12.139     | 5.459     | 2.22x   | 2.26x                    |

The last column is the forward `quant_fp4(weight)` amortized over grad-accum 8 — worth **2-8%**
per shape, and **+1.8-2.0% end to end** (measured twice, see *The weight cache* below). An earlier version
of this table claimed up to +21% by also subtracting the backward's `rht128_requant(w)`. That is
**not cacheable**: it consumes a Hadamard that `rerotate_hadamard` re-randomizes on every
backward, which is the whole point of the estimator.

The tensor cores do their part — the fp4 GEMMs alone run 1.55–2.80x the bf16 ones. What eats it
is quantization, which at d12's 768-wide shapes costs about as much as the GEMM it feeds
(0.487 ms of GEMM against 0.543 ms of cast). **The margin grows with model dim**: roughly
break-even on block shapes at d16, a clear win from d20 up. d12 comes out ahead anyway because
the whole step is not GEMM, and because nanochat's bf16 `Linear` pays an fp32→bf16 weight cast
per call that the quantized paths replace.

`lm_head` at its real widths (this tokenizer is 32,768), M=16,384:

| shape            | bf16 GEMMs | fp4 GEMMs | quant | total | speedup   |
|------------------|------------|-----------|-------|-------|-----------|
| d12 `768→32768`  | 30.71 ms   | 23.63     | 8.09  | 31.72 | 0.97x     |
| d20 `1280→32768` | 46.36      | 27.97     | 8.57  | 36.55 | **1.27x** |
| d24 `1536→32768` | 55.33      | 30.13     | 8.80  | 38.93 | **1.42x** |
| ~~`768→65536`~~  | 50.93      | 48.37     | 18.11 | 66.48 | 0.77x     |

The last row is **not a shape in this model** — an earlier pass assumed a 65,536 vocab, and the
"lm_head is the weak shape, at 0.76x" claim came from it. It reproduces exactly at that width, so
the measurement was right and the shape was wrong. At the true width lm_head is neutral at d12
and a clear win from d20 up, so **it belongs in the fp4 path**.

The mechanism: `rht128_quant_eden` on grad_output costs **2.62 ms, and 4.93 ms transposed,
independent of K** — it scales with M x N, which vocab fixes. That is ~7.5 ms of the ~8-9 ms
quantization total, so as K grows the GEMM saving grows (1.30x -> 1.66x -> 1.84x) while
quantization stays flat. fp4 at lm_head improves monotonically with depth.

## Where the time goes (d20 / dbs 4, 1 GPU, `SSSL`, grad-accum 16 — captured 2026-08-18)

Three matched captures on the current tree, `scripts/profile_train.py --steps 3 --skip-ncu`, one
GPU, same `.venv` (cu128), arms run back to back from a common 45 C start. This is the first
kernel-level map of the NVFP4 arm; every earlier throughput claim in this file is a wall-clock
A/B with no decomposition behind it.

| arm                                | GPU ms/step | distinct kernels |   vs bf16 |    vs fp8 |
|------------------------------------|------------:|-----------------:|----------:|----------:|
| bf16                               |     5,658.9 |              176 |     1.00x |         — |
| `--fp8`                            |     4,566.8 |              246 |     1.24x |     1.00x |
| **`--nvfp4 --nvfp4-weight-cache`** | **3,639.8** |              190 | **1.55x** | **1.25x** |

The nvfp4 total reconciles with wall clock to **0.2%** (3,639.8 ms captured against 3,631 ms/step
measured in the windowing A/B), which is the check that the capture is measuring the real step.

### The nvfp4 step, by group

| group                                            | ms/step |    % | n/step |
|:-------------------------------------------------|--------:|-----:|-------:|
| GEMM fp4 (block-scaled)                          |   990.1 | 27.2 |  5,808 |
| pointwise (elementwise, norm, cast)              |   794.0 | 21.8 | 19,611 |
| reduction (amax / softmax / CE)                  |   584.3 | 16.1 |  5,153 |
| attention (flash fwd+bwd)                        |   501.3 | 13.8 |  1,280 |
| **fp4 rotate + quantize, bwd** (`rht128_eden`)   |   314.3 |  8.6 |  3,872 |
| optimizer / Muon                                 |   186.8 |  5.1 |     67 |
| **fp4 quantize, forward** (`four_six`)           |   114.8 |  3.2 |  2,057 |
| **fp4 requantize, bwd** (`rht128_requant`)       |   104.2 |  2.9 |  3,872 |
| **fp4 scale conversion** (`eden_convert_scales`) |    35.3 |  1.0 |  7,744 |
| memcpy / memset                                  |    11.4 |  0.3 |      8 |
| GEMM bf16 (unconverted layers)                   |     2.4 |  0.1 |    719 |
| other / unclassified                             |     0.9 |  0.0 |    137 |

Three rows are identical across all three arms to within 0.6% — attention (501.3 / 503.9 / 500.4),
Muon (186.8 / 187.2 / 187.5) and memcpy (11.4) — which is the cross-check that the grouping is
not moving work between rows. The 2.4 ms bf16 GEMM row is `ve_gate` and `smear_gate`, the 11 of
132 layers that are not 128-aligned: negligible, as this file has always claimed.

### Where the 927 ms against fp8 actually comes from

The step-level delta decomposes exactly (fp8 4,566.8 − nvfp4 3,639.8 = **927.0 ms**):

|                                                  |                              ms |
|:-------------------------------------------------|--------------------------------:|
| GEMM: 1,985.8 → 992.5                            |                      **−993.3** |
| explicit fp4 quantize rows (none in the fp8 arm) |                      **+568.6** |
| reduction row: 781.7 → 584.3                     |                          −197.4 |
| pointwise row: 1,096.9 → 794.0                   |                          −302.9 |
| attention + Muon + memcpy                        |                              ~0 |
| **net**                                          | **−925.0** (vs −927.0 measured) |

Two things follow, and the second is the surprising one.

**The GEMMs deliver.** 1,985.8 → 992.5 ms is **2.00x over fp8** and 3.67x over bf16, and the whole
fp4 GEMM inventory is a *single* kernel — `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_
ue4m3xe2m1_..._128x128x256`, 5,808 launches at a 89.8 us median. That is the arithmetic win NVFP4
is for, and at d20 it arrives in full.

**Quantization costs 568.6 ms (15.6% of step), i.e. 57% of the GEMM time it feeds** — better than
the ~100% this file records at d12's 768-wide shapes, and consistent with its claim that the
margin grows with model dim. Of that, **453.8 ms (12.5%) is the backward**: `rht128_eden` on the
two grad_output operands is 314.3 ms, `rht128_requant` on the two saved forward operands 104.2,
and the scale-conversion pass 35.3.

But fp4 *also* spends **500 ms/step less** than fp8 on the pointwise and reduction rows, because
fp8's cast/clamp/amax fuses into its producers and inflates them, while fp4's quantize is an
opaque `custom_op` that leaves the producer writing plain bf16. So the two recipes pay for
quantization in different columns, and the honest comparison is the sum of everything that is
not GEMM, attention, Muon or memcpy: bf16 pays 1,322.4 ms, fp8 pays 1,878.6, and fp4 pays
1,946.9. **So both quantized recipes add ~560–625 ms of glue over bf16, and fp4's is ~68 ms/step
*more* expensive than fp8's — not less, despite moving half the bytes.** That is the headroom, and it is why "fuse the
cast into
its producer" is the item this map supports most strongly.

### The amax pre-pass is 148 ms/step

`quant_fp4` computes `torch.linalg.vector_norm(x, inf)` before every activation quantize, because
NVFP4's per-tensor scale must be known before the kernel runs. Inductor fuses that reduction into
the producer, so it is not a separate kernel — but it is measurable by name:

| arm   | kernels mentioning `linalg_vector_norm` | launches |
|-------|----------------------------------------:|---------:|
| nvfp4 |                           204.4 ms/step |    2,644 |
| fp8   |                                    56.0 |       52 |
| bf16  |                                    56.0 |       52 |

**~148 ms/step (4.1%) is the fp4 amax pre-pass.** Treat it as an upper bound on what a
TransformerEngine-style delayed-scaling recipe would save: those kernels compute other things
too, so removing the reduction makes them cheaper, not free. It also removes a
producer → amax → quantize serialization that the number does not capture.

### It is not launch-bound — 98.7% busy, at 1.7x the fp8 launch count

`scripts/gpu_idle_report.py` on the same captures. This needed re-answering rather than assuming:
`perf-log.md` experiment 5 rejected CUDA graphs at the fp8 launch count, and the fp4 arm issues
**58,573 GPU ops/step against fp8's 34,357**.

|                               |            nvfp4 |     fp8 |
|-------------------------------|-----------------:|--------:|
| GPU ops/step                  |           58,573 |  34,357 |
| GPU busy                      |       **98.73%** |  99.51% |
| sub-2 us gaps (launch-shaped) | 0.724% of window |  0.261% |
| median inter-kernel gap       |          0.35 us | 0.35 us |

**The verdict holds: the step is GPU-bound and any CUDA-graph or launch-batching win is capped at
0.72% of wall clock** (~26.7 ms/step), before graphs' own costs. The card is 0.8 points less busy
than under fp8, not launch-starved.

The host side is where the fp4 path is visibly untidy, without it yet mattering:

| API                     |   ms/step | calls/step |
|-------------------------|----------:|-----------:|
| `cudaStreamSynchronize` |     880.6 |         17 |
| `cuLaunchKernel`        |     394.9 |     22,777 |
| `cudaLaunchKernel`      |     310.4 |     13,967 |
| **`cudaMemset`**        | **296.7** |  **7,744** |

That `cudaMemset` row is the per-call `cudaMemset(max_scale, 0, 4)` in the eden and requant
launchers — 7,744 calls at 38 us of CPU each, 13% of all API time, zeroing four bytes. Net CPU
work (API total minus the synchronize the CPU spends waiting) is 1,387 ms/step against a 3,692 ms
step, so there is 2.7x headroom and none of this is on the critical path today — against fp8's
3.1x. Worth fixing on principle and because the margin shrinks as the GPU gets faster, but it is
**not** where the throughput is.

### The roofline: which fp4 kernel needs which fix

`profile_train.py` drilled the top 8 with ncu SpeedOfLight (8/8 returned data). Low on **both**
SM% and DRAM% means latency- or occupancy-bound, i.e. a fusion or launch-geometry problem; high
DRAM% means the only lever is moving fewer bytes.

| kernel                                           | % GPU | SM % |   DRAM % | limited by                      |
|:-------------------------------------------------|------:|-----:|---------:|:--------------------------------|
| `cutlass3x_...block_scaled` (fp4 GEMM)           |  27.2 | 66.3 |     19.9 | compute, with 34 points spare   |
| `four_six_fp4_kernel` (fwd quantize)             |   3.2 | 46.0 | **85.0** | **bandwidth, near roofline**    |
| `rht128_eden_tma_kernel<false>` (bwd)            |   5.2 | 27.4 | **76.5** | **bandwidth**                   |
| `rht128_eden_tma_kernel<true>` (bwd, transposed) |   3.5 | 26.2 | **77.7** | **bandwidth**                   |
| `rht128_requant_kernel` (bwd)                    |   2.9 | 38.8 |     32.7 | **neither — latency/occupancy** |
| `gemm_64x64_32x6` (Muon Newton-Schulz)           |   3.1 | 63.3 |     18.8 | compute                         |

The two `rht128_eden` variants are drilled **separately**: they share a name, so one ncu sample
cannot speak for both, and the report now suppresses the columns rather than guessing (see
*Gradient accumulation* below for what that guard was added to catch). Drilled per variant they
agree to ~1 point, so the reading holds — but it is measured, not assumed.

This settles the question the time ranking could not, and it moves two items:

**The forward quantize is already at the byte roofline (85% DRAM).** So there is *no* arithmetic
win available in it — which is the opposite of `perf-log.md` experiment 22/23, where the
fp8-sourced fp4 cast was ALU-bound on a per-element `div_rn` and a division-free rewrite bought
+3.9%. Do not go looking for that here; Quartet's 4/6 kernel is bandwidth-limited, so the only
lever is **fewer bytes**: fuse it into its producer so `x` is not re-read in bf16, and drop the
amax pre-pass so it is not read twice.

**`rht128_eden` is bandwidth-bound too (76.5 / 77.7% DRAM)**, and it has a specific removable cost: it
writes per-group scales to a bf16 scratch buffer, which `eden_convert_scales_kernel` then reads
back and rewrites as fp8 — a round trip that exists only because the per-tensor scale is not
known until the pass finishes. An amax *history* makes it known in advance, which folds those
7,744 launches and 35.3 ms away and shortens the main pass.

**`rht128_requant` is the one with a kernel-side fix** — 38.8% SM against 32.7% DRAM is idle on
both roofs, the signature of occupancy or launch geometry, at 26.4 us over 3,872 launches.

### What the map ranks

In measured order, largest first:

1. **Fuse the quantize into its producer.** The glue comparison says fp4's quantization is net
   ~68 ms worse than fp8's *because* it cannot fuse, and the roofline says both quantize kernels
   are bandwidth-bound — so bytes are exactly the currency. Needs delayed scaling first.
2. **Delayed scaling (TE's amax history), ≤148 ms (4.1%) directly** — and it is worth more than
   that indirectly: it is the prerequisite for (1), and it retires the eden scratch round-trip
   and its 7,744-launch conversion pass.
3. **The backward quantize, 453.8 ms (12.5%).** `rht128_eden` (314.3) wants fewer bytes;
   `rht128_requant` (104.2) wants occupancy. Different fixes, same row.
4. ~~**The fp4 GEMM's 34 points of SM headroom, on 990.1 ms.**~~ **Answered 2026-08-18: not by
   algorithm selection.** cuBLASLt has exactly one block-scaled fp4 kernel for these shapes, and
   picking the best candidate for every shape is **1.001x** — see *The GEMM headroom is not an
   algorithm choice* below. The headroom is real; nothing in cuBLAS reaches it.
5. **The pointwise row's launch count**: 9,049 (bf16) → 19,611 (fp4), **+10,562 launches/step**.
   The `out * alpha` half of this attribution is now **measured wrong** (see *The per-tensor scale
   in the GEMM epilogue*): those multiplies were fused into their consumers, not launched
   separately, and deleting them is worth ~27 ms of ALU rather than 5,808 kernels. What is left of
   the entry is `rerotate_hadamard`'s per-Linear-per-micro-step re-randomization (~3,872), which
   queue B3 addresses. Read this as the standing warning it now is: the arithmetic matching a
   launch count is not evidence about which kernel the work is in.

Not supported by this map: CUDA graphs (0.72% ceiling), anything aimed at the unconverted bf16
layers (2.4 ms), and any arithmetic rewrite of the forward cast (it is at the byte roofline).

### Gradient accumulation is 208.4 ms/step, and it is TE's `fuse_wgrad_accumulation` — **done**

Two `at::native::vectorized_elementwise_kernel` rows carry **208.4 ms/step (5.7%)**, identical in
all three arms:

| functor                     | ms/step | n/step |      med |
|:----------------------------|--------:|-------:|---------:|
| `CUDAFunctor_add<float>`    |   135.7 |  2,040 |  34.0 us |
| `CUDAFunctor_add<BFloat16>` |    72.7 |    165 | 440.6 us |

They are plain adds, and the launch counts identify them: `2,040 = 136 params x 15`, and
`165 = 11 x 15`, where 15 is `grad_accum - 1` — the micro-steps on which `AccumulateGrad` has an
existing `.grad` to add to rather than store into. So this is **gradient accumulation**, which is
precision-independent (masters and their grads are fp32 whatever the matmuls do) and is exactly
what TransformerEngine's `fuse_wgrad_accumulation` removes, by having the wgrad GEMM accumulate
into an fp32 `main_grad` in its epilogue (`beta=1`) instead of writing a fresh tensor for a
separate kernel to add. `perf-log.md` experiment 14 did this for fp8 and measured +0.7-1.9%.

The bf16 row is the embedding tables, whose gradients come from `embedding_dense_backward` rather
than a GEMM, so only the 135.7 ms fp32 row is addressable this way.

**Landed 2026-08-18 as `--nvfp4-fuse-wgrad`, +4.07%** — see *Fused wgrad accumulation* below. The
fp32 row goes to 0.4 ms, and the widening casts this entry did not count are worth another 34.8 ms
against the fp32 epilogue's +45.0 ms on the GEMM row.

**An earlier version of this section read these two rows at "SM 0.0%, DRAM 0.3%" and called them
latency-bound optimizer work. That was wrong, and the way it was wrong is worth keeping**:
`kernel_report.py`'s `short_name()` collapsed four different `vectorized_elementwise_kernel`
instantiations onto one handle, and the ncu join then pinned one arbitrary sample's roofline —
taken from a 0.0 ms `FillFunctor` — onto all four rows. A name collision in that join does not
lose data, it *invents* it. The tool now disambiguates on the functor and refuses to attribute a
roofline to any short name covering more than one raw kernel, printing what it suppressed.

## The GEMM headroom is not an algorithm choice — queue A1, 2026-08-18

The GEMM row is 990.1 ms/step (27.2%) at 66.3% SM and 19.9% DRAM, so *The roofline* left 34 points
of compute roofline unused and ranked "pick a better algorithm" fourth. **There is no better
algorithm.** `scripts/probe_fp4_gemm.py` autotunes every GEMM the model launches — the real
inventory, each Linear shape x its three GEMMs x 16 micro-steps, which reproduces the profile's
5,808 launches/step — against `_scaled_mm` on the same warm card:

|                                                        | GEMM row, ms/step |
|--------------------------------------------------------|------------------:|
| `torch._scaled_mm`                                     |             987.1 |
| this fork's launcher, 1 MiB workspace (torch's budget) |             987.7 |
| this fork's launcher, 32 MiB, best candidate per shape |         **985.8** |

**1.001x**, and end to end **0.0%**: 72,707 tok/s against 72,693 for the same recipe without the
flag, with the reference repeat at 72,663 — a 0.04% spread, so the null is tight rather than
merely unresolved. (d20/dbs 4, 2 GPUs, grad-accum 16, medians of steps 15-19.) The projection
also lands 0.3% from the profile's measured 990.1 ms, which is the check that the probe is
timing the real inventory.

The reason is that **cuBLASLt has one block-scaled fp4 kernel here**: every candidate at every
shape is `algoId 70, tile 128x128, stages 256xAUTO` — the `cutlass3x_sm120_bstensorop_*_128x128x256`
the profile counted. The list varies only in cluster shape (2x1x1 / 1x2x1) and whether split-K is
on. Split-K needs 6.55 MB of workspace, which is why torch's 1 MiB query never sees it; it wins
exactly one shape of nine, `1280x1280x8192` at 1.02x (72.1 → 70.5 us), worth 1.3 ms of a 3,640 ms
step. The 34 points are real and nothing in cuBLAS reaches them — that would take a different
kernel.

**What landed is the seam, and it is why A1 was worth doing anyway.** `--nvfp4-lt-gemm` routes the
three GEMMs through `nanochat/sm120/csrc/fp4_gemm.cpp`: one plan per shape, autotuned inside the
first real step and verified against `_scaled_mm` before it is trusted, falling back per shape if
it is not. At all nine shapes it verifies **bit-identical** (`err 0`), and the autotune costs 3.4 s
once. A2 (per-tensor scale as a device-pointer alpha) and A3 (beta=1 into an fp32 `main_grad`) are
epilogue changes `_scaled_mm` cannot express at all, so they need this and now have it — and
because the seam is measured neutral, whatever they move is theirs. They have since moved +0.35%
and **+4.07%**, so a 0.0% item was worth landing on its own terms.

Four things worth not rediscovering:

- **`_scaled_mm` *is* cuBLASLt for fp4, and its descriptor is readable.** `CUBLASLT_LOG_LEVEL=2`
  prints the whole recipe — the operand swap, `VEC16_UE4M3` scale modes, the 1 MiB workspace. The
  launcher was written from that trace rather than from the documentation.
- **The heuristic query needs the scale pointers.** With a `VEC16_UE4M3` scale mode set,
  `cublasLtMatmulAlgoGetHeuristic` returns `NOT_SUPPORTED` until `A_SCALE_POINTER`/`B_SCALE_POINTER`
  are on the descriptor, even though it never dereferences them.
- **Time the two arms alternately.** The first version of the probe timed `_scaled_mm` once, after
  each plan's autotune — always at the hottest point of that shape's soak. It read 1.15x on a shape
  that alternating A/B/A puts at 1.02x. The same bias is still present in the per-shape `vs_ref` the
  training log prints, which is why that column is indicative and the probe is the number.
- **Config attributes are not all 32-bit.** `CLUSTER_SHAPE_ID` and `INNER_SHAPE_ID` are `uint16_t`
  and cuBLASLt rejects a wrong-width read rather than truncating, so a naive getter reports every
  cluster shape as "not set" — which hid half the axis the candidate list actually varies on.
  (`refactor`'s `pinned_gemm.cu` has the same getter and the same blind spot.)

## The per-tensor scale in the GEMM epilogue — queue A2, 2026-08-18

`--nvfp4-epilogue-alpha` (needs `--nvfp4-lt-gemm`) hands cuBLASLt the per-tensor scale as a
device-pointer alpha, so `out * alpha` never runs. **+0.35%**, and a real if small numerics
improvement. Batched arms, d20/dbs 4, 2 GPUs, medians of steps 15-19:

| arm                        |      tok/s |
|----------------------------|-----------:|
| `--nvfp4-lt-gemm`          |     72,823 |
| `+ --nvfp4-epilogue-alpha` | **73,079** |
| `--nvfp4-lt-gemm` (repeat) |     72,773 |

A 0.07% reference spread, so +0.35% is five times the noise floor, and the alpha arm is ahead of
both reference arms at every one of the five step indices.

**The profile says the mechanism is not the one the queue assumed, and the difference matters.**
Matched captures (1 GPU, 3 steps, same config) put the step at 3,635.6 → 3,615.6 ms, **−20.0 ms**:

| group     |           ms/step |              n/step |
|:----------|------------------:|--------------------:|
| GEMM fp4  |     981.6 → 981.6 |       5,808 → 5,808 |
| pointwise | 795.6 → **784.3** | 19,611 → **24,939** |
| reduction | 583.8 → **573.8** |       5,153 → 5,153 |

The epilogue costs the GEMM nothing — that row does not move by 0.1 ms. But the pointwise row gets
*faster while gaining 5,328 launches*, which only makes sense once the per-kernel diff is read:
every kernel that changed is an existing fused kernel with `mul` dropped from its name, at
roughly the same cost. `triton_poi_fused__to_copy_mul_pow_relu_view_14` (320 launches, 77.27 ms)
becomes `triton_poi_fused__to_copy_pow_relu_view_14` (320, 77.24); the `linalg_vector_norm` variant
goes 59.20 → 58.09. **So `out * alpha` was already fused into its consumer nearly everywhere, as
`fp4_mm`'s comment always claimed** — it was never 5,808 separate passes over [M, N], and A2's
ceiling was never the ~300 ms that framing implies. It is ~27 ms of ALU spread across the
consumers, which is what actually came out.

What went *in* is `triton_poi_fused_mul_5`: **0 → 3,872 launches, 4.12 ms**, plus 1,776 more on
`triton_poi_fused_mul_7`. Those are the alpha products themselves — 121 Linears x 16 micro-steps
x (1 forward + 2 backward) = exactly 5,808 — which now have to be *materialized*, because
`a_ts * b_ts` feeds an opaque custom op and Inductor can no longer keep it inside a fusion. **6.9
ms/step of 0-d multiplies is the price of the 27, and removing it would roughly double A2** (queue
A2b): the quantize kernels already write these scales, so one of them writing the product costs a
store in a kernel that is already running.

**The numerics improve, and the improvement is worth nothing.** Both halves of that are measured,
and the second is the one that matters. The epilogue scales the fp32 accumulator and rounds to
bf16 **once** where the separate multiply rounds and then scales, so against the exact fp32
product of the dequantized operands it is **1.3-1.6x lower relative MSE** — real, structural, and
true at every shape. But put that term in the actual budget:

| rel MSE, against the fp32 product of the *unquantized* operands |                |
|:----------------------------------------------------------------|---------------:|
| fp4 quantization of the operands                                |        1.51e-2 |
| output rounding, two roundings (separate multiply)              |        ~1.1e-5 |
| output rounding, one (epilogue)                                 |        ~7.9e-6 |
| **improvement in total GEMM error**                             | **0.00-0.11%** |

Quantization dominates the rounding by ~1,500x, so A2 moves the total error by a tenth of a
percent. **Nothing this small can show up in bpb**, and it should not be argued as a reason to
adopt anything.

That also retires an observation this entry made in its first draft. The alpha arm's training loss
is lower at every step index by ~0.0075, where the two reference arms agree to 0.001, and it is
tempting to read that as the mechanism showing through. **It cannot be** — the mechanism is three
orders of magnitude too small to produce it. What it is is one trajectory of a computation that
changed, which is precisely the single-seed luck experiment 24 caught (the "+0.006" that turned out
to be seed 42 being the most fp4-pessimistic of eight). Do not bank it, in either direction.

The tolerance does follow from the rounding, though: a plan is verified against `_scaled_mm * alpha`
at one bf16 ulp rather than bit-equality, which is safe because alpha here is ~1e-6, so a dropped or
doubled alpha misses by six orders of magnitude rather than by rounding.

Two mechanical notes. `CUBLASLT_POINTER_MODE_DEVICE` puts **both** alpha and beta in device memory,
so a device-alpha plan needs a device beta as well (the launcher keeps a static `[1.0, 0.0]`). And
pointer mode, like the block-scale pointers, has to be on the descriptor *before* the heuristic
query — it is a capability the candidate list is filtered on. With it set the list is unchanged:
same `algoId 70`, same tile, same count of candidates.

## Fused wgrad accumulation — queue A3, 2026-08-18

`--nvfp4-fuse-wgrad` (needs `--nvfp4-lt-gemm`) gives the wgrad GEMM a `beta=1` epilogue with an
fp32 accumulator as both C and D, so each micro-step's weight gradient joins the running sum
inside the GEMM instead of being written bf16 for autograd to widen and add. **+4.07%** — the
largest single item measured on this branch, against the tightest reference this file has. Batched
arms, d20/dbs 4, 2 GPUs, grad-accum 16, medians of steps 15-19:

| arm                                               |      tok/s |       MFU |           peak |
|---------------------------------------------------|-----------:|----------:|---------------:|
| `--nvfp4-lt-gemm --nvfp4-epilogue-alpha`          |     73,209 |     47.70 |     19,587 MiB |
| `+ --nvfp4-fuse-wgrad`                            | **76,193** | **49.64** | **19,421 MiB** |
| `--nvfp4-lt-gemm --nvfp4-epilogue-alpha` (repeat) |     73,220 |     47.70 |     19,587 MiB |

The two reference arms agree to **0.015%**, so this is 270x the noise floor, and the fused arm's
*slowest* step (76,056) beats the references' fastest (73,289). It also **costs 166 MiB less**
rather than more. Neutral was the expectation — the flat fp32 buffer replaces the `.grad`
autograd would have allocated for the same 121 parameters — and the 166 MiB it came in under
that is not explained here; it is reproducible and small, so it is recorded rather than argued.

**The profile confirms the mechanism the queue asked it to confirm.** Matched captures (1 GPU,
3 steps, same config) put the step at 3,610.4 → 3,472.7 ms, **−137.7 ms**:

| group                                       |             ms/step |              n/step |
|:--------------------------------------------|--------------------:|--------------------:|
| GEMM fp4                                    | 979.1 → **1,024.1** |       5,808 → 5,808 |
| pointwise                                   |   783.9 → **613.9** | 24,939 → **21,189** |
| `CUDAFunctor_add<float>` (inside pointwise) |     135.6 → **0.4** |     2,040 → **225** |

The add row does not shrink, it **goes away**: the 225 launches left are `15 params x 15
micro-steps`, the fp32 parameters that are not NVFP4 Linears and still accumulate the old way.
The 1,936 wgrads per step (121 layers x 16) move from the `..._f32_bf16_bf16_...` cutlass variant
to `..._f32_f32_f32_...` at the same shapes, and the pointwise row loses 3,750 launches — the
1,815 adds plus 1,935 bf16→fp32 widening casts, one per wgrad.

**The estimate was right, and only half for the reason it gave.** The queue priced this at the
135.7 ms of the `CUDAFunctor_add<float>` row (3.7%) and got 140 ms (4.07%). But that row is only
the add; the widening casts it did not count are worth another 34.8 ms, and the fp32 epilogue's
own C-read and D-write cost the GEMM row **+45.0 ms**. Those two nearly cancel, which is luck in
composition rather than a good model — the reusable lesson is the one A2 already taught from the
other side: **price the whole op you are replacing, not the one kernel you can see in the
ranking.**

The numerics move the same way A2's did and matter as little. On the backward's own operands the
fused wgrad is **2.0x lower relative MSE** against the exact fp32 product of the dequantized
operands (5.2e-6 against 1.06e-5) — it rounds zero times where the unfused path rounds twice, at
the GEMM output and again after the per-tensor scale. That also makes the fused gradient
systematically ~0.1% *larger* in magnitude, which is magnitude the bf16 rounding was losing, and
is why the equivalence test bounds relative MSE rather than a projection ratio — a projection
reads a real accuracy gain as a failure. Training loss at step 19 was 7.6539 fused against
7.6497 / 7.6473 for the two references, i.e. outside their own 0.0024 spread. **Do not bank
that, in either direction**: it is one seed of a computation that changed, which is the trap A2's
entry documents and experiment 24 caught. The numerics claim here is the rel-MSE one and nothing
more.

Four things worth not rediscovering:

- **cuBLASLt does offer an fp32 D for the block-scaled fp4 kernels**, from the same `algoId 70 /
  tile 128x128 / stages 256xAUTO` family and with the same 6-7 candidates the bf16 plans get. No
  new axis, and no fallback needed anywhere. That was the one thing that could have killed this
  outright.
- **An accumulating plan has to carry the device alpha.** `beta=1` adds the *scaled* product, so
  a per-tensor scale applied afterwards would rescale the whole running sum. The launcher refuses
  to build a host-alpha accum plan rather than letting that be a runtime surprise, and it means
  `--nvfp4-fuse-wgrad` folds alpha into the wgrad whether or not `--nvfp4-epilogue-alpha` is set.
- **`save_for_backward` on the accumulator is safe here only because of the ordering.** The
  buffer is mutated by the backward that unpacked it and zeroed after `optimizer.step()`, so the
  version it was saved at is always the version at unpack and autograd never sees a modified
  saved tensor. That is the same constraint the weight cache already documents, and violating it
  — refreshing or zeroing between a forward and its backward — corrupts silently in both cases.
- **DDP needed nothing.** nanochat has no DDP wrapper; gradients sync inside `optimizer.step()`
  from `p.grad`, and `attach()` hands the optimizer the same fp32 dtype at the same shape it
  always read. `MuonAdamW`'s reduce/gather paths are untouched.

The bf16 half of the accumulation row is untouched: `CUDAFunctor_add<BFloat16>` (72.7 ms, 165
launches) is the embedding tables, whose gradients come from `embedding_dense_backward` and never
pass through a GEMM epilogue.

## Strong scaling: fp4 moves the same bytes, so its comms tax is larger — 2026-08-26

**fp4 does not move more data between GPUs than fp8. It moves the same bytes in a shorter step.**
NCCL volume is set by parameter count and gradient dtype: `MuonAdamW` reduce-scatters fp32 grads
and all-gathers fp32 params (`nanochat/optim.py:283-416`), and nothing in `sm120/nvfp4.py` or
`sm120/fp8_backend.py` touches a collective. `--nvfp4-fuse-wgrad` if anything pins grads to fp32.
So the *fraction* of the step spent in NCCL rises by whatever the arithmetic speedup is.

Five arms, d12 / dbs 8 / total-batch 524288 / 100 steps, evals off, cooldown to 40 C, tok/s at
step 99 (`dev-ignore/fp4-scaling/`). `ga` 32-on-1 vs 16-on-2 holds work per step identical, and
`tok_per_sec = total_batch_size / dt` (`base_train.py:683`) is global, so the ratio is the
strong-scaling factor directly:

| arm                      | start C | tok/s @99 |    MFU |   dt/step |
|--------------------------|--------:|----------:|-------:|----------:|
| `--nvfp4` 2 GPU          |      36 |   232,678 | 39.90% | 2253.3 ms |
| `--nvfp4` 2 GPU (repeat) |      40 |   231,885 | 39.77% | 2261.0 ms |
| `--fp8` 2 GPU            |      40 |   193,025 | 33.10% | 2716.2 ms |
| `--nvfp4` 1 GPU          |      39 |   118,180 | 40.53% | 4436.3 ms |
| `--fp8` 1 GPU            |      40 |    97,778 | 33.54% | 5362.0 ms |

|           | scaling 2g/1g | comms tax |
|-----------|--------------:|----------:|
| `--fp8`   |    **1.974x** | **1.29%** |
| `--nvfp4` |    **1.962x** | **1.89%** |

**The tax roughly doubles in relative terms and is still under 2%.** fp8's 1.29% scaled by fp4's
1.209x single-GPU speedup predicts **1.56%**; the two fp4 2-GPU arms bracket that at 1.56% (cold
arm) and 1.89% (repeat). Read the direction as confirmed and the magnitude as ~0.3 pts wide —
the repeat spread is 0.34%, the same size as the effect.

Three things fall out:

- **fp4 is +20.1% over fp8 here**, not the +29.4% recorded at d20/1 GPU. Narrower `n_embd` and
  the extra rank both cost; at 1 GPU the same pair is +20.9%, so the shape explains most of it.
- **fp4's per-GPU MFU is 40.5% against fp8's 33.5%** — the wider gap than tok/s suggests, because
  MFU is scored on the bf16 roof either way.
- **`--fp8` scales 1.974x here against `perf-log.md`'s 1.987x on `refactor`.** Consistent: this
  branch has none of that stack, so the step is longer and comms hides better, not worse.

Not a concern at this scale, and the lever if it ever were is bf16 reductions rather than
anything in the fp4 path. The byte problem fp4 actually has is DRAM, not PCIe — the quantize
kernels sit at 76-85% of the DRAM roofline (*Where the time goes*), which is queue B2.

## Depth, batch size and what fits

Measured with `--window-pattern L --nvfp4`, eval + sampling + checkpointing all on, 4 steps:

| depth   | dbs   | peak       | headroom    | tok/s      | MFU       |
|---------|-------|------------|-------------|------------|-----------|
| d12     | 8     | 10,427 MiB | 14.0 GiB    | 99,593     | 39.9%     |
| d20     | 2     | 13,105     | 11.4 GiB    | 28,612     | 41.9%     |
| **d20** | **4** | **16,796** | **7.7 GiB** | **31,011** | **45.4%** |
| d20     | 8     | OOM        | —           | —          | —         |
| d24     | 2     | 21,129     | 3.3 GiB     | 18,260     | 43.6%     |
| d24     | 4     | OOM        | —           | —          | —         |

**d24 does complete end to end at dbs 2** — full loop, validation descending, model and optimizer
checkpoints written. perf-log.md's "d24 does not fit here at any batch size" is too strong; what
is true is that it fits *only* at dbs 1–2, with 3.3 GiB spare.

**Peak memory is parameter and optimizer state, not activations.** At d24 it moves by 1.3 MiB
across every configuration tried — dbs 1 vs 2, `SSSL` vs `L`, eval/sampling/checkpointing off vs
on. d24 is 1.384B params, 44% of it `value_embeds` (604M). Neither `--fp8` nor `--nvfp4` touches
that: both keep fp32 masters and quantize on the fly, so at d24 all three arms land within 6 MiB
of each other (bf16 21,122.16 / fp8 21,122.16 / nvfp4 21,127.51). **NVFP4 is not a memory
optimization.** If d24 needs to fit at a useful batch size, the lever is `value_embeds` or the
optimizer state.

d20 is the box's sweet spot: 8 GiB cheaper than d24, a real batch size, and the best MFU measured.

## The weight cache

`--nvfp4-weight-cache`. The forward's `quant_fp4(weight)` runs every micro-step, but the weight
is an fp32 master that nothing in the grad-accumulation loop moves, so re-quantizing it per
micro-step is (accum-1)/accum redundant. `refresh_weight_caches()` does it once per optimizer
step instead.

d20 dbs=4, batched arms with the reference repeated (`scripts/arm_batch.sh`), run twice independently:

| run    | no cache | cache      | no cache (repeat) | repeat spread | delta      |
|--------|----------|------------|-------------------|---------------|------------|
| first  | 30,877   | **31,476** | 30,824            | 0.17%         | **+2.03%** |
| second | 30,973   | **31,464** | 30,860            | 0.37%         | **+1.77%** |

So **+1.8 to +2.0%**. Each run clears its own repeat spread by 5-10x, but neither clears
perf-log.md's 2% cross-session bar on its own — two independent measurements agreeing in sign and
magnitude is what carries it, not either number alone. Quote the range, not the better half.

It *costs* memory: at d24/dbs2 the peak went 21,129 → 21,566 MiB, i.e. **+437 MiB** for the fp4
weight copies and their block scales (~0.6 bytes per matmul parameter over 730M). At d20/dbs4 the
peak instead moved *down* 16 MiB -- not investigated, but consistent with the peak there falling
somewhere other than the optimizer step, where the cache buffers are not the binding term.

Refresh **after `optimizer.step()`**, never between a forward and its backward: the forward hands
those buffers to `save_for_backward` and the refresh rewrites them in place.

Equivalence is *not* testable from a training run -- the backward draws a fresh Hadamard rotation
and EDEN seeds every step and nothing seeds them, so no two runs agree bit for bit. **Measured
2026-08-18** rather than assumed: two identical `--nvfp4 --seed 42` runs agree at steps 0 and 1
and diverge at step 2, where `--fp8` does not (queue C1). The forward
has no randomness, so `TestWeightCache` in `tests/test_nvfp4.py` proves it exactly there, with a
negative control that a *stale* cache must be detectably different.

## Windowing: ~~`--window-pattern L` is worth +43.7%~~ — reversed, `SSSL` now wins by 5.7%

FA3 does not load on sm_120 (`_load_flash_attention_3` gates on capability major 8/9), so
attention used to run on SDPA — and `_sdpa_attention` only takes the fast `is_causal=True` path
when the window spans the sequence. The default `SSSL` puts 3 of every 4 layers on a 512 window
at Tq=2048, which fell through to a materialized `[2048, 2048]` mask and off flash entirely:

| d24, dbs 2, before the fix | tok/s  | MFU   |
|----------------------------|--------|-------|
| `SSSL` (default)           | 12,784 | 21.3% |
| `L`                        | 18,362 | 43.8% |

**Fixed 2026-08-18 by porting `perf-log.md` experiment 13's windowed flash** — `sm120/attention.py`
routes a causal sliding window through `aten::_flash_attention_forward`, which takes
`window_size_left` directly and reaches the same `pytorch_flash` kernels SDPA's fast path already
uses. No new CUDA. Importing `nanochat.sm120` installs it, and `base_train.py` now prints
`✓ windowed flash attention` instead of warning.

d20 / dbs 4, 1 GPU, `--nvfp4 --nvfp4-weight-cache`, batched arms with the reference repeated,
tok/s at step 13:

| arm          | tok/s      | MFU    | peak       |
|--------------|------------|--------|------------|
| `L`          | 34,167     | 49.98% | 16,781 MiB |
| **`SSSL`**   | **36,095** | 47.03% | 16,781 MiB |
| `L` (repeat) | 34,106     | 49.89% | 16,781 MiB |

**`SSSL` is +5.7% over `L`**, against a reference repeat spread of 0.18% — so the sign that this
file recorded was not just wrong in magnitude, it was wrong in direction. The number matches
`perf-log.md`'s +6.2% at d12 and d16 fp8, i.e. the effect is depth- and precision-flat. The `SSSL`
arm also started 9 C hotter than the first `L` arm, which makes +5.7% conservative.

Read the MFU column with care: it *falls* while throughput rises, because `bf16_mfu` counts the
attention the window skips. Memory is identical to the last MiB across all three arms.

**Every NVFP4-vs-fp8 number elsewhere in this file predates this fix**, and was measured with
~75% of layers off the flash path inflating attention's share of the step. That dilutes a
Linear-only optimization. **How much it was worth to the NVFP4-vs-fp8 ratio is still unmeasured**
— the d20 re-run of 2026-08-18 (*Result*, +29.4%) is not the experiment that settles it, because
the 1.25x it improves on was itself captured after this fix and the difference is accounted for
by A1-A3. What is unambiguously still diluted is **d12**: those arms have never been re-run, so
the +7.0% figure that headline used to carry remains an underestimate by an unknown amount.

## Run cost on this box

Token budget is `target_param_data_ratio × (transformer_matrices + lm_head)` — the shard count
is only the data pool. At speedrun's ratio 8, with the tok/s above:

| depth | scaling params | tokens | shards | runtime         |
|-------|----------------|--------|--------|-----------------|
| d12   | 110M           | 0.88B  | 16     | 2.5 h           |
| d20   | 435M           | 3.48B  | 65     | 31 h            |
| d24   | 730M           | 5.84B  | 108    | 89 h (3.7 days) |

Measured on the local corpus: 252.8M chars/shard at 4.693 chars/token = **~53.9M tokens/shard**,
so the 170 train shards `runs/speedrun.sh` downloads hold ~9.16B tokens. That sizing matches the
*default* ratio 12 for d24 (8.76B ≈ 162 shards), not speedrun's own ratio 8.

Runtimes are lower bounds: they come from 4-step runs, which predate thermal decay, and this box
loses a few percent of steady-state throughput as it heats.

## Build

The kernels need `cuda_fp4.h` intrinsics that lower to `cvt.rn.satfinite.e2m1x2.f32`, so they
compile `-gencode arch=compute_120a,code=sm_120a` and need **CUDA ≥ 12.8 — not 13.x**. 12.8 is
the release that added sm_120 and NVFP4 together; nothing in this path requires CUDA 13.

| thing                              | actually needs                                            |
|------------------------------------|-----------------------------------------------------------|
| sm_120 at all                      | CUDA ≥ 12.8 (the cu128 torch wheel ships native `sm_120`) |
| NVFP4 `_scaled_mm`                 | CUDA ≥ 12.8 — verified working on `.venv` (cu128)         |
| compiling these kernels            | CUDA ≥ 12.8                                               |
| ~~**`.venv-cu130` specifically**~~ | ~~nothing above — only that this box has no 12.x `nvcc`~~ |

**That last row is obsolete: 12.8 is installed now, and `.venv` is the venv to use.** All of A1
was measured on it. `cpp_extension` refuses a CUDA *major* mismatch between `nvcc` and the torch
build, so what matters is that a toolkit of the right major exists — `/usr/local/cuda-12.8` for
`.venv`/cu128 and 13.2/13.3 for `.venv-cu130`/cu130 — and `ext.resolve_cuda_home()` picks it per
venv, printing which nvcc it chose. `ext.unavailable_reason()` reports a mismatch it cannot fix,
and `tests/test_nvfp4.py` skips on it.

Built on first use via `torch.utils.cpp_extension.load`, cached under
`$NANOCHAT_BASE_DIR/torch_extensions/quartet2-torch<v>-cu<v>/`. First build is a few minutes.
The directory is keyed by torch build on purpose: with a shared one, the cu128 venv silently
loaded the cu130 venv's `.so` and *ran*, which is worse than failing.

`--nvfp4-lt-gemm`'s launcher is a **second, separate** extension (`fp4gemm-torch<v>-cu<v>/`), and
it is host-only C++ — no device code, so no nvcc, no gencode, and it builds in seconds. That is
not just tidiness: nvcc 12.8 hits an internal compiler error on GCC 13's `<bits/stl_construct.h>`
at `-std=c++20`, which this repo requires everywhere, so a `.cu` with the same contents does not
compile at all here. It links `-lcublasLt`, but at runtime the SONAME is already loaded by torch,
so it shares torch's copy — the same library `_scaled_mm` calls, whatever built it.

No new Python dependencies. Two of upstream's were designed out:

- **flashinfer** (`mm_fp4`) → `torch._scaled_mm`, which already does block-scaled NVFP4 here.
  The kernels emit block scales in exactly the cutlass 128x4 swizzle it indexes, so this is a
  drop-in. (Row-major scales do not error — they return confident garbage. There is a negative
  control test for it.)
- **scipy** (`linalg.hadamard`) → a four-line Sylvester construction, verified bit-identical to
  scipy's for n=2/16/128 when scipy happens to be installed.

## Layout

```
nanochat/sm120/nvfp4.py            NVFP4Linear, convert_to_nvfp4_training, fp4_mm, dequantize
nanochat/sm120/fp4_gemm.py         --nvfp4-lt-gemm: one autotuned cuBLASLt plan per GEMM shape
nanochat/sm120/csrc/fp4_gemm.cpp   ours: the launcher itself. Host-only, so no nvcc and no gencode
nanochat/sm120/quartet/
    csrc/*.cu, *.cuh               upstream verbatim @ 5f2a47e (Apache-2.0)
    csrc/binding.cpp               ours: pybind11/at::Tensor, replacing upstream's nanobind
    ext.py                         JIT build + availability gate
    quant.py, rht.py               ported wrappers (_quartet2 -> ext.load())
    reference.py                   pure-torch NVFP4, the test oracle
tests/test_nvfp4.py                54 tests
scripts/bench_nvfp4.py             per-shape probe (forward only; see caveat below)
scripts/probe_fp4_gemm.py          the GEMM inventory, autotuned against _scaled_mm (queue A1)
scripts/arm_batch.sh               batched-arm throughput A/B (A, B, A) for any two flag sets
scripts/probe_fwd_rht.py           forward-GEMM error with/without a rotation, on captured tensors
```

The profiling harness this file quotes measurements from — `scripts/profile_train.py`,
`kernel_report.py`, `ncu_report.py`, `gpu_idle_report.py` and `nanochat/sm120/profiling.py`'s
`--profile-steps` — was **not** ported into this repo. Where a section below cites one, it is
recording where a number came from on the branch it was measured on, not naming a script you can
run here.

Upstream's `linear.py` and `pseudoquant.py` are not vendored — `nvfp4.py` and `reference.py`
replace them. The seam in files upstream owns is small: two flags and one conversion block in
`scripts/base_train.py`, plus widening `disable_fp8`'s predicate so evaluation drops back to
bf16 for NVFP4 the same way it does for fp8. Nothing in `nanochat/` points into `sm120/`.

`NVFP4Linear` subclasses nanochat's `Linear`, not `nn.Linear` — the invariant from commit
c620d36. Getting that wrong removes the layer from `GPT.num_matmul_params()` and under-reports
FLOPs/token; there is a test pinning it.

## Numerics: C1, eight paired seeds — cleared, weakly

**24 runs, 2026-08-19, `dev-ignore/overnight/c1/`.** perf-log.md experiment 18's protocol at the
config it calibrated on — d12 / dbs 8 / 2 GPU / 100 steps, `--eval-tokens 2097152`, no cooldown
(bpb does not depend on thermal state). Eight seeds x {`--fp8`, `--nvfp4`-A, `--nvfp4`-B}: the
third arm is a **replicate**, needed because `--nvfp4` is not run-to-run reproducible (see *The
weight cache*). Each seed carries its own fp8 reference, so a bad run contaminates one delta
rather than all eight — the trap experiment 18 lost two flags to.

| seed | fp8      | nvfp4-A  | nvfp4-B  | delta     | replicate |
|------|----------|----------|----------|-----------|-----------|
| 42   | 1.464083 | 1.477265 | 1.476734 | +0.012916 | 0.000531  |
| 43   | 1.461437 | 1.481243 | 1.481490 | +0.019929 | 0.000247  |
| 44   | 1.468248 | 1.479530 | 1.479773 | +0.011404 | 0.000243  |
| 45   | 1.466406 | 1.457611 | 1.457648 | −0.008777 | 0.000037  |
| 46   | 1.471653 | 1.462465 | 1.461963 | −0.009439 | 0.000502  |
| 47   | 1.462397 | 1.475208 | 1.475029 | +0.012722 | 0.000179  |
| 48   | 1.458821 | 1.478994 | 1.478845 | +0.020099 | 0.000149  |
| 49   | 1.472422 | 1.460864 | 1.460134 | −0.011923 | 0.000730  |

**mean +0.005866 ± 0.009610 (2 sem), band −0.0037 .. +0.0155, signs 5+/3−.** The band spans zero,
so `--nvfp4` **clears** — and it survives the heavy-tail check that matters here: leave-one-out
over all eight seeds leaves *every* subset spanning zero, so the verdict does not rest on one
point.

Three things this settles that the queue was guessing at:

- **The distribution is bimodal, and no seed produced a small delta.** Five land +0.011 to +0.020,
  three land −0.009 to −0.012, |Δ| is between 0.0088 and 0.0201 everywhere. NVFP4 always moves bpb
  by ~0.013 at 100 steps; only the *sign* is seed-dependent. That is trajectory reshuffling in a
  chaotic regime, not a small systematic — and it means **mean ± 2 sem is a poor summary here**:
  the median is +0.0121, double the mean.
- **The replicate null is negligible**: mean 0.00033, max 0.00073, i.e. **seed sd is 58x replicate
  sd**. The nondeterminism is real but irrelevant at this horizon, so **future batteries can drop
  the B arm** and run 16.
- **This is a much weaker clear than `--fp4-fwd`'s.** Experiment 24 cleared at +0.0006 ± 0.0016,
  4+/4−; this is +0.0059 ± 0.0096 at 5+/3− — a mean 10x larger against a band 6x wider. Same
  verdict on the same criterion, far less of it.

**What it does not settle is the horizon, which is the whole question.** A ±0.013 seed-noise cloud
centred near zero is exactly what experiment 24 saw before experiment 25 found a clean monotone
+0.0147 at ratio 12. C1 measures a *level* at one horizon and cannot see a *trend*. C2 is the
experiment.

## Numerics: C2, the horizon — the deficit is real, and the speed nearly pays for it

**4 runs at the full ratio-12 horizon (2,520 steps), 2026-08-19, `dev-ignore/overnight/c2/`.**
2 seeds x {`--fp8`, `--nvfp4`}, d12 / dbs 8 / 2 GPU, horizon from `--target-param-data-ratio 12`
rather than `--num-iterations` — the LR schedule derives from it, so this is a different
experiment from C1, not a longer one.

| seed | fp8      | nvfp4    | delta         | fp8 CORE | nvfp4 CORE | CORE delta |
|------|----------|----------|---------------|----------|------------|------------|
| 42   | 0.833421 | 0.847110 | **+0.013689** | 0.1543   | 0.1317     | −0.0226    |
| 43   | 0.833425 | 0.847187 | **+0.013762** | 0.1501   | 0.1488     | −0.0013    |

**The two seeds agree on the deficit to 7e-5**, and the fp8 arms agree to 4e-6 — the sharpest
measurement in either log. The paired deficit is monotone from step 500 and **still widening at
the last eval**: +0.0071 at 500, +0.0105 at 1000, +0.0120 at 2000, +0.0137 at 2520, with the two
seeds tracking each other to within 0.0017 the whole way. This is a compounding systematic, and
**C1's "seeds disagree on the sign" was purely a short-horizon artifact** — at 100 steps the
trajectory noise is +/-0.013 and swamps it; by ratio 12 the seed variance has washed out entirely.

**CORE is not resolved and should not be quoted.** The two CORE deltas disagree by 0.0213 — the
nvfp4 arm's own seed spread is 0.0171 against fp8's 0.0042 — so seed 42's −0.0226 is one noisy
run, not a finding. n=2 cannot resolve CORE at this spread.

### The verdict is *not* experiment 25's, and the difference is the point

```
bpb deficit at ratio 12                    : +0.0137   (both seeds)
nvfp4 speedup, measured on these arms      : 1.199x    (95.3 vs 114.3 min)
extra steps affordable at equal wall clock : ~500      (2520 -> 3020)
projected bpb gain from those steps        : -0.0114   (local slope, decaying 0.721/250 steps)
net at equal wall clock                    : +0.0023 bpb behind
```

perf-log.md experiment 25 killed `--fp4-fwd` for spending ~15% of training progress to buy **3.9%**
of wall clock, a ~4x net loss. NVFP4 spends about the same ~15% to buy **19.9%** — **roughly
break-even, and about 0.002 bpb on the wrong side of it.** Same shape of cost, five times the
compensation. That is a different decision, and this file should not inherit experiment 25's
conclusion by analogy.

**The break-even number is an extrapolation, not a measurement**, and that is its weakness: the
0.0114 comes from projecting the last measured slope forward 500 steps with a fitted decay. The
direct experiment is one equal-wall-clock pair — `--nvfp4` at 3,020 steps against `--fp8` at 2,520
— which answers the question instead of modelling it. Queue C3.

## Numerics: C5, the recipe — lm_head is not where the deficit lives, and 80/20 is a loss

**5 arms at the ratio-12 horizon (2,520 steps), 2026-09-03, `dev-ignore/overnight/c5/`.**
d12 / dbs 8 / 2 GPU / seed 42, C2's protocol exactly (`--total-batch-size 524288 --eval-every 250
--eval-tokens 2097152`). Two of the five re-measure C2's own arms, because **C2 was run in the
sibling checkout** (`sm120_nanochat`, branch `quartet-te`) which has no `nvfp4_numerics.py` — its
constants are a cross-tree reference, not a within-tree one.

| arm                | flags beyond the base                                         | bpb          | min     |
|--------------------|---------------------------------------------------------------|--------------|---------|
| `fp8-w65`          | `--fp8`                                                       | **0.833176** | 112.35m |
| `nvfp4-plain-w65`  | `--nvfp4`                                                     | **0.846993** | 94.05m  |
| `nvfp4-recipe-w65` | `+ --nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd` | **0.844619** | 95.83m  |
| `fp8-w20`          | `--fp8 --warmdown-ratio 0.2`                                  | **0.840602** | 112.21m |
| `nvfp4-recipe-w20` | recipe `+ --warmdown-ratio 0.2`                               | **0.853487** | 95.83m  |

**The parity check passes.** The plain deficit reproduces at **+0.013817** against C2's +0.013689
(s42) and +0.013762 (s43), and the trajectory tracks C2 step for step: +0.0090 / +0.0105 / +0.0119
/ +0.0138 at 500 / 1000 / 2000 / 2500, against C2's +0.0071 / +0.0105 / +0.0120 / +0.0137. Both
w65 arms land ~2e-4 *below* their C2 counterparts (fp8 −0.000245, nvfp4 −0.000117) — real
cross-tree drift, ~60x C2's 4e-6 fp8 seed spread, and visible at step 2 as 10.362546 here against
10.362545 there. It cancels in every paired delta, which is what the pairings use.

### The four pairings

| pair                   | delta         | reading                                             |
|------------------------|---------------|-----------------------------------------------------|
| recipe w65 − fp8 w65   | **+0.011443** | the recipe closes 17% of the deficit and leaves 83% |
| recipe w65 − plain w65 | **−0.002374** | the recipe's gain in isolation                      |
| fp8 w20 − fp8 w65      | **+0.007426** | 80/20 costs fp8 three times what the recipe buys    |
| recipe w20 − fp8 w20   | **+0.012885** | the deficit is *worse* under NVIDIA's schedule      |

**The recipe shifts the curve, it does not flatten it** — the distinction that decides what the
result means. Its gain is flat across training (0.0036 / 0.0026 / 0.0027 / 0.0020 / 0.0022 at
500 → 2500) while the deficit underneath keeps widening at the same rate as plain NVFP4's. Taking
lm_head out of fp4 removes a roughly constant offset; it does not touch the compounding
systematic. **So the cost at d12 is forward noise in the blocks, not the output head** — which is
the C2 decision rule's third branch, and it lands there despite lm_head being a quarter of d12's
linear compute.

**And it is nearly free in time**, which is the one unambiguously good result: 95.83m against
plain's 94.05m, 1.8 min for 0.0024 bpb, at 1.172x fp8's wall clock. The `fp8-fwd` path costs
about what its flag help predicted (17.9 ms against 15.6 all-fp4 at the d12 lm_head shape).

**80/20 is a loss at this horizon, for both precisions** (fp8 +0.0074, recipe +0.0089), and it
does not rescue the recipe — the deficit *grows* to +0.0129. The 100-step d24 pair that suggested
otherwise (fp8 1.302880, nvfp4 1.288498) was inside noise, as its own note said. Note the w20 arms
are still descending steeply at the last eval (fp8 0.8906 → 0.8412 over the final 500 steps
against w65's 0.8557 → 0.8337), so 80/20 is a schedule built for a longer horizon than ratio 12;
this measures it at the horizon we train at, not at the one NVIDIA does.

### Configuration note — `NANOCHAT_FA2_SWINDOW=1` is load-bearing for *numerics*, not only speed

C2 ran with it; the first c5 launch did not, and the difference is not confined to wall clock
(2.71 vs 4.38 s/step, 33.2% vs 20.5% MFU — 3 of every 4 layers are windowed under `SSSL`). The
SDPA mask path changes the arithmetic: step-1 loss 10.384376 against 10.384375 with FA2. Batch
relaunched with it set; all five arms carry `✓ FA2 windowed flash attention` in their logs.

**A determinism claim made here on 2026-09-03 was wrong and is withdrawn.** Two 3-step `--fp8
--seed 42` runs diverged under SDPA (10.362544 vs 10.362545) and agreed under FA2, which was
written up as "bitwise deterministic under FA2". Three steps cannot support that:
`dev/LOG_sm120.md:27-31` already records that `--fp8` at d12/dbs8/2 GPU agrees at steps 0-2 and
*then* drifts, reaching 7.4e-5 by step 19 (suspected DDP reduction order), and warns in terms that
the donor's 3-step determinism table was too short to see it. The check repeated that mistake.
**Neither precision is bit-reproducible here**; SDPA merely surfaces the drift ~17 steps earlier.
What licenses the paired comparisons is not determinism but the measured replicate null at this
horizon — mean 0.00033, max 0.00073 (*Numerics: C1*) with the C2 fp8 seeds agreeing to 4e-6 —
which resolves anything above ~0.001.

### What C5 leaves

The recipe is not adoptable on its own at d12: +0.0114 sits outside the 0.003 band the arm plan
set for adoption, and outside the 0.003-0.010 band where the remaining recipe levers would be
worth a batch each. `--nvfp4-bf16-blocks` is still worth one arm — it attacks the block
forwards, which is where C5 says the cost actually is, unlike the lm_head exclusion. ~~`--nvfp4-scaling
delayed` and `--nvfp4-bwd-source bf16` are backward-side and C5 gives no reason to expect them to
move a forward-noise deficit.~~ **That dismissal assumed its conclusion** — see *Numerics: C6*,
which shows `fp8-fwd` changes the backward operand source as well as the forward precision, so C5
never isolated the forward at all. ~~**Value-level stochastic rounding in the forward is the lever this points at**~~ —
the target was
right, the ranking was not: *Numerics: C8* prices the forward at −0.0132 and puts the
**Hadamard-rotated forward** ahead of SR (free, existing kernels, no bias/variance tradeoff).

Keep the recipe flags regardless of adoption: −0.0024 bpb for +1.9% wall clock is the best
exchange rate on either list.

## Numerics: C6 — an optimized fp8 erases NVFP4's speed advantage at d12

**2 arms at the ratio-12 horizon (2,520 steps), 2026-09-03, `dev-ignore/overnight/c6/`.**
C5's protocol exactly, so both pair against c5 arms in the same tree and session.

| arm                         | flags beyond the base                    | bpb          | vs `c5-fp8-w65` | min    | peak    |
|-----------------------------|------------------------------------------|--------------|-----------------|--------|---------|
| `fp8-delayed-nt-w65`        | `--fp8 --fp8-scaling delayed --wgrad-nt` | **0.833571** | **+0.000395**   | 95.46  | 10.6 GB |
| `nvfp4-recipe-blocks24-w65` | recipe `+ --nvfp4-bf16-blocks 2,4`       | **0.841941** | **+0.008765**   | 100.09 | 11.5 GB |

### Delayed fp8 scaling plus NT wgrad is free — and that is the headline

**+0.000395 bpb for 15% less wall clock** (95.46 against `c5-fp8-w65`'s 112.35) and 0.8 GB less
memory. The delta is flat across the run — +0.0004 at step 750, +0.0004 at 2520 — so unlike the
NVFP4 deficit it does not compound, and it sits at the edge of the ~0.001 the replicate null can
resolve. Treat it as neutral.

This kills a contradiction the 50-step README tables could never settle: they read +0.0134/+0.0187
(worse) at d24 and −0.0093/−0.0130 (better) at d12, i.e. one noise unit in each direction. The
donor's experiment 10 (`dev/LOG_sm120.md:17-19`) claimed −0.022 over 8 paired seeds. The truth at
d12/ratio 12 is **zero**, which no 50-step run could have seen. Note this is `--fp8-scaling`;
**queue B1 is the NVFP4 counterpart `--nvfp4-scaling delayed` and remains unmeasured end to end**,
though this raises the prior that it too is cheap.

**The consequence for NVFP4 is the real result.** The fp8 baseline is now faster than every NVFP4
arm on record at d12:

| config                  | min    | vs best fp8 | bpb deficit |
|-------------------------|--------|-------------|-------------|
| `fp8-delayed-nt`        | 95.46  | —           | —           |
| `nvfp4-plain`           | 94.05  | 1.015x      | +0.0138     |
| `nvfp4-recipe`          | 95.83  | 0.996x      | +0.0114     |
| `nvfp4-recipe-blocks24` | 100.09 | 0.954x      | +0.0088     |

**C2's "break-even at equal wall clock" does not survive.** That verdict rested on NVFP4 being
1.199x faster, buying ~500 extra steps to spend on the deficit. Against an optimized fp8 the
margin is 1.015x for the arm carrying the full +0.0138, and every configuration that narrows the
deficit is *slower* than fp8. At d12 there is no wall-clock budget left to buy the deficit back.

This is depth-specific and does not transfer: the same README rows put NVFP4 at 585k against
optimized fp8's 482k at d24 (1.21x), because bigger GEMMs use fp4 better. **d12 is now settled
against NVFP4; d24 is where the economics could still work**, and it is 8,352 steps at ratio 12
(batch 1,048,576, 8.76B tokens) or 13,920 at ratio 20.

### Block exclusion is the first lever that attenuates rather than offsets

`--nvfp4-bf16-blocks 2,4` on top of the recipe — first 2 and last 4 blocks plus lm_head in
fp8-fwd, 36 of 80 Linears left in fp4 — closes **37% of the deficit** cumulatively (+0.013817 ->
+0.008765), against the recipe's 17% alone.

The trajectory is what distinguishes it from C5. The recipe's gain was flat; this one **grows**
(−0.0002 at step 250, −0.0012 at 750, −0.0024 at 1500, −0.0028 at 2500), and the deficit's growth
rate is roughly halved:

| arm              | deficit @250  | @2520         | growth        |
|------------------|---------------|---------------|---------------|
| plain            | +0.006221     | +0.013817     | +0.007596     |
| recipe           | +0.005261     | +0.011443     | +0.006182     |
| **+ blocks 2,4** | **+0.005041** | **+0.008765** | **+0.003724** |

Halving the fp4 block count roughly halves the growth. A two-point linear fit gives ~0.00042 per
fp4 block with a ~0.0012 intercept, i.e. the compounding term looks proportional to the number of
fp4 blocks, with a residual that is not.

C5's probe→bpb calibration held: predicted −0.0019 for this step, measured −0.002678, so the probe
understates by ~40% but ranks correctly. That makes `scripts/probe_nvfp4_numerics.py` a usable
minutes-long screen for future candidates.

### The confound — C5 and C6 never isolated the forward

`--nvfp4-exclude-precision fp8-fwd` changes **two** things on the excluded layers: an fp8 forward
*and* a bf16-sourced NVFP4 backward (`dev/nvfp4-transformer-engine.md:118-121`). So neither C5's
−0.0024 nor C6's −0.0027 attributes to forward precision on its own, and the claim in *What C5
leaves* that "forward noise in the blocks is the cost" is not established by these arms.

The alternative looked equally good at the time. The ladder in
`nvfp4-transformer-engine.md:135-141` shows `--nvfp4-bwd-source bf16` is the only lever that makes
the backward an unbiased estimator (1.00/0.90 bits per 4x averaging against the default's
0.29/0.54, because the default re-quantizes saved fp4 operands and "is biased against the true
gradient"). A biased gradient estimator is exactly what would produce a compounding deficit, so
C6's halved growth rate seemed equally explained by six fewer biased backwards as by six fewer
fp4 forwards.

**C7 refuted that the same day** — the unbiased backward is *worse*, by more than the whole recipe
gains (see *Numerics: C7*). Since C6's excluded layers took an fp8 forward **plus** a backward
change now known to be harmful, C6's −0.0027 has to come from the forward, and understates it.
That inference is sound for C6's gain specifically, and *Numerics: C8* then established the
general claim directly — the forward main effect at fixed backward is −0.0132, more than the whole
plain deficit. What is still unmeasured is the fp4 backward's own share: every arm on record has
fp4 backward GEMMs, and no flag gives a high-precision backward under an fp4 forward.

## Numerics: C7 — the unbiased backward is the worst arm in the series

**1 arm at the ratio-12 horizon (2,520 steps), 2026-09-03, `dev-ignore/overnight/c7/`.**
`--nvfp4 --nvfp4-bwd-source bf16`, C5's protocol, 96.44m, 14.9 GB peak (+0.9 over plain).
Designed to break C5/C6's confound: it changes only where the backward's fp4 operands are
quantized *from*, leaving the forward bit-identical — verified, step-0 loss 10.397527 matches
plain NVFP4 exactly.

| arm                        | bpb          | deficit vs fp8 | growth 250->2520 |
|----------------------------|--------------|----------------|------------------|
| `c5-fp8-w65`               | 0.833176     | —              | —                |
| `c5-nvfp4-plain-w65`       | 0.846993     | +0.013817      | +0.007596        |
| `c5-nvfp4-recipe-w65`      | 0.844619     | +0.011443      | +0.006182        |
| `c6-...-blocks24-w65`      | 0.841941     | +0.008765      | +0.003724        |
| **`c7-nvfp4-bwdbf16-w65`** | **0.853135** | **+0.019959**  | **+0.012614**    |

**It is 44% worse than plain NVFP4** (+0.0200 against +0.0138), and the damage is in the
compounding term: growth two-thirds *higher* than plain, the mirror image of blocks-2,4 halving
it. The gap opens monotonically from step 1000 (+0.0010, +0.0029, +0.0034, +0.0055, +0.0063,
+0.0064) — an order of magnitude past the replicate null, so not noise.

### Three things this settles

**TE's "avoid double quantization errors" does not hold here.** Quantizing the backward operands
from bf16 instead of requantizing the saved fp4 costs more than the entire lm_head recipe gains,
four times over. Quartet-II's choice is load-bearing, not incidental.

**Forward/backward consistency beats fidelity to bf16.** The default differentiates the function
the forward actually computed; bwd-source bf16 differentiates the un-quantized function. Unbiased
against bf16 but inconsistent with its own forward is worse than biased against bf16 and
consistent — and it compounds, which is what a systematic gradient mismatch accumulating over
2,520 steps should do.

**The probe's gradient columns anti-predict bpb.** This was its best variant by a wide margin
(1.00/0.90 bits per 4x against the default's 0.29/0.54) and it is the worst run on record. The
forward `dloss` column has now called C5, C6 and C7 correctly; the backward columns have one clean
miss. **Use `scripts/probe_nvfp4_numerics.py` as a forward screen only.** Its "the backward levers
do not move the model-level numbers" claim rests on the forward-only variant having identical
`dloss` — but `dloss` is a forward metric at a fixed checkpoint, so the backward cannot move it by
construction. That is tautological, not evidence.

### What is still unmeasured

**The fp4 backward's own contribution to the deficit.** The 2x2 is:

|             | fp4-source bwd  | bf16-source bwd |
|-------------|-----------------|-----------------|
| fp4 forward | plain, +0.0138  | **C7, +0.0200** |
| fp8 forward | not expressible | **C8, +0.0067** |

Every cell runs fp4 backward GEMMs, and "fp8 forward with the fp4-source backward" cannot be
built: the backward requantizes the operands the forward saved, so if the forward is not fp4 there
is nothing to requantize. Forward precision and backward source are coupled by construction —
which is why C5 and C6 moved both at once.

**C7's right-hand column is what makes C8 readable**: both run
`_FP8FwdMatmul.backward`'s settings (`rht="all", bwd_source="bf16"`, `nvfp4_fp8fwd.py:47-52`), so
C8 − C7 varies the forward alone. That contrast is *Numerics: C8* and it is −0.0132. What remains
unmeasured is the empty cell — the fp4 backward's own share — which needs a bf16 backward mode in
`_NVFP4Matmul.backward` (~an afternoon) to fill directly rather than bound.

## Numerics: C8 — the fp4 forward carries the deficit, and it is worth more than all of it

**1 arm at the ratio-12 horizon (2,520 steps), 2026-09-03, `dev-ignore/overnight/c8/`.**
`--nvfp4 --nvfp4-exclude c_q,c_k,c_v,c_proj,c_fc,lm_head --nvfp4-exclude-precision fp8-fwd` —
suffix matching puts all 73 convertible Linears in fp8-fwd, 0/80 left as `NVFP4Linear`. 104.06m,
12.0 GB peak. **0.839906.**

Designed as the other half of C7. `_FP8FwdMatmul.backward` hardcodes `rht="all",
bwd_source="bf16"` (`nvfp4_fp8fwd.py:47-52`), which is exactly what C7 ran model-wide, so **C8 − C7
varies the forward GEMM precision and nothing else** — the isolation C5 and C6 could not give,
because `fp8-fwd` couples the two. Verified at startup: step-0 loss 10.397516, the fp8 value, not
NVFP4's 10.397527.

| arm                           | bpb          | deficit vs fp8 | growth 250->2520 | fp4 block forwards |
|-------------------------------|--------------|----------------|------------------|--------------------|
| `c5-fp8-w65`                  | 0.833176     | —              | —                | —                  |
| `c5-nvfp4-plain-w65`          | 0.846993     | +0.013817      | +0.007596        | 12                 |
| `c5-nvfp4-recipe-w65`         | 0.844619     | +0.011443      | +0.006182        | 12                 |
| `c6-...-blocks24-w65`         | 0.841941     | +0.008765      | +0.003724        | 6                  |
| `c7-nvfp4-bwdbf16-w65`        | 0.853135     | +0.019959      | +0.012614        | 12                 |
| **`c8-nvfp4-fp8fwd-all-w65`** | **0.839906** | **+0.006730**  | **+0.003300**    | **0**              |

### The forward main effect is −0.0132, and it compounds

C8 − C7 at each eval: −0.0039, −0.0034, −0.0031, −0.0053, −0.0074, −0.0095, −0.0105, −0.0119,
−0.0121, −0.0133, **−0.013229 at 2520**. Monotone from step 750 and still widening at the end.

**Swapping fp4 → fp8 in the forward recovers more than the entire plain-NVFP4 deficit** (+0.0138).
That settles what C5 asked and could not isolate: the cost is the fp4 forward. Not lm_head
(−0.0024 in C5), and emphatically not the backward operand source, which is worth −0.0062 in the
*wrong* direction (C7).

### Growth rate is roughly proportional to fp4 forward count

| fp4 block forwards | arm        | growth    |
|--------------------|------------|-----------|
| 12                 | plain      | +0.007596 |
| 6                  | blocks 2,4 | +0.003724 |
| 0                  | C8         | +0.003300 |

C6's two-point fit predicted ~0.0012 at zero blocks; C8 came in at 0.0033. The overshoot is
plausibly the bf16-source backward C8 carries — C7 shows that lever alone raises growth from
+0.0076 to +0.0126. So the compounding term looks like *fp4 forward count plus a backward-source
penalty*, and C8 is paying the second while having eliminated the first.

### C8 is the best NVFP4 arm on record — while handicapped

+0.0067 against fp8, carrying a backward C7 measured at +0.0062 of self-inflicted damage. Those
need not be additive, but if they are even roughly so, **an fp8 forward over the default
fp4-source backward would land near fp8 parity**. That configuration cannot be built today (no
saved fp4 operands to requantize), which makes it a hint about the ceiling rather than a
candidate: it says the headroom a better fp4 forward is chasing is the full +0.0138, not some
fraction of it.

### What this justifies

~~**The Hadamard-rotated forward moves to the front of the queue, ahead of stochastic
rounding.**~~ — **measured 2026-09-03 and reversed before it was built.** Rotating both forward
operands is arithmetically free, but on real activations it *costs* 0.06–0.08 effective bits of
forward accuracy rather than buying any, and the fused-EDEN form B0 proposed reusing costs 0.24 on
73 of 73 layers. See *B0: the rotated forward is a loss* below.

~~**So value-level stochastic rounding is now first**, by elimination~~ — **no: SR is a backward
lever, and C8's term is the forward.** NVIDIA's ablation (arXiv 2509.25149 §4.4, App. E.3) rounds
gradients stochastically and weights/activations to nearest, and reports that SR on activations or
weights *diverges* at 1.2B — it amplifies the rounding error the forward sees. Its place here is
making TE's unrotated-dgrad placement honest, not the forward deficit. What C8 measured is
unchanged — the fp4 forward is still where the deficit lives, and the headroom is still the full
+0.0138 — but the only forward lever left is *selection*, which layers run fp8, and the next
section prices it on the fp4-trained checkpoint: mostly adapted away.

**None of this rescues d12.** C8 runs 104.06m against `--fp8 --fp8-scaling delayed --wgrad-nt`'s
95.46m, so an fp8 forward over an fp4 backward is both slower and behind at this depth. C8's value
is diagnostic — it prices the headroom that justifies the kernel work, and the work pays off at
d24, where NVFP4 still holds ~1.21x (*Numerics: C6*).

## B0: the rotated forward is a loss, and the block scale is why — 2026-09-03

`scripts/probe_fwd_rht.py`, two checkpoints, one real batch each, all 73 convertible Linears.
Capture `(x, W)` per layer, run the fp4 forward GEMM with and without the rotation, and score
effective bits against the fp32 product of the same operands. Positive is better:

| rotation                                                    | c2-fp8-s42                | c2-nvfp4-s42        |
|-------------------------------------------------------------|---------------------------|---------------------|
| `transform_rht128` + 4/6 (what `--nvfp4-fwd-rht` would run)  | **−0.0798** (53/73 worse) | **−0.0592** (19/73) |
| `rht128_quant_eden` (what B0's row proposed reusing)         | **−0.2425** (73/73 worse) | **−0.2206** (73/73) |
| `quant_fp4` EDEN, unrotated — control for the row above      | −0.1680 (72/73)           | −0.1654 (73/73)     |
| torch, 2 / 4 / 8-wide (inside one block)                     | −0.012 / −0.020 / −0.027  | +0.023 / +0.017 / +0.009 |
| torch, 16-wide                                               | −0.0376                   | −0.0021             |
| torch, 32-wide                                               | −0.0592                   | −0.0338             |
| torch, 64-wide                                               | −0.0732                   | −0.0502             |
| torch, 128-wide                                              | −0.0764                   | −0.0579             |

**Monotone in width from 16 up, and nothing pays at any width**: inside a block (2/4/8-wide)
the effect is within ±0.02 bits and flips sign between checkpoints. The EDEN row is mostly not
the rotation: unrotated EDEN costs −0.17 on its own (a stochastic block scale against 4/6's
min-MSE choice), so the rotation's share of −0.24 is the same −0.06 to −0.08 the top row measures.

### Why, and it generalises

The harm is concentrated exactly where the outlier-flattening story predicts a *gain*. Sorting by
damage puts `mlp.c_proj` of every block at the top (−0.36 to −0.86 bits) and `lm_head` next
(−0.36), and those are the layers whose input `amax/rms` runs 42 to 1287 against ~7 for the rest.

**NVFP4 scales per 16 elements, so a concentrated outlier is already isolated.** The one block of
16 that contains it takes a large E4M3 scale; the other 47 blocks of a 768-wide row keep small
scales and full precision. Those layers are in fact the ones the unrotated forward quantizes
*best*: `mlp.c_proj`'s plain effective bits (3.5–3.96) are the highest in the model against
2.8–3.5 elsewhere, because a relu² output is sparse and its zeros are exact. A 128-wide rotation spreads that outlier's energy over all 128 lanes,
so all 8 blocks in the group inherit the magnitude: one bad block is traded for eight mediocre
ones, and summed over K that is worse. Rotation is the fix for a scale that is *coarse relative to
the outlier* — per-tensor fp8, which is where QuaRot, SpinQuant and TE's wgrad RHT come from. A
1-in-16 block scale has already done that job.

This is not an argument against the *backward's* RHT, which is there for a different reason: EDEN
needs the rotation to make its block scale an unbiased estimator, not to widen dynamic range.

### The design the probe saved

Worth recording, because the queue row understated the work and the next rotation idea will hit
the same wall. **"Rotations cancelling across the GEMM" holds for the forward GEMM alone, not end
to end.** The forward would rotate along K; the backward's `rht128_requant` transposes *and*
rotates along M (`quartet/quant.py:338`), a different axis, so the forward's K-rotation lands on
the *uncontracted* axis of the requant output and survives into both backward results:
`dgrad = (E·W)·Aᵀ` and `wgrad = (Eᵀ·X)·Aᵀ`, where `A = swizzle_hadamard(had)`. Undoing it needs a
second unrotated fp4 weight in the cache (for dgrad) plus a once-per-optimizer-step un-rotation of
the fp32 gradient accumulator (for wgrad, which is exact because `A` is fixed and accumulation is
linear). That is an M-sized change, not the S the row implied.

Two mechanical facts the probe pinned along the way, both reusable:

- `transform_rht128(h[:16], x)` is `x·Aᵀ` blockwise **in natural column order**, and
  `rht128_quant_eden` writes the same order (cosine 0.995 against it, against 0.005 for the
  unrotated control). So the one-kernel-family warning in `nvfp4_numerics.py` is about the
  effective matrix being `swizzle_hadamard(h)` rather than `h`, not about the output columns
  moving; its docstring now says so.
- `A` is orthonormal to 2e-4: the bf16 rounding of 128^-0.5 leaves `A·Aᵀ = 0.99979·I`, which is
  ~20x below bf16's own rounding of the output and would not have needed correcting.

### What it costs to know this

Nothing but the probe — the lever was never built. The gate was designed as step 2 of the
implementation plan and run first, which is the order to keep: **a forward-error probe on real
captured tensors is ~2 minutes and prices any forward-numerics lever before the ~1.7 h horizon
arm, and before the code.** `--nvfp4-fwd-rht` does not exist and should not be added.

## Forward noise by layer type, and which checkpoint to probe — 2026-09-03

`scripts/probe_nvfp4_numerics.py` gained `fp8fwd-<type>` variants: one layer type in `fp8-fwd`,
the rest fp4. Units are 768² GEMMs (a block is 12, the trunk 144, lm_head 43). "closed" is the
share of the all-fp4 loss delta removed, one batch, step 2520 of each checkpoint.

| in fp8-fwd                          | units | fp8-trained (c2-fp8-s42) | fp4-trained (c2-nvfp4-s42) |
|-------------------------------------|------:|-------------------------:|---------------------------:|
| nothing (all fp4)                   |     0 | +0.0768                  | +0.0121                    |
| lm_head (C5's recipe)               |    43 | +0.0422 (45%)            | +0.0124 (0%)               |
| lm_head + last block (`0,1`)        |    55 | +0.0291 (62%)            | +0.0128 (0%)               |
| every `attn.c_proj`                 |    12 | +0.0657 (14%)            | +0.0125 (0%)               |
| every `mlp.c_proj`                  |    48 | +0.0623 (19%)            | +0.0117 (4%)               |
| every `mlp.c_fc`                    |    48 | +0.0622 (19%)            | +0.0126 (0%)               |
| every `c_q`/`c_k`/`c_v`             |    36 | +0.0626 (19%)            | +0.0108 (11%)              |
| lm_head + first 2 / last 4 (C6)     |   115 | +0.0147 (81%)            | +0.0090 (25%)              |
| everything (C8)                     |   187 | +0.0027 (97%)            | +0.0055 (54%)              |

**The fp8-trained checkpoint measures a perturbation the fp4 run mostly never pays.** A model
trained in fp4 has adapted to its own forward noise — residual +0.012 against +0.077 — and
lm_head and the last block, 62% of the perturbation on the fp8 checkpoint, cost it nothing. That
is C5's "the recipe only shifts the curve", in two minutes. On this batch the fp4-trained model is
0.044 worse *in bf16* than the fp8-trained one and loses a further 0.012 to its own forward, so
most of the deficit lives in the weights the noisy forward trained, not in the noise at eval time,
and a static probe sees only the latter. Read the fp8-checkpoint numbers as an upper bound (they
over-predicted C5 by 2.6x and C6 by 2.2x) and probe the fp4-trained checkpoint alongside.

**Single-layer greedy selection does not work from one batch.** Excluding each of the 73 layers
alone sums to +0.236 of removal against a whole delta of +0.077: a one-layer delta is the
first-order term g·δ, whose sign belongs to the batch (block 0's `mlp.c_fc` reads as *helping*
fp4). A greedy top-12 by removal per unit predicted −0.018 and landed at +0.033 — no better than
every `attn.c_proj` (+0.032) or the last block (+0.029). Rank by type or by block, never by layer.

What it leaves: every forward lever a probe can price is priced — rotation (B0), SR (diverges),
selection (diffuse, and adapted away) — and the forward quantizer is already a per-block min-MSE
choice. What remains is training dynamics, which only arms answer, and the arm that decides
anything is C3 at d24.

## Queue

Everything below is priced from *Where the time goes* on a 3,639.8 ms step. Two standing warnings
from `perf-log.md`: **marginals do not add** (items that attack the same work are partly
substitutes, so a stack of "+3%"s is not +12%), and **a sub-2% wall-clock delta is not evidence**
unless it is a batched-arm A/B (`scripts/arm_batch.sh`, reference repeated).

### A. No numerics gate — land these first

| #          | item                                                                                                                                                              | measured price                                                                                                                                                                               | depends on | size |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|------|
| ~~**A1**~~ | ~~cuBLASLt path for the three fp4 GEMMs~~ — **done 2026-08-18, `--nvfp4-lt-gemm`**                                                                                | **1.001x on the GEMM row, 0.0% end to end** — one kernel, no algorithm to choose. It lands as the seam only, and A2/A3 are unblocked                                                         | —          | L    |
| ~~**A2**~~ | ~~per-tensor scale as device-pointer alpha~~ — **done 2026-08-18, `--nvfp4-epilogue-alpha`**                                                                      | **+0.35%** (0.07% spread) and one bf16 rounding fewer. The `[M, N]` pass it "deletes" turned out to be fused already; what it wins is ~27 ms of ALU, less 6.9 ms of new alpha products       | A1         | S    |
| **A2b**    | **Produce `a_ts * b_ts` without a kernel per GEMM** — have a quantize kernel store the product it already has the inputs for                                      | **6.9 ms/step and 5,808 launches**, i.e. roughly doubles A2                                                                                                                                  | A2         | S    |
| ~~**A3**~~ | ~~fused wgrad accumulation, `beta=1` into an fp32 `main_grad`~~ — **done 2026-08-18, `--nvfp4-fuse-wgrad`**                                                       | **+4.07%** (0.015% spread), −166 MiB peak. The `CUDAFunctor_add<float>` row goes 135.6 → 0.4 ms, its widening casts take another 34.8, and the fp32 epilogue gives 45.0 back on the GEMM row | A1         | M    |
| **A4**     | **QKV quantize dedup** — `c_q`/`c_k`/`c_v` quantize the same `x` three times, and requantize it three times in the backward                                       | 2 of 3 forward quantizes and 2 of 3 backward requants per attention block                                                                                                                    | —          | M    |
| **A5**     | **`rht128_requant` occupancy** — 38.8% SM against 32.7% DRAM is idle on both roofs                                                                                | 104.2 ms/step (2.9%)                                                                                                                                                                         | —          | M    |
| **A6**     | **Launcher hygiene** — current stream instead of the legacy default, hoist the per-call `cudaGetDevice`/`cudaFuncSetAttribute`/occupancy query, `cudaMemsetAsync` | 296.7 ms/step of CPU, but **0% of wall clock today** (GPU 98.73% busy). Do it for CUDA-graph capture and future headroom, not for throughput                                                 | —          | S    |

A1 bought nothing on its own (*The GEMM headroom is not an algorithm choice*) and A2, built on it,
bought +0.35% — but the seam they laid is what A3 then cashed for **+4.07%**, five times what
`perf-log.md` experiment 14 measured for the same change under fp8. Chained, the `--nvfp4-lt-gemm
--nvfp4-epilogue-alpha --nvfp4-fuse-wgrad` stack is +4.4% over plain `--nvfp4` — each delta was
measured on top of the previous one, so the product is fair, but the stack has never been run as a
single arm (queue D3). `--nvfp4` now turns all of it on, so that stack *is* the NVFP4 path and
the `--no-` forms are how the reference arms get built. The two estimates that fed this queue failed in
opposite directions and for the same reason: **a queue price names one kernel, and the op you are
replacing is usually several.** A2's launch-count inference was 10x optimistic; A3's measured
135.7 ms row came in 3% low, because the widening casts and the new epilogue traffic were both
missing from it and happen to nearly cancel. A4 is **bit-identical** in its cheap
form (quantize `x` once, hand the result to all three Linears); the fused `[M, 3N]` GEMM is a
follow-on that shares one per-tensor scale and is therefore not.

### B. Numerics gate — bigger, and they need C

| #          | item                                                                                                                                                                                                                                                                                                                                                                         | measured price                                                                                                                                                                                 | depends on | size  |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------|
| ~~**B1**~~ | ~~**Delayed scaling** (TE's amax history), replacing the `vector_norm` pre-pass~~ — **built 2026-08-31, `--nvfp4-scaling delayed`**                                                                                                                                                                                                                                          | ≤148 ms/step (4.1%) direct, **still not measured end to end** — but C6 measured the *fp8* counterpart `--fp8-scaling delayed` as free at this horizon, which raises the prior; B2/B4 unblocked | —          | M     |
| **B2**     | **Fuse the quantize into its producer** so `x` is never re-read in bf16                                                                                                                                                                                                                                                                                                      | the ~68 ms glue gap against fp8; both quantize kernels are at 76-85% DRAM, so bytes are the currency                                                                                           | B1         | L     |
| ~~**B0**~~ | ~~**Hadamard-rotate the *forward***~~ — **ruled out 2026-09-03, before it was built.** The rotation cancels across the forward GEMM, but on real activations it *loses* 0.06–0.08 effective bits (0.24 in the `rht128_quant_eden` form the row proposed), worst at exactly the outlier-heavy layers it was meant to help. It also does not cancel past the forward — see *B0: the rotated forward is a loss* | the probe cost 2 minutes | — | — |
| **B3**     | **Hold the RHT sign pattern across the grad-accum window**, re-randomizing per optimizer step                                                                                                                                                                                                                                                                                | makes `rht128_requant(w)` cacheable — the withdrawn "+21%" claim in *Where the speed comes from*                                                                                               | —          | **S** |
| **B4**     | **Fold the eden scratch round-trip** (bf16 scales written, read back, rewritten as fp8)                                                                                                                                                                                                                                                                                      | 35.3 ms + 7,744 launches + most of A6's `cudaMemset`                                                                                                                                           | B1         | M     |

B1 was **Python-only** as predicted: `four_six_fp4_kernel` takes the amax as a read-only device
pointer and `quant_fp4` already accepted an `amax=` override nothing passed, so it is
`nanochat/sm120/nvfp4_state.py` plus one argument through `_NVFP4Matmul.forward`, with no CUDA
touched. `DelayedScaleState` came from the fp8 port and now lives in `sm120/delayed_scale.py`,
generalised over roles and constants. See *Delayed activation scaling* below for what it does and
what is still unmeasured. B3 is the cheapest item on either list and has the largest unmeasured
upside — do it early, but it *is* a change to the gradient estimator.

### C. The shipping gate, which nothing above clears

| #          | item                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | why    | size |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|------|
| ~~**C1**~~ | ~~Eight paired seeds, `--nvfp4` against `--fp8`~~ — **done 2026-08-19: clears, weakly.** +0.005866 ± 0.009610, 5+/3−, leave-one-out robust. See *Numerics: C1* above                                                                                                                                                                                                                                                                                                                                                                                                 | 2.0 h  |
| ~~**C2**~~ | ~~A horizon run at the real token budget~~ — **done 2026-08-19: +0.0137 bpb at ratio 12**, both seeds agreeing to 7e-5, monotone and still widening. But 19.9% faster makes it **~break-even at equal wall clock**, not experiment 25's 4x loss. See *Numerics: C2*                                                                                                                                                                                                                                                                                                  | 5.5 h  |
| **C3**     | **Re-scope against C6's faster fp8 before running this.** At d12 the margin it was meant to price is gone (1.015x), so the experiment is now a d24 one. **The equal-wall-clock pair — the decision experiment.** `--nvfp4 --num-iterations 3020` against `--fp8 --num-iterations 2520`, same wall clock to ~1%. C2's break-even verdict rests on extrapolating a decaying slope 500 steps; this measures it. Note the LR schedule derives from `num_iterations`, so the nvfp4 arm is a *different* schedule, which is the honest comparison but not a controlled one | ~3.5 h |
| **C4**     | **CORE at more seeds.** C2 left CORE unresolved — the two deltas disagree by 0.0213 against an nvfp4 seed spread of 0.0171. It is the only axis that can still move C2's verdict, and 2 seeds cannot see it                                                                                                                                                                                                                                                                                                                                                          | ~3.5 h |
| ~~**C5**~~ | ~~Does NVIDIA's recipe close C2's deficit?~~ — **done 2026-09-03: no, it closes 17%.** `--nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd` leaves **+0.0114** against fp8 and shifts the curve without flattening it, so the d12 cost is block forward noise, not the head. `--warmdown-ratio 0.2` is a loss at this horizon for both precisions. See *Numerics: C5*                                                                                                                                                                                         | 8.7 h  |
| ~~**C6**~~ | ~~Block exclusion, and can fp8 go faster?~~ — **done 2026-09-03.** `--fp8-scaling delayed --wgrad-nt` is **free** (+0.000395 bpb, −15% wall clock), which **erases NVFP4's speed advantage at d12** and retires C2's break-even verdict. `--nvfp4-bf16-blocks 2,4` closes 37% cumulatively and is the first lever to *attenuate* the deficit's growth rather than offset it. See *Numerics: C6*                                                                                                                                                                      | 3.3 h  |
| ~~**C7**~~ | ~~Forward noise or backward bias?~~ — **done 2026-09-03: backward bias is refuted.** `--nvfp4-bwd-source bf16` is the **worst arm on record** (+0.0200, growth +0.0126 against plain's +0.0138/+0.0076): the unbiased backward costs 4x what the lm_head recipe gains. TE's double-quantization rationale does not hold, and the probe's gradient columns anti-predict bpb. See *Numerics: C7*                                                                                                                                                                       | 1.6 h  |
| ~~**C8**~~ | ~~The forward main effect, cleanly.~~ — **done 2026-09-03: the fp4 forward is the deficit.** All 73 Linears in `fp8-fwd` against C7's identical backward isolates forward precision: **−0.0132**, monotone and still widening, i.e. more than the whole plain deficit. Lands at +0.0067 while carrying C7's harmful backward. **Promotes the Hadamard-rotated forward ahead of stochastic rounding.** See *Numerics: C8*                                                                                                                                             | 1.7 h  |

C1, C2 (2026-08-19), C5-C8 (2026-09-03) are run; **C3 and C4 are what is left of the original
gate, and C6 has changed what C3 would measure.** The rule below
still governs anything from B that lands next — start a fresh battery as soon as the first
numerics-affecting item does, not after all of them, since a failed battery over a stack of four
changes does not say which one failed. Run it at
**d12/dbs 8/2 GPU/100 steps**, the config experiment 18 calibrated its null on and the only one
C2 is affordable at (ratio 12 is 1.32B tokens at d12 against 3.48B at d20).

**Both blockers found while scoping C1 are now cleared — 2026-08-18.**

- ~~`--seed` does not exist on this branch~~ — **ported** from `refactor` verbatim, so the two
  branches do not drift: `compute_init(device_type="cuda", seed=42)` (`common.py:174`) feeding
  `torch.manual_seed`/`torch.cuda.manual_seed`, and `--seed` in `base_train.py` at the
  `compute_init` call. Default 42, so every earlier run reproduces unchanged.
- **The NVFP4 arm is *not* run-to-run reproducible, and `--fp8` is.** Settled by five 3-step
  d12/dbs 8/2-GPU arms (`dev-ignore/determ-check/`):

| pair                   | step 0      | step 1      | step 2                 |                     |
|------------------------|-------------|-------------|------------------------|---------------------|
| `--nvfp4 --seed 42` x2 | 10.397527 = | 10.384410 = | 10.362745 vs 10.362743 | **diverges**        |
| `--fp8 --seed 42` x2   | 10.397516 = | 10.384375 = | 10.362544 =            | deterministic       |
| `--nvfp4 --seed 43`    | 10.396896 ≠ | —           | —                      | seed control passes |

*The weight cache*'s claim stands: the backward's fresh Hadamard rotation and EDEN seeds are not
reproducible, notwithstanding that `new_seed()` is a `torch.randint` off the CPU RNG
`compute_init` seeds. The fp8 pair is the control that makes this NVFP4-specific rather than
general nondeterminism — same DDP/NCCL path, identical to 6 dp. **Three steps is the minimum that
sees it**: step 0 is a pure forward and step 1 still agrees exactly, so a shorter check returns a
false "deterministic".

**So C1 is 24 runs, not 16: 8 seeds x {`--fp8`, `--nvfp4`-A, `--nvfp4`-B}** (~1.8 h). The A-vs-B
deltas give the NVFP4 replicate null *at the same seeds*, and the fp8-vs-A deltas give the effect
— judged against a null calibrated on the arm that carries the noise, not experiment 18's fp8 one.
What is unknown is how 2e-6 at step 2 compounds by step 100, which is the chaotic regime
perf-log.md warns about and Muon amplifies. **If the replicate null comes back comparable to the
±0.011 seed band, C1 cannot resolve the effect at n=8 and the honest move is to go straight to
C2.**

### D. Bookkeeping

| #          | item                                                                                                                                                                                                                                                                                                                       |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ~~**D1**~~ | ~~Re-measure the *Result* table on the current tree~~ — **done 2026-08-18.** Five arms, d20/dbs 4, repeat spread 0.31%: NVFP4 is **+29.4% over `--fp8`** and **+59.8% over bf16**. The 1.25x it replaces was already post-window-fix, so the move is A1-A3, not the window. See *Result*                                   |
| ~~**D2**~~ | ~~Commit the pending `dev/nvfp4-quartet.md` and `scripts/kernel_report.py`~~ — **done**, `cf5b41f`                                                                                                                                                                                                                         |
| ~~**D3**~~ | ~~Measure the stack as one arm~~ — **done 2026-08-18**, same batch as D1: **+7.01%** (37,532 vs 35,073), against ~+6.3% from chaining the marginals, which therefore under-counted. It also priced what nothing had: **plain NVFP4 with the stack off is +21.0% over `--fp8`**, so the stack is secondary to the fp4 GEMMs |
| **D4**     | **Re-measure the d12 arms too.** D1 covered d20 only, so the original +7.0%/~+29% headline — d12/dbs 4, pre-window-fix — has no current-tree replacement. Same batch shape, ~40 min                                                                                                                                        |

### Ruled out by measurement

- **CUDA graphs / launch batching** — 98.73% GPU busy, sub-2 us gaps are 0.72% of wall clock.
- **An arithmetic rewrite of the forward cast** — 85% DRAM, already at the byte roofline. This is
  the opposite of `perf-log.md` experiment 23, where an ALU-bound cast gave up +3.9% to a
  division-free form. Do not repeat that here.
- **The unconverted bf16 layers** — 2.4 ms/step.
- **`--warmdown-ratio 0.2` (NVIDIA's 80/20 schedule)** — a loss at ratio 12 for both precisions
  (fp8 +0.0074, nvfp4 recipe +0.0089) and it widens the deficit to +0.0129. Both arms are still
  descending steeply at the last eval, so it is a schedule for a longer horizon than this one.
  See *Numerics: C5*.
- **`--nvfp4-bwd-source bf16` (TE's "quantize both orientations from the high-precision input")** —
  +0.0200 against plain's +0.0138 and a two-thirds higher growth rate. Forward/backward
  consistency beats fidelity to bf16 here. See *Numerics: C7*.
- **A Hadamard-rotated forward (queue B0)** — −0.0798 effective bits at 128-wide and −0.2425
  fused with EDEN (73 of 73 layers), monotone in rotation width across two checkpoints. NVFP4's
  per-16 block scale already isolates the outliers the rotation was meant to flatten, so spreading
  them costs. See *B0: the rotated forward is a loss*.
- **NVFP4 at d12 on wall-clock grounds** — against `--fp8-scaling delayed --wgrad-nt` the speed
  margin is 1.015x for the arm carrying the full deficit, and every configuration that narrows
  the deficit is slower than fp8. See *Numerics: C6*. This is d12-specific; d24 is untested and
  the same tables put NVFP4 at 1.21x there.

## Caveats and open work

- **`scripts/bench_nvfp4.py --backward` is not trustworthy.** It scored NVFP4 at 0.63x of bf16
  over the d12 inventory while a matched `base_train.py` batch measured it faster than both bf16
  and fp8. Forward-only rankings did match. Settle throughput with `base_train.py` arms.
- **Numerics: C1, C2 and C5-C8 are run; the gate is now C3/C4.** C1 clears weakly at 100 steps
  (+0.0059 ± 0.0096), C2 measures **+0.0137 bpb at ratio 12** with both seeds agreeing to 7e-5 —
  but at 19.9% faster that is **~break-even at equal wall clock**, not a kill. CORE is unresolved.
  The replicate null came back 58x below the seed spread, so future batteries can use 16 runs.
- ~~**Re-measure NVFP4 vs fp8 at d20/dbs 4 on the current tree**~~ — **done 2026-08-18**, and it
  is now the *Result* table: **+29.4%** over `--fp8`, up from 1.25x. The **d12** arms are still
  outstanding (queue D4).
- ~~Re-measure `lm_head` at its real width~~ — done: 0.97x at d12, 1.27x at d20, 1.42x at d24.
- ~~The weight is re-quantized every micro-step~~ — done: `--nvfp4-weight-cache`, +1.8-2.0%.
- **`lm_head` may not convert at all** if `padded_vocab_size` is not a multiple of 128 — it is
  padded to 64. This repo's tokenizer is 32,768, so it converts; swap in a tokenizer whose
  padded vocab is an odd multiple of 64 and the biggest GEMM in the model silently drops to
  bf16. The conversion prints what it left in bf16.
- **Layers whose features are not 128-aligned stay bf16**: `ve_gate` and `smear_gate`, 7 of 80
  at d12. Negligible in FLOPs.
- ~~**No side streams or CUDA graphs, and the launchers do host work per call**~~ — **measured
  2026-08-18, and it is not on the critical path.** The vendored kernels do launch on the legacy
  default stream rather than `getCurrentCUDAStream()`, every launcher re-does `cudaGetDevice` +
  `cudaFuncSetAttribute` + an occupancy query, the eden/requant paths add a `cudaMemset` (7,744
  calls, **296.7 ms/step of CPU, 13% of all API time**, to zero four bytes) and the TMA paths
  re-encode a `CUtensorMap` per call. But the GPU is **98.73% busy** at 58,573 ops/step and the
  launch-shaped idle is **0.72% of wall clock**, so this is tidiness and future-proofing, not
  throughput. The default stream still blocks CUDA-graph capture and any side-stream work.
- **Muon's Newton-Schulz is 186.8 ms/step (5.1%) at d20/dbs 4 on one GPU**, against the 0.73%
  `perf-log.md` records at d12/2 GPU — the optimizer is amortized over 16 micro-steps either way,
  but there are more parameters per rank here and the step is shorter. Not an NVFP4 item; noted
  because it is easy to misread as GEMM time (it shares a cutlass kernel family with bf16 model
  GEMMs, and `kernel_report.py` separates them by tile).
- Not combined with the `refactor` branch's fp8 stack (pinned GEMMs, delayed scaling, NT wgrad).
  Those are orthogonal and this branch does not have them — though the *Where the time goes* map
  now prices delayed scaling here at ≤148 ms/step and points at producer fusion as the largest
  item, so "orthogonal" understates it: they are the top of the queue.
