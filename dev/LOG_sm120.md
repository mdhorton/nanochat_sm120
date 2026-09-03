
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-31 windowed flash (nanochat/sm120/attention.py), d12 dbs8, 2x RTX Pro 4000
--fp8 --window-pattern L      186k toks/s
--fp8 SSSL (windowed flash)   195k toks/s  (+5%; was 120k via SDPA mask)
bf16: L 165k, SSSL 173k

2026-08-31 windowed flash made opt-in (NANOCHAT_FA2_SWINDOW=1)
Default is now upstream-faithful: no install, S layers take the SDPA mask.
The install moved out of module scope, so importing nanochat.sm120.nvfp4 (i.e. --nvfp4)
no longer enables it as a side effect. Zero changes to upstream-owned files -- base_train's
existing `_windowed_impl is None` predicates already express the switch.

2026-08-31 delayed fp8 scaling (--fp8-scaling), ported from sm120_nanochat@refactor
Its experiment 10 measures +10.7% at d12 and +0.022 bpb better than dynamic over 8 paired
seeds; neither is re-measured here yet. Landed: nanochat/sm120/{recipe,fp8_state,fp8_backend}.py,
a Float8Backend seam in nanochat/fp8.py, --seed, and 12 tests. The rest of that stack is TODO.md.

The seam is bitwise identical to the code it replaced -- same out/grad_input/grad_weight on all
four d12 GEMM shapes and through Float8Linear.forward, checked twice: against the default backend,
and against SM120Backend with no delayed state, which is what bare --fp8 now runs. So the bare arm
is still the 195k baseline. It had to be checked in eager on one GPU, because the end-to-end loss
curve cannot serve as the gate:

--fp8 at d12/dbs8/2 GPU is NOT bit-reproducible run to run. Two runs of identical code agree at
steps 0-2 and then drift, reaching 7.4e-5 in the loss by step 19. The donor's determinism table
calls --fp8 deterministic, but it compared three steps -- too short to see this. Suspect DDP
reduction order; a 1-GPU control is untried.

2026-08-31 delayed NVFP4 activation scaling (--nvfp4-scaling delayed), queue B1
quant_fp4's torch.linalg.vector_norm pre-pass over the activation is ~148 ms/step (4.1%) at d12 --
an upper bound, since Inductor folds it into producers that compute other things. The amax now
comes from a history instead. Python-only: the kernel already took `const float* amax_ptr` and
quant_fp4 already had an unused `amax=`. Landed nanochat/sm120/{delayed_scale,nvfp4_state}.py,
--nvfp4-scaling, and 12 tests. Not measured end to end yet.

The reading comes off the e4m3 block scales, not the tensor: block_scale = e4m3(group_max /
(amax_assumed * inv_scales_max)), so amax = max(block_scales) * amax_assumed * inv_scales_max --
1/32 the bytes. Biased high by <=1.5x (4/6 picks per group between inv_val 1/6 and 1/4 and that
choice does not cancel), which is the safe direction.

Why this is a weaker perturbation than the fp8 equivalent, measured rather than argued: the fp4
codes are x/(block_scale*scale) and block_scale*scale ~= group_max*inv_val_max, so the assumed
amax cancels -- it survives only in which e4m3 bucket each block scale rounds into. Error against
the fp32 product, one layer, warmed history: dynamic 0.12329, delayed 0.12357, and delayed is flat
to 0.1236 across margin 1.0 -> 4.0. Different rounding, not worse rounding. That is one layer's
forward, not 100 steps of training dynamics, so it does not replace the bpb battery.

The corridor has two walls and only one is guarded: too low and block scales clip at 448 (the
search handles it), too high and they flush to e4m3 zero, silently zeroing a whole group of 16.
The knee is ~256x, so the seed is 100 here rather than fp8's 1e3, which sits at it.

2026-09-02 NVFP4 numerics levers from the Transformer Engine recipe (branch nvfp4-numerics)
Four opt-in flags in nanochat/sm120/nvfp4_numerics.py, all needing --nvfp4 and defaulting to the
Quartet-II behaviour: --nvfp4-exclude / --nvfp4-bf16-blocks (bf16 layer selection),
--nvfp4-bwd-source bf16 (quantize W^T/X^T from the bf16 source instead of the saved fp4),
--nvfp4-rht wgrad|none (TE's RHT placement), --nvfp4-weight-2d (16x16 weight blocks shared by
fprop and dgrad, built in torch once per step in the weight cache). Plus
scripts/probe_nvfp4_numerics.py (one batch of a trained checkpoint, minutes) and 40 tests.
Python only; no CUDA touched. Details and the probe tables: dev/nvfp4-transformer-engine.md.

What the probe says at d12: lm_head is the noise (excluding it removes 45% of the loss delta and
takes the gradient from 0.84 to 0.98 cosine); the backward levers do not move model-level
numbers; the default fp4-source backward is biased against the true gradient and
--nvfp4-bwd-source bf16 is the only unbiased one; 2D weights are a net loss on this batch.

Two things learned the hard way: the rht128 kernels store rotated groups in a permuted order, so
a GEMM must take both operands from one kernel family (mixed = cosine ~0.1, no error); and
EDEN's unbiasedness *is* the random rotation -- unrotated EDEN plateaus, so --nvfp4-rht wgrad/none
are biased-dgrad ablations until a value-level SR kernel exists (no cvt.rs on sm_120).

Costs, unmeasured: bwd-source bf16 saves x in bf16 (+2 B/elem, ~+3 GB at d12/dbs 8); rht wgrad/none
add transpose copies and amax passes; weight-2d runs a torch quantizer per optimizer step.
bpb unmeasured for every lever. README untouched.

2026-09-02 --nvfp4-exclude-precision fp8
Excluding lm_head in bf16 cost 6% tok/s at d12/2 GPU (234k -> 220k); lm_head is the largest GEMM and
fp4 vs bf16 on it is a real win even at d12. The excluded layers can now run as tensorwise
Float8Linear instead: same forward-noise removal on the probe (+0.0422 vs +0.0419 loss delta),
gradient cosine 0.93 vs 0.98 (bf16) vs 0.84 (all-fp4). Not --fp8: that flag converts the whole
model and stays mutually exclusive with --nvfp4. tok/s and bpb unmeasured.

2026-09-02 --nvfp4-exclude-precision fp8-fwd (nanochat/sm120/nvfp4_fp8fwd.py)
fp8 for lm_head stayed at 220k: at the lm_head shape the fp8 backward is as slow as bf16's (the
[tokens, vocab] gradient's amax/cast/transpose passes eat the GEMM saving) and the backward is 2/3
of the layer's FLOPs. Per micro-step, compiled: all-fp4 15.6 ms, fp8 22.1, bf16 24.8, fp8+NT
wgrad+delayed 18.4, fp8 forward over the NVFP4 backward 17.9. The last is now a flag value: fp8
forward GEMM (weight cached per step), Quartet backward from the bf16 operands, so unbiased. Probe:
same loss delta as fp8 (+0.0422 vs +0.0768 all-fp4). 48 tests. tok/s and bpb unmeasured.

2026-09-03 B0 (Hadamard-rotated forward) ruled out by probe, before it was built
scripts/probe_fwd_rht.py: rotating both forward operands loses 0.06-0.08 effective bits on real
activations, worst on mlp.c_proj and lm_head, whose sparse spiky inputs the per-16 block scale
already quantizes best. The fused-EDEN form loses 0.24, of which 0.17 is EDEN's stochastic scale
against 4/6 and not the rotation. No width from 2 to 128 pays. The rotation also would not cancel
past the forward GEMM (rht128_requant rotates the other axis). Two tests pin the kernel facts it
rests on. Value-level SR is a backward lever only -- NVIDIA's ablation (2509.25149) has SR on
forward tensors diverging -- so it is not the forward candidate the notes had queued.
probe_nvfp4_numerics gained fp8fwd-<type> variants. On the fp4-trained checkpoint the residual
forward noise is +0.012 against +0.077 on the fp8-trained one, and excluding lm_head or the last
block closes nothing there: the model adapted, and most of the deficit is in the trained weights,
which a static probe cannot see. Single-layer greedy selection from one batch is first-order noise
(removals sum to 3x the whole). Details: dev/nvfp4-quartet.md, *B0* and *Forward noise by layer type*.
