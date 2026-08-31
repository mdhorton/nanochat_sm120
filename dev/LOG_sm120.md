
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-31 windowed flash (nanochat/sm120/attention.py), d12 dbs8, 2x RTX Pro 4000
--fp8 --window-pattern L      186k toks/s
--fp8 SSSL (windowed flash)   195k toks/s  (+5%; was 120k via SDPA mask)
bf16: L 165k, SSSL 173k

2026-08-31 windowed flash made opt-in (NANOCHAT_FA2_WINDOWED_FLASH=1)
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
