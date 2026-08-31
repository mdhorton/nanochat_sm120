
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
