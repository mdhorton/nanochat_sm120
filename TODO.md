# TODO

## The rest of the sm120 fp8 stack

`--fp8-scaling` and `--wgrad-nt` are ported
(`nanochat/sm120/{recipe,fp8_state,fp8_backend,fp8_pinned}.py`, `csrc/pinned_gemm.cu`). Four more
flags exist in the sibling fork `/remote/projects/pycharm/sm120_nanochat`, branch **`refactor`** —
read them with `git -C /remote/projects/pycharm/sm120_nanochat show refactor:<path>`. Its
`dev/perf-log.md` and `dev/perf-log-experiments.md` carry the measurements below.

| flag                 | d12       | d16   | needs | notes                                            |
|----------------------|-----------|-------|-------|--------------------------------------------------|
| `--pin-gemm all`     | +6.0%     | +0.8% | —     | ext now in-tree; −512 MiB; collapses with depth  |
| `--fp8-weight-cache` | +1.0%     | +1.5% | —     | Python-only; +218–448 MiB                        |
| `--fuse-wgrad-accum` | +0.7–1.9% | +1.4% | —     | ext now in-tree; TE's `fuse_wgrad_accumulation`  |
| `--muon-autotune`    | +0.7%     | +0.7% | —     | precision-independent; `--nvfp4` can take it too |

Marginals do not add — the donor's six compound to +22.7% against +18.1% measured at d16. The
full stack measured **240,038 tok/s** at d12/dbs 8/2 GPU against 180,890 with none of it.

`fp8_state.py` was cut in half on the way in: `WeightCastCache`, `WgradAccumStore` and their
factories are the two rows above that need no new kernel. `fp8_backend.py` overrides `wgrad` and
inherits `mm_fwd`/`mm_dgrad`; the donor overrides all three, routing the other two through
`fp8_pinned.mm` for `--pin-gemm`.

### The CUDA extension, now in-tree

`nanochat/sm120/csrc/pinned_gemm.cu` is the donor's `dev/custom_gemm/pinned_gemm.cu` verbatim, so
it still carries the `accum` and `fast_accum` plan kinds `--fuse-wgrad-accum` and `--pin-gemm`
need — both are Python-only additions to `fp8_pinned.py` from here.

Its build helper (`fp8_pinned._ext`) deliberately does **not** reuse
`quartet.ext.resolve_cuda_home()`, which is major-matching and resolves torch's cu12.8. Half of
what `--pin-gemm`/`--wgrad-nt` are worth is that the extension links the *system*
`libcublasLt.so.13.6.0.2`: under 12.8 every fp8 GEMM lands on `sm89_xmma_*` (Ada kernels on a
Blackwell card) instead of `nvjet_sm120_*`. It builds against `/usr/local/cuda` (13.3) instead,
keys its build directory on the toolkit rather than on `torch.version.cuda`, and saves/restores
`CUDA_HOME` around the build because quartet and `fp4_gemm` both mutate it globally.

Two things the earlier note got wrong, both settled by building it:

- **A cu13 toolkit against a cu12 torch is fine.** `cpp_extension`'s major-mismatch `RuntimeError`
  is raised from `_check_cuda_version`, which only the *setup.py* path calls — JIT `load()` never
  does. The resulting `.so` needs `libcublasLt.so.13` and `libcudart.so.13` alongside torch's
  cu12 libs and the two runtimes coexist.
- **No RPATH is strictly needed**, since `ldconfig` already resolves `libcublasLt.so.13` through
  `/usr/local/cuda/targets/x86_64-linux/lib` — but one is set anyway, because that path is a
  symlink an OS update can repoint.

Acceptance criterion, and it is a test (`tests/test_fp8_wgrad_nt.py`):
`objdump -p <build_dir>/pinned_gemm.so | grep NEEDED` shows `libcublasLt.so.13`.

All three guards are in: `ExtensionUnavailable` raises rather than falling back (a build failure
that fell back would read ~2.5% low while still printing `✓`), `_warn` puts a per-shape rejection
on stderr, and `PerfStack.report_once` drains the plan log after the first backward and says so
when the NT wgrad is on but no plans were built. `report_once` gates on
`fp8_pinned.wgrad_nt()`, not on the flag, so `--wgrad-nt` without `--fp8` warns once and stays
quiet.

Measured here, d12/dbs 8/2 GPU, `--fp8 --fp8-scaling delayed` + `NANOCHAT_FA2_SWINDOW=1`,
one `arm_batch.sh` batch: **205,028 → 228,734 tok/s (+11.6%)**, peak 9,654 → 10,640 MiB
(+986 MiB, against the donor's +1,680 at d16), bpb 1.665911 → 1.663725. The NT arm started 10 °C
hotter than the baseline, so the gain is if anything understated. The four plan lines all report
`err 0` against the TN reference, at 4.3–6.4× on the wgrad including the transpose copies.

### The design decision, deferred

`--fp8-weight-cache` and `--fuse-wgrad-accum` each add an `after_step` hook, at which point
`scripts/base_train.py` has two sets of per-step hooks at one call site — `perf.*` beside the
inline `nvfp4_main_grads` / `refresh_nvfp4_weight_caches` calls. That is the failure mode
`recipe.py` exists to prevent. Resolve it then by **subsuming** the nvfp4 wiring into `recipe.py`
(moving it verbatim; `_apply_fp8`/`_apply_nvfp4` as separate bodies sharing one hook surface), not
by leaving both.

Build the regression net first: `tests/test_nvfp4.py` never imports `scripts.base_train`, so
nothing currently covers that wiring. Capture a CLI-contract golden — stdout, stderr and exit code
for ~10 flag combinations including all three error paths (`--nvfp4 --no-nvfp4-lt-gemm
--nvfp4-epilogue-alpha`, an orphan `--nvfp4-weight-cache`, `--nvfp4 --fp8`) — before the move and
replay it after.

`recipe.resolve()` was deliberately not ported: with `--pin-wgrad` dropped it had nothing to
normalize. Add it back when a flag needs cross-validation, and call it from `base_train` before
`user_config` is snapshotted so checkpoints record effective values.

### Not to port

- `--fp4-fwd` — killed for good: +0.0147 bpb at ratio 12 against a 3.9% saving, a ~4× net loss.
- `--pin-wgrad` — deprecated alias for a flag this repo never had.
- `--compile-mode` — neutral to −1.2%, and 122–424 s of compile.
- `cublaslt_algos.cu` — only used by an unported probe.

### The cuBLAS pin, deliberately not done

The donor recommends `[tool.uv] override-dependencies = ["nvidia-cublas==13.6.0.2"]` against a
−35% collapse it measured on cuBLAS 13.0.0.19. That version reaches a venv only through a torch
**cu130** wheel; this repo is on 2.9.1+cu128 with `nvidia-cublas-cu12`, so the pin is a cu13
package name in a cu12 graph — a no-op at best, unresolvable at worst.

The real exposure here is the mirror image: `/usr/local/cuda` is a symlink (currently 13.3, with
`libcublasLt.so.13.6.0.2`), so a system update could silently repoint what the JIT extensions link.
Defend that with a `cublaslt_version()` print and a `< 130600` warning, not dependency surgery.
**Revisit the day this repo moves to a cu130 torch**, when the pin becomes mandatory.

## NVFP4 queue, after B1

`--nvfp4-scaling delayed` landed (queue B1). What it unblocked, from `dev/nvfp4-quartet.md`:

- **B2 — fuse the quantize into its producer** so `x` is never re-read in bf16. The ~68 ms glue
  gap against fp8; `four_six_fp4_kernel` is at 85% DRAM, so bytes are the currency. Largest item
  on that list.
- **B4 — fold the eden scratch round-trip** (35.3 ms + 7,744 launches + most of A6's `cudaMemset`).
  The backward writes bf16 scratch block scales that `eden_convert_scales_kernel` reads back and
  rewrites as e4m3, purely because the per-tensor scale is not known until the pass finishes. A
  history makes it known in advance -- but unlike B1 this needs an `amax` **input** on kernels
  that have none, i.e. editing `group_transform_and_eden.cu` / `rht128_eden.cu`, which are
  vendored Quartet **verbatim @ 5f2a47e**. Different risk class from B1.
- **A2b — the forward alpha `x_ts * w_ts` is now constant across the grad-accum window** under a
  history, so its 6.9 ms/step over 5,808 launches collapses to a once-per-step multiply.
- **A4 — QKV dedup.** `c_q`/`c_k`/`c_v` quantize the same `x`, so they now also carry three
  identical histories and run three identical readback reductions. A4 collapses all of it.

Not done, and worth a line: `DelayedScaleState` has a `saturated`/`headroom` counter's worth of
information in `update()`'s `raw >= self.max` mask but does not expose it. Surfacing it would turn
the one thing that can actually go wrong here -- block-scale saturation -- from an unknown into an
observable, and a run that never saturates is one where the delayed and dynamic quantizations
agree by construction.

## Measurement harness

Not ported, and `dev/nvfp4-quartet.md` says so: `scripts/profile_train.py`, `kernel_report.py`,
`ncu_report.py`, `gpu_idle_report.py` and `nanochat/sm120/profiling.py`. Every ms/step figure in
that file and in this one was measured with them. `kernel_report.py` ranks by *headroom*
(`time_share × (1 − max(SM%,DRAM%)/100)`) as well as by time, which is what finds a tunable kernel
rather than a big one; `gpu_idle_report.py`'s docstring notes the CUDA-graph verdict was reached at
the fp8 launch count and needs re-answering for NVFP4's 1.7× launches.

Porting them would also let `PerfStack.after_backward` carry an NVTX range, and would fix two
dangling references: `scripts/arm_batch.sh`'s header cites `dev/perf-log.md`, and
`dev/nvfp4-quartet.md` cites "experiment 18/24/25" — none of which exist in this repo.

Note `recipe.add_args` deliberately omits the donor's `--profile-steps`: this repo already defines
that flag (`base_train.py:92`, paired with `--profile-start`) and argparse raises on the duplicate.
When the harness lands, have `PerfStack` read the existing flags rather than adding its own.

## Verification debt

`--fp8` on this box is **not** bit-reproducible across runs. Two runs of identical code at d12/dbs
8/2 GPU agree at steps 0–2 and then drift, reaching 7.4e-5 in the loss by step 19. The donor's
determinism table calls `--fp8` deterministic, but it checked three steps — too short to see this.
Consequences:

- A "same loss curve" gate needs a control pair to calibrate the drift; it cannot be read alone.
- The `--nvfp4` C1/C2 batteries were designed against the assumption that fp8 is the deterministic
  control arm. That assumption is weaker than recorded — worth re-reading those conclusions.
- Suspect DDP/NCCL reduction order. Untested: whether a 1-GPU run is reproducible.

## Comm/compute overlap for PCIe multi-GPU (design only, 2026-09-01)

Context: RTX PRO 6000 boxes have no NVLink, only P2P over PCIe Gen5. Data parallel is the right
architecture there (TP all-reduces activations per block; PP exists to fit models that fit
anyway), so the only lever is hiding the per-step exchange. `MuonAdamW.step` already overlaps
*within* the step (all reduces launched, then per-group wait/compute/gather), but nothing hides
behind the backward that produces the grads or the forward that consumes the params. Exposed tax
measured here: 1.3% fp8 / 1.9% nvfp4 on 2 Gen4 cards; grows ~linearly with card count at fixed
total batch, and fp4's shorter step pays more.

**Measure first.** One profile with NCCL kernels visible, to split the exchange into what the
optimizer's own overlap already hides and what sits on the critical path. Only worth building at
4+ cards.

Two independent seams:

- **Reduce-scatter during the last micro-step's backward.** Bucket by layer, not by shape: the
  shape-stacked Muon groups only complete when block 0's backward finishes. Trigger: a full
  backward hook on each `Block`, armed on the last micro-step only -- fires after the block's
  whole backward, so it covers bf16, fp8 and the fused-wgrad accumulator alike. `step()` then
  consumes per-layer futures instead of launching reduces.
- **All-gather during the next step's forward.** Issue gathers in forward order; a pre-forward
  hook on each block waits on its own future. The nvfp4 weight-cache refresh and the fp8 weight
  cast read the gathered params, so they move into the same per-block hook.

Costs and constraints:

- `torch.compile` wraps the whole GPT; hooks that wait on futures inside it graph-break or need
  compiled autograd. Compile per block, hooks in eager between blocks -- a `base_train.py` change.
- The window is the last backward: ~4% of the step at grad-accum 16 (cannot hide an 8% tax),
  ~a third at grad-accum 2-4, i.e. the large-dbs geometry 96 GB cards run. Payoff depends on
  batch geometry, not just card count.
- `WgradAccumStore.flat` must not be zeroed until every per-layer reduce has read its view.
- Fewer bytes is the other lever, smaller than overlap: Linear weights (Muon) go across as fp32
  both ways; embeddings are already bf16 (44% of d24 params). bf16 reduce-scatter of Muon grads
  is close to what Newton-Schulz already sees; the bf16 all-gather is the delicate side.
