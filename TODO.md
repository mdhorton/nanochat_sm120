# TODO

## The rest of the sm120 fp8 stack

`--fp8-scaling`, `--wgrad-nt` and `--pin-gemm` are ported
(`nanochat/sm120/{recipe,fp8_state,fp8_backend,fp8_pinned}.py`, `csrc/pinned_gemm.cu`). Three more
flags exist in the sibling fork `/remote/projects/pycharm/sm120_nanochat`, branch **`refactor`** —
read them with `git -C /remote/projects/pycharm/sm120_nanochat show refactor:<path>`. Its
`dev/perf-log.md` and `dev/perf-log-experiments.md` carry the measurements below.

| flag | d12 | d16 | needs | notes |
|---|---|---|---|---|
| `--fp8-weight-cache` | +1.0% | +1.5% | — | Python-only; +218–448 MiB |
| `--fuse-wgrad-accum` | +0.7–1.9% | +1.4% | — | ext now in-tree; TE's `fuse_wgrad_accumulation` |
| `--muon-autotune` | +0.7% | +0.7% | — | precision-independent; `--nvfp4` can take it too |

Marginals do not add — the donor's six compound to +22.7% against +18.1% measured at d16. The
full stack measured **240,038 tok/s** at d12/dbs 8/2 GPU against 180,890 with none of it.

`--pin-gemm` landed at +1.7% here, not the donor's +6.0%: on cuBLASLt 13.6 the autotuner
re-picks cuBLASLt's own first candidate on eight of eleven shapes, so what is left is the
workspace gap against `_scaled_mm` rather than a mispicked algorithm. See dev/LOG_sm120.md.
That is a reason to expect the other three marginals to read low here too.

`fp8_state.py` was cut in half on the way in: `WeightCastCache`, `WgradAccumStore` and their
factories are the two rows above that need no new kernel.

### The CUDA extension, now in-tree

`nanochat/sm120/csrc/pinned_gemm.cu` is the donor's `dev/custom_gemm/pinned_gemm.cu` verbatim, so
it still carries the beta=1 `accum` plan kind `--fuse-wgrad-accum` needs — a Python-only addition
to `fp8_pinned.py` from here (`mm_accum`, `mm_wgrad_accum_nt`, `_build_accum_plan`).

Its build helper (`fp8_pinned._ext`) deliberately does **not** reuse
`quartet.ext.resolve_cuda_home()`, which is major-matching and resolves torch's cu12.8. Half of
what `--wgrad-nt` is worth is that the extension links the *system*
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
when either flag is on but no plans were built. `report_once` gates on `fp8_pinned.enabled()`,
not on the flags, so `--pin-gemm`/`--wgrad-nt` without `--fp8` warn once and stay quiet.

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
