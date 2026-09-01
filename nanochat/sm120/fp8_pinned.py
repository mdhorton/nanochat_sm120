"""The natural-layout (NT) fp8 weight-gradient GEMM (`--wgrad-nt`).

`grad_weight[m,n] = go[k,m].T @ x[k,n]`. FP8 on cuBLASLt is TN-only on Ada/Hopper -- both
operands k-major -- so `nanochat/fp8.py` feeds it `go.t().contiguous()` and `_to_col_major(x)`,
two pure-copy kernels that are 4.6% of a training step. sm120's cuBLASLt accepts a K stride of
more than one, so the same product runs with both operands read exactly as they sit and the
copies never enter the graph: +8.4% at d12, +8.5% at d16, and numerically free
(+0.0001 +/- 0.0014 bpb over 8 paired seeds in the donor's experiment 18).

`torch._scaled_mm` cannot express that layout, so the GEMM goes through `csrc/pinned_gemm.cu`
instead. One plan per shape, autotuned on first use and then verified against the very
transpose-copy path it deletes before it is trusted. The NT kernel is 5-15% slower per launch
than TN -- the win is the copies, not the GEMM -- which is why the plan is keyed and autotuned
per layout rather than replayed from the TN pick.

A plan that fails verification warns and falls back to the TN path for that shape. A *build*
failure raises: `--wgrad-nt` was passed explicitly, and a run that silently ignores it reads
~2.5% low while still printing its enablement line, which is how the donor's experiment 16 lost
two arms.

This is the NT slice of `sm120_nanochat@refactor:nanochat/sm120/fp8_pinned.py`; the `--pin-gemm`
surface that shares this extension (`_MODE`, `mm`, `_build_plan`) is not ported -- see TODO.md.
`csrc/pinned_gemm.cu` is that fork's file verbatim, so it still carries the `accum` and
`fast_accum` plan kinds those flags need.
"""
import os
import sys

import torch

from nanochat.sm120 import gemm_cache

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "pinned_gemm.cu")

_WGRAD_NT = False      # --wgrad-nt
_EXT = None
# (m, n, k, go_dtype, x_dtype, "nt") -> PinnedGemm or None (fall back to the TN path).
_PLANS = {}
_LOG = []              # human-readable record of what got pinned, for the caller to print


def configure_wgrad_nt(enabled):
    """Enable the natural-layout wgrad. Called once from the recipe before training starts."""
    global _WGRAD_NT
    _WGRAD_NT = bool(enabled)


def configure_plan_cache(mode, exhaustive=False, max_candidates=512):
    """Persist and reuse autotuned plans (--gemm-plan-cache), and optionally search exhaustively.

    Forwards to gemm_cache so the recipe has one import to reach the whole NT wgrad surface.
    """
    gemm_cache.configure(mode, exhaustive=exhaustive, max_candidates=max_candidates)


def wgrad_nt():
    """Trace-time predicate: `SM120Backend.wgrad` branches on this, so with it False the
    transpose copies stay in the graph and nothing here runs."""
    return _WGRAD_NT


def log_lines():
    # Cache header first, then the per-plan lines. gemm_cache returns [] when the cache is off,
    # which keeps report_once's "built no plans" warning reachable.
    return gemm_cache.log_lines() + list(_LOG)


class ExtensionUnavailable(RuntimeError):
    """The JIT extension could not be built, so the NT wgrad cannot run at all.

    Raised rather than silently falling back to `_scaled_mm`: --wgrad-nt was passed explicitly,
    and a run that ignores it reads ~2.5% low while still printing its enablement line. A
    per-shape rejection is different -- that is a numerical guard doing its job, and it warns
    and falls back.
    """


def _warn(line):
    """Surface a plan failure the moment it happens, not only into _LOG.

    _LOG is drained by `recipe.PerfStack.report_once` after the first backward; a harness that
    never calls it would otherwise see a failure nowhere at all.
    """
    _LOG.append(line)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"WARNING: pinned GEMM {line.strip()}", file=sys.stderr, flush=True)


def _cuda_home():
    """The toolkit to build against: the newest one installed, not the one torch was built with.

    Deliberately NOT `quartet.ext.resolve_cuda_home()`, which matches torch's CUDA *major* and
    would pick 12.8 here. That matters because this extension links its own cuBLASLt: under 12.x
    every fp8 GEMM lands on `sm89_xmma_*` -- Ada kernels on a Blackwell card -- instead of
    `nvjet_sm120_*`. The JIT path does not run torch's major-mismatch check, so a 13.x toolkit
    against a cu12 torch builds fine, and is the configuration the +8.4% was measured in.

    `/usr/local/cuda` is what update-alternatives points at the highest-priority toolkit, which
    is also what a bare `cpp_extension.load()` would have used before quartet mutated the global.
    """
    from torch.utils import cpp_extension

    for path in ("/usr/local/cuda", cpp_extension.CUDA_HOME):
        if path and os.path.exists(os.path.join(path, "bin", "nvcc")):
            return path
    return cpp_extension.CUDA_HOME


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load

        from nanochat.common import get_base_dir
        from nanochat.sm120.quartet.ext import _nvcc_version

        cuda_home = _cuda_home()
        nvcc = _nvcc_version(cuda_home)
        # Keyed on the *toolkit*, not on torch.version.cuda the way the quartet and fp4_gemm
        # extensions are: those link torch's already-loaded cuBLASLt, this one links the
        # toolkit's, so the toolkit is what makes two build directories different.
        toolkit = f"nvcc{nvcc[0]}.{nvcc[1]}" if nvcc else "unknown"
        tag = f"pinnedgemm-torch{torch.__version__.split('+')[0]}-{toolkit}"
        build_dir = os.path.join(get_base_dir(), "torch_extensions", tag)
        os.makedirs(build_dir, exist_ok=True)
        # quartet.ext.load() and fp4_gemm._ext() both set these globally and never put them
        # back; neither runs under --fp8, but a leaked 12.8 would silently undo the choice above.
        saved_home, saved_env = cpp_extension.CUDA_HOME, os.environ.get("CUDA_HOME")
        cpp_extension.CUDA_HOME = cuda_home
        os.environ["CUDA_HOME"] = cuda_home
        try:
            # -std=c++20 in both: torch 2.9.1's ATen/core/List_inl.h does not compile under
            # GCC 13 at C++17. The rpath pins the toolkit whose cuBLASLt this was verified
            # against -- /usr/local/cuda is a symlink an OS update can repoint under us.
            _EXT = load(name="pinned_gemm", sources=[_SRC], build_directory=build_dir,
                        extra_cflags=["-O3", "-std=c++20"],
                        extra_cuda_cflags=["-O3", "-std=c++20"],
                        extra_ldflags=["-lcublasLt", f"-Wl,-rpath,{cuda_home}/lib64"],
                        verbose=os.environ.get("NANOCHAT_VERBOSE_BUILD") == "1")
        except Exception as e:
            raise ExtensionUnavailable(
                f"could not build the pinned_gemm extension ({type(e).__name__}: {e}). "
                "--wgrad-nt cannot work without it; drop the flag to run the wgrad on "
                "_scaled_mm, or fix the build (a missing setuptools in a fresh venv is enough)."
            ) from e
        finally:
            cpp_extension.CUDA_HOME = saved_home
            if saved_env is None:
                os.environ.pop("CUDA_HOME", None)
            else:
                os.environ["CUDA_HOME"] = saved_env
        # The cache keys on the cuBLASLt this extension actually linked -- the toolkit's, not
        # torch's -- so it can only learn its identity once the build has succeeded. Injected
        # this way round because gemm_cache importing fp8_pinned back would be a cycle.
        gemm_cache.set_env(lt_version=_EXT.cublaslt_version(), ext_tag=tag)
    return _EXT


# Autotune budget. The screening pass over every candidate doubles as the soak that takes the
# card off its cold boost; only the finalists are then scored round-robin with alternating order.
_MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS = 64, 20, 8, 5
# A restored plan skips phase 1 and therefore skips the soak; this stands in for it.
_SOAK_ITERS = 200
# How far a restored plan may re-time above what was stored before it is worth saying so.
_STALE_US_RATIO = 1.15


def _fallback(a, b, a_scale, b_scale, out_dtype=torch.bfloat16):
    return torch._scaled_mm(a, b, scale_a=a_scale, scale_b=b_scale,
                            out_dtype=out_dtype, use_fast_accum=False)


def _tn_operands(go, x):
    """The transpose copies the TN form needs -- the thing this module exists to delete."""
    return go.t().contiguous(), x.t().contiguous().t()


def _time_fn(fn, iters=_ITERS):
    for _ in range(3):
        fn()
    t0, t1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    t1.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters


def _tuned_provenance(info):
    """Which search produced this pick. Keyed on the requested tier, not on what enumeration
    happened to return, so it always matches the tier the cache stores."""
    if gemm_cache.exhaustive():
        return (f"TUNED exhaustive ({int(info.get('kept', 0))}/"
                f"{int(info.get('enumerated', 0))} cand, "
                f"{info.get('tune_ms', 0.0) / 1000.0:.1f} s)")
    return f"TUNED heuristic ({int(info.get('heuristic_count', 0))} cand)"


def _build_wgrad_nt_plan(go, x, go_scale, x_scale):
    """Get an NT wgrad plan -- restored from the cache when one fits, otherwise autotuned -- and
    verify it against the TN path it replaces.

    The reference is built through the same transpose copies the plan exists to delete -- a
    startup-only cost. ref_us times those copies too, because they are part of the op being
    replaced: the fair comparison is (t() + _scaled_mm) against the NT launch alone.

    At most two passes: a cached plan that will not restore, or that fails verification, warns
    and falls through to a fresh tune exactly once, so a poisoned entry cannot loop.
    """
    k, m = go.shape                    # k = tokens, m = out_features (weight rows)
    n = x.shape[1]                     # n = in_features (weight cols)
    cache_key = gemm_cache.plan_key(m, n, k, go.dtype, x.dtype, "nt")
    want_budget = gemm_cache.budget(_MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS)

    for allow_cache in (True, False):
        plan = _ext().PinnedGemm(m, n, k, go.dtype, x.dtype, False, accum=False, nt=True)
        entry = gemm_cache.lookup(cache_key, want_budget) if allow_cache else None

        if entry is not None:
            cfg = {name: float(v) for name, v in entry["algo"].items()}
            if not plan.restore(go, x, go_scale, x_scale, cfg):
                _warn(f"  {m}x{n}x{k} nt: STALE cache entry (algo "
                      f"{entry['algo'].get('algo_id')} would not restore), re-tuning")
                gemm_cache.invalidate(cache_key, "restore")
                continue
            # A hit skips phase 1, which was also the soak that takes the card off its cold
            # boost. Without a stand-in, ref_us below would be measured on a colder card than a
            # tuned run's and the printed ratio would not be comparable between the two.
            #
            # plan.time, not _time_fn(plan.run): the tuner timed a bare launch loop in C++, and
            # run() adds an output allocation and a pybind crossing per call. Comparing the two
            # would read a fixed overhead as a stale plan on every small shape.
            plan.time(go, x, go_scale, x_scale, _SOAK_ITERS)
            info = dict(entry)
            info.update(cfg)
            info["us"] = plan.time(go, x, go_scale, x_scale, _ITERS)
        else:
            info = plan.autotune(
                go, x, go_scale, x_scale, _MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS,
                gemm_cache.max_candidates() if gemm_cache.exhaustive() else 0)

        ref = _fallback(*_tn_operands(go, x), go_scale, x_scale)
        got = plan.run(go, x, go_scale, x_scale)
        scale = ref.abs().max().clamp(min=1e-6)
        err = (got.float() - ref.float()).abs().max() / scale
        if got.shape != ref.shape or not torch.isfinite(err) or err > 0.05:
            if entry is not None:
                _warn(f"  {m}x{n}x{k} nt: CACHED plan FAILED verification, max rel err "
                      f"{float(err):.4g}, re-tuning")
                gemm_cache.invalidate(cache_key, "verify")
                continue
            _warn(f"  {m}x{n}x{k} nt: REJECTED, max rel err {float(err):.4g} vs t()+_scaled_mm")
            return None

        ref_us = _time_fn(lambda: _fallback(*_tn_operands(go, x), go_scale, x_scale))
        vs_ref = ref_us / info["us"] if info["us"] > 0 else 0.0
        if entry is None:
            gemm_cache.store(cache_key, plan.config(), info, want_budget)
            prov = _tuned_provenance(info)
        else:
            prov = (f"CACHED {entry['tier']}, tuned {entry['tuned_at']} "
                    f"(stored {entry['us']:.1f} us)")
            # Numerics cannot see a *different but valid* restored kernel -- a wrong tile is
            # still arithmetically correct. A large timing gap is the only signal that one came
            # back, or that the card is in a different clock state than when this was tuned.
            if info["us"] > entry["us"] * _STALE_US_RATIO:
                _warn(f"  {m}x{n}x{k} nt: cached plan re-times {info['us']:.1f} us vs "
                      f"{entry['us']:.1f} us stored -- restored kernel may differ, or the card "
                      f"is in a different clock state")
        _LOG.append(f"  {m}x{n}x{k} nt: algo {int(info['algo_id'])} tile {int(info['tile'])} "
                    f"splitK {int(info['splitk'])} | t()+_scaled_mm {ref_us:.1f} -> "
                    f"{info['us']:.1f} us (vs_ref {vs_ref:.2f}x) | err {float(err):.2g} | {prov}")
        return plan
    return None


# An opaque custom op, not a plain function: `_Float8Matmul` is `allow_in_graph`, so AOTAutograd
# traces its *backward* too and the plan builder would be handed FakeTensors ("we are erroneously
# tracing into a custom kernel"). Everything that touches the extension lives behind this
# boundary; the shape decisions stay outside, where they are compile-time constants.
@torch.library.custom_op("nanochat::pinned_wgrad_nt", mutates_args=())
def pinned_wgrad_nt(go: torch.Tensor, x: torch.Tensor, go_scale: torch.Tensor,
                    x_scale: torch.Tensor) -> torch.Tensor:
    k, m = go.shape
    n = x.shape[1]
    key = (m, n, k, go.dtype, x.dtype, "nt")
    if key not in _PLANS:
        try:
            _PLANS[key] = _build_wgrad_nt_plan(go, x, go_scale, x_scale)
        except ExtensionUnavailable:  # nothing can be pinned at all -- do not fall back silently
            raise
        except Exception as e:  # a bad plan must never take training down with it
            _warn(f"  {m}x{n}x{k} nt: FAILED to build ({type(e).__name__}: {e})")
            _PLANS[key] = None
    plan = _PLANS[key]
    if plan is None:
        return _fallback(*_tn_operands(go, x), go_scale, x_scale)
    return plan.run(go, x, go_scale, x_scale)


@pinned_wgrad_nt.register_fake
def _(go, x, go_scale, x_scale):
    return go.new_empty((go.shape[1], x.shape[1]), dtype=torch.bfloat16)


def mm_wgrad_nt(go, x, go_scale, x_scale, out_dtype):
    """grad_weight = go.T @ x with both operands in their natural layout (--wgrad-nt)."""
    if out_dtype != torch.bfloat16:  # the pinned kernel only produces bf16
        return _fallback(*_tn_operands(go, x), go_scale, x_scale, out_dtype)
    return torch.ops.nanochat.pinned_wgrad_nt(go, x, go_scale, x_scale)
