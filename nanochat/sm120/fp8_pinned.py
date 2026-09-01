"""Pinned cuBLASLt algorithms for the fp8 GEMMs: `--pin-gemm` and `--wgrad-nt`.

Two independent flags over one JIT extension, `csrc/pinned_gemm.cu`. `--pin-gemm` changes which
algorithm runs; `--wgrad-nt` changes the layout the wgrad operands are read in. Either is
useful without the other.

## --pin-gemm {off,attn,wgrad,all}

cuBLAS picks one algorithm per shape from its heuristic and `torch._scaled_mm` gives no way to
ask for another. The donor's experiment 7 measures that pick losing 15-42% on every wgrad shape
and 5-25% on fwd/dgrad. Each plan is autotuned on the first call for its shape -- enumerate the
candidates, time them, keep the fastest -- then verified against `_scaled_mm` before it is
trusted, so the algorithm is always chosen for the exact descriptors training uses rather than
replayed from a recorded id.

`attn` pins only the attention wgrad (768x768x16384 at d12), the one whose win is per-cycle
efficiency rather than clock; `wgrad` pins every wgrad shape; `all` adds fwd and dgrad and is
what the donor's +6.0% at d12 was measured with. The win collapses with depth: +0.8% at d16.

## --wgrad-nt

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

`csrc/pinned_gemm.cu` is `sm120_nanochat@refactor:dev/custom_gemm/pinned_gemm.cu` verbatim, so
it also carries the beta=1 `accum` plan kind `--fuse-wgrad-accum` needs; that flag's surface
(`mm_accum`, `mm_wgrad_accum_nt`) is not ported -- see TODO.md.
"""
import os
import sys

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "pinned_gemm.cu")

_MODE = "off"          # --pin-gemm: off | attn | wgrad | all
_WGRAD_NT = False      # --wgrad-nt: the layout axis, independent of _MODE
_EXT = None
# (m, n, k, a_dtype, b_dtype, last) -> PinnedGemm or None (fall back). `last` is the layout
# axis of the key: True/False (fast_accum, TN operands) or "nt".
_PLANS = {}
_LOG = []              # human-readable record of what got pinned, for the caller to print

# fwd is the only role that fast-accumulates, matching `Float8Backend.mm_fwd`. Kept here rather
# than passed in so the two stay in one place: a plan is tuned for a fast_accum setting, and
# tuning one setting then launching the other would pin an algorithm chosen for a different op.
_FAST_ACCUM = {"fwd": True, "dgrad": False, "wgrad": False}


def configure(mode):
    """Set the pinning mode. Called once from the recipe before training starts."""
    global _MODE
    assert mode in ("off", "attn", "wgrad", "all"), mode
    _MODE = mode


def configure_wgrad_nt(enabled):
    """Enable the natural-layout wgrad. Called once from the recipe before training starts."""
    global _WGRAD_NT
    _WGRAD_NT = bool(enabled)


def wgrad_nt():
    """Trace-time predicate: `SM120Backend.wgrad` branches on this, so with it False the
    transpose copies stay in the graph and nothing here runs."""
    return _WGRAD_NT


def enabled():
    """True when anything here is live -- what `PerfStack.report_once` gates its report on."""
    return _MODE != "off" or _WGRAD_NT


def log_lines():
    return list(_LOG)


class ExtensionUnavailable(RuntimeError):
    """The JIT extension could not be built, so nothing here can run at all.

    Raised rather than silently falling back to `_scaled_mm`: --pin-gemm/--wgrad-nt were passed
    explicitly, and a run that ignores them reads ~2.5% low while still printing its enablement
    line. A per-shape rejection is different -- that is a numerical guard doing its job, and it
    warns and falls back.
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
                "--pin-gemm and --wgrad-nt cannot work without it; drop the flags to run on "
                "_scaled_mm, or fix the build (a missing setuptools in a fresh venv is enough)."
            ) from e
        finally:
            cpp_extension.CUDA_HOME = saved_home
            if saved_env is None:
                os.environ.pop("CUDA_HOME", None)
            else:
                os.environ["CUDA_HOME"] = saved_env
    return _EXT


# Autotune budget. The screening pass over every candidate doubles as the soak that takes the
# card off its cold boost; only the finalists are then scored round-robin with alternating order.
_MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS = 64, 20, 8, 5


def _fallback(a, b, a_scale, b_scale, out_dtype=torch.bfloat16, fast_accum=False):
    return torch._scaled_mm(a, b, scale_a=a_scale, scale_b=b_scale,
                            out_dtype=out_dtype, use_fast_accum=fast_accum)


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


def _build_wgrad_nt_plan(go, x, go_scale, x_scale):
    """Autotune an NT wgrad plan and verify it against the TN path it replaces.

    The reference is built through the same transpose copies the plan exists to delete -- a
    startup-only cost. ref_us times those copies too, because they are part of the op being
    replaced: the fair comparison is (t() + _scaled_mm) against the NT launch alone.
    """
    k, m = go.shape                    # k = tokens, m = out_features (weight rows)
    n = x.shape[1]                     # n = in_features (weight cols)
    plan = _ext().PinnedGemm(m, n, k, go.dtype, x.dtype, False, accum=False, nt=True)
    info = plan.autotune(go, x, go_scale, x_scale, _MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS)

    ref = _fallback(*_tn_operands(go, x), go_scale, x_scale)
    got = plan.run(go, x, go_scale, x_scale)
    scale = ref.abs().max().clamp(min=1e-6)
    err = (got.float() - ref.float()).abs().max() / scale
    if got.shape != ref.shape or not torch.isfinite(err) or err > 0.05:
        _warn(f"  {m}x{n}x{k} nt: REJECTED, max rel err {float(err):.4g} vs t()+_scaled_mm")
        return None

    ref_us = _time_fn(lambda: _fallback(*_tn_operands(go, x), go_scale, x_scale))
    vs_ref = ref_us / info["us"] if info["us"] > 0 else 0.0
    _LOG.append(f"  {m}x{n}x{k} nt: algo {int(info['algo_id'])} tile {int(info['tile'])} "
                f"splitK {int(info['splitk'])} | t()+_scaled_mm {ref_us:.1f} -> "
                f"{info['us']:.1f} us (vs_ref {vs_ref:.2f}x) | err {float(err):.2g}")
    return plan


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


# ---------------------------------------------------------------------------
# --pin-gemm: the same three GEMMs, run on a chosen algorithm instead of the heuristic's.
# ---------------------------------------------------------------------------


def _build_plan(a, b, a_scale, b_scale, fast_accum):
    """Autotune a plan for this shape, verify it against `_scaled_mm`, or return None."""
    m, k = a.shape
    n = b.shape[1]
    plan = _ext().PinnedGemm(m, n, k, a.dtype, b.dtype, fast_accum)
    info = plan.autotune(a, b, a_scale, b_scale, _MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS)

    # Trust nothing until it matches the op it replaces. A different algorithm reduces in a
    # different order, so this is a tolerance check and not equality -- but a transposed or
    # mis-scaled result misses by orders of magnitude, which is the actual risk here.
    ref = _fallback(a, b, a_scale, b_scale, fast_accum=fast_accum)
    got = plan.run(a, b, a_scale, b_scale)
    scale = ref.abs().max().clamp(min=1e-6)
    err = (got.float() - ref.float()).abs().max() / scale
    if got.shape != ref.shape or not torch.isfinite(err) or err > 0.05:
        _warn(f"  {m}x{n}x{k}: REJECTED, max rel err {float(err):.4g} vs _scaled_mm")
        return None

    # Two ratios answering different questions. vs_algo is what a better *algorithm* buys over
    # cuBLASLt's own first candidate reached through this same extension; vs_ref is what the pin
    # buys over the op it actually replaces. They diverge because torch queries the heuristic
    # with its own (smaller) workspace, so `_scaled_mm` may never be offered this algorithm --
    # and without the second number a 1.00x vs_algo reads as "nothing to win" when the
    # end-to-end arm says otherwise.
    ref_us = _time_fn(lambda: _fallback(a, b, a_scale, b_scale, fast_accum=fast_accum))
    vs_algo = info["heuristic_us"] / info["us"] if info["us"] > 0 else 0.0
    vs_ref = ref_us / info["us"] if info["us"] > 0 else 0.0
    _LOG.append(f"  {m}x{n}x{k}{' fa' if fast_accum else ''}: algo {int(info['algo_id'])} "
                f"tile {int(info['tile'])} splitK {int(info['splitk'])} "
                f"reduction {int(info['reduction'])} | "
                f"_scaled_mm {ref_us:.1f} / cand0 {info['heuristic_us']:.1f} -> "
                f"{info['us']:.1f} us (vs_ref {vs_ref:.2f}x, vs_algo {vs_algo:.2f}x) | "
                f"err {float(err):.2g}")
    return plan


# Opaque for the same reason `pinned_wgrad_nt` is: `_Float8Matmul` is `allow_in_graph`, so the
# plan builder would otherwise be handed FakeTensors. The mode and shape decisions stay outside
# in `mm`, where they are compile-time constants.
@torch.library.custom_op("nanochat::pinned_mm", mutates_args=())
def pinned_mm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
              fast_accum: bool) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    key = (m, n, k, a.dtype, b.dtype, fast_accum)
    if key not in _PLANS:
        try:
            _PLANS[key] = _build_plan(a, b, a_scale, b_scale, fast_accum)
        except ExtensionUnavailable:  # nothing can be pinned at all -- do not fall back silently
            raise
        except Exception as e:  # a bad plan must never take training down with it
            _warn(f"  {m}x{n}x{k}: FAILED to build ({type(e).__name__}: {e})")
            _PLANS[key] = None
    plan = _PLANS[key]
    if plan is None:
        return _fallback(a, b, a_scale, b_scale, fast_accum=fast_accum)
    return plan.run(a, b, a_scale, b_scale)


@pinned_mm.register_fake
def _(a, b, a_scale, b_scale, fast_accum):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16)


# Verification holds the [m, n] output twice -- the pinned result and the `_scaled_mm` reference
# -- on top of live activations, so a huge output costs far more peak memory than a pin can win.
# lm_head fwd is the only GEMM that trips this at d12/d16: its 16384x32768 bf16 output is 1.07 GB
# and pinning it cost the donor +4.2 GB of peak for a measured 1.00x. Everything else is under
# 100 MB.
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _pins(role, m, n):
    if _MODE == "off":
        return False
    if m * n * 2 > _MAX_OUTPUT_BYTES:  # bf16 output
        return False
    if _MODE == "all":
        return True
    if role != "wgrad":
        return False
    # The attention projections are square, which is what makes them the worst case: a 768x768
    # output reducing over every token.
    return _MODE == "wgrad" or m == n


def mm(a, b, a_scale, b_scale, out_dtype, role):
    """Drop-in for `_scaled_mm` on one of the three fp8 GEMMs of a Linear.

    Falls back to `_scaled_mm` whenever this role/shape is not pinned, the plan failed
    verification, or the output dtype is not the bf16 the pinned kernel produces. The mode is
    read at trace time, so with --pin-gemm off the compiled graph is upstream's.
    """
    fast_accum = _FAST_ACCUM[role]
    if out_dtype != torch.bfloat16 or not _pins(role, a.shape[0], b.shape[1]):
        return _fallback(a, b, a_scale, b_scale, out_dtype, fast_accum)
    return torch.ops.nanochat.pinned_mm(a, b, a_scale, b_scale, fast_accum)
