"""The NVFP4 GEMMs driven through cuBLASLt directly (`--nvfp4-lt-gemm`).

`torch._scaled_mm` reaches the block-scaled fp4 kernels through cuBLASLt but gives no way to ask
for a different algorithm, and it queries the heuristic with a 1 MiB workspace. This routes the
three GEMMs of an `NVFP4Linear` through `csrc/fp4_gemm.cpp` instead, autotuning one plan per
shape at first use with a 32 MiB workspace.

**On throughput this is a null result and the flag is off**: cuBLASLt has one block-scaled fp4
kernel for these shapes, so the best candidate for every GEMM in the model is 1.001x on the GEMM
row and 0.0% end to end against a 0.04% reference spread (`dev/nvfp4-quartet.md`, *The GEMM
headroom is not an algorithm choice*). What it delivers is the seam the next two queue items
need -- a device-pointer alpha (the per-tensor scale folded into the epilogue, deleting an [M, N]
pass per GEMM) and a beta=1 accumulate into an fp32 `main_grad` -- neither of which `_scaled_mm`
can express. Both have since landed here: `--nvfp4-epilogue-alpha` (A2) and
`--nvfp4-fuse-wgrad` (A3, driven from `nvfp4.WgradAccumStore`), each its own plan kind keyed
separately in `_PLANS`.

Each plan is autotuned in the production configuration rather than replaying a recorded algorithm
id, then verified against `_scaled_mm` before being trusted; at every shape in the real model it
comes out bit-identical. A plan that fails verification warns and falls back for that shape; a
build failure raises, because a run that silently ignores the flag it was given is how
`dev/perf-log.md` experiment 16 lost two arms.
"""
import os
import sys

import torch

FP4 = torch.float4_e2m1fn_x2

_ENABLED = False
_EPILOGUE_ALPHA = False
_EXT = None
_PLANS = {}            # (m, n, k, kind) -> Fp4Gemm or None (fall back); kind in "", "a", "acc"
_LOG = []              # human-readable record of what got pinned, for the caller to print

# Verification holds the [m, n] output twice -- the pinned result and the `_scaled_mm` reference
# -- on top of live activations, since the first call happens inside a real training step. At
# d20/dbs4 the largest fp4 GEMM is lm_head's 8192x32768 forward at 537 MB, which fits in this
# box's 7.7 GiB of headroom; the cap is here so a shape nobody anticipated cannot OOM a run.
_MAX_OUTPUT_BYTES = 1024 * 1024 * 1024

# Autotune budget, and the protocol behind it, are the fp8 pinned GEMM's: screen every candidate
# (which doubles as the soak that takes the card off its cold boost), then score the finalists
# round-robin with alternating order and take the median.
_MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS = 64, 20, 8, 5


class ExtensionUnavailable(RuntimeError):
    """The JIT extension could not be built, so nothing can be pinned at all."""


def configure(enabled, epilogue_alpha=False):
    """Turn the launcher on. Called once from base_train before training starts.

    Validated before anything is assigned: a configure() that raises must not leave the module
    half-on, or the caller's error handling decides which kernels a run uses.
    """
    if epilogue_alpha and not enabled:
        raise ValueError("the epilogue alpha needs the cuBLASLt launcher (--nvfp4-lt-gemm)")
    global _ENABLED, _EPILOGUE_ALPHA
    _ENABLED = bool(enabled)
    _EPILOGUE_ALPHA = bool(epilogue_alpha)


def enabled():
    """Trace-time predicate: `fp4_mm` branches on this, so with it False nothing here runs."""
    return _ENABLED


def epilogue_alpha():
    """Trace-time predicate: with it True, `mm` applies the per-tensor scale and the caller
    must not apply it again."""
    return _EPILOGUE_ALPHA


def log_lines():
    return list(_LOG)


def _warn(line):
    """Surface a plan failure the moment it happens, not only into _LOG."""
    _LOG.append(line)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"WARNING: pinned fp4 GEMM {line.strip()}", file=sys.stderr, flush=True)


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load

        from nanochat.common import get_base_dir
        from nanochat.sm120.quartet.ext import resolve_cuda_home

        # Same toolkit problem as the Quartet kernels: cpp_extension refuses a CUDA major
        # mismatch and its default CUDA_HOME follows /usr/local/cuda, which is 13.x here.
        cuda_home, _ = resolve_cuda_home()
        if cuda_home and cuda_home != cpp_extension.CUDA_HOME:
            cpp_extension.CUDA_HOME = cuda_home
            os.environ["CUDA_HOME"] = cuda_home
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "fp4_gemm.cpp")
        # Keyed by the torch build, like the Quartet extension: two venvs on different CUDA
        # majors sharing a build directory silently hands one venv's .so to the other.
        tag = f"fp4gemm-torch{torch.__version__.split('+')[0]}-cu{torch.version.cuda}"
        build_dir = os.path.join(get_base_dir(), "torch_extensions", tag)
        os.makedirs(build_dir, exist_ok=True)
        try:
            # with_cuda for a source with no device code: it is what adds the CUDA include and
            # library paths, and the file is a .cpp precisely so nvcc never sees it (see the
            # source header). At runtime the SONAME libcublasLt.so.12 is already loaded by torch,
            # so the loader hands this extension torch's copy -- one library, the same one
            # `_scaled_mm` calls, whatever the build-time toolkit was.
            _EXT = load(name="fp4_gemm_sm120", sources=[src], build_directory=build_dir,
                        extra_cflags=["-O3", "-std=c++20"],
                        extra_ldflags=["-lcublasLt"], with_cuda=True,
                        verbose=os.environ.get("NANOCHAT_VERBOSE_BUILD") == "1")
        except Exception as e:
            raise ExtensionUnavailable(
                f"could not build the fp4_gemm extension ({type(e).__name__}: {e}). "
                "The launcher cannot work without it; pass --no-nvfp4-lt-gemm to run the "
                "NVFP4 path on _scaled_mm instead."
            ) from e
    return _EXT


def preload():
    """Build the extension now instead of on the first GEMM. Returns the cuBLASLt version.

    The launcher is on by default under `--nvfp4`, so a toolchain that cannot build it has to
    say so at startup rather than from inside the first backward, where the failure arrives
    mid-step with a stack that points at an autograd Function.
    """
    return _ext().cublaslt_version()


def _fallback(a, b, a_scale, b_scale):
    return torch._scaled_mm(a.view(FP4), b.view(FP4).t(), scale_a=a_scale, scale_b=b_scale,
                            out_dtype=torch.bfloat16)


def _time_scaled_mm(a, b, a_scale, b_scale, iters=_ITERS):
    """Time the op being replaced, so the plan log says what the pin bought at this shape.

    The plan's own `heuristic_us` is cuBLASLt candidate 0 reached through this extension, which
    is not the same thing as `_scaled_mm`: torch queries with a 1 MiB workspace, so it may not
    even be offered the algorithm this picks.

    **Read the resulting `vs_ref` as indicative only.** It lands at the end of the shape's
    autotune, i.e. at the hottest point of a soak the reference did not share, which on this
    power-capped box is worth several percent either way. `scripts/probe_fp4_gemm.py` alternates
    the arms and takes medians; that is the number to quote.
    """
    for _ in range(3):
        _fallback(a, b, a_scale, b_scale)
    t0, t1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    t0.record()
    for _ in range(iters):
        _fallback(a, b, a_scale, b_scale)
    t1.record()
    t1.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters


def _build_plan(a, b, a_scale, b_scale, alpha=None):
    """Autotune a plan for this shape, verify it against `_scaled_mm`, or return None."""
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    plan = _ext().Fp4Gemm(m, n, k, _ext().WORKSPACE_BYTES, alpha is not None)
    info = plan.autotune(a, b, a_scale, b_scale, _MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS)

    # Trust nothing until it matches the op it replaces. Both compute the same fp4 products, so
    # only the reduction order differs -- but a swapped operand, a mis-set scale mode or a
    # row-major scale block fails this by orders of magnitude, which is the actual risk.
    ref = _fallback(a, b, a_scale, b_scale)
    if alpha is None:
        got = plan.run(a, b, a_scale, b_scale)
        tol = 0.05
    else:
        # A device-alpha plan is deliberately *not* bit-identical: it scales the fp32 accumulator
        # and rounds once, where the reference rounds to bf16 and then multiplies. That is one
        # bf16 ulp (~0.4%), in the accurate direction -- though only 0.1% of the GEMM's total
        # error, which fp4 quantization dominates by ~1,500x, so it is not a numerics argument
        # for anything. The tolerance allows the ulp and nothing structural: alpha is ~1e-6 here,
        # so a dropped or doubled alpha misses by six orders of magnitude.
        ref = ref * alpha
        got = plan.run_alpha(a, b, a_scale, b_scale, alpha)
        tol = 0.02
    ref_us = _time_scaled_mm(a, b, a_scale, b_scale)
    scale = ref.abs().max().clamp(min=1e-6)
    err = (got.float() - ref.float()).abs().max() / scale
    if got.shape != ref.shape or not torch.isfinite(err) or err > tol:
        _warn(f"  {m}x{n}x{k}: REJECTED, max rel err {float(err):.4g} vs _scaled_mm")
        return None

    # Two ratios, and they answer different questions. vs_algo is what a better *algorithm* buys
    # over cuBLASLt's own first candidate reached the same way; vs_ref is what the pin buys over
    # the op it actually replaces. They diverge when `_scaled_mm` is not offered this algorithm.
    vs_algo = info["heuristic_us"] / info["us"] if info["us"] > 0 else 0.0
    vs_ref = ref_us / info["us"] if info["us"] > 0 else 0.0
    _LOG.append(f"  {m}x{n}x{k}{' a' if alpha is not None else ''}: "
                f"algo {int(info['algo_id'])} tile {int(info['tile'])} "
                f"stages {int(info['stages'])} splitK {int(info['splitk'])} "
                f"cluster {int(info['cluster'])} | "
                f"cand0 algo {int(info['heuristic_algo_id'])} tile {int(info['heuristic_tile'])} "
                f"| {int(info['ran'])}/{int(info['candidates'])} ran | "
                f"_scaled_mm {ref_us:.1f} / cand0 {info['heuristic_us']:.1f} -> {info['us']:.1f} us "
                f"(vs_ref {vs_ref:.2f}x, vs_algo {vs_algo:.2f}x) | err {float(err):.2g}")
    return plan


def _build_accum_plan(a, b, a_scale, b_scale, alpha):
    """Autotune a beta=1 fp32-accumulate plan and verify it against `_scaled_mm * alpha` + add.

    `c0` is drawn at the scaled product's own magnitude, so the two failure modes this has to
    separate are both far outside the tolerance rather than inside rounding: a plan that ignored
    beta misses by ~0.6 relative, and one that ignored alpha by ~1e6 (alpha is ~1e-6 here).
    Measured on 512x384x256, the plan itself lands at 1.7e-3 -- one bf16 rounding, which is what
    the reference has and the epilogue does not.
    """
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    plan = _ext().Fp4Gemm(m, n, k, _ext().WORKSPACE_BYTES, True, True)
    info = plan.autotune(a, b, a_scale, b_scale, _MAX_ALGOS, _ITERS, _FINALISTS, _ROUNDS)

    prod = _fallback(a, b, a_scale, b_scale).float() * alpha.float()
    c0 = torch.randn(m, n, device=a.device) * prod.abs().mean().clamp(min=1e-30)
    ref = prod + c0
    got = c0.clone()
    plan.run_accum(a, b, a_scale, b_scale, alpha, got)
    scale = ref.abs().max().clamp(min=1e-30)
    err = (got - ref).abs().max() / scale
    if got.shape != ref.shape or not torch.isfinite(err) or err > 0.02:
        _warn(f"  {m}x{n}x{k} accum: REJECTED, max rel err {float(err):.4g} vs _scaled_mm+add")
        return None

    # The op this replaces, all of it: the GEMM, the per-tensor scale, the widening to fp32 and
    # the add into the accumulator -- which is what `AccumulateGrad` runs today, spread over
    # three kernels instead of one epilogue.
    acc = c0.clone()
    for _ in range(3):
        acc += _fallback(a, b, a_scale, b_scale).float() * alpha.float()
    t0, t1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    t0.record()
    for _ in range(_ITERS):
        acc += _fallback(a, b, a_scale, b_scale).float() * alpha.float()
    t1.record()
    t1.synchronize()
    ref_us = t0.elapsed_time(t1) * 1000.0 / _ITERS
    vs_ref = ref_us / info["us"] if info["us"] > 0 else 0.0
    _LOG.append(f"  {m}x{n}x{k} accum: "
                f"algo {int(info['algo_id'])} tile {int(info['tile'])} "
                f"stages {int(info['stages'])} splitK {int(info['splitk'])} "
                f"cluster {int(info['cluster'])} | {int(info['ran'])}/{int(info['candidates'])} "
                f"ran | _scaled_mm+add {ref_us:.1f} -> {info['us']:.1f} us "
                f"(vs_ref {vs_ref:.2f}x) | err {float(err):.2g}")
    return plan


# An opaque custom op, not a plain function: the plan builder must never be handed FakeTensors,
# and the autograd Function around it is traced by AOTAutograd on both halves. Everything that
# touches the extension lives behind this boundary; the enablement decision stays outside, where
# it is a compile-time constant.
@torch.library.custom_op("nanochat::fp4_pinned_mm", mutates_args=())
def fp4_pinned_mm(a: torch.Tensor, b: torch.Tensor,
                  a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    key = (m, n, k, "")
    if key not in _PLANS:
        try:
            _PLANS[key] = _build_plan(a, b, a_scale, b_scale)
        except ExtensionUnavailable:  # nothing can be pinned at all -- do not fall back silently
            raise
        except Exception as e:  # a bad plan must never take training down with it
            _warn(f"  {m}x{n}x{k}: FAILED to build ({type(e).__name__}: {e})")
            _PLANS[key] = None
    plan = _PLANS[key]
    if plan is None:
        return _fallback(a, b, a_scale, b_scale)
    return plan.run(a, b, a_scale, b_scale)


@fp4_pinned_mm.register_fake
def _(a, b, a_scale, b_scale):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


# The same GEMM with the per-tensor scale as the epilogue's alpha (queue A2): `out * alpha` was a
# full read-and-write pass over [m, n] wherever Inductor could not fuse it into a consumer, and
# for the two backward GEMMs it never could.
@torch.library.custom_op("nanochat::fp4_pinned_mm_alpha", mutates_args=())
def fp4_pinned_mm_alpha(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
                        b_scale: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    key = (m, n, k, "a")
    if key not in _PLANS:
        try:
            _PLANS[key] = _build_plan(a, b, a_scale, b_scale, alpha)
        except ExtensionUnavailable:  # nothing can be pinned at all -- do not fall back silently
            raise
        except Exception as e:  # a bad plan must never take training down with it
            _warn(f"  {m}x{n}x{k} alpha: FAILED to build ({type(e).__name__}: {e})")
            _PLANS[key] = None
    plan = _PLANS[key]
    if plan is None:
        return _fallback(a, b, a_scale, b_scale) * alpha
    return plan.run_alpha(a, b, a_scale, b_scale, alpha)


@fp4_pinned_mm_alpha.register_fake
def _(a, b, a_scale, b_scale, alpha):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


# The wgrad with gradient accumulation folded into its epilogue (queue A3): beta=1 and
# C = D = the layer's fp32 accumulator, so the micro-step's contribution joins the running sum
# inside the GEMM instead of being written bf16, widened and added by autograd's AccumulateGrad.
# `mutates_args` is what lets functionalization handle the in-place write when the backward is
# traced; the op returns nothing, because the mutation *is* the result.
@torch.library.custom_op("nanochat::fp4_pinned_mm_accum", mutates_args=("out",))
def fp4_pinned_mm_accum(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
                        b_scale: torch.Tensor, alpha: torch.Tensor, out: torch.Tensor) -> None:
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    key = (m, n, k, "acc")
    if key not in _PLANS:
        try:
            _PLANS[key] = _build_accum_plan(a, b, a_scale, b_scale, alpha)
        except ExtensionUnavailable:  # nothing can be pinned at all -- do not fall back silently
            raise
        except Exception as e:  # a bad plan must never take training down with it
            _warn(f"  {m}x{n}x{k} accum: FAILED to build ({type(e).__name__}: {e})")
            _PLANS[key] = None
    plan = _PLANS[key]
    if plan is None:
        # Same semantics, three kernels instead of one: the sum still runs in fp32, so a
        # fallen-back shape is slower and not less accurate.
        out.add_(_fallback(a, b, a_scale, b_scale).float() * alpha.float())
        return
    plan.run_accum(a, b, a_scale, b_scale, alpha, out)


@fp4_pinned_mm_accum.register_fake
def _(a, b, a_scale, b_scale, alpha, out):
    return None


def mm(a, b, a_scale, b_scale, alpha=None):
    """`a @ b.T` for packed-uint8 NVFP4 operands, through a pinned cuBLASLt algorithm.

    With `alpha` (a 0-d fp32 device tensor) the per-tensor scale is applied **inside the GEMM
    epilogue** and the caller must not apply it again. Falls back to `_scaled_mm` when the
    launcher is off or the output is too large to verify -- and the fallback applies alpha
    itself, so the contract is the same either way.
    """
    if not _ENABLED or a.shape[0] * b.shape[0] * 2 > _MAX_OUTPUT_BYTES:
        out = _fallback(a, b, a_scale, b_scale)
        return out if alpha is None else out * alpha
    if alpha is None:
        return torch.ops.nanochat.fp4_pinned_mm(a, b, a_scale, b_scale)
    return torch.ops.nanochat.fp4_pinned_mm_alpha(a, b, a_scale, b_scale, alpha)


def mm_accum(a, b, a_scale, b_scale, alpha, out):
    """`out += alpha * a @ b.T`, accumulating in fp32 inside the GEMM epilogue.

    `out` is the caller's fp32 gradient accumulator and is mutated in place; nothing is
    returned. The per-tensor scale is always folded in here -- see `fuse_wgrad()`.

    The verification in `_build_accum_plan` holds three [m, n] fp32 tensors on top of live
    activations, so the cap is applied to that rather than to the accumulator itself, which the
    caller allocated either way.
    """
    if not _ENABLED or a.shape[0] * b.shape[0] * 4 > _MAX_OUTPUT_BYTES:
        out.add_(_fallback(a, b, a_scale, b_scale).float() * alpha.float())
        return
    torch.ops.nanochat.fp4_pinned_mm_accum(a, b, a_scale, b_scale, alpha, out)
