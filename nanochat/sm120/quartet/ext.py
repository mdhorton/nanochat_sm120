"""JIT build/load of the vendored Quartet-II CUDA kernels.

Built on first use with `torch.utils.cpp_extension.load` and cached under the nanochat base
dir, so nothing has to change in `pyproject.toml`/`uv.lock` and there is no separate build step.
The first import costs a few minutes; after that it is a hash check.

**The real requirement is CUDA >= 12.8, not 13.x.** The kernels use `cuda_fp4.h` intrinsics
that lower to `cvt.rn.satfinite.e2m1x2.f32`, which appears in 12.8 -- the same release that
added sm_120 and NVFP4. Nothing here needs CUDA 13.

`cpp_extension` hard-errors on a CUDA *major* mismatch between `nvcc` and the torch build, so
each venv needs a toolkit of its own major. Both are installed here (12.8 for `.venv`/cu128,
13.2 and 13.3 for `.venv-cu130`/cu130), and `resolve_cuda_home` picks the right one per venv --
no `CUDA_HOME` juggling at the call site, and it says which nvcc it chose when it overrides.
Build products are cached per torch build, so the two never share a `.so`.
"""
import os
import threading

import torch

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# binding.cpp is ours (see its header); the rest is upstream verbatim.
_SOURCES = [
    "binding.cpp",
    "group_transform_and_eden.cu",
    "group_transform.cu",
    "rht128.cu",
    "rht128_ws.cu",
    "rht128_tma.cu",
    "rht128_eden.cu",
    "rht128_requant.cu",
    "round_four_six.cu",
    "round_eden_fp4.cu",
    "dq_tp_q.cu",
]

_lib = None
_lock = threading.Lock()


def _arch_flag():
    """The `sm_120a`-style gencode this device needs.

    The `a` suffix is not optional: the fp4 conversion and `ldmatrix.trans` PTX these kernels
    emit are architecture-*specific* features, unavailable in the portable `sm_120` target.
    """
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (10, 0):
        raise RuntimeError(
            f"Quartet-II NVFP4 kernels need Blackwell (sm_100+); this device is sm_{major}{minor}. "
            "FP4 tensor cores do not exist before Blackwell -- Hopper stops at fp8."
        )
    return f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"


def _nvcc_version(cuda_home):
    """(major, minor) of the nvcc under `cuda_home`, or None if there isn't a usable one."""
    import subprocess

    if cuda_home is None:
        return None
    try:
        out = subprocess.check_output([os.path.join(cuda_home, "bin", "nvcc"), "--version"],
                                      stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    for token in out.split():
        if token.startswith("V") and token[1:2].isdigit():
            parts = token[1:].split(".")
            return int(parts[0]), int(parts[1])
    return None


def _torch_cuda():
    """(major, minor) this torch was built against, or None."""
    if torch.version.cuda is None:
        return None
    return tuple(int(v) for v in torch.version.cuda.split(".")[:2])


def resolve_cuda_home():
    """A CUDA_HOME whose nvcc *major* matches this torch build. Returns (path, nvcc) or (None, None).

    `cpp_extension` hard-errors on a major mismatch, and its default CUDA_HOME follows
    `/usr/local/cuda`, which `update-alternatives` points at the highest-priority toolkit --
    13.3 on this box. That is the wrong one for a cu12 torch *even when* a 12.8 toolkit is
    installed right next to it, so the default alone would make `.venv` unbuildable for no good
    reason. Rather than make every caller export CUDA_HOME, look next door: side-by-side
    `/usr/local/cuda-<major>.<minor>` installs are the normal apt layout.

    A CUDA_HOME already in the environment is honoured **when its major matches**, and only
    then. Not "explicit always wins": on this box the shell profile exports CUDA_HOME *and*
    CUDA_PATH as `/usr/local/cuda`, the generic symlink -- ambient config naming no particular
    toolkit, rather than anyone's decision. Treating that as authoritative made a cu12 torch
    unbuildable while the toolkit it needed sat in the next directory along.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    want = _torch_cuda()
    if want is None:
        return None, None

    current = _nvcc_version(CUDA_HOME)
    if current is not None and current[0] == want[0]:
        return CUDA_HOME, current

    # The configured toolkit cannot build for this torch. Look for a sibling that can --
    # side-by-side /usr/local/cuda-<ver> is the standard apt layout.
    import glob

    candidates = []
    for path in sorted(glob.glob("/usr/local/cuda-*")):
        got = _nvcc_version(path)
        if got is not None and got[0] == want[0]:
            # Sort key: exact minor match first, then version, then the longer (more specific)
            # path -- /usr/local/cuda-12 and /usr/local/cuda-12.8 are the same toolkit.
            candidates.append((got == want, got, path))
    if candidates:
        _, got, path = max(candidates)
        return path, got
    return CUDA_HOME, current


def unavailable_reason():
    """None if the kernels can be built here, else a human-readable reason why not."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (10, 0):
        return f"needs Blackwell (sm_100+), this device is sm_{major}{minor}"
    want = _torch_cuda()
    if want is None:
        return "this torch build has no CUDA"
    if want < (12, 8):
        return f"needs a CUDA 12.8+ torch for cuda_fp4.h, this one is {torch.version.cuda}"
    # cpp_extension refuses a CUDA *major* mismatch between nvcc and the torch build, so check
    # it here rather than letting it surface as a build failure minutes in.
    cuda_home, nvcc = resolve_cuda_home()
    if nvcc is None:
        return "no nvcc found (set CUDA_HOME)"
    if nvcc[0] != want[0]:
        return (f"nvcc is CUDA {nvcc[0]}.{nvcc[1]} but torch was built against "
                f"{torch.version.cuda}; cpp_extension requires matching major versions. "
                f"Install a CUDA {want[0]}.x toolkit, or set CUDA_HOME to one.")
    return None


def load():
    """Build (once) and return the extension module. Raises if this box cannot build it."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is not None:
            return _lib
        reason = unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"cannot load the Quartet-II NVFP4 kernels: {reason}")

        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load as _load

        from nanochat.common import get_base_dir

        # Point the build at the toolkit whose major matches this torch (see resolve_cuda_home).
        # cpp_extension caches CUDA_HOME in a module global at import time and every nvcc path
        # goes through it, so overriding the global is what actually redirects the build.
        cuda_home, nvcc = resolve_cuda_home()
        if cuda_home and cuda_home != cpp_extension.CUDA_HOME:
            # Say so rather than silently redirecting: which nvcc built the kernels is exactly
            # the sort of thing you want in the log when chasing a numerics or perf oddity.
            print(f"[nvfp4] building with CUDA {nvcc[0]}.{nvcc[1]} at {cuda_home} "
                  f"(torch is cu{torch.version.cuda}; {cpp_extension.CUDA_HOME} does not match)")
            cpp_extension.CUDA_HOME = cuda_home
            os.environ["CUDA_HOME"] = cuda_home

        # Keyed by the torch build, not just the name: this box has two venvs on different CUDA
        # majors, and a shared directory silently hands one venv's .so to the other. That
        # happened to run, which is worse than failing.
        tag = f"quartet2-torch{torch.__version__.split('+')[0]}-cu{torch.version.cuda}"
        build_dir = os.path.join(get_base_dir(), "torch_extensions", tag)
        os.makedirs(build_dir, exist_ok=True)

        _lib = _load(
            name="quartet2_sm120",
            sources=[os.path.join(_CSRC, s) for s in _SOURCES],
            build_directory=build_dir,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",          # upstream's CMakeLists sets CMAKE_CUDA_STANDARD 20
                "-lineinfo",
                "--use_fast_math",     # upstream passes -ffast-math, which nvcc spells this way
                _arch_flag(),
                "--expt-relaxed-constexpr",
                # cpp_extension defines these; upstream's CMake build does not, and the kernels
                # rely on the implicit __nv_bfloat16 <-> float conversions they suppress. Listed
                # after the defaults so the -U wins.
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            ],
            # rht128_tma.cu builds its TMA descriptors with the driver API (cuTensorMapEncodeTiled);
            # cpp_extension links cudart but not the driver stub.
            extra_ldflags=["-lcuda"],
            verbose=os.environ.get("NANOCHAT_VERBOSE_BUILD") == "1",
        )
        return _lib
