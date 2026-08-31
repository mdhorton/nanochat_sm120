"""sm_120 (consumer/workstation Blackwell) specific code.

Nothing upstream points into this package -- the arrows only go the other way, so the files
karpathy/nanochat owns keep a narrow seam and the fork stays cheap to rebase.

Importing this package installs the windowed-flash fast path, which SDPA cannot express and
which the `SSSL` default depends on. The NVFP4 pieces stay inert until --nvfp4 converts the
Linear layers.

| module          | what                                                                    |
|-----------------|-------------------------------------------------------------------------|
| `attention`     | windowed flash attention: the fast path SDPA's mask emulation replaces  |
| `nvfp4`         | NVFP4 training: `NVFP4Linear` and `convert_to_nvfp4_training`            |
| `fp4_gemm`      | `--nvfp4-lt-gemm`: this fork's cuBLASLt launcher for the fp4 GEMMs      |
| `quartet/`      | vendored Quartet-II kernels (arXiv 2601.22813) and their Python wrappers |
"""
from . import attention  # noqa: F401  -- registers flash_attention._windowed_impl

__all__ = ["attention"]
