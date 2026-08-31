"""sm_120 (consumer/workstation Blackwell) specific code.

Nothing upstream points into this package -- the arrows only go the other way, so the files
karpathy/nanochat owns keep a narrow seam and the fork stays cheap to rebase.

Importing this package installs the windowed-flash fast path, which SDPA cannot express and
which the `SSSL` default depends on.

| module          | what                                                                    |
|-----------------|-------------------------------------------------------------------------|
| `attention`     | windowed flash attention: the fast path SDPA's mask emulation replaces  |
"""
from . import attention  # noqa: F401  -- registers flash_attention._windowed_impl

__all__ = ["attention"]
