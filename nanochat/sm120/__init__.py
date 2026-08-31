"""sm_120 (consumer/workstation Blackwell) specific code.

Nothing upstream points into this package -- the arrows only go the other way, so the files
karpathy/nanochat owns keep a narrow seam and the fork stays cheap to rebase.

Both fast paths here are **opt-in**, so an unconfigured run behaves exactly like upstream
nanochat and A/B arms compare what they claim to. Set `NANOCHAT_FA2_WINDOWED_FLASH=1` to route
sliding-window attention through the flash kernels instead of an SDPA mask; `--nvfp4` converts
the Linear layers. Neither happens on its own.

The windowed switch is an environment variable rather than a flag because it has to be global:
`base_train`, `chat_sft` and `chat_rl` each import this package at module top but parse their
args ~50 lines later, and the eval paths (`base_eval`, `engine`) take no flags at all. A flag
would mean an argparse block plus import-ordering surgery in three scripts and still miss the
rest. `attention.install()` is available for a caller that wants to decide in code.

| module          | what                                                                    |
|-----------------|-------------------------------------------------------------------------|
| `attention`     | windowed flash attention: the fast path SDPA's mask emulation replaces  |
| `nvfp4`         | NVFP4 training: `NVFP4Linear` and `convert_to_nvfp4_training`            |
| `fp4_gemm`      | `--nvfp4-lt-gemm`: this fork's cuBLASLt launcher for the fp4 GEMMs      |
| `quartet/`      | vendored Quartet-II kernels (arXiv 2601.22813) and their Python wrappers |
"""
import os

from . import attention  # noqa: F401  -- the module, not the install; see WINDOWED_FLASH_ENV

WINDOWED_FLASH_ENV = "NANOCHAT_FA2_WINDOWED_FLASH"


def _env_opted_in():
    """Truthy spellings only, so `NANOCHAT_FA2_WINDOWED_FLASH=0` reads as off rather than as set."""
    return os.environ.get(WINDOWED_FLASH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


if _env_opted_in():
    attention.install()

__all__ = ["attention", "WINDOWED_FLASH_ENV"]
