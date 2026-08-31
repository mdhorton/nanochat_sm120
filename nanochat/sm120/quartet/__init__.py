"""Quartet-II NVFP4 kernels, vendored from IST-DASLab/Quartet-II.

Upstream: https://github.com/IST-DASLab/Quartet-II @ 5f2a47e, Apache-2.0.
Paper: "Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient
Estimation", Panferov, Schultheis, Tabesh, Alistarh (ICML 2026), arXiv:2601.22813.

What is vendored, and what changed:

  `csrc/*.cu`, `*.cuh`   verbatim. They take raw pointers and include no binding headers, so
                         they need no edits to build against torch.
  `csrc/binding.cpp`     ours. Upstream binds through nanobind + dlpack, which needs
                         scikit-build-core, CMake and a cp312 stable-ABI wheel; this is
                         pybind11 over `at::Tensor` with the same names and contracts.
  `ext.py`               ours. JIT build via `cpp_extension.load`, cached.
  `quant.py`, `rht.py`   ported: `_quartet2` -> `ext.load()`, and `scipy.linalg.hadamard` ->
                         a four-line Sylvester construction (`rht.hadamard_matrix`).

Upstream's `linear.py` is *not* vendored. It gets its GEMM from `flashinfer.mm_fp4`; the
nanochat-facing port in `nanochat/sm120/nvfp4.py` uses `torch._scaled_mm` instead, which
already does block-scaled NVFP4 on this box and needs no extra dependency. Upstream's
`pseudoquant.py` is not vendored either -- `reference.py` here is a much smaller pure-torch
NVFP4, written from the format rather than ported, so the tests measure against something
independent of upstream's CUDA.
"""
