"""Hadamard transforms: the bf16-in/bf16-out kernels, and the matrices they take.

Ported from upstream `quartet2/rht.py`, plus :func:`hadamard_matrix`, which replaces upstream's
`scipy.linalg.hadamard` call. `scipy.linalg.hadamard` *is* Sylvester's construction, and 128 is
a power of two, so building it with four lines of torch is exact and saves a dependency.
"""
import torch

from .ext import load


@torch.library.custom_op("quartet2::transform_128", mutates_args=("out",))
def _group_transform_128(out: torch.Tensor, x: torch.Tensor, h: torch.Tensor, transpose: bool) -> None:
    load().group_transform_128(out, h, x, transpose)


@torch.library.custom_op("quartet2::transform_rht128", mutates_args=("out",))
def _transform_rht128(out: torch.Tensor, x: torch.Tensor, h: torch.Tensor, transpose: bool) -> None:
    load().transform_rht128(out, h, x, transpose)


def hadamard_matrix(n=128, dtype=torch.bfloat16, device="cuda"):
    """The n x n Sylvester Hadamard, scaled by n^-0.5 so the transform is orthonormal.

    n must be a power of two. Matches `scipy.linalg.hadamard(n) * n**-0.5` exactly (both are
    Sylvester's construction, and every entry is +-1 -- representable in any float dtype).
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of two, got {n}"
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h * n**-0.5).to(dtype)


def rerotate_hadamard(h):
    """Re-randomize the rotation by flipping column signs.

    Called once per backward: the Hadamard alone is a fixed rotation, and it is the *random*
    part that makes the transform spread outliers differently on each step. Signs go on the
    last dim -- the inner dim for a TN GEMM.
    """
    signs = torch.randint(0, 2, (h.size(0),), device=h.device, dtype=h.dtype) * 2 - 1
    return h * signs[None, :]


def transform_128(*, out: torch.Tensor = None, h: torch.Tensor, x: torch.Tensor, transpose: bool = False):
    """`x h^t` in bf16, with a full 128x128 `h`."""
    assert h.dim() == 2 and h.shape == (128, 128)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.is_cuda
    assert h.is_contiguous()

    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    rows, cols = x.shape
    if transpose:
        # TODO transpose with more dims doesn't pass tests
        assert len(original_shape) == 2
        cols, rows = rows, cols
    if out is None:
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _group_transform_128(out, x, h, transpose)
    out = out.reshape(original_shape)
    if transpose:
        out = out.reshape(out.T.shape)
    return out


def transform_rht128(*, out: torch.Tensor = None, h: torch.Tensor, x: torch.Tensor, transpose: bool = False):
    """`x h^t` in bf16, with `h` given as the top 16x128 slice of a 128x128 Hadamard."""
    assert h.dim() == 2 and h.shape == (16, 128)
    assert x.dtype == torch.bfloat16
    assert x.is_cuda
    assert x.is_contiguous()
    assert h.is_cuda
    assert h.is_contiguous()

    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    rows, cols = x.shape
    if transpose:
        # TODO transpose with more dims doesn't pass tests
        assert len(original_shape) == 2
        cols, rows = rows, cols
    if out is None:
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _transform_rht128(out, x, h, transpose)
    out = out.reshape(original_shape)
    if transpose:
        out = out.reshape(out.T.shape)
    return out


def swizzle_hadamard(hadamard_matrix):
    """The row reordering the rht128 kernels regenerate internally.

    For a Sylvester Hadamard `h` (with any column sign flips), this identity holds::

        transform_128(x, swizzle_hadamard(h)) == transform_rht128(x, h[:16, :])

    For debugging and testing only -- it is not efficient.
    """
    r = torch.empty_like(hadamard_matrix)
    reorder = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
    for k in range(0, 128, 64):
        for l in range(4):
            for j in range(16):
                r[k + 16 * l + j, :] = hadamard_matrix[k + 4 * l + reorder[j], :]
    return r
