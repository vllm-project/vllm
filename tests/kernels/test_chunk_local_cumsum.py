# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for the chunk_local_cumsum GDN/FLA Triton kernels."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

# 3D hits the scalar kernel, 4D the vector one; T=100 leaves a partial tail chunk.
SHAPES = [(1, 64, 4), (2, 100, 8), (1, 64, 4, 32), (2, 100, 4, 64)]

HEAD_SHAPES = [(4,), (4, 32)]

CHUNK_SIZES = [32, 64]

# output_dtype=None keeps the input dtype.
NO_CAST = (torch.float32, torch.float32, 1e-2)
DOWNCAST_BF16 = (torch.bfloat16, None, 5e-2)

DTYPE_CASES = [
    NO_CAST,
    (torch.bfloat16, torch.float32, 1e-2),
    DOWNCAST_BF16,
]

VARLEN_DTYPE_CASES = [NO_CAST, DOWNCAST_BF16]


@pytest.fixture(autouse=True)
def default_device():
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    yield
    torch.set_default_device(None)


def ref_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inclusive prefix (or suffix) sum inside each chunk along T."""
    if cu_seqlens is not None:
        bounds = zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        return torch.cat(
            [ref_local_cumsum(g[:, s:e], chunk_size, reverse) for s, e in bounds],
            dim=1,
        )
    T = g.shape[1]
    pad = -T % chunk_size
    if pad:
        g = torch.cat([g, g.new_zeros(g.shape[0], pad, *g.shape[2:])], dim=1)
    x = g.float().unflatten(1, (-1, chunk_size))
    x = x.flip(2).cumsum(2).flip(2) if reverse else x.cumsum(2)
    return x.flatten(1, 2)[:, :T]


def assert_matches_reference(
    g: torch.Tensor,
    chunk_size: int,
    *,
    reverse: bool,
    tol: float,
    output_dtype: torch.dtype | None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> None:
    out = chunk_local_cumsum(
        g,
        chunk_size,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=output_dtype,
    )
    expected_dtype = output_dtype or g.dtype
    assert out.dtype == expected_dtype
    assert out.shape == g.shape

    ref = ref_local_cumsum(g, chunk_size, reverse, cu_seqlens).to(expected_dtype)
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype,output_dtype,tol", DTYPE_CASES)
def test_chunk_local_cumsum(
    shape: tuple[int, ...],
    chunk_size: int,
    reverse: bool,
    dtype: torch.dtype,
    output_dtype: torch.dtype | None,
    tol: float,
) -> None:
    g = torch.randn(*shape, dtype=dtype)
    assert_matches_reference(
        g, chunk_size, reverse=reverse, tol=tol, output_dtype=output_dtype
    )


@pytest.mark.parametrize("seq_lens", [[64], [1, 63, 100]])
@pytest.mark.parametrize("head_shape", HEAD_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype,output_dtype,tol", VARLEN_DTYPE_CASES)
def test_chunk_local_cumsum_varlen(
    seq_lens: list[int],
    head_shape: tuple[int, ...],
    chunk_size: int,
    reverse: bool,
    dtype: torch.dtype,
    output_dtype: torch.dtype | None,
    tol: float,
) -> None:
    cu_seqlens = torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)
    g = torch.randn(1, int(cu_seqlens[-1]), *head_shape, dtype=dtype)
    assert_matches_reference(
        g,
        chunk_size,
        reverse=reverse,
        tol=tol,
        output_dtype=output_dtype,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize("head_shape", HEAD_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_local_cumsum_precomputed_chunk_indices(
    head_shape: tuple[int, ...],
    chunk_size: int,
) -> None:
    """GDN/KDA always hand the kernel indices computed by the caller."""
    dtype, output_dtype, tol = NO_CAST
    cu_seqlens = torch.tensor([0, 1, 64, 164], dtype=torch.int32)
    g = torch.randn(1, int(cu_seqlens[-1]), *head_shape, dtype=dtype)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    assert_matches_reference(
        g,
        chunk_size,
        reverse=False,
        tol=tol,
        output_dtype=output_dtype,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert torch.equal(
        chunk_local_cumsum(
            g, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        ),
        chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens),
    )
