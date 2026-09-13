# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for _copy_h2d_dst_strided in routed_experts.py.

The helper must be a drop-in replacement for ``dst.copy_(src)`` for every
source layout produced by fused-MoE checkpoint loading (see issue #31624):
contiguous slabs, transpose views of memory-mapped tensors, and TP-narrowed
transpose views with unit inner stride.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts import (
    _copy_h2d_dst_strided,
)

CUDA_DTYPES = [torch.float32, torch.bfloat16, torch.float8_e4m3fn]
CPU_DTYPES = [torch.float32, torch.bfloat16]


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-level equality: a copy must reproduce the source exactly,
    and this stays true for dtypes whose NaNs defeat torch.equal."""
    return torch.equal(
        a.cpu().contiguous().view(torch.uint8),
        b.cpu().contiguous().view(torch.uint8),
    )


def _make_src(layout: str, dtype: torch.dtype) -> torch.Tensor:
    """Build a source tensor mimicking checkpoint loading layouts."""
    # Disk layout of one fused expert slab: [hidden_in, hidden_out].
    # Keep values within fp8_e4m3 range (max 448): out-of-range casts
    # saturate to NaN, and NaN != NaN breaks torch.equal even for
    # byte-identical copies.
    base = (
        (torch.arange(48 * 32, dtype=torch.float32) % 448.0).reshape(48, 32).to(dtype)
    )
    if layout == "contiguous":
        return base
    if layout == "transposed":
        # Runtime orientation view, as created by llama4.py.
        return base.transpose(-1, -2)
    if layout == "tp_narrowed":
        # Transpose view narrowed along the sharded dim (TP>1): unit inner
        # stride with a padded row pitch.
        return base.transpose(-1, -2).narrow(0, 8, 16)
    raise ValueError(layout)


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "tp_narrowed"])
@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_same_device_matches_plain_copy(layout: str, dtype: torch.dtype):
    src = _make_src(layout, dtype)
    expected = torch.empty_like(src)
    expected.copy_(src)
    actual = torch.empty_like(src)
    _copy_h2d_dst_strided(actual, src)
    assert _same_bytes(actual, expected)


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "tp_narrowed"])
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h2d_matches_plain_copy(layout: str, dtype: torch.dtype):
    src = _make_src(layout, dtype)
    expected = torch.empty(src.shape, dtype=dtype, device="cuda")
    expected.copy_(src)
    actual = torch.empty(src.shape, dtype=dtype, device="cuda")
    _copy_h2d_dst_strided(actual, src)
    assert _same_bytes(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h2d_strided_dst_view():
    """The destination may itself be a narrow of the parameter tensor."""
    src = _make_src("transposed", torch.float32)
    param = torch.zeros(64, 48, device="cuda")
    dst = param.narrow(0, 8, 32).narrow(1, 0, 48)
    expected = param.clone().narrow(0, 8, 32).narrow(1, 0, 48)
    expected.copy_(src)
    _copy_h2d_dst_strided(dst, src)
    assert _same_bytes(dst, expected)


def test_scalar_and_1d_fallback():
    scalar_src = torch.tensor(3.5)
    scalar_dst = torch.empty(())
    _copy_h2d_dst_strided(scalar_dst, scalar_src)
    assert scalar_dst.item() == pytest.approx(3.5)

    vec_src = torch.arange(8, dtype=torch.float32)[::2]
    vec_dst = torch.empty(4)
    _copy_h2d_dst_strided(vec_dst, vec_src)
    assert torch.equal(vec_dst, vec_src.contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h2d_dtype_conversion():
    """copy_ semantics include implicit dtype casts; the flip must too."""
    src = _make_src("transposed", torch.float32)
    expected = torch.empty(src.shape, dtype=torch.bfloat16, device="cuda")
    expected.copy_(src)
    actual = torch.empty(src.shape, dtype=torch.bfloat16, device="cuda")
    _copy_h2d_dst_strided(actual, src)
    assert _same_bytes(actual, expected)
