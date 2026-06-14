# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant kernel implementations for common ops.

Verifies correctness (vs torch reference) and the batch-invariance property
(result for one item is bitwise identical regardless of other items in the
batch) for:
  - bmm (Triton bmm_kernel)
  - log_softmax (Triton _log_softmax_kernel)
  - softmax (pure PyTorch deterministic implementation)
  - mean (Triton mean_kernel)
"""

import pytest
import torch
from utils import skip_unsupported_device

from vllm.model_executor.layers.batch_invariant import (
    bmm_batch_invariant,
    log_softmax,
    mean_batch_invariant,
    mean_dim,
    softmax_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


# ---------------------------------------------------------------------------
# BMM tests — bmm_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_device
@pytest.mark.parametrize(
    "B,M,K,N",
    [
        (1, 32, 64, 16),
        (4, 64, 128, 64),
        (8, 512, 1024, 512),
        (2, 1, 64, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_correctness(B, M, K, N, dtype):
    """bmm_batch_invariant matches torch.bmm within tolerance."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a = torch.randn(B, M, K, dtype=dtype, device=device)
    b = torch.randn(B, K, N, dtype=dtype, device=device)

    expected = torch.bmm(a, b)
    actual = bmm_batch_invariant(a, b)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_device
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_batch_invariance(dtype):
    """Same slice gives bitwise-identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    a_single = torch.randn(1, 64, 32, dtype=dtype, device=device)
    b_single = torch.randn(1, 32, 128, dtype=dtype, device=device)

    out_single = bmm_batch_invariant(a_single, b_single)

    # Embed the same slice in a larger batch with random neighbors
    a_batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    b_batch = torch.randn(8, 32, 128, dtype=dtype, device=device)
    a_batch[5] = a_single[0]
    b_batch[5] = b_single[0]

    out_batch = bmm_batch_invariant(a_batch, b_batch)

    assert torch.equal(out_single[0], out_batch[5])


# ---------------------------------------------------------------------------
# Log-softmax tests — _log_softmax_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_device
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_correctness(rows, cols, dtype):
    """Triton log_softmax matches torch.log_softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.log_softmax(x, dim=-1)
    actual = log_softmax(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_device
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = log_softmax(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = log_softmax(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Softmax tests — pure PyTorch (exp/sum, no Triton kernel)
# ---------------------------------------------------------------------------


@skip_unsupported_device
@pytest.mark.parametrize("rows,cols", [(1, 128), (16, 1024), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_correctness(rows, cols, dtype):
    """Deterministic softmax matches torch.softmax."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(rows, cols, dtype=dtype, device=device)

    expected = torch.softmax(x, dim=-1)
    actual = softmax_batch_invariant(x, dim=-1)

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_device
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_batch_invariance(dtype):
    """Same row gives identical result regardless of other rows."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 2048, dtype=dtype, device=device)
    out_single = softmax_batch_invariant(row, dim=-1)

    batch = torch.randn(16, 2048, dtype=dtype, device=device)
    batch[7] = row[0]
    out_batch = softmax_batch_invariant(batch, dim=-1)

    assert torch.equal(out_single[0], out_batch[7])


# ---------------------------------------------------------------------------
# Mean reduction tests — mean_kernel Triton kernel
# ---------------------------------------------------------------------------


@skip_unsupported_device
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((16, 128), 1),
        ((4, 64, 32), 1),
        ((8, 256), 0),
        ((2, 16, 64), 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_dim_correctness(shape, dim, dtype):
    """Triton mean_dim matches torch.mean."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    x = torch.randn(shape, dtype=dtype, device=device)

    expected = torch.mean(x.float(), dim=dim).to(dtype)
    actual = mean_dim(x, dim=dim)
    if actual.dtype != dtype:
        actual = actual.to(dtype)

    rtol, atol = (1e-2, 1e-2) if dtype == torch.float32 else (1e-1, 1e-1)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@skip_unsupported_device
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_batch_invariance(dtype):
    """Same slice gives identical result regardless of batch neighbors."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 512, dtype=dtype, device=device)
    out_single = mean_dim(row, dim=1)

    batch = torch.randn(8, 512, dtype=dtype, device=device)
    batch[3] = row[0]
    out_batch = mean_dim(batch, dim=1)

    assert torch.equal(out_single[0], out_batch[3])


@skip_unsupported_device
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_multi_dim_batch_invariance(dtype):
    """mean_batch_invariant over multiple dims is batch-invariant."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)

    row = torch.randn(1, 64, 32, dtype=dtype, device=device)
    out_single = mean_batch_invariant(row, dim=[1, 2])

    batch = torch.randn(8, 64, 32, dtype=dtype, device=device)
    batch[5] = row[0]
    out_batch = mean_batch_invariant(batch, dim=[1, 2])

    assert torch.equal(out_single[0], out_batch[5])
