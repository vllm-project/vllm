# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.transform import deterministic_hadamard_matrix
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        "These tests require hadacore_transform, not supported on ROCm.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("hidden_dim", [2**n for n in range(10)])
def test_hadacore(batch_size, hidden_dim, dtype=torch.bfloat16, device="cuda"):
    x = torch.eye(hidden_dim, dtype=dtype, device=device)
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float64, device="cuda"
    ) / math.sqrt(hidden_dim)

    y = ops.hadacore_transform(x.clone())
    y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
    assert torch.allclose(y, y_true)

    y = ops.hadacore_transform(y)
    assert torch.allclose(y, x)


@pytest.mark.parametrize(
    "num_rows",
    [1, 17, 53, 65, 165, 173, 257, 513, 1025, 1041, 1537, 2049, 4097, 8193],
)
@pytest.mark.parametrize("hidden_dim", [64, 128, 256])
def test_hadacore_odd_row_counts(
    num_rows, hidden_dim, dtype=torch.bfloat16, device="cuda"
):
    """Verify hadacore_transform handles odd row counts without corruption (#63514)."""
    torch.manual_seed(num_rows)
    x = torch.randn(num_rows, hidden_dim, dtype=dtype, device=device)
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float64, device=device
    ) / math.sqrt(hidden_dim)

    y = ops.hadacore_transform(x.clone(), inplace=True)
    y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
    torch.testing.assert_close(y, y_true, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("inplace", [False, True])
def test_hadacore_non_contiguous(inplace, dtype=torch.bfloat16, device="cuda"):
    """Non-contiguous views must not read or write their padding columns (#70383)."""
    base = torch.full((4, 96), 99.0, dtype=dtype, device=device)
    x = base[:, :64]
    torch.manual_seed(0)
    x.copy_(torch.randn(4, 64, dtype=dtype, device=device))
    expected = ops.hadacore_transform(x.contiguous().clone(), inplace=False)

    y = ops.hadacore_transform(x, inplace=inplace)

    torch.testing.assert_close(y, expected, atol=0.05, rtol=0.02)
    if inplace:
        torch.testing.assert_close(x, expected, atol=0.05, rtol=0.02)
    torch.testing.assert_close(
        base[:, 64:], torch.full_like(base[:, 64:], 99.0), atol=0, rtol=0
    )


@pytest.mark.parametrize("inplace", [False, True])
def test_hadacore_empty_batch(inplace, dtype=torch.bfloat16, device="cuda"):
    """An empty batch must return instead of launching a zero-block grid (#70383)."""
    x = torch.empty((0, 64), dtype=dtype, device=device)
    y = ops.hadacore_transform(x, inplace=inplace)
    assert y.shape == x.shape
    assert y.numel() == 0
