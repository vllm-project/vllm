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


@pytest.mark.parametrize(("batch_size", "hidden_dim"), [(28, 64), (42, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hadacore_partial_chunk(batch_size, hidden_dim, dtype):
    """The last CTA must wait for its valid async loads without reading past x."""
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float32, device="cuda"
    ) / math.sqrt(hidden_dim)
    for _ in range(20):
        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
        expected = (x.float() @ hadamard.T).to(dtype)
        actual = ops.hadacore_transform(x.clone(), inplace=True)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
