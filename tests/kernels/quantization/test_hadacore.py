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


@pytest.mark.parametrize("hidden_dim", [2**n for n in range(10)])
@pytest.mark.parametrize("inplace", [False, True])
def test_hadacore(hidden_dim, inplace, dtype=torch.bfloat16, device="cuda"):
    x = torch.eye(hidden_dim, dtype=dtype, device=device)
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float64, device="cuda"
    ) / math.sqrt(hidden_dim)

    y = ops.hadacore_transform(x.clone(), inplace=inplace)
    y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
    assert torch.allclose(y, y_true)

    y = ops.hadacore_transform(y, inplace=inplace)
    assert torch.allclose(y, x)
