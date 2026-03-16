# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for optimized router GEMM kernel

Run `pytest tests/kernels/moe/test_router_gemm.py`.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("input_dim", [360, 720, 1440, 2880])
@pytest.mark.parametrize("output_dim", [32, 64, 128])
def test_tinygemm2(batch_size, input_dim, output_dim):
    set_random_seed(0)
    x = torch.randn(batch_size, input_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_dim, input_dim, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(output_dim, device="cuda", dtype=torch.bfloat16)

    output = ops.tinygemm2(x, weight, bias)
    output_ref = torch.nn.functional.linear(x, weight, bias)

    assert output.shape == (batch_size, output_dim)
    assert torch.allclose(output, output_ref, rtol=1e-2, atol=1e-2)
