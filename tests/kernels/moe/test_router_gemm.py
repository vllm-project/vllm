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
    not (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(100)
        )
    ),
    reason="This test only runs on CUDA Hopper or Blackwell platform.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("input_dim", [360, 720, 1440, 2880])
@pytest.mark.parametrize("output_dim", [32, 64, 128])
def test_gpt_oss_router_gemm(batch_size, input_dim, output_dim):
    set_random_seed(0)
    x = torch.randn(batch_size, input_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_dim, input_dim, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(output_dim, device="cuda", dtype=torch.bfloat16)

    output = ops.gpt_oss_router_gemm(x, weight, bias)
    output_ref = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
