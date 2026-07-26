# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the moe_fused_mul_sum Triton kernel.

Tests the fused weighted summation of MoE expert outputs, both with and
without expert-map masking for Expert Parallelism.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

NUM_TOKENS = [1, 16, 64, 128]
TOP_KS = [2, 4, 8]
HIDDEN_SIZES = [128, 512, 4096]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SEEDS = [0]


def ref_moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return (
        inputs.float() * topk_weights.unsqueeze(-1).float()
    ).sum(dim=1).to(inputs.dtype)


def ref_moe_fused_mul_sum_with_expert_map(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    valid = expert_map[topk_ids] >= 0
    return (
        inputs.float()
        * topk_weights.unsqueeze(-1).float()
        * valid.unsqueeze(-1).float()
    ).sum(dim=1).to(inputs.dtype)


def get_tolerances(dtype: torch.dtype):
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1.3e-6)
    return dict(atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton kernel requires CUDA, ROCm, or XPU",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_moe_fused_mul_sum(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
):
    torch.set_default_device(DEVICE)
    set_random_seed(seed)

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype)
    topk_weights = torch.randn(num_tokens, top_k, dtype=dtype).softmax(dim=-1)

    result = moe_fused_mul_sum(inputs, topk_weights)
    expected = ref_moe_fused_mul_sum(inputs, topk_weights)

    torch.testing.assert_close(result, expected, **get_tolerances(dtype))


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton kernel requires CUDA, ROCm, or XPU",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_moe_fused_mul_sum_with_expert_map(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
):
    torch.set_default_device(DEVICE)
    set_random_seed(seed)

    num_experts = 16
    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype)
    topk_weights = torch.randn(num_tokens, top_k, dtype=dtype).softmax(dim=-1)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))

    # Expert map: mark some experts as invalid (-1) to simulate
    # Expert Parallelism where not all experts are on this rank
    expert_map = torch.arange(num_experts)
    expert_map[num_experts // 2:] = -1

    result = moe_fused_mul_sum(
        inputs,
        topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    expected = ref_moe_fused_mul_sum_with_expert_map(
        inputs,
        topk_weights,
        topk_ids,
        expert_map,
    )

    torch.testing.assert_close(result, expected, **get_tolerances(dtype))
