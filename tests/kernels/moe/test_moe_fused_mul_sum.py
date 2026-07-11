# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for moe_fused_mul_sum, the fused weighted-sum-of-experts kernel.

Guards against out-of-bounds expert_map reads when topk_ids contains -1
sentinels for invalid token/expert pairs under expert parallelism
(https://github.com/vllm-project/vllm/issues/47281).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("CUDA-like platform required", allow_module_level=True)


def ref_impl(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor | None,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    weights = topk_weights.float()
    if expert_map is not None:
        assert topk_ids is not None
        valid = topk_ids >= 0
        local_ids = expert_map[topk_ids.clamp(min=0)]
        valid &= local_ids >= 0
        weights = weights * valid
    out = (inputs.float() * weights.unsqueeze(-1)).sum(dim=1)
    return out.to(inputs.dtype)


def make_expert_map(
    num_global_experts: int, num_local_experts: int, device: str
) -> torch.Tensor:
    expert_map = torch.full((num_global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )
    return expert_map


@pytest.mark.parametrize("num_tokens", [1, 7, 256])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("hidden_size", [128, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_moe_fused_mul_sum(num_tokens, top_k, hidden_size, dtype):
    torch.manual_seed(0)
    device = "cuda"
    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)

    out = moe_fused_mul_sum(inputs, topk_weights)
    ref = ref_impl(inputs, topk_weights, None, None)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 7, 256])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("hidden_size", [128, 2048])
@pytest.mark.parametrize("with_invalid_ids", [False, True])
def test_moe_fused_mul_sum_expert_map(num_tokens, top_k, hidden_size, with_invalid_ids):
    """Expert-parallel case: entries routed to other ranks (expert_map == -1)
    must not contribute, and -1 sentinels in topk_ids must not cause
    out-of-bounds reads of expert_map."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    num_global_experts, num_local_experts = 64, 16

    inputs = torch.randn(num_tokens, top_k, hidden_size, dtype=dtype, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    topk_ids = torch.randint(
        0, num_global_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    if with_invalid_ids:
        invalid = torch.rand(topk_ids.shape, device=device) < 0.3
        topk_ids[invalid] = -1
    expert_map = make_expert_map(num_global_experts, num_local_experts, device)

    out = moe_fused_mul_sum(
        inputs, topk_weights, topk_ids=topk_ids, expert_map=expert_map
    )
    torch.cuda.synchronize()
    ref = ref_impl(inputs, topk_weights, topk_ids, expert_map)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
