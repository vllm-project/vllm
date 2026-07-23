# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TopKWeightAndReduceContiguous.

Covers both the eager (``mul_`` + ``moe_sum``) path and the fused
``moe_fused_mul_sum`` path, which is selected by tensor size, so the shapes
below straddle the ``_FUSED_MUL_SUM_MIN_NUMEL`` threshold.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    _FUSED_MUL_SUM_MIN_NUMEL,
    TopKWeightAndReduceContiguous,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
# num_tokens=8 stays under the fused threshold; 512 crosses it for top_k=8.
@pytest.mark.parametrize("num_tokens", [8, 512])
@pytest.mark.parametrize("hidden", [2048, 7168])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
@pytest.mark.parametrize("preallocated_output", [False, True])
def test_topk_weight_and_reduce_contiguous(
    num_tokens: int,
    hidden: int,
    top_k: int,
    apply_router_weight_on_input: bool,
    preallocated_output: bool,
):
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    fused_expert_output = torch.randn(
        num_tokens, top_k, hidden, device=device, dtype=dtype
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_ids = torch.randint(
        0, 64, (num_tokens, top_k), device=device, dtype=torch.int64
    )

    output = None
    if preallocated_output:
        output = torch.empty(num_tokens, hidden, device=device, dtype=dtype)

    if apply_router_weight_on_input:
        # Weights are assumed already applied upstream: reduce is a plain sum.
        ref = fused_expert_output.float().sum(dim=1)
    else:
        ref = (fused_expert_output.float() * topk_weights.float().unsqueeze(-1)).sum(
            dim=1
        )

    result = TopKWeightAndReduceContiguous().apply(
        output=output,
        # apply() may mutate the input in the eager branch; pass a copy.
        fused_expert_output=fused_expert_output.clone(),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )

    assert result.shape == (num_tokens, hidden)
    if preallocated_output:
        assert result.data_ptr() == output.data_ptr()
    # Tolerance accommodates bf16 rounding; a wrong reduction/weighting would
    # miss by ~O(weight * value), far outside this band.
    torch.testing.assert_close(result.float(), ref, atol=1e-1, rtol=5e-2)


def test_fused_threshold_is_reachable():
    # Guard against a regression that sets the gate so high the fused path is
    # effectively dead for realistic high-throughput EP shapes.
    assert _FUSED_MUL_SUM_MIN_NUMEL <= 512 * 8 * 4096
