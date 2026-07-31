# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused MoE topk weight-and-reduce kernel.

Run `pytest tests/kernels/moe/test_moe_fused_mul_sum.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [1, 17, 256]
TOP_KS = [2, 8]
HIDDEN_SIZE = 512
NUM_EXPERTS = 16


def _reference(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    valid = topk_ids >= 0
    if expert_map is not None:
        valid &= expert_map[topk_ids.clamp(min=0)] >= 0
    weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
    return (inputs.float() * weights.float().unsqueeze(-1)).sum(dim=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("use_expert_map", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_invalid_slots_excluded(num_tokens: int, top_k: int, use_expert_map: bool):
    """Slots marked -1 must not contribute, and must not be dereferenced.

    The expert GEMM never writes rows for topk slots whose expert id is -1
    (a non-local expert under EP, or an all2all padding row), so those rows
    hold stale workspace data. Filling them with NaN here means any failure to
    mask them shows up as NaN in the output rather than a small numeric drift.
    Indexing `expert_map` with -1 is also an out-of-bounds read.
    """
    set_random_seed(0)
    device = "cuda"

    inputs = torch.randn(
        num_tokens, top_k, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # Mark roughly half the slots invalid and poison the rows behind them.
    invalid = torch.rand(num_tokens, top_k, device=device) < 0.5
    topk_ids[invalid] = -1
    inputs[invalid] = float("nan")

    expert_map = None
    if use_expert_map:
        # Second half of the experts is non-local, so valid ids can still map
        # to -1 -- exercising the second masking condition.
        expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
        local = NUM_EXPERTS // 2
        expert_map[:local] = torch.arange(local, dtype=torch.int32, device=device)
        inputs[(topk_ids >= 0) & (expert_map[topk_ids.clamp(min=0)] < 0)] = float("nan")

    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert not out.isnan().any(), "invalid slots leaked into the reduction"
    ref = _reference(inputs.nan_to_num(), topk_weights, topk_ids, expert_map)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
