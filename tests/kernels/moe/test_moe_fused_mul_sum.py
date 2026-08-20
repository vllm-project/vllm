# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused MoE topk weight-and-reduce kernel.

Run `pytest tests/kernels/moe/test_moe_fused_mul_sum.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
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

    # The _reference implementation zeros out outputs. We need to match that here,
    # otherwise the test fails.
    outputs = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert not out.isnan().any(), "invalid slots leaked into the reduction"
    ref = _reference(inputs.nan_to_num(), topk_weights, topk_ids, expert_map)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def _local_expert_map(device: str) -> torch.Tensor:
    """First half of the experts is local, second half maps to -1 (non-local)."""
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    local = NUM_EXPERTS // 2
    expert_map[:local] = torch.arange(local, dtype=torch.int32, device=device)
    return expert_map


SENTINEL = 42.0  # exact in bf16; marks output the kernel must leave untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nonlocal_row_is_zeroed():
    """A real row whose positive ids all map to non-local experts is written 0.

    Such a row is not padding (its ids are >= 0, so row_valid > 0 and the block
    is not early-returned). Every slot is then masked out by expert_map, so the
    accumulator stays 0 and 0 is stored -- the row must not be left holding stale
    buffer data, because downstream reduction treats it as a real token.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    local = NUM_EXPERTS // 2
    expert_map = _local_expert_map(device)

    # Poison inputs so any failure to mask the non-local slots surfaces as NaN.
    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    # Every id lands in the non-local half [local, NUM_EXPERTS).
    topk_ids = torch.randint(
        local, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert not out.isnan().any(), "non-local row leaked NaN into the output"
    torch.testing.assert_close(out.float(), torch.zeros_like(out.float()))


@pytest.mark.parametrize("use_expert_map", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_padding_row_left_untouched(use_expert_map: bool):
    """An all -1 padding row is skipped: its output row is left untouched.

    This is the CUDA-graph decode contract -- padding rows past num_recv are
    never read downstream, so the kernel elides them (whole-padding blocks
    early-return) rather than zeroing. The skip is gated on topk_ids alone, so
    it must hold whether or not expert_map is passed (the humming path passes
    None); we pin both.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    expert_map = _local_expert_map(device) if use_expert_map else None

    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    topk_ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=device)

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    assert torch.all(out == SENTINEL), "padding row output was modified"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nonlocal_and_padding_adjacent():
    """A non-local real row next to an all -1 padding row.

    Each CTA owns one token row: the non-local row must be zeroed (its slots are
    all masked by expert_map) while the padding row beside it is left untouched.
    Guards against a change that lets one row's fate leak into its neighbour.
    """
    set_random_seed(0)
    device = "cuda"
    num_tokens, top_k = 17, 8
    local = NUM_EXPERTS // 2
    expert_map = _local_expert_map(device)

    inputs = torch.full(
        (num_tokens, top_k, HIDDEN_SIZE),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    # All rows are non-local real rows except row 1, an all -1 padding row.
    topk_ids = torch.randint(
        local, NUM_EXPERTS, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[1] = -1

    outputs = torch.full(
        (num_tokens, HIDDEN_SIZE), SENTINEL, dtype=torch.bfloat16, device=device
    )
    out = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        outputs=outputs,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    padding[1] = True
    assert torch.all(out[1] == SENTINEL), "padding row beside a real row was modified"
    written = out[~padding].float()
    assert not written.isnan().any(), "non-local row leaked NaN into the output"
    torch.testing.assert_close(written, torch.zeros_like(written))
