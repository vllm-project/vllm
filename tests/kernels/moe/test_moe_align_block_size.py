# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE align block size function.

Run `pytest tests/kernels/moe/test_moe_align_block_size.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size,
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [1, 3, 256, 2256, 4096]
NUM_EXPERTS = [32, 160, 256, 257]
TOP_KS = [1, 2, 16, 32]
BLOCK_SIZES = [32, 128]
set_random_seed(0)


def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """
    Verify that actual_sorted_ids follows the correct expert-level sorting.
    The kerne limplementation may or may not preserve original token order
    in topk_ids in the final sorted_ids however this does not impact quality.
    """
    # Group tokens by expert from the golden implementation
    golden_expert_tokens = _group_tokens_by_expert(
        golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in golden_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"golden={golden_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block : current_block + expert_blocks_needed] = (
                expert_id_new
            )

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad


@pytest.mark.parametrize("m", NUM_TOKENS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("pad_sorted_ids", [False, True])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_moe_align_block_size(
    m: int, topk: int, num_experts: int, block_size: int, pad_sorted_ids: bool
):
    """Test moe_align_block_size without expert mapping"""
    topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="cuda")[:topk]
        topk_ids[i] = experts

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            pad_sorted_ids=pad_sorted_ids,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)

    # For sorted_token_ids, verify block-level correctness rather than exact
    # order Tokens within each expert's blocks can be in any order, but expert
    # regions must be correct
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )

    total_tokens = m * topk
    assert actual_num_tokens.item() % block_size == 0, (
        "num_tokens_post_pad should be divisible by block_size"
    )
    assert actual_num_tokens.item() >= total_tokens, (
        "num_tokens_post_pad should be at least total_tokens"
    )
    valid_tokens = actual_sorted_ids[actual_sorted_ids < total_tokens]
    assert len(valid_tokens) == total_tokens, (
        f"Should have exactly {total_tokens} valid tokens, got {len(valid_tokens)}"
    )
    assert (actual_expert_ids >= 0).all() and (actual_expert_ids < num_experts).all(), (
        "expert_ids should contain valid expert indices"
    )


@pytest.mark.parametrize("m", [16, 32, 2048])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_moe_align_block_size_with_expert_map(
    m: int, topk: int, num_experts: int, block_size: int
):
    """Test moe_align_block_size with expert mapping (EP scenario)"""
    topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="cuda")[:topk]
        topk_ids[i] = experts

    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            expert_map=expert_map,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )


def test_moe_align_block_size_deterministic():
    m, topk, num_experts, block_size = 128, 2, 32, 64

    torch.manual_seed(42)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), device="cuda", dtype=torch.int32
    )

    # expect the results to be reproducible
    results = []
    for _ in range(5):
        sorted_ids, expert_ids, num_tokens = moe_align_block_size(
            topk_ids=topk_ids, block_size=block_size, num_experts=num_experts
        )
        results.append((sorted_ids.clone(), expert_ids.clone(), num_tokens.clone()))

    for i in range(1, len(results)):
        assert torch.equal(results[0][0], results[i][0]), (
            "sorted_ids should be deterministic"
        )
        assert torch.equal(results[0][1], results[i][1]), (
            "expert_ids should be deterministic"
        )
        assert torch.equal(results[0][2], results[i][2]), (
            "num_tokens should be deterministic"
        )


@pytest.mark.parametrize("max_tokens_per_batch", [13, 16, 512])
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
@pytest.mark.parametrize("simulate_empty_batches", [False, True])
def test_batched_moe_align_block_size(
    max_tokens_per_batch: int,
    num_experts: int,
    block_size: int,
    simulate_empty_batches: bool,
):
    def ref_outputs(
        expert_num_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = expert_num_tokens.size(0)

        # Round up so each batch can be split to blocks evenly.
        Msum = round_up(max_tokens_per_batch, block_size) * E
        ref_sorted_ids = torch.empty((Msum,), dtype=torch.int32)
        ref_expert_ids = torch.empty((Msum // block_size,), dtype=torch.int32)
        ref_num_tokens_post_pad = torch.empty((1,), dtype=torch.int32)

        # Intialize
        sentinel = E * max_tokens_per_batch
        ref_sorted_ids.fill_(sentinel)
        ref_expert_ids.fill_(-1)

        # Fill ref_sorted_ids
        i = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            token_offset = expert_id * max_tokens_per_batch
            for j in range(expert_nt):
                ref_sorted_ids[i] = token_offset + j
                i += 1
            # round up i to the next block_size
            i = round_up(i, block_size)

        ref_num_tokens_post_pad[0] = i

        # Fill expert_ids
        nt_ceil_sum = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            expert_ids_offset = nt_ceil_sum // block_size
            ceil_expert_nt = round_up(int(expert_nt.item()), block_size)
            num_blocks = ceil_expert_nt // block_size
            for x in range(num_blocks):
                ref_expert_ids[expert_ids_offset + x] = expert_id
            nt_ceil_sum += ceil_expert_nt

        return (
            ref_sorted_ids.to("cuda"),
            ref_expert_ids.to("cuda"),
            ref_num_tokens_post_pad.to("cuda"),
        )

    # Compute expert_num_tokens
    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch,
        size=(num_experts,),
        device="cpu",
        dtype=torch.int32,
    )
    if simulate_empty_batches:
        # mark half the batches to have 0 tokens
        zero_batches = torch.randperm(num_experts)[: num_experts // 2]
        expert_num_tokens[zero_batches] = 0

    # ref outputs
    ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad = ref_outputs(
        expert_num_tokens
    )

    # outputs
    sorted_ids, expert_ids, num_tokens_post_pad = batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens.to("cuda")
    )

    assert ref_sorted_ids.size() == sorted_ids.size(), (
        f"{ref_sorted_ids.size()} vs {sorted_ids.size()}"
    )
    assert ref_expert_ids.size() == expert_ids.size(), (
        f"{ref_expert_ids.size()} vs {expert_ids.size()}"
    )
    assert ref_num_tokens_post_pad.size() == num_tokens_post_pad.size(), (
        f"{ref_num_tokens_post_pad.size()} vs {num_tokens_post_pad.size()}"
    )
    torch.testing.assert_close(ref_sorted_ids, sorted_ids, atol=0, rtol=0)
    torch.testing.assert_close(ref_expert_ids, expert_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        ref_num_tokens_post_pad, num_tokens_post_pad, atol=0, rtol=0
    )
