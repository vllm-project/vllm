# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm import _custom_ops as ops


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def sample_data(num_experts, max_loras, num_tokens, topk_num):
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.zeros((num_tokens,), dtype=torch.int32)

    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        token_lora_mapping[i] = random.randint(0, max_loras - 1)

    return topk_ids.to("cuda"), token_lora_mapping.to("cuda")


@pytest.mark.parametrize("num_tokens", [100, 200, 1024, 4096])  # 81920
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [2, 32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(
    num_tokens, topk_num, num_experts, max_loras, block_size
):
    # sample data
    random.seed(1)
    topk_ids, token_lora_mapping = sample_data(
        num_experts, max_loras, num_tokens, topk_num
    )

    # compute paddings
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device="cuda",
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,), num_experts, dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.zeros((max_loras,), dtype=torch.int32, device="cuda")
    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device="cuda")
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32, device="cuda")

    # call kernel
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    # verify values
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1, block_size)

    for lora_idx in range(max_loras):
        for token_idx in range(sorted_token_ids.size(1)):
            block = sorted_token_ids[lora_idx][token_idx]
            indices = block[block != topk_ids.numel()]
            if indices.numel() > 0:
                expert_id = expert_ids[lora_idx][token_idx]
                assert torch.all(topk_ids.view(-1)[indices] == expert_id)


@pytest.mark.parametrize("num_tokens", [100, 1024])
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("global_num_experts", [64])
@pytest.mark.parametrize("local_num_experts", [8])
@pytest.mark.parametrize("max_loras", [2])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size_with_expert_map(
    num_tokens,
    topk_num,
    global_num_experts,
    local_num_experts,
    max_loras,
    block_size,
):
    """Test that expert_map correctly remaps global expert IDs to local."""
    random.seed(1)
    topk_ids, token_lora_mapping = sample_data(
        global_num_experts, max_loras, num_tokens, topk_num
    )

    # Create expert_map: first local_num_experts are local (0..local-1),
    # rest are -1 (not on this rank)
    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.int32, device="cuda"
    )
    for i in range(local_num_experts):
        expert_map[i] = i

    # Use global_num_experts for padding (matches act_decorator change)
    max_num_tokens_padded = topk_ids.numel() + global_num_experts * (
        block_size - 1
    )
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device="cuda",
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    num_tokens_post_pad = torch.zeros(
        (max_loras,), dtype=torch.int32, device="cuda"
    )
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device="cuda"
    )
    lora_ids = torch.arange(
        max_loras + 2, dtype=torch.int32, device="cuda"
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        global_num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        expert_map,
    )

    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1, block_size)

    for lora_idx in range(max_loras):
        for token_idx in range(sorted_token_ids.size(1)):
            block = sorted_token_ids[lora_idx][token_idx]
            indices = block[block != topk_ids.numel()]
            if indices.numel() > 0:
                expert_id = expert_ids[lora_idx][token_idx].item()
                # Expert IDs should be local (0..local-1) or -1
                assert -1 <= expert_id < local_num_experts, (
                    f"Expected local expert ID in [-1, {local_num_experts}), "
                    f"got {expert_id}"
                )
                # All tokens in block should map to same global expert,
                # and that global expert should map to this local ID
                global_ids = topk_ids.view(-1)[indices]
                for gid in global_ids:
                    assert expert_map[gid.item()].item() == expert_id


if __name__ == "__main__":
    pytest.main([__file__])
