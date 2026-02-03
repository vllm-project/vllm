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
@pytest.mark.parametrize("num_experts", [16, 64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [1, 2, 16, 32])
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
    num_virtual_experts = num_experts * max_loras
    max_num_tokens_padded = topk_ids.numel() + num_virtual_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,),
        dtype=torch.int32,
        device="cuda",
    )
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")
    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device="cuda")
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32, device="cuda")

    # call kernel
    ops.moe_lora_align_block_size(
        topk_ids,
        lora_ids,
        adapter_enabled,
        token_lora_mapping,
        num_virtual_experts,
        max_loras,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        None,  # expert_map
    )

    # verify values
    n_tokens_padded = num_tokens_post_pad.item()
    num_blocks = n_tokens_padded // block_size
    top_k = topk_ids.size(1)
    numel = topk_ids.numel()
    num_tokens = topk_ids.size(0)

    # Count expected tokens: tokens with valid lora_id and enabled adapter
    expected_count = 0
    for token_id in range(num_tokens):
        lora_id = token_lora_mapping[token_id].item()
        if 0 <= lora_id < max_loras and adapter_enabled[lora_id].item() == 1:
            expected_count += top_k  # Each token has top_k expert slots

    # Count actual sorted tokens and verify assignments
    actual_count = 0
    for block_idx in range(num_blocks):
        virtual_expert = expert_ids[block_idx].item()
        if virtual_expert < 0:
            continue

        # Decode virtual expert -> (lora_id, expert_id)
        lora_id = virtual_expert // num_experts
        expert_id = virtual_expert % num_experts

        block_start = block_idx * block_size
        block_end = min(block_start + block_size, n_tokens_padded)

        for token_idx in sorted_token_ids[block_start:block_end]:
            token_idx_val = token_idx.item()
            if token_idx_val >= numel:
                # Padding token
                continue

            actual_count += 1
            orig_token = token_idx_val // top_k

            # Verify LoRA assignment
            assert token_lora_mapping[orig_token].item() == lora_id, "Lora_id mismatch."

            # Verify expert assignment
            assert topk_ids.view(-1)[token_idx_val].item() == expert_id, (
                "Expert_id mismatch."
            )

    # Verify all expected tokens were sorted
    assert actual_count == expected_count, "Token count mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
