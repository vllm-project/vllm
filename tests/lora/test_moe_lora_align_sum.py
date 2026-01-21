# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


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

    return topk_ids.to(current_platform.device_name), token_lora_mapping.to(
        current_platform.device_name
    )


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
        device=current_platform.device_name,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        num_experts,
        dtype=torch.int32,
        device=current_platform.device_name,
    )
    num_tokens_post_pad = torch.zeros(
        (max_loras,),
        dtype=torch.int32,
        device=current_platform.device_name,
    )
    adapter_enabled = torch.ones(
        (max_loras + 1,),
        dtype=torch.int32,
        device=current_platform.device_name,
    )
    lora_ids = torch.arange(
        max_loras + 2,
        dtype=torch.int32,
        device=current_platform.device_name,
    )

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


if __name__ == "__main__":
    pytest.main([__file__])
