# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.library import custom_op

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    get_moe_align_tensor_size,
    moe_align_block_size,
)


@custom_op("humming::humming_moe_align", mutates_args=())
def humming_moe_align(
    configs: list[int],
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(configs) > 0 and len(configs) % 3 == 0
    shape_m = topk_ids.size(0)

    for i in range(len(configs) // 3):
        if shape_m >= configs[i * 3] and shape_m <= configs[i * 3 + 1]:
            block_size = configs[i * 3 + 2]
            break
    else:
        raise ValueError(f"Could not find a matching block_size for shape_m={shape_m}")

    def make_empty_tensor(size):
        return torch.empty((size,), dtype=torch.int32, device=topk_ids.device)

    def get_tensor_size(target_block_size):
        return get_moe_align_tensor_size(
            topk_ids=topk_ids,
            num_experts=num_experts,
            block_size=target_block_size,
            pad_sorted_ids=False,
        )

    max_sorted_ids_size = get_tensor_size(64)[0]
    max_expert_ids_size = get_tensor_size(8)[1]

    sorted_ids = make_empty_tensor(max_sorted_ids_size)
    expert_ids = make_empty_tensor(max_expert_ids_size)
    num_tokens_post_pad = make_empty_tensor(1)

    sorted_ids_size, experts_ids_size = get_tensor_size(block_size)

    moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
        pad_sorted_ids=False,
        ignore_invalid_experts=True,
        sorted_ids=sorted_ids[:sorted_ids_size],
        expert_ids=expert_ids[:experts_ids_size],
        num_tokens_post_pad=num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
