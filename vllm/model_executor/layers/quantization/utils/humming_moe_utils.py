# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def humming_moe_align(
    configs: list[int],
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(configs) > 0 and len(configs) % 3 == 0
    # NOTE: we choose moe_block_size based on
    #       num_tokens * top_k (= topk_ids.nelement())
    shape_m = topk_ids.nelement()

    for i in range(len(configs) // 3):
        if shape_m > configs[i * 3] and shape_m <= configs[i * 3 + 1]:
            block_size = configs[i * 3 + 2]
            break
    else:
        raise ValueError(f"Could not find a matching block_size for shape_m={shape_m}")

    return moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
        pad_sorted_ids=False,
        ignore_invalid_experts=True,
    )
