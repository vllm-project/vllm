# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.fused_moe.utils import _fp8_perm


def _moe_permute(
    curr_hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    curr_topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    block_m: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
           Optional[torch.Tensor]]:
    """
    Determine the sorted_token_ids, expert_ids for the given problem size.
    Permute the hidden states and scales according to `sorted_token_ids`.
    """
    top_k_num = curr_topk_ids.shape[1]

    tokens_in_chunk, _ = curr_hidden_states.shape

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(curr_topk_ids,
                             block_m,
                             global_num_experts,
                             expert_map,
                             pad_sorted_ids=True))

    inv_perm: Optional[torch.Tensor] = None

    num_tokens = top_k_num * tokens_in_chunk
    sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
    expert_ids = torch.repeat_interleave(expert_ids, block_m, dim=0)
    inv_perm = torch.argsort(sorted_token_ids)[:num_tokens]

    # Permute according to sorted token ids.
    curr_hidden_states = _fp8_perm(curr_hidden_states,
                                   sorted_token_ids // top_k_num)

    if a1q_scale is not None:
        a1q_scale = a1q_scale[sorted_token_ids // top_k_num]

    return (curr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
            inv_perm)


def _moe_unpermute_and_reduce(
    out: torch.Tensor,
    curr_hidden: torch.Tensor,
    inv_perm: Optional[torch.Tensor],
    topk_weight: torch.Tensor,
) -> None:
    """
    Unpermute the final result and apply topk_weights, then perform the final
    reduction on the hidden states.
    """
    M, topk = topk_weight.shape
    K = curr_hidden.shape[-1]
    if inv_perm is not None:
        curr_hidden = curr_hidden[inv_perm, ...]
    curr_hidden = curr_hidden.view(-1, topk, K)
    curr_hidden.mul_(topk_weight.view(M, -1, 1))
    ops.moe_sum(curr_hidden, out)
