# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

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
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Determine the sorted_token_ids, expert_ids for the given problem size.
    Permute the hidden states and scales according to `sorted_token_ids`.
    """
    top_k_num = curr_topk_ids.size(1)

    tokens_in_chunk = curr_hidden_states.size(0)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(curr_topk_ids,
                             block_m,
                             global_num_experts,
                             expert_map,
                             pad_sorted_ids=True))

    inv_perm: Optional[torch.Tensor] = None

    num_tokens = top_k_num * tokens_in_chunk
    expert_ids = torch.repeat_interleave(expert_ids, block_m, dim=0)
    inv_perm = torch.argsort(sorted_token_ids)[:num_tokens]

    # Permute according to sorted token ids.
    sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)

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
    apply_router_weight_on_input: bool,
) -> None:
    """
    Unpermute the final result and apply topk_weights, then perform the final
    reduction on the hidden states.
    """
    M, topk = topk_weight.size()
    K = curr_hidden.size(-1)
    if inv_perm is not None:
        curr_hidden = curr_hidden[inv_perm, ...]
    curr_hidden = curr_hidden.view(-1, topk, K)
    if not apply_router_weight_on_input:
        curr_hidden.mul_(topk_weight.view(M, -1, 1))
    ops.moe_sum(curr_hidden, out)


def moe_permute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    topk: int,
    n_expert: int,
    n_local_expert: int,
    expert_map: Optional[torch.Tensor] = None,
    align_block_size: Optional[int] = None,
    fill_invalid_expert: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function expands and permutes activation to gather uncontinuous tokens
      for each expert.
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - topk_weights (torch.Tensor): topk expert route weight for each token.
    - topk_ids (torch.Tensor): topk expert route id for each token.
    - token_expert_indices (torch.Tensor): indice for expanded hidden.
    - topk (int): The number of top-k experts to select.
    - n_expert (int): The number of expert.
    - n_local_expert (int): The number of expert in current EP rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - align_block_size (Optional[int]): align group gemm block size for deepgemm
    - fill_invalid_expert(int): fill expert id in m_indices for invalid expert
      to workaround DeepGemm unsupported -1 in m_indices
    Returns:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for standard grouped gemm. if enable 'align_block_size'
       expert_first_token_offset will align up to 'align_block_size'.
    - src_row_id2dst_row_id_map (torch.Tensor): idx map for moe_unpermute.
    - m_indices: m_indices for grouped gemm in deepgemm,`m_indices[i]` records
    the group which the j-th row of the LHS belong to.`
    """
    n_token, n_hidden = hidden_states.size()
    assert (n_hidden * hidden_states.element_size()
            ) % 16 == 0, "permue kernel need hidden dim align to 16B"
    permuted_row_size = n_token * topk
    if align_block_size is not None:
        permuted_row_size = (permuted_row_size + n_expert *
                             (align_block_size - 1) + align_block_size -
                             1) // align_block_size * align_block_size

    permuted_hidden_states = torch.empty(
        (permuted_row_size, n_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    m_indices = torch.full((permuted_row_size, ),
                           fill_invalid_expert,
                           dtype=torch.int32,
                           device=hidden_states.device)
    expert_first_token_offset = torch.empty(n_local_expert + 1,
                                            dtype=torch.int64,
                                            device=hidden_states.device)
    src_row_id2dst_row_id_map = torch.empty((n_token, topk),
                                            dtype=torch.int32,
                                            device=hidden_states.device)
    torch.ops._moe_C.moe_permute(hidden_states, topk_weights, topk_ids,
                                 token_expert_indices, expert_map, n_expert,
                                 n_local_expert, topk, align_block_size,
                                 permuted_hidden_states,
                                 expert_first_token_offset,
                                 src_row_id2dst_row_id_map, m_indices)
    return (permuted_hidden_states, expert_first_token_offset,
            src_row_id2dst_row_id_map, m_indices)


def moe_unpermute(
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_row_id2dst_row_id_map: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
    topk: int,
    n_expert: int,
    n_local_expert: int,
) -> torch.Tensor:
    """
    This function expands and permutes activation to gathering uncontinuous
      tokens for each expert.
    Parameters:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - topk_weights (torch.Tensor): topk expert route weight for each token.
    - topk_ids (torch.Tensor): topk expert route id for each token.
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for grouped gemm.
    - topk (int): The number of top-k experts to select.
    - n_expert (int): The number of expert.
    - n_local_expert (int): The number of expert in current EP rank.
    Returns:
    - hidden_states (torch.Tensor): The reduced and unpermuted activation
      tensor.
    """
    n_token, n_hidden = topk_weights.size(0), permuted_hidden_states.size(-1)
    assert (n_hidden * permuted_hidden_states.element_size()
            ) % 16 == 0, "unpermue kernel need hidden dim align to 16B"
    hidden_states = torch.empty((n_token, n_hidden),
                                dtype=permuted_hidden_states.dtype,
                                device=permuted_hidden_states.device)

    torch.ops._moe_C.moe_unpermute(permuted_hidden_states, topk_weights,
                                   topk_ids, src_row_id2dst_row_id_map,
                                   expert_first_token_offset, n_expert,
                                   n_local_expert, topk, hidden_states)
    return hidden_states


def moe_permute_unpermute_supported():
    return torch.ops._moe_C.moe_permute_unpermute_supported()
