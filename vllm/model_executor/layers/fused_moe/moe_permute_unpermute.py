import torch
from typing import List
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk


def moe_permute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    topk: int,
    n_expert: int,
) -> List[torch.Tensor]:
    n_token, n_hidden = hidden_states.shape
    permuted_hidden_states = torch.empty(
        (n_token * topk, n_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_first_token_offset = torch.empty(
        n_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    src_row_id2dst_row_id_map = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_weights,
        topk_ids,
        token_expert_indices,
        n_expert,
        topk,
        permuted_hidden_states,
        expert_first_token_offset,
        src_row_id2dst_row_id_map,
    )
    return [
        permuted_hidden_states,
        expert_first_token_offset,
        src_row_id2dst_row_id_map,
    ]


def moe_unpermute(permuted_hidden_states: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  token_expert_indices: torch.Tensor,
                  src_row_id2dst_row_id_map: torch.Tensor,
                  topk: int,
                  n_expert: int 
) -> torch.Tensor:
    n_token, n_hidden = topk_weights.shape[0], permuted_hidden_states.shape[-1]
    hidden_states = torch.empty((n_token, n_hidden), dtype=permuted_hidden_states.dtype, 
                                device=permuted_hidden_states.device)

    torch.ops._moe_C.moe_unpermute(permuted_hidden_states, topk_weights,
                                   topk_ids, src_row_id2dst_row_id_map,
                                   n_expert, topk, hidden_states)
    return hidden_states

