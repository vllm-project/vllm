import torch
from vllm import _custom_ops as ops
from typing import Optional


def moe_permute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    topk: int,
    n_expert: int,
    n_local_expert:int,
    expert_map: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
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
    - n_localexpert (int): The number of expert in current EP rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices 
        from the global expert space to the local expert space of the expert 
        parallel shard.
    Returns:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for grouped gemm.
    - src_row_id2dst_row_id_map (torch.Tensor): idx map for moe_unpermute.
    """
    n_token, n_hidden = hidden_states.shape
    permuted_hidden_states = torch.empty(
        (n_token * topk, n_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_first_token_offset = torch.empty(
        n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    src_row_id2dst_row_id_map = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_weights,
        topk_ids,
        token_expert_indices,
        expert_map, 
        n_expert,
        n_local_expert,
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
                  src_row_id2dst_row_id_map: torch.Tensor,
                  topk: int,
                  n_expert: int 
) -> torch.Tensor:
    """
    This function expands and permutes activation to gathering uncontinuous 
      tokens for each expert.
    Parameters:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - topk_weights (torch.Tensor): topk expert route weight for each token.
    - topk_ids (torch.Tensor): topk expert route id for each token.
    - topk (int): The number of top-k experts to select.
    - n_expert (int): The number of expert.
    Returns:
    - hidden_states (torch.Tensor): The reduced and unpermuted activation tensor.  
    """   
    n_token, n_hidden = topk_weights.shape[0], permuted_hidden_states.shape[-1]
    hidden_states = torch.empty((n_token, n_hidden), dtype=permuted_hidden_states.dtype, 
                                device=permuted_hidden_states.device)

    torch.ops._moe_C.moe_unpermute(permuted_hidden_states, topk_weights,
                                   topk_ids, src_row_id2dst_row_id_map,
                                   n_expert, topk, hidden_states)
    return hidden_states

