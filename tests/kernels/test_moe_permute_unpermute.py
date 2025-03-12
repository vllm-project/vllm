# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE permute/unpermute kernel

Run `pytest tests/kernels/test_moe_permute_unpermute.py`.
"""

import pytest
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute, moe_unpermute
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from typing import Optional
import numpy as np  # np sort is stable
from vllm.platforms import current_platform

NUM_EXPERTS = [16, 64]
TOP_KS = [2, 4, 8]
EP_SIZE = [1, 4, 16]

def torch_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    topk: int,
    n_expert: int,
    n_local_expert:int,
    start_expert:int,
    expert_map: Optional[torch.Tensor]=None
) -> list[torch.Tensor]:
    # print(topk_ids)
    n_token = hidden_states.shape[0]
    ## torch sort is unstable
    # sorted_topk_ids, sorted_indices = torch.sort(topk_ids.flatten()) 
    if expert_map is not None :
        is_local_expert = (expert_map[topk_ids] != -1)
        not_local_expert = (expert_map[topk_ids] == -1)
        topk_ids = is_local_expert * (topk_ids - start_expert) + not_local_expert * (topk_ids + n_expert)
        # print(topk_ids)

    sorted_indices = np.argsort(topk_ids.flatten().cpu().numpy(), kind="stable") 
    sorted_topk_ids = topk_ids.flatten()[sorted_indices]

    dst_row_id2src_row_id_map = token_expert_indices.flatten()[sorted_indices]
    permuted_hidden_states = hidden_states[dst_row_id2src_row_id_map % n_token, ...]

    expert_first_token_offset = torch.zeros(n_local_expert + 1, dtype=torch.int64, device="cuda")
    idx = 0
    for i in range(0, n_local_expert):
        cnt = 0
        while idx < sorted_topk_ids.numel() and sorted_topk_ids[idx] == i:
            cnt += 1
            idx += 1
        expert_first_token_offset[i+1] = expert_first_token_offset[i] + cnt

    _, src2dst_idx = torch.sort(dst_row_id2src_row_id_map)
    src_row_id2dst_row_id_map = torch.arange(0, n_token * topk, device="cuda", 
                                             dtype=torch.int32)[
                                             src2dst_idx].reshape((n_token, topk))

    # print(expert_first_token_offset)
    return [permuted_hidden_states, expert_first_token_offset, 
            src_row_id2dst_row_id_map]

def torch_unpermute(permuted_hidden_states: torch.Tensor,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    token_expert_indices: torch.Tensor,
                    src_row_id2dst_row_id_map: torch.Tensor,
                    expert_first_token_offset: torch.Tensor, 
                    topk: int,
                    n_expert: int 
) -> torch.Tensor:
    # ignore token not in local expert
    permuted_hidden_states[expert_first_token_offset[-1]:, ...] = 0 
    idx = src_row_id2dst_row_id_map.flatten()[
          token_expert_indices.flatten()].reshape(token_expert_indices.shape)
    output = permuted_hidden_states[idx, 
                                    ...] * topk_weights[..., None]
    output = output.sum(dim=1).to(permuted_hidden_states.dtype)
    return output

@pytest.mark.parametrize("n_token", [1, 33, 64, 222, 1024])
@pytest.mark.parametrize("n_hidden", [128, 1024, 2048])
@pytest.mark.parametrize("n_expert", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ep_size", EP_SIZE)
def test_moe_permute_unpermute(
    n_token: int, n_hidden: int, topk: int, n_expert: int, ep_size: int, dtype: torch.dtype
):
    ep_rank= 0 #np.random.randint(0, ep_size)
    expert_map = None
    n_local_expert = n_expert
    if(ep_size != 1) :
        n_local_expert, expert_map = determine_expert_map(ep_size, ep_rank, n_expert)
        expert_map = expert_map.cuda()
    start_expert = n_local_expert * ep_rank
    current_platform.seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), device="cuda").to(dtype)
    gating_output = torch.randn((n_token, n_expert), device="cuda").to(dtype)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        hidden_states, gating_output, topk, False
    )

    gold0, gold1, gold2 = torch_permute(hidden_states, 
                                        topk_ids, 
                                        token_expert_indices, 
                                        topk, n_expert,n_local_expert,
                                        start_expert, expert_map=expert_map)

    result0, result1, result2 = moe_permute(hidden_states, 
                                            topk_weights, topk_ids, 
                                            token_expert_indices, 
                                            topk, n_expert, n_local_expert, expert_map
                                            )
    # print(gold0, result0)
    # print(gold1, result1)
    # print(gold2, result2)

    # check expert_first_token_offset
    torch.testing.assert_close(gold1,
                               result1,
                               atol=0,
                               rtol=0)
    # check src_row_id2dst_row_id_map
    torch.testing.assert_close(gold2,
                               result2,
                               atol=0,
                               rtol=0)
    # check permuted_hidden_states, only token for [0 : n_local_expert-1] expert 
    torch.testing.assert_close(gold0[:gold1[n_local_expert], ...],
                               result0[:gold1[n_local_expert], ...],
                               atol=0,
                               rtol=0)

    # add a random tensor to simulate group gemm 
    result0 = 0.5 * result0 + torch.randn_like(result0)

    result3 = moe_unpermute(result0, topk_weights, topk_ids, 
                            result2,  result1, topk, n_expert, n_local_expert)
    gold3 = torch_unpermute(result0, topk_weights, topk_ids, 
                            token_expert_indices, result2, result1, topk,n_expert)

    # print(result3)
    # print(gold3)

    torch.testing.assert_close(result3,
                            gold3,
                            atol=2e-2,
                            rtol=0)


# test_moe_permute_unpermute(1024, 2048, 4, 8, 2, torch.float16)