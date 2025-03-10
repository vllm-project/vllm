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
from typing import List
import numpy as np  # np sort is stable
from vllm.platforms import current_platform

NUM_EXPERTS = [16, 64]
TOP_KS = [2, 4, 8]

def torch_permute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    topk: int,
    n_expert: int,
) -> list[torch.Tensor]:
    # print(topk_ids)
    n_token = hidden_states.shape[0]
    ## torch sort is unstable
    # sorted_topk_ids, sorted_indices = torch.sort(topk_ids.flatten()) 
    
    sorted_indices = np.argsort(topk_ids.flatten().cpu().numpy(), kind="stable") 
    sorted_topk_ids = topk_ids.flatten()[sorted_indices]

    dst_row_id2src_row_id_map = token_expert_indices.flatten()[sorted_indices]
    permuted_hidden_states = hidden_states[dst_row_id2src_row_id_map % n_token, ...]

    expert_first_token_offset = torch.zeros(n_expert + 1, dtype=torch.int64, device="cuda")
    idx = 0
    for i in range(0, n_expert):
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
                    topk: int,
                    n_expert: int 
) -> torch.Tensor:
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
def test_moe_permute_unpermute(
    n_token: int, n_hidden: int, topk: int, n_expert: int, dtype: torch.dtype
):
    current_platform.seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), device="cuda", dtype=dtype) / 10
    gating_output = torch.randn((n_token, n_expert), device="cuda", dtype=dtype) 
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        hidden_states, gating_output, topk, False
    )
    gold0, gold1, gold2 = torch_permute(hidden_states, 
                                        topk_weights, topk_ids, 
                                        token_expert_indices, 
                                        topk, n_expert)
    result0, result1, result2 = moe_permute(hidden_states, 
                                            topk_weights, topk_ids, 
                                            token_expert_indices, 
                                            topk, n_expert)
    # print(gold0, result0)
    # print(gold1, result1)
    # print(gold2, result2)
    torch.testing.assert_close(gold0,
                               result0,
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(gold1,
                               result1,
                               atol=0,
                               rtol=0)
    torch.testing.assert_close(gold2,
                               result2,
                               atol=0,
                               rtol=0)
    # add a random tensor to simulate group gemm 
    result0 = 0.5 * result0 + torch.randn_like(result0)

    result3 = moe_unpermute(result0, topk_weights, topk_ids, 
                            result2, topk,n_expert)
    gold3 = torch_unpermute(result0, topk_weights, topk_ids, 
                            token_expert_indices, result2, topk,n_expert)

    torch.testing.assert_close(result3,
                            gold3,
                            atol=2e-2,
                            rtol=0)

