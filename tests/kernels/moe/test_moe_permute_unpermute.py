# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE permute/unpermute kernel

Run `pytest tests/kernels/test_moe_permute_unpermute.py`.
"""

from typing import Optional

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute, moe_permute_unpermute_supported, moe_unpermute)
from vllm.model_executor.layers.fused_moe.utils import (
        blockwise_moe_fp8_quantize_and_permute)
from vllm.platforms import current_platform

from vllm.model_executor.layers.fused_moe.utils import _fp8_quantize, _fp8_perm
from vllm import _custom_ops as ops

NUM_EXPERTS = [16, 64, 256]
TOP_KS = [2, 6, 8]
EP_SIZE = [1, 4, 16]
current_platform.seed_everything(0)


def torch_permute(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        #   token_expert_indices: torch.Tensor,
        topk: int,
        n_expert: int,
        n_local_expert: int,
        start_expert: int,
        expert_map: Optional[torch.Tensor] = None,
        align_block_size: Optional[int] = None,
        fill_invalid_expert: int = -1) -> list[torch.Tensor]:
    n_token, n_hidden = hidden_states.shape[0], hidden_states.shape[1]
    if expert_map is not None:
        is_local_expert = (expert_map[topk_ids] != -1)
        not_local_expert = (expert_map[topk_ids] == -1)
        topk_ids = is_local_expert * (
            topk_ids - start_expert) + not_local_expert * (topk_ids + n_expert)
    token_expert_indices = torch.arange(0,
                                        n_token * topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device).reshape(
                                            (n_token, topk))

    sorted_topk_ids, sorted_indices = torch.sort(topk_ids.flatten(),
                                                 stable=True)
    dst_row_id2src_row_id_map = token_expert_indices.flatten()[sorted_indices]

    expert_first_token_offset = torch.zeros(n_local_expert + 1,
                                            dtype=torch.int64,
                                            device="cuda")
    idx = 0
    for i in range(0, n_local_expert):
        cnt = 0
        while idx < sorted_topk_ids.numel() and sorted_topk_ids[idx] == i:
            cnt += 1
            idx += 1
        expert_first_token_offset[i + 1] = expert_first_token_offset[i] + cnt

    _, src2dst_idx = torch.sort(dst_row_id2src_row_id_map)
    valid_row_idx = []
    if align_block_size is None:

        permuted_hidden_states = hidden_states[dst_row_id2src_row_id_map //
                                               topk, ...]
        permuted_row_size = permuted_hidden_states.shape[0]
        m_indices = torch.empty(permuted_row_size,
                                device="cuda",
                                dtype=torch.int32).fill_(fill_invalid_expert)
        for i in range(1, n_local_expert + 1):
            first_token_offset = expert_first_token_offset[i - 1]
            last_token_offset = expert_first_token_offset[i]
            m_indices[first_token_offset:last_token_offset] = i - 1
        src_row_id2dst_row_id_map = torch.arange(
            0, n_token * topk, device="cuda",
            dtype=torch.int32)[src2dst_idx].reshape((n_token, topk))
        valid_row_idx += [i for i in range(expert_first_token_offset[-1])]
        dst_row_id2src_row_id_map[
            expert_first_token_offset[-1]:] = n_token * topk
        return [
            permuted_hidden_states, expert_first_token_offset,
            src_row_id2dst_row_id_map, dst_row_id2src_row_id_map, m_indices,
            valid_row_idx
        ]
    else:
        permuted_row_size = (topk * n_token + n_expert *
                             (align_block_size - 1) + align_block_size -
                             1) // align_block_size * align_block_size
        permuted_idx = torch.full((permuted_row_size, ),
                                  n_token * topk,
                                  dtype=torch.int32,
                                  device=hidden_states.device)
        permuted_hidden_states = torch.empty((permuted_row_size, n_hidden),
                                             device="cuda",
                                             dtype=hidden_states.dtype)
        align_src_row_id2dst_row_id = torch.empty(n_token * topk,
                                                  device="cuda",
                                                  dtype=torch.int32)
        align_expert_first_token_offset = torch.zeros_like(
            expert_first_token_offset)
        m_indices = torch.empty(permuted_row_size,
                                device="cuda",
                                dtype=torch.int32).fill_(fill_invalid_expert)
        # get align_permuted_hidden_states,
        # valid row_idx and align_expert_first_token_offset
        for i in range(1, n_local_expert + 1):
            first_token_offset = expert_first_token_offset[i - 1]
            last_token_offset = expert_first_token_offset[i]
            n_token_in_expert = last_token_offset - first_token_offset
            align_expert_first_token_offset[
                i] = align_expert_first_token_offset[
                    i - 1] + (n_token_in_expert + align_block_size -
                              1) // align_block_size * align_block_size
            align_first_token_offset = align_expert_first_token_offset[i - 1]
            align_last_token_offset = align_expert_first_token_offset[i]
            dst_row_id2src_row_id_in_expert = dst_row_id2src_row_id_map[
                first_token_offset:first_token_offset + n_token_in_expert]
            # store token in current expert with align_first_token_offset
            permuted_hidden_states[align_first_token_offset:\
                                   align_first_token_offset+n_token_in_expert,\
                                      ...] = hidden_states[\
                                       dst_row_id2src_row_id_in_expert // topk,\
                                          ...]
            permuted_idx[align_first_token_offset:\
                         align_first_token_offset+\
                         n_token_in_expert] = dst_row_id2src_row_id_in_expert
            # set current expert m_indices
            m_indices[align_first_token_offset:align_last_token_offset] = i - 1
            valid_row_idx += [
                i for i in range(align_first_token_offset,
                                 align_first_token_offset + n_token_in_expert)
            ]
        # get align_src_row_id2dst_row_id
        for i in range(n_token * topk):
            eid = sorted_topk_ids[i]
            if (eid >= n_local_expert):
                # check token not in local expert
                align_src_row_id2dst_row_id[
                    i] = align_expert_first_token_offset[-1]
                continue
            first_token_offset = expert_first_token_offset[eid]
            align_first_token_offset = align_expert_first_token_offset[eid]
            token_offset = i - first_token_offset
            align_src_row_id2dst_row_id[
                i] = align_first_token_offset + token_offset
        align_src_row_id2dst_row_id = align_src_row_id2dst_row_id[\
            src2dst_idx].reshape((n_token, topk))
        return [
            permuted_hidden_states, align_expert_first_token_offset,
            align_src_row_id2dst_row_id, permuted_idx, m_indices, valid_row_idx
        ]


def torch_unpermute(permuted_hidden_states: torch.Tensor,
                    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                    token_expert_indices: torch.Tensor,
                    src_row_id2dst_row_id_map: torch.Tensor,
                    valid_row_idx: torch.Tensor, topk: int,
                    n_expert: int) -> torch.Tensor:
    # ignore invalid row
    n_hidden = permuted_hidden_states.shape[1]
    mask = torch.zeros(permuted_hidden_states.shape[0],
                       dtype=bool,
                       device="cuda")
    mask[valid_row_idx] = True
    permuted_hidden_states[~mask] = 0

    permuted_hidden_states = permuted_hidden_states[
        src_row_id2dst_row_id_map.flatten(), ...]
    permuted_hidden_states = permuted_hidden_states.view(-1, topk, n_hidden)
    output = (permuted_hidden_states * topk_weights.unsqueeze(2)).sum(1).to(
        permuted_hidden_states.dtype)
    return output


@pytest.mark.parametrize("n_token", [1, 33, 1024, 5000])
@pytest.mark.parametrize("n_hidden", [2048, 7168])
@pytest.mark.parametrize("n_expert", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("align_block_size", [None, 128])
def test_moe_permute_unpermute(n_token: int, n_hidden: int, topk: int,
                               n_expert: int, ep_size: int, dtype: torch.dtype,
                               align_block_size: Optional[int]):
    if not moe_permute_unpermute_supported():
        pytest.skip("moe_permute_unpermute is not supported on this platform.")
    fill_invalid_expert = 0
    ep_rank = np.random.randint(0, ep_size)
    expert_map = None
    n_local_expert = n_expert
    if (ep_size != 1):
        n_local_expert, expert_map = determine_expert_map(
            ep_size, ep_rank, n_expert)
        expert_map = expert_map.cuda()
    start_expert = n_local_expert * ep_rank
    current_platform.seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden), device="cuda").to(dtype)
    gating_output = torch.randn((n_token, n_expert), device="cuda").to(dtype)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        hidden_states, gating_output, topk, False)
    (gold_permuted_hidden_states, gold_expert_first_token_offset,
     gold_inv_permuted_idx, gold_permuted_idx, gold_m_indices,
     valid_row_idx) = torch_permute(
         hidden_states,
         topk_ids,
         # token_expert_indices,
         topk,
         n_expert,
         n_local_expert,
         start_expert,
         expert_map=expert_map,
         align_block_size=align_block_size,
         fill_invalid_expert=fill_invalid_expert)

    (permuted_hidden_states, _, expert_first_token_offset, inv_permuted_idx,
     m_indices) = moe_permute(hidden_states=hidden_states,
                              a1q_scale=None,
                              topk_ids=topk_ids,
                              n_expert=n_expert,
                              n_local_expert=n_local_expert,
                              expert_map=expert_map,
                              align_block_size=align_block_size,
                              fill_invalid_expert=fill_invalid_expert)

    # check expert_first_token_offset
    torch.testing.assert_close(gold_expert_first_token_offset,
                               expert_first_token_offset,
                               atol=0,
                               rtol=0)
    # check src_row_id2dst_row_id_map
    torch.testing.assert_close(gold_inv_permuted_idx.flatten(),
                               inv_permuted_idx,
                               atol=0,
                               rtol=0)
    # check mindice
    # current kernel usage assumes deepgemm requires align_block_size
    # when it's not provided then we don't compute m_indices (for cutlass)
    if align_block_size is not None:
        torch.testing.assert_close(gold_m_indices, m_indices, atol=0, rtol=0)

    # check permuted_hidden_states, only valid token
    torch.testing.assert_close(gold_permuted_hidden_states[valid_row_idx],
                               permuted_hidden_states[valid_row_idx],
                               atol=0,
                               rtol=0)
    # add a random tensor to simulate group gemm
    result0 = 0.5 * permuted_hidden_states + torch.randn_like(
        permuted_hidden_states)
    result4 = torch.empty_like(hidden_states)
    moe_unpermute(result4, result0, topk_weights, inv_permuted_idx,
                  expert_first_token_offset)

    gold4 = torch_unpermute(result0, topk_weights, topk_ids,
                            token_expert_indices, inv_permuted_idx,
                            valid_row_idx, topk, n_local_expert)
    # check unpermuted hidden
    torch.testing.assert_close(result4, gold4, atol=2e-2, rtol=0)

# @pytest.mark.parametrize("m", [1, 33, 1024, 5000])
# @pytest.mark.parametrize("k", [2048, 7168])
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("ep_size", [1]) #EP_SIZE) # TODO provide tests for ep_size > 1
@pytest.mark.parametrize("m", [3])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("e", [4])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("ep_size", [1])
def test_blockwise_moe_fp8_quantize_and_permute(
    m: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    ep_size: int,
):
    torch.manual_seed(42)

    group_size = 128

    # Create input tensors
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    a_scale = torch.randn((m, k // group_size),
                              dtype=torch.float32, device="cuda")

    ground_a = a.clone()
    ground_a_scale = a_scale.clone()

    print("a_scale first:", a_scale)
    
    score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
    _, topk_ids, _ = fused_topk(a,
                                               score,
                                               topk,
                                               renormalize=False)

    # permuted_a = torch.empty(m * topk, k, dtype=torch.float8_e4m3fn, device="cuda")

    # Call the function
    (permuted_a, permuted_a_scale, expert_offsets,
     permuted_idx) = blockwise_moe_fp8_quantize_and_permute(
        a,
        None,
        topk_ids,
        e,
        group_size,
        transpose_scales=False,
        # permuted_a=permuted_a,
    )

    print("a_scale second:", a_scale)

    ground_a, ground_a_scale = _fp8_quantize(
            ground_a,
            A_scale=None,
            per_act_token=False,
            block_shape=[group_size, group_size],
        )
    a_map = torch.argsort(permuted_idx) // topk
    problem_sizes = torch.empty((e, 3), dtype=torch.int32, device="cuda")

    for expert in range(e):
        problem_sizes[expert][0] = expert_offsets[expert + 1] - expert_offsets[expert]
        problem_sizes[expert][1] = 128
        problem_sizes[expert][2] = k

    print("topk ids:", topk_ids)
    ground_a = _fp8_perm(ground_a, a_map)
    print("a_map:", a_map)
    print("ground scale:", ground_a_scale)
    ground_a_scale = ground_a_scale[a_map]
    print("mapped ground scale:", ground_a_scale)
    # ground_a_scale = ops.transpose_cutlass_moe_a_scales(
    #             ground_a_scale, expert_offsets[:-1], problem_sizes)

    _, permquant_scale = _fp8_quantize(
            a,
            A_scale=None,
            per_act_token=False,
            block_shape=[group_size, group_size],
            expert_offsets=expert_offsets[:-1],
            problem_sizes=problem_sizes,
            idx_map=permuted_idx,
        )
    print("permquant_scale:", permquant_scale)

    # Optionally, check that the output is not all zeros
    # print("a:", a)
    # print("a_scale:", a_scale)
    # print("topk_ids:", topk_ids)
    # print("expert_offsets:", expert_offsets)
    # print("permuted_idx:", permuted_idx)
    print("permuted_a:", permuted_a)
    print("permuted_a_scale:", permuted_a_scale)
    print("ground_a:", ground_a)
    print("ground_a_scale:", ground_a_scale)

    # print("permuted_a shape:", permuted_a.shape)
    # print("ground_a shape:", ground_a.shape)
    # print("permuted_a_scale shape:", permuted_a_scale.shape)
    # print("ground_a_scale shape:", ground_a_scale.shape)
    # retrieved_a = permuted_a.float() * permuted_a_scale
    # retrieved_ground_a = ground_a.float() * ground_a_scale
    # print((retrieved_a - retrieved_ground_a).abs())
    # print((retrieved_a - retrieved_ground_a).abs().max())

    torch.testing.assert_close(ground_a, permuted_a, atol=0, rtol=0)
    torch.testing.assert_close(ground_a_scale, permuted_a_scale, atol=0, rtol=0)
