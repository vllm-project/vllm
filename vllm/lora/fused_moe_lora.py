# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Tuple

import torch
import triton
import triton.language as tl

_LORA_PTR_DICT: Dict[Tuple[int, ...], torch.tensor] = {}


def _get_ptr(lora_weights: List[torch.Tensor], device: torch.device):
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`, 
    After this, it remains constant and subsequent usage is through LUT.
    Refer to: 
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:
        return ptr_tensor

    tensor_ptrs = []
    for lora_weight in lora_weights:
        tensor_ptrs.append(lora_weight.data_ptr())
    ptr_tensor = torch.tensor(tensor_ptrs, device=device)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


@triton.jit
def fused_moe_lora(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    slice_a_size: tl.constexpr,
    slice_c_size: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    # calculate pid_m,pid_n
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_idx)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # get the expert_id to process curr shard
    expert_id = tl.load(expert_ids_ptr + lora_idx * stride_el + pid_m)
    if expert_id >= 64:
        return

    # if expert_id != 0:
    #     return

    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.bfloat16))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + stride_tl * lora_idx +
                         offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # get a_ptrs,b_ptrs
    a_ptrs = cur_a_ptr + (offs_token[:, None] // top_k * stride_am +
                          offs_k[None, :] * stride_ak)
    b_ptrs = cur_b_ptr + lora_idx * stride_bl + expert_id * stride_be + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_w13_lora(qcurr_hidden_states: torch.Tensor,
                       w13_lora_a_stacked: List[torch.Tensor],
                       w13_lora_b_stacked: List[torch.Tensor],
                       topk_weights: torch.Tensor,
                       sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                       num_tokens_post_padded: torch.Tensor, max_lora_rank: int,
                       top_k_num: int, config,
                       intermediate_cache1: torch.Tensor):

    w1_lora_a_stacked = w13_lora_a_stacked[0]
    w1_lora_b_stacked = w13_lora_b_stacked[0]

    #============begin============
    N = max_lora_rank
    M = qcurr_hidden_states.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]
    #====================================
    w13_intermediate_cache1 = torch.zeros(
        (2 * M * top_k_num * (max_lora_rank + w1_output_dim_size)),
        dtype=torch.bfloat16,
        device=qcurr_hidden_states.device)

    w1_a_inter_size = M * top_k_num * max_lora_rank
    w1_b_inter_size = M * top_k_num * w1_output_dim_size

    w1_a_intermediate_cache1 = w13_intermediate_cache1[:w1_a_inter_size].view(
        M, top_k_num, max_lora_rank)
    # w3_a_intermediate_cache1 = w13_intermediate_cache1[w1_a_inter_size:2 *
    #                                                    w1_a_inter_size].view(
    #                                                        M, top_k_num,
    #                                                        max_lora_rank)

    w1_b_intermediate_cache1 = w13_intermediate_cache1[2 * w1_a_inter_size:2 *
                                                       w1_a_inter_size +
                                                       w1_b_inter_size].view(
                                                           M, top_k_num,
                                                           w1_output_dim_size)
    w3_b_intermediate_cache1 = w13_intermediate_cache1[2 * w1_a_inter_size +
                                                       w1_b_inter_size:].view(
                                                           M, top_k_num,
                                                           w1_output_dim_size)

    #====================================

    b_ptr = _get_ptr(w13_lora_a_stacked, qcurr_hidden_states.device)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
        len(w13_lora_a_stacked),
        w13_lora_a_stacked[0].shape[0],
    )

    fused_moe_lora[grid](qcurr_hidden_states,
                         b_ptr,
                         w1_a_intermediate_cache1,
                         topk_weights,
                         sorted_token_ids,
                         expert_ids,
                         num_tokens_post_padded,
                         N,
                         K,
                         EM,
                         num_tokens,
                         qcurr_hidden_states.stride(0),
                         qcurr_hidden_states.stride(1),
                         w1_lora_a_stacked.stride(0),
                         w1_lora_a_stacked.stride(1),
                         w1_lora_a_stacked.stride(3),
                         w1_lora_a_stacked.stride(2),
                         w1_a_intermediate_cache1.stride(1),
                         w1_a_intermediate_cache1.stride(2),
                         sorted_token_ids.stride(0),
                         expert_ids.stride(0),
                         num_slice_a=1,
                         num_slice_c=2,
                         slice_a_size=qcurr_hidden_states.numel(),
                         slice_c_size=w1_a_intermediate_cache1.numel(),
                         top_k=top_k_num,
                         MUL_ROUTED_WEIGHT=False,
                         **config)

    b_ptr = _get_ptr(w13_lora_b_stacked, qcurr_hidden_states.device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_a_intermediate_cache1 = w1_a_intermediate_cache1.view(
        -1, w1_a_intermediate_cache1.shape[2])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
        len(w13_lora_b_stacked),
        w13_lora_b_stacked[0].shape[0],
    )
    fused_moe_lora[grid](w1_a_intermediate_cache1,
                         b_ptr,
                         w1_b_intermediate_cache1,
                         topk_weights,
                         sorted_token_ids,
                         expert_ids,
                         num_tokens_post_padded,
                         N,
                         K,
                         EM,
                         num_tokens,
                         w1_a_intermediate_cache1.stride(0),
                         w1_a_intermediate_cache1.stride(1),
                         w1_lora_b_stacked.stride(0),
                         w1_lora_b_stacked.stride(1),
                         w1_lora_b_stacked.stride(3),
                         w1_lora_b_stacked.stride(2),
                         w1_b_intermediate_cache1.stride(1),
                         w1_b_intermediate_cache1.stride(2),
                         sorted_token_ids.stride(0),
                         expert_ids.stride(0),
                         num_slice_a=2,
                         num_slice_c=2,
                         slice_a_size=w1_a_intermediate_cache1.numel(),
                         slice_c_size=w1_b_intermediate_cache1.numel(),
                         top_k=1,
                         MUL_ROUTED_WEIGHT=False,
                         **config)
    intermediate_cache1[:, :, :N] += w1_b_intermediate_cache1
    intermediate_cache1[:, :, N:] += w3_b_intermediate_cache1


def fused_moe_w2_lora(intermediate_cache2, w2_lora_a_stacked: torch.Tensor,
                      w2_lora_b_stacked: torch.Tensor,
                      topk_weights: torch.Tensor,
                      sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                      num_tokens_post_padded: torch.Tensor, max_lora_rank: int,
                      top_k_num: int, config):
    EM = sorted_token_ids.shape[1]
    M = topk_weights.shape[0]
    num_tokens = topk_weights.numel()
    device = intermediate_cache2.device

    w2_a_intermediate_cache1 = torch.zeros((M * top_k_num, max_lora_rank),
                                           dtype=torch.bfloat16,
                                           device=device)
    w2_b_intermediate_cache1 = torch.zeros(
        (M, top_k_num, w2_lora_b_stacked.shape[2]),
        dtype=torch.bfloat16,
        device=device)

    b_ptr = _get_ptr([w2_lora_a_stacked], device)

    w2_lora_a_in = intermediate_cache2.view(-1, intermediate_cache2.shape[-1])
    w2_lora_a_out = w2_a_intermediate_cache1.view(-1, top_k_num, max_lora_rank)

    K = w2_lora_a_stacked.shape[3]
    N = max_lora_rank

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
        1,  #slices
        w2_lora_a_stacked.shape[0],  # max_loras
    )

    fused_moe_lora[grid](intermediate_cache2,
                         b_ptr,
                         w2_a_intermediate_cache1,
                         topk_weights,
                         sorted_token_ids,
                         expert_ids,
                         num_tokens_post_padded,
                         N,
                         K,
                         EM,
                         num_tokens,
                         w2_lora_a_in.stride(0),
                         w2_lora_a_in.stride(1),
                         w2_lora_a_stacked.stride(0),
                         w2_lora_a_stacked.stride(1),
                         w2_lora_a_stacked.stride(3),
                         w2_lora_a_stacked.stride(2),
                         w2_lora_a_out.stride(1),
                         w2_lora_a_out.stride(2),
                         sorted_token_ids.stride(0),
                         expert_ids.stride(0),
                         num_slice_a=1,
                         num_slice_c=1,
                         slice_a_size=intermediate_cache2.numel(),
                         slice_c_size=w2_a_intermediate_cache1.numel(),
                         top_k=1,
                         MUL_ROUTED_WEIGHT=False,
                         **config)

    K = max_lora_rank
    N = w2_lora_b_stacked.shape[2]

    b_ptr = _get_ptr([w2_lora_b_stacked], device)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
        1,  #slices
        w2_lora_a_stacked.shape[0],  # max_loras
    )

    fused_moe_lora[grid](w2_a_intermediate_cache1,
                         b_ptr,
                         w2_b_intermediate_cache1,
                         topk_weights,
                         sorted_token_ids,
                         expert_ids,
                         num_tokens_post_padded,
                         N,
                         K,
                         EM,
                         num_tokens,
                         w2_a_intermediate_cache1.stride(0),
                         w2_a_intermediate_cache1.stride(1),
                         w2_lora_b_stacked.stride(0),
                         w2_lora_b_stacked.stride(1),
                         w2_lora_b_stacked.stride(3),
                         w2_lora_b_stacked.stride(2),
                         w2_b_intermediate_cache1.stride(1),
                         w2_b_intermediate_cache1.stride(2),
                         sorted_token_ids.stride(0),
                         expert_ids.stride(0),
                         num_slice_a=1,
                         num_slice_c=1,
                         slice_a_size=w2_a_intermediate_cache1.numel(),
                         slice_c_size=w2_b_intermediate_cache1.numel(),
                         top_k=1,
                         MUL_ROUTED_WEIGHT=True,
                         **config)
    return w2_b_intermediate_cache1
