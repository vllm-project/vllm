# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Taken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py
and updated to fit vllm needs and terminology.
"""

import functools
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.triton_utils import tl, triton
from vllm.utils import round_up


@functools.cache
def deep_gemm_block_shape() -> list[int]:
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg
    block = dg.get_m_alignment_for_contiguous_layout()
    return [block, block]


def expert_num_tokens_round_up_and_sum(expert_num_tokens: torch.Tensor,
                                       alignment: int) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) +
           (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M(M: int, num_topk: int, local_num_experts: int,
                      alignment: int,
                      expert_tokens_meta: Optional[mk.ExpertTokensMetadata]):

    if ((expert_tokens_meta is not None)
            and (expert_tokens_meta.expert_num_tokens_cpu is not None)):
        return expert_num_tokens_round_up_and_sum(
            expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment)

    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum


@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return ((x + y - 1) // y) * y


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum,
                                mask=offset_cumsum < num_experts,
                                other=0)
    tokens_per_expert = round_up_128(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum,
             cumsum,
             mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in,
                          mask=mask)
        to_copy_s = tl.load(recv_x_scale + token_id * recv_x_scale_stride0 +
                            offset_in_s,
                            mask=mask_s)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 +
                                topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id,
                                                 1)
                tl.store(
                    output_index + token_id * output_index_stride0 +
                    topk_index, dest_token_index)
                output_tensor_ptr = (output_tensor +
                                     dest_token_index * output_tensor_stride0)
                output_tensor_scale_ptr = (
                    output_tensor_scale +
                    dest_token_index * output_tensor_scale_stride0)
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s,
                         to_copy_s,
                         mask=mask_s)


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0

    _fwd_kernel_ep_scatter_1[(grid, )](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid, )](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(recv_topk_ids +
                                cur_token * recv_topk_ids_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                source_token_index = tl.load(input_index +
                                             cur_token * input_index_stride0 +
                                             topk_index)
                acc_weight = tl.load(recv_topk_weight +
                                     cur_token * recv_topk_weight_stride0 +
                                     topk_index)
                tmp = tl.load(input_tensor +
                              source_token_index * input_tensor_stride0 +
                              cur_block * BLOCK_D + off_d)
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor + cur_token * output_tensor_stride0 +
            cur_block * BLOCK_D + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = min(hidden_size, 1024)
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def deepgemm_moe_permute(aq: torch.Tensor,
                         aq_scale: torch.Tensor,
                         topk_ids: torch.Tensor,
                         local_num_experts: int,
                         expert_map: Optional[torch.Tensor],
                         expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
                         aq_out: Optional[torch.Tensor] = None):

    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, (
        "The kernel uses -1 to represent invalid topk_ids")
    H = aq.size(1)
    device = aq.device

    block_m = deep_gemm_block_shape()[0]
    block_k = deep_gemm_block_shape()[1]

    M_sum = compute_aligned_M(M=topk_ids.size(0),
                              num_topk=topk_ids.size(1),
                              local_num_experts=local_num_experts,
                              alignment=block_m,
                              expert_tokens_meta=expert_tokens_meta)

    expert_start_loc = torch.empty((local_num_experts),
                                   device=device,
                                   dtype=torch.int32)

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    aq_scale_out = torch.empty((M_sum, H // block_k),
                               device=device,
                               dtype=torch.float32)

    maybe_has_empty_blocks = ((expert_tokens_meta is None)
                              or (expert_tokens_meta.expert_num_tokens_cpu
                                  is None))
    expert_ids_init = torch.zeros if maybe_has_empty_blocks else torch.empty

    expert_ids = expert_ids_init((M_sum), device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens = None
    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    else:
        expert_num_tokens = count_expert_num_tokens(topk_ids,
                                                    local_num_experts,
                                                    expert_map)

    ep_scatter(recv_x=aq,
               recv_x_scale=aq_scale,
               recv_topk=topk_ids,
               num_recv_tokens_per_expert=expert_num_tokens,
               expert_start_loc=expert_start_loc,
               expert_map=expert_map,
               output_tensor=aq_out,
               output_tensor_scale=aq_scale_out,
               m_indices=expert_ids,
               output_index=inv_perm)

    return aq_out, aq_scale_out, expert_ids, inv_perm


def deepgemm_unpermute_and_reduce(
        a: torch.Tensor,  # Grouped gemm output
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_perm: torch.Tensor,
        expert_map: Optional[torch.Tensor],
        output: torch.Tensor):

    return ep_gather(input_tensor=a,
                     recv_topk_ids=topk_ids,
                     recv_topk_weight=topk_weights,
                     input_index=inv_perm,
                     expert_map=expert_map,
                     output_tensor=output)
