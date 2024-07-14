# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, grouped_topk, moe_align_block_size, try_get_optimal_moe_config)


@triton.jit
def _fused_moe_kernel_a16w4_perchannel(
    # Pointers to matrices
    A,
    B,
    C,
    scale_b_ptr,
    zero_points_ptr,
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
    # moving by 1 element in a particular dimension. E.g. `stride_am` is how
    # much to increase `A` by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_be,
    stride_scale_bn,
    stride_scale_bk,
    stride_zero_points_e,
    stride_zero_points_n,
    stride_zero_points_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    add_zero_points: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am +
                  offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                            offs_bn[None, :] * stride_bn)

    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_SIZE_N * 2 + tl.arange(
            0, 2 * BLOCK_SIZE_N)
        zero_points_ptrs = (zero_points_ptr +
                            off_experts * stride_zero_points_e +
                            offs_zero_points)
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points_ptr.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs,
                                   mask=offs_zero_points < 2 * N,
                                   other=_ZERO_POINT0)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    _A0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=B.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B,
        # generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=_A0)
        b_int4_two = tl.load(b_ptrs,
                             mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                             other=_B0)

        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)
        b = tl.interleave(b_int4_l, b_int4_h).to(A.dtype.element_ty)

        if add_zero_points:
            b -= zero_points_vals[None, :]

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_scale = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    scale_ptrs = (scale_b_ptr + off_experts * stride_scale_be +
                  offs_scale * stride_scale_bn)
    _SCALE0 = tl.zeros([1], dtype=scale_b_ptr.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    accumulator *= scales[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(A.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N * 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_kernel_a16w4_subchannel(
    # Pointers to matrices
    A,
    B,
    C,
    scale_b_ptr,
    zero_points_ptr,
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
    # moving by 1 element in a particular dimension. E.g. `stride_am` is how
    # much to increase `A` by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_be,
    stride_scale_bn,
    stride_scale_bk,
    stride_zero_points_e,
    stride_zero_points_n,
    stride_zero_points_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    add_zero_points: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am +
                  offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                            offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    _A0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=B.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B,
        # generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=_A0)
        b_int4_two = tl.load(b_ptrs,
                             mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                             other=_B0)  # [K x N]

        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)
        b = tl.interleave(b_int4_l,
                          b_int4_h).to(A.dtype.element_ty)  # [K x 2N]

        # dequantize weight
        if add_zero_points:
            offs_zp_n = (pid_n * BLOCK_SIZE_N * 2 +
                         tl.arange(0, 2 * BLOCK_SIZE_N)) % (2 * N)
            _ZERO_POINT0 = tl.zeros([1],
                                    dtype=zero_points_ptr.dtype.element_ty)
            # offs_zp_k = tl.arange(0, 1)
            zp_ptrs = (zero_points_ptr + off_experts * stride_zero_points_e +
                       offs_zp_n * stride_zero_points_n + k)
            zero_points_vals = tl.load(zp_ptrs)
            b = b - zero_points_vals

        offs_scale_n = pid_n * BLOCK_SIZE_N * 2 + tl.arange(
            0, 2 * BLOCK_SIZE_N)
        _SCALE0 = tl.zeros([1], dtype=scale_b_ptr.dtype.element_ty)
        scale_b_ptrs = (scale_b_ptr + off_experts * stride_scale_be +
                        offs_scale_n * stride_scale_bn + k)
        scales_val = tl.load(scale_b_ptrs,
                             mask=offs_scale_n < 2 * N,
                             other=_SCALE0)
        b = b * scales_val[None, :]

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        # accumulator *= scales_val[None, :]
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(A.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N * 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_a16w4_forward(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scale_b: torch.Tensor,
    zero_points: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) with A16W4
    using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K),
        where '*' can be any shape representing batches and K is the feature
        dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K),
        where E is the number of experts, K is the input feature dimension, 
        and N is the output feature dimension.
        It should pack alone N dimension
    - C: The output cache tensor with shape (M, topk, N),
        where M is the total number of tokens post padding, topk is the number
        of times each token is repeated,
        and N is the output feature dimension.
    - scale_b / zero_points: Tensors that used to dequant int4 B, 
                             where dequant_B = (B - zero_points) * scale_b,
        for perchannel case, the shape of scale_b and zero_points is (E, N),
        for subchannel case, the shape of scale_b and zero_points is
            (E, N, K // channel_size).
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are 
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block.
        It determines which expert matrix from B should be used for 
        each block in A.

    This kernel performs the multiplication of a token by its corresponding 
    expert matrix as determined by `expert_ids`.

    The sorting of `sorted_token_ids` by expert index and padding ensures 
    divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency 
    in block matrix multiplication across different blocks processed 
    by the same expert.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert B.shape[1] % 16 == 0 and B.shape[2] % 16 == 0

    add_zero_points = zero_points is not None
    is_perchannel = scale_b.dim() == 2  # (E, N)

    grid = (
        triton.cdiv(sorted_token_ids.shape[0], config['BLOCK_SIZE_M']) *
        triton.cdiv(B.shape[1], config['BLOCK_SIZE_N']),
        1,
        1,
    )

    kwargs = [
        A,
        B,
        C,
        scale_b,
        zero_points,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),  # E
        B.stride(1),  # N
        B.stride(2),  # K
        C.stride(1),
        C.stride(2),
        scale_b.stride(0),  # E
        scale_b.stride(1),  # N
        scale_b.stride(-1),  # K
    ]

    kwargs += ([1, 1, 1] if not add_zero_points else [
        zero_points.stride(0),
        zero_points.stride(1),
        zero_points.stride(-1)
    ])

    const_kwargs = {
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "top_k": top_k,
        "num_warps": 4
    }

    if add_zero_points:
        const_kwargs.update({"add_zero_points": True})
    else:
        const_kwargs.update({"add_zero_points": False})

    if not is_perchannel:
        k_per_scale = B.shape[-1] // scale_b.shape[-1]
        config['BLOCK_SIZE_K'] = k_per_scale

    const_kwargs.update(config)

    if is_perchannel:
        fuse_moe_a16w4 = _fused_moe_kernel_a16w4_perchannel
    else:
        fuse_moe_a16w4 = _fused_moe_kernel_a16w4_subchannel

    fuse_moe_a16w4[grid](*kwargs, **const_kwargs)


def fused_experts_gptq(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w1_qzeros: torch.Tensor,
                       w1_scales: torch.Tensor,
                       w2: torch.Tensor,
                       w2_qzeros: torch.Tensor,
                       w2_scales: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       override_config: Optional[Dict[str, Any]] = None,
                       quantize_bits: int = 4) -> torch.Tensor:
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # only for float16
    assert hidden_states.dtype == torch.float16

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    if quantize_bits == 4:  # 2 int4 are packed as an int8
        N = N * 2
        invoke_fused_moe_kernel = fused_moe_a16w4_forward
        dtype_str = "a16w4"
    else:
        raise ValueError(f"Unsupported quantize_bits: {quantize_bits}")

    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        dtype_str,
        override_config=override_config,
    )

    config = get_config_func(M)

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE:
            # will only happen in the last chunk
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            # reload config to get better performance on the last chunk
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], E))

        invoke_fused_moe_kernel(curr_hidden_states, w1, intermediate_cache1,
                                w1_scales, w1_qzeros, curr_topk_weights,
                                curr_topk_ids, sorted_token_ids, expert_ids,
                                num_tokens_post_padded, False,
                                topk_ids.shape[1], config)

        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                                w2_scales, w2_qzeros, curr_topk_weights,
                                curr_topk_ids, sorted_token_ids, expert_ids,
                                num_tokens_post_padded, True, 1, config)

        torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                  dim=1,
                  out=out_hidden_states[begin_chunk_idx:end_chunk_idx])
    return out_hidden_states


def fused_moe_gptq(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_scales: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w2_scales: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    quantize_bits: int = 4,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1_qweight.shape[
        0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    else:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    return fused_experts_gptq(hidden_states,
                              w1_qweight,
                              w1_qzeros,
                              w1_scales,
                              w2_qweight,
                              w2_qzeros,
                              w2_scales,
                              topk_weights,
                              topk_ids,
                              inplace=inplace,
                              override_config=override_config,
                              quantize_bits=quantize_bits)
