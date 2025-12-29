# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from .utils import supports_pdl

_LORA_PTR_DICT: dict[tuple[int, ...], torch.Tensor] = {}

def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(w.data_ptr() for w in lora_weights)
    ptr_tensor = _LORA_PTR_DICT.get(key)
    if ptr_tensor is not None:
        return ptr_tensor

    ptr_tensor = torch.tensor(
        [w.data_ptr() for w in lora_weights],
        device=device,
        dtype=torch.uint64,
    )
    _LORA_PTR_DICT[key] = ptr_tensor
    return ptr_tensor

_A_WS_CACHE: dict[tuple, torch.Tensor] = {}

def _get_a_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_slices: int,
    M: int,
    top_k_num: int,
    rank: int,
    must_zero: bool,
) -> torch.Tensor:
    """
    Reuse a_intermediate_cache1 to reduce allocator & memset overhead.
    - If must_zero: zero_() (needed when shrink SPLIT_K>1 uses atomic_add).
    - Else: leave uninitialized (safe because masked stores/loads ensure
      unwritten rows are never read for contribution).
    """
    key = (device.type, device.index, dtype, num_slices, M, top_k_num, rank)
    buf = _A_WS_CACHE.get(key)
    if buf is None:
        buf = torch.empty((num_slices, M, top_k_num, rank), device=device, dtype=dtype)
        _A_WS_CACHE[key] = buf
        if must_zero:
            buf.zero_()
    else:
        if must_zero:
            buf.zero_()
    return buf



@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
        "MAX_LORAS_TOTAL",
    ]
)
def _fused_moe_lora_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    lora_ids,
    adapter_enabled,
    MAX_LORAS_TOTAL,  # python int runtime scalar
    # strides
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
    slice_a_size,
    slice_c_size,
    # meta
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_TO_C: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx).to(tl.int64)
    if lora_id == -1:
        return

    if tl.load(adapter_enabled + lora_id) == 0:
        return

    max_loras_total = tl.full((), MAX_LORAS_TOTAL, tl.int64)

    # pid mapping
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # expert id
    ind_e = lora_id * stride_el + pid_m
    expert_id = tl.load(
        expert_ids_ptr + ind_e,
        mask=ind_e < max_loras_total * stride_el,
        other=-1,
    )
    if expert_id == -1:
        return

    # slice pointers
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    # N offsets (no modulo)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    cn_mask = offs_cn < N

    # K offsets
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # token ids
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_ind = lora_id * stride_tl + offs_token_id
    offs_token = tl.load(
        sorted_token_ids_ptr + token_ind,
        mask=token_ind < max_loras_total * stride_tl,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    # A/B ptrs
    a_ptrs = cur_a_ptr + (
        (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_cn[None, :] * stride_bn
    )

    if USE_GDC and IS_PRIMARY:
        tl.extra.cuda.gdc_launch_dependents()

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for kk in range(0, grid_k):
        k_remaining = K - kk * (BLOCK_SIZE_K * SPLIT_K)

        b_mask = (offs_k[:, None] < k_remaining) & (cn_mask[None, :])
        if USE_B_L2_CACHE:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
        else:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= moe_weight[:, None]

    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & cn_mask[None, :]

    if SPLIT_K == 1:
        if ADD_TO_C:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, (acc + prev).to(c_ptr.dtype.element_ty), mask=c_mask)
        else:
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask, sem="relaxed")


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    w1_lora_a = lora_a_stacked[0]
    use_gdc = supports_pdl(qcurr_hidden_states.device)

    b_ptr = _get_ptr(lora_a_stacked, device)
    max_loras_total = sorted_token_ids.shape[0]  # python int

    grid0 = (
        split_k
        * triton.cdiv(EM, block_size_m)
        * triton.cdiv(N, block_size_n)
    )
    grid = (grid0, len(lora_a_stacked), lora_ids.numel())

    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        max_loras_total,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a.stride(0),
        w1_lora_a.stride(1),
        w1_lora_a.stride(3),
        w1_lora_a.stride(2),
        a_intermediate_cache1.stride(2),
        a_intermediate_cache1.stride(3),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        top_k=1 if mul_routed_weight else top_k_num,
        MUL_ROUTED_WEIGHT=False,
        ADD_TO_C=False,
        USE_B_L2_CACHE=True,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=split_k,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        IS_PRIMARY=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    EM: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    w1_lora_b = lora_b_stacked[0]

    K = max_lora_rank
    N = w1_output_dim_size

    a2d = a_intermediate_cache1.view(-1, a_intermediate_cache1.shape[3])
    out_view = output[:, :, offset : offset + num_slices * N]

    use_gdc = supports_pdl(a2d.device)
    max_loras_total = sorted_token_ids.shape[0]

    grid0 = triton.cdiv(EM, block_size_m) * triton.cdiv(N, block_size_n)
    grid = (grid0, len(lora_b_stacked), lora_ids.numel())

    _fused_moe_lora_kernel[grid](
        a2d,
        b_ptr,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        max_loras_total,
        a2d.stride(0),
        a2d.stride(1),
        w1_lora_b.stride(0),
        w1_lora_b.stride(1),
        w1_lora_b.stride(3),
        w1_lora_b.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=a2d.numel() // num_slices,
        slice_c_size=N,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_TO_C=True,
        USE_B_L2_CACHE=True,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=split_k,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        IS_PRIMARY=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b = lora_b_stacked[0]

    num_experts = lora_a_stacked[0].shape[1]
    M = topk_weights.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b.shape[2]

    # When using fully_sharded or shrink_split_k>1 (atomic_add), it must be zeroed.
    must_zero_a = fully_sharded or (shrink_split_k != 1)

    a_intermediate_cache1 = _get_a_workspace(
        device=device,
        dtype=output.dtype,
        num_slices=num_slices,
        M=M,
        top_k_num=top_k_num,
        rank=max_lora_rank,
        must_zero=must_zero_a,
    )

    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        device,
        N=max_lora_rank,
        M=M,
        EM=EM,
        K=K,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_slices=num_slices,
        block_size_m=shrink_block_size_m,
        block_size_n=shrink_block_size_n,
        block_size_k=shrink_block_size_k,
        group_size_m=shrink_group_size_m,
        num_warps=shrink_num_warps,
        num_stages=shrink_num_stages,
        split_k=shrink_split_k,
        mul_routed_weight=mul_routed_weight,
    )

    if fully_sharded:
        if max_lora_rank == w1_lora_b.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(a_intermediate_cache1)
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(a_intermediate_cache1)
            max_lora_rank = a_intermediate_cache1.shape[-1]

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        device,
        EM=EM,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_slices=num_slices,
        max_lora_rank=max_lora_rank,
        w1_output_dim_size=w1_output_dim_size,
        block_size_m=expand_block_size_m,
        block_size_n=expand_block_size_n,
        block_size_k=expand_block_size_k,
        group_size_m=expand_group_size_m,
        num_warps=expand_num_warps,
        num_stages=expand_num_stages,
        split_k=expand_split_k,
        mul_routed_weight=mul_routed_weight,
        offset=offset,
    )


def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=_fused_moe_lora_shrink_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_expand_fake,
    )

    fused_moe_lora = torch.ops.vllm.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.vllm.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.vllm.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand