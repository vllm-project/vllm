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

_LORA_PTR_DICT: dict[tuple[int, ...], torch.tensor] = {}


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):
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
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
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
    MAX_LORAS_TOTAL,  # python int scalar
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

    # ---- scalars as uint32/int32 (cheaper than int64) ----
    max_loras_u32 = tl.full((), MAX_LORAS_TOTAL, tl.uint32)
    num_valid_u32 = tl.full((), num_valid_tokens, tl.uint32)
    num_experts_u32 = tl.full((), num_experts, tl.uint32)

    # lora_id: int32
    lora_id_i32 = tl.load(lora_ids + lora_idx).to(tl.int32)
    lora_id_u32 = lora_id_i32.to(tl.uint32)

    # valid lora_id: (0 <= lora_id < MAX_LORAS_TOTAL)
    # Use unsigned comparison to automatically exclude -1: (-1) -> 0xFFFF_FFFF,
    # which will not be < max_loras_u32
    if lora_id_u32 >= max_loras_u32:
        return

    if tl.load(adapter_enabled + lora_id_i32) == 0:
        return

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

    # token count per (lora, expert) shard
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id_i32)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # expert_id: int32 -> uint32 bounds check
    ind_e = lora_id_i32 * stride_el + pid_m
    expert_id_i32 = tl.load(
        expert_ids_ptr + ind_e,
        mask=ind_e < (max_loras_u32.to(tl.int32) * stride_el),
        other=-1,
    ).to(tl.int32)
    expert_id_u32 = expert_id_i32.to(tl.uint32)
    if expert_id_u32 >= num_experts_u32:
        return

    # slice pointers
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    # N offsets: no modulo + cn_mask
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
    cn_mask = offs_cn < N

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # token ids: int32, other=-1
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int32)
    token_ind = lora_id_i32 * stride_tl + offs_token_id
    offs_token_i32 = tl.load(
        sorted_token_ids_ptr + token_ind,
        mask=token_ind < (max_loras_u32.to(tl.int32) * stride_tl),
        other=-1,
    ).to(tl.int32)

    # Using unsigned comparison: negative numbers automatically become large numbers
    # -> mask false
    offs_token_u32 = offs_token_i32.to(tl.uint32)
    token_mask = offs_token_u32 < num_valid_u32

    safe_token_i32 = tl.where(token_mask, offs_token_i32, 0)

    # A/B ptrs
    a_ptrs = cur_a_ptr + (
        (safe_token_i32[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        cur_b_ptr
        + lora_id_i32 * stride_bl
        + expert_id_i32 * stride_be
        + offs_k[:, None] * stride_bk
        + offs_cn[None, :] * stride_bn
    )

    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
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
        moe_weight = tl.load(
            topk_weights_ptr + safe_token_i32, mask=token_mask, other=0.0
        )
        acc *= moe_weight[:, None]

    c_ptrs = (
        cur_c_ptr
        + stride_cm * safe_token_i32[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = token_mask[:, None] & cn_mask[None, :]

    out_ty = c_ptr.dtype.element_ty
    acc_out = acc.to(out_ty)

    if SPLIT_K == 1:
        if ADD_TO_C:
            # Perform addition with out_ty, matching the rounding path of torch’s output(dtype)+=delta(dtype).
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)  # already out_ty
            tl.store(c_ptrs, prev + acc_out, mask=c_mask)
        else:
            tl.store(c_ptrs, acc_out, mask=c_mask)
    else:
        #When SPLIT_K != 1, atomic is still used (note: expand will force split_k=1 to prevent atomic accumulation on non-zero output).
        tl.atomic_add(c_ptrs, acc_out, mask=c_mask, sem="relaxed")

@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, M, top_k_num, max_lora_rank)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    # adding for kernel
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
    use_gdc: bool = False,
) -> None:
    # N here is max_lora_rank, passed from caller
    w1_lora_a = lora_a_stacked[0]
    b_ptr = _get_ptr(lora_a_stacked, device)
    max_loras_total = sorted_token_ids.shape[0]

    grid0 = (
        split_k
        * triton.cdiv(EM, block_size_m)
        * triton.cdiv(N, block_size_n)
    )
    grid = (grid0, len(lora_a_stacked), lora_ids.numel())


    # a_intermediate_cache1 is now a view sliced from a larger flat buffer. The stride between slices must be calculated using stride(0)，
    # and can no longer be obtained with numel() // num_slices (otherwise, when capacity > size, it will cause incorrect slice address calculation / overlapping writes).
    slice_c_size = a_intermediate_cache1.stride(0)

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
        slice_a_size=qcurr_hidden_states.numel(),  # It has no effect when num_slice_a=1.
        slice_c_size=slice_c_size,
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

# A workspace: only one expandable large buffer is retained for each combination of (device, dtype, num_slices, top_k_num).
_A_WS_BUF: dict[tuple, torch.Tensor] = {}
_A_WS_CAP: dict[tuple, tuple[int, int]] = {}  # (cap_M, cap_rank)


def _round_up_pow2(x: int) -> int:
    """Round x up to the nearest power of two to reduce thrashing and fragmentation caused by repeated realloc operations."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


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
    Returns a view with shape (num_slices, M, top_k_num, rank).
    Internally, only one "large buffer" is maintained; it is expanded when capacity is insufficient, avoiding separate tensors for each M.
    """
    key = (device.type, device.index, dtype, num_slices, top_k_num)

    buf = _A_WS_BUF.get(key)
    cap = _A_WS_CAP.get(key)
    cap_M, cap_rank = cap if cap is not None else (0, 0)

    need_realloc = (buf is None) or (cap_M < M) or (cap_rank < rank)
    if need_realloc:
        # Expansion strategy: round up to the nearest power of two large enough, reducing the probability of frequent reallocs.
        new_cap_M = _round_up_pow2(max(M, cap_M))
        new_cap_rank = _round_up_pow2(max(rank, cap_rank))

        new_buf = torch.empty(
            (num_slices, new_cap_M, top_k_num, new_cap_rank),
            device=device,
            dtype=dtype,
        )
        _A_WS_BUF[key] = new_buf
        _A_WS_CAP[key] = (new_cap_M, new_cap_rank)
        buf = new_buf

    # Slice to the currently requested size (view, no copy).
    view = buf[:, :M, :, :rank]

    # Note: A workspace must be zeroed when shrink_split_k != 1 or under fully_sharded.
    if must_zero:
        view.zero_()

    return view

# B(delta) workspace: only one expandable large buffer is retained per (device, dtype, num_slices, top_k_num) combination.
_B_WS_BUF: dict[tuple, torch.Tensor] = {}
_B_WS_CAP: dict[tuple, tuple[int, int]] = {}  # (cap_M, cap_N)


def _get_b_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_slices: int,
    M: int,
    top_k_num: int,
    N: int,
    must_zero: bool,
) -> torch.Tensor:
    """
    Returns a view with shape (num_slices, M, top_k_num, N).
    When split_k > 1, this is used to hold deltas for atomic_add (must be cleared first).
    """
    key = (device.type, device.index, dtype, num_slices, top_k_num)

    buf = _B_WS_BUF.get(key)
    cap = _B_WS_CAP.get(key)
    cap_M, cap_N = cap if cap is not None else (0, 0)

    need_realloc = (buf is None) or (cap_M < M) or (cap_N < N)
    if need_realloc:
        new_cap_M = _round_up_pow2(max(M, cap_M))
        new_cap_N = _round_up_pow2(max(N, cap_N))

        new_buf = torch.empty(
            (num_slices, new_cap_M, top_k_num, new_cap_N),
            device=device,
            dtype=dtype,
        )
        _B_WS_BUF[key] = new_buf
        _B_WS_CAP[key] = (new_cap_M, new_cap_N)
        buf = new_buf

    view = buf[:, :M, :, :N]

    # When split_k > 1, the accumulation of deltas relies on atomic_add and must be cleared (only the actually used slices are cleared to avoid a full memset).
    if must_zero:
        view.zero_()

    return view

@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    # adding for kernel
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
    offset: int = 0,
    use_gdc: bool = False,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    w1_lora_b = lora_b_stacked[0]

    # GEMM dims for expand: A: (M * top_k, rank), B: (rank, N_out)
    K = max_lora_rank
    N = w1_output_dim_size

    out_view = output[:, :, offset : offset + num_slices * N]
    max_loras_total = sorted_token_ids.shape[0]

    grid0 = triton.cdiv(EM, block_size_m) * triton.cdiv(N, block_size_n)
    grid = (grid0, len(lora_b_stacked), lora_ids.numel())

    # No longer do .view(-1, rank) on a_intermediate_cache1.
    # Because a_intermediate_cache1 may be a view sliced from a buffer with a larger cap_rank, each row has padding, and view would fail.
    # We directly pass the padded layout to Triton and explicitly tell it the strides for the token and slice axes.
    #
    # Token axis (flattening (M, top_k)): the stride between adjacent tokens is always stride(2) (i.e., cap_rank), as derived:
    # (m, tk) -> (m, tk+1) differs by stride(2); crossing rows (tk=top_k-1 -> tk=0, m+1) also differs by stride(2).
    a_stride_am = a_intermediate_cache1.stride(2)  # = cap_rank
    a_stride_ak = a_intermediate_cache1.stride(3)  # = 1
    a_slice_a_size = a_intermediate_cache1.stride(0)  # actual stride between slices (includes cap_M/cap_rank)

    # In out_view, each slice corresponds to an N chunk in the last dimension; stride between slices = N * stride(last_dim)
    out_slice_c_size = N * out_view.stride(2)

    if split_k == 1:
        # Fast path: directly perform ADD_TO_C on out_view within the kernel (no additional delta workspace needed).
        _fused_moe_lora_kernel[grid](
            a_intermediate_cache1,
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
            a_stride_am,
            a_stride_ak,
            w1_lora_b.stride(0),
            w1_lora_b.stride(1),
            w1_lora_b.stride(3),
            w1_lora_b.stride(2),
            # out_view treated as 2D (token, N): token axis stride = out_view.stride(1), column stride = out_view.stride(2)
            out_view.stride(1),
            out_view.stride(2),
            sorted_token_ids.stride(0),
            expert_ids.stride(0),
            slice_a_size=a_slice_a_size,
            slice_c_size=out_slice_c_size,
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
            SPLIT_K=1,
            USE_GDC=use_gdc,
            launch_pdl=use_gdc,
            IS_PRIMARY=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    # Stable and performant path (split_k>1):
    # First write deltas to b_ws (must be cleared) using atomic_add, then add_ them to out_view in one go.
    b_ws = _get_b_workspace(
        device=device,
        dtype=output.dtype,
        num_slices=num_slices,
        M=M,
        top_k_num=top_k_num,
        N=N,
        must_zero=True,
    )

    # b_ws is also a view sliced from a buffer with a larger cap_N; slice stride must use stride(0)
    b_slice_c_size = b_ws.stride(0)

    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        b_ws,
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
        a_stride_am,
        a_stride_ak,
        w1_lora_b.stride(0),
        w1_lora_b.stride(1),
        w1_lora_b.stride(3),
        w1_lora_b.stride(2),
        # b_ws treated as 2D (token, N): token axis stride = b_ws.stride(2)=cap_N, column stride = b_ws.stride(3)=1
        b_ws.stride(2),
        b_ws.stride(3),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=a_slice_a_size,
        slice_c_size=b_slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_TO_C=False,          # only write delta
        USE_B_L2_CACHE=True,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=split_k,         # retain parallelism
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        IS_PRIMARY=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Do not use out_view.view(...) (could also fail with certain strides); use as_strided to construct a 4D view, zero-copy and more stable.
    # out_view: (M, top_k, num_slices*N) -> out_4d: (M, top_k, num_slices, N)
    out_4d = out_view.as_strided(
        size=(M, top_k_num, num_slices, N),
        stride=(out_view.stride(0), out_view.stride(1), out_slice_c_size, out_view.stride(2)),
    )

    # b_ws: (num_slices, M, top_k, N) -> (M, top_k, num_slices, N)
    out_4d.add_(b_ws.permute(1, 2, 0, 3))




@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
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
    assert (
        sorted_token_ids.dim()
        == expert_ids.dim()
        == topk_weights.dim()
        == qcurr_hidden_states.dim()
        == 2
    )
    assert (
        sorted_token_ids.shape[0]
        == expert_ids.shape[0]
        == num_tokens_post_padded.shape[0]
    )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b = lora_b_stacked[0]

    num_experts = lora_a_stacked[0].shape[1]
    M = topk_weights.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b.shape[2]

    # fully_sharded or shrink_split_k != 1 (atomic_add) must be zeroed
    must_zero_a = fully_sharded or (shrink_split_k != 1)

    # A workspace: may be a padded view (non-contiguous), but shrink/expand kernels already handle strides correctly.
    a_intermediate_cache1 = _get_a_workspace(
        device=device,
        dtype=output.dtype,
        num_slices=num_slices,
        M=M,
        top_k_num=top_k_num,
        rank=max_lora_rank,
        must_zero=must_zero_a,
    )

    use_gdc = supports_pdl(device) and not fully_sharded

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
        use_gdc=use_gdc,
    )

    if fully_sharded:
        # Critical fix: TP collective requires contiguous; our workspace is a slice view, often non-contiguous.
        a_tp = a_intermediate_cache1
        if not a_tp.is_contiguous():
            # Only copy the "actual view size" (M * top_k * rank * num_slices), not the padding.
            a_tp = a_tp.contiguous()

        if max_lora_rank == w1_lora_b.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(a_tp)
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(a_tp)
            # After all_gather, rank becomes full rank.
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
        N=max_lora_rank,
        M=M,
        EM=EM,
        K=K,
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
        use_gdc=use_gdc,
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
