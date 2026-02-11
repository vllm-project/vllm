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


def _merge_adapter_token_lists(
    sorted_token_ids: torch.Tensor,  # (max_loras, max_num_tokens_padded)
    expert_ids: torch.Tensor,  # (max_loras, max_num_m_blocks)
    num_tokens_post_padded: torch.Tensor,  # (max_loras,)
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    block_size_m: int,
    total_m_blocks_ub: int,
    num_valid_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Merges per-adapter token lists into a single flat list, eliminating
    the per-adapter grid dimension from the kernel launch.

    Fully GPU-based — no CPU sync, CUDA graph capture safe.

    Uses a CPU-computed upper bound (total_m_blocks_ub) for output tensor
    sizes and kernel grid sizing. Excess blocks have block_adapter_map=-1
    and are skipped by the kernel with a single load + compare.

    Returns:
        merged_sorted_token_ids: (total_tokens_ub,) flat token IDs
        merged_expert_ids: (total_m_blocks_ub,) flat expert assignments
        block_adapter_map: (total_m_blocks_ub,) maps M-block → adapter ID
        total_m_blocks_ub: upper bound on M-blocks (for grid sizing)
    """
    max_loras_dim = sorted_token_ids.shape[0]
    max_padded = sorted_token_ids.shape[1]
    max_m_blocks = expert_ids.shape[1]
    device = sorted_token_ids.device

    total_tokens_ub = total_m_blocks_ub * block_size_m

    # Handle edge case
    if total_m_blocks_ub == 0 or max_loras_dim == 0:
        return (
            torch.full((block_size_m,), num_valid_tokens,
                       dtype=sorted_token_ids.dtype, device=device),
            torch.full((1,), -1, dtype=expert_ids.dtype, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            1,
        )

    # --- Build per-adapter-slot active flags (all on GPU) ---
    # active_flags[i] = 1 if adapter slot i is active, 0 otherwise
    n_ids = min(lora_ids.shape[0], max_loras_dim)
    ae_size = adapter_enabled.shape[0]

    # Allocate with +1 garbage slot to absorb invalid lora_id writes
    active_flags_ext = torch.zeros(
        max_loras_dim + 1, dtype=torch.int32, device=device
    )
    if n_ids > 0 and ae_size > 0:
        lids = lora_ids[:n_ids]
        valid = (lids >= 0) & (lids < max_loras_dim) & (lids < ae_size)
        safe_lids = lids.clamp(min=0, max=ae_size - 1)
        enabled = adapter_enabled[safe_lids] > 0
        active_mask = (valid & enabled).to(torch.int32)
        # Redirect invalid lora_ids to garbage slot (index max_loras_dim)
        target = torch.where(valid, lids.long(), max_loras_dim)
        active_flags_ext.scatter_(0, target, active_mask)
    active_flags = active_flags_ext[:max_loras_dim]

    # --- Per-adapter m-block and token counts (GPU) ---
    m_blocks_per = (
        (num_tokens_post_padded.int() + block_size_m - 1) // block_size_m
    ) * active_flags
    ntpp_active = num_tokens_post_padded.int() * active_flags

    # --- Exclusive prefix sums for merged offsets (GPU) ---
    m_block_offsets = torch.zeros(
        max_loras_dim, dtype=torch.int32, device=device
    )
    if max_loras_dim > 1:
        m_block_offsets[1:] = torch.cumsum(m_blocks_per[:-1], dim=0)
    token_offsets = m_block_offsets.long() * block_size_m

    # --- Allocate merged output tensors (upper bound + garbage slot) ---
    garbage_m = total_m_blocks_ub
    garbage_t = total_tokens_ub

    merged_sorted = torch.full(
        (total_tokens_ub + block_size_m,),  # +block_size_m for garbage
        num_valid_tokens,
        dtype=sorted_token_ids.dtype,
        device=device,
    )
    merged_experts = torch.full(
        (total_m_blocks_ub + 1,), -1,  # +1 for garbage
        dtype=expert_ids.dtype, device=device,
    )
    block_adapter_map = torch.full(
        (total_m_blocks_ub + 1,), -1,  # +1 for garbage
        dtype=torch.int32, device=device,
    )

    # --- Scatter m-block data (expert_ids + block_adapter_map) ---
    local_m = torch.arange(
        max_m_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0).expand(max_loras_dim, -1)

    m_valid = local_m < m_blocks_per.unsqueeze(1)
    global_m = (m_block_offsets.unsqueeze(1) + local_m).long()
    global_m = torch.where(m_valid, global_m, garbage_m)

    merged_experts.scatter_(
        0, global_m.reshape(-1), expert_ids.reshape(-1)
    )

    adapter_ids = torch.arange(
        max_loras_dim, dtype=torch.int32, device=device
    ).unsqueeze(1).expand(-1, max_m_blocks)
    block_adapter_map.scatter_(
        0, global_m.reshape(-1), adapter_ids.reshape(-1)
    )

    # --- Scatter token data (sorted_token_ids) ---
    local_t = torch.arange(
        max_padded, dtype=torch.int64, device=device
    ).unsqueeze(0).expand(max_loras_dim, -1)

    t_valid = local_t < ntpp_active.unsqueeze(1).long()
    global_t = token_offsets.unsqueeze(1) + local_t
    global_t = torch.where(t_valid, global_t, garbage_t)

    merged_sorted.scatter_(
        0, global_t.reshape(-1), sorted_token_ids.reshape(-1)
    )

    return (
        merged_sorted[:total_tokens_ub],
        merged_experts[:total_m_blocks_ub],
        block_adapter_map[:total_m_blocks_ub],
        total_m_blocks_ub,
    )


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
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    lora_ids,
    adapter_enabled,
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
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
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
    lora_id = tl.load(lora_ids + lora_idx)

    if lora_id == -1:
        # Early exit for the no-lora case.
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        # Early exit for the no moe lora case.
        return
    max_loras = tl.num_programs(axis=2)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
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
    # get the expert_id to process curr shard
    ind = lora_id * stride_el + pid_m
    expert_id = tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)
    if expert_id == -1:
        return
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_ind = stride_tl * lora_id + offs_token_id
    offs_token = tl.load(
        sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0
    )
    token_mask = offs_token < num_valid_tokens

    # get a_ptrs,b_ptrs
    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        tl.extra.cuda.gdc_launch_dependents()

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # GDC wait waits for ALL programs in the prior kernel to complete
    # before continuing.
    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        # pre-fetch lora weight
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "slice_a_size",
        "slice_c_size",
    ]
)
def _fused_moe_lora_merged_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    block_adapter_map_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension.
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
):
    """
    Merged variant of the fused MoE LoRA kernel.

    Instead of using program_id(axis=2) to iterate over max_loras adapter
    slots (many of which are inactive or over-allocated), this kernel
    operates on a pre-merged flat token list. The block_adapter_map tells
    each M-block which adapter's LoRA weights to use.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m, pid_n
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

    # Look up which adapter this M-block belongs to
    lora_id = tl.load(block_adapter_map_ptr + pid_m)
    if lora_id == -1:
        # No work for this block (inactive or beyond merged data)
        return

    # Look up the expert for this M-block
    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    # get a_ptr, b_ptr, c_ptr for the current slice
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Flat indexing into merged sorted_token_ids
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # get a_ptrs, b_ptrs
    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    if USE_GDC and IS_PRIMARY:
        tl.extra.cuda.gdc_launch_dependents()

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "K_input",
        "output_dim",
        "offset",
    ]
)
def _fused_moe_lora_fused_se_kernel(
    # Input (qcurr_hidden_states)
    input_ptr,
    # Weight pointer tables (from _get_ptr)
    lora_a_ptr_table,
    lora_b_ptr_table,
    # Output (final output, 2D view: M*top_k x total_output_dim)
    output_ptr,
    # Routing weights
    topk_weights_ptr,
    # Merged indexing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    block_adapter_map_ptr,
    # Dimensions
    K_input,
    output_dim,
    EM,
    num_valid_tokens,
    # Input strides
    stride_input_m,
    stride_input_k,
    # lora_A strides: shape (max_loras, num_experts, rank, K)
    # passed as (adapter, expert, K, rank) following existing convention
    stride_a_l,
    stride_a_e,
    stride_a_k,
    stride_a_n,
    # lora_B strides: shape (max_loras, num_experts, output_dim, rank)
    # passed as (adapter, expert, rank, output_dim)
    stride_b_l,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    # Output strides (2D flattened)
    stride_out_m,
    stride_out_n,
    # Column offset in output
    offset,
    # Constexprs
    RANK: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K_SHRINK: tl.constexpr,
    BLOCK_N_EXPAND: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """
    Fused shrink+expand kernel for MoE LoRA.

    Combines the shrink (input @ lora_A) and expand (intermediate @ lora_B)
    phases into a single kernel launch. The intermediate tensor
    (BLOCK_SIZE_M x RANK) stays in registers, eliminating global memory
    traffic and one kernel launch.

    Each program handles one (M-block, slice) pair:
    1. Shrink: iterate over K_input, accumulate intermediate in fp32
    2. Cast intermediate to compute dtype, apply routing weight
    3. Expand: loop over N-tiles of output_dim, write directly to output
    """
    pid_m = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    # Look up adapter and expert for this M-block
    lora_id = tl.load(block_adapter_map_ptr + pid_m)
    if lora_id == -1:
        return
    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    # Load token IDs for this M-block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64
    )
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # Load lora_A base pointer for this slice
    cur_lora_a_ptr = tl.load(lora_a_ptr_table + slice_id).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    # ===== Shrink phase =====
    # intermediate[BLOCK_M, RANK] = input[BLOCK_M, K] @ lora_A[K, RANK]
    intermediate = tl.zeros((BLOCK_SIZE_M, RANK), dtype=tl.float32)

    offs_rank = tl.arange(0, RANK).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K_SHRINK)

    # Base pointers for input (indexed by base token = offs_token // top_k)
    input_ptrs = input_ptr + (
        offs_token[:, None] // top_k * stride_input_m
        + offs_k[None, :] * stride_input_k
    )

    # Base pointers for lora_A weight
    a_weight_ptrs = (
        cur_lora_a_ptr
        + lora_id * stride_a_l
        + expert_id * stride_a_e
        + offs_k[:, None] * stride_a_k
        + offs_rank[None, :] * stride_a_n
    )

    grid_k = tl.cdiv(K_input, BLOCK_K_SHRINK)
    for k in range(0, grid_k):
        k_remaining = K_input - k * BLOCK_K_SHRINK
        # Load input tile [BLOCK_M, BLOCK_K]
        a_tile = tl.load(
            input_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        # Load lora_A tile [BLOCK_K, RANK]
        w_tile = tl.load(
            a_weight_ptrs,
            mask=offs_k[:, None] < k_remaining,
            other=0.0,
        )
        intermediate += tl.dot(a_tile, w_tile)
        input_ptrs += BLOCK_K_SHRINK * stride_input_k
        a_weight_ptrs += BLOCK_K_SHRINK * stride_a_k

    # Apply routing weight while still in fp32 (better precision)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0
        )
        intermediate = intermediate * moe_weight[:, None]

    # Cast to weight dtype for expand dot (matches lora_B dtype, which
    # may differ from output dtype when output comes from quantized MoE).
    # Must be AFTER routing weight mul to avoid fp32 promotion from
    # topk_weights breaking the expand tl.dot dtype match.
    intermediate = intermediate.to(input_ptr.dtype.element_ty)

    # ===== Expand phase =====
    # Loop over N-tiles: output += intermediate[BLOCK_M, RANK] @ lora_B[RANK, BLOCK_N]
    cur_lora_b_ptr = tl.load(lora_b_ptr_table + slice_id).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )

    # Output column start for this slice
    output_col_start = slice_id * output_dim + offset

    # Base pointer for lora_B weight (adapter + expert offset)
    b_weight_base = (
        cur_lora_b_ptr + lora_id * stride_b_l + expert_id * stride_b_e
    )

    for n_start in range(0, output_dim, BLOCK_N_EXPAND):
        offs_bn = (n_start + tl.arange(0, BLOCK_N_EXPAND)).to(tl.int64)
        n_mask = offs_bn < output_dim

        # Load lora_B tile [RANK, BLOCK_N]
        b_weight_ptrs = (
            b_weight_base
            + offs_rank[:, None] * stride_b_k
            + offs_bn[None, :] * stride_b_n
        )
        b_tile = tl.load(b_weight_ptrs, mask=n_mask[None, :], other=0.0)

        # Compute output tile [BLOCK_M, BLOCK_N]
        result = tl.dot(intermediate, b_tile)
        result = result.to(output_ptr.dtype.element_ty)

        # Load-add-store to output
        out_col = output_col_start + offs_bn
        out_ptrs = (
            output_ptr
            + offs_token[:, None] * stride_out_m
            + out_col[None, :] * stride_out_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
        existing = tl.load(out_ptrs, mask=out_mask, other=0.0)
        tl.store(out_ptrs, existing + result, mask=out_mask)


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "K_input",
        "output_dim",
        "offset",
    ]
)
def _fused_moe_lora_persistent_se_kernel(
    # Input (qcurr_hidden_states)
    input_ptr,
    # Weight pointer tables (from _get_ptr)
    lora_a_ptr_table,
    lora_b_ptr_table,
    # Output (final output, 2D view: M*top_k x total_output_dim)
    output_ptr,
    # Routing weights
    topk_weights_ptr,
    # Merged indexing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    block_adapter_map_ptr,
    # Dimensions
    K_input,
    output_dim,
    EM,
    num_valid_tokens,
    # Input strides
    stride_input_m,
    stride_input_k,
    # lora_A strides: shape (max_loras, num_experts, rank, K)
    # passed as (adapter, expert, K, rank) following existing convention
    stride_a_l,
    stride_a_e,
    stride_a_k,
    stride_a_n,
    # lora_B strides: shape (max_loras, num_experts, output_dim, rank)
    # passed as (adapter, expert, rank, output_dim)
    stride_b_l,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    # Output strides (2D flattened)
    stride_out_m,
    stride_out_n,
    # Column offset in output
    offset,
    # Constexprs
    RANK: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K_SHRINK: tl.constexpr,
    BLOCK_N_EXPAND: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """
    Persistent fused shrink+expand kernel for MoE LoRA.

    Instead of one CTA per M-block, a fixed number of CTAs (≤ num_SMs)
    each process multiple contiguous M-blocks in a loop. Consecutive
    M-blocks in the merged list often share the same (adapter, expert),
    so lora_A/lora_B tiles stay warm in L2 cache across iterations.

    Degrades gracefully: when num_CTAs >= num_M_blocks, each CTA
    processes at most 1 block (same as the non-persistent kernel).
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    num_ctas = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)

    # Contiguous block assignment for L2 cache locality:
    # consecutive M-blocks processed by the same CTA are likely
    # to share (adapter, expert), keeping lora weights warm in L2.
    blocks_per_cta = tl.cdiv(num_pid_m, num_ctas)
    start_m = pid * blocks_per_cta

    # Hoist slice-invariant pointer loads
    cur_lora_a_ptr = tl.load(lora_a_ptr_table + slice_id).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )
    cur_lora_b_ptr = tl.load(lora_b_ptr_table + slice_id).to(
        tl.pointer_type(input_ptr.dtype.element_ty)
    )

    # Hoist loop-invariant values
    offs_rank = tl.arange(0, RANK).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K_SHRINK)
    grid_k = tl.cdiv(K_input, BLOCK_K_SHRINK)
    output_col_start = slice_id * output_dim + offset

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    # ===== Persistent loop over contiguous M-blocks =====
    for local_idx in range(blocks_per_cta):
        pid_m = start_m + local_idx
        if pid_m < num_pid_m:
            # Look up adapter and expert for this M-block
            lora_id = tl.load(block_adapter_map_ptr + pid_m)
            if lora_id != -1:
                expert_id = tl.load(expert_ids_ptr + pid_m)
                if expert_id != -1:
                    # Load token IDs for this M-block
                    offs_token_id = (
                        pid_m * BLOCK_SIZE_M
                        + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
                    )
                    offs_token = tl.load(
                        sorted_token_ids_ptr + offs_token_id
                    )
                    token_mask = offs_token < num_valid_tokens

                    # ===== Shrink phase =====
                    intermediate = tl.zeros(
                        (BLOCK_SIZE_M, RANK), dtype=tl.float32
                    )

                    input_ptrs = input_ptr + (
                        offs_token[:, None] // top_k * stride_input_m
                        + offs_k[None, :] * stride_input_k
                    )
                    a_weight_ptrs = (
                        cur_lora_a_ptr
                        + lora_id * stride_a_l
                        + expert_id * stride_a_e
                        + offs_k[:, None] * stride_a_k
                        + offs_rank[None, :] * stride_a_n
                    )

                    for k in range(0, grid_k):
                        k_remaining = K_input - k * BLOCK_K_SHRINK
                        a_tile = tl.load(
                            input_ptrs,
                            mask=token_mask[:, None]
                            & (offs_k[None, :] < k_remaining),
                            other=0.0,
                        )
                        w_tile = tl.load(
                            a_weight_ptrs,
                            mask=offs_k[:, None] < k_remaining,
                            other=0.0,
                        )
                        intermediate += tl.dot(a_tile, w_tile)
                        input_ptrs += BLOCK_K_SHRINK * stride_input_k
                        a_weight_ptrs += BLOCK_K_SHRINK * stride_a_k

                    # Apply routing weight while still in fp32
                    if MUL_ROUTED_WEIGHT:
                        moe_weight = tl.load(
                            topk_weights_ptr + offs_token,
                            mask=token_mask,
                            other=0,
                        )
                        intermediate = intermediate * moe_weight[:, None]

                    # Cast to weight dtype AFTER routing mul to avoid
                    # fp32 promotion breaking expand tl.dot dtype match
                    intermediate = intermediate.to(
                        input_ptr.dtype.element_ty
                    )

                    # ===== Expand phase =====
                    b_weight_base = (
                        cur_lora_b_ptr
                        + lora_id * stride_b_l
                        + expert_id * stride_b_e
                    )

                    for n_start in range(0, output_dim, BLOCK_N_EXPAND):
                        offs_bn = (
                            n_start + tl.arange(0, BLOCK_N_EXPAND)
                        ).to(tl.int64)
                        n_mask = offs_bn < output_dim

                        b_weight_ptrs = (
                            b_weight_base
                            + offs_rank[:, None] * stride_b_k
                            + offs_bn[None, :] * stride_b_n
                        )
                        b_tile = tl.load(
                            b_weight_ptrs, mask=n_mask[None, :], other=0.0
                        )

                        result = tl.dot(intermediate, b_tile)
                        result = result.to(output_ptr.dtype.element_ty)

                        out_col = output_col_start + offs_bn
                        out_ptrs = (
                            output_ptr
                            + offs_token[:, None] * stride_out_m
                            + out_col[None, :] * stride_out_n
                        )
                        out_mask = token_mask[:, None] & n_mask[None, :]
                        existing = tl.load(
                            out_ptrs, mask=out_mask, other=0.0
                        )
                        tl.store(
                            out_ptrs, existing + result, mask=out_mask
                        )


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, num_tokens, top_k_num, max_lora_rank)
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
    ## adding for kernel
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
    w1_lora_a_stacked = lora_a_stacked[0]
    use_gdc = supports_pdl(qcurr_hidden_states.device)
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        lora_a_stacked[0].shape[0],
    )
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
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
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
        IS_PRIMARY=True,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    b_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, output_dim_size)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
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
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    use_gdc = supports_pdl(a_intermediate_cache1.device)
    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,  # Set split_k = 1 for expand calls
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
    }

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        lora_b_stacked[0].shape[0],
    )
    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        b_intermediate_cache1,
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
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        b_intermediate_cache1.stride(2),
        b_intermediate_cache1.stride(3),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=b_intermediate_cache1.numel() // num_slices,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        IS_PRIMARY=False,
        **expand_config,
    )
    for i in range(num_slices):
        output[:, :, i * N + offset : (i + 1) * N + offset] += b_intermediate_cache1[i]


@torch.inference_mode()
def _fused_moe_lora_shrink_merged(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    merged_sorted_token_ids: torch.Tensor,  # (total_tokens,) flat
    merged_expert_ids: torch.Tensor,  # (total_m_blocks,) flat
    block_adapter_map: torch.Tensor,  # (total_m_blocks,)
    top_k_num: int,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
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
    w1_lora_a_stacked = lora_a_stacked[0]
    use_gdc = supports_pdl(qcurr_hidden_states.device)
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        1,
    )
    _fused_moe_lora_merged_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        merged_sorted_token_ids,
        merged_expert_ids,
        block_adapter_map,
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
        a_intermediate_cache1.stride(2),
        a_intermediate_cache1.stride(3),
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        top_k=1 if mul_routed_weight else top_k_num,
        MUL_ROUTED_WEIGHT=False,
        IS_PRIMARY=True,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand_merged(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    b_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    merged_sorted_token_ids: torch.Tensor,  # (total_tokens,) flat
    merged_expert_ids: torch.Tensor,  # (total_m_blocks,) flat
    block_adapter_map: torch.Tensor,  # (total_m_blocks,)
    top_k_num: int,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
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
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    use_gdc = supports_pdl(a_intermediate_cache1.device)
    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,
    }

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        1,
    )
    _fused_moe_lora_merged_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        b_intermediate_cache1,
        topk_weights,
        merged_sorted_token_ids,
        merged_expert_ids,
        block_adapter_map,
        N,
        K,
        EM,
        num_tokens,
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        b_intermediate_cache1.stride(2),
        b_intermediate_cache1.stride(3),
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=b_intermediate_cache1.numel() // num_slices,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        IS_PRIMARY=False,
        **expand_config,
    )
    for i in range(num_slices):
        output[:, :, i * N + offset : (i + 1) * N + offset] += b_intermediate_cache1[i]


@torch.inference_mode()
def _fused_moe_lora_shrink_expand_fused(
    output: torch.Tensor,  # (num_tokens, top_k_num, total_output_dim)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K)
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    merged_sorted_token_ids: torch.Tensor,  # flat
    merged_expert_ids: torch.Tensor,  # flat
    block_adapter_map: torch.Tensor,  # flat
    top_k_num: int,
    device: torch.device,
    M: int,
    EM: int,
    num_tokens: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_k_shrink: int,
    block_size_n_expand: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
) -> None:
    """
    Fused shrink+expand wrapper. Launches a kernel that keeps the
    intermediate (BLOCK_M x rank) in registers instead of writing
    to global memory, eliminating one kernel launch and global memory
    round-trip.
    """
    a_ptr_table = _get_ptr(lora_a_stacked, device)
    b_ptr_table = _get_ptr(lora_b_stacked, device)

    w1_lora_a_stacked = lora_a_stacked[0]
    w1_lora_b_stacked = lora_b_stacked[0]

    # View output as 2D for stride calculation
    output_2d = output.view(-1, output.shape[-1])

    K_input = qcurr_hidden_states.shape[1]

    use_gdc = supports_pdl(qcurr_hidden_states.device)

    num_pid_m = triton.cdiv(EM, block_size_m)

    grid = (num_pid_m, num_slices)

    _fused_moe_lora_fused_se_kernel[grid](
        qcurr_hidden_states,
        a_ptr_table,
        b_ptr_table,
        output_2d,
        topk_weights,
        merged_sorted_token_ids,
        merged_expert_ids,
        block_adapter_map,
        K_input,
        w1_output_dim_size,
        EM,
        num_tokens,
        # Input strides
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        # lora_A strides: (adapter, expert, K, rank)
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        # lora_B strides: (adapter, expert, rank, output_dim)
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        # Output strides
        output_2d.stride(0),
        output_2d.stride(1),
        # Column offset
        offset,
        # Constexprs
        RANK=max_lora_rank,
        top_k=1 if mul_routed_weight else top_k_num,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_K_SHRINK=block_size_k_shrink,
        BLOCK_N_EXPAND=block_size_n_expand,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        num_warps=num_warps,
        num_stages=num_stages,
    )


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
    total_m_blocks_ub: int = 0,
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
    w1_lora_b_stacked = lora_b_stacked[0]
    N = max_lora_rank
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]

    # Use caller-provided upper bound, or fall back to conservative bound
    if total_m_blocks_ub <= 0:
        num_experts = lora_a_stacked[0].shape[1]
        max_loras_dim = sorted_token_ids.shape[0]
        total_padded_ub = (
            num_tokens
            + max_loras_dim * num_experts * (shrink_block_size_m - 1)
        )
        total_m_blocks_ub = (
            triton.cdiv(total_padded_ub, shrink_block_size_m) + max_loras_dim
        )

    # Merge per-adapter token lists into flat layout (fully GPU, graph-safe)
    merged_sorted, merged_experts, block_adapter_map, total_m_blocks = (
        _merge_adapter_token_lists(
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            lora_ids,
            adapter_enabled,
            shrink_block_size_m,
            total_m_blocks_ub,
            num_tokens,
        )
    )
    EM = total_m_blocks * shrink_block_size_m

    # Use fused shrink+expand when possible (keeps intermediate in registers)
    can_fuse = (
        not fully_sharded
        and max_lora_rank >= 16
        and shrink_split_k == 1
    )

    if can_fuse:
        _fused_moe_lora_shrink_expand_fused(
            output,
            qcurr_hidden_states,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            merged_sorted,
            merged_experts,
            block_adapter_map,
            top_k_num,
            device,
            M,
            EM,
            num_tokens,
            num_slices,
            max_lora_rank,
            w1_output_dim_size,
            shrink_block_size_m,
            shrink_block_size_k,
            expand_block_size_n,
            shrink_group_size_m,
            shrink_num_warps,
            shrink_num_stages,
            mul_routed_weight,
            offset,
        )
    else:
        # Fallback: separate shrink and expand kernels
        # (needed for fully_sharded which requires all_reduce between phases,
        #  rank < 16 which doesn't meet tl.dot minimum, or split_k > 1)
        a_intermediate_cache1 = torch.zeros(
            (num_slices, M, top_k_num, max_lora_rank),
            dtype=output.dtype,
            device=device,
        )

        b_intermediate_cache1 = torch.zeros(
            (num_slices, M, top_k_num, w1_output_dim_size),
            dtype=output.dtype,
            device=device,
        )

        _fused_moe_lora_shrink_merged(
            a_intermediate_cache1,
            qcurr_hidden_states,
            lora_a_stacked,
            topk_weights,
            merged_sorted,
            merged_experts,
            block_adapter_map,
            top_k_num,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_slices,
            shrink_block_size_m,
            shrink_block_size_n,
            shrink_block_size_k,
            shrink_group_size_m,
            shrink_num_warps,
            shrink_num_stages,
            shrink_split_k,
            mul_routed_weight,
        )

        if fully_sharded:
            if max_lora_rank == w1_lora_b_stacked.shape[-1]:
                a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                    a_intermediate_cache1
                )
            else:
                a_intermediate_cache1 = tensor_model_parallel_all_gather(
                    a_intermediate_cache1
                )
                max_lora_rank = a_intermediate_cache1.shape[-1]

        _fused_moe_lora_expand_merged(
            output,
            a_intermediate_cache1,
            b_intermediate_cache1,
            lora_b_stacked,
            topk_weights,
            merged_sorted,
            merged_experts,
            block_adapter_map,
            top_k_num,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_slices,
            max_lora_rank,
            w1_output_dim_size,
            expand_block_size_m,
            expand_block_size_n,
            expand_block_size_k,
            expand_group_size_m,
            expand_num_warps,
            expand_num_stages,
            expand_split_k,
            mul_routed_weight,
            offset,
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
    total_m_blocks_ub: int = 0,
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
