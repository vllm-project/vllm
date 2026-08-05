# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sparse-index assembly and paged K/V gather kernels."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _assemble_context_kernel(
    topk_ptr,
    out_ptr,
    valid_ptr,
    stride_topk_h,
    stride_out_h,
    stride_valid_h,
    sink_len,
    local_start,
    local_len,
    topk_len,
    n_ctx,
    BLOCK: tl.constexpr,
):
    h = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_ctx
    in_sink = offs < sink_len
    in_local = (offs >= sink_len) & (offs < sink_len + local_len)
    topk_pos = offs - sink_len - local_len
    topk_safe = tl.maximum(0, tl.minimum(topk_pos, topk_len - 1))
    selected = tl.load(
        topk_ptr + h * stride_topk_h + topk_safe,
        mask=mask & ~(in_sink | in_local),
        other=-1,
    )
    logical = tl.where(
        in_sink,
        offs,
        tl.where(in_local, local_start + offs - sink_len, selected),
    )
    tl.store(out_ptr + h * stride_out_h + offs, logical, mask=mask)
    tl.store(valid_ptr + h * stride_valid_h + offs, logical >= 0, mask=mask)


@triton.jit
def _assemble_context_batch_kernel(
    topk_ptr,
    seq_lens_ptr,
    out_ptr,
    valid_ptr,
    stride_topk_b,
    stride_topk_h,
    stride_out_b,
    stride_out_h,
    stride_valid_b,
    stride_valid_h,
    sink_size,
    local_size,
    topk_len,
    n_ctx,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + b)
    sink_len = tl.minimum(sink_size, seq_len)
    local_start = tl.maximum(sink_len, seq_len - local_size)
    local_len = seq_len - local_start
    offs = tl.arange(0, BLOCK)
    mask = offs < n_ctx
    in_sink = offs < sink_len
    in_local = (offs >= sink_len) & (offs < sink_len + local_len)
    topk_pos = offs - sink_len - local_len
    topk_safe = tl.maximum(0, tl.minimum(topk_pos, topk_len - 1))
    selected = tl.load(
        topk_ptr + b * stride_topk_b + h * stride_topk_h + topk_safe,
        mask=mask & ~(in_sink | in_local),
        other=-1,
    )
    logical = tl.where(
        in_sink,
        offs,
        tl.where(in_local, local_start + offs - sink_len, selected),
    )
    # Past the true n_ctx for this request (variable local length): mark invalid.
    in_range = offs < (sink_len + local_len + topk_len)
    logical = tl.where(in_range, logical, -1)
    tl.store(
        out_ptr + b * stride_out_b + h * stride_out_h + offs, logical, mask=mask
    )
    tl.store(
        valid_ptr + b * stride_valid_b + h * stride_valid_h + offs,
        logical >= 0,
        mask=mask,
    )


@triton.jit
def _paged_gather_kv_kernel(
    key_ptr,
    value_ptr,
    block_table_ptr,
    logical_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_l_h,
    stride_l_t,
    stride_ok_h,
    stride_ok_t,
    stride_ok_d,
    stride_ov_h,
    stride_ov_t,
    stride_ov_d,
    block_size,
    n_ctx,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    logical = tl.load(logical_ptr + h * stride_l_h + t * stride_l_t)
    valid_token = logical >= 0
    logical_safe = tl.maximum(logical, 0)
    logical_block = logical_safe // block_size
    token_offset = logical_safe - logical_block * block_size
    physical_block = tl.load(block_table_ptr + logical_block)
    valid_token = valid_token & (physical_block >= 0)
    physical_safe = tl.maximum(physical_block, 0)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < head_dim
    k_offset = (
        physical_safe * stride_k_b
        + token_offset * stride_k_t
        + h * stride_k_h
        + offs_d * stride_k_d
    )
    v_offset = (
        physical_safe * stride_v_b
        + token_offset * stride_v_t
        + h * stride_v_h
        + offs_d * stride_v_d
    )
    key = tl.load(key_ptr + k_offset, mask=mask & valid_token, other=0.0)
    value = tl.load(value_ptr + v_offset, mask=mask & valid_token, other=0.0)
    tl.store(
        out_k_ptr + h * stride_ok_h + t * stride_ok_t + offs_d * stride_ok_d,
        key,
        mask=mask,
    )
    tl.store(
        out_v_ptr + h * stride_ov_h + t * stride_ov_t + offs_d * stride_ov_d,
        value,
        mask=mask,
    )


@triton.jit
def _paged_gather_kv_batch_kernel(
    key_ptr,
    value_ptr,
    block_table_ptr,
    logical_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_l_b,
    stride_l_h,
    stride_l_t,
    stride_ok_b,
    stride_ok_h,
    stride_ok_t,
    stride_ok_d,
    stride_ov_b,
    stride_ov_h,
    stride_ov_t,
    stride_ov_d,
    block_size,
    n_ctx,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    logical = tl.load(
        logical_ptr + b * stride_l_b + h * stride_l_h + t * stride_l_t
    )
    valid_token = logical >= 0
    logical_safe = tl.maximum(logical, 0)
    logical_block = logical_safe // block_size
    token_offset = logical_safe - logical_block * block_size
    physical_block = tl.load(block_table_ptr + b * stride_bt_b + logical_block)
    valid_token = valid_token & (physical_block >= 0)
    physical_safe = tl.maximum(physical_block, 0)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < head_dim
    k_offset = (
        physical_safe * stride_k_b
        + token_offset * stride_k_t
        + h * stride_k_h
        + offs_d * stride_k_d
    )
    v_offset = (
        physical_safe * stride_v_b
        + token_offset * stride_v_t
        + h * stride_v_h
        + offs_d * stride_v_d
    )
    key = tl.load(key_ptr + k_offset, mask=mask & valid_token, other=0.0)
    value = tl.load(value_ptr + v_offset, mask=mask & valid_token, other=0.0)
    tl.store(
        out_k_ptr
        + b * stride_ok_b
        + h * stride_ok_h
        + t * stride_ok_t
        + offs_d * stride_ok_d,
        key,
        mask=mask,
    )
    tl.store(
        out_v_ptr
        + b * stride_ov_b
        + h * stride_ov_h
        + t * stride_ov_t
        + offs_d * stride_ov_d,
        value,
        mask=mask,
    )


def assemble_context(
    seq_len: int,
    topk: torch.Tensor,
    sink_size: int,
    local_size: int,
    out: torch.Tensor | None = None,
    valid_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble sink/local/retrieval indices in one launch.

    When ``out`` (and optionally ``valid_out``) are provided and wide enough,
    the kernel writes into the first ``n_ctx`` columns of those preallocated
    buffers instead of allocating fresh tensors, and a view of that width is
    returned. This keeps the hot decode path allocation-free.
    """
    heads, topk_len = topk.shape
    sink_len = min(int(sink_size), int(seq_len))
    local_start = max(sink_len, int(seq_len) - int(local_size))
    local_len = int(seq_len) - local_start
    n_ctx = sink_len + local_len + topk_len
    if out is not None and out.shape[0] == heads and out.shape[1] >= n_ctx:
        out = out[:, :n_ctx]
    else:
        out = torch.empty(heads, n_ctx, dtype=torch.int64, device=topk.device)
    if (
        valid_out is not None
        and valid_out.shape[0] == heads
        and valid_out.shape[1] >= n_ctx
    ):
        valid = valid_out[:, :n_ctx]
    else:
        valid = torch.empty(heads, n_ctx, dtype=torch.bool, device=topk.device)
    block = triton.next_power_of_2(n_ctx)
    _assemble_context_kernel[(heads,)](
        topk,
        out,
        valid,
        topk.stride(0),
        out.stride(0),
        valid.stride(0),
        sink_len,
        local_start,
        local_len,
        topk_len,
        n_ctx,
        BLOCK=block,
        num_warps=4,
    )
    return out, valid


def assemble_context_batch(
    seq_lens: torch.Tensor,
    topk: torch.Tensor,
    sink_size: int,
    local_size: int,
    out: torch.Tensor | None = None,
    valid_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched assemble: topk [B, Hkv, topk], seq_lens [B] on device."""
    batch, heads, topk_len = topk.shape
    # For long-context sparse decode every request exceeds sink+local, so
    # n_ctx is identical across the batch. Pad to that fixed width.
    n_ctx = int(sink_size) + int(local_size) + topk_len
    if (
        out is not None
        and out.shape[0] == batch
        and out.shape[1] == heads
        and out.shape[2] >= n_ctx
    ):
        out = out[:, :, :n_ctx]
    else:
        out = torch.empty(
            batch, heads, n_ctx, dtype=torch.int64, device=topk.device
        )
    if (
        valid_out is not None
        and valid_out.shape[0] == batch
        and valid_out.shape[1] == heads
        and valid_out.shape[2] >= n_ctx
    ):
        valid = valid_out[:, :, :n_ctx]
    else:
        valid = torch.empty(
            batch, heads, n_ctx, dtype=torch.bool, device=topk.device
        )
    # Preserve the scheduler's native integer dtype.  Casting the one-element
    # decode tensor from int32 to int64 launched an elementwise copy for every
    # layer; Triton integer arithmetic supports either dtype directly.
    seq_lens_dev = seq_lens.to(device=topk.device).contiguous()
    block = triton.next_power_of_2(n_ctx)
    _assemble_context_batch_kernel[(batch, heads)](
        topk.contiguous(),
        seq_lens_dev,
        out,
        valid,
        topk.stride(0),
        topk.stride(1),
        out.stride(0),
        out.stride(1),
        valid.stride(0),
        valid.stride(1),
        int(sink_size),
        int(local_size),
        topk_len,
        n_ctx,
        BLOCK=block,
        num_warps=4,
    )
    return out, valid


@triton.jit
def _paged_gather_from_topk_batch_kernel(
    key_ptr,
    value_ptr,
    block_table_ptr,
    topk_ptr,
    seq_lens_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_topk_b,
    stride_topk_h,
    stride_ok_b,
    stride_ok_h,
    stride_ok_t,
    stride_ok_d,
    stride_ov_b,
    stride_ov_h,
    stride_ov_t,
    stride_ov_d,
    block_size,
    sink_size,
    local_size,
    topk_len,
    n_ctx,
    head_dim,
    max_blocks,
    num_physical_blocks,
    BLOCK_D: tl.constexpr,
):
    """Fused sink/local/topk logical-id assembly + paged K/V gather."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    seq_len = tl.load(seq_lens_ptr + b)
    sink_len = tl.minimum(sink_size, seq_len)
    local_start = tl.maximum(sink_len, seq_len - local_size)
    local_len = seq_len - local_start
    in_sink = t < sink_len
    in_local = (t >= sink_len) & (t < sink_len + local_len)
    topk_pos = t - sink_len - local_len
    in_topk = (t >= sink_len + local_len) & (topk_pos < topk_len)
    topk_safe = tl.maximum(0, tl.minimum(topk_pos, topk_len - 1))
    selected = tl.load(
        topk_ptr + b * stride_topk_b + h * stride_topk_h + topk_safe,
        mask=in_topk,
        other=-1,
    )
    logical = tl.where(
        in_sink,
        t,
        tl.where(in_local, local_start + t - sink_len, selected),
    )
    valid_token = (
        (in_sink | in_local | in_topk)
        & (logical >= 0)
        & (logical < seq_len)
    )
    logical_safe = tl.maximum(logical, 0)
    logical_block = logical_safe // block_size
    token_offset = logical_safe - logical_block * block_size
    in_table = logical_block < max_blocks
    physical_block = tl.load(
        block_table_ptr + b * stride_bt_b + logical_block,
        mask=valid_token & in_table,
        other=-1,
    )
    valid_token = (
        valid_token
        & in_table
        & (physical_block >= 0)
        & (physical_block < num_physical_blocks)
    )
    physical_safe = tl.maximum(
        0, tl.minimum(physical_block, num_physical_blocks - 1)
    )
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < head_dim
    # ``physical_block`` is int32. Large KV pools can exceed 262,144 blocks,
    # so physical_block * stride_k_b may overflow 32-bit address arithmetic.
    physical_safe_i64 = physical_safe.to(tl.int64)
    token_offset_i64 = token_offset.to(tl.int64)
    h_i64 = h.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)
    k_offset = (
        physical_safe_i64 * stride_k_b
        + token_offset_i64 * stride_k_t
        + h_i64 * stride_k_h
        + offs_d_i64 * stride_k_d
    )
    v_offset = (
        physical_safe_i64 * stride_v_b
        + token_offset_i64 * stride_v_t
        + h_i64 * stride_v_h
        + offs_d_i64 * stride_v_d
    )
    key = tl.load(key_ptr + k_offset, mask=mask & valid_token, other=0.0)
    value = tl.load(value_ptr + v_offset, mask=mask & valid_token, other=0.0)
    tl.store(
        out_k_ptr
        + b * stride_ok_b
        + h * stride_ok_h
        + t * stride_ok_t
        + offs_d * stride_ok_d,
        key,
        mask=mask,
    )
    tl.store(
        out_v_ptr
        + b * stride_ov_b
        + h * stride_ov_h
        + t * stride_ov_t
        + offs_d * stride_ov_d,
        value,
        mask=mask,
    )


def paged_gather_kv_from_topk_batch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: torch.Tensor,
    block_size: int,
    sink_size: int,
    local_size: int,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    output_bthd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused batched gather: assemble sink/local/topk then load K/V.

    Args:
        key_cache / value_cache: [num_blocks, block_size, Hkv, D]
        block_table: [B, max_blocks]
        seq_lens: [B] (CPU or CUDA)
        topk: [B, Hkv, final_topk]
    Returns:
        out_k / out_v: [B, Hkv, T, D], or [B, T, Hkv, D] when
            ``output_bthd`` is true.
    """
    batch, heads, topk_len = topk.shape
    head_dim = key_cache.shape[-1]
    n_ctx = int(sink_size) + int(local_size) + topk_len
    output_shape = (
        (batch, n_ctx, heads, head_dim)
        if output_bthd
        else (batch, heads, n_ctx, head_dim)
    )
    if (
        out_k is None
        or out_k.shape != output_shape
        or out_k.dtype != key_cache.dtype
        or out_k.device != key_cache.device
    ):
        out_k = torch.empty(
            output_shape,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
    if (
        out_v is None
        or out_v.shape != out_k.shape
        or out_v.dtype != out_k.dtype
        or out_v.device != out_k.device
    ):
        out_v = torch.empty_like(out_k)
    bt = block_table.contiguous()
    topk_c = topk.contiguous()
    # Keep the scheduler's native integer dtype to avoid an int32 -> int64
    # elementwise conversion in every layer of the decode hot path.
    seq_lens_dev = seq_lens.to(device=topk.device).contiguous()
    out_h_dim = 2 if output_bthd else 1
    out_t_dim = 1 if output_bthd else 2

    _paged_gather_from_topk_batch_kernel[(batch, heads, n_ctx)](
        key_cache,
        value_cache,
        bt,
        topk_c,
        seq_lens_dev,
        out_k,
        out_v,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        bt.stride(0),
        topk_c.stride(0),
        topk_c.stride(1),
        out_k.stride(0),
        out_k.stride(out_h_dim),
        out_k.stride(out_t_dim),
        out_k.stride(3),
        out_v.stride(0),
        out_v.stride(out_h_dim),
        out_v.stride(out_t_dim),
        out_v.stride(3),
        int(block_size),
        int(sink_size),
        int(local_size),
        topk_len,
        n_ctx,
        head_dim,
        int(bt.shape[1]),
        int(key_cache.shape[0]),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out_k, out_v


def paged_gather_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-head sparse K/V from paged cache in one CUDA launch."""
    heads, n_ctx = logical_ids.shape
    head_dim = key_cache.shape[-1]
    out_k = torch.empty(
        heads, n_ctx, head_dim, dtype=key_cache.dtype, device=key_cache.device
    )
    out_v = torch.empty_like(out_k)
    _paged_gather_kv_kernel[(heads, n_ctx)](
        key_cache,
        value_cache,
        block_table,
        logical_ids,
        out_k,
        out_v,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        logical_ids.stride(0),
        logical_ids.stride(1),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        int(block_size),
        n_ctx,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out_k, out_v


def paged_gather_kv_batch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_ids: torch.Tensor,
    block_size: int,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched paged gather.

    Args:
        key_cache / value_cache: [num_blocks, block_size, Hkv, D]
        block_table: [B, max_blocks]
        logical_ids: [B, Hkv, n_ctx]
    Returns:
        out_k / out_v: [B, Hkv, n_ctx, D]
    """
    batch, heads, n_ctx = logical_ids.shape
    head_dim = key_cache.shape[-1]
    if (
        out_k is None
        or out_k.shape != (batch, heads, n_ctx, head_dim)
        or out_k.dtype != key_cache.dtype
        or out_k.device != key_cache.device
    ):
        out_k = torch.empty(
            batch,
            heads,
            n_ctx,
            head_dim,
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
    if (
        out_v is None
        or out_v.shape != out_k.shape
        or out_v.dtype != out_k.dtype
        or out_v.device != out_k.device
    ):
        out_v = torch.empty_like(out_k)
    bt = block_table.contiguous()
    lids = logical_ids.contiguous()
    _paged_gather_kv_batch_kernel[(batch, heads, n_ctx)](
        key_cache,
        value_cache,
        bt,
        lids,
        out_k,
        out_v,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        bt.stride(0),
        lids.stride(0),
        lids.stride(1),
        lids.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_k.stride(3),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        out_v.stride(3),
        int(block_size),
        n_ctx,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out_k, out_v
