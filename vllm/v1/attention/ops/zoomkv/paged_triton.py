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
