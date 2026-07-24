# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch block-summary finalization for decode block boundaries."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _compact_completed_slots_kernel(
    slots_ptr, out_ptr, count_ptr, n, block_size, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots_ptr + i, mask=i < n, other=-1)
    complete = (i < n) & (slot >= 0) & ((slot % block_size) == block_size - 1)
    count_ptrs = count_ptr + tl.zeros((BLOCK,), tl.int32)
    pos = tl.atomic_add(count_ptrs, 1, mask=complete)
    tl.store(out_ptr + pos, slot, mask=complete)


def compact_completed_slots(slots: torch.Tensor, block_size: int) -> torch.Tensor:
    """Device-side compaction with a fixed upper bound; unused entries stay -1."""
    n = slots.numel()
    capacity = min(n, triton.cdiv(n, block_size) + 1024)
    out = torch.full((capacity,), -1, dtype=slots.dtype, device=slots.device)
    count = torch.zeros((), dtype=torch.int32, device=slots.device)
    _compact_completed_slots_kernel[(triton.cdiv(n, 256),)](
        slots, out, count, n, block_size, BLOCK=256
    )
    return out


@triton.jit
def _finalize_block_summary_kernel(
    key_ptr,
    slots_ptr,
    chunk_min_ptr,
    chunk_max_ptr,
    centroid_ptr,
    packed_ptr,
    valid_ptr,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_meta_b,
    stride_meta_h,
    stride_meta_d,
    stride_p_b,
    stride_p_h,
    stride_p_pack,
    stride_p_t,
    block_size,
    head_dim,
    num_blocks,
):
    slot_i = tl.program_id(0)
    h = tl.program_id(1)
    dim_block = tl.program_id(2)
    slot = tl.load(slots_ptr + slot_i)
    physical_block = slot // block_size
    offset = slot - physical_block * block_size
    complete = (slot >= 0) & (offset == block_size - 1) & (physical_block < num_blocks)
    physical_safe = tl.maximum(0, tl.minimum(physical_block, num_blocks - 1))
    toks = tl.arange(0, 16)[:, None]
    dims_local = tl.arange(0, 32)[None, :]
    dims = dim_block * 32 + dims_local
    x = tl.load(
        key_ptr
        + physical_safe * stride_k_b
        + toks * stride_k_t
        + h * stride_k_h
        + dims * stride_k_d,
        mask=complete & (toks < block_size) & (dims < head_dim),
        other=0.0,
    ).to(tl.float32)
    mn = tl.min(x, axis=0)
    mx = tl.max(x, axis=0)
    centroid = tl.sum(x, axis=0) / block_size
    meta_offset = (
        physical_safe * stride_meta_b + h * stride_meta_h + dims * stride_meta_d
    )
    dim_mask = complete & (dims < head_dim)
    tl.store(chunk_min_ptr + meta_offset, mn, mask=dim_mask)
    tl.store(chunk_max_ptr + meta_offset, mx, mask=dim_mask)
    tl.store(centroid_ptr + meta_offset, centroid, mask=dim_mask)

    # Match the reference's bf16 quantization arithmetic exactly.
    shifts = (tl.arange(0, 8)[None, :] * 4).to(tl.int32)
    for pack_group in tl.static_range(0, 4):
        pack_dims = dim_block * 32 + pack_group * 8 + tl.arange(0, 8)[None, :]
        pack_x = tl.load(
            key_ptr
            + physical_safe * stride_k_b
            + toks * stride_k_t
            + h * stride_k_h
            + pack_dims * stride_k_d,
            mask=complete & (toks < block_size) & (pack_dims < head_dim),
            other=0.0,
        ).to(tl.bfloat16)
        pack_min = tl.min(pack_x, axis=0).to(tl.bfloat16)
        pack_max = tl.max(pack_x, axis=0).to(tl.bfloat16)
        scale = ((pack_max - pack_min) / 15.0).to(tl.bfloat16)
        scale = tl.maximum(scale, 1.0e-8).to(tl.bfloat16)
        ratio = ((pack_x - pack_min[None, :]) / scale[None, :]).to(tl.bfloat16)
        codes = tldevice.rint(tl.maximum(0.0, tl.minimum(15.0, ratio))).to(tl.int32)
        packed = tl.sum(codes << shifts, axis=1)
        packed_offset = (
            physical_safe * stride_p_b
            + h * stride_p_h
            + (dim_block * 4 + pack_group) * stride_p_pack
            + tl.arange(0, 16) * stride_p_t
        )
        tl.store(
            packed_ptr + packed_offset,
            packed,
            mask=complete
            & ((dim_block * 32 + pack_group * 8) < head_dim)
            & (tl.arange(0, 16) < block_size),
        )
    # Benign same-value stores from all head/dim-block programs.
    tl.store(valid_ptr + physical_safe, 1, mask=complete)


def finalize_completed_slots(
    key_cache: torch.Tensor,
    slots: torch.Tensor,
    block_summary,
) -> None:
    """Finalize any slot whose block offset is block_size-1, asynchronously."""
    flat_slots = slots.reshape(-1)
    grid = (
        flat_slots.numel(),
        block_summary.num_kv_heads,
        triton.cdiv(block_summary.head_dim, 32),
    )
    _finalize_block_summary_kernel[grid](
        key_cache,
        flat_slots,
        block_summary.chunk_min,
        block_summary.chunk_max,
        block_summary.centroid,
        block_summary.packed,
        block_summary.valid,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        block_summary.chunk_min.stride(0),
        block_summary.chunk_min.stride(1),
        block_summary.chunk_min.stride(2),
        block_summary.packed.stride(0),
        block_summary.packed.stride(1),
        block_summary.packed.stride(2),
        block_summary.packed.stride(3),
        block_summary.block_size,
        block_summary.head_dim,
        block_summary.num_blocks,
        num_warps=4,
    )
