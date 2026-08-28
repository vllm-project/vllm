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


def compact_completed_slots(
    slots: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side compaction with a fixed upper bound.

    ``count`` carries the number of valid entries, so callers do not need
    unused entries in ``out`` to be initialized to -1.
    """
    n = slots.numel()
    capacity = min(n, triton.cdiv(n, block_size) + 1024)
    if (
        out is None
        or out.shape[0] < capacity
        or out.dtype != slots.dtype
        or out.device != slots.device
    ):
        out = torch.empty((capacity,), dtype=slots.dtype, device=slots.device)
    else:
        out = out[:capacity]
    if count is None or count.device != slots.device or count.dtype != torch.int32:
        count = torch.empty((), dtype=torch.int32, device=slots.device)
    count.zero_()
    _compact_completed_slots_kernel[(triton.cdiv(n, 256),)](
        slots, out, count, n, block_size, BLOCK=256
    )
    return out, count


@triton.jit
def _finalize_block_summary_kernel(
    key_ptr,
    slots_ptr,
    slot_count_ptr,
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
    HAS_COUNT: tl.constexpr,
):
    slot_i = tl.program_id(0)
    h = tl.program_id(1)
    dim_block = tl.program_id(2)
    has_slot = slot_i >= 0
    if HAS_COUNT:
        has_slot = slot_i < tl.load(slot_count_ptr)
    slot = tl.load(slots_ptr + slot_i, mask=has_slot, other=-1)
    physical_block = slot // block_size
    offset = slot - physical_block * block_size
    complete = (
        has_slot
        & (slot >= 0)
        & (offset == block_size - 1)
        & (physical_block < num_blocks)
    )
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


@triton.jit
def _finalize_block_summary_small_kernel(
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
    HKV: tl.constexpr,
    D_BLOCKS: tl.constexpr,
):
    slot_i = tl.program_id(0)
    slot = tl.load(slots_ptr + slot_i)
    physical_block = slot // block_size
    offset = slot - physical_block * block_size
    complete = (slot >= 0) & (offset == block_size - 1) & (physical_block < num_blocks)
    physical_safe = tl.maximum(0, tl.minimum(physical_block, num_blocks - 1))
    toks = tl.arange(0, 16)[:, None]
    dims_local = tl.arange(0, 32)[None, :]
    shifts = (tl.arange(0, 8)[None, :] * 4).to(tl.int32)

    # Full CUDA Graph replays this node every decode step. Avoid executing all
    # masked reductions on the 15/16 steps that do not complete a child block.
    if complete:
        for h in tl.static_range(0, HKV):
            for dim_block in tl.static_range(0, D_BLOCKS):
                dims = dim_block * 32 + dims_local
                x = tl.load(
                    key_ptr
                    + physical_safe * stride_k_b
                    + toks * stride_k_t
                    + h * stride_k_h
                    + dims * stride_k_d,
                    mask=(toks < block_size) & (dims < head_dim),
                    other=0.0,
                ).to(tl.float32)
                mn = tl.min(x, axis=0)
                mx = tl.max(x, axis=0)
                centroid = tl.sum(x, axis=0) / block_size
                meta_offset = (
                    physical_safe * stride_meta_b
                    + h * stride_meta_h
                    + dims * stride_meta_d
                )
                dim_mask = dims < head_dim
                tl.store(chunk_min_ptr + meta_offset, mn, mask=dim_mask)
                tl.store(chunk_max_ptr + meta_offset, mx, mask=dim_mask)
                tl.store(centroid_ptr + meta_offset, centroid, mask=dim_mask)

                for pack_group in tl.static_range(0, 4):
                    pack_dims = (
                        dim_block * 32
                        + pack_group * 8
                        + tl.arange(0, 8)[None, :]
                    )
                    pack_x = tl.load(
                        key_ptr
                        + physical_safe * stride_k_b
                        + toks * stride_k_t
                        + h * stride_k_h
                        + pack_dims * stride_k_d,
                        mask=(toks < block_size) & (pack_dims < head_dim),
                        other=0.0,
                    ).to(tl.bfloat16)
                    pack_min = tl.min(pack_x, axis=0).to(tl.bfloat16)
                    pack_max = tl.max(pack_x, axis=0).to(tl.bfloat16)
                    scale = ((pack_max - pack_min) / 15.0).to(tl.bfloat16)
                    scale = tl.maximum(scale, 1.0e-8).to(tl.bfloat16)
                    ratio = (
                        (pack_x - pack_min[None, :]) / scale[None, :]
                    ).to(tl.bfloat16)
                    codes = tldevice.rint(
                        tl.maximum(0.0, tl.minimum(15.0, ratio))
                    ).to(tl.int32)
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
                        mask=((dim_block * 32 + pack_group * 8) < head_dim)
                        & (tl.arange(0, 16) < block_size),
                    )
        tl.store(valid_ptr + physical_safe, 1)


def finalize_completed_slots(
    key_cache: torch.Tensor,
    slots: torch.Tensor,
    block_summary,
    slot_count: torch.Tensor | None = None,
) -> None:
    """Finalize any slot whose block offset is block_size-1, asynchronously."""
    flat_slots = slots.reshape(-1)
    if flat_slots.numel() == 0:
        return
    if (
        slot_count is None
        and flat_slots.numel() <= 256
        and block_summary.num_kv_heads <= 8
        and block_summary.head_dim <= 256
    ):
        _finalize_block_summary_small_kernel[(flat_slots.numel(),)](
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
            HKV=block_summary.num_kv_heads,
            D_BLOCKS=triton.cdiv(block_summary.head_dim, 32),
            num_warps=4,
        )
        return
    grid = (
        flat_slots.numel(),
        block_summary.num_kv_heads,
        triton.cdiv(block_summary.head_dim, 32),
    )
    count_arg = slot_count if slot_count is not None else block_summary.valid
    _finalize_block_summary_kernel[grid](
        key_cache,
        flat_slots,
        count_arg,
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
        HAS_COUNT=slot_count is not None,
        num_warps=4,
    )


@triton.jit
def _finalize_parent_summary_kernel(
    block_table_ptr,
    seq_lens_ptr,
    parent_min_ptr,
    parent_max_ptr,
    parent_valid_ptr,
    parent_first_child_ptr,
    chunk_min_ptr,
    chunk_max_ptr,
    child_valid_ptr,
    start_block,
    factor,
    num_blocks,
    head_dim,
    batch,
    max_parents,
    scan_all,
    bt_stride_b,
    bt_stride_n,
    meta_stride_p,
    meta_stride_h,
    meta_stride_d,
    HKV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    parent = tl.program_id(0)
    b = tl.program_id(1)
    if b >= batch or parent >= max_parents:
        return

    seq_len = tl.load(seq_lens_ptr + b).to(tl.int32)
    block_size = 16
    last_completed = seq_len // block_size - 1
    rel_last = last_completed - start_block
    n_chunks = tl.maximum(0, rel_last + 1)
    n_parent = n_chunks // factor
    if parent >= n_parent:
        return

    if not scan_all:
        # Pure decode: only finalize the parent that just completed.
        if seq_len % block_size != 0:
            return
        if rel_last < 0 or (rel_last % factor) != (factor - 1):
            return
        target_parent = rel_last // factor
        if parent != target_parent:
            return

    child_begin = parent * factor
    child_end = child_begin + factor
    if child_end > n_chunks:
        return

    first_phys = -1
    anchor_phys = -1
    all_valid = True
    for child_off in tl.static_range(0, 16):
        child = child_begin + child_off
        if child >= n_chunks:
            all_valid = False
        else:
            phys = tl.load(
                block_table_ptr
                + b * bt_stride_b
                + (start_block + child) * bt_stride_n
            ).to(tl.int32)
            valid = (phys >= 0) & (phys < num_blocks) & tl.load(
                child_valid_ptr + phys
            )
            all_valid = all_valid & valid
            if child_off == 0:
                first_phys = phys
            if child_off == (factor - 1):
                anchor_phys = phys

    if not all_valid:
        return
    if anchor_phys < 0:
        return

    dims = tl.arange(0, BLOCK)
    for h in tl.static_range(0, HKV):
        dim_mask = dims < head_dim
        pmin = tl.full((BLOCK,), float("inf"), tl.float32)
        pmax = tl.full((BLOCK,), float("-inf"), tl.float32)
        for child_off in tl.static_range(0, 16):
            child = child_begin + child_off
            phys = tl.load(
                block_table_ptr
                + b * bt_stride_b
                + (start_block + child) * bt_stride_n
            ).to(tl.int32)
            base = phys * meta_stride_p + h * meta_stride_h
            cmin = tl.load(
                chunk_min_ptr + base + dims * meta_stride_d, mask=dim_mask, other=0.0
            ).to(tl.float32)
            cmax = tl.load(
                chunk_max_ptr + base + dims * meta_stride_d, mask=dim_mask, other=0.0
            ).to(tl.float32)
            pmin = tl.minimum(pmin, cmin)
            pmax = tl.maximum(pmax, cmax)
        out_base = anchor_phys * meta_stride_p + h * meta_stride_h
        tl.store(
            parent_min_ptr + out_base + dims * meta_stride_d,
            pmin,
            mask=dim_mask,
        )
        tl.store(
            parent_max_ptr + out_base + dims * meta_stride_d,
            pmax,
            mask=dim_mask,
        )
    tl.store(parent_valid_ptr + anchor_phys, 1)
    tl.store(parent_first_child_ptr + anchor_phys, first_phys)


def finalize_parent_summaries(
    block_table: torch.Tensor,
    block_summary,
    *,
    start_block: int,
    seq_lens: torch.Tensor,
    scan_all: bool,
    max_parents: int | None = None,
) -> None:
    """Aggregate 16 completed child summaries into anchor-indexed parent pools."""
    if block_table.numel() == 0 or seq_lens.numel() == 0:
        return
    batch = seq_lens.numel()
    factor = block_summary.blocks_per_parent
    if max_parents is None:
        available = max(0, block_table.shape[1] - start_block)
        max_parents = max(1, (available + factor - 1) // factor)
    if max_parents <= 0:
        return
    if not seq_lens.is_cuda:
        seq_lens = seq_lens.to(device=block_table.device, dtype=torch.int32)
    elif seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(dtype=torch.int32)
    bt = block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)
    hkv = block_summary.num_kv_heads
    if hkv > 8:
        raise ValueError(
            f"parent finalize Triton path supports up to 8 kv heads, got {hkv}"
        )
    block = 128 if block_summary.head_dim >= 128 else 64
    _finalize_parent_summary_kernel[(max_parents, batch)](
        bt,
        seq_lens,
        block_summary.parent_min,
        block_summary.parent_max,
        block_summary.parent_valid,
        block_summary.parent_first_child,
        block_summary.chunk_min,
        block_summary.chunk_max,
        block_summary.valid,
        start_block,
        factor,
        block_summary.num_blocks,
        block_summary.head_dim,
        batch,
        max_parents,
        scan_all,
        bt.stride(0),
        bt.stride(1),
        block_summary.chunk_min.stride(0),
        block_summary.chunk_min.stride(1),
        block_summary.chunk_min.stride(2),
        HKV=hkv,
        BLOCK=block,
        num_warps=4,
    )
