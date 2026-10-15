# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered updates of compressed index keys."""

import numpy as np
import torch

from vllm.triton_utils import tl, triton

libdevice = tl.extra.cuda.libdevice


@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _update_decode_values(
    row,
    slot,
    dim,
    key,
    z,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    active = slot >= 0
    page = tl.where(active, slot // BLOCK_SIZE, PAGES + row)
    fresh = (slot % 8 == 0) | ~active
    last = tl.load(LastSlot + page)
    tl.device_assert(
        fresh | (last == slot - 1), "Step5 CSA tail missing for a partial region"
    )
    numerator = tl.load(Numerator + page * 256 + dim)
    denominator = tl.load(Denominator + page * 256 + dim)
    maximum = tl.load(Maximum + page)
    numerator = tl.where(fresh, 0, numerator)
    denominator = tl.where(fresh, 0, denominator)
    maximum = tl.where(fresh, -float("inf"), maximum)
    updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
    scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
    weight = libdevice.exp(z - updated_max)
    numerator = numerator * scale + weight * key
    denominator = denominator * scale + weight
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    region = tl.maximum(slot, 0) % BLOCK_SIZE // 8
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(Mean + (page * (BLOCK_SIZE // 8) + region) * 256 + dim, summary_fp8)
    tl.store(Numerator + page * 256 + dim, numerator)
    tl.store(Denominator + page * 256 + dim, denominator)
    tl.store(Maximum + page, updated_max)
    tl.store(LastSlot + page, tl.where(active & (slot % 8 != 7), slot, -1))


@triton.jit
def _update_decode(
    Key,
    Z,
    Slots,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    KEY_STRIDE: tl.constexpr,
    Z_STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.arange(0, 256)
    slot = tl.load(Slots + row)
    key = tl.load(Key + row * KEY_STRIDE + dim).to(tl.float32)
    z = tl.load(Z + row * Z_STRIDE + dim).to(tl.float32)
    _update_decode_values(
        row,
        slot,
        dim,
        key,
        z,
        Mean,
        Numerator,
        Denominator,
        Maximum,
        LastSlot,
        PAGES,
        BLOCK_SIZE,
    )


def update_decode(
    key, z, slots, mean, numerator, denominator, maximum, last_slot, pages, block_size
):
    if slots.numel():
        _update_decode[(slots.numel(),)](
            key,
            z,
            slots,
            mean,
            numerator,
            denominator,
            maximum,
            last_slot,
            key.stride(0),
            z.stride(0),
            pages,
            block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _accumulate_prefill(
    Key,
    Z,
    Metadata,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    State,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    region_row = tl.program_id(0)
    metadata = Metadata + region_row * 12
    region = tl.load(metadata)
    if region < 0:
        return
    first = tl.load(metadata + 1)
    page = region * 8 // BLOCK_SIZE
    fresh = first == 0
    last_slot = tl.load(LastSlot + page)
    # Checked on device (enable with TRITON_DEBUG=1): a host check would
    # synchronize every layer.
    tl.device_assert(
        fresh | (last_slot == region * 8 + first - 1),
        "Step5 CSA tail missing for a partial region",
    )
    dim = tl.arange(0, 256)
    numerator = tl.load(Numerator + page * 256 + dim)
    denominator = tl.load(Denominator + page * 256 + dim)
    maximum = tl.load(Maximum + page)
    numerator = tl.where(fresh, 0, numerator)
    denominator = tl.where(fresh, 0, denominator)
    maximum = tl.where(fresh, -float("inf"), maximum)
    for offset in range(8):
        row = tl.load(metadata + 4 + offset)
        valid = row >= 0
        key = tl.load(
            Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.float32)
        z = tl.load(
            Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.float32)
        updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
        scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
        weight = libdevice.exp(z - updated_max)
        updated_num = numerator * scale + weight * key
        updated_den = denominator * scale + weight
        numerator = tl.where(valid, updated_num, numerator)
        denominator = tl.where(valid, updated_den, denominator)
        maximum = tl.where(valid, updated_max, maximum)
    tl.store(State + region_row * 513 + dim, numerator)
    tl.store(State + region_row * 513 + 256 + dim, denominator)
    tl.store(State + region_row * 513 + 512, maximum)


@triton.jit
def _commit_prefill(
    Metadata,
    State,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    BLOCK_SIZE: tl.constexpr,
):
    region_row = tl.program_id(0)
    metadata = Metadata + region_row * 12
    region = tl.load(metadata)
    if region < 0:
        return
    final_slot = tl.load(metadata + 2)
    final_region = tl.load(metadata + 3) != 0
    page = region * 8 // BLOCK_SIZE
    dim = tl.arange(0, 256)
    numerator = tl.load(State + region_row * 513 + dim)
    denominator = tl.load(State + region_row * 513 + 256 + dim)
    maximum = tl.load(State + region_row * 513 + 512)
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(Mean + region * 256 + dim, summary_fp8)
    tl.store(Numerator + page * 256 + dim, numerator, mask=final_region)
    tl.store(Denominator + page * 256 + dim, denominator, mask=final_region)
    tl.store(Maximum + page, maximum, mask=final_region)
    tl.store(LastSlot + page, final_slot, mask=final_region)


@triton.jit
def _build_prefill_groups(
    Slots,
    Positions,
    Owners,
    QueryStart,
    Metadata,
    POSITION_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_WIDTH: tl.constexpr,
):
    # One program per token row. A group is the run of one request's rows in
    # an eight-token region; its first row writes the metadata, others -1.
    row = tl.program_id(0)
    output = Metadata + row.to(tl.int64) * 12
    owner = tl.load(Owners + row)
    start = tl.load(QueryStart + owner)
    end = tl.load(QueryStart + owner + 1)
    limit = end
    if PREFIX_WIDTH > 0:
        # Rows before the journaled last window; skipped requests keep none.
        limit = tl.maximum(start, end - PREFIX_WIDTH)
        limit = tl.where(tl.load(Slots + start) >= 0, limit, start)
    position = tl.load(Positions + row * POSITION_STRIDE)
    head = (row == start) | (position % 8 == 0)
    offset = tl.arange(0, 8)
    step = offset - position % 8
    rows = row + step
    in_group = head & (row < limit) & (step >= 0) & (rows < limit)
    slot = tl.load(Slots + rows, in_group, other=-1)
    valid = in_group & (slot >= 0)
    first = tl.min(tl.where(valid, offset, 8), 0)
    last = tl.max(tl.where(valid, offset, -1), 0)
    region = tl.max(tl.where(valid, slot // 8, -1), 0)
    # Final region of its page in this batch: the request's next region is
    # absent or starts a new page.
    next_row = row + 8 - position % 8
    next_region = (position // 8 + 1) * 8
    final = (next_row >= limit) | (next_region % BLOCK_SIZE == 0)
    tl.store(output, tl.where(first < 8, region, -1))
    tl.store(output + 1, first)
    tl.store(output + 2, tl.where(last != 7, region * 8 + last, -1))
    tl.store(output + 3, final.to(tl.int64))
    tl.store(output + 4 + offset, tl.where(valid, rows, -1).to(tl.int64))


def build_prefill_groups(
    slots, positions, owners, query_start_loc, block_size, prefix_width=0
) -> torch.Tensor:
    """Summary update metadata on device, without host slots or a sync.

    Returns ``[rows, 12]`` metadata; non-head rows have region -1. With
    ``prefix_width``, only rows before each request's last ``prefix_width``
    rows are grouped (speculative journal prefix).
    """
    rows = slots.shape[0]
    metadata = torch.empty((rows, 12), dtype=torch.int64, device=slots.device)
    if rows:
        _build_prefill_groups[(rows,)](
            slots,
            positions,
            owners,
            query_start_loc,
            metadata,
            positions.stride(0),
            block_size,
            prefix_width,
        )
    return metadata


def make_prefill_metadata_from_slots(
    slots: np.ndarray, block_size: int, device
) -> torch.Tensor | None:
    """Vectorized ``make_prefill_metadata`` for rows grouped by region.

    Rows with ``slot < 0`` are skipped. Regions keep first-appearance order and
    a page's final region is its last one in that order, as in the dict form.
    """
    rows = np.flatnonzero(slots >= 0)
    if not rows.size:
        return None
    slot = slots[rows].astype(np.int64)
    unique, first, inverse = np.unique(
        slot // 8, return_index=True, return_inverse=True
    )
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    grouped = np.full((unique.size, 8), -1, dtype=np.int64)
    grouped[rank[inverse], slot % 8] = rows
    regions = unique[order]
    valid = grouped >= 0
    first_offset = valid.argmax(1)
    last_offset = 7 - valid[:, ::-1].argmax(1)
    pages = regions * 8 // block_size
    _, reversed_index = np.unique(pages[::-1], return_index=True)
    final_region = np.zeros(regions.size, dtype=np.int64)
    final_region[regions.size - 1 - reversed_index] = 1
    metadata = np.column_stack(
        (
            regions,
            first_offset,
            np.where(last_offset != 7, regions * 8 + last_offset, -1),
            final_region,
            grouped,
        )
    )
    return torch.from_numpy(metadata).pin_memory().to(device, non_blocking=True)


def make_prefill_metadata(groups, block_size, device) -> torch.Tensor:
    regions = list(groups)
    last = {region * 8 // block_size: region for region in regions}
    metadata = []
    for region, rows in groups.items():
        offsets = [i for i, row in enumerate(rows) if row >= 0]
        final_slot = region * 8 + offsets[-1]
        metadata.append(
            [
                region,
                offsets[0],
                final_slot if offsets[-1] != 7 else -1,
                int(last[region * 8 // block_size] == region),
                *rows,
            ]
        )
    return (
        torch.tensor(metadata, dtype=torch.int64)
        .pin_memory()
        .to(device, non_blocking=True)
    )


def update_prefill(
    key, z, groups, mean, numerator, denominator, maximum, last_slot, block_size
):
    """Accumulate grouped rows; ``groups`` may be prebuilt metadata."""
    if isinstance(groups, torch.Tensor):
        metadata = groups
    elif groups:
        metadata = make_prefill_metadata(groups, block_size, key.device)
    else:
        return
    if not metadata.shape[0]:
        return
    state = torch.empty(
        (metadata.shape[0], 513), dtype=torch.float32, device=key.device
    )
    # Rows with region < 0 are not group starts and are skipped.
    _accumulate_prefill[(metadata.shape[0],)](
        key,
        z,
        metadata,
        numerator,
        denominator,
        maximum,
        last_slot,
        state,
        key.stride(0),
        key.stride(2),
        z.stride(0),
        z.stride(2),
        block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _commit_prefill[(metadata.shape[0],)](
        metadata,
        state,
        mean,
        numerator,
        denominator,
        maximum,
        last_slot,
        block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit
def _update_request_journal(
    Key,
    Z,
    Slots,
    QueryStart,
    SeqLens,
    RequestState,
    JournalValues,
    JournalLast,
    JournalTag,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    RESTORE: tl.constexpr,
    UPDATE: tl.constexpr,
):
    # Journal rows are [request state, position % WIDTH]: the page state before
    # that window row, tagged with its slot. A window has at most WIDTH rows,
    # so its positions never collide, and the next window restores the state
    # before its first row only if this request wrote it at that exact slot.
    request = tl.program_id(0)
    state = tl.load(RequestState + request)
    start = tl.load(QueryStart + request)
    end = tl.load(QueryStart + request + 1)
    if state >= 0 and start < end:
        first = tl.load(Slots + start)
        if first >= 0:
            seq = tl.load(SeqLens + request)
            dim = tl.arange(0, 256)
            if RESTORE:
                journal = state * WIDTH + (seq - (end - start)) % WIDTH
                if tl.load(JournalTag + journal) == first:
                    page = first // BLOCK_SIZE
                    values = JournalValues + journal * 513
                    tl.store(Numerator + page * 256 + dim, tl.load(values + dim))
                    tl.store(
                        Denominator + page * 256 + dim, tl.load(values + 256 + dim)
                    )
                    tl.store(Maximum + page, tl.load(values + 512))
                    tl.store(LastSlot + page, tl.load(JournalLast + journal))
                tl.debug_barrier()
            if UPDATE:
                for row in range(tl.maximum(start, end - WIDTH), end):
                    slot = tl.load(Slots + row)
                    if slot >= 0:
                        page = slot // BLOCK_SIZE
                        journal = state * WIDTH + (seq - (end - row)) % WIDTH
                        values = JournalValues + journal * 513
                        tl.store(values + dim, tl.load(Numerator + page * 256 + dim))
                        tl.store(
                            values + 256 + dim,
                            tl.load(Denominator + page * 256 + dim),
                        )
                        tl.store(values + 512, tl.load(Maximum + page))
                        tl.store(JournalLast + journal, tl.load(LastSlot + page))
                        tl.store(JournalTag + journal, slot)
                        key = tl.load(
                            Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE
                        ).to(tl.float32)
                        z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE).to(
                            tl.float32
                        )
                        _update_decode_values(
                            request,
                            slot,
                            dim,
                            key,
                            z,
                            Mean,
                            Numerator,
                            Denominator,
                            Maximum,
                            LastSlot,
                            PAGES,
                            BLOCK_SIZE,
                        )
                    tl.debug_barrier()


def update_request_journal(
    key,
    z,
    slots,
    query_start_loc,
    seq_lens,
    request_state,
    journal_values,
    journal_last,
    journal_tag,
    mean,
    numerator,
    denominator,
    maximum,
    last_slot,
    pages,
    block_size,
    width,
    *,
    restore=True,
    update=True,
):
    """Speculative summary update with device-side undo, graph capturable.

    Rows ``[max(start, end - width), end)`` of each request are journaled;
    earlier rows must be applied separately between a restore-only and an
    update-only call.
    """
    requests = request_state.shape[0]
    if requests:
        _update_request_journal[(requests,)](
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            request_state,
            journal_values,
            journal_last,
            journal_tag,
            mean,
            numerator,
            denominator,
            maximum,
            last_slot,
            key.stride(0),
            key.stride(2),
            z.stride(0),
            z.stride(2),
            pages,
            block_size,
            width,
            restore,
            update,
            num_warps=4,
            enable_fp_fusion=False,
        )
