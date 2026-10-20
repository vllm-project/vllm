# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA region summaries, compressed once when a region's eighth row arrives.

A region summary is an online softmax over the region's eight index-key rows,
accumulated in position order. Only closed regions are ever scored, so the raw
``[key | z]`` rows of each request's open region wait in a circular buffer
(``Step5CSATailRing``) and the summary is written once, by the row that closes
the region. A rejected draft that closed a region leaves a summary nobody
reads; it is rewritten when the region closes again.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

libdevice = tldevice


@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _accumulate(numerator, denominator, maximum, key, z):
    updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
    scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
    weight = libdevice.exp(z - updated_max)
    return (
        numerator * scale + weight * key,
        denominator * scale + weight,
        updated_max,
    )


@triton.jit
def _store_summary(
    Mean,
    slot,
    dim,
    numerator,
    denominator,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write the summary of the region holding KV ``slot``; blocks of ``Mean``
    are ``MEAN_BLOCK`` bytes apart."""
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    address = slot // BLOCK_SIZE * MEAN_BLOCK + slot % BLOCK_SIZE // 8 * 256
    tl.store(Mean + address + dim, summary_fp8)


@triton.jit
def _replay_ring(Ring, base, first, last, dim, RING_CAPACITY: tl.constexpr):
    """Accumulate the ring rows at positions ``[first, last)`` in order."""
    numerator = tl.zeros((256,), tl.float32)
    denominator = tl.zeros((256,), tl.float32)
    maximum = tl.max(tl.full((256,), -float("inf"), tl.float32), 0)
    for position in range(first, last):
        row = Ring + (base + position % RING_CAPACITY) * 512
        key = tl.load(row + dim).to(tl.float32)
        z = tl.load(row + 256 + dim).to(tl.float32)
        numerator, denominator, maximum = _accumulate(
            numerator, denominator, maximum, key, z
        )
    return numerator, denominator, maximum


@triton.jit
def _update_decode_values(
    slot,
    position,
    base,
    dim,
    key,
    z,
    Mean,
    Ring,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Store an open-region row, or summarize the region this row closes."""
    if (slot >= 0) & (base >= 0):
        if position % 8 == 7:
            numerator, denominator, maximum = _replay_ring(
                Ring, base, position - 7, position, dim, RING_CAPACITY
            )
            numerator, denominator, maximum = _accumulate(
                numerator, denominator, maximum, key, z
            )
            _store_summary(
                Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
            )
        else:
            row = Ring + (base + position % RING_CAPACITY) * 512
            tl.store(row + dim, key.to(tl.bfloat16))
            tl.store(row + 256 + dim, z.to(tl.bfloat16))


@triton.jit
def _update_decode(
    Key,
    Z,
    Slots,
    Positions,
    RingBase,
    Mean,
    Ring,
    KEY_STRIDE: tl.constexpr,
    Z_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.arange(0, 256)
    key = tl.load(Key + row * KEY_STRIDE + dim).to(tl.float32)
    z = tl.load(Z + row * Z_STRIDE + dim).to(tl.float32)
    _update_decode_values(
        tl.load(Slots + row),
        tl.load(Positions + row),
        tl.load(RingBase + row),
        dim,
        key,
        z,
        Mean,
        Ring,
        RING_CAPACITY,
        MEAN_BLOCK,
        BLOCK_SIZE,
    )


def summary_layout(mean: torch.Tensor) -> tuple[int, int]:
    """(bytes between blocks, tokens per block) of a ``[B, R, 256]`` cache."""
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    return mean.stride(0), mean.shape[1] * 8


def update_decode(key, z, slots, positions, ring_base, mean, ring, capacity):
    if slots.numel():
        _update_decode[(slots.numel(),)](
            key,
            z,
            slots,
            positions,
            ring_base,
            mean,
            ring,
            key.stride(0),
            z.stride(0),
            capacity,
            *summary_layout(mean),
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _update_verify(
    Key,
    Z,
    Slots,
    QueryStart,
    SeqLens,
    RingBase,
    Mean,
    Ring,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per request: its rows update the same open region in order.
    request = tl.program_id(0)
    start = tl.load(QueryStart + request)
    end = tl.load(QueryStart + request + 1)
    if start < end:
        base = tl.load(RingBase + start)
        seq = tl.load(SeqLens + request)
        dim = tl.arange(0, 256)
        for row in range(start, end):
            slot = tl.load(Slots + row)
            if (slot >= 0) & (base >= 0):
                position = seq - (end - row)
                ring_row = Ring + (base + position % RING_CAPACITY) * 512
                key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
                z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
                tl.store(ring_row + dim, key.to(tl.bfloat16))
                tl.store(ring_row + 256 + dim, z.to(tl.bfloat16))
                if position % 8 == 7:
                    tl.debug_barrier()
                    numerator, denominator, _ = _replay_ring(
                        Ring, base, position - 7, position + 1, dim, RING_CAPACITY
                    )
                    _store_summary(
                        Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
                    )
            tl.debug_barrier()


def update_verify(
    key, z, slots, query_start_loc, seq_lens, ring_base, mean, ring, capacity
):
    """Uniform MTP verify update; graph capturable, no host metadata."""
    requests = seq_lens.shape[0]
    if requests:
        _update_verify[(requests,)](
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            ring_base,
            mean,
            ring,
            key.stride(0),
            key.stride(2),
            z.stride(0),
            z.stride(2),
            capacity,
            *summary_layout(mean),
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _build_prefill_groups(
    Slots,
    Positions,
    Owners,
    QueryStart,
    Metadata,
    POSITION_STRIDE: tl.constexpr,
):
    # One program per token row. A group is the run of one request's rows in
    # an eight-token region; its first row writes the metadata, others -1.
    row = tl.program_id(0)
    output = Metadata + row.to(tl.int64) * 10
    owner = tl.load(Owners + row)
    start = tl.load(QueryStart + owner)
    end = tl.load(QueryStart + owner + 1)
    position = tl.load(Positions + row * POSITION_STRIDE)
    head = (row == start) | (position % 8 == 0)
    offset = tl.arange(0, 8)
    step = offset - position % 8
    rows = row + step
    in_group = head & (step >= 0) & (rows < end)
    slot = tl.load(Slots + rows, in_group, other=-1)
    valid = in_group & (slot >= 0)
    first = tl.min(tl.where(valid, offset, 8), 0)
    region = tl.max(tl.where(valid, slot // 8, -1), 0)
    tl.store(output, tl.where(first < 8, region, -1))
    tl.store(output + 1, first)
    tl.store(output + 2 + offset, tl.where(valid, rows, -1).to(tl.int64))


def build_prefill_groups(slots, positions, owners, query_start_loc) -> torch.Tensor:
    """Summary update metadata on device, without host slots or a sync.

    Returns ``[rows, 10]`` metadata ``[region, first offset, rows[8]]``;
    non-head rows have region -1.
    """
    rows = slots.shape[0]
    metadata = torch.empty((rows, 10), dtype=torch.int64, device=slots.device)
    if rows:
        _build_prefill_groups[(rows,)](
            slots,
            positions,
            owners,
            query_start_loc,
            metadata,
            positions.stride(0),
        )
    return metadata


@triton.jit
def _update_prefill(
    Key,
    Z,
    Metadata,
    Positions,
    RingBase,
    Mean,
    Ring,
    POSITION_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    metadata = Metadata + tl.program_id(0) * 10
    region = tl.load(metadata)
    # Only a group holding its region's eighth row closes the region.
    if (region >= 0) & (tl.load(metadata + 9) >= 0):
        first = tl.load(metadata + 1)
        head = tl.load(metadata + 2 + first)
        dim = tl.arange(0, 256)
        # Rows before ``first`` were stored by an earlier step.
        position = tl.load(Positions + head * POSITION_STRIDE)
        numerator, denominator, maximum = _replay_ring(
            Ring,
            tl.load(RingBase + head),
            position - first,
            position,
            dim,
            RING_CAPACITY,
        )
        for offset in range(8):
            row = tl.load(metadata + 2 + offset)
            valid = row >= 0
            key = tl.load(
                Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE, mask=valid, other=0
            ).to(tl.float32)
            z = tl.load(
                Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE, mask=valid, other=0
            ).to(tl.float32)
            updated = _accumulate(numerator, denominator, maximum, key, z)
            numerator = tl.where(valid, updated[0], numerator)
            denominator = tl.where(valid, updated[1], denominator)
            maximum = tl.where(valid, updated[2], maximum)
        _store_summary(
            Mean, region * 8, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
        )


@triton.jit
def _write_ring(
    Key,
    Z,
    RingSlots,
    Ring,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(RingSlots + row)
    if slot >= 0:
        dim = tl.arange(0, 256)
        key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
        z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
        tl.store(Ring + slot * 512 + dim, key.to(tl.bfloat16))
        tl.store(Ring + slot * 512 + 256 + dim, z.to(tl.bfloat16))


def update_prefill(
    key, z, metadata, positions, ring_base, ring_slots, mean, ring, capacity
):
    """Summarize the regions this step closes, then store open-region rows.

    Every ring read of the step precedes its writes, so a long chunk cannot
    overwrite open-region rows it still has to read.
    """
    rows = metadata.shape[0]
    if not rows:
        return
    strides = (key.stride(0), key.stride(2), z.stride(0), z.stride(2))
    _update_prefill[(rows,)](
        key,
        z,
        metadata,
        positions,
        ring_base,
        mean,
        ring,
        positions.stride(0),
        *strides,
        capacity,
        *summary_layout(mean),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _write_ring[(ring_slots.shape[0],)](key, z, ring_slots, ring, *strides, num_warps=4)
