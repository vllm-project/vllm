# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive a compact head: upconvert its copied rows and complete a predicted pull.

A compact head copies, with the copy engines, the first rows of every peer's packed
buffer into the staging and upconverts them into the peer's slot of the BF16 region.
The copy engines need a size the host knows, but how many rows a rank must take
from a peer (the first `U` rows of the peer's packed buffer) is computed on the
device. Instead of reading `U` back, the host copies a predicted number of rows and
enqueues, in stream order after the copies, one launch of the kernel below for all
peers. It reads the exact counts on the device and upconverts every peer's rows
`[0, U)` into the region with the arithmetic of
`cp_gather_and_upconvert_fp8_kv_cache`: rows the copy engines brought from the
staging, and the rows `[predicted, U)` of a peer whose prediction was short straight
from the peer-mapped buffer the copy engines read, which it also stores into the
staging a copy would have filled.

The host therefore never needs `U`: no query, no wait, and the same launch whether
the prediction was enough or not. Copied rows past `U` are never addressed and are
not upconverted. The grid follows the rows the host copied; each program walks
its rows in strides, so a prediction that falls short by any amount still
completes. One launch serves every peer, where an upconvert per peer took one
launch each, and it moves every row with 16-byte accesses.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.cuda_direct import (
    DirectKernel,
    KernelLaunch,
    direct_launch,
)

# Sources one launch serves (the peers of a rank), rows a program moves at once,
# warps per program, and programs per source at most.
MAX_SOURCES = 7
RECEIVE_ROWS = 8
RECEIVE_WARPS = 2
MAX_RECEIVE_BLOCKS = 16384
# One raw fp8_ds_mla cache entry and the BF16 row it upconverts to.
FP8_ENTRY = 656
FP8_NOPE = 512
FP8_SCALE_OFFSET = 512
FP8_ROPE_OFFSET = 528
ROPE_DIM = 64
ROW_DIM = FP8_NOPE + ROPE_DIM
ENTRY_WORDS = FP8_ENTRY // 4
# The 128-element tiles of the fp8 part, one float scale each.
SCALE_TILES = 4

_ENTRY = tl.constexpr(FP8_ENTRY)
_NOPE = tl.constexpr(FP8_NOPE)
_SCALE_OFFSET = tl.constexpr(FP8_SCALE_OFFSET)
_ROPE_OFFSET = tl.constexpr(FP8_ROPE_OFFSET)
_ROPE_DIM = tl.constexpr(ROPE_DIM)
_ROW_DIM = tl.constexpr(ROW_DIM)
_WORDS = tl.constexpr(ENTRY_WORDS)
_WORDS_BLOCK = tl.constexpr(triton.next_power_of_2(ENTRY_WORDS))
_TILES = tl.constexpr(SCALE_TILES)
_TILE = tl.constexpr(FP8_NOPE // SCALE_TILES)


@triton.jit
def _upconvert_rows(entry, rows, live, out_row, region_ptr):
    """Upconvert the raw entries at `entry` (one address per row) to rows
    `out_row + rows` of `region`: fp8 to float (exact), times the float scale of
    its 128-element tile, rounded to BF16; the RoPE half is copied."""
    tiles = tl.arange(0, _TILES)
    cols = tiles[:, None] * _TILE + tl.arange(0, _TILE)[None, :]
    m3 = live[:, None, None]
    q = tl.load(entry[:, None, None] + cols[None, :, :], mask=m3, other=0)
    scale_at = entry[:, None, None] + _SCALE_OFFSET + tiles[None, :, None] * 4
    scale = tl.load(scale_at.to(tl.pointer_type(tl.float32)), mask=m3, other=0.0)
    nope = (q.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale).to(tl.bfloat16)
    out = (out_row + rows.to(tl.int64)) * _ROW_DIM
    tl.store(region_ptr + out[:, None, None] + cols[None, :, :], nope, mask=m3)
    m2 = live[:, None]
    rcols = tl.arange(0, _ROPE_DIM)
    rope_at = entry[:, None] + _ROPE_OFFSET + rcols[None, :] * 2
    rope = tl.load(rope_at.to(tl.pointer_type(tl.int16)), mask=m2, other=0)
    tl.store(
        region_ptr.to(tl.pointer_type(tl.int16))
        + out[:, None]
        + _NOPE
        + rcols[None, :],
        rope,
        mask=m2,
    )


@triton.jit(
    do_not_specialize=[
        "b0", "b1", "b2", "b3", "b4", "b5", "b6",
        "limit", "stage_rows", "shard_rows", "stride",
    ]
)  # fmt: skip
def _receive_kernel(
    anchor_ptr,
    deltas_ptr,
    ends_ptr,
    slots_ptr,
    parts_ptr,
    region_ptr,
    b0: tl.int32,
    b1: tl.int32,
    b2: tl.int32,
    b3: tl.int32,
    b4: tl.int32,
    b5: tl.int32,
    b6: tl.int32,
    limit: tl.int32,
    stage_rows: tl.int32,
    shard_rows: tl.int32,
    stride: tl.int32,
    ROWS: tl.constexpr,
):
    """Source `i` (grid axis 1) has `count = min(ends[s + 1] - ends[s], limit)`
    rows, `s = slots[i]`; its first `b_i` rows were copied to rows
    `parts[i] * stage_rows + row` of the staging, the raw entries at `anchor`.
    Upconvert rows `[0, count)` to rows `s * shard_rows + row` of `region`: rows
    below `b_i` from the staging, the others from the buffer at
    `anchor + deltas[i]` bytes, which are also stored to the staging. Program `b`
    (axis 0) takes rows `b * ROWS + k * stride + [0, ROWS)` of either range.

    Each range has a base address of its own, so the compiler knows every row
    is 16-byte aligned and loads it with wide accesses."""
    i = tl.program_id(1)
    slot = tl.load(slots_ptr + i).to(tl.int64)
    count = tl.minimum(tl.load(ends_ptr + slot + 1) - tl.load(ends_ptr + slot), limit)
    first = tl.program_id(0) * ROWS
    if first >= count:
        return
    copied = (
        b0 * (i == 0).to(tl.int32)
        + b1 * (i == 1).to(tl.int32)
        + b2 * (i == 2).to(tl.int32)
        + b3 * (i == 3).to(tl.int32)
        + b4 * (i == 4).to(tl.int32)
        + b5 * (i == 5).to(tl.int32)
        + b6 * (i == 6).to(tl.int32)
    )
    staged = anchor_ptr + tl.load(parts_ptr + i).to(tl.int64) * stage_rows * _ENTRY
    peer = anchor_ptr + tl.multiple_of(tl.load(deltas_ptr + i), 16)
    out_row = slot * shard_rows
    # The rows the copy engines brought, from the staging.
    staged_rows = tl.minimum(copied, count)
    for r in range(0, tl.cdiv(staged_rows - first, stride)):
        rows = first + r * stride + tl.arange(0, ROWS)
        live = rows < staged_rows
        _upconvert_rows(
            staged + rows.to(tl.int64) * _ENTRY, rows, live, out_row, region_ptr
        )
    # The rows a short prediction missed, from the peer; into the staging too.
    wcols = tl.arange(0, _WORDS_BLOCK)
    for r in range(0, tl.cdiv(count - copied - first, stride)):
        rows = copied + first + r * stride + tl.arange(0, ROWS)
        live = rows < count
        offset = rows.to(tl.int64) * _ENTRY
        mw = live[:, None] & (wcols[None, :] < _WORDS)
        word_at = (peer + offset)[:, None] + wcols[None, :] * 4
        words = tl.load(word_at.to(tl.pointer_type(tl.int32)), mask=mw, other=0)
        stage_at = (staged + offset)[:, None] + wcols[None, :] * 4
        tl.store(stage_at.to(tl.pointer_type(tl.int32)), words, mask=mw)
        _upconvert_rows(peer + offset, rows, live, out_row, region_ptr)


def receive_kernel() -> DirectKernel | None:
    """`_receive_kernel` loaded for direct launches (`launch_receive`), or None."""
    i32 = torch.int32
    return direct_launch(
        _receive_kernel,
        {
            "anchor_ptr": torch.uint8,
            "deltas_ptr": torch.int64,
            "ends_ptr": i32,
            "slots_ptr": i32,
            "parts_ptr": i32,
            "region_ptr": torch.bfloat16,
        },
        ROWS=RECEIVE_ROWS,
        num_warps=RECEIVE_WARPS,
    )


def receive_blocks(copied: Sequence[int], limit: int) -> int:
    """Blocks per source of a launch: enough for the rows the host copied (at
    least one), at most `MAX_RECEIVE_BLOCKS`; rows past them are walked in
    strides."""
    rows = max(min(int(n), limit) for n in copied)
    return max(1, min(MAX_RECEIVE_BLOCKS, triton.cdiv(rows, RECEIVE_ROWS)))


def launch_receive(
    launch: KernelLaunch,
    stream: int,
    copied: Sequence[int],
    limit: int,
    region_ptr: int,
    shard_rows: int,
) -> None:
    """Enqueue, on `stream`, the upconvert of a compact head's rows: one driver
    call.

    `launch` holds the kernel's buffers: source `i` holds raw `fp8_ds_mla` entries
    and starts `deltas[i]` bytes from `anchor_ptr` (`peer_deltas`); its first
    `copied[i]` rows were copied to rows `parts[i] * stage_rows + row` of the
    staging at `anchor_ptr` (host-known). The kernel upconverts its rows
    `[0, min(ends[s + 1] - ends[s], limit))`, `s = slots[i]`, to rows
    `s * shard_rows + row` of the BF16 region at `region_ptr`: rows below
    `copied[i]` from the staging, the others from the source, which it also stores
    to the staging. `ends` (int32, on the device) is read when the kernel runs, so
    it may be written by work enqueued earlier on the stream; nothing is read on
    the host.
    """
    slots, first = launch.slots, launch.index["b0"]
    for offset, rows in enumerate(copied):
        slots[first + offset] = rows
    blocks = receive_blocks(copied, limit)
    launch.set("limit", limit)
    launch.set("region_ptr", region_ptr)
    launch.set("shard_rows", shard_rows)
    launch.set("stride", blocks * RECEIVE_ROWS)
    launch(stream, blocks, len(copied))


def peer_deltas(anchor: torch.Tensor, sources: Sequence[torch.Tensor]) -> torch.Tensor:
    """Where each source buffer starts, in bytes from `anchor`'s first element: the
    `deltas` of `launch_receive` (int64, on `anchor`'s device). The sources are
    in this process (peer-mapped symmetric memory), and the values hold while
    neither they nor `anchor` are reallocated."""
    base = anchor.data_ptr()
    return torch.tensor(
        [source.data_ptr() - base for source in sources],
        dtype=torch.int64,
        device=anchor.device,
    )
