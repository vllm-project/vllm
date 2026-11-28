# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Put a chunk's own KV rows into the region the DCP KV prefetch gathers.

Every rank holds every rank's new `kv_c`/`k_pe` rows before the cache write
(`kv_ce_overlap`, "The chunk's own rows"). Two kernels place them without a
tensor operation per rank:

* `_stage_slots_kernel` gives each gathered row its slot in a scratch block
  cache, rank-major, so the cache-write kernel quantizes rank `p`'s rows, in
  token order, into the contiguous entries from `p * per_rank`. When
  `HAS_RESTORE` is False (no PCP reordering, e.g. TP + DCP), token `g` is row
  `g` directly.
* `_stage_rows_kernel` upconverts every rank's entries straight into that
  rank's rows `[old_p, new_p)` of the region, with the arithmetic of
  `cp_gather_and_upconvert_fp8_kv_cache`, driven by a small device table: one
  launch for all ranks.

`_plan_check_kernel` belongs to the plan those rows follow: every rank derives
the step's plan by itself and this kernel traps a rank whose peers derived
another.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.cuda_direct import DirectKernel, direct_launch

# Rows one block of each kernel handles.
STAGE_ROWS = 16
SLOT_BLOCK = 1024
# One raw fp8_ds_mla cache entry and the BF16 row it upconverts to.
FP8_ENTRY = 656
FP8_NOPE = 512
FP8_SCALE_OFFSET = 512
FP8_ROPE_OFFSET = 528
ROPE_DIM = 64
ROW_DIM = FP8_NOPE + ROPE_DIM

_ENTRY = tl.constexpr(FP8_ENTRY)
_NOPE = tl.constexpr(FP8_NOPE)
_SCALE_OFFSET = tl.constexpr(FP8_SCALE_OFFSET)
_ROPE_OFFSET = tl.constexpr(FP8_ROPE_OFFSET)
_ROPE_DIM = tl.constexpr(ROPE_DIM)
_ROW_DIM = tl.constexpr(ROW_DIM)


@triton.jit(
    do_not_specialize=[
        "context_start", "new_rows", "interleave", "dcp_size", "per_rank",
    ],
    do_not_specialize_on_alignment=["restore_ptr"],
)  # fmt: skip
def _stage_slots_kernel(
    restore_ptr,
    slots_ptr,
    context_start: tl.int64,
    new_rows: tl.int32,
    interleave: tl.int32,
    dcp_size: tl.int32,
    per_rank: tl.int32,
    BLOCK: tl.constexpr,
    HAS_RESTORE: tl.constexpr,
):
    """`slots[row] = owner * per_rank + shard_row - old[owner]` for token
    `context_start + g`, `g < new_rows`, where `row = restore[g]` when
    `HAS_RESTORE` is True and `row = g` otherwise. Rows no token maps to keep
    their value."""
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = g < new_rows
    token = context_start + g.to(tl.int64)
    owner = token // interleave % dcp_size
    shard_row = token // (interleave * dcp_size) * interleave + token % interleave
    base = context_start // interleave // dcp_size * interleave
    remainder = context_start - base * dcp_size
    old = base + tl.minimum(tl.maximum(remainder - owner * interleave, 0), interleave)
    if HAS_RESTORE:
        row = tl.load(restore_ptr + g, mask=live, other=0)
    else:
        row = g.to(tl.int64)
    tl.store(slots_ptr + row, owner * per_rank + shard_row - old, mask=live)


@triton.jit(do_not_specialize_on_alignment=["sources_ptr", "targets_ptr", "counts_ptr"])
def _stage_rows_kernel(
    cache_ptr,
    sources_ptr,
    targets_ptr,
    counts_ptr,
    region_ptr,
    ROWS: tl.constexpr,
):
    """Rank `p` (grid axis 1): upconvert the raw entries `sources[p] + i` of the
    contiguous entries at `cache` to rows `targets[p] + i` of `region`,
    `i < counts[p]`. Block `b` (axis 0) takes `i` in `[b * ROWS, (b + 1) * ROWS)`.

    The arithmetic is that of `cp_gather_and_upconvert_fp8_kv_cache`: fp8 to float
    (exact), times the float scale of its 128-element tile, rounded to BF16; the
    RoPE half is copied.
    """
    p = tl.program_id(1)
    count = tl.load(counts_ptr + p)
    first = tl.program_id(0) * ROWS
    if first >= count:
        return
    rows = first + tl.arange(0, ROWS)
    live = (rows < count)[:, None]
    entry = cache_ptr + (tl.load(sources_ptr + p) + rows).to(tl.int64)[:, None] * _ENTRY
    out = (tl.load(targets_ptr + p) + rows).to(tl.int64)[:, None] * _ROW_DIM
    cols = tl.arange(0, _NOPE)
    q = tl.load(entry + cols[None, :], mask=live, other=0)
    scale_at = entry + _SCALE_OFFSET + (cols[None, :] // 128) * 4
    scale = tl.load(scale_at.to(tl.pointer_type(tl.float32)), mask=live, other=0.0)
    nope = (q.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(region_ptr + out + cols[None, :], nope, mask=live)
    rcols = tl.arange(0, _ROPE_DIM)
    rope_at = entry + _ROPE_OFFSET + rcols[None, :] * 2
    rope = tl.load(rope_at.to(tl.pointer_type(tl.int16)), mask=live, other=0)
    tl.store(
        region_ptr.to(tl.pointer_type(tl.int16)) + out + _NOPE + rcols[None, :],
        rope,
        mask=live,
    )


def stage_slots_kernel(has_restore: bool = True) -> DirectKernel | None:
    """`_stage_slots_kernel` loaded for direct launches, or None."""
    i64 = torch.int64
    return direct_launch(
        _stage_slots_kernel,
        {"restore_ptr": i64, "slots_ptr": i64},
        BLOCK=SLOT_BLOCK,
        HAS_RESTORE=has_restore,
    )


def stage_rows_kernel() -> DirectKernel | None:
    """`_stage_rows_kernel` loaded for direct launches, or None."""
    i32 = torch.int32
    return direct_launch(
        _stage_rows_kernel,
        {
            "cache_ptr": torch.uint8,
            "sources_ptr": i32,
            "targets_ptr": i32,
            "counts_ptr": i32,
            "region_ptr": torch.bfloat16,
        },
        ROWS=STAGE_ROWS,
    )


@triton.jit(debug=True, do_not_specialize=["slot"])
def _plan_check_kernel(
    words_ptr,
    deltas_ptr,
    slot: tl.int32,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Trap unless every rank's plan word `slot` equals this rank's. Rank `p`'s
    words (int64) start `deltas[p]` bytes from `words_ptr`, this rank's own."""
    ranks = tl.arange(0, BLOCK)
    live = ranks < WORLD
    at = words_ptr + tl.load(deltas_ptr + ranks, mask=live, other=0) + slot * 8
    theirs = tl.load(at.to(tl.pointer_type(tl.int64)), mask=live, other=0)
    mine = tl.load((words_ptr + slot * 8).to(tl.pointer_type(tl.int64)))
    tl.device_assert(
        (theirs == mine) | (ranks >= WORLD),
        "DCP KV prefetch: the ranks planned a step differently",
    )


def plan_check_kernel(world: int) -> DirectKernel | None:
    """`_plan_check_kernel` loaded for direct launches, or None."""
    return direct_launch(
        _plan_check_kernel,
        {"words_ptr": torch.uint8, "deltas_ptr": torch.int64},
        WORLD=world,
        BLOCK=triton.next_power_of_2(world),
    )
