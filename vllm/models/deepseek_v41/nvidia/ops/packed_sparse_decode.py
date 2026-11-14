# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack a request's speculative positions into one FlashMLA head tile.

On SM90 the sparse decode kernel tiles query heads by 64, so a TP8 rank with
8 real heads wastes 7/8 of every MMA. The 1 + K positions of a speculative
decode request attend nearly the same keys, so their heads can share a tile:
slot ``s`` of the tile holds position ``s``'s heads, the key list becomes the
union of the positions' keys, and a per-key bitmask restores each position's
own key set.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _merge_slot_indices_kernel(
    out_idx_ptr,
    out_mask_ptr,
    out_len_ptr,
    idx_ptr,
    idx_stride,
    len_ptr,
    out_width,
    WIDTH: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    first = req * SLOTS

    # Gather every slot's valid keys with its slot bit, sort by key, then keep
    # the first entry of each run with the run's OR'd bits.
    offs = tl.arange(0, BLOCK)
    slot = offs // WIDTH
    col = offs % WIDTH
    in_slot = slot < SLOTS
    slot_len = tl.load(len_ptr + first + slot, mask=in_slot, other=0)
    live = in_slot & (col < slot_len)
    key = tl.load(
        idx_ptr + (first + slot).to(tl.int64) * idx_stride + col,
        mask=live,
        other=-1,
    )
    live = live & (key >= 0)
    # Dead entries sort last; the low 8 bits carry the slot bit.
    big = tl.where(live, key.to(tl.int64), 0x3FFFFFFF)
    packed = (big << 8) | tl.where(live, 1 << slot, 0).to(tl.int64)
    packed = tl.sort(packed)
    keys = packed >> 8
    bits = (packed & 0xFF).to(tl.int32)

    # OR the bits within each run of equal keys (runs are at most SLOTS).
    acc = bits
    for d in tl.static_range(1, SLOTS):
        nxt = tl.gather(packed, tl.minimum(offs + d, BLOCK - 1), 0)
        same = (nxt >> 8) == keys
        same = same & (offs + d < BLOCK)
        acc = acc | tl.where(same, (nxt & 0xFF).to(tl.int32), 0)
    prev = tl.gather(packed, tl.maximum(offs - 1, 0), 0)
    head = (offs == 0) | ((prev >> 8) != keys)
    valid = head & (keys != 0x3FFFFFFF)

    pos = tl.cumsum(valid.to(tl.int32), 0) - 1
    write = valid & (pos < out_width)
    base = req.to(tl.int64) * out_width
    tl.store(out_idx_ptr + base + pos, keys.to(tl.int32), mask=write)
    tl.store(out_mask_ptr + base + pos, acc.to(tl.uint8), mask=write)
    n = tl.minimum(tl.sum(valid.to(tl.int32), 0), out_width)
    tl.store(out_len_ptr + req, n)
    # Clear the tail so stale keys past n never leak into the kernel.
    for start in range(0, out_width, BLOCK):
        tail = start + offs
        clear = (tail >= n) & (tail < out_width)
        tl.store(out_idx_ptr + base + tail, -1, mask=clear)
        tl.store(out_mask_ptr + base + tail, 0, mask=clear)


def merge_slot_indices(
    indices: torch.Tensor, lens: torch.Tensor, slots: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union the key lists of each request's slots.

    Args:
        indices: ``[num_reqs * slots, width]`` int32 global KV slots, ``-1``
            invalid; request ``r`` owns rows ``r * slots`` onward.
        lens: ``[num_reqs * slots]`` int32, valid prefix length of each row.
        slots: query rows per request.

    Returns:
        ``(indices [num_reqs, 1, W], mask [num_reqs, 1, W], lens [num_reqs])``
        where ``W = slots * width`` rounded up to 64 and bit ``s`` of
        ``mask`` says whether slot ``s`` attends that key.

    """
    num_reqs = indices.shape[0] // slots
    width = indices.shape[-1]
    out_width = triton.cdiv(slots * width, 64) * 64
    out_idx = torch.empty(
        (num_reqs, 1, out_width), dtype=torch.int32, device=indices.device
    )
    out_mask = torch.empty(
        (num_reqs, 1, out_width), dtype=torch.uint8, device=indices.device
    )
    out_len = torch.empty(num_reqs, dtype=torch.int32, device=indices.device)
    if num_reqs > 0:
        _merge_slot_indices_kernel[(num_reqs,)](
            out_idx,
            out_mask,
            out_len,
            indices,
            indices.stride(0),
            lens,
            out_width,
            WIDTH=width,
            SLOTS=slots,
            BLOCK=triton.next_power_of_2(slots * width),
            num_warps=8,
        )
    return out_idx, out_mask, out_len


def pack_slot_rows(
    x: torch.Tensor, num_reqs: int, slots: int, rows: int, tile_rows: int
) -> torch.Tensor:
    """``[num_reqs * slots, H_pad, D]`` -> ``[num_reqs, 1, tile_rows, D]``.

    Slot ``s`` of request ``r`` takes the first ``rows`` heads of token
    ``r * slots + s``. Tile rows past ``slots * rows`` are left uninitialized:
    no mask bit selects them, so the kernel never lets them attend a key.
    """
    d = x.shape[-1]
    packed = x.new_empty(num_reqs, 1, tile_rows, d)
    packed.view(num_reqs, tile_rows // rows, rows, d)[:, :slots].copy_(
        x[: num_reqs * slots].view(num_reqs, slots, x.shape[1], d)[:, :, :rows]
    )
    return packed


def unpack_slot_rows(
    packed: torch.Tensor, out: torch.Tensor, slots: int, rows: int
) -> None:
    """Inverse of ``pack_slot_rows`` into ``out[:, :rows]``."""
    num_reqs, d = packed.shape[0], packed.shape[-1]
    out[: num_reqs * slots].view(num_reqs, slots, out.shape[1], d)[:, :, :rows].copy_(
        packed.view(num_reqs, -1, rows, d)[:, :slots]
    )
