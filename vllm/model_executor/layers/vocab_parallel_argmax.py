# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused vocab-parallel argmax for ``LogitsProcessor.get_top_tokens``.

Each rank reduces its vocabulary shard to one ``(max, token id)`` pair per
4096-wide tile and per row, the pairs are all-gathered in rank-major order,
and a second kernel picks the global maximum. Ties resolve to the lowest token
id and NaN wins, like ``torch.argmax`` on the full logits. Compared with the
eager reduction this replaces, it skips the in-place padding mask, the index
arithmetic and the stack, and gathers along dim 0 so no transpose is needed.
"""

import torch

from vllm.triton_utils import tl, triton

_TILE = 4096
# Token ids travel as fp32 next to the maxima, which is exact below 2**24.
MAX_VOCAB_SIZE = 1 << 24


@triton.jit
def _argmax_combine(left_value, left_id, right_value, right_id):
    left_nan = left_value != left_value  # noqa: PLR0124
    right_nan = right_value != right_value  # noqa: PLR0124
    tied = (left_value == right_value) | (left_nan & right_nan)
    take_right = (
        (right_nan & ~left_nan)
        | (~left_nan & ~right_nan & (right_value > left_value))
        | (tied & (right_id < left_id))
    )
    return (
        tl.where(take_right, right_value, left_value),
        tl.where(take_right, right_id, left_id),
    )


@triton.jit
def _local_argmax_kernel(
    logits_ptr,
    logits_stride,
    partials_ptr,
    num_valid,
    vocab_start,
    num_tiles,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    local_ids = tile * TILE + tl.arange(0, TILE)
    valid = local_ids < num_valid
    values = tl.load(
        logits_ptr + row * logits_stride + local_ids,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    # Masked columns carry an id past every real one, so they lose all ties.
    ids = tl.where(valid, local_ids + vocab_start, 2**30)
    value, index = tl.reduce((values, ids), 0, _argmax_combine)
    dst = partials_ptr + (row * num_tiles + tile) * 2
    tl.store(dst, value)
    tl.store(dst + 1, index.to(tl.float32))


@triton.jit(do_not_specialize=["num_rows"])
def _merge_argmax_kernel(
    partials_ptr,
    out_ptr,
    num_rows,
    num_tiles,
    world_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    valid = slots < world_size * num_tiles
    rank = slots // num_tiles
    tile = slots % num_tiles
    offsets = ((rank * num_rows + row) * num_tiles + tile) * 2
    values = tl.load(partials_ptr + offsets, mask=valid, other=float("-inf"))
    ids = tl.load(partials_ptr + offsets + 1, mask=valid, other=2**30).to(tl.int32)
    _, index = tl.reduce((values, ids), 0, _argmax_combine)
    tl.store(out_ptr + row, index.to(tl.int64))


def local_argmax_partials(
    logits: torch.Tensor, num_valid: int, vocab_start: int
) -> torch.Tensor:
    """``[num_rows, num_tiles, 2]`` fp32 ``(max, global id)`` per tile.

    Only the first ``num_valid`` columns of ``logits`` are real tokens, with
    global ids starting at ``vocab_start``. The tile count follows the shard
    width, which is the same on every rank, so the partials can be gathered.
    """
    num_rows, width = logits.shape
    num_tiles = max(1, triton.cdiv(width, _TILE))
    partials = logits.new_empty((num_rows, num_tiles, 2), dtype=torch.float32)
    if num_rows > 0:
        _local_argmax_kernel[(num_rows, num_tiles)](
            logits,
            logits.stride(0),
            partials,
            num_valid,
            vocab_start,
            num_tiles,
            TILE=_TILE,
            num_warps=4,
        )
    return partials


def merge_argmax_partials(gathered: torch.Tensor, world_size: int) -> torch.Tensor:
    """Global argmax from rank-major ``local_argmax_partials`` outputs."""
    num_rows = gathered.shape[0] // world_size
    num_tiles = gathered.shape[1]
    out = gathered.new_empty(num_rows, dtype=torch.int64)
    if num_rows > 0:
        _merge_argmax_kernel[(num_rows,)](
            gathered,
            out,
            num_rows,
            num_tiles,
            world_size,
            BLOCK=triton.next_power_of_2(world_size * num_tiles),
            num_warps=4,
        )
    return out
