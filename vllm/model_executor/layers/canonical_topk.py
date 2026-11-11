# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical form of a sparse-indexer top-k: a deterministic set in ascending order.

The radix top-k kernels (persistent, cooperative, filtered) take output slots by
`atomicAdd`, so the order of the selected indices depends on thread scheduling,
and so does the set when several scores tie with the k-th one: the kernels keep
whichever tied candidates reach the last slots first. `canonicalize_topk_` rewrites
a top-k result so that it depends only on the scores: every candidate scoring above
the k-th score, then the lowest-index candidates that tie with it, all in ascending
index order. Scores are compared with the kernels' own ordered key
(`convert_to_uint32_v2`), so -0.0/+0.0 and NaN rank exactly as in the selection.

Three kernels, no host synchronization: the k-th key per row, then per-block counts
of candidates above / tied with it, then each block writes its kept candidates at
their ascending positions.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK = 2048
# Largest key, above any score's: marks empty slots when taking the minimum.
_EMPTY_KEY = 2**33


def _ordered_key(scores: torch.Tensor) -> torch.Tensor:
    """The kernels' ordered FP32 key, as int64 (larger key = larger score)."""
    bits = scores.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 2**31, 0xFFFFFFFF - bits, bits + 2**31)


@triton.jit
def _key(x):
    bits = x.to(tl.float32).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(bits >= 2147483648, 4294967295 - bits, bits + 2147483648)


@triton.jit
def _threshold_kernel(
    logits_ptr,
    stride_row,
    idx_ptr,
    stride_idx,
    thresh_ptr,
    num_valid_ptr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    EMPTY: tl.constexpr,
):
    """Per row: the smallest key among the selected candidates, and their count."""
    row = tl.program_id(0)
    slots = tl.arange(0, K_POW2)
    idx = tl.load(idx_ptr + row * stride_idx + slots, mask=slots < K, other=-1)
    valid = idx >= 0
    x = tl.load(logits_ptr + row * stride_row + idx, mask=valid, other=0.0)
    key = tl.where(valid, _key(x), EMPTY)
    tl.store(thresh_ptr + row, tl.min(key, 0))
    tl.store(num_valid_ptr + row, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def _count_kernel(
    logits_ptr,
    stride_row,
    row_ends_ptr,
    thresh_ptr,
    counts_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    """Per (row, block): candidates above the threshold key and tied with it."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < tl.load(row_ends_ptr + row)
    x = tl.load(logits_ptr + row * stride_row + cols, mask=in_row, other=0.0)
    key = _key(x)
    t = tl.load(thresh_ptr + row)
    base = counts_ptr + (row * num_blocks + blk) * 2
    tl.store(base, tl.sum((in_row & (key > t)).to(tl.int32), 0))
    tl.store(base + 1, tl.sum((in_row & (key == t)).to(tl.int32), 0))


@triton.jit
def _write_kernel(
    logits_ptr,
    stride_row,
    row_ends_ptr,
    thresh_ptr,
    num_valid_ptr,
    counts_ptr,
    out_ptr,
    stride_out,
    num_blocks,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    NB_POW2: tl.constexpr,
):
    """Per (row, block): write the kept candidates at their ascending positions."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    num_valid = tl.load(num_valid_ptr + row)
    if blk == 0:
        # The kept candidates fill [0, num_valid); clear the rest.
        slots = tl.arange(0, K_POW2)
        tl.store(
            out_ptr + row * stride_out + slots,
            tl.full([K_POW2], -1, tl.int32),
            mask=(slots >= num_valid) & (slots < K),
        )
    # Counts of the whole row and of the blocks before this one.
    b = tl.arange(0, NB_POW2)
    in_range = b < num_blocks
    row_counts = counts_ptr + row * num_blocks * 2 + b * 2
    above_b = tl.load(row_counts, mask=in_range, other=0)
    tied_b = tl.load(row_counts + 1, mask=in_range, other=0)
    need = num_valid - tl.sum(above_b, 0)
    before = b < blk
    above_before = tl.sum(tl.where(before, above_b, 0), 0)
    tied_before = tl.sum(tl.where(before, tied_b, 0), 0)

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < tl.load(row_ends_ptr + row)
    x = tl.load(logits_ptr + row * stride_row + cols, mask=in_row, other=0.0)
    key = _key(x)
    t = tl.load(thresh_ptr + row)
    above = in_row & (key > t)
    tied = in_row & (key == t)
    tied_i = tied.to(tl.int32)
    tie_rank = tied_before + tl.cumsum(tied_i, 0) - tied_i
    keep = above | (tied & (tie_rank < need))
    keep_i = keep.to(tl.int32)
    pos = (
        above_before
        + tl.minimum(tl.maximum(need, 0), tied_before)
        + tl.cumsum(keep_i, 0)
        - keep_i
    )
    # Guard the output if `indices` was not an exact top-k of `logits`.
    tl.store(out_ptr + row * stride_out + pos, cols.to(tl.int32), mask=keep & (pos < K))


def canonicalize_topk_(
    logits: torch.Tensor, indices: torch.Tensor, row_ends: torch.Tensor
) -> None:
    """Rewrite `indices` in place into its canonical form (see module docstring).

    logits: [rows, cols] scores the top-k was taken over (fp32/bf16/fp16), with
        unit column stride; row r's candidates are columns [0, row_ends[r]).
    indices: [rows, k] int32 column indices selected per row, -1 for empty slots.
    row_ends: [rows] int32.

    Each row keeps the same number of valid entries; only which tied candidates
    are kept and the order change. Assumes `indices` is an exact top-k of
    `logits` (every selected score >= every unselected one).
    """
    rows, k = indices.shape
    if rows == 0 or k == 0 or logits.shape[1] == 0:
        return
    assert logits.stride(1) == 1 and indices.stride(1) == 1
    assert indices.dtype == torch.int32 and row_ends.dtype == torch.int32
    device = logits.device
    num_blocks = triton.cdiv(logits.shape[1], _BLOCK)
    thresh = torch.empty(rows, dtype=torch.int64, device=device)
    num_valid = torch.empty(rows, dtype=torch.int32, device=device)
    counts = torch.empty(rows, num_blocks, 2, dtype=torch.int32, device=device)
    k_pow2 = triton.next_power_of_2(k)
    _threshold_kernel[(rows,)](
        logits,
        logits.stride(0),
        indices,
        indices.stride(0),
        thresh,
        num_valid,
        K=k,
        K_POW2=k_pow2,
        EMPTY=_EMPTY_KEY,
    )
    grid = (rows, num_blocks)
    _count_kernel[grid](
        logits, logits.stride(0), row_ends, thresh, counts, num_blocks, BLOCK=_BLOCK
    )
    _write_kernel[grid](
        logits,
        logits.stride(0),
        row_ends,
        thresh,
        num_valid,
        counts,
        indices,
        indices.stride(0),
        num_blocks,
        K=k,
        K_POW2=k_pow2,
        BLOCK=_BLOCK,
        NB_POW2=triton.next_power_of_2(num_blocks),
    )
