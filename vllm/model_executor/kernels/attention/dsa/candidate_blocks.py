# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_IDX_BITS = tl.constexpr(20)
_IDX_MASK = tl.constexpr((1 << 20) - 1)
_NEG_INF = tl.constexpr(-float("inf"))


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


@triton.jit
def _pack_key(score, index, valid):
    """Order-preserving int64 key: float order << _IDX_BITS | ~index.

    Flipping the sign bit (all bits when negative) maps IEEE order onto
    unsigned order. The index rides inverted in the low bits so ties resolve to
    the smaller index, matching torch.topk. NaN is forced above +inf because
    torch.topk ranks it highest, and _block_scores_kernel can produce it.
    Invalid lanes pack to 0, below every valid key.
    """
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    ordered = tl.where(score != score, 0xFFFFFFFF, ordered)
    tie = (_IDX_MASK - index).to(tl.int64)
    return tl.where(valid, (ordered.to(tl.int64) << _IDX_BITS) | tie, 0)


@triton.jit
def _scan_range_topk(scores, row, lo, hi, nblocks, K_PAD: tl.constexpr, TILE):
    """Running top-K of packed keys over blocks [lo, hi) of one row."""
    best = tl.zeros((K_PAD,), dtype=tl.int64)
    for start in range(lo, hi, TILE):
        cols = start + tl.arange(0, TILE)
        valid = cols < hi
        score = tl.load(scores + row * nblocks + cols, valid, other=_NEG_INF)
        # -inf marks padding; it must never take a slot.
        key = _pack_key(score, cols, valid & (score != _NEG_INF))
        best = tl.topk(tl.join(best, tl.topk(key, K_PAD)).reshape(2 * K_PAD), K_PAD)
    return best


@triton.jit
def _live_blocks(starts, ends, row, nblocks, BLOCK_SIZE, HAS_STARTS, ROW_REPEAT):
    """Blocks this row can populate, matching _block_scores_kernel's extent.

    The logits workspace is padded to the full width, so past a row's end every
    block scores -inf and can never win a slot. Bounding the scan here is what
    torch.topk cannot do: it has no per-row extent.
    """
    bound = row // ROW_REPEAT
    start = tl.load(starts + bound) if HAS_STARTS else 0
    end = tl.load(ends + bound)
    live = (end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
    return tl.minimum(tl.maximum(live, 1), nblocks).to(tl.int32)


@triton.jit
def _active_chunks(live, NUM_CHUNKS: tl.constexpr, TILE: tl.constexpr):
    """Chunks worth running: each should own a full tile, else short contexts
    pay NUM_CHUNKS tile-wide top-ks to look at almost nothing."""
    return tl.minimum(tl.maximum((live + TILE - 1) // TILE, 1), NUM_CHUNKS)


@triton.jit
def _write_candidates(
    best,
    out,
    row,
    stride_row,
    stride_col,
    OUT_K: tl.constexpr,
    K_PAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    slots = tl.arange(0, K_PAD)
    index = (_IDX_MASK - (best & _IDX_MASK)).to(tl.int32)
    tl.store(
        out + row * stride_row + slots * stride_col,
        tl.where(best != 0, index, -1),
        slots < OUT_K,
    )
    if TAIL > 0:
        # Fewer blocks than requested candidates: pad the unused slots.
        rest = K_PAD + tl.arange(0, TAIL)
        tl.store(out + row * stride_row + rest * stride_col, -1, rest < OUT_K)


@triton.jit(do_not_specialize=["nblocks"])
def _row_topk_kernel(
    scores,
    starts,
    ends,
    out,
    nblocks,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    OUT_K: tl.constexpr,
    K_PAD: tl.constexpr,
    TAIL: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hi = _live_blocks(starts, ends, row, nblocks, BLOCK_SIZE, HAS_STARTS, ROW_REPEAT)
    best = _scan_range_topk(scores, row, 0, hi, nblocks, K_PAD, TILE)
    _write_candidates(best, out, row, stride_row, stride_col, OUT_K, K_PAD, TAIL)


@triton.jit(do_not_specialize=["nblocks"])
def _row_topk_partial_kernel(
    scores,
    starts,
    ends,
    partial,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    K_PAD: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    live = _live_blocks(starts, ends, row, nblocks, BLOCK_SIZE, HAS_STARTS, ROW_REPEAT)
    # Split on device from this row's live extent: a host-side bound would cost
    # a sync, and the padded width says nothing about the context.
    active = _active_chunks(live, NUM_CHUNKS, TILE)
    if chunk < active:
        split = (live + active - 1) // active
        lo = chunk * split
        hi = tl.minimum(lo + split, live)
        best = _scan_range_topk(scores, row, lo, hi, nblocks, K_PAD, TILE)
        slots = tl.arange(0, K_PAD)
        tl.store(partial + row * (NUM_CHUNKS * K_PAD) + chunk * K_PAD + slots, best)


@triton.jit(do_not_specialize=["nblocks"])
def _row_topk_merge_kernel(
    partial,
    starts,
    ends,
    out,
    nblocks,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    OUT_K: tl.constexpr,
    K_PAD: tl.constexpr,
    TAIL: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    live = _live_blocks(starts, ends, row, nblocks, BLOCK_SIZE, HAS_STARTS, ROW_REPEAT)
    active = _active_chunks(live, NUM_CHUNKS, TILE)
    lanes = tl.arange(0, NUM_CHUNKS * K_PAD)
    # Chunks the partial pass skipped hold stale keys; mask them out.
    keys = tl.load(
        partial + row * (NUM_CHUNKS * K_PAD) + lanes,
        (lanes // K_PAD) < active,
        other=0,
    )
    _write_candidates(
        tl.topk(keys, K_PAD), out, row, stride_row, stride_col, OUT_K, K_PAD, TAIL
    )


# One wave of workgroups on MI355X; below this a program-per-row leaves most
# CUs idle and splitting the row pays for the extra merge.
_TARGET_PROGRAMS = 256
# The merge top-k spans NUM_CHUNKS*K_PAD lanes whether or not those chunks ran,
# so more chunks buy parallelism at a cost the low-row cases cannot absorb.
_MAX_CHUNKS = 8
_TILE = 512


def _rocm_select_topk(
    scores: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    nblocks: int,
    topk_blocks: int,
    out: torch.Tensor,
    row_repeat: int,
    block_size: int,
) -> None:
    """Fused block top-k for ROCm, replacing topk + _store_candidates_kernel.

    torch.topk runs a multi-pass radix select that sorts values it does not
    need -- the caller only consumes the *set* of block ids -- and scans the
    padding because it cannot see per-row extents.
    """
    rows = scores.shape[0]
    k_pad = triton.next_power_of_2(min(topk_blocks, nblocks))

    num_chunks = 1
    if rows < _TARGET_PROGRAMS:
        # Sized only to fill the machine; how many actually run is decided per
        # row on device from its live extent.
        want = min(
            triton.cdiv(_TARGET_PROGRAMS, max(rows, 1)),
            _MAX_CHUNKS,
            max(1, nblocks // _TILE),
        )
        num_chunks = max(1, triton.next_power_of_2(want + 1) // 2)

    common = dict(
        BLOCK_SIZE=block_size,
        HAS_STARTS=row_ks is not None,
        ROW_REPEAT=row_repeat,
        OUT_K=topk_blocks,
        K_PAD=k_pad,
        TAIL=triton.next_power_of_2(topk_blocks - k_pad) if topk_blocks > k_pad else 0,
        TILE=_TILE,
    )
    if num_chunks == 1:
        _row_topk_kernel[(rows,)](
            scores, row_ks, row_ke, out, nblocks, *out.stride(), **common
        )
        return

    partial = scores.new_empty((rows, num_chunks * k_pad), dtype=torch.int64)
    _row_topk_partial_kernel[(rows, num_chunks)](
        scores,
        row_ks,
        row_ke,
        partial,
        nblocks,
        BLOCK_SIZE=block_size,
        HAS_STARTS=row_ks is not None,
        ROW_REPEAT=row_repeat,
        NUM_CHUNKS=num_chunks,
        K_PAD=k_pad,
        TILE=_TILE,
    )
    _row_topk_merge_kernel[(rows,)](
        partial,
        row_ks,
        row_ke,
        out,
        nblocks,
        *out.stride(),
        NUM_CHUNKS=num_chunks,
        **common,
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    if current_platform.is_rocm():
        _rocm_select_topk(
            scores, row_ks, row_ke, nblocks, topk_blocks, out, row_repeat, block_size
        )
        return
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )
