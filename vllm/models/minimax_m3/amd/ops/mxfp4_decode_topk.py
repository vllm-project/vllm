# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton decode top-k for the gfx942 MXFP4 indexer, bit-compatible with AITER.

``aiter.ops.msa_block_select.pa_sparse_block_topk`` runs one workgroup per
(row, index head), so at small batch a single workgroup scans the whole row and
its radix narrowing is latency bound. This
selector splits a row into chunks across programs (bitonic ``tl.topk`` per
tile, last-arriving program merges), the scheme the fp8 indexer's
``_decode_topk_fused_kernel`` uses, but reproduces AITER's selection exactly:

* key = order-preserving u32 of the fp32 score, then the block index, so equal
  scores rank the *larger* block first (AITER packs ``ordered << 32 | idx``);
* NaN ranks as -inf, ``-0.0`` stays below ``+0.0`` (no canonicalisation);
* winners come out in descending key order (AITER's rank order), rows shorter
  than ``topk`` are padded with -1;
* the sparse block table / context length are emitted exactly as AITER's
  ``emit_sparse_block_table_rows`` (full blocks packed in rank order, partial
  tail block last, zeros after, every kv head of the index head's group).

CUDA-graph safe: the grid depends only on launch-time ``max_seq_len`` and the
row count; the per-row completion counter is reset by the merging program.
"""

import torch

from vllm.triton_utils import tl, triton

# Launch geometry, measured on MI300X (harness/topk_time.py, TOPK_NOTES.md):
# few rows split each row into up to TOPK_MAX_CHUNKS programs of >= one
# 256-block tile; more rows use 512-block tiles and split only past 1024
# blocks per chunk, since the merge then costs more than it saves.
TOPK_MAX_CHUNKS = 16
TOPK_TARGET_GRID = 64
TOPK_FEW_ROWS = 8
TOPK_FEW_ROWS_TILE = 256
TOPK_MANY_ROWS_TILE = 512
TOPK_MANY_ROWS_CHUNK_MIN = 1024
# Rows (chunks) of at most this many blocks take a narrower single-tile
# select: the bitonic top-k cost grows with the tile width, which dominates
# short-context decode at high batch.
TOPK_SHORT_TILE = 256
# auto dispatch: rows * heads * max_block above this goes to AITER, whose
# one-workgroup-per-row radix select wins once every CU already has a row of
# long-context work (at a 512k max_seq_len: more than 64 (row, head) pairs).
# Only launch-time values are used, never the real lens.
TOPK_TRITON_MAX_WORK = 64 * 4096


@triton.jit
def _aiter_key(score, index, valid):
    # AITER: (live && v == v) ? v : -inf, then u ^= sign ? ~0 : 0x80000000.
    score = tl.where(score != score, float("-inf"), score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
    key = (ordered.to(tl.int64) << 16) | index.to(tl.int64)
    return tl.where(valid, key, -1)


@triton.jit
def _emit_topk_and_table(
    winners,  # [BLOCK_T] int64 keys, descending; -1 = empty
    bt_tile,  # [SHORT_K] int32 block_table[req, 0:SHORT_K] (PREFETCHED only)
    topk_ptr,
    block_table_ptr,
    sparse_bt_ptr,
    sparse_ctx_ptr,
    pid_b,
    pid_h,
    req_id,
    causal_len,
    num_blocks,
    num_heads,
    num_kv_heads,
    stride_topk_h,
    stride_topk_b,
    stride_bt_b,
    stride_sbt_b,
    topk: tl.constexpr,
    block_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EMIT: tl.constexpr,
    PREFETCHED: tl.constexpr,
):
    off_t = tl.arange(0, BLOCK_T)
    blk = (winners & 0xFFFF).to(tl.int32)
    blk = tl.where((off_t < topk) & (off_t < num_blocks) & (winners >= 0), blk, -1)
    tl.store(
        topk_ptr + pid_h * stride_topk_h + pid_b * stride_topk_b + off_t,
        blk,
        mask=off_t < topk,
    )
    if EMIT:
        self_blk = tl.where(causal_len > 0, (causal_len - 1) // block_size, 0)
        valid = (off_t < topk) & (causal_len > 0) & (blk >= 0) & (blk <= self_blk)
        is_full = valid & (blk < self_blk)
        full_i = is_full.to(tl.int32)
        n_valid = tl.sum(valid.to(tl.int32), axis=0)
        n_full = tl.sum(full_i, axis=0)
        slot = tl.where(is_full, tl.cumsum(full_i, axis=0) - full_i, n_full)
        if PREFETCHED:
            # bt_tile holds blocks [0, SHORT_K); a single-chunk launch can
            # still select past it on long rows, which falls back to a load.
            near = blk < bt_tile.shape[0]
            page = tl.gather(bt_tile, tl.where(valid & near, blk, 0), 0)
            far = tl.load(
                block_table_ptr + req_id * stride_bt_b + blk,
                mask=valid & (~near),
                other=0,
            )
            page = tl.where(near, page, far)
        else:
            page = tl.load(
                block_table_ptr + req_id * stride_bt_b + blk, mask=valid, other=0
            )
        has_tail = n_valid > n_full
        ctx = tl.where(
            has_tail,
            n_full * block_size + causal_len - self_blk * block_size,
            tl.minimum(n_valid * block_size, causal_len),
        )
        off_j = tl.arange(0, pages_per_block)
        off_w = tl.arange(0, BLOCK_T * pages_per_block)
        heads_per_group = tl.maximum(num_kv_heads // num_heads, 1)
        for i in range(0, heads_per_group):
            kv_head = pid_h * heads_per_group + i
            row = pid_b * num_kv_heads + kv_head
            sbt_row = sparse_bt_ptr + row.to(tl.int64) * stride_sbt_b
            base = page * (pages_per_block * num_kv_heads) + kv_head
            tl.store(
                sbt_row + slot[:, None] * pages_per_block + off_j[None, :],
                base[:, None] + off_j[None, :] * num_kv_heads,
                mask=valid[:, None],
            )
            tl.store(
                sbt_row + off_w,
                tl.zeros_like(off_w),
                mask=(off_w >= n_valid * pages_per_block)
                & (off_w < topk * pages_per_block),
            )
            tl.store(sparse_ctx_ptr + row, ctx)


@triton.jit(do_not_specialize=["query_len", "num_kv_heads"])
def _mxfp4_decode_topk_kernel(
    score_ptr,  # [H, rows, width] fp32
    partial_ptr,  # [NUM_CHUNKS, H, rows, BLOCK_T] int64
    counter_ptr,  # [H, >= rows] int32, zero between launches
    topk_ptr,  # [H, rows, topk] int32
    seq_lens_ptr,  # [num_reqs] int32
    block_table_ptr,  # [num_reqs, >= max_block] int32 (page-16 rebased)
    sparse_bt_ptr,  # [rows * num_kv_heads, topk * pages_per_block] int32
    sparse_ctx_ptr,  # [rows * num_kv_heads] int32
    query_len,
    num_kv_heads,
    score_width,  # readable blocks per score row (launch-time constant)
    bt_width,  # readable blocks per block_table row (launch-time constant)
    stride_s_h,
    stride_s_b,
    stride_p_c,
    stride_p_h,
    stride_p_b,
    stride_c_h,
    stride_topk_h,
    stride_topk_b,
    stride_bt_b,
    stride_sbt_b,
    topk: tl.constexpr,
    block_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EMIT: tl.constexpr,
    CHUNK_MIN: tl.constexpr,
    SHORT_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    num_heads = tl.num_programs(1)
    req_id = pid_b // query_len
    tok = pid_b - req_id * query_len
    off_k = tl.arange(0, BLOCK_K)
    off_s = tl.arange(0, SHORT_K)
    row_ptr = (
        score_ptr + pid_h.to(tl.int64) * stride_s_h + pid_b.to(tl.int64) * stride_s_b
    )
    # Chunk 0 always starts at block 0, so its first tile (and the request's
    # first SHORT_K block-table entries) are issued before seq_lens resolves:
    # the three loads overlap instead of chaining. Masked by launch-time widths
    # only; the causal mask is applied to the keys below.
    if SHORT_K < BLOCK_K:
        s0 = tl.load(
            row_ptr + off_s, mask=(off_s < score_width) & (pid_c == 0), other=0.0
        )
    else:
        s0 = tl.load(
            row_ptr + off_k, mask=(off_k < score_width) & (pid_c == 0), other=0.0
        )
    bt0 = tl.zeros([SHORT_K], dtype=tl.int32)
    if EMIT:
        bt0 = tl.load(
            block_table_ptr + req_id * stride_bt_b + off_s,
            mask=(off_s < bt_width) & (pid_c == 0),
            other=0,
        )
    seq_len = tl.load(seq_lens_ptr + req_id)
    causal_len = tl.maximum(seq_len - query_len + tok + 1, 0)
    num_blocks = (causal_len + block_size - 1) // block_size

    # A row is split only into chunks of at least CHUNK_MIN blocks: below that
    # the merge costs more than the serial tiles it saves.
    active = tl.maximum(
        1, tl.minimum(NUM_CHUNKS, (num_blocks + CHUNK_MIN - 1) // CHUNK_MIN)
    )
    if pid_c >= active:
        return
    per = num_blocks // active
    extra = num_blocks - per * active
    start = pid_c * per + tl.minimum(pid_c, extra)
    count = per + (pid_c < extra).to(tl.int32)

    if SHORT_K < BLOCK_K:
        if count <= SHORT_K:
            valid_s = off_s < count
            if pid_c == 0:  # noqa: SIM108 (Triton: keep the load in a branch)
                s_short = s0
            else:
                s_short = tl.load(row_ptr + start + off_s, mask=valid_s, other=0.0)
            best = tl.topk(_aiter_key(s_short, start + off_s, valid_s), BLOCK_T)
        else:
            valid_w = off_k < count
            s_wide = tl.load(row_ptr + start + off_k, mask=valid_w, other=0.0)
            best = tl.topk(_aiter_key(s_wide, start + off_k, valid_w), BLOCK_T)
    else:
        valid = off_k < count
        if pid_c == 0:  # noqa: SIM108 (Triton: keep the load in a branch)
            s = s0
        else:
            s = tl.load(row_ptr + start + off_k, mask=valid, other=0.0)
        best = tl.topk(_aiter_key(s, start + off_k, valid), BLOCK_T)
    for off in tl.range(BLOCK_K, count, BLOCK_K):
        valid = off_k < count - off
        s = tl.load(row_ptr + start + off + off_k, mask=valid, other=0.0)
        tile = tl.topk(_aiter_key(s, start + off + off_k, valid), BLOCK_T)
        best = tl.topk(tl.cat(best, tile, can_reorder=True), BLOCK_T)

    if NUM_CHUNKS > 1:  # noqa: SIM102 (constexpr guard)
        if active > 1:
            off_t = tl.arange(0, BLOCK_T)
            tl.store(
                partial_ptr
                + pid_c * stride_p_c
                + pid_h * stride_p_h
                + pid_b * stride_p_b
                + off_t,
                best,
            )
            counter = counter_ptr + pid_h * stride_c_h + pid_b
            arrival = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
            if arrival != active - 1:
                return
            off_m = tl.arange(0, NUM_CHUNKS * BLOCK_T)
            c = off_m // BLOCK_T
            cand = tl.load(
                partial_ptr
                + c * stride_p_c
                + pid_h * stride_p_h
                + pid_b * stride_p_b
                + off_m % BLOCK_T,
                mask=c < active,
                other=-1,
                volatile=True,
            )
            best = tl.topk(cand, BLOCK_T)
            tl.atomic_xchg(counter, 0, sem="acq_rel", scope="gpu")
            _emit_topk_and_table(
                best,
                bt0,
                topk_ptr,
                block_table_ptr,
                sparse_bt_ptr,
                sparse_ctx_ptr,
                pid_b,
                pid_h,
                req_id,
                causal_len,
                num_blocks,
                num_heads,
                num_kv_heads,
                stride_topk_h,
                stride_topk_b,
                stride_bt_b,
                stride_sbt_b,
                topk,
                block_size,
                pages_per_block,
                BLOCK_T,
                EMIT,
                False,
            )
            return

    # Single chunk: pages of blocks < BLOCK_K come from the prefetched bt0.
    _emit_topk_and_table(
        best,
        bt0,
        topk_ptr,
        block_table_ptr,
        sparse_bt_ptr,
        sparse_ctx_ptr,
        pid_b,
        pid_h,
        req_id,
        causal_len,
        num_blocks,
        num_heads,
        num_kv_heads,
        stride_topk_h,
        stride_topk_b,
        stride_bt_b,
        stride_sbt_b,
        topk,
        block_size,
        pages_per_block,
        BLOCK_T,
        EMIT,
        True,
    )


def decode_topk_launch_policy(
    max_block: int,
    rows: int,
    heads: int,
    block_k: int = 0,
    max_chunks: int = 0,
    chunk_min: int = 0,
) -> tuple[int, int, int]:
    """(num_chunks, block_k, chunk_min) for a launch-time block bound and row
    count. Chunks of at most ``TOPK_SHORT_TILE`` blocks additionally take a
    narrower single-tile select inside the kernel."""
    few = rows * heads <= TOPK_FEW_ROWS
    if not block_k:
        block_k = TOPK_FEW_ROWS_TILE if few else TOPK_MANY_ROWS_TILE
    if not chunk_min:
        chunk_min = block_k if few else TOPK_MANY_ROWS_CHUNK_MIN
    chunk_min = max(chunk_min, block_k)
    want = max(1, triton.cdiv(max_block, chunk_min))
    if max_chunks:
        target = min(max_chunks, want)
    elif few:
        target = min(TOPK_MAX_CHUNKS, TOPK_TARGET_GRID // max(1, rows * heads), want)
    else:
        target = min(TOPK_MAX_CHUNKS, want)
    target = max(1, target)
    return 1 << (target.bit_length() - 1), block_k, chunk_min


def mxfp4_decode_topk(
    score: torch.Tensor,  # [H, rows, width] fp32
    topk_idx: torch.Tensor,  # [H, rows, topk] int32
    block_table: torch.Tensor,  # [num_reqs, >= max_block] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    *,
    sparse_bt: torch.Tensor | None,
    sparse_ctx: torch.Tensor | None,
    max_seq_len: int,
    block_size: int,
    query_len: int,
    num_kv_heads: int,
    pages_per_block: int,
    counter: torch.Tensor | None = None,  # [H, >= rows] int32 zeros
    block_k: int = 0,
    max_chunks: int = 0,
    num_warps: int = 0,
    chunk_min: int = 0,
) -> torch.Tensor:
    """Drop-in for ``pa_sparse_block_topk`` on uniform-query-length decode."""
    heads, rows, _ = score.shape
    topk = topk_idx.size(2)
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert rows % query_len == 0 and seq_lens.numel() == rows // query_len
    assert num_kv_heads % heads == 0
    if rows == 0:
        return topk_idx
    max_block = triton.cdiv(max_seq_len, block_size)
    assert max_block < 0xFFFF
    num_chunks, block_k, chunk_min = decode_topk_launch_policy(
        max_block, rows, heads, block_k, max_chunks, chunk_min
    )
    block_t = triton.next_power_of_2(topk)
    short_k = max(min(TOPK_SHORT_TILE, block_k), block_t)
    emit = sparse_bt is not None
    if emit:
        assert sparse_ctx is not None
        assert sparse_bt.shape == (rows * num_kv_heads, topk * pages_per_block)
        assert sparse_bt.stride(1) == 1 and sparse_ctx.stride(0) == 1
    else:
        sparse_bt = topk_idx
        sparse_ctx = topk_idx
    if num_chunks > 1:
        partial = torch.empty(
            (num_chunks, heads, rows, block_t), dtype=torch.int64, device=score.device
        )
        if counter is None:
            counter = torch.zeros((heads, rows), dtype=torch.int32, device=score.device)
        assert counter.dtype == torch.int32 and counter.stride(1) == 1
        assert counter.shape[0] == heads and counter.shape[1] >= rows
    else:
        partial = topk_idx
        counter = topk_idx
    _mxfp4_decode_topk_kernel[(rows, heads, num_chunks)](
        score,
        partial,
        counter,
        topk_idx,
        seq_lens,
        block_table,
        sparse_bt,
        sparse_ctx,
        query_len,
        num_kv_heads,
        score.size(2),
        block_table.size(1),
        score.stride(0),
        score.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        counter.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        block_table.stride(0),
        sparse_bt.stride(0),
        topk=topk,
        block_size=block_size,
        pages_per_block=pages_per_block,
        NUM_CHUNKS=num_chunks,
        BLOCK_K=block_k,
        BLOCK_T=block_t,
        EMIT=emit,
        CHUNK_MIN=chunk_min,
        SHORT_K=short_k,
        num_warps=num_warps or 4,
    )
    return topk_idx


def triton_topk_preferred(
    rows: int, heads: int, max_seq_len: int, block_size: int
) -> bool:
    """``auto`` dispatch for a decode launch. Depends only on launch-time
    values, so a captured graph keeps its choice."""
    max_block = triton.cdiv(max_seq_len, block_size)
    return max_block < 0xFFFF and rows * heads * max_block <= TOPK_TRITON_MAX_WORK
