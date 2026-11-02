# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer scores directly from paged summary keys."""

import torch

from vllm.triton_utils import tl, triton

from .sparse_indexer import DecodeIndexerMetadata


@triton.jit
def _direct_scores(
    Q,
    W,
    Mean,
    Table,
    Slots,
    SeqLens,
    Scores,
    Active,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Q_TOKEN: tl.constexpr,
    Q_GROUP: tl.constexpr,
    W_TOKEN: tl.constexpr,
    W_GROUP: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
    ROW_GROUPS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
):
    worker = tl.program_id(0)
    row = tl.program_id(1)
    # Rows are (token, KV group) pairs; each request has QUERY_LEN tokens.
    # A token at offset i of its request sees seq_len - (QUERY_LEN - 1 - i).
    token = row // ROW_GROUPS
    kv_group = row % ROW_GROUPS
    request = token // QUERY_LEN
    active = tl.load(Slots + token * SLOT_STRIDE) >= 0
    seq = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64) - (
        QUERY_LEN - 1 - token % QUERY_LEN
    )
    position = tl.where(active, seq - 1, 0)
    current = position >> 3
    length = current + 1
    if worker == 0:
        tl.store(Active + row, active)
        tl.store(Positions + row, position)
        tl.store(Current + row, current)
        tl.store(Lengths + row, length)
        tl.store(TopKLengths + row, current)
    live_tiles = tl.cdiv(tl.minimum(tl.maximum(length, 0), CAPACITY).to(tl.int32), TILE)
    for tile in range(worker, live_tiles, WORKERS):
        region = tile * TILE + tl.arange(0, TILE)
        col = region // REGIONS_PER_BLOCK
        physical = tl.load(
            Table + request * TABLE_ROW + col, col < TABLE_COLS, other=-1
        )
        valid = (region < CAPACITY) & (physical >= 0) & (physical < CACHE_PAGES)
        d = tl.arange(0, 256)
        h = tl.arange(0, 16)
        raw = tl.load(
            Mean
            + (physical.to(tl.int64) * MEAN_BLOCK + region % REGIONS_PER_BLOCK * 256)[
                :, None
            ]
            + d[None, :],
            valid[:, None],
            other=0,
        )
        key = raw.to(tl.float8e4nv, bitcast=True)
        query = tl.load(
            Q.to(tl.pointer_type(tl.uint8))
            + token * Q_TOKEN
            + kv_group * Q_GROUP
            + h[None, :] * 256
            + d[:, None],
            h[None, :] < HEADS,
            other=0,
        )
        query = query.to(tl.float8e4nv, bitcast=True)
        dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
        dot = tl.maximum(dot, 0.0)
        w = tl.load(W + token * W_TOKEN + kv_group * W_GROUP + h, h < HEADS, other=0)
        # FI's FP32 epilogue has four independent FMA chains followed by
        # left-associated addition. Keep this order, not a tree reduction.
        sums = tl.full((TILE, 4), 0, tl.float32)
        lane = tl.arange(0, 4)
        for group in tl.static_range(HEADS // 4):
            index = group * 4 + lane
            a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
            weights = tl.gather(w, index, 0)
            sums = tl.fma(a, weights[None, :], sums)
        a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
        b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
        c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
        e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
        result = ((a + b) + c) + e
        tl.store(Scores + row * SCORE_ROW + region, result, region < CAPACITY)


def decode_summary_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    mean: torch.Tensor,
    block_table: torch.Tensor,
    slots: torch.Tensor,
    seq_lens: torch.Tensor,
    capacity: int,
    *,
    num_sms: int,
    row_groups: int = 1,
    query_len: int = 1,
) -> tuple[torch.Tensor, DecodeIndexerMetadata]:
    """Read raw FP8 summaries without repacking keys or allocating unit scales.

    Only the 128-region rounded live prefix is written. Consumers must use
    lengths/topk_lengths to ignore the unspecified score tail. Each group of
    ``row_groups`` query rows is one token of ``slots``; each ``query_len``
    tokens are one request of ``block_table``/``seq_lens``.
    """
    # query: [tokens, groups, 8, 256] and weights: [tokens, groups, 8], each
    # group's heads contiguous; groups may be the outer storage dimension.
    tokens, groups = query.shape[:2]
    assert groups == row_groups and query.stride(3) == 1 and query.stride(2) == 256
    assert weights.shape == query.shape[:3] and weights.stride(2) == 1
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    batch = tokens * groups
    scores = torch.empty(
        (batch, triton.cdiv(capacity, 256) * 256),
        device=query.device,
        dtype=torch.float32,
    )
    active = torch.empty(batch, device=query.device, dtype=torch.bool)
    positions = torch.empty(batch, device=query.device, dtype=torch.int64)
    current = torch.empty_like(positions)
    lengths = torch.empty(batch, device=query.device, dtype=torch.int32)
    topk_lengths = torch.empty_like(lengths)
    # Bound idle CTAs for short contexts; each worker walks its live tiles.
    workers = min(triton.cdiv(capacity, 128), triton.cdiv(2 * num_sms, batch))
    _direct_scores[(workers, batch)](
        query,
        weights,
        mean,
        block_table,
        slots,
        seq_lens,
        scores,
        active,
        positions,
        current,
        lengths,
        topk_lengths,
        query.stride(0),
        query.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_table.shape[1],
        block_table.stride(0),
        slots.stride(0),
        seq_lens.stride(0),
        mean.shape[0],
        mean.shape[1],
        mean.stride(0),
        capacity,
        scores.stride(0),
        8,
        128,
        workers,
        row_groups,
        query_len,
        num_warps=4,
        enable_fp_fusion=False,
    )
    metadata = DecodeIndexerMetadata(
        active, positions, current, lengths, block_table, topk_lengths
    )
    return scores[:, :capacity], metadata


@triton.jit(do_not_specialize=["TABLE_ROW", "REGIONS"])
def _paged_scores_sm90(
    Q,
    KV,
    W,
    Lengths,
    Table,
    Output,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    region = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    valid = (region < REGIONS) & (region < length)
    physical = tl.load(
        Table + batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
        valid,
        other=0,
    ).to(tl.int64)
    d = tl.arange(0, 256)
    h = tl.arange(0, 16)
    key = tl.load(
        KV
        + physical[:, None] * KV_PAGE
        + (region % PAGE_SIZE)[:, None] * 256
        + d[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    query = tl.load(
        Q.to(tl.pointer_type(tl.uint8))
        + batch * Q_BATCH
        + (row % NEXT_N) * Q_NEXT
        + h[None, :] * Q_HEAD
        + d[:, None],
        h[None, :] < HEADS,
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
    dot = tl.maximum(dot, 0.0)
    weights = tl.load(W + row * W_ROW + h * W_HEAD, h < HEADS, other=0)
    # Match the four independent FP32 FMA chains used by the decode indexer.
    sums = tl.full((TILE, 4), 0, tl.float32)
    lane = tl.arange(0, 4)
    for group in tl.static_range(HEADS // 4):
        index = group * 4 + lane
        a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
        w = tl.gather(weights, index, 0)
        sums = tl.fma(a, w[None, :], sums)
    a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
    b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
    c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
    e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
    scales = tl.load(
        (KV + physical * KV_PAGE + PAGE_SIZE * 256 + (region % PAGE_SIZE) * 4).to(
            tl.pointer_type(tl.float32)
        ),
        valid,
        other=0,
    )
    result = (((a + b) + c) + e) * scales
    tl.store(
        Output + row * REGIONS + region,
        tl.where(valid, result, float("-inf")),
        region < REGIONS,
    )


def paged_mqa_logits_sm90(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    schedule_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score Step5's packed FP8 summary pages on Hopper.

    The packed page holds all FP8 keys followed by one FP32 scale per key.
    Query pairs share a page table and length, as in FlashInfer's interface.
    """
    batch, next_n, heads, dim = q.shape
    assert q.dtype == torch.float8_e4m3fn and dim == 256
    assert heads in (4, 8, 16) and q.stride(-1) == 1
    assert kv_fused.dtype == torch.uint8 and kv_fused.is_contiguous()
    assert kv_fused.shape[2:] == (1, 260)
    assert weights.shape == (batch * next_n, heads)
    output = torch.empty(
        (batch * next_n, max_seq_len), dtype=torch.float32, device=q.device
    )
    if batch and max_seq_len:
        _paged_scores_sm90[(batch * next_n, triton.cdiv(max_seq_len, 32))](
            q,
            kv_fused,
            weights,
            seq_lens,
            block_tables,
            output,
            *q.stride()[:3],
            *weights.stride(),
            kv_fused.stride(0),
            *block_tables.stride(),
            seq_lens.stride(0),
            max_seq_len,
            next_n,
            heads,
            kv_fused.shape[1],
            32,
            num_warps=4,
        )
    return output
