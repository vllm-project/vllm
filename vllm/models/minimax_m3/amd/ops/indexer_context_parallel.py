# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in M3 compute-only index context partitioning, independent of global DCP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _context_score(
    Q,
    Cache,
    Table,
    Lengths,
    Scores,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SCORE_HEAD_STRIDE: tl.constexpr,
    SCORE_ROW_STRIDE: tl.constexpr,
    SCORE_BLK_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    GLOBAL_BLOCKS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    CHUNK: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
):
    request = tl.program_id(0)
    chunk = tl.program_id(1)
    n = tl.arange(0, N)
    token, head = n // HEADS, n % HEADS
    row = request * QUERY_LEN + token
    d = tl.arange(0, 128)
    q = tl.load(
        Q
        + row[None, :] * Q_TOKEN_STRIDE
        + head[None, :] * Q_HEAD_STRIDE
        + d[:, None],
        mask=n[None, :] < HEADS * QUERY_LEN,
        other=0,
    )
    length = tl.load(Lengths + request)
    cutoff = length - QUERY_LEN + token + 1
    pos = tl.arange(0, 128)
    # Launch num_stages=1: pipelined masked KV loads still evaluate
    # addresses on ROCm and HSA-fault on dummy capture pages.
    for local in range(chunk * CHUNK, tl.minimum((chunk + 1) * CHUNK, LOCAL_BLOCKS)):
        block = local * WORLD + RANK
        valid = (block < GLOBAL_BLOCKS) & (block * 128 < length)
        page = tl.load(
            Table + request * TABLE_STRIDE + block, mask=valid, other=0
        ).to(tl.int64)
        page = tl.where((page >= 0) & (page < NUM_PAGES), page, 0)
        k_mask = valid & (pos[:, None] < 128)
        cache_off = (
            page * (128 * 128)
            + pos[:, None].to(tl.int64) * 128
            + d[None, :].to(tl.int64)
        )
        k = tl.load(Cache + cache_off, mask=k_mask, other=0.0)
        dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
        dot = tl.where(
            valid & (block * 128 + pos[:, None] < cutoff[None, :]),
            dot,
            float("-inf"),
        )
        score = tl.max(dot, 0)
        tl.store(
            Scores
            + head.to(tl.int64) * SCORE_HEAD_STRIDE
            + row.to(tl.int64) * SCORE_ROW_STRIDE
            + local * SCORE_BLK_STRIDE,
            score,
            mask=n < HEADS * QUERY_LEN,
        )


def indexer_context_scores(
    idx_q,
    index_cache,
    block_table,
    seq_lens,
    max_seq_len,
    rank,
    world_size,
    max_query_len,
    sm_scale,
    out=None,
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks.

    ``out``, when given, is a ``[heads, >=tokens, >=local]`` buffer written in
    place (stable address for CUDA graph capture).
    """
    if (
        world_size < 1
        or not 0 <= rank < world_size
        or max_seq_len < 1
        or max_query_len < 1
    ):
        raise ValueError("invalid context partition or query geometry")
    if (
        idx_q.ndim != 3
        or idx_q.shape[2] != 128
        or idx_q.dtype != torch.bfloat16
        or idx_q.stride(2) != 1
    ):
        raise ValueError(
            "index queries must be BF16 [tokens,heads,128] with contiguous D"
        )
    tokens, heads, _ = idx_q.shape
    if heads < 1 or seq_lens.ndim != 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("query rows must equal batch * max_query_len")
    if (
        index_cache.ndim != 3
        or index_cache.shape[1:] != (128, 128)
        or not index_cache.is_contiguous()
    ):
        raise ValueError("index cache must be contiguous [pages,128,128]")
    if index_cache.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError("unsupported index cache dtype")
    num_pages = int(index_cache.shape[0])
    if num_pages < 1:
        raise ValueError("index cache has no pages")
    blocks = triton.cdiv(max_seq_len, 128)
    if (
        block_table.ndim != 2
        or block_table.shape[0] != seq_lens.numel()
        or block_table.shape[1] < blocks
        or block_table.stride(1) != 1
    ):
        raise ValueError("block table does not cover the requested context")
    if (
        block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("block table and contiguous lengths must be int32")
    if any(
        not x.is_cuda or x.device != idx_q.device
        for x in (idx_q, index_cache, block_table, seq_lens)
    ):
        raise ValueError("all score inputs must be on the same GPU")
    local = triton.cdiv(blocks, world_size)
    if out is None:
        scores = torch.empty(
            (heads, tokens, local), dtype=torch.float32, device=idx_q.device
        )
    else:
        if (
            out.ndim != 3
            or out.shape[0] != heads
            or out.shape[1] < tokens
            or out.shape[2] < local
            or out.dtype != torch.float32
            or out.device != idx_q.device
        ):
            raise ValueError("out must be fp32 [heads, >=tokens, >=local] on device")
        scores = out[:, :tokens, :local]
    if tokens:
        # Number of grid-y chunks: cap at local_blocks, target ~64 CTAs/req.
        target = max(1, min(local, 64 // max(1, seq_lens.numel())))
        chunks = 1 << (target.bit_length() - 1)
        _context_score[(seq_lens.numel(), chunks)](
            idx_q,
            index_cache,
            block_table,
            seq_lens,
            scores,
            Q_TOKEN_STRIDE=idx_q.stride(0),
            Q_HEAD_STRIDE=idx_q.stride(1),
            TABLE_STRIDE=block_table.stride(0),
            SCORE_HEAD_STRIDE=scores.stride(0),
            SCORE_ROW_STRIDE=scores.stride(1),
            SCORE_BLK_STRIDE=scores.stride(2),
            HEADS=heads,
            QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=blocks,
            NUM_PAGES=num_pages,
            RANK=rank,
            WORLD=world_size,
            CHUNK=triton.cdiv(local, chunks),
            N=max(16, triton.next_power_of_2(heads * max_query_len)),
            SCALE=sm_scale * 1.4426950409,
            num_stages=1,
        )
    return scores
