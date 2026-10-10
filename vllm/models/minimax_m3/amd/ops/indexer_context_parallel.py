# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in M3 compute-only index context partitioning, independent of global DCP."""

import torch

from vllm.triton_utils import tl, triton

# Workgroups the score grid aims for, and the floor on how few blocks one
# chunk may walk: the query tile is loaded once outside the block loop, so a
# chunk down to a single block pays that fixed cost for nothing.
DECODE_SCORE_TARGET_GRID = 1 << 14
DECODE_SCORE_MIN_BLOCKS = 3

_SCORE_SHAPE_ARGS = (
    "TABLE_STRIDE",
    "SCORE_HEAD_STRIDE",
    "SCORE_ROW_STRIDE",
    "SCORE_BLK_STRIDE",
    "LOCAL_BLOCKS",
    "GLOBAL_BLOCKS",
    "NUM_PAGES",
)


def _decode_score_chunks(batch: int, local_blocks: int) -> int:
    """Chunks one score row is split across.

    A count, not a size: the grid is ``(request, chunk)``, so this is the
    second grid dim. The round trip through the size drops chunk counts that
    do not divide the blocks and would launch programs that only return.
    """
    if local_blocks <= 0:
        return 1
    target = max(1, DECODE_SCORE_TARGET_GRID // max(1, batch))
    chunks = min(1 << (target.bit_length() - 1), local_blocks)
    chunks = min(chunks, max(1, triton.cdiv(local_blocks, DECODE_SCORE_MIN_BLOCKS)))
    return triton.cdiv(local_blocks, triton.cdiv(local_blocks, chunks))


@triton.jit(
    do_not_specialize=_SCORE_SHAPE_ARGS,
    do_not_specialize_on_alignment=_SCORE_SHAPE_ARGS,
)
def _context_score(
    Q,
    Cache,
    Table,
    Lengths,
    Scores,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    TABLE_STRIDE,
    SCORE_HEAD_STRIDE,
    SCORE_ROW_STRIDE,
    SCORE_BLK_STRIDE,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    NUM_PAGES,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    CHUNK: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
):
    request = tl.program_id(0)
    chunk = tl.program_id(1)
    length = tl.load(Lengths + request)
    # This chunk's slice of this rank's stride of the blocks the request
    # reaches, all resolved before the loop so the body carries no predicate.
    # A load under an `if` is control dependent, which stops the pipeliner
    # from issuing the next iteration's key tile early; on a loop this far
    # into bandwidth that exposed latency is most of the runtime.
    end = tl.minimum(tl.cdiv(length, 128), GLOBAL_BLOCKS)
    scan = tl.minimum(LOCAL_BLOCKS, tl.cdiv(tl.maximum(end - RANK, 0), WORLD))
    lo = chunk * CHUNK
    hi = tl.minimum(lo + CHUNK, scan)
    if lo >= hi:
        return

    n = tl.arange(0, N)
    token, head = n // HEADS, n % HEADS
    row = request * QUERY_LEN + token
    d = tl.arange(0, 128)
    q = tl.load(
        Q + row[None, :] * Q_TOKEN_STRIDE + head[None, :] * Q_HEAD_STRIDE + d[:, None],
        mask=n[None, :] < HEADS * QUERY_LEN,
        other=0.0,
    )
    cutoff = length - QUERY_LEN + token + 1
    pos = tl.arange(0, 128)
    s_base = head.to(tl.int64) * SCORE_HEAD_STRIDE + row.to(tl.int64) * SCORE_ROW_STRIDE
    for local in tl.range(lo, hi):
        block = local * WORLD + RANK
        page = tl.load(Table + request * TABLE_STRIDE + block).to(tl.int64)
        # Branchless clamp, not a masked load: profiling and capture batches
        # run against a block table that was never populated, and a masked
        # load still evaluates the address on ROCm.
        page = tl.minimum(tl.maximum(page, 0), NUM_PAGES - 1)
        k = tl.load(Cache + page * (128 * 128) + pos[:, None] * 128 + d[None, :])
        dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
        dot = tl.where(block * 128 + pos[:, None] < cutoff[None, :], dot, float("-inf"))
        tl.store(
            Scores + s_base + local * SCORE_BLK_STRIDE,
            tl.max(dot, 0),
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

    ``max_seq_len`` sizes the grid and must be a capture-stable bound, not the
    batch's own longest sequence: the grid and ``CHUNK`` are baked into a CUDA
    graph at capture, where the dummy batch is one token long, and a replay
    sized off that would score only the first block of each shard. Pass
    ``max_model_len``; the per-request causal bound clips the real work and
    chunks past the end return immediately.

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
    if index_cache.dtype != torch.bfloat16:
        # The score loop applies no per-token scale; an fp8 index cache needs
        # the AITER indexer (#57909).
        raise ValueError(
            f"index cache must be bf16 for CP scoring, got {index_cache.dtype}"
        )
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
    out_ok = (
        out is not None
        and out.ndim == 3
        and out.shape[0] == heads
        and out.shape[1] >= tokens
        and out.shape[2] >= local
        and out.dtype == torch.float32
        and out.device == idx_q.device
    )
    if out_ok:
        scores = out[:, :tokens, :local]
    else:
        # Profiling dummy batches can exceed the persistent decode buffer;
        # do not record a fresh allocation into a CUDA graph.
        if out is not None and torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "out must be fp32 [heads, >=tokens, >=local] on device: "
                f"got {tuple(out.shape)} {out.dtype} {out.device}; "
                f"need heads={heads} tokens={tokens} local={local} "
                f"device={idx_q.device}"
            )
        scores = torch.empty(
            (heads, tokens, local), dtype=torch.float32, device=idx_q.device
        )
    if tokens:
        batch = seq_lens.numel()
        chunks = min(local, _decode_score_chunks(batch, local))
        # CHUNK bounds the score loop and the pipelining this kernel is built
        # around reads that bound, so it is rounded to a power of two while
        # the strides stay runtime. Rounding up never drops a block: a wider
        # chunk over correspondingly fewer programs still spans the shard.
        chunk = triton.next_power_of_2(triton.cdiv(local, chunks))
        _context_score[(batch, triton.cdiv(local, chunk))](
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
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=blocks,
            NUM_PAGES=num_pages,
            HEADS=heads,
            QUERY_LEN=max_query_len,
            RANK=rank,
            WORLD=world_size,
            CHUNK=chunk,
            N=max(16, triton.next_power_of_2(heads * max_query_len)),
            SCALE=sm_scale * 1.4426950409,
            num_stages=3,
        )
    return scores
