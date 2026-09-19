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
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    GLOBAL_BLOCKS: tl.constexpr,
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
        Q + row[None, :] * Q_TOKEN_STRIDE + head[None, :] * Q_HEAD_STRIDE + d[:, None],
        mask=n[None, :] < HEADS * QUERY_LEN,
        other=0,
    )
    length = tl.load(Lengths + request)
    cutoff = length - QUERY_LEN + token + 1
    pos = tl.arange(0, 128)
    for local in range(chunk * CHUNK, tl.minimum((chunk + 1) * CHUNK, LOCAL_BLOCKS)):
        block = local * WORLD + RANK
        valid = (block < GLOBAL_BLOCKS) & (block * 128 < length)
        score = tl.full((N,), float("-inf"), tl.float32)
        if valid:
            page = tl.load(Table + request * TABLE_STRIDE + block).to(tl.int64)
            k = tl.load(Cache + page * 128 * 128 + pos[:, None] * 128 + d[None, :])
            dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
            dot = tl.where(
                block * 128 + pos[:, None] < cutoff[None, :], dot, float("-inf")
            )
            score = tl.max(dot, 0)
        tl.store(
            Scores + (head * TOKENS + row) * LOCAL_BLOCKS + local,
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
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks."""
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
    scores = torch.empty(
        (heads, tokens, local), dtype=torch.float32, device=idx_q.device
    )
    if tokens:
        chunks = min(local, _decode_score_chunks(seq_lens.numel(), local))
        _context_score[(seq_lens.numel(), chunks)](
            idx_q,
            index_cache,
            block_table,
            seq_lens,
            scores,
            Q_TOKEN_STRIDE=idx_q.stride(0),
            Q_HEAD_STRIDE=idx_q.stride(1),
            TABLE_STRIDE=block_table.stride(0),
            TOKENS=tokens,
            HEADS=heads,
            QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=blocks,
            RANK=rank,
            WORLD=world_size,
            CHUNK=triton.cdiv(local, chunks),
            N=max(16, triton.next_power_of_2(heads * max_query_len)),
            SCALE=sm_scale * 1.4426950409,
            num_stages=3,
        )
    return scores
