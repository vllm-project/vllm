"""MiniMax-M3 sparse indexer: top-k candidate exchange for context parallelism.

After each rank scores its own shard via indexer_context_parallel, ranks
exchange only the top-k candidates instead of the full score matrix. Payload
is O(topk) rather than O(blocks) -- critical for long-context AgentX workloads.

Correctness: rank R keeps topk candidates. A global winner on R cannot be
missing from R's top-k (it would have beaten topk blocks that are in the
global set anyway). k/P per rank would NOT be safe.

Forced blocks (init=sink, local=sliding-window) must be pinned in BOTH
_local_topk (so they arrive at merge) AND _merge_topk (so they survive the
global sort). Pinning only at merge is wrong -- a forced block that lost its
rank's local top-k never arrives.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _force(score, block, valid, local_start, INIT_BLOCKS: tl.constexpr):
    score = tl.where(valid & (block < INIT_BLOCKS), 1e30, score)
    return tl.where(valid & (block >= local_start), 1e29, score)


@triton.jit
def _local_topk(
    Scores, Keys, Lengths,
    TOKENS: tl.constexpr, HEADS: tl.constexpr, QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr, GLOBAL_BLOCKS: tl.constexpr,
    RANK: tl.constexpr, WORLD: tl.constexpr, TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr, LOCAL_KEEP: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    causal_blocks = (length - QUERY_LEN + token + 128) // 128
    local_start = tl.maximum(0, causal_blocks - LOCAL_KEEP)
    s_row = Scores + (head * TOKENS + row) * LOCAL_BLOCKS
    off = tl.arange(0, BLOCK_SIZE_K)
    local_valid = off < LOCAL_BLOCKS
    block = off * WORLD + RANK
    valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
    score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
    score = _force(score, block, valid, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
    for start in tl.range(BLOCK_SIZE_K, LOCAL_BLOCKS, BLOCK_SIZE_K):
        off = start + tl.arange(0, BLOCK_SIZE_K)
        local_valid = off < LOCAL_BLOCKS
        block = off * WORLD + RANK
        valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
        score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
        score = _force(score, block, valid, local_start, INIT_BLOCKS)
        tile = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    tl.store(Keys + (head * TOKENS + row) * TOPK + off_t, winners, mask=off_t < TOPK)


@triton.jit
def _merge_topk(
    Keys, Indices, Lengths, Table, SparseBt, SparseCtx,
    TABLE_STRIDE: tl.constexpr, SBT_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr, QUERY_LEN: tl.constexpr, TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr, LOCAL_KEEP: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, PAGES_PER_BLOCK: tl.constexpr,
    KEYS_PER_SHARD: tl.constexpr, SRC_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr, REAL_CANDIDATES: tl.constexpr,
    CANDIDATES: tl.constexpr, BLOCK_SIZE_T: tl.constexpr, EMIT: tl.constexpr,
):
    row = tl.program_id(0)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    causal_len = length - QUERY_LEN + token + 1
    valid_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, valid_blocks - LOCAL_KEEP)
    off = tl.arange(0, CANDIDATES)
    src = (off // KEYS_PER_SHARD) * SRC_STRIDE + row * ROW_STRIDE + off % KEYS_PER_SHARD
    key = tl.load(Keys + src, mask=off < REAL_CANDIDATES, other=0)
    block = (key & 0xFFFF).to(tl.int32) - 1
    real = key != 0
    score = ((key >> 16) & 0xFFFFFFFF).to(tl.uint32)
    bits = score ^ tl.where(score >> 31 != 0, 0x80000000, 0xFFFFFFFF)
    value = bits.to(tl.float32, bitcast=True)
    value = _force(value, block, real, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(value, block + 1, real), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (winners & 0xFFFF).to(tl.int32) - 1
    topk_idx = tl.where(off_t < tl.minimum(TOPK, valid_blocks), topk_idx, -1)
    tl.store(Indices + row * TOPK + off_t, topk_idx, mask=off_t < TOPK)
    if EMIT:
        _emit_sparse_block_table_row(
            topk_idx, Table + request * TABLE_STRIDE,
            SparseBt + row * SBT_STRIDE, SparseCtx + row,
            causal_len, TOPK, 0, 128, PAGES_PER_BLOCK, NUM_KV_HEADS, BLOCK_SIZE_T,
        )


def local_candidate_keys(
    scores, seq_lens, topk, rank, world_size, max_query_len,
    global_blocks, init_blocks=0, local_blocks=0,
):
    """Reduce [heads,tokens,local] scores to [heads,tokens,topk] packed int64 keys."""
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [heads,tokens,local_blocks]")
    heads, tokens, local = scores.shape
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid context partition")
    if max_query_len < 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("score rows must equal batch * max_query_len")
    if topk < 1 or topk > 512:
        raise ValueError("unsupported top-k")
    if min(init_blocks, local_blocks) < 0:
        raise ValueError("forced-block counts must be non-negative")
    _require_packable(global_blocks)
    if not scores.is_contiguous() or seq_lens.dtype != torch.int32:
        raise ValueError("scores must be contiguous and lengths int32")
    keys = torch.empty((heads, tokens, topk), dtype=torch.int64, device=scores.device)
    if tokens:
        width = max(16, triton.next_power_of_2(local), triton.next_power_of_2(topk))
        width = min(width, 1024)
        _local_topk[(tokens, heads)](
            scores, keys, seq_lens,
            TOKENS=tokens, HEADS=heads, QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local, GLOBAL_BLOCKS=global_blocks,
            RANK=rank, WORLD=world_size, TOPK=topk,
            INIT_BLOCKS=init_blocks, LOCAL_KEEP=local_blocks,
            BLOCK_SIZE_K=width, BLOCK_SIZE_T=triton.next_power_of_2(topk),
            num_warps=DECODE_TOPK_NUM_WARPS,
        )
    return keys


def merge_candidate_keys(
    keys, block_table, seq_lens, topk, init_blocks, local_blocks, max_query_len
):
    """Select global top-k from gathered [world,tokens,topk] or [tokens,world*topk] keys."""
    if keys.dtype != torch.int64 or keys.ndim not in (2, 3):
        raise ValueError("keys must be int64 [tokens,world*topk] or [world,tokens,topk]")
    if keys.ndim == 3:
        shards, tokens, per_shard = keys.shape
        src_stride, row_stride, key_stride = keys.stride()
        candidates = shards * per_shard
    else:
        tokens, candidates = keys.shape
        per_shard, src_stride = candidates, 0
        row_stride, key_stride = keys.stride()
    if key_stride != 1:
        raise ValueError("the candidate axis must be contiguous")
    if max_query_len < 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("key rows must equal batch * max_query_len")
    if topk < 1 or topk > candidates:
        raise ValueError("top-k must not exceed the gathered candidate count")
    if min(init_blocks, local_blocks) < 0:
        raise ValueError("forced-block counts must be non-negative")
    if block_table.ndim != 2 or block_table.shape[0] != seq_lens.numel():
        raise ValueError("block table must have one row per request")
    if (
        block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or block_table.stride(1) != 1
    ):
        raise ValueError("metadata must be int32 with contiguous inner dimensions")
    if any(
        not x.is_cuda or x.device != keys.device for x in (keys, block_table, seq_lens)
    ):
        raise ValueError("merge inputs must share a GPU")
    indices = torch.empty((1, tokens, topk), dtype=torch.int32, device=keys.device)
    output, _ = _alloc_emit(tokens, 1, topk, block_table, True, keys.device)
    if tokens:
        _merge_topk[(tokens,)](
            keys, indices, seq_lens, block_table, output[0], output[1],
            TABLE_STRIDE=block_table.stride(0), SBT_STRIDE=output[0].stride(0),
            TOKENS=tokens, QUERY_LEN=max_query_len, TOPK=topk,
            INIT_BLOCKS=init_blocks, LOCAL_KEEP=local_blocks,
            NUM_KV_HEADS=1, PAGES_PER_BLOCK=8,
            KEYS_PER_SHARD=per_shard, SRC_STRIDE=src_stride, ROW_STRIDE=row_stride,
            REAL_CANDIDATES=candidates, CANDIDATES=triton.next_power_of_2(candidates),
            BLOCK_SIZE_T=triton.next_power_of_2(topk), EMIT=True,
            num_warps=DECODE_TOPK_NUM_WARPS,
        )
    return indices, *output
