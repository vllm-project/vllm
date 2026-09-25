# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (gfx950/CDNA4) indexer impl for MiniMax M3.

Scores index blocks and selects the top-k with AITER's fp8 MFMA kernels. Only
the scorer differs between the two sides of the batch: prefill's rows are
ragged, each with its own causal reach, so they take
``pa_sparse_block_score_prefill``, while decode's are uniform per request and
take ``pa_sparse_block_score_decode``. The two share one tile body in AITER, so
they agree block for block, and both hand a ``[heads, rows, blocks]`` fp32
score to the same ``pa_sparse_block_topk``.

That top-k also emits the attend's page table. The winners are already in its
workgroup's LDS, so resolving them through the block table there costs one wave
and saves the attend a second pass over the selection.

Indexer CP is the one case AITER cannot serve, because its decode pair reads
the whole index cache on the rank that owns the row. That path keeps Triton
kernels of its own -- score a stride of the blocks, exchange candidates, merge
-- and they emit the same page table in the same numbering, so the two phases
of a batch can share one buffer whichever pair wrote which rows.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_cp import exchange_candidates
from vllm.models.minimax_m3.amd.ops.sparse_pa import ASM_PAGE_SIZE
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
    MiniMaxM3IndexerPrefillMetadata,
)

# The single source of truth for whether the AITER attend was asked for; this
# indexer is only usable alongside it.
from vllm.models.minimax_m3.common.sparse_attention import (
    _minimax_m3_aiter_sparse_pa_requested,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# MiniMax-M3 score/top-k shape contract
MSA_TOPK_BLOCKS = 16
MSA_SCORE_TYPE = "max"
MSA_SPARSE_BLOCK_SIZE = 128
MSA_INDEX_HEAD_DIM = 128
# The fp8 MFMA the score kernels are built on exists on these targets only.
SUPPORTED_ARCHS = ("gfx950",)

# Wave width the top-k is written against: it gives one lane per output slot
# and reads the score row in wave-wide strips.
WAVE_SIZE = 64
# Per-lane register slots the top-k can hold, which is what caps the context:
# a row may span at most SLOTS_MAX * WAVE_SIZE blocks.
SLOTS_MAX = 128
MAX_SUPPORTED_BLOCKS = SLOTS_MAX * WAVE_SIZE
# MFMA columns the score pass has, one per (query token, index head) pair.
MFMA_COLS = 16

# Workgroups the CP score grid aims for, and the floor on how few blocks one
# chunk may walk. The query tile is loaded once outside the block loop, so a
# chunk down to a single block pays that fixed cost for nothing.
DECODE_SCORE_TARGET_GRID = 1 << 14
DECODE_SCORE_MIN_BLOCKS = 3
DECODE_TOPK_NUM_WARPS = 8
DECODE_TOPK_TILE = 512


def _score_buffer(
    heads: int, rows: int, max_seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Score buffer for ``rows`` query rows, padded as the top-k requires.

    Lanes read whole wave-wide strips with no tail guard and each holds a
    power-of-two count of them, so the block axis is padded past the block
    count the context actually needs.

    Left uninitialized on purpose: the score pass writes every block up to the
    longest row it covers and the top-k reads only the blocks its own row can
    see, so nothing downstream observes the padded tail. Filling it would cost
    a write over what is, at long context, the largest tensor in the indexer.
    """
    blocks = cdiv(max(max_seq_len, 1), block_size)
    width = next_power_of_2(cdiv(blocks, WAVE_SIZE)) * WAVE_SIZE
    return torch.empty((heads, rows, width), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# The indexer-CP decode scoring & top-k kernels
# ---------------------------------------------------------------------------


def _decode_score_chunks(batch: int, max_block: int) -> int:
    """Chunks one score row is split across.

    A count, not a size: the grid is (request, chunk), so this IS the second
    grid dim and must stay shape-constant for a cudagraph to replay it. The
    round trip through the size is not redundant -- a count that does not
    divide the blocks leaves trailing chunks that start past the end and launch
    only to return.
    """
    target = max(1, DECODE_SCORE_TARGET_GRID // max(1, batch))
    if max_block <= 0:
        return 1
    chunks = min(1 << (target.bit_length() - 1), max_block)
    chunks = min(chunks, max(1, cdiv(max_block, DECODE_SCORE_MIN_BLOCKS)))
    return cdiv(max_block, cdiv(max_block, chunks))


def _require_packable(max_block: int) -> None:
    """The packed key spends its low 16 bits on the 1-based block id.

    Raised rather than asserted: this follows from ``--max-model-len`` and the
    block size, and under ``python -O`` an assertion would vanish and let the
    id wrap into the tie-break field, which is a top-k that quietly picks the
    wrong blocks.
    """
    if max_block >= 0xFFFF:
        raise ValueError(
            f"the packed top-k key addresses at most {0xFFFF - 1} blocks, got "
            f"{max_block}"
        )


@triton.jit
def _pack_score_key(score, index, valid):
    """One int64 ordering exactly like (score descending, block id descending).

    fp32 already compares like a sign-magnitude integer, so flipping the whole
    word for negatives and just the sign bit for positives gives an unsigned
    key with the same order. The 1-based block id rides in the low 16 bits, so
    equal scores resolve to the higher id. Bit 48 keeps every real candidate
    above the zero a masked-off lane carries, so padding always loses.

    The tie rule is load-bearing: block scores are a max over 128 keys from a
    3-mantissa-bit fp8 cache, so two blocks land on the same fp32 score often
    enough to matter. Which one wins does not change the selection -- both are
    in it -- but it changes their order in sparse_bt, which is the order the
    attend accumulates them in.
    """
    bits = score.to(tl.uint32, bitcast=True)
    # -0.0 and 0.0 are one score with two bit patterns; give them one key.
    bits = tl.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    # A NaN bitcasts above +inf and would win every selection it entered. Send
    # it to the bottom, below -inf, whose image is nonzero. Testing the bits
    # rather than `x != x` keeps this in the integer domain and leaves the
    # infinities exactly where they belong.
    ordered = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | index.to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _force(score, block, valid, local_start, INIT_BLOCKS: tl.constexpr):
    """Lift the always-selected blocks above every scored one.

    Two tiers so the sink blocks outrank the sliding window, matching the
    order the AITER score pass encodes them in; both sit above any real score.
    """
    score = tl.where(valid & (block < INIT_BLOCKS), 1e30, score)
    return tl.where(valid & (block >= local_start), 1e29, score)


@triton.jit
def _emit_sparse_block_table_row(
    topk_idx,  # [BLOCK_SIZE_T] selected 128-block ids, -1 for the pads
    bt_row,  # block table already offset to this request
    sbt_row,  # sparse_bt already offset to this (token, kv-head) row
    sctx_ptr,  # sparse_ctx already offset to that same row
    causal_len,  # keys this query token may attend (its position + 1)
    topk,
    pid_h,
    block_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Resolve the winners through the block table into the attend's page table.

    The same table ``pa_sparse_block_topk`` emits on the AITER path, in the
    same page-16 numbering with the pages of one block folded head-minor, so
    the two phases can write disjoint row ranges of one shared buffer.
    """
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # Tail block: the 128-block holding this token's last causal key. It is the
    # only selected block that can be partial, so it has to land last -- ctx
    # counts the leading full blocks and then however much of it is real.
    self_blk = (causal_len - 1) // block_size
    bt_blk = tl.where(off_t < topk, topk_idx, -1)
    bt_valid = (bt_blk >= 0) & (bt_blk <= self_blk)
    bt_is_tail = bt_valid & (bt_blk == self_blk)
    bt_is_full = bt_valid & (bt_blk < self_blk)
    bt_n_full = tl.sum(bt_is_full.to(tl.int32), axis=0)
    bt_n_valid = tl.sum(bt_valid.to(tl.int32), axis=0)
    bt_earlier_full = tl.cumsum(bt_is_full.to(tl.int32), axis=0) - bt_is_full.to(
        tl.int32
    )
    bt_slot = tl.where(bt_is_full, bt_earlier_full, bt_n_full)  # tail -> n_full

    bt_logical_page = tl.load(bt_row + bt_blk, mask=bt_valid, other=0).to(tl.int32)
    bt_base_phys = bt_logical_page * pages_per_block * NUM_KV_HEADS + pid_h
    bt_dst_base = bt_slot * pages_per_block

    # The pages of one block are contiguous in the destination but NUM_KV_HEADS
    # apart in the source, so this scatters a [slot, page] rectangle rather than
    # looping the page axis: one masked store on a path that runs per (query
    # token, kv head).
    pj = tl.arange(0, pages_per_block)
    tl.store(
        sbt_row + bt_dst_base[:, None] + pj[None, :],
        bt_base_phys[:, None] + pj[None, :] * NUM_KV_HEADS,
        mask=bt_valid[:, None],
    )
    off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
    tl.store(
        sbt_row + off_w,
        tl.zeros_like(off_w),
        mask=off_w >= bt_n_valid * pages_per_block,
    )

    bt_tail_tokens = causal_len - self_blk * block_size
    bt_has_tail = tl.sum(bt_is_tail.to(tl.int32), axis=0) > 0
    bt_ctx = bt_n_full * block_size + tl.where(bt_has_tail, bt_tail_tokens, 0)
    bt_ctx = tl.where(
        bt_has_tail, bt_ctx, tl.minimum(bt_n_valid * block_size, causal_len)
    )
    tl.store(sctx_ptr, bt_ctx)


# Runtime rather than constexpr across the three chain kernels, and excluded
# from Triton's own int specialization: each is a stride or a bound derived
# from ``max_seq_len`` or the decode width, so their cardinality over a serving
# run is unbounded, and a constexpr is a compile cache key. Captured steps pay
# that once, but a step carrying a prefill chunk runs eager, and there a cold
# combination stalls the host mid-chain for as long as the compile takes.
_SCORE_SHAPE_ARGS = ("TABLE_STRIDE", "TOKENS", "LOCAL_BLOCKS", "GLOBAL_BLOCKS")
_TOPK_SHAPE_ARGS = ("TOKENS", "LOCAL_BLOCKS", "GLOBAL_BLOCKS")
_MERGE_SHAPE_ARGS = ("TABLE_STRIDE", "TOKENS", "SRC_STRIDE", "HEAD_STRIDE")


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
    TOKENS,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    CHUNK: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Per-block max score over this rank's stride of the 128-blocks.

    The causal bound is taken against the true length and the GLOBAL block id,
    so a shard never shifts it.

    The dot takes the query's dtype and casts the cache to it, so an fp8 query
    keeps the whole thing on v_mfma_f32_16x16x32_fp8_fp8 -- the instruction the
    AITER pass is built on -- and a bf16 one widens the cache instead. Same
    answer either way to within fp32 accumulation order.
    """
    request = tl.program_id(0)
    chunk = tl.program_id(1)
    length = tl.load(Lengths + request)
    # This chunk's slice of this rank's stride of the blocks the request
    # reaches, all resolved before the loop so that the body needs no
    # predicate. A load under an `if` is control dependent, which stops the
    # pipeliner from issuing the next iteration's key tile early, and on a loop
    # this far into bandwidth that exposed latency is most of the runtime.
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
    # other is typed rather than the integer 0, which cannot cast to fp8 and is
    # what stops the whole kernel compiling for an fp8 query.
    q = tl.load(
        Q + row[None, :] * Q_TOKEN_STRIDE + head[None, :] * Q_HEAD_STRIDE + d[:, None],
        mask=n[None, :] < HEADS * QUERY_LEN,
        other=0.0,
    )
    cutoff = length - QUERY_LEN + token + 1
    pos = tl.arange(0, 128)
    for local in tl.range(lo, hi):
        block = local * WORLD + RANK
        page = tl.load(Table + request * TABLE_STRIDE + block).to(tl.int64)
        k = tl.load(Cache + page * 128 * 128 + pos[:, None] * 128 + d[None, :])
        dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
        dot = tl.where(block * 128 + pos[:, None] < cutoff[None, :], dot, float("-inf"))
        tl.store(
            Scores + (head * TOKENS + row) * LOCAL_BLOCKS + local,
            tl.max(dot, 0),
            mask=n < HEADS * QUERY_LEN,
        )


@triton.jit(
    do_not_specialize=_TOPK_SHAPE_ARGS,
    do_not_specialize_on_alignment=_TOPK_SHAPE_ARGS,
)
def _local_topk(
    Scores,
    Keys,
    Lengths,
    TOKENS,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Pack this shard's [heads, tokens, local] scores to its own top-k keys.

    The id inside each key is the GLOBAL block id, so the merge never needs to
    know which rank a candidate came from.

    Forced blocks are pinned HERE as well as in the merge. Pinning only at the
    merge loses them: a forced block that lost its own shard's top-k never
    arrives to be pinned. Pinning here costs nothing, because the candidate it
    evicts could not have placed globally anyway -- that block lost to k-1
    other candidates on this rank plus the forced one, and the global top-k is
    k wide.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    causal_len = length - QUERY_LEN + token + 1
    causal_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, causal_blocks - LOCAL_KEEP)
    s_row = Scores + (head * TOKENS + row) * LOCAL_BLOCKS

    scan = tl.minimum(LOCAL_BLOCKS, tl.cdiv(tl.maximum(causal_blocks - RANK, 0), WORLD))
    off = tl.arange(0, BLOCK_SIZE_K)
    local_valid = off < scan
    block = off * WORLD + RANK
    valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
    score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
    score = _force(score, block, valid, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
    for start in tl.range(BLOCK_SIZE_K, scan, BLOCK_SIZE_K):
        off = start + tl.arange(0, BLOCK_SIZE_K)
        local_valid = off < scan
        block = off * WORLD + RANK
        valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
        score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
        score = _force(score, block, valid, local_start, INIT_BLOCKS)
        tile = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    tl.store(Keys + (head * TOKENS + row) * TOPK + off_t, winners, mask=off_t < TOPK)


@triton.jit(
    do_not_specialize=_MERGE_SHAPE_ARGS,
    do_not_specialize_on_alignment=_MERGE_SHAPE_ARGS,
)
def _merge_topk(
    Keys,
    Indices,
    Lengths,
    Table,
    SparseBt,
    SparseCtx,
    TABLE_STRIDE,
    SBT_STRIDE: tl.constexpr,
    TOKENS,
    QUERY_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    KEYS_PER_SHARD: tl.constexpr,
    SRC_STRIDE,
    HEAD_STRIDE,
    ROW_STRIDE: tl.constexpr,
    REAL_CANDIDATES: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Merge gathered candidates into the global top-k, and emit the page table.

    One program per (query row, owned head): the shard axis is the exchange's,
    and this rank keeps only the run of heads it owns.

    Forced blocks are re-pinned here even though the shard pass already pinned
    them, because the pin has to survive the score round trip: this is what
    keeps a forced block ahead of a real block that happens to score above
    1e29. It is not what makes forced blocks arrive -- see ``_local_topk``.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    causal_len = tl.load(Lengths + request) - QUERY_LEN + token + 1
    valid_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, valid_blocks - LOCAL_KEEP)

    off = tl.arange(0, CANDIDATES)
    # Read with a stride rather than compacting: the exchange hands back a view
    # of its gather, and two integer ops beat the copy that flattening the
    # shard axis into the row axis would cost. CANDIDATES is rounded up to a
    # power of two; the pad lanes read 0, the same key an empty shard slot
    # carries, so _pack_score_key ranks them below every real candidate.
    src = (
        (off // KEYS_PER_SHARD) * SRC_STRIDE
        + head * HEAD_STRIDE
        + row * ROW_STRIDE
        + off % KEYS_PER_SHARD
    )
    key = tl.load(Keys + src, mask=off < REAL_CANDIDATES, other=0)
    # Unpack the id, re-pin forced blocks, repack. Padding lanes carry key 0.
    block = (key & 0xFFFF).to(tl.int32) - 1
    real = key != 0
    score = ((key >> 16) & 0xFFFFFFFF).to(tl.uint32)
    # Inverse of the order-preserving image: a set top bit means the original
    # was positive (it was flipped in), a clear one means the whole word was.
    bits = score ^ tl.where(score >> 31 != 0, 0x80000000, 0xFFFFFFFF)
    value = bits.to(tl.float32, bitcast=True)
    value = _force(value, block, real, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(value, block + 1, real), BLOCK_SIZE_T)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (winners & 0xFFFF).to(tl.int32) - 1
    topk_idx = tl.where(off_t < tl.minimum(TOPK, valid_blocks), topk_idx, -1)
    idx_row = Indices + (head * TOKENS + row) * TOPK
    tl.store(idx_row + off_t, topk_idx, mask=off_t < TOPK)
    _emit_sparse_block_table_row(
        topk_idx,
        Table + request * TABLE_STRIDE,
        SparseBt + (row * NUM_KV_HEADS + head) * SBT_STRIDE,
        SparseCtx + row * NUM_KV_HEADS + head,
        causal_len,
        TOPK,
        head,
        128,
        PAGES_PER_BLOCK,
        NUM_KV_HEADS,
        BLOCK_SIZE_T,
    )


def shard_scores(
    index_query: torch.Tensor,  # [tokens, heads, 128]
    index_cache: torch.Tensor,  # [pages, 128, 128]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    global_blocks: int,
    rank: int,
    world: int,
    query_len: int,
    scale: float,
) -> torch.Tensor:
    """[heads, tokens, ceil(global_blocks/world)] scores over this rank's blocks.

    Takes the block count rather than a length on purpose. The top-k bounds
    its causal reach against the same total, and the two kernels agree only if
    it is the same number; derived here off a block size it would be a second
    one that merely happens to match.
    """
    tokens, heads, _ = index_query.shape
    # Handed straight to the dot in whatever dtype the fused QK-norm emitted,
    # which is fp8 whenever the index cache is e4m3. The kernel indexes the
    # head dim directly, so only that stride has to be unit -- which is what
    # lets a CP rank score the replicated projection without compacting it.
    assert index_query.stride(2) == 1, "the index query's head dim must be contiguous"
    local = cdiv(global_blocks, world)
    batch = seq_lens.numel()
    scores = torch.empty(
        (heads, tokens, local), dtype=torch.float32, device=index_query.device
    )
    if tokens == 0:
        return scores
    chunks = min(local, _decode_score_chunks(batch, local))
    # Rounded to a power of two rather than made runtime like the strides
    # above, because CHUNK bounds the score loop and the pipelining this
    # kernel is built around reads that bound. Rounding up never drops a
    # block: a wider chunk over correspondingly fewer programs still spans
    # everything this rank owns.
    chunk = next_power_of_2(cdiv(local, chunks))
    chunks = cdiv(local, chunk)
    _context_score[(batch, chunks)](
        index_query,
        index_cache,
        block_table,
        seq_lens,
        scores,
        Q_TOKEN_STRIDE=index_query.stride(0),
        Q_HEAD_STRIDE=index_query.stride(1),
        TABLE_STRIDE=block_table.stride(0),
        TOKENS=tokens,
        HEADS=heads,
        QUERY_LEN=query_len,
        LOCAL_BLOCKS=local,
        GLOBAL_BLOCKS=global_blocks,
        RANK=rank,
        WORLD=world,
        CHUNK=chunk,
        N=max(16, next_power_of_2(heads * query_len)),
        SCALE=scale * 1.4426950409,
        num_stages=3,
    )
    return scores


def candidate_keys(
    scores: torch.Tensor,  # [heads, tokens, local]
    seq_lens: torch.Tensor,
    *,
    topk: int,
    rank: int,
    world: int,
    query_len: int,
    global_blocks: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """[heads, tokens, topk] packed keys, the full k of this rank's shard.

    The full k and never k/world: the global winners may lie entirely inside
    one shard, so a rank that kept fewer could drop one.
    """
    heads, tokens, local = scores.shape
    _require_packable(global_blocks)
    keys = torch.empty((heads, tokens, topk), dtype=torch.int64, device=scores.device)
    if tokens == 0:
        return keys
    # The tile has to be at least as wide as the top-k it selects, since the
    # first one runs before any cat, and no wider than the row it walks.
    width = min(DECODE_TOPK_TILE, max(next_power_of_2(local), next_power_of_2(topk)))
    _local_topk[(tokens, heads)](
        scores,
        keys,
        seq_lens,
        TOKENS=tokens,
        HEADS=heads,
        QUERY_LEN=query_len,
        LOCAL_BLOCKS=local,
        GLOBAL_BLOCKS=global_blocks,
        RANK=rank,
        WORLD=world,
        TOPK=topk,
        INIT_BLOCKS=init_blocks,
        LOCAL_KEEP=local_blocks,
        BLOCK_SIZE_K=width,
        BLOCK_SIZE_T=next_power_of_2(topk),
        num_warps=DECODE_TOPK_NUM_WARPS,
    )
    return keys


def merge_and_emit(
    keys: torch.Tensor,  # [shards, heads, tokens, topk] packed keys
    topk_idx: torch.Tensor,  # [heads, tokens, topk] int32, written in place
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,  # page-16 rebase of the ATTEND's table
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    *,
    query_len: int,
    num_kv_heads: int,
    pages_per_block: int,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Global top-k of the gathered candidates, plus the attend's page table."""
    shards, heads, tokens, per_shard = keys.shape
    topk = topk_idx.shape[2]
    candidates = shards * per_shard
    assert keys.stride(3) == 1, "the candidate axis must be contiguous"
    assert topk_idx.shape[:2] == (heads, tokens)
    # assert the number of index heads is equal to the number of kv heads
    assert heads == num_kv_heads, (
        f"this rank owns {heads} index head(s) but {num_kv_heads} kv head(s); "
        "the merge's emitted page table needs them 1:1"
    )
    if tokens == 0:
        return
    _merge_topk[(tokens, heads)](
        keys,
        topk_idx,
        seq_lens,
        block_table,
        sparse_bt,
        sparse_ctx,
        TABLE_STRIDE=block_table.stride(0),
        SBT_STRIDE=sparse_bt.stride(0),
        TOKENS=tokens,
        QUERY_LEN=query_len,
        TOPK=topk,
        INIT_BLOCKS=init_blocks,
        LOCAL_KEEP=local_blocks,
        NUM_KV_HEADS=num_kv_heads,
        PAGES_PER_BLOCK=pages_per_block,
        KEYS_PER_SHARD=per_shard,
        SRC_STRIDE=keys.stride(0),
        HEAD_STRIDE=keys.stride(1),
        ROW_STRIDE=keys.stride(2),
        REAL_CANDIDATES=candidates,
        CANDIDATES=next_power_of_2(candidates),
        # Exactly next_pow2(topk): the emit's zero-fill spans
        # BLOCK_SIZE_T * pages_per_block unmasked and the sparse_bt row is only
        # topk * pages_per_block wide, so a padded BLOCK_SIZE_T writes past the
        # row into the next allocation.
        BLOCK_SIZE_T=next_power_of_2(topk),
        num_warps=DECODE_TOPK_NUM_WARPS,
    )


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Adds the per-row shape the ragged top-k needs.

    The uniform decode rows are recovered inside the kernel from ``seq_lens`` and
    the shared query length, which also clamps cudagraph padding rows to nothing.
    Prefill rows have no such shape, so it is materialized here once per forward
    and shared by every layer -- which is also where the emitted page table's
    tail block comes from, since a block count alone does not say how many tokens
    the last block holds.
    """

    # [num_prefill_tokens] int32, causal block count per row.
    prefill_num_valid_pages: torch.Tensor | None = None
    # [num_prefill_tokens] int32, the request each prefill row belongs to.
    prefill_row_req_id: torch.Tensor | None = None
    # [num_prefill_tokens] int32, causal token count (position + 1) per row.
    prefill_kv_lens: torch.Tensor | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """The Triton indexer's metadata plus the prefill rows' causal shape."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        self.sparse_block_size = int(
            text_config.sparse_attention_config["sparse_block_size"]
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Companions to the base's num_valid_pages_buffer, for the two vectors
        # only the emitted table needs.
        self.row_req_id_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.kv_lens_buffer = torch.empty(max_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Decode-first batch: context lengths into the stable cudagraph buffer.
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: MiniMaxM3IndexerPrefillMetadata | None = None
        prefill_num_valid_pages: torch.Tensor | None = None
        prefill_row_req_id: torch.Tensor | None = None
        prefill_kv_lens: torch.Tensor | None = None
        if num_prefills > 0:
            cu_seqlens_q = (query_start_loc[num_decodes:] - num_decode_tokens).to(
                torch.int32
            )
            prefill_metadata = MiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )
            # A prefill row sees its own position, so its causal length and block
            # count both follow from that alone; the request it belongs to comes
            # from the query offsets. Prefill batches are never captured, so the
            # stable buffers are only being reused here, not required.
            positions = common_attn_metadata.positions
            assert positions is not None
            row_positions = positions[num_decode_tokens:num_tokens]
            prefill_num_valid_pages = self.num_valid_pages_buffer[
                num_decode_tokens:num_tokens
            ]
            prefill_num_valid_pages.copy_(
                row_positions // self.sparse_block_size + 1, non_blocking=True
            )
            prefill_kv_lens = self.kv_lens_buffer[num_decode_tokens:num_tokens]
            prefill_kv_lens.copy_(row_positions + 1, non_blocking=True)
            prefill_row_req_id = self.row_req_id_buffer[num_decode_tokens:num_tokens]
            prefill_row_req_id.copy_(
                torch.searchsorted(
                    cu_seqlens_q[1:].contiguous(),
                    torch.arange(
                        num_prefill_tokens,
                        dtype=torch.int32,
                        device=cu_seqlens_q.device,
                    ),
                    right=True,
                ),
                non_blocking=True,
            )

        decode_metadata: MiniMaxM3IndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            assert num_decode_tokens == num_decodes * decode_query_len
            decode_metadata = MiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        return MiniMaxM3IndexerMSAMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
            prefill_num_valid_pages=prefill_num_valid_pages,
            prefill_row_req_id=prefill_row_req_id,
            prefill_kv_lens=prefill_kv_lens,
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Fp8 score + top-k for both prefill and decode."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Both passes are v_mfma_f32_16x16x32_fp8_fp8 with no bf16 instantiation,
        # so an index cache of any other dtype would be read as e4m3 bytes.
        # The selector will not get here, but nothing else may either.
        if self.indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError(
                "The MSA indexer requires an fp8 e4m3 index cache, got "
                f"indexer_kv_dtype={self.indexer_kv_dtype!r}"
            )
        # Shared, stable-address page table + per-row context bound the top-k
        # emits for the attend. Owned by the model so one allocation serves
        # every layer; left None when it reserved none, in which case the
        # attend rebuilds the table itself.
        self.sparse_bt_buffer: torch.Tensor | None = None
        self.sparse_ctx_buffer: torch.Tensor | None = None
        # Indexer CP, assigned by the wrapper for the same reason the buffers
        # are: the impl's base ``__init__`` lives in common and knows none of
        # this. ``cp_world == 1`` is the tensor-parallel path, where
        # ``num_index_heads`` is already only this rank's heads.
        self.indexer_cp = False
        self.cp_world = 1
        self.cp_rank = 0

    @property
    def num_owned_index_heads(self) -> int:
        """Index heads whose selection this rank ends up holding.

        Under CP the scoring pass runs every head on every rank, but the merge
        leaves each rank only the run it owns -- so this, not
        ``num_index_heads``, is the width of the top-k the attend reads and of
        everything downstream of the exchange. It equals this rank's KV head
        count, which is what the emitted page table is laid out against.
        """
        return self.num_index_heads // self.cp_world

    @property
    def pages_per_block(self) -> int:
        """Physical pages one selected block expands into for the attend."""
        return self.block_size // ASM_PAGE_SIZE

    def _table_rows(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The page table and context rows covering score rows ``[lo, hi)``.

        One table row per (token, kv head), head minor, which is the order
        ``pa_decode_gluon`` reads once it flattens the cache. Slices of the
        shared buffers rather than copies, so the attend sees the writes; a
        model that reserved no buffers gets throwaway ones, and the attend
        rebuilds the table itself in that case.
        """
        kv_heads = self.num_kv_heads
        if self.sparse_bt_buffer is None or self.sparse_ctx_buffer is None:
            rows = (hi - lo) * kv_heads
            device = self.index_cache.kv_cache.device
            return (
                torch.empty(
                    (rows, self.topk_blocks * self.pages_per_block),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.empty(rows, dtype=torch.int32, device=device),
            )
        return (
            self.sparse_bt_buffer[lo * kv_heads : hi * kv_heads],
            self.sparse_ctx_buffer[lo * kv_heads : hi * kv_heads],
        )

    def _decode(
        self,
        index_query: torch.Tensor,  # [num_decode_tokens, num_index_heads, D]
        kv: torch.Tensor,
        d: MiniMaxM3IndexerDecodeMetadata,
        decode_topk: torch.Tensor,
        decode_page16_block_table: torch.Tensor,
        sparse_bt: torch.Tensor,
        sparse_ctx: torch.Tensor,
    ) -> None:
        """Score the blocks, take the top-k, emit the attend's page table.

        AITER's pair does both in one rank's worth of work, which is what CP
        exists not to do, so the sharded case takes the Triton chain instead:
        score this rank's stride of the blocks, cut that to this rank's own
        full top-k, exchange the keys, merge them into the global selection.
        """
        if self.indexer_cp:
            # Derived once for the whole chain: the score pass sizes this
            # rank's shard off it and the top-k bounds its causal reach
            # against it, and the two are only consistent if it is the same
            # number rather than two that agree because the gate pins the
            # block size.
            global_blocks = cdiv(d.max_seq_len, self.block_size)
            scores = shard_scores(
                index_query,
                kv,
                d.block_table,
                d.seq_lens,
                global_blocks=global_blocks,
                rank=self.cp_rank,
                world=self.cp_world,
                query_len=d.decode_query_len,
                scale=self.scale,
            )
            keys = candidate_keys(
                scores,
                d.seq_lens,
                topk=self.topk_blocks,
                rank=self.cp_rank,
                world=self.cp_world,
                query_len=d.decode_query_len,
                global_blocks=global_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )
            merge_and_emit(
                exchange_candidates(keys),
                decode_topk,
                d.seq_lens,
                decode_page16_block_table,
                sparse_bt,
                sparse_ctx,
                query_len=d.decode_query_len,
                num_kv_heads=self.num_kv_heads,
                pages_per_block=self.pages_per_block,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )
            return

        from aiter.ops.msa_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_topk,
        )

        heads, tokens = decode_topk.shape[:2]
        score = _score_buffer(
            heads, tokens, d.max_seq_len, self.block_size, index_query.device
        )
        pa_sparse_block_score_decode(
            index_query,
            kv,
            score,
            d.block_table,
            d.seq_lens,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            query_len=d.decode_query_len,
            max_seq_len=d.max_seq_len,
        )
        pa_sparse_block_topk(
            score,
            decode_topk,
            decode_page16_block_table,
            d.seq_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
            max_seq_len=d.max_seq_len,
            block_size=self.block_size,
            query_len=d.decode_query_len,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=self.pages_per_block,
        )

    def _prefill(
        self,
        index_query: torch.Tensor,  # [num_prefill_tokens, num_index_heads, D]
        kv: torch.Tensor,
        p: MiniMaxM3IndexerPrefillMetadata,
        prefill_topk: torch.Tensor,
        prefill_page16_block_table: torch.Tensor,
        sparse_bt: torch.Tensor,
        sparse_ctx: torch.Tensor,
        num_valid_pages: torch.Tensor,
        row_req_id: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> None:
        """Score the blocks, take the top-k, emit the attend's page table.

        AITER both sides, CP or not: prefill has a whole chunk of query rows to
        spread over the machine and nothing to gain from sharding the block
        axis on top of that, so the sharded chain the decode path takes buys it
        only an exchange it would otherwise not pay for.
        """
        from aiter.ops.msa_attention import (
            pa_sparse_block_score_prefill,
            pa_sparse_block_topk,
        )

        if self.indexer_cp:
            # The scorer indexes the head axis directly rather than taking its
            # stride, so this rank's run of the replicated projection has to be
            # compacted. Tensor-parallel rows arrive contiguous already.
            owned = self.num_owned_index_heads
            lo = self.cp_rank * owned
            index_query = index_query[:, lo : lo + owned, :].contiguous()

        heads, total_q = prefill_topk.shape[:2]
        score = _score_buffer(
            heads, total_q, p.max_seq_len, self.block_size, index_query.device
        )
        pa_sparse_block_score_prefill(
            index_query,
            kv,
            score,
            p.block_table,
            p.cu_seqlens_q,
            p.seq_lens,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            max_query_len=p.max_query_len,
            max_seq_len=p.max_seq_len,
        )
        pa_sparse_block_topk(
            score,
            prefill_topk,
            prefill_page16_block_table,
            p.seq_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
            max_seq_len=p.max_seq_len,
            block_size=self.block_size,
            num_valid_pages=num_valid_pages,
            row_req_id=row_req_id,
            kv_lens=kv_lens,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=self.pages_per_block,
        )

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
        # The emitted page table addresses the main cache, whose blocks are a
        # different group's than the index cache's, so the top-k resolves the
        # selection through the attend's block table -- only the score pass reads
        # the index cache and takes the indexer's. It has to be the page-16
        # rebase of that table, which the attend's own metadata builder does
        # once per step; the indexer's metadata cannot reach another group's
        # blocks, so the layer reads it off the attend and hands it in.
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        # Both sides write into the single shared persistent buffer (decode at
        # [:, :nd], prefill at [:, nd:]) and return views into it. The top-k
        # takes the head/row strides, so these slices need no copy back.
        buf = self.topk_indices_buffer
        if buf is None:
            buf = torch.empty(
                (self.num_owned_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            assert decode_page16_block_table is not None, (
                "the MSA indexer's top-k emits the attend's page table and "
                "needs the page-16 rebase of the attend's decode block table"
            )
            decode_topk = buf[:, :nd, :]
            sparse_bt, sparse_ctx = self._table_rows(0, nd)
            # A decode row's causal length is seq_len - query_len + token + 1,
            # which every kernel in the chain derives itself, so speculative
            # rows need no extra per-row shape.
            self._decode(
                iq[:nd],
                kv,
                d,
                decode_topk,
                decode_page16_block_table,
                sparse_bt,
                sparse_ctx,
            )

        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None
            assert prefill_page16_block_table is not None, (
                "the MSA indexer's top-k emits the attend's page table and "
                "needs the page-16 rebase of the attend's prefill block table"
            )
            assert md.prefill_num_valid_pages is not None
            assert md.prefill_row_req_id is not None
            assert md.prefill_kv_lens is not None
            prefill_topk = buf[:, nd:num_tokens, :]
            sparse_bt, sparse_ctx = self._table_rows(nd, num_tokens)
            self._prefill(
                iq[nd:],
                kv,
                p,
                prefill_topk,
                prefill_page16_block_table,
                sparse_bt,
                sparse_ctx,
                md.prefill_num_valid_pages,
                md.prefill_row_req_id,
                md.prefill_kv_lens,
            )

        return decode_topk, prefill_topk


class MiniMaxM3MSAIndexer(nn.Module):
    """``MiniMaxM3Indexer``'s surface over the MSA impl.
    Fp8 score + top-k for both prefill and decode.
    """

    def __init__(
        self,
        *,
        impl_cls: type[MiniMaxM3IndexerMSAImpl],
        sparse_bt_buffer: torch.Tensor | None = None,
        sparse_ctx_buffer: torch.Tensor | None = None,
        indexer_cp: bool = False,
        **impl_kwargs,
    ) -> None:
        super().__init__()
        self.impl = impl_cls(**impl_kwargs)
        # Assigned rather than passed: the impl's base ``__init__`` lives in
        # common and takes neither the table buffers nor the CP topology.
        self.impl.sparse_bt_buffer = sparse_bt_buffer
        self.impl.sparse_ctx_buffer = sparse_ctx_buffer
        self.impl.indexer_cp = indexer_cp
        if indexer_cp:
            self.impl.cp_world = get_tensor_model_parallel_world_size()
            self.impl.cp_rank = get_tensor_model_parallel_rank()

    @property
    def index_cache(self) -> MiniMaxM3IndexerCache:
        return self.impl.index_cache

    @property
    def num_index_heads(self) -> int:
        return self.impl.num_index_heads

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(
            index_query,
            decode_page16_block_table=decode_page16_block_table,
            prefill_page16_block_table=prefill_page16_block_table,
        )


def msa_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    indexer_kv_dtype: IndexerKVDType,
    max_model_len: int,
    score_type: str = "max",
) -> str | None:
    """Return why this config cannot use this indexer, or None if it can.

    Checks platform (ROCm/gfx950), the AITER sparse PA attend, index-cache
    dtype, the kernels' shape contract and the max context in blocks.
    ``select_msa_indexer_impl_cls`` logs the string and falls back when it is
    not None.
    """
    if not current_platform.is_rocm():
        return (
            "needs ROCm for the fp8 MFMA score/top-k, "
            f"got platform={current_platform.device_type!r}"
        )
    if not _minimax_m3_aiter_sparse_pa_requested():
        # The top-k emits the attend's page table in page-16 numbering, which
        # only addresses the interleaved cache the AITER attend reads. Paired
        # with any other attend there is nowhere valid to write it, so this
        # indexer is not usable on its own.
        return (
            "needs the AITER sparse PA attend, whose page table its top-k "
            "emits (rocm_aiter_ops + shuffle KV cache layout)"
        )
    if indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
        # The score kernels are fp8 MFMA; there is no bf16 instantiation.
        return (
            f"needs an fp8 e4m3 index cache, got indexer_kv_dtype={indexer_kv_dtype!r}"
        )
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return f"needs {' or '.join(SUPPORTED_ARCHS)} for the fp8 MFMA"
    if score_type != MSA_SCORE_TYPE:
        return f"needs score_type={MSA_SCORE_TYPE!r}, got score_type={score_type!r}"
    if topk_blocks != MSA_TOPK_BLOCKS:
        return f"needs topk_blocks={MSA_TOPK_BLOCKS}, got topk_blocks={topk_blocks}"
    if sparse_block_size != MSA_SPARSE_BLOCK_SIZE:
        return (
            f"needs sparse_block_size={MSA_SPARSE_BLOCK_SIZE}, "
            f"got sparse_block_size={sparse_block_size}"
        )
    if index_head_dim != MSA_INDEX_HEAD_DIM:
        return (
            f"needs index_head_dim={MSA_INDEX_HEAD_DIM}, "
            f"got index_head_dim={index_head_dim}"
        )

    max_blocks = math.ceil(max_model_len / sparse_block_size)
    if max_blocks > MAX_SUPPORTED_BLOCKS:
        return (
            f"max_model_len={max_model_len} needs {max_blocks} blocks per row, "
            f"more than the {MAX_SUPPORTED_BLOCKS} the top-k is compiled for"
        )
    # The CP selector packs a 1-based block id into the low 16 bits of its key.
    # Looser than the slot cap, so unreachable while that one holds; kept
    # because it belongs to a different kernel and the two move independently.
    if max_blocks >= 0xFFFF:
        return (
            f"max_model_len={max_model_len} needs {max_blocks} blocks per row, "
            f"more than the packed key's {0xFFFF - 1}"
        )
    return None


def select_msa_indexer_impl_cls(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    indexer_kv_dtype: IndexerKVDType,
    score_type: str = "max",
) -> type[MiniMaxM3IndexerMSAImpl] | None:
    """The MSA indexer impl if this config can use it, else None.

    ``None`` sends the caller to the platform-neutral ``MiniMaxM3Indexer``,
    which on ROCm means the Triton indexer -- and a bf16-only one, so an fp8
    index cache that lands here has nowhere to go.
    """
    reason = msa_indexer_unsupported_reason(
        topk_blocks=topk_blocks,
        sparse_block_size=sparse_block_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        indexer_kv_dtype=indexer_kv_dtype,
        max_model_len=get_current_vllm_config().model_config.max_model_len,
        score_type=score_type,
    )
    if reason is not None:
        logger.info_once("MiniMax M3 indexer: MSA unavailable (%s)", reason)
        return None
    logger.info_once(
        "MiniMax M3 indexer: selected MSA (Triton fp8 MFMA score + top-k) "
        "[topk_blocks=%d, indexer_kv_dtype=%s]",
        topk_blocks,
        indexer_kv_dtype,
    )
    return MiniMaxM3IndexerMSAImpl
