# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.

Index queries score each 128-token block of index keys (max over the block),
then the top-k blocks (plus forced init/local blocks) are selected per query
token. Adapted to vLLM's paged KV cache: the KV page size is forced to equal the
sparse block size (128), so one sparse block maps to exactly one page.

Index-K cache layout (vLLM): ``(num_blocks, 128, idx_head_dim)`` (one shared
key vector per token).

Only the paths MiniMax M3 uses are implemented: score_type="max", index value
disabled (score-only indexer), and shared index keys. Each local index-query
head selects its own block ids for the block-sparse attention kernels in
``sparse_attn``.
"""

import torch

from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    PAGES_PER_SPARSE_BLOCK,
    _write_sparse_block_table_row_from_values,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128
DECODE_SCORE_BALANCED_PROGRAM_BUDGET = 1024
DECODE_SCORE_HIGH_BATCH_PROGRAM_BUDGET = 768
MAX_DECODE_SCORE_BALANCED_REQUESTS = 11
DECODE_SCORE_DEFAULT_TARGET_GRID = 512
DECODE_SCORE_GFX950_HIGH_BATCH_TARGET_GRID = 1024
MIN_DECODE_SCORE_GFX950_HIGH_BATCH_REQUESTS = 20
MAX_DECODE_SCORE_GFX950_HIGH_BATCH_REQUESTS = 64
MAX_DECODE_SCORE_KV_CHUNKS = 256
DECODE_TOPK_BLOCKS_PER_CHUNK = 512
MAX_DECODE_TOPK_FAST_CHUNKS = 16
DECODE_TOPK_TARGET_GRID = 64


def _decode_score_program_budget(
    num_reqs: int,
    head_dim: int,
    query_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    *,
    is_gfx950: bool,
) -> int | None:
    if not (
        is_gfx950
        and head_dim == 128
        and query_dtype == torch.bfloat16
        and cache_dtype == torch.bfloat16
    ):
        return None
    if 1 <= num_reqs <= 8:
        return DECODE_SCORE_BALANCED_PROGRAM_BUDGET
    if 9 <= num_reqs <= MAX_DECODE_SCORE_BALANCED_REQUESTS:
        return DECODE_SCORE_HIGH_BATCH_PROGRAM_BUDGET
    return None


def _decode_score_split_launch_policy(
    num_reqs: int,
    head_dim: int,
    query_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    *,
    is_gfx950: bool,
) -> tuple[int, bool]:
    """Choose the generic split-K launch and high-batch specialization."""
    use_high_batch_config = (
        is_gfx950
        and MIN_DECODE_SCORE_GFX950_HIGH_BATCH_REQUESTS
        <= num_reqs
        <= MAX_DECODE_SCORE_GFX950_HIGH_BATCH_REQUESTS
        and head_dim == 128
        and query_dtype == torch.bfloat16
        and cache_dtype == torch.bfloat16
    )
    target_grid = (
        DECODE_SCORE_GFX950_HIGH_BATCH_TARGET_GRID
        if use_high_batch_config
        else DECODE_SCORE_DEFAULT_TARGET_GRID
    )
    target = max(
        1,
        min(
            MAX_DECODE_SCORE_KV_CHUNKS,
            target_grid // max(1, num_reqs),
        ),
    )
    return 1 << (target.bit_length() - 1), use_high_batch_config


def _decode_topk_launch_policy(
    max_block: int,
    total_q: int,
    num_idx_heads: int,
    topk: int,
    *,
    is_gfx950: bool,
) -> tuple[int, bool, bool]:
    """Choose the graph grid and bounded-context selector specialization."""
    if (
        is_gfx950
        and topk == 16
        and 0 < max_block <= MAX_DECODE_TOPK_FAST_CHUNKS * DECODE_TOPK_BLOCKS_PER_CHUNK
    ):
        return MAX_DECODE_TOPK_FAST_CHUNKS, True, True

    target = max(
        1,
        min(
            MAX_DECODE_TOPK_FAST_CHUNKS,
            DECODE_TOPK_TARGET_GRID // max(1, total_q * num_idx_heads),
        ),
    )
    return 1 << (target.bit_length() - 1), False, False


# ---------------------------------------------------------------------------
# Bitonic top-k helpers (layout-agnostic).
# ---------------------------------------------------------------------------
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k)
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Top-k selection over per-token block scores (layout-agnostic). block_size_q
# is 1 for M3, so top-k is computed per query token.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. The score scale is omitted
# because decode only consumes block ordering.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = q_offsets < decode_query_len
    q_ids = pid_r * decode_query_len + q_offsets

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks_q = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks_q - local_blocks)
    # Query vectors for all index heads in a small spec-decode block.
    q = tl.load(
        q_ptr
        + q_ids[None, :] * stride_q_n
        + h_offsets[None, :] * stride_q_h
        + off_d[:, None] * stride_q_d,
        mask=q_mask[None, :],
        other=0.0,
    )  # [D,HQ]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[:, None] < kv_len[None, :]
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N,D]
        if BLOCK_SIZE_HQ == 1:
            # Degenerate GEMV (q is [D,1]): vectorized fp32 multiply + reduce
            # instead of an MFMA tile. Numerically equivalent to tl.dot.
            q_vec = tl.sum(q, axis=1).to(tl.float32)  # [D]
            kq = tl.sum(k.to(tl.float32) * q_vec[None, :], axis=1)[:, None]  # [N,1]
        else:
            # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k
            # are loaded in their stored dtype (bf16 or e4m3) and the MMA
            # accumulates in fp32 so the per-block max score is exact for the
            # fp8 indexer too.
            kq = tl.dot(k, q, out_dtype=tl.float32)  # [N,HQ]
        kq = tl.where(pos_mask & q_mask[None, :], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [HQ]
        is_visible_block = blk < num_blocks_q
        is_init = (blk < init_blocks) & is_visible_block
        is_local = (blk >= local_start) & is_visible_block
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
            score,
            mask=q_mask,
        )


@triton.jit
def _ceil_div_nonnegative(value, divisor):
    return value // divisor + tl.where(value % divisor != 0, 1, 0)


@triton.jit
def _decode_index_score_mapped_range(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    pid_r,
    seq_len,
    chunk_start_block,
    blocks_per_program,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = q_offsets < decode_query_len
    q_ids = pid_r * decode_query_len + q_offsets

    query_pos = seq_len - decode_query_len + q_offsets
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks_q = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    chunk_end_block = tl.minimum(
        chunk_start_block + blocks_per_program,
        num_blocks,
    )

    if chunk_start_block < chunk_end_block:
        off_k = tl.arange(0, BLOCK_SIZE_K)
        off_d = tl.arange(0, head_dim)
        bt_row = block_table_ptr + pid_r * stride_bt_b
        local_start = tl.maximum(0, num_blocks_q - local_blocks)
        q = tl.load(
            q_ptr
            + q_ids[None, :] * stride_q_n
            + h_offsets[None, :] * stride_q_h
            + off_d[:, None] * stride_q_d,
            mask=q_mask[None, :],
            other=0.0,
        )
        for blk in tl.range(chunk_start_block, chunk_end_block):
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = blk * BLOCK_SIZE_K + off_k
            pos_mask = pos[:, None] < kv_len[None, :]
            k = tl.load(
                ik_cache_ptr
                + page * stride_ik_blk
                + off_k[:, None] * stride_ik_pos
                + off_d * stride_ik_d,
            )
            if BLOCK_SIZE_HQ == 1:
                q_vec = tl.sum(q, axis=1).to(tl.float32)
                kq = tl.sum(
                    k.to(tl.float32) * q_vec[None, :],
                    axis=1,
                )[:, None]
            else:
                kq = tl.dot(k, q, out_dtype=tl.float32)
            kq = tl.where(
                pos_mask & q_mask[None, :],
                kq,
                float("-inf"),
            )
            score = tl.max(kq, axis=0)
            is_visible_block = blk < num_blocks_q
            is_init = (blk < init_blocks) & is_visible_block
            is_local = (blk >= local_start) & is_visible_block
            score = tl.where(
                is_local,
                1e29,
                tl.where(is_init, 1e30, score),
            )
            tl.store(
                score_ptr
                + h_offsets * stride_s_h
                + q_ids * stride_s_n
                + blk * stride_s_k,
                score,
                mask=q_mask,
            )


@triton.jit(do_not_specialize=["score_program_budget", "decode_query_len"])
def _decode_index_score_balanced_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    seq_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    NUM_REQUESTS: tl.constexpr,
    score_program_budget,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    total_blocks = tl.zeros((), dtype=tl.int32)
    for req in tl.static_range(0, NUM_REQUESTS):
        req_seq_len = tl.maximum(tl.load(seq_lens + req), 0)
        total_blocks += _ceil_div_nonnegative(
            req_seq_len,
            BLOCK_SIZE_K,
        )
    blocks_per_program = tl.maximum(
        1,
        _ceil_div_nonnegative(
            total_blocks,
            score_program_budget,
        ),
    )

    program_offset = tl.zeros((), dtype=tl.int32)
    pid_r = tl.zeros((), dtype=tl.int32)
    chunk_start_block = tl.zeros((), dtype=tl.int32)
    selected_seq_len = tl.zeros((), dtype=tl.int32)
    owns_work = tl.zeros((), dtype=tl.int32)
    for req in tl.static_range(0, NUM_REQUESTS):
        req_seq_len = tl.maximum(tl.load(seq_lens + req), 0)
        req_num_blocks = _ceil_div_nonnegative(
            req_seq_len,
            BLOCK_SIZE_K,
        )
        req_programs = _ceil_div_nonnegative(
            req_num_blocks,
            blocks_per_program,
        )
        owns_request = (pid >= program_offset) & (pid < program_offset + req_programs)
        pid_r = tl.where(owns_request, req, pid_r)
        local_program = tl.where(
            owns_request,
            pid - program_offset,
            0,
        )
        chunk_start_block = tl.where(
            owns_request,
            local_program * blocks_per_program,
            chunk_start_block,
        )
        selected_seq_len = tl.where(
            owns_request,
            req_seq_len,
            selected_seq_len,
        )
        owns_work += owns_request.to(tl.int32)
        program_offset += req_programs
    if owns_work == 0:
        return

    _decode_index_score_mapped_range(
        q_ptr,
        ik_cache_ptr,
        score_ptr,
        block_table_ptr,
        pid_r,
        selected_seq_len,
        chunk_start_block,
        blocks_per_program,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        stride_q_n,
        stride_q_h,
        stride_q_d,
        stride_ik_blk,
        stride_ik_pos,
        stride_ik_d,
        stride_s_h,
        stride_s_n,
        stride_s_k,
        stride_bt_b,
        BLOCK_SIZE_K,
        BLOCK_SIZE_Q,
    )


# ---------------------------------------------------------------------------
# Decode top-k. Each CTA keeps a packed score/id top-k. Short rows emit
# directly; long rows synchronize their live CTAs and the last arrival merges
# the partial lists and optionally builds the sparse page table.
# ---------------------------------------------------------------------------
@triton.jit
def _decode_topk_key(score, index, valid):
    score = tl.where(score != score, -1e30, score)
    score = tl.where(score == 0.0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    tie = (0xFFFF - index).to(tl.int64)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | tie
    return tl.where(valid, key, 0)


@triton.jit
def _store_decode_topk(
    winners,
    topk_ptr,
    attention_block_table_ptr,
    sparse_bt_ptr,
    sparse_ctx_ptr,
    pid_b,
    pid_h,
    req_id,
    query_pos,
    num_blocks,
    topk: tl.constexpr,
    block_size: tl.constexpr,
    pages_per_sparse_block: tl.constexpr,
    block_page_stride: tl.constexpr,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_attention_bt_b,
    stride_sparse_bt_b,
    BLOCK_SIZE_T: tl.constexpr,
    EMIT_SPARSE_TABLE: tl.constexpr,
):
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (0xFFFF - (winners & 0xFFFF)).to(tl.int32) - 1
    topk_idx = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx, -1)
    tl.store(
        topk_ptr
        + pid_h * stride_topk_h
        + pid_b * stride_topk_b
        + off_t * stride_topk_t,
        topk_idx,
        mask=off_t < topk,
    )
    if EMIT_SPARSE_TABLE:
        _write_sparse_block_table_row_from_values(
            topk_idx,
            attention_block_table_ptr + req_id * stride_attention_bt_b,
            sparse_bt_ptr + pid_b * stride_sparse_bt_b,
            sparse_ctx_ptr + pid_b,
            query_pos,
            topk,
            block_size,
            pages_per_sparse_block,
            block_page_stride,
            BLOCK_SIZE_T,
        )


@triton.jit
def _merge_store_decode_topk(
    partial_ptr,
    counter,
    topk_ptr,
    attention_block_table_ptr,
    sparse_bt_ptr,
    sparse_ctx_ptr,
    active_chunks,
    pid_b,
    pid_h,
    req_id,
    query_pos,
    num_blocks,
    stride_partial_c,
    stride_partial_h,
    stride_partial_b,
    stride_partial_t,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_attention_bt_b,
    stride_sparse_bt_b,
    topk: tl.constexpr,
    block_size: tl.constexpr,
    pages_per_sparse_block: tl.constexpr,
    block_page_stride: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MERGE_CHUNKS: tl.constexpr,
    EMIT_SPARSE_TABLE: tl.constexpr,
):
    candidate_width: tl.constexpr = MERGE_CHUNKS * BLOCK_SIZE_T
    off_candidate = tl.arange(0, candidate_width)
    candidate_chunk = off_candidate // BLOCK_SIZE_T
    candidate_slot = off_candidate % BLOCK_SIZE_T
    candidates = tl.load(
        partial_ptr
        + candidate_chunk * stride_partial_c
        + pid_h * stride_partial_h
        + pid_b * stride_partial_b
        + candidate_slot * stride_partial_t,
        mask=candidate_chunk < active_chunks,
        other=0,
        volatile=True,
    )
    winners = tl.topk(candidates, BLOCK_SIZE_T)
    _store_decode_topk(
        winners,
        topk_ptr,
        attention_block_table_ptr,
        sparse_bt_ptr,
        sparse_ctx_ptr,
        pid_b,
        pid_h,
        req_id,
        query_pos,
        num_blocks,
        topk,
        block_size,
        pages_per_sparse_block,
        block_page_stride,
        stride_topk_h,
        stride_topk_b,
        stride_topk_t,
        stride_attention_bt_b,
        stride_sparse_bt_b,
        BLOCK_SIZE_T,
        EMIT_SPARSE_TABLE,
    )
    tl.atomic_xchg(counter, 0, sem="acq_rel", scope="gpu")


@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_topk_fused_kernel(
    score_ptr,  # [num_idx_heads, total_q, max_block]
    partial_ptr,  # [NUM_TOPK_CHUNKS, num_idx_heads, total_q, BLOCK_SIZE_T]
    counter_ptr,  # [num_idx_heads, total_q]
    topk_ptr,  # [num_idx_heads, total_q, topk]
    seq_lens_ptr,  # [num_reqs]
    attention_block_table_ptr,  # [num_reqs, max_blocks]
    sparse_bt_ptr,  # [total_q, topk * PAGES_PER_SPARSE_BLOCK]
    sparse_ctx_ptr,  # [total_q]
    decode_query_len,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_partial_c,
    stride_partial_h,
    stride_partial_b,
    stride_partial_t,
    stride_counter_h,
    stride_counter_b,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_attention_bt_b,
    stride_sparse_bt_b,
    topk: tl.constexpr,
    block_size: tl.constexpr,
    pages_per_sparse_block: tl.constexpr,
    block_page_stride: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    EMIT_SPARSE_TABLE: tl.constexpr,
    SINGLE_TILE_GUARANTEED: tl.constexpr,
    ADAPTIVE_FINAL_MERGE: tl.constexpr,
):
    tl.static_assert(topk <= BLOCK_SIZE_T)
    tl.static_assert(BLOCK_SIZE_T <= BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens_ptr + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    desired_chunks = (num_blocks + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    active_chunks = tl.maximum(1, tl.minimum(NUM_TOPK_CHUNKS, desired_chunks))
    if pid_chunk >= active_chunks:
        return

    blocks_per_chunk = num_blocks // active_chunks
    extra_chunks = num_blocks - blocks_per_chunk * active_chunks
    chunk_start = pid_chunk * blocks_per_chunk + tl.minimum(pid_chunk, extra_chunks)
    chunk_blocks = blocks_per_chunk + (pid_chunk < extra_chunks)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    if SINGLE_TILE_GUARANTEED:
        valid = off_k < chunk_blocks
        score = tl.load(
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + (chunk_start + off_k) * stride_s_k,
            mask=valid,
            other=-1e30,
        ).to(tl.float32)
        index = (chunk_start + off_k + 1).to(tl.int32)
        local_topk = tl.topk(
            _decode_topk_key(score, index, valid),
            BLOCK_SIZE_T,
        )
    else:
        score_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + (chunk_start + off_k) * stride_s_k
        )
        valid = off_k < chunk_blocks
        score = tl.load(score_ptrs, mask=valid, other=-1e30).to(tl.float32)
        index = (chunk_start + off_k + 1).to(tl.int32)
        local_topk = tl.topk(
            _decode_topk_key(score, index, valid),
            BLOCK_SIZE_T,
        )
        score_ptrs += BLOCK_SIZE_K * stride_s_k
        for offset in tl.range(BLOCK_SIZE_K, chunk_blocks, BLOCK_SIZE_K):
            valid = off_k < chunk_blocks - offset
            score = tl.load(score_ptrs, mask=valid, other=-1e30).to(tl.float32)
            index = (chunk_start + offset + off_k + 1).to(tl.int32)
            tile_topk = tl.topk(
                _decode_topk_key(score, index, valid),
                BLOCK_SIZE_T,
            )
            local_topk = tl.topk(
                tl.cat(local_topk, tile_topk, can_reorder=True),
                BLOCK_SIZE_T,
            )
            score_ptrs += BLOCK_SIZE_K * stride_s_k

    if active_chunks == 1:
        _store_decode_topk(
            local_topk,
            topk_ptr,
            attention_block_table_ptr,
            sparse_bt_ptr,
            sparse_ctx_ptr,
            pid_b,
            pid_h,
            req_id,
            query_pos,
            num_blocks,
            topk,
            block_size,
            pages_per_sparse_block,
            block_page_stride,
            stride_topk_h,
            stride_topk_b,
            stride_topk_t,
            stride_attention_bt_b,
            stride_sparse_bt_b,
            BLOCK_SIZE_T,
            EMIT_SPARSE_TABLE,
        )
        return

    tl.store(
        partial_ptr
        + pid_chunk * stride_partial_c
        + pid_h * stride_partial_h
        + pid_b * stride_partial_b
        + off_t * stride_partial_t,
        local_topk,
    )
    counter = counter_ptr + pid_h * stride_counter_h + pid_b * stride_counter_b
    arrival = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrival != active_chunks - 1:
        return

    if ADAPTIVE_FINAL_MERGE:
        if active_chunks <= 2:
            _merge_store_decode_topk(
                partial_ptr,
                counter,
                topk_ptr,
                attention_block_table_ptr,
                sparse_bt_ptr,
                sparse_ctx_ptr,
                active_chunks,
                pid_b,
                pid_h,
                req_id,
                query_pos,
                num_blocks,
                stride_partial_c,
                stride_partial_h,
                stride_partial_b,
                stride_partial_t,
                stride_topk_h,
                stride_topk_b,
                stride_topk_t,
                stride_attention_bt_b,
                stride_sparse_bt_b,
                topk,
                block_size,
                pages_per_sparse_block,
                block_page_stride,
                BLOCK_SIZE_T,
                2,
                EMIT_SPARSE_TABLE,
            )
            return
        if active_chunks <= 4:
            _merge_store_decode_topk(
                partial_ptr,
                counter,
                topk_ptr,
                attention_block_table_ptr,
                sparse_bt_ptr,
                sparse_ctx_ptr,
                active_chunks,
                pid_b,
                pid_h,
                req_id,
                query_pos,
                num_blocks,
                stride_partial_c,
                stride_partial_h,
                stride_partial_b,
                stride_partial_t,
                stride_topk_h,
                stride_topk_b,
                stride_topk_t,
                stride_attention_bt_b,
                stride_sparse_bt_b,
                topk,
                block_size,
                pages_per_sparse_block,
                block_page_stride,
                BLOCK_SIZE_T,
                4,
                EMIT_SPARSE_TABLE,
            )
            return
        if active_chunks <= 8:
            _merge_store_decode_topk(
                partial_ptr,
                counter,
                topk_ptr,
                attention_block_table_ptr,
                sparse_bt_ptr,
                sparse_ctx_ptr,
                active_chunks,
                pid_b,
                pid_h,
                req_id,
                query_pos,
                num_blocks,
                stride_partial_c,
                stride_partial_h,
                stride_partial_b,
                stride_partial_t,
                stride_topk_h,
                stride_topk_b,
                stride_topk_t,
                stride_attention_bt_b,
                stride_sparse_bt_b,
                topk,
                block_size,
                pages_per_sparse_block,
                block_page_stride,
                BLOCK_SIZE_T,
                8,
                EMIT_SPARSE_TABLE,
            )
            return
        _merge_store_decode_topk(
            partial_ptr,
            counter,
            topk_ptr,
            attention_block_table_ptr,
            sparse_bt_ptr,
            sparse_ctx_ptr,
            active_chunks,
            pid_b,
            pid_h,
            req_id,
            query_pos,
            num_blocks,
            stride_partial_c,
            stride_partial_h,
            stride_partial_b,
            stride_partial_t,
            stride_topk_h,
            stride_topk_b,
            stride_topk_t,
            stride_attention_bt_b,
            stride_sparse_bt_b,
            topk,
            block_size,
            pages_per_sparse_block,
            block_page_stride,
            BLOCK_SIZE_T,
            16,
            EMIT_SPARSE_TABLE,
        )
        return

    candidate_width: tl.constexpr = NUM_TOPK_CHUNKS * BLOCK_SIZE_T
    off_candidate = tl.arange(0, candidate_width)
    candidate_chunk = off_candidate // BLOCK_SIZE_T
    candidate_slot = off_candidate % BLOCK_SIZE_T
    candidates = tl.load(
        partial_ptr
        + candidate_chunk * stride_partial_c
        + pid_h * stride_partial_h
        + pid_b * stride_partial_b
        + candidate_slot * stride_partial_t,
        mask=candidate_chunk < active_chunks,
        other=0,
        volatile=True,
    )
    winners = tl.topk(candidates, BLOCK_SIZE_T)
    _store_decode_topk(
        winners,
        topk_ptr,
        attention_block_table_ptr,
        sparse_bt_ptr,
        sparse_ctx_ptr,
        pid_b,
        pid_h,
        req_id,
        query_pos,
        num_blocks,
        topk,
        block_size,
        pages_per_sparse_block,
        block_page_stride,
        stride_topk_h,
        stride_topk_b,
        stride_topk_t,
        stride_attention_bt_b,
        stride_sparse_bt_b,
        BLOCK_SIZE_T,
        EMIT_SPARSE_TABLE,
    )
    tl.atomic_xchg(counter, 0, sem="acq_rel", scope="gpu")


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the
    result is written into ``out[:, :total_q, :]`` instead of a fresh tensor --
    used to keep the top-k output at a stable address for cudagraph capture.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    _topk_index_kernel[grid_topk](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    *,
    attention_block_table: torch.Tensor | None = None,
    sparse_block_table_out: torch.Tensor | None = None,
    sparse_context_lens_out: torch.Tensor | None = None,
    block_page_stride: int | None = None,
    completion_counter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score followed by fused adaptive top-k selection.

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    The optional sparse-table arguments fuse current-layer table construction
    into the selector. They must be provided together. ``completion_counter``
    provides stable per-query synchronization storage for CUDA graphs. It must
    be zero before its first launch and must not be shared by overlapping
    selector invocations; every completed launch resets its active entries.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    assert 0 < decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    emit_sparse_table = attention_block_table is not None
    if emit_sparse_table and (
        sparse_block_table_out is None
        or sparse_context_lens_out is None
        or block_page_stride is None
    ):
        raise ValueError(
            "MiniMax-M3 fused decode sparse-table arguments must be provided together"
        )
    if not emit_sparse_table and (
        sparse_block_table_out is not None
        or sparse_context_lens_out is not None
        or block_page_stride is not None
    ):
        raise ValueError(
            "MiniMax-M3 fused decode sparse-table arguments must be provided together"
        )
    if emit_sparse_table:
        assert attention_block_table is not None
        assert sparse_block_table_out is not None
        assert sparse_context_lens_out is not None
        assert block_page_stride is not None
        if num_idx_heads != 1:
            raise ValueError(
                "MiniMax-M3 fused decode sparse-table construction requires "
                f"one index head, got {num_idx_heads}"
            )
        if block_page_stride not in (
            PAGES_PER_SPARSE_BLOCK,
            2 * PAGES_PER_SPARSE_BLOCK,
        ):
            raise ValueError(
                "MiniMax-M3 fused decode sparse-table page stride must be "
                f"{PAGES_PER_SPARSE_BLOCK} or {2 * PAGES_PER_SPARSE_BLOCK}, "
                f"got {block_page_stride}"
            )
        expected_sparse_shape = (total_q, topk * PAGES_PER_SPARSE_BLOCK)
        if sparse_block_table_out.shape != expected_sparse_shape:
            raise ValueError(
                "MiniMax-M3 fused decode sparse block table has shape "
                f"{tuple(sparse_block_table_out.shape)}, expected "
                f"{expected_sparse_shape}"
            )
        if sparse_context_lens_out.shape != (total_q,):
            raise ValueError(
                "MiniMax-M3 fused decode sparse context lengths have shape "
                f"{tuple(sparse_context_lens_out.shape)}, expected {(total_q,)}"
            )
        if (
            attention_block_table.dim() != 2
            or attention_block_table.shape[0] != seq_lens.shape[0]
        ):
            raise ValueError(
                "MiniMax-M3 fused decode attention block table must be "
                "[num_requests, max_blocks]"
            )
        int32_tensors = (
            attention_block_table,
            sparse_block_table_out,
            sparse_context_lens_out,
        )
        if any(tensor.dtype != torch.int32 for tensor in int32_tensors):
            raise ValueError("MiniMax-M3 fused decode sparse tables require int32")
        if any(tensor.device != idx_q.device for tensor in int32_tensors):
            raise ValueError(
                "MiniMax-M3 fused decode sparse tables must share the query device"
            )
        if (
            attention_block_table.stride(1) != 1
            or sparse_block_table_out.stride(1) != 1
            or sparse_context_lens_out.stride(0) != 1
        ):
            raise ValueError(
                "MiniMax-M3 fused decode sparse tables require contiguous rows"
            )
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    if max_block >= 0xFFFF:
        raise ValueError(
            "MiniMax-M3 decode top-k supports fewer than 65535 sparse blocks"
        )
    if emit_sparse_table:
        assert attention_block_table is not None
        if attention_block_table.shape[1] < max_block:
            raise ValueError(
                "MiniMax-M3 fused decode attention block table is shorter "
                f"than the required {max_block} sparse blocks"
            )
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    is_gfx950 = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        is_gfx950 = on_gfx950()
    # Multi-head spec decode scores a wider head-position tile per K block;
    # reduce stages to ease memory/register pressure on the fallback path.
    score_kwargs = pdl_kwargs.copy()
    if num_idx_heads > 1 and max_decode_query_len > 1:
        score_kwargs.update({"num_warps": 4, "num_stages": 2})

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # Use the configured max decode length to avoid Triton recompiles when
    # switching between qlen=1 and spec-decode verification batches.
    BLOCK_SIZE_Q = triton.next_power_of_2(max_decode_query_len)
    num_reqs = seq_lens.shape[0]
    score_program_budget = _decode_score_program_budget(
        num_reqs,
        head_dim,
        idx_q.dtype,
        index_kv_cache.dtype,
        is_gfx950=is_gfx950,
    )
    grid_score: tuple[int, ...]
    if score_program_budget is not None:
        grid_score = (score_program_budget + num_reqs - 1,)
        _decode_index_score_balanced_kernel[grid_score](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            num_idx_heads,
            head_dim,
            init_blocks,
            local_blocks,
            num_reqs,
            score_program_budget,
            decode_query_len,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            num_warps=2,
            num_stages=1,
        )
    else:
        # Increase independent work for the measured high-batch gfx950 decode
        # range while preserving the deployed split for every other shape.
        num_kv_chunks, use_high_batch_config = _decode_score_split_launch_policy(
            num_reqs,
            head_dim,
            idx_q.dtype,
            index_kv_cache.dtype,
            is_gfx950=is_gfx950,
        )
        if use_high_batch_config and not (
            num_idx_heads > 1 and max_decode_query_len > 1
        ):
            score_kwargs.update({"num_warps": 2, "num_stages": 1})
        grid_score = (num_reqs, num_kv_chunks)
        _decode_index_score_kernel[grid_score](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            num_idx_heads,
            head_dim,
            init_blocks,
            local_blocks,
            decode_query_len,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            num_kv_chunks=num_kv_chunks,
            USE_PDL=use_pdl,
            **score_kwargs,
        )

    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )
    # The launch grid remains shape-constant for CUDA graphs. Each query uses
    # only the number of chunks needed for its live context.
    (
        num_topk_chunks,
        single_tile_guaranteed,
        adaptive_final_merge,
    ) = _decode_topk_launch_policy(
        max_block,
        batch,
        num_idx_heads,
        topk,
        is_gfx950=is_gfx950,
    )
    block_size_t = triton.next_power_of_2(topk)
    topk_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.int64,
        device=idx_q.device,
    )
    if completion_counter is None:
        active_counter = torch.zeros(
            (num_idx_heads, batch),
            dtype=torch.int32,
            device=idx_q.device,
        )
    else:
        if (
            completion_counter.dim() != 2
            or completion_counter.shape[0] != num_idx_heads
            or completion_counter.shape[1] < batch
            or completion_counter.dtype != torch.int32
            or completion_counter.device != idx_q.device
            or not completion_counter.is_contiguous()
        ):
            raise ValueError(
                "MiniMax-M3 completion counter must be contiguous int32 "
                "[num_idx_heads, >=total_q] on the query device"
            )
        active_counter = completion_counter[:, :batch]

    selector_attention_block_table = block_table
    selector_sparse_block_table = topk_idx
    selector_sparse_context_lens = seq_lens
    selector_block_page_stride = PAGES_PER_SPARSE_BLOCK
    selector_sparse_block_stride = topk_idx.stride(1)
    if emit_sparse_table:
        assert attention_block_table is not None
        assert sparse_block_table_out is not None
        assert sparse_context_lens_out is not None
        assert block_page_stride is not None
        selector_attention_block_table = attention_block_table
        selector_sparse_block_table = sparse_block_table_out
        selector_sparse_context_lens = sparse_context_lens_out
        selector_block_page_stride = block_page_stride
        selector_sparse_block_stride = sparse_block_table_out.stride(0)
    _decode_topk_fused_kernel[(batch, num_idx_heads, num_topk_chunks)](
        score,
        topk_partial,
        active_counter,
        topk_idx,
        seq_lens,
        selector_attention_block_table,
        selector_sparse_block_table,
        selector_sparse_context_lens,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_partial.stride(0),
        topk_partial.stride(1),
        topk_partial.stride(2),
        topk_partial.stride(3),
        active_counter.stride(0),
        active_counter.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        selector_attention_block_table.stride(0),
        selector_sparse_block_stride,
        topk=topk,
        block_size=SPARSE_BLOCK_SIZE,
        pages_per_sparse_block=PAGES_PER_SPARSE_BLOCK,
        block_page_stride=selector_block_page_stride,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        BLOCK_SIZE_K=512,
        BLOCK_SIZE_T=block_size_t,
        EMIT_SPARSE_TABLE=emit_sparse_table,
        SINGLE_TILE_GUARANTEED=single_tile_guaranteed,
        ADAPTIVE_FINAL_MERGE=adaptive_final_merge,
        num_warps=4 if adaptive_final_merge else 8,
        num_stages=2,
    )
    return topk_idx
