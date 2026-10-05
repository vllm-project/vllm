# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate exchange for MiniMax-M3 Triton indexer CP (graph-safe).

Matches PR #57909: each rank keeps, per index head, its own top-k over a 1/P
block shard, all-gathers the packed keys (payload independent of context
length), then merges per head. The merge is exact only when every shard
scored the same head, so callers must score every global head on every
rank. Forced init/local blocks are pinned on both the shard and the merge so
they cannot be dropped before the exchange.
"""

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    PAGES_PER_SPARSE_BLOCK,
    _write_sparse_block_table_row_from_values,
)
from vllm.triton_utils import tl, triton

DECODE_TOPK_TILE = 512
DECODE_TOPK_NUM_WARPS = 8

_TOPK_SHAPE_ARGS = (
    "SCORE_HEAD_STRIDE",
    "SCORE_ROW_STRIDE",
    "SCORE_BLK_STRIDE",
    "KEY_HEAD_STRIDE",
    "KEY_ROW_STRIDE",
    "LOCAL_BLOCKS",
    "GLOBAL_BLOCKS",
)
_MERGE_SHAPE_ARGS = (
    "SRC_STRIDE",
    "HEAD_STRIDE",
    "ROW_STRIDE",
    "IDX_HEAD_STRIDE",
    "IDX_ROW_STRIDE",
    "TABLE_STRIDE",
    "SBT_STRIDE",
)


@triton.jit
def _pack_score_key(score, index, valid):
    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    ordered = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | index.to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _force(score, block, valid, local_start, INIT_BLOCKS: tl.constexpr):
    score = tl.where(valid & (block < INIT_BLOCKS), 1e30, score)
    return tl.where(valid & (block >= local_start), 1e29, score)


@triton.jit(
    do_not_specialize=_TOPK_SHAPE_ARGS,
    do_not_specialize_on_alignment=_TOPK_SHAPE_ARGS,
)
def _local_topk_keys(
    Scores,
    Keys,
    Lengths,
    SCORE_HEAD_STRIDE,
    SCORE_ROW_STRIDE,
    SCORE_BLK_STRIDE,
    KEY_HEAD_STRIDE,
    KEY_ROW_STRIDE,
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
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    causal_len = length - QUERY_LEN + token + 1
    causal_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, causal_blocks - LOCAL_KEEP)
    s_row = Scores + head * SCORE_HEAD_STRIDE + row * SCORE_ROW_STRIDE
    scan = tl.minimum(LOCAL_BLOCKS, tl.cdiv(tl.maximum(causal_blocks - RANK, 0), WORLD))
    off = tl.arange(0, BLOCK_SIZE_K)
    local_valid = off < scan
    block = off * WORLD + RANK
    valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
    score = tl.load(s_row + off * SCORE_BLK_STRIDE, mask=local_valid, other=-1e30)
    score = _force(score, block, valid, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
    for start in tl.range(BLOCK_SIZE_K, scan, BLOCK_SIZE_K):
        off = start + tl.arange(0, BLOCK_SIZE_K)
        local_valid = off < scan
        block = off * WORLD + RANK
        valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
        score = tl.load(s_row + off * SCORE_BLK_STRIDE, mask=local_valid, other=-1e30)
        score = _force(score, block, valid, local_start, INIT_BLOCKS)
        tile = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    tl.store(
        Keys + head * KEY_HEAD_STRIDE + row * KEY_ROW_STRIDE + off_t,
        winners,
        mask=off_t < TOPK,
    )


@triton.jit(
    do_not_specialize=_MERGE_SHAPE_ARGS,
    do_not_specialize_on_alignment=_MERGE_SHAPE_ARGS,
)
def _merge_topk_keys(
    Keys,
    Indices,
    Lengths,
    Table,
    SparseBt,
    SparseCtx,
    SRC_STRIDE,
    HEAD_STRIDE,
    ROW_STRIDE,
    IDX_HEAD_STRIDE,
    IDX_ROW_STRIDE,
    TABLE_STRIDE,
    SBT_STRIDE,
    QUERY_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    KEYS_PER_SHARD: tl.constexpr,
    REAL_CANDIDATES: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    EMIT_SPARSE_TABLE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    causal_len = tl.load(Lengths + request) - QUERY_LEN + token + 1
    valid_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, valid_blocks - LOCAL_KEEP)
    off = tl.arange(0, CANDIDATES)
    src = (
        (off // KEYS_PER_SHARD) * SRC_STRIDE
        + head * HEAD_STRIDE
        + row * ROW_STRIDE
        + off % KEYS_PER_SHARD
    )
    key = tl.load(Keys + src, mask=off < REAL_CANDIDATES, other=0)
    block = (key & 0xFFFF).to(tl.int32) - 1
    real = key != 0
    score_bits = ((key >> 16) & 0xFFFFFFFF).to(tl.uint32)
    bits = score_bits ^ tl.where(score_bits >> 31 != 0, 0x80000000, 0xFFFFFFFF)
    value = bits.to(tl.float32, bitcast=True)
    value = _force(value, block, real, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(value, block + 1, real), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (winners & 0xFFFF).to(tl.int32) - 1
    topk_idx = tl.where(off_t < tl.minimum(TOPK, valid_blocks), topk_idx, -1)
    tl.store(
        Indices + head * IDX_HEAD_STRIDE + row * IDX_ROW_STRIDE + off_t,
        topk_idx,
        mask=off_t < TOPK,
    )
    if EMIT_SPARSE_TABLE:
        abs_pos = causal_len - 1
        _write_sparse_block_table_row_from_values(
            topk_idx,
            Table + request * TABLE_STRIDE,
            SparseBt + row * SBT_STRIDE,
            SparseCtx + row,
            abs_pos,
            TOPK,
            BLOCK_SIZE,
            PAGES_PER_BLOCK,
            BLOCK_PAGE_STRIDE,
            BLOCK_SIZE_T,
        )


def local_topk_keys(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk: int,
    rank: int,
    world: int,
    query_len: int,
    global_blocks: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write ``[heads, tokens, topk]`` packed int64 keys for this shard."""
    heads, tokens, local = scores.shape
    if global_blocks >= 0xFFFF:
        raise ValueError("packed CP keys support fewer than 65535 sparse blocks")
    if tokens == 0:
        return out
    width = min(
        DECODE_TOPK_TILE,
        max(triton.next_power_of_2(max(local, 1)), triton.next_power_of_2(topk)),
    )
    _local_topk_keys[(tokens, heads)](
        scores,
        out,
        seq_lens,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        out.stride(0),
        out.stride(1),
        QUERY_LEN=query_len,
        LOCAL_BLOCKS=local,
        GLOBAL_BLOCKS=global_blocks,
        RANK=rank,
        WORLD=world,
        TOPK=topk,
        INIT_BLOCKS=init_blocks,
        LOCAL_KEEP=local_blocks,
        BLOCK_SIZE_K=width,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        num_warps=DECODE_TOPK_NUM_WARPS,
    )
    return out


def aiter_all_gather(x: torch.Tensor) -> torch.Tensor | None:
    """AITER IPC all-gather over the TP group as ``[world, *x.shape]``, or None.

    Same path as PR #57909: the payload is viewed as int32 so AITER's pybind
    layer (which has no float64 after it relabels integers as floats) accepts
    it -- int64 keys and bf16 index queries alike. Concatenates along dim 0,
    then views back to ``x``'s dtype.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    comm = rocm_aiter_ops.get_aiter_allreduce()
    if comm is None or comm.disabled:
        return None
    if not x.is_contiguous() or (x.shape[-1] * x.element_size()) % 4:
        return None
    packed = x.view(torch.int32)
    if not comm.should_custom_ag(packed):
        return None
    gathered = comm.custom_all_gather(packed, dim=0)
    if gathered is None:
        return None
    world = get_tp_group().world_size
    return gathered.view(x.dtype).view(world, *x.shape)


def all_gather(x: torch.Tensor, dest: torch.Tensor, group) -> torch.Tensor:
    """All-gather contiguous ``x`` to ``[world, *x.shape]``.

    Prefers AITER's IPC all-gather; otherwise gathers into ``dest``, a
    caller-owned contiguous ``[world, *x.shape]`` buffer, so CUDA graphs never
    record a fresh allocation or a non-contiguous NCCL payload.
    """
    gathered = aiter_all_gather(x)
    if gathered is not None:
        return gathered
    dist.all_gather_into_tensor(
        dest.view(-1, *x.shape[1:]), x, group=group.device_group
    )
    return dest


def merge_topk_keys(
    keys: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    query_len: int,
    init_blocks: int,
    local_blocks: int,
    attention_block_table: torch.Tensor | None = None,
    sparse_block_table_out: torch.Tensor | None = None,
    sparse_context_lens_out: torch.Tensor | None = None,
    block_page_stride: int | None = None,
) -> torch.Tensor:
    """Merge gathered ``[world, heads, tokens, topk]`` keys into global top-k."""
    shards, heads, tokens, per_shard = keys.shape
    topk = topk_idx.shape[2]
    candidates = shards * per_shard
    if tokens == 0:
        return topk_idx
    emit = attention_block_table is not None
    dummy = topk_idx
    _merge_topk_keys[(tokens, heads)](
        keys,
        topk_idx,
        seq_lens,
        attention_block_table if emit else dummy,
        sparse_block_table_out if emit else dummy,
        sparse_context_lens_out if emit else dummy,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        attention_block_table.stride(0) if attention_block_table is not None else 0,
        sparse_block_table_out.stride(0) if sparse_block_table_out is not None else 0,
        QUERY_LEN=query_len,
        TOPK=topk,
        INIT_BLOCKS=init_blocks,
        LOCAL_KEEP=local_blocks,
        KEYS_PER_SHARD=per_shard,
        REAL_CANDIDATES=candidates,
        CANDIDATES=triton.next_power_of_2(max(candidates, 1)),
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        EMIT_SPARSE_TABLE=emit,
        BLOCK_SIZE=128,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_PAGE_STRIDE=block_page_stride or PAGES_PER_SPARSE_BLOCK,
        num_warps=DECODE_TOPK_NUM_WARPS,
    )
    return topk_idx
