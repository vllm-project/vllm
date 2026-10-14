# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in sharded QSA selector under DCP: the exact global top-k merge.

With ``AttentionConfig.qsa_dcp_selector="sharded"`` each DCP rank stores and
scores only the compressed states it owns and takes a local top-k. A state in
the global top-k is also in its owner's local top-k, so one all-gather of the
local ``(score, global state)`` candidates and one stable sort per row give
the global top-k of the replicated selector.
"""

import torch

from vllm.distributed import get_dcp_group
from vllm.models.qwen4_exp.common.qsa_cache import QSADCPShard
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["dcp_rank"])
def _pack_qsa_dcp_candidates_kernel(
    logits_ptr,
    local_indices_ptr,
    packed_ptr,
    stride_logits_row,
    stride_indices_row,
    dcp_rank,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
) -> None:
    """Write one row of local top-k states as ``(score, global state)`` fp32.

    Padding entries (``-1``) become ``(-inf, -1)``. fp32 holds state ids
    below 2**24 exactly.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_K)
    in_row = columns < TOPK
    local = tl.load(
        local_indices_ptr + row * stride_indices_row + columns, mask=in_row, other=-1
    )
    valid = local >= 0
    safe = tl.maximum(local, 0).to(tl.int64)
    score = tl.load(
        logits_ptr + row * stride_logits_row + safe,
        mask=in_row & valid,
        other=-float("inf"),
    )
    rounds = safe // DCP_INTERLEAVE
    global_ids = (
        rounds * DCP_WORLD + dcp_rank
    ) * DCP_INTERLEAVE + safe % DCP_INTERLEAVE
    global_ids = tl.where(valid, global_ids, -1)
    packed_row = packed_ptr + (row * TOPK + columns) * 2
    tl.store(packed_row, score, mask=in_row)
    tl.store(packed_row + 1, global_ids.to(tl.float32), mask=in_row)


@triton.jit(do_not_specialize=["stride_rank"])
def _select_qsa_dcp_topk_kernel(
    gathered_bits_ptr,
    output_ptr,
    stride_rank,
    stride_output_row,
    TOPK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Select one row's global top-k from the rank-major candidate gather.

    ``gathered_bits_ptr`` is the ``[world, rows, TOPK, 2]`` all-gather viewed
    as int32. Each candidate gets an int64 key: the score bits, flipped so
    that integer order is float order, then the inverted global id. Real keys
    are unique (a state has one owner), so the descending sort orders by
    score, then by global id ascending, padding last, whatever the rank
    order.
    """
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_N)
    in_range = lanes < DCP_WORLD * TOPK
    rank = (lanes // TOPK).to(tl.int64)
    column = (lanes % TOPK).to(tl.int64)
    pair = gathered_bits_ptr + rank * stride_rank + (row * TOPK + column) * 2
    score_bits = tl.load(pair, mask=in_range, other=0).to(tl.int64)
    global_ids = (
        tl.load(pair + 1, mask=in_range, other=0)
        .to(tl.float32, bitcast=True)
        .to(tl.int64)
    )
    # Flip the magnitude bits of negative floats so signed integer order
    # matches float order.
    score_key = tl.where(score_bits < 0, score_bits ^ 0x7FFFFFFF, score_bits)
    key = score_key * 4294967296 + (4294967295 - global_ids)
    # -2**63 (int64 min) marks padding.
    key = tl.where(in_range & (global_ids >= 0), key, -9223372036854775808)
    key = tl.sort(key, 0, descending=True)
    selected = tl.where(
        key == -9223372036854775808, -1, 4294967295 - (key & 0xFFFFFFFF)
    )
    tl.store(
        output_ptr + row * stride_output_row + lanes,
        selected.to(tl.int32),
        mask=lanes < TOPK,
    )


def pack_qsa_dcp_candidates(
    logits: torch.Tensor, local_indices: torch.Tensor, shard: QSADCPShard
) -> torch.Tensor:
    """Pack local top-k rows as ``[rows, k, 2]`` fp32 (score, global state)."""
    rows, topk = local_indices.shape
    assert logits.shape[0] == rows and logits.stride(1) == 1
    assert local_indices.stride(1) == 1
    packed = torch.empty((rows, topk, 2), dtype=torch.float32, device=logits.device)
    if rows:
        _pack_qsa_dcp_candidates_kernel[(rows,)](
            logits,
            local_indices,
            packed,
            logits.stride(0),
            local_indices.stride(0),
            shard.rank,
            TOPK=topk,
            BLOCK_K=triton.next_power_of_2(topk),
            DCP_WORLD=shard.world_size,
            DCP_INTERLEAVE=shard.interleave,
            num_warps=4,
        )
    return packed


def select_qsa_dcp_topk(rank_major: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Write the global top-k of a rank-major candidate gather to ``out``.

    ``rank_major`` is ``[world, rows, topk, 2]`` fp32 (a dim-0 all-gather of
    ``pack_qsa_dcp_candidates`` rows); ``out`` is ``[rows, topk]`` int32.
    """
    world, rows, topk, _ = rank_major.shape
    assert rank_major.is_contiguous() and rank_major.dtype == torch.float32
    assert out.shape == (rows, topk) and out.stride(1) == 1
    block_n = triton.next_power_of_2(world * topk)
    if rows:
        _select_qsa_dcp_topk_kernel[(rows,)](
            rank_major.view(torch.int32),
            out,
            rank_major.stride(0),
            out.stride(0),
            TOPK=topk,
            DCP_WORLD=world,
            BLOCK_N=block_n,
            num_warps=min(16, max(4, block_n // 256)),
        )
    return out


def merge_qsa_dcp_topk(
    logits: torch.Tensor, block_indices: torch.Tensor, shard: QSADCPShard
) -> None:
    """Replace local top-k states with the global top-k of the DCP group.

    ``block_indices`` holds this rank's local top-k as rank-local states
    (columns of ``logits``); it receives global states. Exact up to ties at a
    rank's local cut-off: the radix top-k keeps an arbitrary subset of states
    tied at its k-th score, as it does without DCP, so the merged set can
    differ from a fully ordered global top-k only among equal scores. Every
    rank must call this with the same number of rows (one all-gather).
    """
    if shard.world_size == 1:
        return
    packed = pack_qsa_dcp_candidates(logits, block_indices, shard)
    gathered = get_dcp_group().all_gather(packed, dim=0)
    select_qsa_dcp_topk(gathered.view(shard.world_size, *packed.shape), block_indices)


@triton.jit(do_not_specialize=["dcp_rank"])
def _expand_localize_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    dcp_rank,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    BLOCK_W: tl.constexpr,
) -> None:
    """``_expand_qsa_indices_kernel`` then ``_qsa_localize_dcp_kernel``, fused.

    One program holds a whole row. It expands the row as the expand kernel
    does, keeps the tokens this DCP rank owns (``DCP_INTERLEAVE`` is in
    tokens) and writes them as rank-local indices to the front of the row
    in their original order, ``-1`` after them, and their count in the
    trailing column. The two stores cover disjoint columns.
    """
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    count = expanded_count + tail_count

    safe_token = tl.maximum(token, 0)
    owned = (
        valid
        & (columns < count)
        & (safe_token // DCP_INTERLEAVE % DCP_WORLD == dcp_rank)
    )
    local_token = (
        safe_token // (DCP_INTERLEAVE * DCP_WORLD)
    ) * DCP_INTERLEAVE + safe_token % DCP_INTERLEAVE
    owned_flags = owned.to(tl.int32)
    target = tl.cumsum(owned_flags, axis=0) - 1
    num_owned = tl.sum(owned_flags, axis=0)
    row_output = output_ptr + row * stride_output_row
    tl.store(row_output + target * stride_output_column, local_token, mask=owned)
    tl.store(
        row_output + columns * stride_output_column,
        -1,
        mask=(columns >= num_owned) & (columns < OUTPUT_WIDTH),
    )
    tl.store(row_output + OUTPUT_WIDTH * stride_output_column, num_owned)


def expand_localize_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
    token_shard: QSADCPShard,
) -> None:
    """``expand_qsa_block_indices`` then ``qsa_localize_dcp_indices``, in one launch.

    ``block_indices`` holds global states (after ``merge_qsa_dcp_topk``) and
    ``visible_blocks`` the global visible counts. ``token_shard`` is this
    rank's share of the main KV cache, in tokens. ``out`` receives the
    selected tokens this rank owns as rank-local indices plus their count,
    the rows ``Qwen4ExpQSAFlashAttentionImpl.localize_dcp_selection_`` writes.
    """
    assert token_topk % compress_ratio == 0
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (
        query_positions.numel(),
        token_topk // compress_ratio,
    )
    assert out.shape == (block_indices.shape[0], output_width + 1)
    if not out.shape[0]:
        return
    _expand_localize_qsa_indices_kernel[(out.shape[0],)](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        token_shard.rank,
        BLOCK_TOPK=token_topk // compress_ratio,
        COMPRESS_RATIO=compress_ratio,
        DCP_WORLD=token_shard.world_size,
        DCP_INTERLEAVE=token_shard.interleave,
        BLOCK_W=triton.next_power_of_2(output_width),
        num_warps=8,
    )


__all__ = [
    "expand_localize_qsa_block_indices",
    "merge_qsa_dcp_topk",
    "pack_qsa_dcp_candidates",
    "select_qsa_dcp_topk",
]
