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


__all__ = [
    "merge_qsa_dcp_topk",
    "pack_qsa_dcp_candidates",
    "select_qsa_dcp_topk",
]
