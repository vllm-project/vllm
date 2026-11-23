# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton launches of the DCP KV prefetch kernels, for their tests.

vLLM launches these kernels only directly, by their loaded handles
(`cuda_direct`). The Triton interpreter on CPU, and a device run that compares a
direct launch with Triton's, launch them through Triton instead, with the grids
and arguments below. The tests that run in a subprocess prepend this file's
source to their script.
"""

from collections.abc import Sequence

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops import kv_ce_rows, kv_ce_topup
from vllm.v1.attention.ops import kv_ce_sel_union as su
from vllm.v1.attention.ops import pcp_dcp_k_replica as kr


def token_slots(restore_idx: torch.Tensor, slots: torch.Tensor, start: int) -> None:
    """Enqueue `_token_slots_kernel` on the current stream for the tokens
    `restore_idx` covers. `slots` (int64) holds -1 where no token may map."""
    rows = restore_idx.shape[0]
    assert restore_idx.dtype == slots.dtype == torch.int64
    assert restore_idx.is_contiguous() and slots.is_contiguous()
    if rows == 0:
        return
    kr._token_slots_kernel[(triton.cdiv(rows, kr.SLOT_BLOCK),)](
        restore_idx, slots, start, rows, BLOCK=kr.SLOT_BLOCK
    )


def stage_slots(
    restore_idx: torch.Tensor | None,
    slots: torch.Tensor,
    context_start: int,
    interleave: int,
    dcp_size: int,
    per_rank: int,
    new_rows: int | None = None,
) -> None:
    """Enqueue `_stage_slots_kernel` for the tokens `restore_idx` covers (or
    `new_rows` tokens in identity order when `restore_idx` is None), on the
    current stream. `slots` (int64) holds -1 where no token may map."""
    if restore_idx is not None:
        n = restore_idx.shape[0]
        assert restore_idx.dtype == slots.dtype == torch.int64
        assert restore_idx.is_contiguous() and slots.is_contiguous()
        ptr_arg = restore_idx
        has_restore = True
    else:
        assert new_rows is not None
        n = new_rows
        assert slots.dtype == torch.int64 and slots.is_contiguous()
        ptr_arg = slots
        has_restore = False
    if n == 0:
        return
    kv_ce_rows._stage_slots_kernel[(triton.cdiv(n, kv_ce_rows.SLOT_BLOCK),)](
        ptr_arg,
        slots,
        context_start,
        n,
        interleave,
        dcp_size,
        per_rank,
        BLOCK=kv_ce_rows.SLOT_BLOCK,
        HAS_RESTORE=has_restore,
    )


def stage_rows(
    cache: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    region: torch.Tensor,
    max_count: int,
) -> None:
    """Enqueue `_stage_rows_kernel` on the current stream: one program column per
    entry of `counts`, enough blocks for `max_count` rows (host-known; at least
    every count)."""
    assert cache.dtype == torch.uint8 and cache.is_contiguous()
    assert cache.shape[-1] == kv_ce_rows.FP8_ENTRY
    assert sources.dtype == targets.dtype == counts.dtype == torch.int32
    assert sources.numel() == targets.numel() == counts.numel()
    assert region.dtype == torch.bfloat16 and region.shape[1] == kv_ce_rows.ROW_DIM
    if max_count <= 0:
        return
    kv_ce_rows._stage_rows_kernel[
        (triton.cdiv(max_count, kv_ce_rows.STAGE_ROWS), counts.numel())
    ](cache, sources, targets, counts, region, ROWS=kv_ce_rows.STAGE_ROWS)


def plan_check(words: torch.Tensor, deltas: torch.Tensor, slot: int) -> None:
    """Enqueue `_plan_check_kernel` on the current stream. `words` (int64) are
    this rank's plan words and `deltas` (int64) the byte offsets of every rank's
    words from them."""
    assert words.dtype == deltas.dtype == torch.int64
    world = deltas.numel()
    kv_ce_rows._plan_check_kernel[(1,)](
        words.view(torch.uint8),
        deltas,
        slot,
        WORLD=world,
        BLOCK=triton.next_power_of_2(world),
    )


def receive_rows(
    stage: torch.Tensor,
    deltas: torch.Tensor,
    ends: torch.Tensor,
    slot_ids: torch.Tensor,
    part_ids: torch.Tensor,
    copied: Sequence[int],
    limit: int,
    stage_rows: int,
    region: torch.Tensor,
    shard_rows: int,
) -> None:
    """Enqueue, on the current stream, the upconvert of a compact head's rows
    (`kv_ce_topup.launch_receive`, with tensors for the buffers it binds)."""
    sources = len(copied)
    assert 0 < sources <= kv_ce_topup.MAX_SOURCES
    assert deltas.dtype == torch.int64 and deltas.numel() == sources
    assert slot_ids.dtype == part_ids.dtype == ends.dtype == torch.int32
    assert slot_ids.numel() == part_ids.numel() == sources
    assert stage.dim() == 2 and stage.is_contiguous()
    assert stage.dtype == torch.uint8 and stage.shape[1] == kv_ce_topup.FP8_ENTRY
    assert region.dtype == torch.bfloat16 and region.shape[1] == kv_ce_topup.ROW_DIM
    blocks = kv_ce_topup.receive_blocks(copied, limit)
    padded = [int(rows) for rows in copied] + [0] * (kv_ce_topup.MAX_SOURCES - sources)
    kv_ce_topup._receive_kernel[(blocks, sources)](
        stage,
        deltas,
        ends,
        slot_ids,
        part_ids,
        region,
        *padded,
        limit,
        stage_rows,
        shard_rows,
        blocks * kv_ce_topup.RECEIVE_ROWS,
        ROWS=kv_ce_topup.RECEIVE_ROWS,
        num_warps=kv_ce_topup.RECEIVE_WARPS,
    )


def mark_selection(
    p: torch.Tensor, mask: torch.Tensor, shard_rows: int, head_rows: int, epoch: int
) -> None:
    """Tag the old rows of `p` with `epoch` in `mask`."""
    tokens, k = p.shape
    if tokens == 0:
        return
    assert p.is_contiguous() and p.dtype == torch.int32
    assert 0 < epoch <= su.EPOCHS
    block = min(su.MARK_BLOCK, triton.next_power_of_2(k))
    su._mark_kernel[(tokens, triton.cdiv(k, block))](
        p, mask, shard_rows, head_rows, epoch, K=k, BLOCK=block
    )


def union_marks(
    own: torch.Tensor,
    peers: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    n: int,
    rank: int,
    world_size: int,
    epoch: int,
) -> None:
    """OR this rank's marks with the local copies of the peers' (row `p` of `peers`
    for peer `p`; this rank's row is not read)."""
    assert peers.dim() == 2 and peers.shape[0] >= world_size and peers.stride(1) == 1
    su._union_count_kernel[(triton.cdiv(n, su.SCAN_BLOCK),)](
        own,
        peers,
        peers.stride(0),
        union,
        counts,
        n,
        rank,
        epoch,
        WORLD=world_size,
        BLOCK=su.SCAN_BLOCK,
    )


def scan_union(
    union: torch.Tensor,
    counts: torch.Tensor,
    cum: torch.Tensor,
    ends: torch.Tensor,
    world_size: int,
    head_rows: int,
) -> None:
    """Number the union rows with one flat scan over all slots (`counts` from
    `union_marks`); `ends[r + 1]` is the scan at the end of slot `r`."""
    n = world_size * head_rows
    su._union_scan_kernel[(triton.cdiv(n, su.SCAN_BLOCK),)](
        union,
        counts,
        cum,
        ends,
        n,
        head_rows,
        WORLD=world_size,
        BLOCK=su.SCAN_BLOCK,
        CHUNK=su.SCAN_CHUNK,
    )


def remap_topk(
    p: torch.Tensor,
    cum: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
    shard_rows: int,
    head_rows: int,
) -> None:
    tokens, k = p.shape
    if tokens == 0:
        return
    assert p.is_contiguous() and out.is_contiguous()
    block = min(su.MARK_BLOCK, triton.next_power_of_2(k))
    su._remap_kernel[(tokens, triton.cdiv(k, block))](
        p, cum, ends, out, shard_rows, head_rows, K=k, BLOCK=block
    )


def pack_rows(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    union: torch.Tensor,
    cum: torch.Tensor,
    ends: torch.Tensor,
    rank: int,
    head_rows: int,
    wire: torch.Tensor,
    region_slot: torch.Tensor,
) -> None:
    """Copy the raw cache entries of the rows marked in `union` for `rank`'s slot
    (rows `[rank * head_rows, (rank + 1) * head_rows)`) into rows `cum - ends[rank]`
    of `wire` (one entry per row), and upconvert them into the same rows of
    `region_slot`; one launch."""
    cache = kv_cache.view(torch.uint8)
    entry = wire.shape[1]
    assert cache.dim() == 3 and cache.stride(2) == 1 and cache.shape[2] >= entry
    assert wire.dtype == torch.uint8 and wire.is_contiguous() and entry % 4 == 0
    assert cache.stride(0) % 4 == 0 and cache.stride(1) % 4 == 0
    assert block_table.dim() == 2 and block_table.shape[0] == 1
    assert region_slot.shape[1] == su.ROW_DIM
    words = entry // 4
    su._pack_kernel[(triton.cdiv(head_rows, su.PACK_ROWS),)](
        cache,
        cache.view(torch.int32),
        block_table,
        workspace_starts,
        union,
        cum,
        ends,
        rank,
        wire.view(torch.int32),
        region_slot,
        rank * head_rows,
        head_rows,
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        ROWS=su.PACK_ROWS,
        WORDS=words,
        BLOCK=triton.next_power_of_2(words),
    )
