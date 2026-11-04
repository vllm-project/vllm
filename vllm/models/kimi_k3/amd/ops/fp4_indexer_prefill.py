# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 prefill: gather compact index keys once, then score global causal rows."""

from dataclasses import dataclass, field

import torch

from vllm.distributed import get_dcp_group
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerPrefillMetadata
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from .fp4_indexer import quantize_queries


@dataclass(frozen=True)
class FP4QueryPlan:
    start: int
    end: int
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    batches: torch.Tensor
    schedule: torch.Tensor
    num_ctas: int


@dataclass(frozen=True)
class FP4ChunkPlan:
    all_cu: torch.Tensor
    page_table: torch.Tensor
    first: int
    per_rank: int
    queries: list[FP4QueryPlan]


@dataclass
class FP4PrefillMetadata(DeepseekV32IndexerPrefillMetadata):
    plans: list[FP4ChunkPlan] = field(default_factory=list)


def _local_key_cu(chunk, world, interleave):
    lengths = chunk.cu_seq_lens[1:] - chunk.cu_seq_lens[:-1]
    counts = get_dcp_local_seq_lens(lengths, world, None, interleave)
    return torch.cat(
        (counts.new_zeros((1, world)), counts.cumsum(0, dtype=torch.int32))
    )


def _prepare_chunk(chunk, world, rank, interleave):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    device = chunk.cu_seq_lens.device
    pages = triton.cdiv(chunk.total_seq_lens, 64)
    width = pages * 64
    # The scorer prefetches page IDs one block beyond the final iteration.
    table = torch.zeros(
        (1, triton.cdiv(pages, 4) * 4 + 4), device=device, dtype=torch.int32
    )
    table[0, :pages] = torch.arange(pages, device=device, dtype=torch.int32)
    ks, ke = (
        (chunk.global_cu_seqlen_ks, chunk.global_cu_seqlen_ke)
        if world > 1
        else (chunk.cu_seqlen_ks, chunk.cu_seqlen_ke)
    )
    assert ks is not None and ke is not None
    per_rank = triton.cdiv(chunk.token_end - chunk.token_start, world)
    first = chunk.token_start + rank * per_rank
    last = min(first + per_rank, chunk.token_end)
    row_chunk = max(1, min(8192, (1024**3 - 1) // (4 * width)))
    if row_chunk >= 128:
        row_chunk = row_chunk // 128 * 128
    queries = []
    for start in range(first, last, row_chunk):
        end = min(start + row_chunk, last)
        lo, hi = start - chunk.token_start, end - chunk.token_start
        starts, ends = ks[lo:hi], ke[lo:hi]
        batches = torch.zeros(end - start, device=device, dtype=torch.int32)
        _, schedule, num_ctas = compute_prefill_schedule(
            batches, starts, ends, 256, max(4096, end - start), width
        )
        queries.append(
            FP4QueryPlan(start, end, starts, ends, batches, schedule, num_ctas)
        )
    return FP4ChunkPlan(
        _local_key_cu(chunk, world, interleave), table, first, per_rank, queries
    )


def prepare_prefill_metadata(metadata, world, rank, interleave):
    """Build read-only geometry once for the layers sharing this attention group."""
    return FP4PrefillMetadata(
        metadata.chunks,
        metadata.max_prefill_seq_len,
        [_prepare_chunk(chunk, world, rank, interleave) for chunk in metadata.chunks],
    )


@triton.jit
def _pack_keys(
    Cache,
    Table,
    LocalCu,
    Packed,
    CS: tl.constexpr,
    TS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(LocalCu + req)
    length = tl.load(LocalCu + req + 1) - begin
    page = tl.load(Table + req * TS + row // 64, row < length, 0).to(tl.int64)
    slot = row % 64
    byte = tl.arange(0, 128)
    offset = tl.where(
        byte < 64,
        byte // 16 * 1024 + slot[:, None] * 16 + byte % 16,
        4096 + (byte - 64) * 64 + (slot[:, None] % 16) * 4 + slot[:, None] // 16,
    )
    value = tl.load(
        Cache + page[:, None] * CS + offset,
        (row[:, None] < length) & (byte[None, :] < 68),
        0,
    )
    tl.store(
        Packed + (begin + row[:, None]).to(tl.int64) * 68 + byte[None, :],
        value,
        (row[:, None] < length) & (byte[None, :] < 68),
    )


@triton.jit(do_not_specialize=["Total", "MaxLocal"])
def _stage_keys(
    Packed,
    TokenToSeq,
    GlobalCu,
    AllLocalCu,
    Payload,
    Scales,
    Total,
    MaxLocal,
    World: tl.constexpr,
    Interleave: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(TokenToSeq + token, token < Total, 0)
    begin = tl.load(GlobalCu + req)
    pos = token - begin
    owner = pos // Interleave % World
    local = pos // (Interleave * World) * Interleave + pos % Interleave
    local += tl.load(AllLocalCu + req * World + owner)
    source = owner.to(tl.int64) * MaxLocal + local
    byte = tl.arange(0, 64)
    value = tl.load(
        Packed + source[:, None] * 68 + byte[None, :], token[:, None] < Total, 0
    )
    page, slot = token // 64, token % 64
    target = (
        page[:, None].to(tl.int64) * 4096
        + byte[None, :] // 16 * 1024
        + slot[:, None] * 16
        + byte[None, :] % 16
    )
    tl.store(Payload + target, value, token[:, None] < Total)
    group = tl.arange(0, 4)
    scale = tl.load(
        Packed + source[:, None] * 68 + 64 + group[None, :], token[:, None] < Total, 0
    )
    target_scale = (
        page[:, None].to(tl.int64) * 256
        + group[None, :] * 64
        + slot[:, None] % 16 * 4
        + slot[:, None] // 16
    )
    tl.store(Scales + target_scale, scale, token[:, None] < Total)


def prefill_workspace_bytes(max_keys, max_reqs, world, interleave):
    """Peak prefill scratch of one chunk holding at most ``max_keys`` keys.

    The indexer builder caps each chunk's keys at its prefill buffer size and
    _prepare_chunk keeps each logits tile under 1 GiB. Staging holds the local
    packed keys, their DCP all-gather and the restored pages at once; a rank's
    share exceeds an even split by at most one interleave run per request.
    """
    pages = triton.cdiv(max_keys, 64) * 64
    if world == 1:
        return 1024**3 + 68 * (max_keys + pages)
    local = triton.cdiv(max_keys, world) + max_reqs * interleave
    return 1024**3 + 68 * ((world + 1) * local + pages)


def stage_keys(cache, chunk, world, interleave, all_cu=None):
    """Preserve quantized bytes and swizzled scale rows, including merged strides."""
    device = cache.device
    cu = chunk.cu_seq_lens
    if all_cu is None:
        all_cu = _local_key_cu(chunk, world, interleave)
    max_local = chunk.max_local_total_seq_lens if world > 1 else chunk.total_seq_lens
    packed = torch.empty((max_local, 68), device=device, dtype=torch.uint8)
    local_width = (
        (chunk.max_seqlen_k + world * interleave - 1)
        // (world * interleave)
        * interleave
    )
    _pack_keys[(chunk.num_reqs, triton.cdiv(local_width, 32))](
        cache,
        chunk.block_table,
        chunk.local_cu_seq_lens if world > 1 else chunk.cu_seq_lens,
        packed,
        cache.stride(0),
        chunk.block_table.stride(0),
        32,
    )
    gathered = get_dcp_group().all_gather(packed, dim=0) if world > 1 else packed
    pages = triton.cdiv(chunk.total_seq_lens, 64)
    payload = torch.zeros((pages, 1, 4, 64, 16), device=device, dtype=torch.uint8)
    scales = torch.zeros((pages, 1, 4, 64), device=device, dtype=torch.uint8)
    _stage_keys[(triton.cdiv(chunk.total_seq_lens, 32),)](
        gathered,
        chunk.token_to_seq,
        cu,
        all_cu,
        payload,
        scales,
        chunk.total_seq_lens,
        max_local,
        world,
        interleave,
        32,
    )
    return payload, scales


def prefill_indexer(
    q,
    weights,
    cache,
    out,
    chunk,
    world,
    interleave,
    weight_scale=1.0,
    staged_keys=None,
    pending=None,
    plan=None,
):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_logits_fp4_prefill,
    )
    from aiter.ops.topk import top_k_per_row_prefill

    assert weights.dtype == torch.bfloat16
    if plan is None:
        rank = get_dcp_group().rank_in_group if world > 1 else 0
        plan = _prepare_chunk(chunk, world, rank, interleave)
    payload, scales = (
        stage_keys(cache, chunk, world, interleave, plan.all_cu)
        if staged_keys is None
        else staged_keys
    )
    width = payload.shape[0] * 64
    count = chunk.token_end - chunk.token_start
    local_out = torch.full(
        (plan.per_rank, out.shape[1]), -1, dtype=out.dtype, device=out.device
    )
    for query in plan.queries:
        start, end = query.start, query.end
        starts, ends = query.row_starts, query.row_ends
        qp, qs = quantize_queries(q[start:end])
        logits = torch.empty((end - start, width), device=q.device, dtype=torch.float32)
        flydsl_pa_mqa_logits_fp4_prefill(
            qp,
            qs,
            payload,
            scales,
            plan.page_table,
            weights[start:end],
            query.batches,
            starts,
            ends,
            width,
            weight_scale=weight_scale,
            cta_info=query.schedule,
            n_ctas=query.num_ctas,
            out=logits,
        )
        selected = local_out[start - plan.first : end - plan.first]
        top_k_per_row_prefill(
            logits=logits,
            rowStarts=starts,
            rowEnds=ends,
            indices=selected,
            values=None,
            numRows=end - start,
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            # DCP assigns each query to one rank and shares its selected IDs.
            # Only replicated queries need identical tie-breaking on each rank.
            stable=world == 1,
            k=out.shape[1],
        )
        # The public Kimi indexer output uses sequence-relative global IDs.
        selected.copy_(torch.where(selected >= 0, selected - starts[:, None], -1))
    if pending is not None:
        pending.append((chunk.token_start, chunk.token_end, local_out))
        return
    selected = get_dcp_group().all_gather(local_out, dim=0) if world > 1 else local_out
    out[chunk.token_start : chunk.token_end].copy_(selected[:count])


def prefill_chunks(q, weights, cache, out, metadata, world, interleave, weight_scale):
    """Reuse each request set's keys and exchange all query chunks together."""
    gathered_keys = None
    pending: list[tuple[int, int, torch.Tensor]] = []
    for chunk, plan in zip(metadata.chunks, metadata.plans, strict=True):
        cache_key = (chunk.block_table.data_ptr(), chunk.num_reqs, chunk.total_seq_lens)
        if gathered_keys is None or gathered_keys[0] != cache_key:
            # Release the previous request set's pages before staging the next.
            gathered_keys = None
            gathered_keys = (
                cache_key,
                stage_keys(cache, chunk, world, interleave, plan.all_cu),
            )
        prefill_indexer(
            q,
            weights,
            cache,
            out,
            chunk,
            world,
            interleave,
            weight_scale,
            staged_keys=gathered_keys[1],
            pending=pending,
            plan=plan,
        )
    if not pending:
        return
    mine = (
        torch.cat([rows for _, _, rows in pending])
        if len(pending) > 1
        else pending[0][2]
    )
    selected = get_dcp_group().all_gather(mine, dim=0) if world > 1 else mine
    selected = selected.view(world, mine.shape[0], -1)
    offset = 0
    for start, end, rows in pending:
        count = rows.shape[0]
        out[start:end].copy_(
            selected[:, offset : offset + count].reshape(-1, out.shape[1])[
                : end - start
            ]
        )
        offset += count
