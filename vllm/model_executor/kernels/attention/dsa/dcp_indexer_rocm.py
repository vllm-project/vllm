# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm DCP candidates ordered by descending FP32 score, then ascending token id."""

import math

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    dcp_global_to_local_index,
    dcp_local_to_global_index,
    get_dcp_local_seq_lens,
)

FP8_DTYPE = current_platform.fp8_dtype()


@triton.jit
def _pack_candidates_kernel(
    logits,
    indices,
    row_starts,
    output,
    rank,
    rows: tl.constexpr,
    k: tl.constexpr,
    width: tl.constexpr,
    logits_stride0: tl.constexpr,
    logits_stride1: tl.constexpr,
    indices_stride0: tl.constexpr,
    indices_stride1: tl.constexpr,
    starts_stride: tl.constexpr,
    world_size: tl.constexpr,
    interleave: tl.constexpr,
    has_offsets: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // k, offsets % k
    mask = row < rows
    local = tl.load(
        indices + row * indices_stride0 + col * indices_stride1,
        mask=mask,
        other=-1,
    ).to(tl.int64)
    column = local
    if has_offsets:
        column += tl.load(row_starts + row * starts_stride, mask=mask, other=0)
    valid = mask & (local >= 0) & (column >= 0) & (column < width)
    score = tl.load(
        logits + row * logits_stride0 + column * logits_stride1,
        mask=valid,
        other=float("nan"),
    )
    valid &= tl.abs(score) < float("inf")
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = tl.where(bits < 0, ~bits, bits ^ 0x80000000) - 0x80000000
    global_ids = dcp_local_to_global_index(local, rank, world_size, interleave)
    packed = (ordered << 32) | (0xFFFFFFFF - global_ids)
    tl.store(output + offsets, tl.where(valid, packed, -9223372036854775808), mask)


def pack_dcp_candidates(
    logits: torch.Tensor,
    indices: torch.Tensor,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack each local candidate into one sortable int64 collective element."""
    assert logits.dtype == torch.float32 and indices.dtype == torch.int32
    assert 0 <= rank < world_size and interleave > 0
    invalid = torch.iinfo(torch.int64).min
    if logits.shape[1] == 0:
        return torch.full_like(indices, invalid, dtype=torch.int64)
    output = torch.empty(indices.shape, dtype=torch.int64, device=indices.device)
    if output.numel() == 0:
        return output
    _pack_candidates_kernel[(triton.cdiv(output.numel(), 256),)](
        logits,
        indices,
        row_starts if row_starts is not None else logits,
        output,
        rank,
        *indices.shape,
        logits.shape[1],
        *logits.stride(),
        *indices.stride(),
        row_starts.stride(0) if row_starts is not None else 0,
        world_size,
        interleave,
        row_starts is not None,
        BLOCK=256,
    )
    return output


@triton.jit
def _decode_selected_candidates(
    Keys,
    Output,
    Width: tl.constexpr,
    Count: tl.constexpr,
    KS: tl.constexpr,
    OS: tl.constexpr,
    OCOL: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    keys = tl.load(Keys + row * KS + col, col < Count, other=-9223372036854775808)
    ids = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
    ids = tl.where(keys != -9223372036854775808, ids, -1)
    tl.store(Output + row * OS + col * OCOL, ids, col < Width)


def select_dcp_candidates(gathered: torch.Tensor, out: torch.Tensor) -> None:
    """Select global candidates without floating-point conversion of token ids.

    The selected set follows the packed (score, token id) order, but the ids are
    written in no particular order within a row.
    """
    if not out.numel():
        return
    count = min(out.shape[1], gathered.shape[1])
    if count == 0:
        out.fill_(-1)
        return
    keys = gathered.topk(count, dim=-1, sorted=False).values
    _decode_selected_candidates[(out.shape[0], triton.cdiv(out.shape[1], 256))](
        keys,
        out,
        out.shape[1],
        count,
        keys.stride(0),
        out.stride(0),
        out.stride(1),
        B=256,
    )


# --- Global-key prefill: all-gather every rank's index keys for one chunk ----


@triton.jit
def _pack_local_keys(
    Cache,
    Table,
    Cu,
    Packed,
    cache_page_bytes,
    table_stride,
    PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    """Copy this rank's index keys out of the cache into one row per token.

    A cache page holds ``PAGE`` value rows of ``HEAD_DIM`` bytes followed by
    ``PAGE`` four-byte scales, the layout
    ``cp_gather_indexer_k_quant_cache_triton`` reads. Values and scale travel
    together so the all-gather below moves one buffer instead of two.
    """
    req = tl.program_id(0)
    row = tl.program_id(1) * 16 + tl.arange(0, 16)
    begin = tl.load(Cu + req)
    length = tl.load(Cu + req + 1) - begin
    page = tl.load(Table + req * table_stride + row // PAGE, row < length, 0).to(
        tl.int64
    )
    slot = row % PAGE
    byte = tl.arange(0, TILE)
    offset = tl.where(
        byte[None, :] < HEAD_DIM,
        slot[:, None] * HEAD_DIM + byte[None, :],
        PAGE * HEAD_DIM + slot[:, None] * 4 + byte[None, :] - HEAD_DIM,
    )
    mask = (row[:, None] < length) & (byte[None, :] < WIDTH)
    value = tl.load(Cache + page[:, None] * cache_page_bytes + offset, mask, 0)
    tl.store(
        Packed + (begin + row[:, None]).to(tl.int64) * WIDTH + byte[None, :],
        value,
        mask,
    )


@triton.jit(do_not_specialize=["Total", "MaxLocal"])
def _restore_global_keys(
    Packed,
    TokenToSeq,
    GlobalCu,
    AllLocalCu,
    Keys,
    Scales,
    Total,
    MaxLocal,
    World: tl.constexpr,
    Interleave: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Undo the DCP interleave: rank-major gathered rows to token order."""
    token = tl.program_id(0) * 16 + tl.arange(0, 16)
    req = tl.load(TokenToSeq + token, token < Total, 0)
    pos = token - tl.load(GlobalCu + req)
    owner, local = dcp_global_to_local_index(pos, World, Interleave)
    local += tl.load(AllLocalCu + req * World + owner)
    source = owner.to(tl.int64) * MaxLocal + local
    byte = tl.arange(0, HEAD_DIM)
    value = tl.load(
        Packed + source[:, None] * WIDTH + byte[None, :], token[:, None] < Total, 0
    )
    tl.store(
        Keys + token[:, None].to(tl.int64) * HEAD_DIM + byte[None, :],
        value,
        token[:, None] < Total,
    )
    b = tl.arange(0, 4)
    scale = tl.load(
        Packed + source[:, None] * WIDTH + HEAD_DIM + b[None, :],
        token[:, None] < Total,
        0,
    )
    tl.store(
        Scales + token[:, None].to(tl.int64) * 4 + b[None, :],
        scale,
        token[:, None] < Total,
    )


def gather_global_indexer_keys(
    kv_cache: torch.Tensor,
    chunk,
    dcp_world_size: int,
    cp_interleave: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank's index keys for one prefill chunk, in global token order.

    Returns ``(keys, scales)`` of shape ``[chunk.total_seq_lens, head_dim]``
    and ``[chunk.total_seq_lens]``: the whole context of every request in the
    chunk, so a row scored against them needs no candidate merge afterwards.
    """
    from vllm.distributed import get_dcp_group

    width = head_dim + 4
    page = kv_cache.shape[1]
    assert kv_cache.shape[2] == width, (
        f"indexer cache row is {kv_cache.shape[2]} bytes, expected {width}"
    )
    device = kv_cache.device
    seq_lens = chunk.cu_seq_lens[1:] - chunk.cu_seq_lens[:-1]
    counts = get_dcp_local_seq_lens(seq_lens, dcp_world_size, None, cp_interleave)
    all_cu = torch.cat(
        (counts.new_zeros((1, dcp_world_size)), counts.cumsum(0, dtype=torch.int32))
    )
    # The per-rank row count is the stride into the gathered buffer, so both
    # the allocation and the restore below must use the same (padded) value.
    rows = chunk.max_local_total_seq_lens
    if envs.VLLM_ROCM_DSA_DCP_PAD_KEY_GATHER:
        # AITER's IPC all-gather only takes a payload whose byte count is a
        # multiple of 16. Rounding the rows up buys that for any width.
        align = 16 // math.gcd(width, 16)
        rows = triton.cdiv(rows, align) * align
    packed = torch.empty((rows, width), dtype=torch.uint8, device=device)
    local_width = (
        triton.cdiv(chunk.max_seqlen_k, dcp_world_size * cp_interleave) * cp_interleave
    )
    _pack_local_keys[chunk.num_reqs, max(1, triton.cdiv(local_width, 16))](
        kv_cache.view(torch.uint8),
        chunk.block_table,
        chunk.local_cu_seq_lens,
        packed,
        kv_cache.stride(0) * kv_cache.element_size(),
        chunk.block_table.stride(0),
        PAGE=page,
        HEAD_DIM=head_dim,
        WIDTH=width,
        TILE=triton.next_power_of_2(width),
    )
    gathered = get_dcp_group().all_gather(packed, dim=0)
    total = chunk.total_seq_lens
    keys = torch.empty((total, head_dim), dtype=FP8_DTYPE, device=device)
    scales = torch.empty((total,), dtype=torch.float32, device=device)
    _restore_global_keys[(triton.cdiv(total, 16),)](
        gathered,
        chunk.token_to_seq,
        chunk.cu_seq_lens,
        all_cu,
        keys.view(torch.uint8),
        scales.view(torch.uint8),
        total,
        rows,
        World=dcp_world_size,
        Interleave=cp_interleave,
        HEAD_DIM=head_dim,
        WIDTH=width,
    )
    return keys, scales
