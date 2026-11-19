# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-RoPE FP4 indexer quantization for the AITER row-group layout.

Each 128-element head has four UE8M0 scales and 64 E2M1 payload bytes.
Cache writes preserve the physical page stride of the merged MLA cache.
"""

import torch

from vllm.triton_utils import tl, triton


def decode_tile_size(max_model_len: int, width: int, topk: int, heads: int) -> int:
    """Bound logits, selection scratch and queries while grouping verify rows."""
    row_bytes = 4 * width + 24 * min(topk, width) + 64 * heads + 256
    return max(32, 32 * 4 * max_model_len // max(1, row_bytes))


def decode_layout(
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    num_rows: int,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    interleave: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group verify rows by request and bound scores in rank-local key space.

    seq_lens are global, uncompressed sequence lengths from indexer metadata.
    Graph padding remains outside the last query boundary, with zero key bounds.
    No device-to-host reads are needed, including for nonuniform verify lengths.
    """
    if not 0 <= dcp_rank < dcp_world_size or interleave < 1 or num_rows < 0:
        raise ValueError("Invalid DCP geometry or query row count")
    if seq_lens.ndim != 1 or query_lens.ndim != 1:
        raise ValueError("Sequence and query lengths must be one-dimensional")
    if seq_lens.numel() < query_lens.numel():
        raise ValueError("Each decode request requires a global sequence length")
    starts = torch.cat(
        (query_lens.new_zeros(1), query_lens.cumsum(0, dtype=torch.int32))
    ).to(torch.int32)
    rows = torch.arange(num_rows, device=query_lens.device, dtype=torch.int32)
    if query_lens.numel() == 0:
        return starts, torch.zeros_like(rows)
    request = torch.searchsorted(starts[1:], rows, right=True).clamp_max(
        query_lens.numel() - 1
    )
    ends = (
        seq_lens[request] - query_lens[request] + rows - starts[request] + 1
    ).clamp_min(0)
    if dcp_world_size > 1:
        full = ends // (dcp_world_size * interleave) * interleave
        ends = full + (ends - full * dcp_world_size - dcp_rank * interleave).clamp(
            0, interleave
        )
    return starts, torch.where(rows < starts[-1], ends, 0).to(torch.int32)


@triton.jit
def _quantize(values):
    values = tl.reshape(values, (4, 32))
    scale = tl.maximum(tl.max(tl.abs(values), 1) / 6.0, 1.0e-4)
    bits = scale.to(tl.int32, bitcast=True)
    exponent = tl.minimum(((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    x = values / scale[:, None]
    a = tl.abs(x)
    # E2M1 levels: 0, .5, 1, 1.5, 2, 3, 4, 6. Ties choose even codes.
    code = (a > 0.25).to(tl.uint8)
    code += (a >= 0.75).to(tl.uint8)
    code += (a > 1.25).to(tl.uint8)
    code += (a >= 1.75).to(tl.uint8)
    code += (a > 2.5).to(tl.uint8)
    code += (a >= 3.5).to(tl.uint8)
    code += (a > 5.0).to(tl.uint8)
    code |= ((x < 0) & (code != 0)).to(tl.uint8) << 3
    low, high = tl.split(tl.reshape(code, (64, 2)))
    return low | (high << 4), exponent.to(tl.uint8)


@triton.jit
def _quantize_query(
    Q, Payload, Scales, QS: tl.constexpr, QH: tl.constexpr, H: tl.constexpr
):
    row, head = tl.program_id(0), tl.program_id(1)
    group = tl.arange(0, 4)
    scale_offsets = row * 256 + group * 64 + (head % 16) * 4 + head // 16
    if head < H:
        values = tl.load(Q + row * QS + head * QH + tl.arange(0, 128)).to(tl.float32)
        packed, scales = _quantize(values)
        tl.store(Payload + (row * H + head) * 64 + tl.arange(0, 64), packed)
        tl.store(Scales + scale_offsets, scales)
    else:
        tl.store(Scales + scale_offsets, tl.full((4,), 0, tl.uint8))


@triton.jit
def _quantize_key(K, Cache, Slots, KS: tl.constexpr, CS: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot >= 0:
        values = tl.load(K + row * KS + tl.arange(0, 128)).to(tl.float32)
        packed, scales = _quantize(values)
        page, offset = slot // 64, slot % 64
        byte = tl.arange(0, 64)
        tl.store(
            Cache + page * CS + (byte // 16) * 1024 + offset * 16 + byte % 16, packed
        )
        # AITER's 64-slot scale axis is a transposed 16 x 4 tile.
        shuffled = (offset % 16) * 4 + offset // 16
        tl.store(Cache + page * CS + 4096 + tl.arange(0, 4) * 64 + shuffled, scales)


def quantize_queries(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [rows, heads, 128] without folding scales into head weights."""
    if q.ndim != 3 or q.shape[2] != 128 or q.stride(2) != 1:
        raise ValueError("FP4 index queries require contiguous 128-element heads")
    rows, heads, _ = q.shape
    if heads not in (16, 32, 64):
        raise ValueError("FP4 row-group queries support 16, 32 or 64 heads")
    payload = torch.empty((rows, heads, 64), device=q.device, dtype=torch.uint8)
    scales = torch.empty((rows, 1, 4, 16, 4), device=q.device, dtype=torch.uint8)
    if rows:
        _quantize_query[(rows, 64)](q, payload, scales, q.stride(0), q.stride(1), heads)
    return payload, scales


def cache_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """View index sections as payload/scales without copying across MLA pages."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, 68)
        or cache.stride(2) != 1
        or cache.stride(1) != 68
        or cache.stride(0) < 4352
    ):
        raise ValueError("FP4 index cache requires uint8 [pages, 64, 68] sections")
    pages, stride, start = cache.shape[0], cache.stride(0), cache.storage_offset()
    payload = cache.as_strided(
        (pages, 1, 4, 64, 16), (stride, 4096, 1024, 16, 1), start
    )
    scales = cache.as_strided((pages, 1, 4, 64), (stride, 256, 64, 1), start + 4096)
    return payload, scales


def store_keys(keys: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor) -> None:
    """Quantize normalized keys into merged cache pages; negative slots are padding."""
    cache_views(cache)
    if keys.ndim != 2 or keys.shape[1] != 128 or keys.stride(1) != 1:
        raise ValueError("FP4 index keys require contiguous 128-element rows")
    if slots.ndim != 1 or slots.numel() != keys.shape[0] or slots.stride(0) != 1:
        raise ValueError("Each key requires one contiguous slot entry")
    if keys.shape[0]:
        _quantize_key[(keys.shape[0],)](
            keys, cache, slots, keys.stride(0), cache.stride(0)
        )


def score_paged(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    query_starts: torch.Tensor,
    row_ends: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Score one bounded query tile using shared-key row-group FP4 MLA logits.

    Weights include the model's head/softmax factors, but not FP4 group scales.
    The AITER reader must support physical page strides for merged caches.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        flydsl_pa_mqa_logits_fp4_rowgroup,
        make_fp4_mqa_plan,
    )

    width = block_table.shape[1] * 64
    output = torch.empty(
        (query.shape[0], width), dtype=torch.float32, device=query.device
    )
    if not output.numel() or query_starts.numel() <= 1:
        return output
    payload, scales = cache_views(cache)
    q_payload, q_scales = quantize_queries(query)
    plan = make_fp4_mqa_plan(
        num_seqs=query_starts.numel() - 1,
        max_qlen=max_query_len,
        num_rows=query.shape[0],
        heads=query.shape[1],
        page_size=64,
        max_seq_len=width,
    )
    return flydsl_pa_mqa_logits_fp4_rowgroup(
        plan,
        q_payload,
        q_scales,
        payload,
        scales,
        block_table,
        weights,
        query_starts,
        row_ends,
        out=output,
    )
