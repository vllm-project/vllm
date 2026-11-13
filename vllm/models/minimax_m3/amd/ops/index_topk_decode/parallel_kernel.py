# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel decode top-16 with runtime shape metadata.

Merge widths cover the host's maximum 8192 score columns. Keeping
rows, strides, and columns dynamic avoids compiling new kernels when
mixed batches change their score layout during serving.
"""

import functools

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _flydsl_key(score, index, valid):
    bits = score.to(tl.int32, bitcast=True)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    ordered = tl.where(
        (ordered > 0x7F800000) | (ordered < -2139095041), -2139095041, ordered
    )
    key = (ordered.to(tl.int64) << 32) | index.to(tl.int64)
    return tl.where(valid, key, -9223372036854775807 - 1)


@triton.jit(do_not_specialize=["SH", "SR", "ROWS", "COLS", "PARTS", "OH", "OR"])
def _m3_parallel_topk_partial(
    score,
    lengths,
    partial,
    output,
    SH,
    SR,
    ROWS,
    COLS,
    QUERY: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS,
    OH,
    OR,
    ADAPTIVE: tl.constexpr,
):
    row, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = tl.load(lengths + row // QUERY)
    live = tl.minimum(COLS, tl.cdiv(tl.maximum(0, seq - QUERY + row % QUERY + 1), 128))
    if ADAPTIVE:  # noqa: SIM102
        if live <= 1024:
            if part != 0:
                return
            if live <= 128:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 128)
            elif live <= 512:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 512)
            else:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 1024)
            rank = tl.arange(0, 16)
            indices = tl.where(rank < live, winners & 0xFFFFFFFF, -1).to(tl.int32)
            tl.store(output + head * OH + row * OR + rank, indices)
            return
    if part * CHUNK >= live:
        return
    index = part * CHUNK + tl.arange(0, CHUNK)
    valid = index < live
    values = tl.load(score + head * SH + row * SR + index, mask=valid, other=0)
    winners = tl.topk(_flydsl_key(values, index, valid), 16)
    out = ((head * ROWS + row) * PARTS + part) * 16 + tl.arange(0, 16)
    tl.store(partial + out, winners)


@triton.jit(do_not_specialize=["OH", "OR", "ROWS", "COLS", "PARTS"])
def _m3_parallel_topk_merge(
    partial,
    lengths,
    output,
    OH,
    OR,
    ROWS,
    COLS,
    QUERY: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS,
    MERGE: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    seq = tl.load(lengths + row // QUERY)
    live = tl.minimum(COLS, tl.cdiv(tl.maximum(0, seq - QUERY + row % QUERY + 1), 128))
    if ADAPTIVE:  # noqa: SIM102
        if live <= 1024:
            return
    offset = tl.arange(0, MERGE)
    keys = tl.load(
        partial + (head * ROWS + row) * PARTS * 16 + offset,
        mask=(offset < PARTS * 16) & (offset // 16 * CHUNK < live),
        other=-9223372036854775807 - 1,
    )
    winners = tl.topk(keys, 16)
    rank = tl.arange(0, 16)
    indices = tl.where(rank < live, winners & 0xFFFFFFFF, -1).to(tl.int32)
    tl.store(output + head * OH + row * OR + rank, indices)


@triton.jit(do_not_specialize=["SH", "SR"])
def _m3_row_topk(score, head, row, live, SH, SR, WIDTH: tl.constexpr):
    index = tl.arange(0, WIDTH)
    values = tl.load(score + head * SH + row * SR + index, mask=index < live, other=0)
    return tl.topk(_flydsl_key(values, index, index < live), 16)


@triton.jit(do_not_specialize=["SH", "SR", "OH", "OR", "ROWS", "COLS", "PARTS"])
def _m3_parallel_topk_fused(
    score,
    lengths,
    partial,
    counters,
    output,
    SH,
    SR,
    OH,
    OR,
    ROWS,
    COLS,
    QUERY: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS,
    MERGE: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
):
    row, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = tl.load(lengths + row // QUERY)
    live = tl.minimum(COLS, tl.cdiv(tl.maximum(0, seq - QUERY + row % QUERY + 1), 128))
    if ADAPTIVE:  # noqa: SIM102 -- remove unused branches at compile time
        if live <= 1024:
            if part != 0:
                return
            if live <= 128:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 128)
            elif live <= 512:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 512)
            else:
                winners = _m3_row_topk(score, head, row, live, SH, SR, 1024)
            rank = tl.arange(0, 16)
            indices = tl.where(rank < live, winners & 0xFFFFFFFF, -1).to(tl.int32)
            tl.store(output + head * OH + row * OR + rank, indices)
            return
    active = tl.maximum(1, tl.cdiv(live, CHUNK))
    if part >= active:
        return
    index = part * CHUNK + tl.arange(0, CHUNK)
    valid = index < live
    values = tl.load(score + head * SH + row * SR + index, mask=valid, other=0)
    winners = tl.topk(_flydsl_key(values, index, valid), 16)
    rank = tl.arange(0, 16)
    if active == 1:
        indices = tl.where(rank < live, winners & 0xFFFFFFFF, -1).to(tl.int32)
        tl.store(output + head * OH + row * OR + rank, indices)
        return
    out = ((head * ROWS + row) * PARTS + part) * 16 + rank
    tl.store(partial + out, winners)
    # The release publishes every CTA's partial; the last arrival acquires them.
    counter = counters + (head * ROWS + row) * COUNTER_STRIDE
    arrival = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrival != active - 1:
        return
    offset = tl.arange(0, MERGE)
    keys = tl.load(
        partial + (head * ROWS + row) * PARTS * 16 + offset,
        mask=offset < active * 16,
        other=-9223372036854775807 - 1,
        volatile=True,
    )
    winners = tl.topk(keys, 16)
    indices = tl.where(rank < live, winners & 0xFFFFFFFF, -1).to(tl.int32)
    tl.store(output + head * OH + row * OR + rank, indices)
    tl.atomic_xchg(counter, 0, sem="acq_rel", scope="gpu")


@functools.cache
def _buffers(heads, rows, cols, device_index, stream_id):
    # Graphs retain these addresses. Each DBO stream owns separate counters.
    device = torch.device("cuda", device_index)
    chunk = 512 if heads * rows <= 16 else 256
    return (
        torch.empty(
            (heads, rows, triton.cdiv(cols, chunk), 16),
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros((heads, rows), dtype=torch.int32, device=device),
    )


def parallel_decode_topk(score, output, lengths, query_len, cols):
    heads, rows, _ = score.shape
    stream = torch.cuda.current_stream(score.device)
    scratch, counters = _buffers(
        heads, rows, cols, score.device.index, stream.cuda_stream
    )
    if heads * rows <= 16:
        parts = triton.cdiv(cols, 512)
        _m3_parallel_topk_fused[(rows, heads, parts)](
            score,
            lengths,
            scratch,
            counters,
            output,
            score.stride(0),
            score.stride(1),
            output.stride(0),
            output.stride(1),
            rows,
            cols,
            query_len,
            512,
            parts,
            256,
            True,
            1,
            num_warps=4,
        )
        return
    parts = triton.cdiv(cols, 256)
    _m3_parallel_topk_partial[(rows, heads, parts)](
        score,
        lengths,
        scratch,
        output,
        score.stride(0),
        score.stride(1),
        rows,
        cols,
        query_len,
        256,
        parts,
        output.stride(0),
        output.stride(1),
        True,
        num_warps=4,
    )
    _m3_parallel_topk_merge[(rows, heads)](
        scratch,
        lengths,
        output,
        output.stride(0),
        output.stride(1),
        rows,
        cols,
        query_len,
        256,
        parts,
        512,
        True,
        num_warps=4,
    )
