# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coarse IndexPool mathematics, independent of cache allocation.

Every decode step runs these per DSA layer, so each stage is a single Triton
kernel instead of a chain of eager elementwise ops.
"""

from typing import Any

import torch

from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import round_up

INDEX_DIM = 128
_ROPE_DIM = 64
_HALF = 32


@triton.jit
def _butterfly(x, BM: tl.constexpr, S: tl.constexpr):
    y = tl.permute(tl.reshape(x, (BM, 16 // S, 2, S)), (0, 1, 3, 2))
    left, right = tl.split(y)
    y = tl.permute(tl.join(left + right, left - right), (0, 1, 3, 2))
    return tl.reshape(y, (BM, 32))


@triton.jit
def _fwht32(x, BM: tl.constexpr):
    """Hadamard butterflies with strides 1..16 inside each 32-wide row."""
    x = _butterfly(x, BM, 1)
    x = _butterfly(x, BM, 2)
    x = _butterfly(x, BM, 4)
    x = _butterfly(x, BM, 8)
    return _butterfly(x, BM, 16)


@triton.jit
def _rope_frequency(
    Pos,
    Inv,
    token,
    live,
    POS_PLANE,
    POS_STRIDE,
    THREE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    j = tl.arange(0, 32)
    inv = tl.load(Inv + j)
    t = tl.load(Pos + token * POS_STRIDE, live, 0).to(tl.float32)
    frequency = t[:, None] * inv[None, :]
    if THREE:
        h = tl.load(Pos + POS_PLANE + token * POS_STRIDE, live, 0).to(tl.float32)
        w = tl.load(Pos + 2 * POS_PLANE + token * POS_STRIDE, live, 0).to(tl.float32)
        if INTERLEAVED:
            is_h = (j % 3 == 1) & (j < S1 * 3)
            is_w = (j % 3 == 2) & (j < S2 * 3)
        else:
            is_h = (j >= S0) & (j < S0 + S1)
            is_w = j >= S0 + S1
        frequency = tl.where(is_h[None, :], h[:, None] * inv[None, :], frequency)
        frequency = tl.where(is_w[None, :], w[:, None] * inv[None, :], frequency)
    return frequency


@triton.jit
def _rotate_fwht_quantize(a0, a1, even, odd, frequency, BM: tl.constexpr):
    """RoPE the last 64 lanes, then FWHT and power-of-two FP8 quantize.

    Inputs are fp32 holding BF16 values: `a0`/`a1` are lanes 0..63 and
    `even`/`odd` the interleaved rotary pairs. The butterflies run in the
    same order as the reference radix-2 transform, so results are bitwise
    identical to it.
    """
    cos = tldevice.cos(frequency)
    sin = tldevice.sin(frequency)
    r0 = (even * cos + -odd * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (odd * cos + even * sin).to(tl.bfloat16).to(tl.float32)
    a0 = _fwht32(a0, BM)
    a1 = _fwht32(a1, BM)
    r0 = _fwht32(r0, BM)
    r1 = _fwht32(r1, BM)
    c0, c1 = a0 + a1, a0 - a1
    d0, d1 = r0 + r1, r0 - r1
    norm: tl.constexpr = 0.08838834764831845
    y0 = (c0 + d0) * norm
    y1 = (c1 + d1) * norm
    y2 = (c0 - d0) * norm
    y3 = (c1 - d1) * norm
    amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(y0), 1), tl.max(tl.abs(y1), 1)),
        tl.maximum(tl.max(tl.abs(y2), 1), tl.max(tl.abs(y3), 1)),
    )
    ratio = tldevice.div_rn(tl.maximum(amax, 1e-4), 448.0)
    scale = tldevice.exp2(tldevice.ceil(tldevice.log2(ratio)))
    inverse = (1.0 / scale)[:, None]
    q0 = tl.clamp(y0 * inverse, -448.0, 448.0).to(tl.float8e4nv)
    q1 = tl.clamp(y1 * inverse, -448.0, 448.0).to(tl.float8e4nv)
    q2 = tl.clamp(y2 * inverse, -448.0, 448.0).to(tl.float8e4nv)
    q3 = tl.clamp(y3 * inverse, -448.0, 448.0).to(tl.float8e4nv)
    return q0, q1, q2, q3, scale


@triton.jit
def _store_quant(Out, row_offset, q0, q1, q2, q3, mask):
    col = tl.arange(0, 32)[None, :]
    base = Out + row_offset[:, None] + col
    mask = mask[:, None]
    tl.store(base, q0.to(tl.uint8, bitcast=True), mask)
    tl.store(base + 32, q1.to(tl.uint8, bitcast=True), mask)
    tl.store(base + 64, q2.to(tl.uint8, bitcast=True), mask)
    tl.store(base + 96, q3.to(tl.uint8, bitcast=True), mask)


@triton.jit
def _encoded_scale(scale, MX: tl.constexpr):
    bits = scale.to(tl.int32, bitcast=True)
    if MX:
        bits = (bits >> 23) * 0x01010101
    return bits


@triton.jit
def _quantize_query_kernel(
    Query,
    HeadWeights,
    Pos,
    Inv,
    Quant,
    Weights,
    tokens,
    QUERY_TOKEN,
    HEAD_WEIGHT_TOKEN,
    POS_PLANE,
    POS_STRIDE,
    HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    THREE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BM: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    token = row // PADDED_HEADS
    head = row % PADDED_HEADS
    in_bounds = token < tokens
    live = in_bounds & (head < HEADS)
    token = tl.where(in_bounds, token, 0).to(tl.int64)
    source = Query + token[:, None] * QUERY_TOKEN + head[:, None] * 128
    col = tl.arange(0, 32)[None, :]
    mask = live[:, None]
    a0 = tl.load(source + col, mask, 0).to(tl.float32)
    a1 = tl.load(source + 32 + col, mask, 0).to(tl.float32)
    even = tl.load(source + 64 + 2 * col, mask, 0).to(tl.float32)
    odd = tl.load(source + 65 + 2 * col, mask, 0).to(tl.float32)
    frequency = _rope_frequency(
        Pos, Inv, token, live, POS_PLANE, POS_STRIDE, THREE, INTERLEAVED, S0, S1, S2
    )
    q0, q1, q2, q3, scale = _rotate_fwht_quantize(a0, a1, even, odd, frequency, BM)
    _store_quant(Quant, row.to(tl.int64) * 128, q0, q1, q2, q3, in_bounds)
    weight = tl.load(HeadWeights + token * HEAD_WEIGHT_TOKEN + head, live, 0)
    weight = weight.to(tl.float32) * WEIGHT_SCALE * scale
    tl.store(Weights + row, tl.where(live, weight, 0.0), in_bounds)


@triton.jit
def _pool_lanes(value_row, Ape, lanes, mask, R: tl.constexpr, BM: tl.constexpr):
    """Softmax over the `R` group members, then the weighted value sum.

    Each member row holds 128 values followed by 128 gate logits.
    """
    gate_row = value_row + 128
    peak = tl.full((BM, 32), float("-inf"), tl.float32)
    for r in tl.static_range(R):
        logit = tl.load(gate_row + r * 256 + lanes, mask, 0)
        peak = tl.maximum(peak, logit + tl.load(Ape + r * 128 + lanes))
    total = tl.zeros((BM, 32), tl.float32)
    for r in tl.static_range(R):
        logit = tl.load(gate_row + r * 256 + lanes, mask, 0)
        total += tldevice.exp(logit + tl.load(Ape + r * 128 + lanes) - peak)
    pooled = tl.zeros((BM, 32), tl.float32)
    for r in tl.static_range(R):
        logit = tl.load(gate_row + r * 256 + lanes, mask, 0)
        weight = tldevice.exp(logit + tl.load(Ape + r * 128 + lanes) - peak) / total
        pooled += tl.load(value_row + r * 256 + lanes, mask, 0) * weight
    return pooled


@triton.jit
def _pool_quantize_write_kernel(
    Projection,
    Ape,
    NormWeight,
    Pos,
    Inv,
    Cache,
    CacheWords,
    Table,
    Owners,
    Groups,
    Active,
    slots,
    POS_PLANE,
    POS_STRIDE,
    CACHE_BLOCK,
    TABLE_ROW,
    TABLE_COLS,
    R: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    MX: tl.constexpr,
    THREE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BM: tl.constexpr,
):
    slot = tl.program_id(0) * BM + tl.arange(0, BM)
    in_bounds = slot < slots
    live = in_bounds & tl.load(Active + slot, in_bounds, 0).to(tl.int1)
    slot64 = tl.where(in_bounds, slot, 0).to(tl.int64)
    col = tl.arange(0, 32)[None, :]
    mask = live[:, None]

    # Lanes are gathered as [0:32), [32:64), rotary evens and rotary odds.
    value_row = Projection + slot64[:, None] * (R * 256)
    pooled0 = _pool_lanes(value_row, Ape, col, mask, R, BM)
    pooled1 = _pool_lanes(value_row, Ape, 32 + col, mask, R, BM)
    pooled2 = _pool_lanes(value_row, Ape, 64 + 2 * col, mask, R, BM)
    pooled3 = _pool_lanes(value_row, Ape, 65 + 2 * col, mask, R, BM)
    square = (
        tl.sum(pooled0 * pooled0, 1)
        + tl.sum(pooled1 * pooled1, 1)
        + tl.sum(pooled2 * pooled2, 1)
        + tl.sum(pooled3 * pooled3, 1)
    )
    rstd = tl.rsqrt(square / 128 + EPS)[:, None]
    a0 = (pooled0 * rstd * tl.load(NormWeight + col)).to(tl.bfloat16)
    a1 = (pooled1 * rstd * tl.load(NormWeight + 32 + col)).to(tl.bfloat16)
    even = (pooled2 * rstd * tl.load(NormWeight + 64 + 2 * col)).to(tl.bfloat16)
    odd = (pooled3 * rstd * tl.load(NormWeight + 65 + 2 * col)).to(tl.bfloat16)

    frequency = _rope_frequency(
        Pos, Inv, slot64, live, POS_PLANE, POS_STRIDE, THREE, INTERLEAVED, S0, S1, S2
    )
    q0, q1, q2, q3, scale = _rotate_fwht_quantize(
        a0.to(tl.float32),
        a1.to(tl.float32),
        even.to(tl.float32),
        odd.to(tl.float32),
        frequency,
        BM,
    )

    group = tl.load(Groups + slot64, live, 0).to(tl.int64)
    owner = tl.load(Owners + slot64, live, 0).to(tl.int64)
    column = tl.minimum(tl.maximum(group // PAGE, 0), TABLE_COLS - 1)
    block = tl.load(Table + owner * TABLE_ROW + column, live, 0).to(tl.int64)
    base = block * CACHE_BLOCK
    page_row = group % PAGE
    _store_quant(Cache, base + page_row * 128, q0, q1, q2, q3, live)
    tl.store(
        CacheWords + (base + PAGE * 128 + page_row * 4) // 4,
        _encoded_scale(scale, MX),
        live,
    )


_inverse_frequencies: dict[tuple[float, torch.device], torch.Tensor] = {}


def _rope_arguments(positions, parameters, device):
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool currently requires default RoPE")
    theta = float(parameters.get("rope_theta", 10000.0))
    key = (theta, device)
    if key not in _inverse_frequencies:
        _inverse_frequencies[key] = theta ** (
            -torch.arange(0, _ROPE_DIM, 2, device=device, dtype=torch.float32)
            / _ROPE_DIM
        )
    three = positions.ndim == 2
    section = parameters.get("mrope_section") if three else (0, 0, 0)
    if three and (positions.shape[0] != 3 or section is None or sum(section) != _HALF):
        raise ValueError("IndexPool requires valid three-axis MRoPE sections")
    return dict(
        Inv=_inverse_frequencies[key],
        POS_PLANE=positions.stride(0) if three else 0,
        POS_STRIDE=positions.stride(-1),
        THREE=three,
        INTERLEAVED=bool(parameters.get("mrope_interleaved", True)),
        S0=section[0],
        S1=section[1],
        S2=section[2],
    )


def quantize_query(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    positions: torch.Tensor,
    parameters: dict[str, Any],
    padded_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE, FWHT and FP8-quantize indexer queries, padding heads with zeros.

    Args:
        query: `[tokens, heads, 128]` BF16 with unit-stride heads.
        head_weights: `[tokens, heads]` per-head scoring weights.
        positions: `[tokens]` or `[3, tokens]` positions.
        parameters: Layer RoPE parameters.
        padded_heads: Head count the MQA logits kernel expects.

    Returns:
        FP8 queries of shape `[tokens, padded_heads, 128]`.
        FP32 weights of shape `[tokens, padded_heads]` with each head's
            quantization scale folded in.

    """
    tokens, heads, dim = query.shape
    assert dim == INDEX_DIM and query.stride(2) == 1 and query.stride(1) == dim
    assert head_weights.stride(-1) == 1
    quant = torch.empty(
        (tokens, padded_heads, dim), dtype=torch.float8_e4m3fn, device=query.device
    )
    weights = torch.empty(
        (tokens, padded_heads), dtype=torch.float32, device=query.device
    )
    rows = tokens * padded_heads
    if rows:
        block = 16
        _quantize_query_kernel[(triton.cdiv(rows, block),)](
            query,
            head_weights,
            positions,
            Quant=quant.view(torch.uint8),
            Weights=weights,
            tokens=tokens,
            QUERY_TOKEN=query.stride(0),
            HEAD_WEIGHT_TOKEN=head_weights.stride(0),
            HEADS=heads,
            PADDED_HEADS=padded_heads,
            WEIGHT_SCALE=heads**-0.5 * INDEX_DIM**-0.5,
            BM=block,
            enable_fp_fusion=False,
            **_rope_arguments(positions, parameters, query.device),
        )
    return quant, weights


def pool_quantize_write(
    projection: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    parameters: dict[str, Any],
    cache: torch.Tensor,
    table: torch.Tensor,
    owners: torch.Tensor,
    groups: torch.Tensor,
    active: torch.Tensor,
    mx_scales: bool,
) -> None:
    """Gate-pool raw keys, RoPE, FWHT, FP8-quantize and write pool pages.

    Args:
        projection: `[slots, ratio, 256]` FP32 value projections followed
            by gate projections.
        ape: `[ratio, 128]` intra-group position bias.
        norm_weight: `[128]` key RMSNorm weight.
        eps: RMSNorm epsilon.
        positions: `[slots]` or `[3, slots]` group positions.
        parameters: Layer RoPE parameters.
        cache: `[blocks, page, 132]` UINT8 pool pages.
        table: `[requests, blocks]` block table.
        owners: `[slots]` request of each slot.
        groups: `[slots]` pool group of each slot.
        active: `[slots]` whether each slot is written.
        mx_scales: Store scales as four UE8M0 bytes instead of FP32.

    """
    slots, ratio, dim = projection.shape
    assert dim == 2 * INDEX_DIM and projection.is_contiguous()
    assert cache.dtype == torch.uint8 and cache.stride(0) % 4 == 0
    if slots == 0:
        return
    page = cache.shape[1]
    block = 16
    _pool_quantize_write_kernel[(triton.cdiv(slots, block),)](
        projection,
        ape.contiguous(),
        norm_weight,
        positions,
        Cache=cache,
        CacheWords=cache.view(torch.int32),
        Table=table,
        Owners=owners,
        Groups=groups,
        Active=active,
        slots=slots,
        CACHE_BLOCK=cache.stride(0),
        TABLE_ROW=table.stride(0),
        TABLE_COLS=table.shape[1],
        R=ratio,
        PAGE=page,
        EPS=eps,
        MX=mx_scales,
        BM=block,
        enable_fp_fusion=False,
        **_rope_arguments(positions, parameters, projection.device),
    )


# COUNT follows the context length below the group budget; keep it runtime.
@triton.jit(do_not_specialize=["COUNT"])
def _expand_groups_kernel(
    Indices,
    Logical,
    Active,
    Out,
    COUNT,
    INDEX_ROW,
    COUNT_P2: tl.constexpr,
    RATIO: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_P2: tl.constexpr,
    OUT_ROW,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, COUNT_P2)
    selected = k < COUNT
    index = tl.load(Indices + row * INDEX_ROW + k, selected, -1)
    sentinel: tl.constexpr = 2147483647
    group = tl.sort(tl.where(index >= 0, index, sentinel))

    position = tl.load(Logical + row)
    live = (position >= 0) & tl.load(Active + row).to(tl.int1)
    upper = tl.where(live, position // RATIO, 0)
    valid = live & (group != sentinel) & (group >= 0) & (group < upper)
    member = tl.arange(0, RATIO)
    out = Out + row * OUT_ROW
    destination = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(
        out + destination[:, None] * RATIO + member[None, :],
        group[:, None] * RATIO + member[None, :],
        valid[:, None],
    )
    history = tl.sum(valid.to(tl.int32), 0) * RATIO
    tail = upper * RATIO + member
    tail_valid = live & (tail <= position)
    tail_destination = history + tl.cumsum(tail_valid.to(tl.int32), 0) - 1
    tl.store(out + tail_destination, tail.to(tl.int32), tail_valid)
    filled = history + tl.sum(tail_valid.to(tl.int32), 0)
    column = tl.arange(0, OUT_P2)
    tl.store(out + column, -1, (column >= filled) & (column < OUT_WIDTH))


def select_expand_groups(
    logits: torch.Tensor,
    upper: torch.Tensor,
    budget: int,
    logical: torch.Tensor,
    active: torch.Tensor,
    ratio: int,
    out: torch.Tensor,
    topk: SparseIndexerTopk,
) -> None:
    """Pick each row's top pool groups and expand them to sorted token indices.

    Args:
        logits: `[rows, width]` FP32 group scores.
        upper: `[rows]` number of complete groups visible to each row.
        budget: Maximum number of recalled groups.
        logical: `[rows]` logical query positions (negative when padded).
        active: `[rows]` whether each row is a live query.
        ratio: Tokens per group.
        out: `[rows, budget * ratio + ratio]` INT32 destination. Live indices
            are packed in ascending order, followed by `-1` padding.
        topk: The sparse indexer top-k stage that selects the groups.

    """
    rows, width = logits.shape
    out_width = budget * ratio + ratio
    assert out.shape == (rows, out_width) and out.stride(1) == 1
    if rows == 0:
        return
    assert logits.stride(1) == 1
    count = min(budget, width)
    # Selects within each row's first `upper` groups, -1 past them. A 32-byte
    # row pitch suits every backend's output stride.
    indices = out.new_empty(rows, round_up(count, 8))[:, :count]
    topk.forward(logits, upper, 1, indices, count, width)
    _expand_groups_kernel[(rows,)](
        indices,
        logical,
        active,
        out,
        COUNT=count,
        INDEX_ROW=indices.stride(0),
        COUNT_P2=triton.next_power_of_2(count),
        RATIO=ratio,
        OUT_WIDTH=out_width,
        OUT_P2=triton.next_power_of_2(out_width),
        OUT_ROW=out.stride(0),
        num_warps=4,
    )
