# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA decode attention read directly from paged KV by selected regions.

Replaces gather-to-scratch plus a ragged FlashInfer call: each program takes
all query heads of one KV group (GQA packing) for one split of a row's
selected regions, loads the regions' pages in place (dequantizing FP8), and a
combine kernel merges splits and clears inactive rows.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

# PDL: kernels launch during the producer's tail and gdc_wait before reading
# its outputs. No early trigger, so their writes stay visible downstream.
_PDL = current_platform.is_arch_support_pdl()
# The AMD Triton backend rejects the launch_pdl keyword, even when False.
_LAUNCH = {"launch_pdl": True} if _PDL else {}


@triton.jit(do_not_specialize=["SCALE"])
def _sparse_decode_split(
    Q,
    Cache,
    Regions,
    Table,
    Positions,
    Active,
    KScale,
    VScale,
    Partial,
    Lse,
    Output,
    Gate,
    SCALE,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    CLEAR_INACTIVE: tl.constexpr,
    GATED: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_HEAD: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    CACHE_GROUP: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    REGION_ROW: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    GROUPS: tl.constexpr,
    ROWS_PER_REQUEST: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    REGION_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    # Row r is KV group r % GROUPS; its request is r // ROWS_PER_REQUEST.
    row = tl.program_id(0)
    split = tl.program_id(1)
    group = row % GROUPS
    request = row // ROWS_PER_REQUEST
    position = tl.load(Positions + row)
    active = tl.load(Active + row) != 0
    # Selections are left-padded with -1 and hold all history below the
    # budget, so only the trailing min(position / 8 + 1, R) slots are live.
    live = tl.minimum(tl.maximum(position, 0) // 8 + 1, REGION_COUNT)
    live_start = REGION_COUNT - live
    chunk = (live + SPLITS - 1) // SPLITS
    first = live_start + split * chunk
    last = tl.minimum(first + chunk, REGION_COUNT)

    m = tl.arange(0, BLOCK_M)
    # 192 = 3 x 64: equal-shape pieces share one layout, so the gather
    # addresses are computed once for all six K/V loads.
    d = tl.arange(0, 64)
    q_base = Q + row.to(tl.int64) * Q_ROW + m[:, None] * Q_HEAD + d[None, :]
    q_mask = (m < HEADS)[:, None]
    q0 = tl.load(q_base, q_mask, other=0.0)
    q1 = tl.load(q_base + 64, q_mask, other=0.0)
    q2 = tl.load(q_base + 128, q_mask, other=0.0)
    if FP8:
        # E4M3 -> FP16 is a native conversion (E4M3 -> BF16 goes via FP32).
        q0, q1, q2 = q0.to(tl.float16), q1.to(tl.float16), q2.to(tl.float16)
    # FP8 -> FP16 is exact; the K scale folds into the logits and the V scale
    # into the partial output instead of rescaling every element.
    scale = SCALE * 1.4426950408889634
    v_scale = 1.0
    if FP8:
        scale = scale * tl.load(KScale)
        v_scale = tl.load(VScale)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    total = tl.zeros((BLOCK_M,), tl.float32)
    acc0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc1 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc2 = tl.zeros((BLOCK_M, 64), tl.float32)
    lane = tl.arange(0, 64)
    cache = Cache + group * CACHE_GROUP
    # Eight regions (64 tokens) per tile.
    for tile in range(first, last, 8):
        slot = tile + lane // 8
        region = tl.load(
            Regions + row.to(tl.int64) * REGION_ROW + slot,
            (slot < last) & active,
            other=-1,
        )
        token = tl.maximum(region, 0) * 8 + lane % 8
        valid = (region >= 0) & (token <= position)
        page = tl.load(
            Table + request * TABLE_ROW + token // BLOCK_SIZE, valid, other=0
        )
        base = (
            cache + page.to(tl.int64) * CACHE_PAGE + (token % BLOCK_SIZE) * CACHE_TOKEN
        )[:, None] + d[None, :]
        mask = valid[:, None]
        k0 = tl.load(base, mask, other=0.0)
        k1 = tl.load(base + 64, mask, other=0.0)
        k2 = tl.load(base + 128, mask, other=0.0)
        v0 = tl.load(base + 192, mask, other=0.0)
        v1 = tl.load(base + 256, mask, other=0.0)
        v2 = tl.load(base + 320, mask, other=0.0)
        if FP8:
            k0, k1, k2 = k0.to(tl.float16), k1.to(tl.float16), k2.to(tl.float16)
            v0, v1, v2 = v0.to(tl.float16), v1.to(tl.float16), v2.to(tl.float16)
        scores = tl.dot(q0, tl.trans(k0))
        scores = tl.dot(q1, tl.trans(k1), scores)
        scores = tl.dot(q2, tl.trans(k2), scores)
        scores = tl.where(valid[None, :], scores * scale, -float("inf"))
        updated = tl.maximum(maximum, tl.max(scores, 1))
        pivot = tl.where(updated == -float("inf"), 0.0, updated)
        alpha = tl.exp2(maximum - pivot)
        weights = tl.exp2(scores - pivot[:, None])
        total = total * alpha + tl.sum(weights, 1)
        probs = weights.to(v0.dtype)
        acc0 = tl.dot(probs, v0, acc0 * alpha[:, None])
        acc1 = tl.dot(probs, v1, acc1 * alpha[:, None])
        acc2 = tl.dot(probs, v2, acc2 * alpha[:, None])
        maximum = updated

    keep = m < HEADS
    out = Output + row.to(tl.int64) * OUT_ROW + m[:, None] * OUT_HEAD + d[None, :]
    clear = CLEAR_INACTIVE & ~active
    inverse = tl.where(total > 0, v_scale / tl.maximum(total, 1e-30), 0.0)
    store = keep[:, None]
    if SPLITS == 1:
        inverse = tl.where(clear, 0.0, inverse)
        sigmoid = _gate(
            Gate, row // GROUPS, group * HEADS + m, keep, GATED, GATE_ROW, GATE_HEAD
        )[:, None]
        dtype = Output.dtype.element_ty
        inverse = inverse[:, None]
        value0 = (acc0 * inverse).to(dtype).to(tl.float32) * sigmoid
        value1 = (acc1 * inverse).to(dtype).to(tl.float32) * sigmoid
        value2 = (acc2 * inverse).to(dtype).to(tl.float32) * sigmoid
        tl.store(out, value0.to(dtype), store)
        tl.store(out + 64, value1.to(dtype), store)
        tl.store(out + 128, value2.to(dtype), store)
    else:
        lse = tl.where(
            total > 0, maximum + tl.log2(tl.maximum(total, 1e-30)), -float("inf")
        )
        out_row = row.to(tl.int64) * SPLITS * HEADS + split * HEADS + m
        tl.store(Lse + out_row, lse, keep)
        partial = Partial + out_row[:, None] * 192 + d[None, :]
        inverse = inverse[:, None]
        tl.store(partial, acc0 * inverse, store)
        tl.store(partial + 64, acc1 * inverse, store)
        tl.store(partial + 128, acc2 * inverse, store)


@triton.jit
def _sparse_decode_combine(
    Partial,
    Lse,
    Active,
    Output,
    Gate,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CLEAR_INACTIVE: tl.constexpr,
    GROUPS: tl.constexpr,
    GATED: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_HEAD: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.arange(0, BLOCK_S)
    d = tl.arange(0, 256)
    partial_row = (row.to(tl.int64) * SPLITS + split) * HEADS + head
    lse = tl.load(Lse + partial_row, split < SPLITS, other=-float("inf"))
    maximum = tl.max(lse, 0)
    weight = tl.where(lse > -float("inf"), tl.exp2(lse - maximum), 0.0)
    weight = weight / tl.maximum(tl.sum(weight, 0), 1e-30)
    partial = tl.load(
        Partial + partial_row[:, None] * 192 + d[None, :],
        (split[:, None] < SPLITS) & (d[None, :] < 192),
        other=0.0,
    )
    merged = tl.sum(tl.where(weight[:, None] > 0, partial * weight[:, None], 0.0), 0)
    if CLEAR_INACTIVE:
        merged = tl.where(tl.load(Active + row) != 0, merged, 0.0)
    dtype = Output.dtype.element_ty
    sigmoid = _gate(
        Gate,
        row // GROUPS,
        row % GROUPS * HEADS + head,
        True,
        GATED,
        GATE_ROW,
        GATE_HEAD,
    )
    tl.store(
        Output + row.to(tl.int64) * OUT_ROW + head * OUT_HEAD + d,
        (merged.to(dtype).to(tl.float32) * sigmoid).to(dtype),
        d < 192,
    )


@triton.jit
def _gate(Gate, token, head, mask, GATED: tl.constexpr, GATE_ROW, GATE_HEAD):
    """Headwise sigmoid gate, rounded like the unfused gate kernel (1 if off)."""
    if GATED:
        gate = tl.load(
            Gate + token.to(tl.int64) * GATE_ROW + head * GATE_HEAD, mask, other=0.0
        ).to(tl.float32)
        sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-gate))
        return sigmoid.to(Gate.dtype.element_ty).to(tl.float32)
    return tl.full(head.shape, 1.0, tl.float32)


def sparse_decode_splits(rows: int) -> int:
    """Splits per row for ~256 programs (about 1.7 waves on B300).

    Graphs are captured per batch size, so the split cannot follow context
    length; each split still covers only the row's live regions.
    """
    splits = 1
    while splits < 32 and rows * splits * 2 <= 256:
        splits *= 2
    return splits


def sparse_decode_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    regions: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    active: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    query_len: int = 1,
    splits: int,
    clear_inactive: bool = True,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> None:
    """Attention for (token, KV group) rows over their selected regions.

    With ``gate`` ([tokens, G * heads]) the headwise sigmoid gate is applied
    to the output, as the separate gate kernel would.

    ``query``/``output``: [rows, heads, 192] with rows = tokens * G;
    ``cache``: [pages, G, block, 384] (K then V); ``regions``: [rows, R]
    region ids (-1 unused); ``positions``/``active``: per row. Rows of one
    request are ``G * query_len`` consecutive rows sharing a table row.
    """
    rows, heads, dim = query.shape
    groups = cache.shape[1]
    assert dim == 192 and cache.shape[-1] == 384 and heads <= 16
    assert query.stride(2) == output.stride(2) == cache.stride(3) == 1
    assert regions.stride(1) == 1 and rows == regions.shape[0]
    fp8 = cache.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if fp8:
        assert k_scale is not None and v_scale is not None
        if cache.dtype == torch.uint8:
            cache = cache.view(current_platform.fp8_dtype())
    if not rows:
        return
    partial = torch.empty(
        (rows * splits * heads if splits > 1 else 1, 192),
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.empty(partial.shape[0], dtype=torch.float32, device=query.device)
    _sparse_decode_split[(rows, splits)](
        query,
        cache,
        regions,
        block_table,
        positions,
        active,
        k_scale,
        v_scale,
        partial,
        lse,
        output,
        gate,
        scale,
        output.stride(0),
        output.stride(1),
        clear_inactive,
        gate is not None,
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        regions.stride(0),
        block_table.stride(0),
        groups,
        groups * query_len,
        heads,
        16,
        regions.shape[1],
        cache.shape[2],
        splits,
        fp8,
        _PDL,
        num_warps=4,
        **_LAUNCH,
    )
    if splits > 1:
        _sparse_decode_combine[(rows, heads)](
            partial,
            lse,
            active,
            output,
            gate,
            output.stride(0),
            output.stride(1),
            heads,
            splits,
            triton.next_power_of_2(splits),
            clear_inactive,
            groups,
            gate is not None,
            gate.stride(0) if gate is not None else 0,
            gate.stride(1) if gate is not None else 0,
            _PDL,
            num_warps=4,
            **_LAUNCH,
        )
