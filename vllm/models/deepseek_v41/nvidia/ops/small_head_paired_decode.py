# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native eight-head query pairing over exact sparse KV unions.

KV loader and split softmax derive from luoyuctl's #59418.
"""

import torch

from vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode import _load_kv_block
from vllm.triton_utils import tl, triton


@triton.jit
def _attend(
    q,
    qt,
    qh,
    rows,
    groups,
    sc,
    sp,
    PS: tl.constexpr,
    si,
    sm,
    sl,
    SW: tl.constexpr,
    ec,
    ep,
    PE: tl.constexpr,
    ei,
    em,
    el,
    EW: tl.constexpr,
    po,
    pl,
    scale,
    EXTRA: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
):
    g, split = tl.program_id(0), tl.program_id(1)
    if g >= tl.load(groups):
        return
    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    row = tl.load(rows + 2 * g + h // 8)
    query = tl.load(
        q + row[:, None].to(tl.int64) * qt + (h % 8)[:, None] * qh + d[None, :],
        row[:, None] >= 0,
        other=0.0,
    )
    sn = tl.load(sl + g)
    sb = tl.cdiv(sn, N)
    blocks = sb
    en = 0
    if EXTRA:
        en = tl.load(el + g)
        blocks += tl.cdiv(en, N)
    per = tl.cdiv(blocks, S)
    begin, end = split * per, tl.minimum((split + 1) * per, blocks)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 512), tl.float32)
    lane = tl.arange(0, N)
    for b in range(begin, end):
        if b < sb:
            offset = b * N
            kv, valid = _load_kv_block(
                sc, sp, PS, si, g.to(tl.int64) * SW, sn, offset, N
            )
            counts = tl.load(
                sm + g.to(tl.int64) * SW + offset + lane, offset + lane < sn, other=0
            ).to(tl.uint32)
        else:
            offset = (b - sb) * N
            kv, valid = _load_kv_block(
                ec, ep, PE, ei, g.to(tl.int64) * EW, en, offset, N
            )
            counts = tl.load(
                em + g.to(tl.int64) * EW + offset + lane, offset + lane < en, other=0
            ).to(tl.uint32)
        multiplicity = tl.where(
            (h < 8)[:, None], counts[None, :] & 65535, counts[None, :] >> 16
        ).to(tl.float32)
        score = tl.dot(query, tl.trans(kv)) * (scale * 1.4426950408889634)
        score = tl.where(
            valid[None, :] & (row >= 0)[:, None] & (multiplicity > 0),
            score,
            -float("inf"),
        )
        nxt = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(nxt == -float("inf"), 0.0, nxt)
        alpha = tl.exp2(maximum - safe)
        probability = tl.exp2(score - safe[:, None]) * multiplicity
        denominator = denominator * alpha + tl.sum(probability, 1)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probability.to(tl.bfloat16), kv
        )
        maximum = nxt
    empty = denominator == 0.0
    safe_denom = tl.where(empty, 1.0, denominator)
    lse = tl.where(empty, -float("inf"), maximum + tl.log2(safe_denom))
    result = accumulator / safe_denom[:, None]
    part = (g.to(tl.int64) * S + split) * 16 + h
    tl.store(po + part[:, None] * 512 + d[None, :], result)
    tl.store(pl + part, lse)


@triton.jit
def _merge(po, pl, rows, groups, sink, out, out_stride, oh, S: tl.constexpr):
    g, h = tl.program_id(0), tl.program_id(1)
    if g >= tl.load(groups):
        return
    row = tl.load(rows + 2 * g + h // 8)
    if row < 0:
        return
    s = tl.arange(0, S)
    d = tl.arange(0, 512)
    part = (g.to(tl.int64) * S + s) * 16 + h
    lse = tl.load(pl + part)
    maximum = tl.max(lse, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(lse - safe)
    denom = tl.sum(weights, 0)
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    output = tl.sum(
        tl.load(po + part[:, None] * 512 + d[None, :]) * weights[:, None], 0
    )
    total_lse = safe + tl.log2(safe_denom)
    sink2 = tl.load(sink + h % 8) * 1.4426950408889634
    output = output / safe_denom / (1.0 + tl.exp2(sink2 - total_lse))
    output = tl.where(denom > 0.0, output, 0.0)
    tl.store(out + row.to(tl.int64) * out_stride + (h % 8) * oh + d, output)


def run(q, sc, ec, sink, scale, out, metadata, partials, config):
    assert q.shape[1] == out.shape[1] == 8 and q.shape[-1] == 512
    assert q.dtype == out.dtype == torch.bfloat16
    assert q.stride(-1) == out.stride(-1) == 1
    if not q.shape[0]:
        return
    splits, block, warps = config
    rows, count, unions = metadata
    si, sm, sl = unions[0]
    has_extra = ec is not None
    ei, em, el = unions[1] if has_extra else unions[0]
    ec = ec if has_extra else sc
    _attend[(q.shape[0], splits)](
        q,
        q.stride(0),
        q.stride(1),
        rows,
        count,
        sc.view(torch.uint8),
        sc.stride(0),
        sc.shape[1],
        si,
        sm,
        sl,
        si.stride(0),
        ec.view(torch.uint8),
        ec.stride(0),
        ec.shape[1],
        ei,
        em,
        el,
        ei.stride(0),
        *partials,
        scale,
        has_extra,
        block,
        splits,
        num_warps=warps,
        num_stages=2,
    )
    _merge[(q.shape[0], 16)](
        *partials,
        rows,
        count,
        sink,
        out,
        out.stride(0),
        out.stride(1),
        splits,
        num_warps=4,
    )
