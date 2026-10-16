# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sliding-window GQA decode with split KV and a fused headwise gate.

FA2 runs one CTA per query head and scans the whole window serially; its
Python wrapper rejects split KV, and FA4 has no head_dim 192 on SM100. Here
one program handles all query heads of a KV head (the GQA group) for one
window split, and a combine kernel merges splits and applies the gate.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .sparse_decode_attention import _PDL, _gate


@triton.jit(do_not_specialize=["SCALE"])
def _swa_decode_split(
    Q,
    KCache,
    VCache,
    Table,
    SeqLens,
    Partial,
    Lse,
    Gate,
    Output,
    SCALE,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_HEAD: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_PAGE: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    # A program covers PACK consecutive tokens of one request and all GROUP
    # query heads of one KV head: rows are (token, head). Packing MTP verify
    # tokens reads their overlapping windows once. A token at offset i of its
    # request sits at seq_len - QUERY_LEN + i and sees the WINDOW keys ending
    # there.
    program = tl.program_id(0)
    split = tl.program_id(1)
    first_token = program // KV_HEADS * PACK
    kv_head = program % KV_HEADS
    request = first_token // QUERY_LEN
    first = tl.load(SeqLens + request) - QUERY_LEN + first_token % QUERY_LEN
    low = tl.maximum(first - WINDOW + 1, 0)
    chunk: tl.constexpr = (WINDOW + PACK - 1 + SPLITS - 1) // SPLITS
    start = low + split * chunk
    end = tl.minimum(first + PACK, start + chunk)

    m = tl.arange(0, BLOCK_M)
    row_token = m // GROUP
    head = kv_head * GROUP + m % GROUP
    live = m < PACK * GROUP
    position = first + row_token
    d0 = tl.arange(0, 128)
    d1 = 128 + tl.arange(0, 64)
    q_base = Q + (first_token + row_token).to(tl.int64) * Q_ROW + head * Q_HEAD
    q0 = tl.load(q_base[:, None] + d0[None, :], live[:, None], other=0.0)
    q1 = tl.load(q_base[:, None] + d1[None, :], live[:, None], other=0.0)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    total = tl.zeros((BLOCK_M,), tl.float32)
    acc0 = tl.zeros((BLOCK_M, 128), tl.float32)
    acc1 = tl.zeros((BLOCK_M, 64), tl.float32)
    scale = SCALE * 1.4426950408889634
    for tile in range(start, end, BLOCK_N):
        n = tile + tl.arange(0, BLOCK_N)
        valid = n < end
        page = tl.load(Table + request * TABLE_ROW + n // BLOCK_SIZE, valid, other=0)
        offset = page.to(tl.int64) * K_PAGE + (n % BLOCK_SIZE) * K_TOKEN
        k_base = KCache + offset + kv_head * K_HEAD
        k0 = tl.load(k_base[:, None] + d0[None, :], valid[:, None], other=0.0)
        k1 = tl.load(k_base[:, None] + d1[None, :], valid[:, None], other=0.0)
        scores = tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))
        visible = (
            valid[None, :]
            & (n[None, :] <= position[:, None])
            & (n[None, :] > position[:, None] - WINDOW)
        )
        scores = tl.where(visible, scores * scale, -float("inf"))
        updated = tl.maximum(maximum, tl.max(scores, 1))
        # Packed rows see different ranges; a row may have no key yet.
        pivot = tl.where(updated == -float("inf"), 0.0, updated)
        alpha = tl.exp2(maximum - pivot)
        weights = tl.exp2(scores - pivot[:, None])
        total = total * alpha + tl.sum(weights, 1)
        v_offset = page.to(tl.int64) * V_PAGE + (n % BLOCK_SIZE) * V_TOKEN
        v_base = VCache + v_offset + kv_head * V_HEAD
        v0 = tl.load(v_base[:, None] + d0[None, :], valid[:, None], other=0.0)
        v1 = tl.load(v_base[:, None] + d1[None, :], valid[:, None], other=0.0)
        probs = weights.to(v0.dtype)
        acc0 = acc0 * alpha[:, None] + tl.dot(probs, v0)
        acc1 = acc1 * alpha[:, None] + tl.dot(probs, v1)
        maximum = updated

    token = first_token + row_token
    out = Output + token.to(tl.int64)[:, None] * OUT_ROW + head[:, None] * OUT_HEAD
    # Match the unfused path: BF16 attention times sigmoid rounded to gate dtype.
    sigmoid = _gate(Gate, token, head, live, True, GATE_ROW, GATE_HEAD)
    inverse = tl.where(total > 0, 1.0 / tl.maximum(total, 1e-30), 0.0)
    if SPLITS == 1:
        _store_gated(
            out, acc0 * inverse[:, None], acc1 * inverse[:, None], sigmoid, live
        )
    else:
        # Base-2 LSE; empty splits keep -inf and zero partials.
        lse = tl.where(
            total > 0, maximum + tl.log2(tl.maximum(total, 1e-30)), -float("inf")
        )
        base = (token.to(tl.int64) * KV_HEADS + kv_head) * SPLITS
        row = (base + split) * GROUP + m % GROUP
        tl.store(Lse + row, lse, live)
        tl.store(
            Partial + row[:, None] * 192 + d0[None, :],
            acc0 * inverse[:, None],
            live[:, None],
        )
        tl.store(
            Partial + row[:, None] * 192 + d1[None, :],
            acc1 * inverse[:, None],
            live[:, None],
        )


@triton.jit
def _store_gated(out, attention0, attention1, sigmoid, live):
    d0 = tl.arange(0, 128)
    d1 = 128 + tl.arange(0, 64)
    dtype = out.dtype.element_ty
    value0 = attention0.to(dtype).to(tl.float32) * sigmoid[:, None]
    value1 = attention1.to(dtype).to(tl.float32) * sigmoid[:, None]
    tl.store(out + d0[None, :], value0.to(dtype), live[:, None])
    tl.store(out + d1[None, :], value1.to(dtype), live[:, None])


@triton.jit
def _swa_decode_combine(
    Partial,
    Lse,
    Gate,
    Output,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_HEAD: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    token = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP
    split = tl.arange(0, BLOCK_S)
    d = tl.arange(0, 256)
    row = (
        (token.to(tl.int64) * KV_HEADS + kv_head) * SPLITS + split
    ) * GROUP + head % GROUP
    lse = tl.load(Lse + row, split < SPLITS, other=-float("inf"))
    maximum = tl.max(lse, 0)
    weight = tl.where(lse > -float("inf"), tl.exp2(lse - maximum), 0.0)
    weight = weight / tl.maximum(tl.sum(weight, 0), 1e-30)
    partial = tl.load(
        Partial + row[:, None] * 192 + d[None, :],
        (split[:, None] < SPLITS) & (d[None, :] < 192),
        other=0.0,
    )
    attention = tl.sum(tl.where(weight[:, None] > 0, partial * weight[:, None], 0.0), 0)
    # Match the unfused path: BF16 attention times sigmoid rounded to gate dtype.
    attention = attention.to(Output.dtype.element_ty).to(tl.float32)
    sigmoid = _gate(Gate, token, head, True, True, GATE_ROW, GATE_HEAD)
    tl.store(
        Output + token.to(tl.int64) * OUT_ROW + head * OUT_HEAD + d,
        attention * sigmoid,
        d < 192,
    )


def swa_decode_plan(
    tokens: int, kv_heads: int, group: int, query_len: int, num_sms: int, window: int
) -> tuple[int, int]:
    """Return (pack, splits) for a decode batch.

    MTP verify tokens of a request are packed into one program when
    (token, KV head) programs already fill about a wave, since their windows
    overlap; the window is then split to fill the GPU.
    """
    pack = (
        query_len
        if query_len > 1 and tokens * kv_heads >= 128 and query_len * group <= 64
        else 1
    )
    return pack, swa_decode_splits(tokens // pack * kv_heads, num_sms, window)


def swa_decode_splits(programs: int, num_sms: int, window: int) -> int:
    """Split the window until (token, KV head) programs reach ~2 waves."""
    splits = 1
    while (
        splits < 8
        and programs * splits * 2 <= 2 * num_sms
        and window // (splits * 2) >= 64
    ):
        splits *= 2
    return splits


def swa_decode_gated(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    window: int,
    query_len: int,
    splits: int,
    pack: int = 1,
) -> None:
    """Decode rows of ``query_len`` tokens per request over a causal window.

    ``query``/``output``: [tokens, heads, 192]; ``key_cache``/``value_cache``:
    [pages, block, kv_heads, 192] views; ``gate``: [tokens, heads]. Rows of
    requests with no visible keys (padding) produce zeros.
    """
    tokens, heads, dim = query.shape
    kv_heads = key_cache.shape[2]
    group = heads // kv_heads
    assert dim == 192 and group <= 16 and heads == group * kv_heads
    assert query.stride(2) == key_cache.stride(3) == value_cache.stride(3) == 1
    assert output.stride(2) == 1 and tokens == seq_lens.shape[0] * query_len
    assert query_len % pack == 0 and pack * group <= 64
    if not tokens:
        return
    partial = torch.empty(
        (tokens * kv_heads * splits * group if splits > 1 else 1, 192),
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.empty(partial.shape[0], dtype=torch.float32, device=query.device)
    _swa_decode_split[(tokens // pack * kv_heads, splits)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial,
        lse,
        gate,
        output,
        scale,
        output.stride(0),
        output.stride(1),
        gate.stride(0),
        gate.stride(1),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        kv_heads,
        group,
        pack,
        max(16, triton.next_power_of_2(pack * group)),
        query_len,
        key_cache.shape[1],
        window,
        splits,
        64,
        _PDL,
        num_warps=4,
        **({"launch_pdl": _PDL} if current_platform.is_cuda() else {}),
    )
    if splits > 1:
        _swa_decode_combine[(tokens, heads)](
            partial,
            lse,
            gate,
            output,
            output.stride(0),
            output.stride(1),
            gate.stride(0),
            gate.stride(1),
            kv_heads,
            group,
            splits,
            triton.next_power_of_2(splits),
            _PDL,
            num_warps=4,
            **({"launch_pdl": _PDL} if current_platform.is_cuda() else {}),
        )
