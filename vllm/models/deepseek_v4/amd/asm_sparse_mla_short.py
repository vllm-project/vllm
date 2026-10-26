# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed sparse-MLA short rows with a retained softmax probability residual."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _short_scores(
    Q,
    QR,
    KV,
    KR,
    Indptr,
    Indices,
    Scores,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    MAX_KV: tl.constexpr,
    KEYS: tl.constexpr,
    HEADS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(Indptr + row)
    length = tl.load(Indptr + row + 1) - start
    if length > 0 and length <= MAX_KV and tl.program_id(2) * 64 < length:
        heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
        cols = tl.program_id(2) * 64 + tl.arange(0, 64)
        valid = cols < length
        slots = tl.load(Indices + start + cols, valid, 0).to(tl.int64)
        dims = tl.arange(0, 512)
        scales = tl.arange(0, 16)
        rope = tl.arange(0, 64)
        qoff = row * QS0 + heads[:, None] * QS1
        q = tl.load(Q + qoff + dims[None, :], dims[None, :] < 448, 0.0)
        qs = tl.load(
            Q.to(tl.pointer_type(tl.uint8)) + qoff + 448 + scales[None, :],
            scales[None, :] < 14,
            127,
        )
        k = tl.load(
            KV + slots[:, None] * 512 + dims[None, :],
            valid[:, None] & (dims[None, :] < 448),
            0.0,
        )
        ks = tl.load(
            KV.to(tl.pointer_type(tl.uint8))
            + slots[:, None] * 512
            + 448
            + scales[None, :],
            valid[:, None] & (scales[None, :] < 14),
            127,
        )
        qr = tl.load(QR + row * RS0 + heads[:, None] * RS1 + rope[None, :])
        kr = tl.load(KR + slots[:, None] * 64 + rope[None, :], valid[:, None], 0.0)
        scores = tl.dot_scaled(q, qs, "e4m3", tl.trans(k), ks, "e4m3")
        scores = tl.dot(qr, tl.trans(kr), scores) * (512.0**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        tl.store(
            Scores + (row * 128 + heads[:, None]) * KEYS + cols[None, :],
            scores,
            cols[None, :] < KEYS,
        )


@triton.jit
def _short_values_impl(
    KV,
    KR,
    Indptr,
    Indices,
    Sink,
    Scores,
    Output,
    OS0: tl.constexpr,
    MAX_KV: tl.constexpr,
    KEYS: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    VALUES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(Indptr + row)
    length = tl.load(Indptr + row + 1) - start
    if length > 0 and length <= MAX_KV:
        heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
        cols = tl.arange(0, KEYS)
        dims = tl.program_id(2) * VALUES + tl.arange(0, VALUES)
        valid = cols < length
        scores = tl.load(
            Scores + (row * 128 + heads[:, None]) * SCORE_STRIDE + cols[None, :],
            valid[None, :],
            -float("inf"),
        )
        sink = tl.load(Sink + heads).to(tl.float32)
        maximum = tl.maximum(tl.max(scores, 1), sink)
        p = tl.exp(scores - maximum[:, None])
        denominator = tl.sum(p, 1) + tl.exp(sink - maximum)
        slots = tl.load(Indices + start + cols, valid, 0).to(tl.int64)
        offset = slots[:, None] * 512
        nope = tl.load(
            KV + offset + dims[None, :], valid[:, None] & (dims[None, :] < 448), 0.0
        )
        scales = tl.load(
            KV.to(tl.pointer_type(tl.uint8)) + offset + 448 + 2 * (dims[None, :] // 64),
            valid[:, None] & (dims[None, :] < 448),
            127,
        )
        rope = tl.load(
            KR + slots[:, None] * 64 + dims[None, :] - 448,
            valid[:, None] & (dims[None, :] >= 448),
            0.0,
        )
        value = tl.where(
            dims[None, :] < 448,
            nope.to(tl.float32) * tl.exp2(scales.to(tl.float32) - 127.0),
            rope.to(tl.float32),
        ).to(tl.bfloat16)
        high = p.to(tl.bfloat16)
        residual = (p - high.to(tl.float32)).to(tl.bfloat16)
        result = tl.dot(residual, value)
        result = tl.dot(high, value, result)
        tl.store(
            Output + row * OS0 + heads[:, None] * 512 + dims[None, :],
            (result / denominator[:, None]).to(tl.bfloat16),
        )


@triton.jit
def _short_values(
    KV,
    KR,
    Indptr,
    Indices,
    Sink,
    Scores,
    Output,
    OS0: tl.constexpr,
    MAX_KV: tl.constexpr,
    KEYS: tl.constexpr,
    HEADS: tl.constexpr,
    VALUES: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Indptr + row + 1) - tl.load(Indptr + row)
    if length > 0 and length <= MAX_KV:
        if KEYS <= 128 or length <= 128:
            _short_values_impl(
                KV,
                KR,
                Indptr,
                Indices,
                Sink,
                Scores,
                Output,
                OS0,
                MAX_KV,
                min(KEYS, 128),
                KEYS,
                HEADS,
                VALUES,
            )
        elif KEYS <= 256 or length <= 256:
            _short_values_impl(
                KV,
                KR,
                Indptr,
                Indices,
                Sink,
                Scores,
                Output,
                OS0,
                MAX_KV,
                min(KEYS, 256),
                KEYS,
                HEADS,
                VALUES,
            )
        else:
            _short_values_impl(
                KV,
                KR,
                Indptr,
                Indices,
                Sink,
                Scores,
                Output,
                OS0,
                MAX_KV,
                KEYS,
                KEYS,
                HEADS,
                VALUES,
            )


def correct_short_packed_rows(
    q: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    kv_rope: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor,
    *,
    block_heads: int = 32,
    num_warps: int = 4,
    block_values: int = 64,
    matrix_instr_nonkdim: int = 16,
    max_short_kv: int = 512,
) -> None:
    """Overwrite short rows; empty graph-padding rows remain undefined."""
    keys = max(64, triton.next_power_of_2(max_short_kv))
    scores = torch.empty((q.shape[0], 128, keys), dtype=torch.float32, device=q.device)
    options = dict(
        num_warps=num_warps, num_stages=1, matrix_instr_nonkdim=matrix_instr_nonkdim
    )
    _short_scores[(q.shape[0], 128 // block_heads, triton.cdiv(keys, 64))](
        q,
        q_rope,
        kv,
        kv_rope,
        indptr,
        indices,
        scores,
        q.stride(0),
        q.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        MAX_KV=max_short_kv,
        KEYS=keys,
        HEADS=block_heads,
        **options,
    )
    _short_values[(q.shape[0], 128 // block_heads, 512 // block_values)](
        kv,
        kv_rope,
        indptr,
        indices,
        sink,
        scores,
        output,
        output.stride(0),
        MAX_KV=max_short_kv,
        KEYS=keys,
        HEADS=block_heads,
        VALUES=block_values,
        **options,
    )
