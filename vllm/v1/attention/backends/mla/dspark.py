# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged DSpark MLA with an anchor-relative window and block masks."""

import torch
import triton as tr
import triton.language as tl

from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend


class DSparkMLAMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reorder_batch_threshold = 0

    def build(
        self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs
    ):
        return common_attn_metadata


class DSparkMLABackend(TritonMLABackend):
    @staticmethod
    def get_name():
        return "GLM_DSPARK_MLA"

    @staticmethod
    def get_builder_cls():
        return DSparkMLAMetadataBuilder


@tr.jit
def partial(
    Q,
    Z,
    T,
    S,
    L,
    OUT,
    E,
    H: tl.constexpr,
    BS: tl.constexpr,
    TS: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    M: tl.constexpr = 16,
    N: tl.constexpr = 32,
    D: tl.constexpr = 1024,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    start = tl.load(S + req)
    end = tl.load(S + req + 1)
    ql = end - start
    length = tl.load(L + req)
    rows = tl.arange(0, M)
    dims = tl.arange(0, D)
    ks = tl.arange(0, N)
    q = tl.load(
        Q + ((start + rows[:, None]) * H + head) * 576 + dims[None, :],
        (rows[:, None] < ql) & (dims[None, :] < 576),
        0,
    )
    anchor = length - ql
    first = 0
    if WINDOW > 0:
        first = tl.maximum(0, anchor - WINDOW)
    chunks = tl.cdiv(tl.maximum(0, length - first), N)
    per_split = tl.cdiv(chunks, SPLITS)
    lo = first + split * per_split * N
    hi = tl.minimum(length, lo + per_split * N)
    maxv = tl.full((M,), float("-inf"), tl.float32)
    sumv = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    for base in range(lo, hi, N):
        ids = base + ks
        block = tl.load(T + req * TS + ids // BS, ids < hi, 0)
        z = tl.load(
            Z + (block[:, None] * BS + ids[:, None] % BS) * 576 + dims[None, :],
            (ids[:, None] < hi) & (dims[None, :] < 576),
            0,
        )
        score = tl.dot(q, tl.trans(z)) * 0.0625
        valid = (rows[:, None] < ql) & (ids[None, :] < hi)
        if CAUSAL:
            valid &= ids[None, :] <= anchor + rows[:, None]
        score = tl.where(valid, score, float("-inf"))
        newmax = tl.maximum(maxv, tl.max(score, 1))
        safe = tl.where(newmax == float("-inf"), 0.0, newmax)
        factor = tl.exp(maxv - safe)
        p = tl.exp(score - safe[:, None])
        acc = acc * factor[:, None] + tl.dot(p.to(z.dtype), z)
        sumv = sumv * factor + tl.sum(p, 1)
        maxv = newmax
    denom = tl.where(sumv > 0, sumv, 1.0)
    ptr = (
        ((start + rows[:, None]) * H + head) * SPLITS * 512
        + split * 512
        + dims[None, :]
    )
    tl.store(
        OUT + ptr, acc / denom[:, None], (rows[:, None] < ql) & (dims[None, :] < 512)
    )
    tl.store(
        E + ((start + rows) * H + head) * SPLITS + split,
        tl.where(sumv > 0, maxv + tl.log(denom), float("-inf")),
        rows < ql,
    )


@tr.jit
def merge(OUT, E, Y, H: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    ss = tl.arange(0, SPLITS)
    rr = tl.arange(0, 512)
    lse = tl.load(E + (row * H + head) * SPLITS + ss)
    m = tl.max(lse, 0)
    safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - safe)
    total = tl.sum(w, 0)
    w = w / tl.where(total > 0, total, 1.0)
    vals = tl.load(OUT + ((row * H + head) * SPLITS + ss[:, None]) * 512 + rr[None, :])
    tl.store(Y + (row * H + head) * 512 + rr, tl.sum(vals * w[:, None], 0))


def attention(q, z, table, starts, lengths, causal, window):
    assert q.is_contiguous() and z.is_contiguous() and q.shape[-1] == z.shape[-1] == 576
    tokens, heads, _ = q.shape
    splits = 16 if not window else 4
    tmp = torch.zeros(
        (tokens, heads, splits, 512), device=q.device, dtype=torch.float32
    )
    lse = torch.full(
        (tokens, heads, splits), float("-inf"), device=q.device, dtype=torch.float32
    )
    out = torch.zeros((tokens, heads, 512), device=q.device, dtype=q.dtype)
    partial[(len(lengths), heads, splits)](
        q,
        z,
        table,
        starts,
        lengths,
        tmp,
        lse,
        heads,
        z.shape[1],
        table.stride(0),
        causal,
        window or 0,
        splits,
        num_warps=8,
    )
    merge[(tokens, heads)](tmp, lse, out, heads, splits, num_warps=4)
    return out
