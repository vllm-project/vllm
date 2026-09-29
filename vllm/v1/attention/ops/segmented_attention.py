# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ROCm token-major segmented attention kernels and dispatch."""

from __future__ import annotations

from functools import lru_cache

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import get_kv_quant_mode
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from .triton_unified_attention import unified_attention

MAX_QUERY_LEN = 8192
MAX_SPLITS = 64
# D128 scratch cap; wider outputs scale this budget with head dimension.
MAX_LONG_EXTEND_WORKSPACE_BYTES = 32 * 1024 * 1024


def segmented_query_capacity(max_query_len: int) -> int:
    """Round a supported query length to its compiled power-of-two capacity."""
    if not 1 <= max_query_len <= MAX_QUERY_LEN:
        raise ValueError(f"Query length must be in [1, {MAX_QUERY_LEN}]")
    return 1 << (max_query_len - 1).bit_length()


def _query_capacity_buckets() -> tuple[int, ...]:
    return tuple(1 << exponent for exponent in range(MAX_QUERY_LEN.bit_length()))


def _long_extend_splits(
    base_splits: int,
    batch: int,
    query_len: int,
    seq_len: int,
    hq: int,
    hk: int,
    dim: int,
    bm: int,
) -> int:
    """Add long-prefix parallelism while bounding the extra split workspace."""
    if seq_len < 32768:
        return base_splits
    groups = batch * hk * triton.cdiv(query_len * (hq // hk), bm)
    split_cap = 16 if seq_len < 131072 else 32
    scratch_per_split = batch * segmented_query_capacity(query_len) * hq * (dim + 1) * 4
    scratch_limit = MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128
    scratch_splits = scratch_limit // scratch_per_split
    scratch_cap = 1 << (scratch_splits.bit_length() - 1) if scratch_splits else 1
    occupancy_splits = triton.next_power_of_2(triton.cdiv(512, groups))
    return max(base_splits, min(split_cap, scratch_cap, occupancy_splits))


@triton.jit
def _segmented_attention_stage(
    Q,
    K,
    V,
    TABLE,
    STARTS,
    LENS,
    KS,
    VS,
    SINKS,
    PART,
    LSE,
    OUT,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    V3: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QCAP: tl.constexpr,
    LIMIT_Q: tl.constexpr,
    SCALE: tl.constexpr,
    FP8: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    PREFIX_FAST: tl.constexpr,
    PV_SPLIT: tl.constexpr,
    QK_PIPELINE: tl.constexpr,
):
    item, mb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, kh = item // HK, item % HK
    first = tl.load(STARTS + seq)
    nq = tl.load(STARTS + seq + 1) - first
    if nq < 1 or nq > LIMIT_Q:
        return
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = QCAP * G
    if mb * BM >= nq * G:
        return
    sequence = tl.load(LENS + seq)
    prefix = sequence - nq
    m = mb * BM + tl.arange(0, BM)
    qpos, qh = m // G, kh * G + m % G
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    if BK == D:
        q_full = tl.load(
            Q + (first + qpos[:, None]) * Q0 + qh[:, None] * Q1 + d[None, :],
            mask=(m < nq * G)[:, None],
            other=0.0,
        )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    if HAS_SINKS:
        sink = tl.load(SINKS + qh, mask=m < nq * G, other=0).to(tl.float32)
        maximum = tl.where(split == 0, sink * 1.4426950408889634, maximum)
        denom = tl.where(split == 0, 1.0, denom)
    if PV_SPLIT:
        half = tl.arange(0, D // 2)
        acc_low = tl.zeros((BM, D // 2), tl.float32)
        acc_high = tl.zeros((BM, D // 2), tl.float32)
    else:
        acc = tl.zeros((BM, D), tl.float32)
    tile_rows = tl.minimum((mb + 1) * BM, nq * G)
    range_begin = 0
    if WINDOW >= 0:
        first_qpos = (mb * BM) // G
        range_begin = tl.maximum(0, prefix + first_qpos - WINDOW)
        range_begin = (range_begin // BN) * BN
    range_end = sequence
    if CAUSAL:
        range_end = tl.minimum(range_end, prefix + tl.cdiv(tile_rows, G))
    span = tl.maximum(0, range_end - range_begin)
    segment = tl.cdiv(span, SPLITS * BN) * BN
    begin = range_begin + split * segment
    end = tl.minimum(begin + segment, range_end)
    k_scale = 1.0
    v_scale = 1.0
    if FP8:
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
    for tail in tl.static_range(2):
        if tail:
            loop_begin = tl.maximum(begin, (end // BN) * BN)
            loop_end = end
        else:
            loop_begin = begin
            loop_end = (end // BN) * BN
        for start in tl.range(loop_begin, loop_end, BN):
            ns = start + n
            valid = tl.full((BN,), True, tl.int1)
            if tail:
                valid = ns < end
            if FP8 and PAGE >= BN:
                page = start // PAGE
                inside = start - page * PAGE + n
                crossed = inside >= PAGE
                page += crossed
                inside = tl.where(crossed, inside - PAGE, inside)
            else:
                page = ns // PAGE
                inside = ns % PAGE
            block = tl.load(TABLE + seq * T0 + page * T1, mask=valid, other=0).to(
                tl.int64
            )
            scores = tl.zeros((BM, BN), tl.float32)
            for ki in tl.range(D // BK, num_stages=QK_PIPELINE):
                kd = ki * BK + tl.arange(0, BK)
                if BK == D:
                    q = q_full
                else:
                    q = tl.load(
                        Q
                        + (first + qpos[:, None]) * Q0
                        + qh[:, None] * Q1
                        + kd[None, :],
                        mask=(m < nq * G)[:, None],
                        other=0.0,
                    )
                k = tl.load(
                    K
                    + block[None, :] * K0
                    + inside[None, :] * K1
                    + kh * K2
                    + kd[:, None] * K3,
                    mask=valid[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, k.to(q.dtype), scores)
            scores *= SCALE * 1.4426950408889634 * k_scale
            # A complete prefix tile is visible to every valid causal row when
            # sliding-window masking is disabled.
            if not (
                PREFIX_FAST
                and CAUSAL
                and WINDOW < 0
                and start + BN <= prefix
                and mb * BM + BM <= nq * G
            ):
                score_mask = valid[None, :] & (m[:, None] < nq * G)
                if CAUSAL:
                    score_mask = score_mask & (ns[None, :] <= prefix + qpos[:, None])
                if WINDOW >= 0:
                    score_mask = score_mask & (
                        ns[None, :] >= prefix + qpos[:, None] - WINDOW
                    )
                scores = tl.where(score_mask, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            p = tl.exp2(scores - safe_max[:, None])
            if PV_SPLIT:
                acc_low *= alpha[:, None]
                acc_high *= alpha[:, None]
                v_low = tl.load(
                    V
                    + block[:, None] * V0
                    + inside[:, None] * V1
                    + kh * V2
                    + half[None, :] * V3,
                    mask=valid[:, None],
                    other=0.0,
                )
                v_high = tl.load(
                    V
                    + block[:, None] * V0
                    + inside[:, None] * V1
                    + kh * V2
                    + (half[None, :] + D // 2) * V3,
                    mask=valid[:, None],
                    other=0.0,
                )
                prob = p.to(Q.dtype.element_ty)
                acc_low = tl.dot(prob, v_low.to(prob.dtype), acc_low)
                acc_high = tl.dot(prob, v_high.to(prob.dtype), acc_high)
            else:
                acc *= alpha[:, None]
                v = tl.load(
                    V
                    + block[:, None] * V0
                    + inside[:, None] * V1
                    + kh * V2
                    + d[None, :] * V3,
                    mask=valid[:, None],
                    other=0.0,
                )
                if FP8 and D == 256 and BM >= 32:
                    # Defer the scalar V dequantization to shorten the D-wide
                    # accumulator live range for the register-heavy 256-D tiles.
                    acc = tl.dot(
                        p.to(Q.dtype.element_ty), v.to(Q.dtype.element_ty), acc
                    )
                elif FP8:
                    acc += (
                        tl.dot(p.to(Q.dtype.element_ty), v.to(Q.dtype.element_ty))
                        * v_scale
                    )
                else:
                    acc = tl.dot(p.to(Q.dtype.element_ty), v, acc)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = safe_max
    norm = tl.where(denom > 0.0, denom, 1.0)
    if PV_SPLIT:
        result = tl.cat(acc_low, acc_high, dim=1) / norm[:, None]
    else:
        result = acc / norm[:, None]
    if FP8 and D == 256 and BM >= 32:
        result *= v_scale
    if SPLITS == 1:
        tl.store(
            OUT + (first + qpos[:, None]) * O0 + qh[:, None] * O1 + d[None, :],
            result,
            mask=(m < nq * G)[:, None],
        )
    else:
        row = ((seq * HK + kh) * ROWS + m) * SPLITS + split
        tl.store(
            PART + row[:, None] * D + d[None, :],
            result,
            mask=(m < nq * G)[:, None],
        )
        tl.store(
            LSE + row,
            maximum + tl.log2(denom),
            mask=m < nq * G,
        )


@triton.jit
def _segmented_attention_reduce(
    PART,
    LSE,
    OUT,
    STARTS,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QCAP: tl.constexpr,
    LIMIT_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    REDUCE_D: tl.constexpr,
):
    m, item, db = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, kh = item // HK, item % HK
    first = tl.load(STARTS + seq)
    nq = tl.load(STARTS + seq + 1) - first
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = QCAP * G
    if nq < 1 or nq > LIMIT_Q or m >= nq * G:
        return
    s = tl.arange(0, SPLITS)
    d = db * REDUCE_D + tl.arange(0, REDUCE_D)
    row = ((seq * HK + kh) * ROWS + m) * SPLITS
    lse = tl.load(LSE + row + s)
    weights = tl.exp2(lse - tl.max(lse, 0))
    part = tl.load(
        PART + (row + s)[:, None] * D + d[None, :],
        mask=d[None, :] < D,
        other=0.0,
    )
    result = tl.sum(part * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(
        OUT + (first + m // G) * O0 + (kh * G + m % G) * O1 + d,
        result,
        mask=d < D,
    )


@lru_cache(maxsize=512)
def select_segmented_config(batch, max_query_len, max_seq_len, hq, hk, dim, fp8):
    """Select the token-major cache configuration for segmented attention."""
    qcap = min(max_query_len, MAX_QUERY_LEN)
    if batch < 1 or qcap < 1:
        raise ValueError("Batch and query capacity must be positive")

    if qcap <= 2:
        if dim == 64 and qcap == 1 and batch == 1 and max_seq_len >= 131072:
            return dict(
                bm=16,
                bn=32,
                bk=64,
                splits=256,
                warps=1,
                stages=1,
            )
        if dim == 64 and max_seq_len <= 256:
            return dict(
                bm=16,
                bn=32 if fp8 else 64,
                bk=64,
                splits=1,
                warps=4,
                stages=1,
            )
        if dim == 64 and max_seq_len >= 32768:
            groups = batch * hk * triton.cdiv(qcap * (hq // hk), 16)
            return dict(
                bm=16,
                bn=32,
                bk=64,
                splits=min(64, triton.next_power_of_2(triton.cdiv(128, groups))),
                warps=4,
                stages=1,
            )
        if batch * hk == 1:
            cfg = dict(
                bm=16,
                bn=64,
                bk=min(128, dim),
                splits=32,
                warps=4,
                stages=1,
            )
        else:
            groups = batch * hk * triton.cdiv(qcap * (hq // hk), 16)
            cfg = dict(
                bm=16,
                bn=32,
                bk=dim,
                splits=min(
                    MAX_SPLITS,
                    triton.next_power_of_2(triton.cdiv(64, groups)),
                ),
                warps=4,
                stages=1,
            )
        while cfg["splits"] > 1 and max_seq_len < cfg["splits"] * cfg["bn"] * 2:
            cfg["splits"] //= 2
        return cfg

    if qcap <= 8:
        bm, bn = 16, 64
        bk, stages = 64, 1
    elif qcap <= 32:
        bm, bk = 32, 64
        bn, stages = (32, 2) if fp8 else (64, 1)
    else:
        bm, bn, bk = 32, 64, 64
        stages = 1 if fp8 else 2
    groups = batch * hk * triton.cdiv(qcap * (hq // hk), bm)
    target = 96
    if fp8 and qcap > 32:
        target = 192
    splits = min(32, triton.next_power_of_2(triton.cdiv(target, groups)))
    while splits > 1 and max_seq_len < splits * 128:
        splits //= 2
    cfg = dict(
        bm=bm,
        bn=bn,
        bk=bk,
        splits=splits,
        warps=4,
        stages=stages,
    )
    if dim == 64 and qcap >= 64 and qcap < max_seq_len <= qcap + 256:
        cfg.update(
            bm=64 if fp8 or qcap <= 256 else 128,
            bn=32,
            bk=64,
            splits=1,
            warps=4,
            stages=1,
        )
    elif dim == 64 and fp8 and max_seq_len <= 256:
        cfg.update(bn=32, splits=1)
    elif dim == 64 and qcap == 8 and max_seq_len >= 32768:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 32)
        cfg.update(
            bm=32,
            bn=32 if fp8 else 64,
            splits=min(64, triton.next_power_of_2(triton.cdiv(256, groups))),
        )
    elif dim == 64 and fp8 and qcap == 8 and max_seq_len >= 4096:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 32)
        cfg.update(
            bm=32,
            bn=64,
            splits=min(32, triton.next_power_of_2(triton.cdiv(128, groups))),
        )
    elif dim == 64 and fp8 and qcap >= 64 and max_seq_len >= 4096:
        groups = batch * hk * triton.cdiv(qcap * (hq // hk), 64)
        cfg.update(
            bm=64,
            bn=32,
            splits=min(8, triton.next_power_of_2(triton.cdiv(1024, groups))),
            stages=1,
        )
    elif dim == 128 and fp8 and qcap <= 256 and max_seq_len <= qcap:
        cfg.update(splits=1)
    elif (
        dim == 128
        and fp8
        and qcap >= 256
        and max_seq_len >= 8192
        and (
            hq // hk >= 4
            or (max_seq_len >= 32768 and (hq // hk >= 2 or qcap >= 512))
            or max_seq_len >= 131072
        )
    ):
        # Shorter prefixes need sufficient grouped-query reuse; at GQA1/Q256
        # the wider tile pays off only once the prefix reaches 131K tokens.
        cfg.update(
            bm=64,
            bn=128,
            bk=64,
            warps=8,
            stages=3,
            waves_per_eu=6,
            prefix_fast=True,
        )
        if qcap >= 512 or max_seq_len >= 32768:
            cfg["qk_pipeline"] = 3
        cfg["splits"] = _long_extend_splits(
            cfg["splits"], batch, qcap, max_seq_len, hq, hk, dim, cfg["bm"]
        )
    elif dim == 256 and fp8 and qcap >= 256 and max_seq_len >= 8192:
        # Wider tiles amortize each FP8 KV load over twice as many queries and
        # keys. Six waves is the best gfx1201 pressure/occupancy tradeoff.
        cfg.update(
            bm=64,
            bn=128,
            bk=64,
            warps=8,
            stages=1,
            waves_per_eu=6,
        )
        cfg["splits"] = _long_extend_splits(
            cfg["splits"], batch, qcap, max_seq_len, hq, hk, dim, cfg["bm"]
        )
    elif dim == 256 and not fp8 and qcap >= 1024 and max_seq_len >= 8192:
        # Two half-width PV accumulators avoid spilling this BF16 D256 tile.
        # Keep Q256 on the incumbent: the wider tile loses at that capacity.
        cfg.update(
            bm=64,
            bn=64,
            bk=64,
            warps=8,
            stages=3,
            waves_per_eu=6,
            pv_split=True,
            qk_pipeline=3,
        )
    elif dim == 128 and not fp8 and qcap > 128:
        if max_seq_len >= 8192:
            cfg.update(
                bm=128,
                bn=32,
                bk=128,
                warps=8,
                stages=1,
                waves_per_eu=6,
            )
            groups = batch * hk * triton.cdiv(qcap * (hq // hk), cfg["bm"])
            cfg["splits"] = min(16, triton.next_power_of_2(triton.cdiv(192, groups)))
        elif qcap >= 4096:
            cfg.update(
                bm=128,
                bn=32,
                bk=128,
                splits=1,
                warps=8,
                stages=1,
                waves_per_eu=6,
            )
        elif qcap > 256 or hq > 10:
            cfg.update(
                bm=64,
                bn=32,
                bk=128,
                splits=1,
                warps=4,
                stages=1,
                waves_per_eu=6,
            )
        else:
            cfg.update(bm=32, bn=64, bk=64, splits=1, warps=4, stages=2)
    return cfg


def segmented_workspace_shapes(batch, query_cap, hq, hk, dim, splits):
    if splits == 1:
        return None
    partial = (batch, hk, query_cap * (hq // hk), splits, dim)
    return partial, partial[:-1]


def reserve_segmented_attention_workspace(
    max_batch,
    hq,
    hk,
    dim,
    max_seq_len,
    *,
    max_tokens=None,
    fp8=False,
):
    """Reserve the largest selected attention workspace before graph capture.

    ``max_tokens`` bounds reachable ``(batch, query_len)`` pairs using one
    longest query and one token for each remaining sequence. Omitting it keeps
    the legacy reservation behavior for callers without scheduler limits.
    """
    if not is_workspace_manager_initialized():
        return
    largest = 0
    previous_capacity = 0
    for query_capacity in _query_capacity_buckets():
        query_lengths = {previous_capacity + 1, query_capacity}
        for query_len in query_lengths:
            if max_tokens is not None and query_len > max_tokens:
                continue
            batch_limit = max_batch
            if max_tokens is not None:
                batch_limit = min(batch_limit, max_tokens - query_len + 1)
            for batch in range(1, batch_limit + 1):
                cfg = select_segmented_config(
                    batch, query_len, max_seq_len, hq, hk, dim, fp8
                )
                splits = cfg["splits"]
                if (
                    dim == 64
                    and query_len <= 512
                    and max_seq_len >= 4096
                    and envs.VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE
                ):
                    split_limit = (
                        512
                        if batch == 1 and query_len == 1 and max_seq_len >= 131072
                        else MAX_SPLITS
                    )
                    splits = min(split_limit, 2 * splits)
                if splits > 1:
                    largest = max(
                        largest,
                        splits * batch * query_capacity * hq,
                    )
        previous_capacity = query_capacity
    if largest:
        current_workspace_manager()._reserve_simultaneous(
            ((largest * dim,), torch.float32), ((largest,), torch.float32)
        )


def can_use_segmented_attention(
    q, k, v, out, kc, vc, table, starts, lengths, k_scale, v_scale
):
    """Check tensor metadata; unsupported feature checks live in the caller."""
    if (
        k is None
        or v is None
        or q.ndim != 3
        or kc.ndim != 4
        or vc.ndim != 4
        or q.dtype not in (torch.bfloat16, torch.float16)
        or out.dtype != q.dtype
        or q.shape != out.shape
        or q.shape[-1] not in (64, 128, 256)
        or kc.dtype != vc.dtype
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or kc.dtype not in (q.dtype, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or table.ndim != 2
        or starts.ndim != 1
        or lengths.ndim != 1
        or table.shape[0] != lengths.numel()
        or starts.numel() != lengths.numel() + 1
        or table.dtype != torch.int32
        or starts.dtype != torch.int32
        or lengths.dtype != torch.int32
        or starts.stride(0) != 1
        or lengths.stride(0) != 1
        or not q.is_cuda
    ):
        return False
    hkv = kc.shape[2]
    dim = q.shape[-1]
    page = kc.shape[1]
    if hkv < 1 or q.shape[1] % hkv or not 1 <= q.shape[1] // hkv <= 16:
        return False
    if (
        k.shape != (q.shape[0], hkv, dim)
        or v.shape != k.shape
        or kc.shape != (kc.shape[0], page, hkv, dim)
        or vc.shape != kc.shape
        or any(
            t.device != q.device for t in (k, v, out, kc, vc, table, starts, lengths)
        )
        or any(t.stride(-1) != 1 for t in (q, k, v, out, kc, vc))
    ):
        return False
    if kc.element_size() == 1:
        return all(
            isinstance(s, torch.Tensor)
            and s.numel() == 1
            and s.dtype == torch.float32
            and s.device == q.device
            for s in (k_scale, v_scale)
        )
    return True


def run_segmented_attention(
    q,
    out,
    kc,
    vc,
    table,
    starts,
    lengths,
    max_query_len,
    max_seq_len,
    k_scale,
    v_scale,
    scale,
    *,
    sliding_window=-1,
    causal=True,
    sinks=None,
    config=None,
    workspace=None,
):
    """Write eligible unified-cache attention rows; leave other requests untouched."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    if batch == 0 or max_query_len == 0 or q.numel() == 0:
        return out
    query_len = min(max_query_len, MAX_QUERY_LEN)
    qcap = segmented_query_capacity(query_len)
    fp8 = kc.element_size() == 1
    hk = kc.shape[2]
    attention_span = max_seq_len
    if sliding_window >= 0:
        attention_span = min(max_seq_len, sliding_window + max_query_len)
    cfg = config or select_segmented_config(
        batch, query_len, attention_span, hq, hk, dim, fp8
    )
    splits = cfg["splits"]
    shapes = segmented_workspace_shapes(batch, qcap, hq, hk, dim, splits)
    if shapes is None:
        partial = lse = out
    elif workspace is not None:
        partial, lse = workspace
    elif is_workspace_manager_initialized():
        partial, lse = current_workspace_manager().get_simultaneous(
            (shapes[0], torch.float32), (shapes[1], torch.float32)
        )
    else:
        partial = torch.empty(shapes[0], device=q.device, dtype=torch.float32)
        lse = torch.empty(shapes[1], device=q.device, dtype=torch.float32)
    _launch_segmented_attention(
        q,
        out,
        kc,
        vc,
        table,
        starts,
        lengths,
        k_scale,
        v_scale,
        sinks,
        scale,
        cfg,
        partial,
        lse,
        qcap,
        sliding_window,
        causal,
        compile_only=False,
    )
    return out


def compile_segmented_attention(
    q,
    out,
    kc,
    vc,
    table,
    starts,
    lengths,
    max_query_len,
    k_scale,
    v_scale,
    scale,
    config,
    workspace,
    *,
    sliding_window=-1,
    causal=True,
    sinks=None,
):
    """Compile one segmented attention configuration without launching it."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    qcap = segmented_query_capacity(min(max_query_len, MAX_QUERY_LEN))
    hk = kc.shape[2]
    shapes = segmented_workspace_shapes(batch, qcap, hq, hk, dim, config["splits"])
    if shapes is None:
        partial = lse = out
    else:
        partial, lse = workspace
    _launch_segmented_attention(
        q,
        out,
        kc,
        vc,
        table,
        starts,
        lengths,
        k_scale,
        v_scale,
        sinks,
        scale,
        config,
        partial,
        lse,
        qcap,
        sliding_window,
        causal,
        compile_only=True,
    )


def _launch_segmented_attention(
    q,
    out,
    kc,
    vc,
    table,
    starts,
    lengths,
    k_scale,
    v_scale,
    sinks,
    scale,
    cfg,
    partial,
    lse,
    qcap,
    sliding_window,
    causal,
    *,
    compile_only,
):
    """Launch or compile the exact stage and reduction specializations."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    hk = kc.shape[2]
    fp8 = kc.element_size() == 1
    splits = cfg["splits"]
    stage_grid = (
        batch * hk,
        triton.cdiv(qcap * (hq // hk), cfg["bm"]),
        splits,
    )
    stage_args = (
        q,
        kc,
        vc,
        table,
        starts,
        lengths,
        k_scale,
        v_scale,
        sinks,
        partial,
        lse,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        *kc.stride(),
        *vc.stride(),
        *table.stride(),
        kc.shape[1],
        hq,
        hk,
        dim,
        qcap,
        MAX_QUERY_LEN,
        scale,
        fp8,
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        splits,
        sliding_window,
        causal,
        sinks is not None,
        cfg.get("prefix_fast", False),
        cfg.get("pv_split", False),
        cfg.get("qk_pipeline"),
    )
    stage_options = dict(
        num_warps=cfg["warps"],
        num_stages=cfg["stages"],
        waves_per_eu=cfg.get("waves_per_eu", 2),
    )
    if compile_only:
        _segmented_attention_stage.warmup(*stage_args, grid=stage_grid, **stage_options)
    else:
        _segmented_attention_stage[stage_grid](*stage_args, **stage_options)
    if splits > 1:
        reduce_d = cfg.get("reduce_d", 64 if batch * qcap * hq < 64 else dim)
        reduce_grid = (
            qcap * (hq // hk),
            batch * hk,
            triton.cdiv(dim, reduce_d),
        )
        reduce_args = (
            partial,
            lse,
            out,
            starts,
            out.stride(0),
            out.stride(1),
            hq,
            hk,
            dim,
            qcap,
            MAX_QUERY_LEN,
            splits,
            reduce_d,
        )
        reduce_options = {"num_warps": cfg.get("reduce_warps", 4)}
        if compile_only:
            _segmented_attention_reduce.warmup(
                *reduce_args, grid=reduce_grid, **reduce_options
            )
        else:
            _segmented_attention_reduce[reduce_grid](*reduce_args, **reduce_options)


def get_segmented_config(*args, **kwargs):
    """Load the tuner on demand to avoid a kernel/tuner import cycle."""
    from .segmented_attention_tuning import get_segmented_config as tuned_config

    return tuned_config(*args, **kwargs)


def segmented_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    kv_cache_dtype: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_query_len: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    *,
    sliding_window: int = -1,
    output_scale: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    causal: bool | torch.Tensor = True,
    softcap: float = 0.0,
) -> None:
    """Run segmented attention when eligible, otherwise use unified Triton."""
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        fp8_dtype = current_platform.fp8_dtype()
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

    from vllm.platforms.rocm import on_gfx1x, on_gfx12x

    segmented_pattern = (
        isinstance(causal, bool)
        and (on_gfx12x() if key_cache.element_size() == 1 else on_gfx1x())
        and not softcap
        and output_scale is None
        and (
            sinks is None
            or (
                sinks.ndim == 1
                and sinks.shape[0] == query.shape[1]
                and sinks.device == query.device
                and sinks.dtype in (torch.float16, torch.bfloat16, torch.float32)
            )
        )
    )
    if (
        segmented_pattern
        and 0 < max_query_len <= MAX_QUERY_LEN
        and (
            can_use_segmented_attention(
                query,
                key,
                value,
                output,
                key_cache,
                value_cache,
                block_table,
                query_start_loc,
                seq_lens,
                k_scale,
                v_scale,
            )
        )
    ):
        attention_span = max_seq_len
        if sliding_window >= 0:
            attention_span = min(max_seq_len, sliding_window + max_query_len)
        config = get_segmented_config(
            query.device,
            query.dtype,
            key_cache.dtype,
            query.shape[1],
            key_cache.shape[2],
            query.shape[2],
            key_cache.shape[1],
            sm_scale,
            len(seq_lens),
            max_query_len,
            attention_span,
            sliding_window,
            causal,
            has_sinks=sinks is not None,
        )
        if config is None:
            config = select_segmented_config(
                len(seq_lens),
                max_query_len,
                attention_span,
                query.shape[1],
                key_cache.shape[2],
                query.shape[2],
                key_cache.element_size() == 1,
            )
        run_segmented_attention(
            query,
            output,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            k_scale,
            v_scale,
            sm_scale,
            sliding_window=sliding_window,
            causal=causal,
            sinks=sinks,
            config=config,
        )
        return

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1) if sliding_window < 0 else (sliding_window, 0),
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        output_scale=output_scale,
        sinks=sinks,
        kv_quant_mode=get_kv_quant_mode(kv_cache_dtype),
    )
