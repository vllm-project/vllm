# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ROCm token-major segmented attention kernels and dispatch."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import get_kv_quant_mode
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from .segmented_attention_config import (
    KV_UPCAST_MODES as KV_UPCAST_MODES,
)
from .segmented_attention_config import (
    MAX_LONG_EXTEND_WORKSPACE_BYTES as MAX_LONG_EXTEND_WORKSPACE_BYTES,
)
from .segmented_attention_config import (
    MAX_QUERY_LEN as MAX_QUERY_LEN,
)
from .segmented_attention_config import (
    MAX_SPLITS as MAX_SPLITS,
)
from .segmented_attention_config import (
    MMPrefixMode,
    segmented_attention_span,
)
from .segmented_attention_config import (
    SegmentedAttentionConfig as SegmentedAttentionConfig,
)
from .segmented_attention_config import (
    SegmentedLaunchConfig as SegmentedLaunchConfig,
)
from .segmented_attention_config import (
    _long_extend_splits as _long_extend_splits,
)
from .segmented_attention_config import (
    _query_capacity_buckets as _query_capacity_buckets,
)
from .segmented_attention_config import (
    _refine_segmented_launch_config as _refine_segmented_launch_config,
)
from .segmented_attention_config import (
    _select_segmented_kv_upcast as _select_segmented_kv_upcast,
)
from .segmented_attention_config import (
    _select_segmented_launch_config as _select_segmented_launch_config,
)
from .segmented_attention_config import (
    segmented_kv_upcast_mode as segmented_kv_upcast_mode,
)
from .segmented_attention_config import (
    segmented_query_capacity as segmented_query_capacity,
)
from .segmented_attention_config import (
    select_segmented_config as select_segmented_config,
)
from .triton_attention_helpers import compute_kv_seq_mask
from .triton_unified_attention import unified_attention


@triton.jit
def _segmented_kv_upcast(
    value, dtype: tl.constexpr, native: tl.constexpr, pack: tl.constexpr = 1
):
    if native:
        if pack == 4 and dtype == tl.bfloat16:
            width: tl.constexpr = value.shape[-1]
            rows: tl.constexpr = value.numel // width
            raw = value.to(tl.uint8, bitcast=True).to(tl.uint32)
            raw = tl.reshape(raw, (rows, width // 4, 4))
            packed = tl.sum(raw << (tl.arange(0, 4)[None, None, :] * 8), 2)
            b0, b1, b2, b3 = tl.inline_asm_elementwise(
                "v_cvt_f32_fp8 $0, $4;\n"
                "v_cvt_f32_fp8 $1, $4 byte_sel:1;\n"
                "v_cvt_f32_fp8 $2, $4 byte_sel:2;\n"
                "v_cvt_f32_fp8 $3, $4 byte_sel:3;\n"
                "v_lshrrev_b32 $0, 16, $0;\n"
                "v_lshrrev_b32 $1, 16, $1;\n"
                "v_lshrrev_b32 $2, 16, $2;\n"
                "v_lshrrev_b32 $3, 16, $3;",
                constraints="=&v,=&v,=&v,=&v,v",
                args=[packed],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=True,
                pack=1,
            )
            bits = tl.reshape(tl.join(tl.join(b0, b2), tl.join(b1, b3)), value.shape)
            return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        raw = value.to(tl.uint8, bitcast=True).to(tl.uint32)
        if dtype == tl.bfloat16:
            # Every OCP E4M3FN value fits exactly in BF16. Keep the assembly
            # result 32-bit, then narrow the bits, to avoid true16 operands
            # and a redundant rounded FP32-to-BF16 conversion.
            bits = tl.inline_asm_elementwise(
                "v_cvt_f32_fp8 $0, $1;\nv_lshrrev_b32 $0, 16, $0;",
                constraints="=v,v",
                args=[raw],
                dtype=tl.uint32,
                is_pure=True,
                pack=1,
            )
            return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        converted = tl.inline_asm_elementwise(
            "v_cvt_f32_fp8 $0, $1;",
            constraints="=v,v",
            args=[raw],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return converted.to(dtype)
    return value.to(dtype)


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
    MM_PREFIX,
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
    HAS_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    MM_PREFIX_CLAMP_SW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    PREFIX_FAST: tl.constexpr,
    PV_SPLIT: tl.constexpr,
    QK_PIPELINE: tl.constexpr,
    QK_UNROLL: tl.constexpr,
    NATIVE_FP8_UPCAST: tl.constexpr,
    NATIVE_FP8_PACK: tl.constexpr,
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
    if HAS_MM_PREFIX:
        # A query tile can see future keys in its image, and unclamped image
        # ranges can also reopen keys before the sliding window. Widen traversal
        # before partitioning; changing just the score mask would miss them.
        first_query = prefix + (mb * BM) // G
        last_query = prefix + (tile_rows - 1) // G
        if MAX_MM_RANGES > 1:
            image_first = tl.full((BM,), 2147483647, tl.int32)
            image_last = tl.full((BM,), -1, tl.int32)
        for ri in range(MAX_MM_RANGES):
            range_start = tl.load(MM_PREFIX + seq * MAX_MM_RANGES * 2 + ri * 2)
            range_stop = tl.load(MM_PREFIX + seq * MAX_MM_RANGES * 2 + ri * 2 + 1)
            intersects = (
                (range_start < range_stop)
                & (first_query <= range_stop)
                & (last_query >= range_start)
            )
            if MAX_MM_RANGES > 1:
                in_image = (
                    (range_start < range_stop)
                    & (prefix + qpos >= range_start)
                    & (prefix + qpos <= range_stop)
                )
                image_first = tl.minimum(
                    image_first, tl.where(in_image, range_start, 2147483647)
                )
                image_last = tl.maximum(image_last, tl.where(in_image, range_stop, -1))
            range_end = tl.where(
                intersects,
                tl.maximum(range_end, tl.minimum(sequence, range_stop + 1)),
                range_end,
            )
            if WINDOW >= 0 and not MM_PREFIX_CLAMP_SW:
                image_begin = tl.maximum(0, range_start) // BN * BN
                range_begin = tl.where(
                    intersects, tl.minimum(range_begin, image_begin), range_begin
                )
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
            for ki in tl.range(
                D // BK, num_stages=QK_PIPELINE, loop_unroll_factor=QK_UNROLL
            ):
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
                if NATIVE_FP8_UPCAST and NATIVE_FP8_PACK == 4:
                    k = _segmented_kv_upcast(k.T, q.dtype, True, 4).T
                else:
                    k = _segmented_kv_upcast(
                        k, q.dtype, NATIVE_FP8_UPCAST, NATIVE_FP8_PACK
                    )
                scores = tl.dot(q, k, scores)
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
                if HAS_MM_PREFIX and MAX_MM_RANGES > 1:
                    # Ranges sharing a query form a connected union. Resolve
                    # its bounds once per row, outside the KV traversal loop.
                    base = tl.full((BM, BN), True, tl.int1)
                    if CAUSAL:
                        base &= ns[None, :] <= prefix + qpos[:, None]
                    image = (ns[None, :] >= image_first[:, None]) & (
                        ns[None, :] <= image_last[:, None]
                    )
                    if WINDOW >= 0:
                        past = ns[None, :] >= prefix + qpos[:, None] - WINDOW
                        base &= past
                        if not CAUSAL:
                            base &= ns[None, :] - prefix - qpos[:, None] <= WINDOW
                        if MM_PREFIX_CLAMP_SW:
                            image &= past
                    score_mask &= base | image
                elif HAS_MM_PREFIX:
                    score_mask &= compute_kv_seq_mask(
                        (prefix + qpos)[:, None],
                        ns,
                        seq,
                        sequence,
                        MM_PREFIX,
                        SLIDING_WINDOW=WINDOW + 1,
                        USE_MM_PREFIX=True,
                        MAX_MM_RANGES=MAX_MM_RANGES,
                        USE_CAUSAL=CAUSAL,
                        MM_PREFIX_CLAMP_SW=MM_PREFIX_CLAMP_SW,
                    )
                else:
                    if CAUSAL:
                        score_mask &= ns[None, :] <= prefix + qpos[:, None]
                    if WINDOW >= 0:
                        score_mask &= ns[None, :] >= prefix + qpos[:, None] - WINDOW
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
                acc_low = tl.dot(
                    prob,
                    _segmented_kv_upcast(
                        v_low, prob.dtype, NATIVE_FP8_UPCAST, NATIVE_FP8_PACK
                    ),
                    acc_low,
                )
                acc_high = tl.dot(
                    prob,
                    _segmented_kv_upcast(
                        v_high, prob.dtype, NATIVE_FP8_UPCAST, NATIVE_FP8_PACK
                    ),
                    acc_high,
                )
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
                if FP8 and ((D == 256 and BM >= 32) or (D == 512 and G == 8)):
                    # Defer the scalar V dequantization to shorten the D-wide
                    # accumulator live range for the register-heavy wide tiles.
                    acc = tl.dot(
                        p.to(Q.dtype.element_ty),
                        _segmented_kv_upcast(
                            v, Q.dtype.element_ty, NATIVE_FP8_UPCAST, NATIVE_FP8_PACK
                        ),
                        acc,
                    )
                elif FP8:
                    acc += (
                        tl.dot(
                            p.to(Q.dtype.element_ty),
                            _segmented_kv_upcast(
                                v,
                                Q.dtype.element_ty,
                                NATIVE_FP8_UPCAST,
                                NATIVE_FP8_PACK,
                            ),
                        )
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
    if FP8 and ((D == 256 and BM >= 32) or (D == 512 and (G == 8 or PV_SPLIT))):
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


def supports_segmented_native_fp8(dtype: torch.dtype, kv_dtype: torch.dtype) -> bool:
    """Native OCP conversion is available for BF16 queries on gfx12."""
    if dtype != torch.bfloat16 or kv_dtype != torch.float8_e4m3fn:
        return False
    from vllm.platforms.rocm import on_gfx12x

    return on_gfx12x()


def segmented_workspace_shapes(batch, query_cap, hq, hk, dim, splits):
    if splits == 1:
        return None
    partial = (batch, hk, query_cap * (hq // hk), splits, dim)
    return partial, partial[:-1]


@lru_cache(maxsize=128)
def segmented_attention_workspace_size(
    max_batch,
    hq,
    hk,
    dim,
    max_seq_len,
    *,
    max_tokens=None,
    fp8=False,
    autotune=False,
    mm_prefix=False,
    sliding_window=-1,
):
    """Size the largest selected attention workspace before graph capture.

    ``max_tokens`` bounds reachable ``(batch, query_len)`` pairs using one
    longest query and one token for each remaining sequence. Omitting it keeps
    the legacy reservation behavior for callers without scheduler limits.
    ``autotune`` reserves additional splits for tuned D64 candidates.
    """
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
                configs: list[SegmentedLaunchConfig] = [
                    select_segmented_config(
                        batch, query_len, max_seq_len, hq, hk, dim, fp8
                    )
                ]
                if mm_prefix:
                    configs.extend(
                        select_segmented_config(
                            batch,
                            query_len,
                            max_seq_len,
                            hq,
                            hk,
                            dim,
                            fp8,
                            sliding_window=sliding_window,
                            native_fp8=native,
                            mm_prefix_mode="clamped",
                            mm_prefix_max_span=max_seq_len,
                            mm_prefix_max_ranges=1,
                        )
                        for native in (False, True)
                    )
                    if autotune:
                        configs.append(
                            _select_segmented_launch_config(
                                batch, query_len, max_seq_len, hq, hk, dim, fp8
                            )
                        )
                splits = max(cfg["splits"] for cfg in configs)
                if dim == 64 and query_len <= 512 and max_seq_len >= 4096 and autotune:
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
    return largest * dim, largest


def get_segmented_attention_workspace(device, sizes):
    """Keep scratch storage stable and separate for each runtime lane/ubatch."""

    def allocate():
        return tuple(
            torch.empty(size, device=device, dtype=torch.float32) for size in sizes
        )

    if is_workspace_manager_initialized():
        return current_workspace_manager().get_persistent_resource(
            ("rocm_segmented_attention", device, sizes), allocate
        )
    return allocate()


def can_use_segmented_attention(
    q, k, v, out, kc, vc, table, starts, lengths, k_scale, v_scale
):
    """Check tensor metadata; unsupported feature checks live in the caller."""
    # Cache readers, including Gemma MTP layers, do not project their own K/V.
    if (k is None) != (v is None):
        return False
    projected_kv = () if k is None else (k, v)
    if (
        q.ndim != 3
        or kc.ndim != 4
        or vc.ndim != 4
        or q.dtype not in (torch.bfloat16, torch.float16)
        or out.dtype != q.dtype
        or q.shape != out.shape
        or q.shape[-1] not in (64, 128, 256, 512)
        or kc.dtype != vc.dtype
        or any(t.dtype != q.dtype for t in projected_kv)
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
        any(t.shape != (q.shape[0], hkv, dim) for t in projected_kv)
        or kc.shape != (kc.shape[0], page, hkv, dim)
        or vc.shape != kc.shape
        or any(
            t.device != q.device
            for t in (*projected_kv, out, kc, vc, table, starts, lengths)
        )
        or any(t.stride(-1) != 1 for t in (q, *projected_kv, out, kc, vc))
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
    mm_prefix_range: torch.Tensor | None = None,
    mm_prefix_clamp_sliding_window: bool = False,
    mm_prefix_max_span: int = 0,
    config: SegmentedLaunchConfig | None = None,
    workspace=None,
) -> torch.Tensor:
    """Write eligible unified-cache attention rows; leave other requests untouched."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    if batch == 0 or max_query_len == 0 or q.numel() == 0:
        return out
    query_len = min(max_query_len, MAX_QUERY_LEN)
    qcap = segmented_query_capacity(query_len)
    fp8 = kc.element_size() == 1
    hk = kc.shape[2]
    mm_mode: MMPrefixMode = (
        "none"
        if mm_prefix_range is None
        else "clamped"
        if mm_prefix_clamp_sliding_window
        else "unclamped"
    )
    attention_span = segmented_attention_span(
        max_seq_len, max_query_len, sliding_window, mm_mode, mm_prefix_max_span
    )
    cfg = config or select_segmented_config(
        batch,
        query_len,
        attention_span,
        hq,
        hk,
        dim,
        fp8,
        sliding_window=sliding_window,
        native_fp8=supports_segmented_native_fp8(q.dtype, kc.dtype),
        mm_prefix_mode=mm_mode,
        mm_prefix_max_span=mm_prefix_max_span,
        mm_prefix_max_ranges=(
            mm_prefix_range.shape[1] if mm_prefix_range is not None else 0
        ),
    )
    splits = cfg["splits"]
    shapes = segmented_workspace_shapes(batch, qcap, hq, hk, dim, splits)
    if shapes is None:
        partial = lse = out
    elif workspace is not None:
        partial, lse = (
            buffer.view(-1)[: shape.numel()].view(shape)
            for buffer, shape in zip(workspace, map(torch.Size, shapes))
        )
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
        mm_prefix_range,
        mm_prefix_clamp_sliding_window,
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
    config: SegmentedLaunchConfig,
    workspace,
    *,
    sliding_window=-1,
    causal=True,
    sinks=None,
    mm_prefix_range: torch.Tensor | None = None,
    mm_prefix_clamp_sliding_window: bool = False,
) -> None:
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
        mm_prefix_range,
        mm_prefix_clamp_sliding_window,
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
    cfg: SegmentedLaunchConfig,
    partial,
    lse,
    qcap,
    sliding_window,
    causal,
    mm_prefix_range,
    mm_prefix_clamp_sliding_window,
    *,
    compile_only,
) -> None:
    """Launch or compile the exact stage and reduction specializations."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    hk = kc.shape[2]
    fp8 = kc.element_size() == 1
    splits = cfg["splits"]
    mode = segmented_kv_upcast_mode(cfg)
    native_fp8_upcast = mode != "normal" and supports_segmented_native_fp8(
        q.dtype, kc.dtype
    )
    stage_grid = (
        batch * hk,
        triton.cdiv(qcap * (hq // hk), cfg["bm"]),
        splits,
    )
    # Query capacity only indexes split scratch; single-pass kernels share it.
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
        mm_prefix_range,
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
        qcap if splits > 1 else 1,
        MAX_QUERY_LEN,
        scale,
        fp8,
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        splits,
        sliding_window,
        causal,
        mm_prefix_range is not None,
        mm_prefix_range.shape[1] if mm_prefix_range is not None else 0,
        mm_prefix_range is not None and mm_prefix_clamp_sliding_window,
        sinks is not None,
        cfg.get("prefix_fast", False) and sliding_window < 0,
        cfg.get("pv_split", False),
        None
        if mm_prefix_range is not None and cfg["bk"] == dim
        else cfg.get("qk_pipeline"),
        None
        if mm_prefix_range is not None and cfg["bk"] == dim
        else cfg.get("qk_unroll"),
        bool(native_fp8_upcast),
        4 if native_fp8_upcast and mode == "native4" else 1,
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


def get_segmented_config(*args: Any, **kwargs: Any) -> SegmentedAttentionConfig | None:
    """Load the tuner on demand to avoid a kernel/tuner import cycle."""
    from vllm.model_executor.warmup.rocm_segmented_attn_autotune_warmup import (
        get_segmented_config as tuned_config,
    )

    return tuned_config(*args, **kwargs)


def segmented_attention(
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
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
    mm_prefix_range: torch.Tensor | None = None,
    mm_prefix_clamp_sliding_window: bool = False,
    mm_prefix_max_span: int = 0,
    workspace=None,
    use_tuned_config: bool = True,
) -> None:
    """Run segmented attention when eligible, otherwise use unified Triton."""
    if mm_prefix_range is not None:
        if (
            mm_prefix_range.ndim != 3
            or mm_prefix_range.shape[0] != seq_lens.numel()
            or mm_prefix_range.shape[2] != 2
            or mm_prefix_range.dtype != torch.int32
            or mm_prefix_range.device != query.device
        ):
            raise ValueError(
                "mm_prefix_range must be an int32 tensor on the query device "
                "with shape (num_seqs, max_ranges, 2)"
            )
        mm_prefix_range = (
            mm_prefix_range.contiguous() if mm_prefix_range.shape[1] else None
        )
    if mm_prefix_clamp_sliding_window and max_query_len == 1:
        # A single query is the last cached token. Its image has no future
        # cached keys, and clamping prevents reopening past-window keys.
        mm_prefix_range = None
        mm_prefix_max_span = 0
    if mm_prefix_range is None:
        mm_prefix_max_span = 0
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        fp8_dtype = current_platform.fp8_dtype()
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

    from vllm.platforms.rocm import on_gfx12x, on_rdna

    segmented_pattern = (
        isinstance(causal, bool)
        and (on_gfx12x() if key_cache.element_size() == 1 else on_rdna())
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
        mm_mode: MMPrefixMode = (
            "none"
            if mm_prefix_range is None
            else "clamped"
            if mm_prefix_clamp_sliding_window
            else "unclamped"
        )
        attention_span = segmented_attention_span(
            max_seq_len, max_query_len, sliding_window, mm_mode, mm_prefix_max_span
        )
        config = None
        if use_tuned_config:
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
                num_query_tokens=query.shape[0],
                mm_prefix_mode=mm_mode,
                mm_prefix_max_span=mm_prefix_max_span,
                mm_prefix_max_ranges=(
                    mm_prefix_range.shape[1] if mm_prefix_range is not None else 0
                ),
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
                sliding_window=sliding_window,
                native_fp8=supports_segmented_native_fp8(query.dtype, key_cache.dtype),
                mm_prefix_mode=mm_mode,
                mm_prefix_max_span=mm_prefix_max_span,
                mm_prefix_max_ranges=(
                    mm_prefix_range.shape[1] if mm_prefix_range is not None else 0
                ),
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
            mm_prefix_range=mm_prefix_range,
            mm_prefix_clamp_sliding_window=mm_prefix_clamp_sliding_window,
            mm_prefix_max_span=mm_prefix_max_span,
            config=config,
            workspace=workspace,
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
        mm_prefix_range=mm_prefix_range,
        mm_prefix_clamp_sliding_window=mm_prefix_clamp_sliding_window,
    )
