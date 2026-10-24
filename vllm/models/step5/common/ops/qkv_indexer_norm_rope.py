# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joint attention QKV and sparse-indexer preparation for decode."""

import torch

from vllm.triton_utils import tl, triton

from .indexer_norm_rope import _indexer_norm_rope
from .summary_update import summary_layout


@triton.jit(do_not_specialize=["SLOT_ROWS"])
def _prepare_main(
    QKV,
    QWeight,
    KWeight,
    Rope,
    Positions,
    Slots,
    QOut,
    KOut,
    VOut,
    KCache,
    VCache,
    token,
    group,
    QH: tl.constexpr,
    KH: tl.constexpr,
    STRIDE: tl.constexpr,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_ROWS,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
    KScale,
    VScale,
    FP8_CACHE: tl.constexpr,
):
    DIM: tl.constexpr = 192
    LANES: tl.constexpr = 32 if ROTARY > 128 else 16
    VECTOR: tl.constexpr = 8 if ROTARY > 128 else 4
    GROUP: tl.constexpr = 128 // LANES
    VALUES: tl.constexpr = triton.cdiv(DIM, LANES * VECTOR) * VECTOR
    h = group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, 256)
    valid = (h < QH + 2 * KH)[:, None] & (d < DIM)[None, :]
    raw = tl.load(QKV + token * STRIDE + h[:, None] * DIM + d[None, :], valid, other=0)
    x = raw.to(tl.float32)
    lane = tl.arange(0, LANES)
    # Match the native QK kernel's sequential FMA and ascending XOR reduction.
    partial = tl.full((GROUP, LANES), 0, tl.float32)
    for i in tl.static_range(VALUES):
        col = lane * VECTOR + i % VECTOR + i // VECTOR * (LANES * VECTOR)
        indices = tl.broadcast_to(col[None, :], (GROUP, LANES))
        value = tl.gather(x, indices, 1)
        partial = tl.fma(value, value, partial)
    for i in tl.static_range(5 if LANES == 32 else 4):
        other = tl.broadcast_to((lane ^ (1 << i))[None, :], (GROUP, LANES))
        partial = partial + tl.gather(partial, other, 1)
    sums = tl.sum(tl.where(lane[None, :] == 0, partial, 0.0), 1)
    inverse = tl.rsqrt(tl.div_rn(sums, 192.0) + EPS)
    weight_ptr = tl.where((h < QH)[:, None], QWeight + d[None, :], KWeight + d[None, :])
    weight = (
        tl.load(weight_ptr, (h < QH + KH)[:, None] & (d < DIM)[None, :], other=0).to(
            tl.float32
        )
        + BIAS
    )
    normalized = (x * inverse[:, None]) * weight
    # Norm is rounded before RoPE; its second half uses an explicit FMA.
    normalized = normalized.to(QKV.dtype.element_ty).to(tl.float32)
    if ROTARY > 0:
        half: tl.constexpr = ROTARY // 2
        partner = tl.where(d < half, d + half, tl.where(d < ROTARY, d - half, d))
        partner_x = tl.gather(
            normalized, tl.broadcast_to(partner[None, :], (GROUP, 256)), 1
        )
        position = tl.load(Positions + token)
        cos = tl.load(Rope + position * ROPE_STRIDE + d % half).to(tl.float32)
        sin = tl.load(Rope + position * ROPE_STRIDE + half + d % half).to(tl.float32)
        low = normalized * cos[None, :] - partner_x * sin[None, :]
        high = tl.fma(partner_x, sin[None, :], normalized * cos[None, :])
        normalized = tl.where(
            (d < ROTARY)[None, :], tl.where((d < half)[None, :], low, high), normalized
        )
    result = tl.where((h < QH + KH)[:, None], normalized.to(QKV.dtype.element_ty), raw)
    # Separate stores avoid selecting pointers with different HIP layouts.
    tl.store(
        QOut + token * QH * DIM + h[:, None] * DIM + d[None, :],
        result,
        valid & (h < QH)[:, None],
    )
    tl.store(
        KOut + token * KH * DIM + (h[:, None] - QH) * DIM + d[None, :],
        result,
        valid & (h >= QH)[:, None] & (h < QH + KH)[:, None],
    )
    tl.store(
        VOut + token * KH * DIM + (h[:, None] - QH - KH) * DIM + d[None, :],
        result,
        valid & (h >= QH + KH)[:, None],
    )
    slot = tl.load(Slots + token, token < SLOT_ROWS, other=-1)
    cache_head = tl.where(h < QH + KH, h - QH, h - QH - KH)
    offset = (
        (slot // BLOCK_SIZE) * CACHE_BLOCK
        + (slot % BLOCK_SIZE) * CACHE_TOKEN
        + cache_head[:, None] * CACHE_HEAD
        + d[None, :]
    )
    if FP8_CACHE:
        scale = tl.where((h < QH + KH)[:, None], tl.load(KScale), tl.load(VScale))
        result = tl.div_rn(result.to(tl.float32), scale)
        result = tl.minimum(tl.maximum(result, -448.0), 448.0)
    tl.store(
        KCache + offset,
        result,
        valid & (h >= QH)[:, None] & (h < QH + KH)[:, None] & (slot >= 0),
    )
    tl.store(VCache + offset, result, valid & (h >= QH + KH)[:, None] & (slot >= 0))


@triton.jit(do_not_specialize=["CSA_ROWS", "TOKENS", "MAIN_SLOT_ROWS"])
def _joint_prepare(
    Q,
    K,
    W,
    QWeight,
    KWeight,
    KBias,
    Rope,
    Positions,
    QOut,
    KOut,
    WOut,
    Z,
    Slots,
    Mean,
    Ring,
    RingBase,
    UPDATE_CSA: tl.constexpr,
    CSA_ROWS,
    Z_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    MEAN_BLOCK_SIZE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    TOKENS,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_BIAS: tl.constexpr,
    NATIVE_ROPE: tl.constexpr,
    MainQKV,
    MainQWeight,
    MainKWeight,
    MainRope,
    MainSlots,
    MainQ,
    MainK,
    MainV,
    MainKC,
    MainVC,
    MAIN_QH: tl.constexpr,
    MAIN_KH: tl.constexpr,
    MAIN_STRIDE: tl.constexpr,
    MAIN_ROTARY: tl.constexpr,
    MAIN_ROPE_STRIDE: tl.constexpr,
    MAIN_SLOT_ROWS,
    MAIN_BLOCK_SIZE: tl.constexpr,
    MAIN_CACHE_BLOCK: tl.constexpr,
    MAIN_CACHE_TOKEN: tl.constexpr,
    MAIN_CACHE_HEAD: tl.constexpr,
    MAIN_EPS: tl.constexpr,
    MAIN_BIAS: tl.constexpr,
    MainKScale,
    MainVScale,
    FP8_CACHE: tl.constexpr,
):
    if tl.program_id(1) <= GROUPS * PAD_HEADS:
        _indexer_norm_rope(
            Q,
            K,
            W,
            QWeight,
            KWeight,
            KBias,
            Rope,
            Positions,
            QOut,
            KOut,
            WOut,
            Z,
            Slots,
            Mean,
            Ring,
            RingBase,
            UPDATE_CSA,
            CSA_ROWS,
            Z_STRIDE,
            SLOT_STRIDE,
            RING_CAPACITY,
            MEAN_BLOCK,
            MEAN_BLOCK_SIZE,
            Q_STRIDE,
            K_STRIDE,
            W_STRIDE,
            HEADS,
            PAD_HEADS,
            GROUPS,
            TOKENS,
            ROTARY,
            ROPE_STRIDE,
            EPS,
            WEIGHT_BIAS,
            NATIVE_ROPE,
        )
    else:
        _prepare_main(
            MainQKV,
            MainQWeight,
            MainKWeight,
            MainRope,
            Positions,
            MainSlots,
            MainQ,
            MainK,
            MainV,
            MainKC,
            MainVC,
            tl.program_id(0),
            tl.program_id(1) - GROUPS * PAD_HEADS - 1,
            MAIN_QH,
            MAIN_KH,
            MAIN_STRIDE,
            MAIN_ROTARY,
            MAIN_ROPE_STRIDE,
            MAIN_SLOT_ROWS,
            MAIN_BLOCK_SIZE,
            MAIN_CACHE_BLOCK,
            MAIN_CACHE_TOKEN,
            MAIN_CACHE_HEAD,
            MAIN_EPS,
            MAIN_BIAS,
            MainKScale,
            MainVScale,
            FP8_CACHE,
        )


def qkv_indexer_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    zero_centered: bool,
    native_rope: bool = False,
    *,
    main_inputs,
    decode_cache=None,
    decode_ring=None,
    decode_z: torch.Tensor | None = None,
    decode_slots: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Prepare indexer/attention tensors and write their separate cache states."""
    (
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        main_rotary,
        main_eps,
        main_bias,
        main_k_scale,
        main_v_scale,
    ) = main_inputs
    fp8_cache = main_kc.dtype in (torch.uint8, torch.float8_e4m3fn)
    if fp8_cache:
        main_kc = main_kc.view(torch.float8_e4m3fn)
        main_vc = main_vc.view(torch.float8_e4m3fn)
    tokens, width = query.shape
    main_q = main_qkv.new_empty((tokens, main_qh * 192))
    main_k = main_qkv.new_empty((tokens, main_kh * 192))
    main_v = torch.empty_like(main_k)
    main_group = 4 if main_rotary > 128 else 8
    if (decode_cache is None) != (decode_ring is None):
        raise ValueError("CSA fusion needs both the summary cache and the ring")
    if decode_cache is not None:
        if decode_z is None or decode_slots is None:
            raise ValueError("CSA fusion needs decode z and slots")
    elif decode_z is not None or decode_slots is not None:
        raise ValueError("CSA decode inputs need a summary cache")
    if decode_slots is not None and decode_slots.numel() > tokens:
        raise ValueError("CSA decode rows must fit the indexer token count")
    # One indexer group per local KV head; outputs are group-major.
    groups = main_kh
    if width % (groups * 256):
        raise ValueError("Indexer query heads must divide evenly into KV groups")
    heads = width // (groups * 256)
    padded_heads = max(heads, 8)
    q = torch.empty(
        (groups, tokens, padded_heads, 256),
        device=query.device,
        dtype=torch.float8_e4m3fn,
    )
    k = torch.empty((tokens, 1, 256), device=key.device, dtype=key.dtype)
    w = torch.empty(
        (groups, tokens, padded_heads), device=weights.device, dtype=torch.float32
    )
    if tokens:
        indexer_programs = groups * padded_heads + 1
        main_programs = triton.cdiv(main_qh + 2 * main_kh, main_group)
        _joint_prepare[(tokens, indexer_programs + main_programs)](
            query,
            key,
            weights,
            q_weight,
            k_weight,
            k_bias,
            cos_sin_cache,
            positions,
            q,
            k,
            w,
            decode_z,
            decode_slots,
            decode_cache.mean if decode_cache is not None else None,
            decode_ring.rows if decode_ring is not None else None,
            decode_ring.metadata().base if decode_ring is not None else None,
            decode_cache is not None,
            decode_slots.numel() if decode_slots is not None else 0,
            decode_z.stride(0) if decode_z is not None else 0,
            decode_slots.stride(0) if decode_slots is not None else 0,
            decode_ring.capacity if decode_ring is not None else 8,
            *(
                summary_layout(decode_cache.mean)
                if decode_cache is not None
                else (0, 8)
            ),
            query.stride(0),
            key.stride(0),
            weights.stride(0),
            heads,
            padded_heads,
            groups,
            tokens,
            cos_sin_cache.shape[1],
            cos_sin_cache.stride(0),
            eps,
            float(zero_centered),
            native_rope,
            main_qkv,
            main_qw,
            main_kw,
            main_rope,
            main_slots,
            main_q,
            main_k,
            main_v,
            main_kc,
            main_vc,
            main_qh,
            main_kh,
            main_qkv.stride(0),
            main_rotary,
            main_rope.stride(0),
            main_slots.numel(),
            main_kc.shape[1],
            main_kc.stride(0),
            main_kc.stride(1),
            main_kc.stride(2),
            main_eps,
            main_bias,
            main_k_scale,
            main_v_scale,
            fp8_cache,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.transpose(0, 1), k, w.transpose(0, 1), main_q, main_k, main_v
