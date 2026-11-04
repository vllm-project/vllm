# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer normalization, rotary embedding and FP8 query preparation."""

import torch

from vllm.triton_utils import tl, triton

from .summary_update import _update_decode_values, summary_layout


@triton.jit
def _rms_norm_variance(x):
    # Preserve the existing CuTeDSL RMSNorm fragment and warp reduction order.
    lane = tl.arange(0, 32)
    partial = tl.full((32,), 0, tl.float32)
    for i in tl.static_range(8):
        value = tl.gather(x, lane * 4 + i % 4 + i // 4 * 128, 0)
        partial = partial + value * value
    for i in tl.static_range(5):
        partial = partial + tl.gather(partial, lane ^ (16 >> i), 0)
    return tl.sum(tl.where(lane == 0, partial, 0), 0) / 256


@triton.jit
def _combine_stats(mean_b, variance_b, count_b, mean_a, variance_a, count_a):
    delta = mean_b - mean_a
    count = count_a + count_b
    inv_count = tl.div_rn(1.0, count)
    na = count_a * inv_count
    nb = count_b * inv_count
    mean = tl.fma(na, mean_a, nb * mean_b)
    variance = tl.fma(delta * delta * count_a, nb, variance_a + variance_b)
    return mean, variance, count


@triton.jit
def _layer_norm_stats(x):
    # Match PyTorch's vectorized LayerNorm: four values per lane, followed
    # by descending warp shuffles and the merge of the two populated warps.
    lane = tl.arange(0, 64)
    mean = tl.full((64,), 0, tl.float32)
    variance = tl.full((64,), 0, tl.float32)
    for i in tl.static_range(4):
        value = tl.gather(x, lane * 4 + i, 0)
        delta = value - mean
        mean = tl.fma(delta, 1.0 / (i + 1), mean)
        variance = tl.fma(delta, value - mean, variance)
    count = tl.full((64,), 4, tl.float32)
    for offset in tl.static_range(5):
        distance = 16 >> offset
        other = tl.where(lane % 32 + distance < 32, lane + distance, lane)
        mean, variance, count = _combine_stats(
            mean,
            variance,
            count,
            tl.gather(mean, other, 0),
            tl.gather(variance, other, 0),
            tl.gather(count, other, 0),
        )
    other = lane ^ 32
    mean, variance, count = _combine_stats(
        mean,
        variance,
        count,
        tl.gather(mean, other, 0),
        tl.gather(variance, other, 0),
        tl.gather(count, other, 0),
    )
    return (
        tl.sum(tl.where(lane == 0, mean, 0), 0),
        tl.sum(tl.where(lane == 0, variance, 0), 0) / 256,
    )


@triton.jit(do_not_specialize=["CSA_ROWS", "TOKENS"])
def _indexer_norm_rope(
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
):
    token = tl.program_id(0)
    group = tl.program_id(1) // PAD_HEADS
    head = tl.program_id(1) % PAD_HEADS
    dim = tl.arange(0, 256)
    if (group < GROUPS) & (head < HEADS):
        x = tl.load(Q + token * Q_STRIDE + (group * HEADS + head) * 256 + dim).to(
            tl.float32
        )
        variance = _rms_norm_variance(x) if WEIGHT_BIAS == 1 else tl.sum(x * x, 0) / 256
        inv = tl.rsqrt(variance + EPS)
        weight = tl.load(QWeight + dim).to(tl.float32) + WEIGHT_BIAS
        normalized = x * inv
        if WEIGHT_BIAS == 0:
            normalized = normalized.to(QWeight.dtype.element_ty).to(tl.float32)
        x = normalized * weight
        # Preserve the activation-typed norm output before RoPE.
        x = x.to(Q.dtype.element_ty).to(tl.float32)
    elif group == GROUPS:
        x = tl.load(K + token * K_STRIDE + dim).to(tl.float32)
        mean, variance = _layer_norm_stats(x)
        centered = x - mean
        inv = tl.rsqrt(variance + EPS)
        weight = tl.load(KWeight + dim).to(tl.float32)
        bias = tl.load(KBias + dim).to(tl.float32)
        x = tl.fma(weight, centered * inv, bias)
        x = x.to(K.dtype.element_ty).to(tl.float32)
    else:
        x = tl.full((256,), 0, tl.float32)

    if (head < HEADS) | (group == GROUPS):
        position = tl.load(Positions + token)
        half = ROTARY // 2
        partner_dim = tl.where(
            dim < half, dim + half, tl.where(dim < ROTARY, dim - half, dim)
        )
        partner = tl.gather(x, partner_dim, 0)
        phase = dim % half
        cos = tl.load(Rope + position * ROPE_STRIDE + phase)
        sin = tl.load(Rope + position * ROPE_STRIDE + half + phase)
        first = x * cos.to(tl.float32)
        second = tl.where(dim < half, -partner, partner) * sin.to(tl.float32)
        if NATIVE_ROPE:
            first = first.to(Q.dtype.element_ty).to(tl.float32)
            second = second.to(Q.dtype.element_ty).to(tl.float32)
        rotated = first + second
        x = tl.where(dim < ROTARY, rotated, x)
    if group == GROUPS:
        tl.store(KOut + token * 256 + dim, x)
        if UPDATE_CSA:  # noqa: SIM102 - Guard absent cache pointers at compile time.
            if token < CSA_ROWS:
                slot = tl.load(Slots + token * SLOT_STRIDE)
                z = tl.load(Z + token * Z_STRIDE + dim).to(tl.float32)
                normalized_key = x.to(K.dtype.element_ty).to(tl.float32)
                _update_decode_values(
                    slot,
                    tl.load(Positions + token),
                    tl.load(RingBase + token),
                    dim,
                    normalized_key,
                    z,
                    Mean,
                    Ring,
                    RING_CAPACITY,
                    MEAN_BLOCK,
                    MEAN_BLOCK_SIZE,
                )
    else:
        # RoPE also materializes BF16/FP16 before the FP8 conversion.
        rounded = x.to(Q.dtype.element_ty)
        offset = (group * TOKENS + token) * PAD_HEADS + head
        tl.store(QOut + offset * 256 + dim, rounded)
        w = tl.load(W + token * W_STRIDE + group * HEADS + head, head < HEADS, other=0)
        tl.store(WOut + offset, w.to(tl.float32) * HEADS**-0.5)


def indexer_norm_rope(
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
    num_groups: int = 1,
    decode_cache=None,
    decode_ring=None,
    decode_z: torch.Tensor | None = None,
    decode_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare indexer Q/K/weights and update the decode CSA state."""
    tokens, width = query.shape
    if (decode_cache is None) != (decode_ring is None):
        raise ValueError("CSA fusion needs both the summary cache and the ring")
    if decode_cache is not None:
        if decode_z is None or decode_slots is None:
            raise ValueError("CSA fusion needs decode z and slots")
    elif decode_z is not None or decode_slots is not None:
        raise ValueError("CSA decode inputs need a summary cache")
    if decode_slots is not None and decode_slots.numel() > tokens:
        raise ValueError("CSA decode rows must fit the indexer token count")
    if num_groups < 1 or width % (num_groups * 256):
        raise ValueError("Indexer query heads must divide evenly into provider groups")
    heads = width // (num_groups * 256)
    padded_heads = max(heads, 8)
    q = torch.empty(
        (num_groups, tokens, padded_heads, 256),
        device=query.device,
        dtype=torch.float8_e4m3fn,
    )
    k = torch.empty((tokens, 1, 256), device=key.device, dtype=key.dtype)
    w = torch.empty(
        (num_groups, tokens, padded_heads), device=weights.device, dtype=torch.float32
    )
    if tokens:
        _indexer_norm_rope[(tokens, num_groups * padded_heads + 1)](
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
            num_groups,
            tokens,
            cos_sin_cache.shape[1],
            cos_sin_cache.stride(0),
            eps,
            float(zero_centered),
            native_rope,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.transpose(0, 1), k, w.transpose(0, 1)
