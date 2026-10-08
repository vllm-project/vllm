# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton store kernel for the mxfp4_mla KV cache.

``concat_and_cache_mla`` takes a single per-tensor scale, so it cannot write
MXFP4's per-group scales. A row is packed data then scales,
``row_bytes(d) == d // 2 + d // 32``, addressed as ``slot * row_bytes``.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mxfp4_mla import (
    E2M1_MAX,
    E2M1_VALUES,
    E8M0_BIAS,
    GROUP_SIZE,
    row_bytes,
    scale_region_offset,
)

# Triton's JIT only reads module globals that are ``tl.constexpr`` instances.
_E2M1_MAX = tl.constexpr(E2M1_MAX)
_E8M0_BIAS = tl.constexpr(E8M0_BIAS)

# Midpoints between consecutive E2M1 magnitudes. Summing ``(a > m)`` over these
# is exactly ``torch.bucketize(a, midpoints, right=False)``: a value sitting on
# a midpoint rounds *down*, matching the reference and AITER.
_M0, _M1, _M2, _M3, _M4, _M5, _M6 = (
    tl.constexpr((lo + hi) / 2) for lo, hi in zip(E2M1_VALUES, E2M1_VALUES[1:])
)


@triton.jit
def _ceil_pow2_exponent(v):
    """Smallest integer ``e`` with ``2**e >= v``, for positive fp32 ``v``.

    Done on the bit pattern rather than with ``log2``: fp32 ``log2`` is not
    exactly representable at powers of two, and rounding one of those up would
    waste a whole exponent step and halve the usable E2M1 range for that group.

    For ``v = m * 2**e`` with ``m`` in ``[1, 2)``, the biased exponent field is
    ``e + 127`` and the mantissa field is zero exactly when ``v`` is a power of
    two. So ``ceil(log2(v))`` is ``e`` when the mantissa is zero and ``e + 1``
    otherwise.
    """
    bits = v.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    return tl.where(mantissa == 0, exponent, exponent + 1)


@triton.jit
def _quantize_e2m1(normalized):
    """fp32 values pre-divided by their group scale -> 4-bit E2M1 codes."""
    a = tl.abs(normalized)
    index = (
        (a > _M0).to(tl.int32)
        + (a > _M1).to(tl.int32)
        + (a > _M2).to(tl.int32)
        + (a > _M3).to(tl.int32)
        + (a > _M4).to(tl.int32)
        + (a > _M5).to(tl.int32)
        + (a > _M6).to(tl.int32)
    )
    # The sign bit is kept even when the magnitude rounds to zero, so a small
    # negative encodes as -0.0 (code 8). This matches AITER's reference exactly;
    # see mxfp4_mla.quantize_to_e2m1_codes.
    sign = (normalized < 0).to(tl.int32) << 3
    return (index | sign).to(tl.uint8)


@triton.jit
def _mxfp4_mla_store_kernel(
    kv_ptr,  # [num_tokens, latent] bf16/fp16/fp32
    slot_mapping_ptr,  # [num_tokens] int64
    cache_ptr,  # uint8, flat; row pitch ROW_BYTES
    kv_stride_token,
    num_slots,
    LATENT: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HALF_GROUP: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
):
    # int64: program_id is int32, so a token-major offset of token *
    # kv_stride_token wraps once the source holds more than 2**31 elements --
    # 4.19M tokens at a 512-wide latent. That is reachable when the whole cache
    # is filled in one launch, and it faults rather than degrades.
    token = tl.program_id(0).to(tl.int64)

    # Slot guard. block_table fills the mapping with PAD_SLOT_ID = -1 before
    # each step, so CUDA-graph batch padding arrives as negative slots. Writing
    # one would compute a negative row base and scribble outside the cache --
    # silent corruption, not a crash. The read kernels guard both ends; match
    # them here.
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0 or slot >= num_slots:
        return

    group = tl.arange(0, NUM_GROUPS)[:, None]
    pair = tl.arange(0, HALF_GROUP)[None, :]

    # Load the group's even and odd elements separately: the packed byte holds
    # (even, odd) as (low, high) nibbles, and deriving absmax from both halves
    # avoids a third read of the same row.
    base = kv_ptr + token * kv_stride_token + group * GROUP
    even = tl.load(base + 2 * pair).to(tl.float32)
    odd = tl.load(base + 2 * pair + 1).to(tl.float32)

    absmax = tl.maximum(
        tl.max(tl.abs(even), axis=1), tl.max(tl.abs(odd), axis=1)
    )  # [NUM_GROUPS]

    # RoundUp: scale = ceil_pow2(absmax / 6.0). Guarantees the scaled peak
    # lands in (3, 6], so nothing saturates at the top code.
    shared_exp = _ceil_pow2_exponent(absmax / _E2M1_MAX)
    encoded = tl.minimum(tl.maximum(shared_exp + _E8M0_BIAS, 1), 254)
    # An all-zero group has no meaningful scale; pin it to 2**0 so the encoding
    # is deterministic and matches the reference.
    encoded = tl.where(absmax == 0, _E8M0_BIAS, encoded)

    # Multiply by the reciprocal rather than divide: 2**-k is exact, so this is
    # bit-identical to dividing by the scale and cheaper.
    inv_scale = tl.exp2(-(encoded - _E8M0_BIAS).to(tl.float32))[:, None]
    low = _quantize_e2m1(even * inv_scale).to(tl.int32)
    high = _quantize_e2m1(odd * inv_scale).to(tl.int32)
    packed = ((low & 0x0F) | ((high & 0x0F) << 4)).to(tl.uint8)

    row = cache_ptr + slot * ROW_BYTES
    tl.store(row + group * HALF_GROUP + pair, packed)
    tl.store(
        row + SCALE_OFFSET + tl.arange(0, NUM_GROUPS),
        encoded.to(tl.uint8),
    )


def store_mxfp4_mla(
    kv_c_normed: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
    group: int = GROUP_SIZE,
) -> None:
    """Quantize the MLA latent to MXFP4 and write it into the paged cache.

    Args:
        kv_c_normed: ``[num_tokens, latent]``, the post-RMSNorm latent.
        slot_mapping: ``[num_tokens]`` global slot indices; negative entries
            (``PAD_SLOT_ID``) are skipped.
        cache: ``uint8``, viewable as ``[num_slots, row_bytes(latent)]``.
        group: number of latent elements sharing one E8M0 scale.

    """
    if kv_c_normed.ndim != 2:
        raise ValueError(
            f"expected kv_c_normed=[tokens, latent], got {kv_c_normed.shape}"
        )
    if cache.dtype != torch.uint8:
        raise ValueError(f"expected a uint8 cache, got {cache.dtype}")

    num_tokens, latent = kv_c_normed.shape
    pitch = row_bytes(latent, group)
    flat = cache.view(-1)
    if flat.numel() % pitch:
        raise ValueError(f"cache of {flat.numel()} bytes is not a multiple of {pitch}")
    num_slots = flat.numel() // pitch

    if num_tokens == 0:
        return

    _mxfp4_mla_store_kernel[(num_tokens,)](
        kv_c_normed,
        slot_mapping,
        flat,
        kv_c_normed.stride(0),
        num_slots,
        LATENT=latent,
        GROUP=group,
        NUM_GROUPS=latent // group,
        HALF_GROUP=group // 2,
        ROW_BYTES=pitch,
        SCALE_OFFSET=scale_region_offset(latent),
        num_warps=4,
    )
