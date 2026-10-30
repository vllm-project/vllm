# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``mxfp4_qsa`` main KV-cache format for the AMD QSA path.

OCP MXFP4 with 32-element groups: E2M1 codes packed two per byte (the low
nibble holds the lower index) and one E8M0 scale per group. One (token, KV
head) record holds K and V, ``record_bytes(d) == 2 * (d // 2 + d // 32)``, 272
bytes for ``head_dim == 256``.

The layer view is the usual ``[blocks, tokens, heads, record_bytes]``, but a
block is stored split rather than as consecutive records. With ``R = tokens *
heads`` records in a block and record index ``r = (token * stride_token + head
* stride_head) // record_bytes``::

    [0,             R * d/2)          K data,   d/2 bytes at r * d/2
    [R * d/2,       R * d)            V data,   d/2 bytes at r * d/2
    [R * d,         R * (d + d/32))   K scales, d/32 bytes at r * d/32
    [R * (d+d/32),  R * (d + d/16))   V scales, d/32 bytes at r * d/32

so every K or V row is one aligned ``d/2``-byte line (128 bytes at d = 256)
and the sparse gather never straddles a row into its scales. The quantizer
matches ``mxfp4_mla`` (vllm-project/vllm#60321): the scale is
``ceil_pow2(absmax / 6)`` so nothing saturates, and E2M1 rounding is
nearest-even.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

GROUP_SIZE = 32
E8M0_BIAS = tl.constexpr(127)
_E2M1_MAX = tl.constexpr(6.0)
# Midpoints between consecutive E2M1 magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6).
# A value on a midpoint goes to the even code index, so the odd-index
# midpoints compare with ``>=``.
_M0 = tl.constexpr(0.25)
_M1 = tl.constexpr(0.75)
_M2 = tl.constexpr(1.25)
_M3 = tl.constexpr(1.75)
_M4 = tl.constexpr(2.5)
_M5 = tl.constexpr(3.5)
_M6 = tl.constexpr(5.0)


def row_bytes(head_dim: int) -> int:
    """Bytes of one packed K (or V) row: data then scales."""
    if head_dim % GROUP_SIZE:
        raise ValueError(f"head dim {head_dim} is not a multiple of {GROUP_SIZE}")
    return head_dim // 2 + head_dim // GROUP_SIZE


def record_bytes(head_dim: int) -> int:
    """Bytes one (token, KV head) record occupies in the cache: K plus V."""
    return 2 * row_bytes(head_dim)


def head_dim_from_record_bytes(num_bytes: int) -> int:
    head_dim = num_bytes * 16 // 17
    if record_bytes(head_dim) != num_bytes:
        raise ValueError(f"{num_bytes} bytes is not an mxfp4_qsa record")
    return head_dim


def block_records(kv_cache: torch.Tensor) -> int:
    """Records per block of a ``[blocks, tokens, heads, record]`` uint8 view.

    The split layout indexes records across the whole block, so the block's
    ``tokens x heads`` records must be dense, in either token-major or
    head-major order.
    """
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 4:
        raise ValueError("mxfp4_qsa caches are [blocks, tokens, heads, bytes] uint8")
    _, tokens, heads, width = kv_cache.shape
    head_dim_from_record_bytes(width)
    strides = (kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3))
    if strides not in ((heads * width, width, 1), (width, tokens * width, 1)):
        raise ValueError(
            f"mxfp4_qsa cache strides {kv_cache.stride()} are not a dense block "
            f"of {tokens} x {heads} records"
        )
    return tokens * heads


@triton.jit
def _ceil_pow2_exponent(v):
    """Smallest integer ``e`` with ``2**e >= v``, for positive fp32 ``v``.

    Read off the bit pattern: fp32 ``log2`` is inexact at powers of two, and
    rounding one of those up would waste a whole E2M1 binade for the group.
    """
    bits = v.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    return tl.where(mantissa == 0, exponent, exponent + 1)


@triton.jit
def _quantize_e2m1(normalized):
    """fp32 values already divided by their group scale -> 4-bit E2M1 codes."""
    a = tl.abs(normalized)
    index = (
        (a > _M0).to(tl.int32)
        + (a >= _M1).to(tl.int32)
        + (a > _M2).to(tl.int32)
        + (a >= _M3).to(tl.int32)
        + (a > _M4).to(tl.int32)
        + (a >= _M5).to(tl.int32)
        + (a > _M6).to(tl.int32)
    )
    # The sign survives a zero magnitude (-0.0 is code 8), as in AITER.
    sign = (normalized < 0).to(tl.int32) << 3
    return index | sign


@triton.jit
def mxfp4_quantize_groups(even, odd):
    """Quantize ``[groups, 16]`` even/odd element pairs of 32-element groups.

    Returns the packed bytes ``[groups, 16]`` and the E8M0 scales ``[groups]``.
    """
    absmax = tl.maximum(tl.max(tl.abs(even), axis=1), tl.max(tl.abs(odd), axis=1))
    shared_exp = _ceil_pow2_exponent(absmax / _E2M1_MAX)
    encoded = tl.minimum(tl.maximum(shared_exp + E8M0_BIAS, 1), 254)
    # An all-zero group has no meaningful scale; pin it to 2**0.
    encoded = tl.where(absmax == 0, E8M0_BIAS, encoded)
    # 2**-k is exact, so multiplying matches dividing by the scale.
    inv_scale = tl.exp2(-(encoded - E8M0_BIAS).to(tl.float32))[:, None]
    low = _quantize_e2m1(even * inv_scale)
    high = _quantize_e2m1(odd * inv_scale)
    packed = ((low & 0x0F) | ((high & 0x0F) << 4)).to(tl.uint8)
    return packed, encoded.to(tl.uint8)


@triton.jit
def _e2m1_to_bf16(code, scale):
    """Decode E2M1 codes times ``2**(scale - 127)`` to BF16, exactly.

    For magnitude index ``m >= 2`` the value is ``(1 + (m & 1) / 2) * 2**((m >>
    1) - 1)``; ``m == 1`` is 0.5. Both put the biased exponent at ``(m >> 1) +
    scale - 1``. The writer clamps scales to ``>= 1``, so the exponent field
    never goes negative.
    """
    magnitude = code & 7
    exponent = (magnitude >> 1) + scale - 1
    mantissa = tl.where(magnitude >= 2, magnitude & 1, 0)
    bits = tl.where(magnitude == 0, 0, (exponent << 7) | (mantissa << 6))
    bits = bits | ((code & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def mxfp4_dequant_rows(packed, scales, HEAD_DIM: tl.constexpr):
    """``[rows, HEAD_DIM // 2]`` bytes and ``[rows, groups]`` scales -> BF16."""
    ROWS: tl.constexpr = packed.shape[0]
    GROUPS: tl.constexpr = HEAD_DIM // 32
    packed = tl.reshape(packed.to(tl.int32), (ROWS, GROUPS, 16))
    exponent = scales.to(tl.int32)[:, :, None]
    low = _e2m1_to_bf16(packed & 0x0F, exponent)
    high = _e2m1_to_bf16(packed >> 4, exponent)
    return tl.reshape(tl.join(low, high), (ROWS, HEAD_DIM))


@triton.jit
def _mxfp4_qsa_store_kernel(
    key_ptr,
    value_ptr,
    slot_mapping_ptr,
    cache_ptr,
    stride_key_token,
    stride_key_head,
    stride_value_token,
    stride_value_head,
    stride_cache_block,
    stride_cache_token,
    stride_cache_head,
    num_slots,
    BLOCK_RECORDS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0 or slot >= num_slots:
        return
    GROUPS: tl.constexpr = HEAD_DIM // 32
    HALF_D: tl.constexpr = HEAD_DIM // 2
    group = tl.arange(0, GROUPS)[:, None]
    pair = tl.arange(0, 16)[None, :]
    element = group * 32 + 2 * pair

    block = slot // PAGE_SIZE
    record = (
        (slot % PAGE_SIZE) * stride_cache_token + head * stride_cache_head
    ) // RECORD_BYTES
    base = cache_ptr + block * stride_cache_block
    byte = group * 16 + pair

    key = key_ptr + token * stride_key_token + head * stride_key_head
    packed, scales = mxfp4_quantize_groups(
        tl.load(key + element).to(tl.float32),
        tl.load(key + element + 1).to(tl.float32),
    )
    tl.store(base + record * HALF_D + byte, packed)
    tl.store(
        base + BLOCK_RECORDS * HEAD_DIM + record * GROUPS + tl.arange(0, GROUPS),
        scales,
    )

    value = value_ptr + token * stride_value_token + head * stride_value_head
    packed, scales = mxfp4_quantize_groups(
        tl.load(value + element).to(tl.float32),
        tl.load(value + element + 1).to(tl.float32),
    )
    tl.store(base + BLOCK_RECORDS * HALF_D + record * HALF_D + byte, packed)
    tl.store(
        base
        + BLOCK_RECORDS * (HEAD_DIM + GROUPS)
        + record * GROUPS
        + tl.arange(0, GROUPS),
        scales,
    )


def store_mxfp4_qsa(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize ``[tokens, heads, head_dim]`` K/V into an mxfp4_qsa cache.

    ``kv_cache`` is the ``[blocks, tokens, heads, record_bytes]`` uint8 view.
    Negative slots (``PAD_SLOT_ID``) are skipped; ``slot_mapping``'s length
    sets the token count, as in ``reshape_and_cache_flash``.
    """
    num_tokens = slot_mapping.shape[0]
    num_blocks, page_size, num_heads, width = kv_cache.shape
    head_dim = head_dim_from_record_bytes(width)
    records = block_records(kv_cache)
    if key.shape[1:] != (num_heads, head_dim) or value.shape[1:] != key.shape[1:]:
        raise ValueError("mxfp4_qsa K/V rows do not match the cache heads")
    assert key.stride(-1) == value.stride(-1) == 1
    if num_tokens == 0:
        return
    _mxfp4_qsa_store_kernel[(num_tokens, num_heads)](
        key,
        value,
        slot_mapping,
        kv_cache,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        num_blocks * page_size,
        BLOCK_RECORDS=records,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        RECORD_BYTES=width,
        num_warps=1,
    )


__all__ = [
    "GROUP_SIZE",
    "block_records",
    "head_dim_from_record_bytes",
    "mxfp4_dequant_rows",
    "mxfp4_quantize_groups",
    "record_bytes",
    "row_bytes",
    "store_mxfp4_qsa",
]
