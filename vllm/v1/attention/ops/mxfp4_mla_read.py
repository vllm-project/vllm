# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton read-side unpack for the mxfp4_mla KV cache.

A row is packed data then scales, ``row_bytes(d) == d // 2 + d // 32``.
Unpacks a gathered tile to bf16 once and hands it to the existing ``tl.dot``
calls. On gfx950 ``tl.dot_scaled`` lowers to the same convert plus a bf16
MFMA, so it would only convert twice.
"""

from __future__ import annotations

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mxfp4_mla import GROUP_SIZE

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX950
else:
    _ON_GFX950 = False

# v_cvt_scalef32_pk_bf16_fp4 exists only on gfx950.
_HW_UNPACK = tl.constexpr(_ON_GFX950)
_GROUP = tl.constexpr(GROUP_SIZE)

# E2M1 magnitudes reconstructed arithmetically rather than from a lookup table,
# so the kernel needs no constant memory. Code index ``c = e << 1 | f``, where
# ``e`` is the 2-bit exponent and ``f`` the 1-bit mantissa. The exponent bias is
# 1, and **``e == 0`` is the subnormal range** -- which is the whole subtlety:
#
#   e=0: value = 0.5 * f          c=0 -> 0.0    c=1 -> 0.5
#   e>0: value = (1 + 0.5f) * 2**(e-1)
#        c=2 -> 1.0 * 2**0 = 1.0        c=5 -> 1.5 * 2**1 = 3.0
#        c=3 -> 1.5 * 2**0 = 1.5        c=6 -> 1.0 * 2**2 = 4.0
#        c=4 -> 1.0 * 2**1 = 2.0        c=7 -> 1.5 * 2**2 = 6.0
#
# Applying the normal-number formula to ``e == 0`` yields 0.75 for code 1, which
# is not an E2M1 value at all.
_E8M0_BIAS = tl.constexpr(127)


# fp32 is 1-8-23, so the exponent field starts at bit 23. Scaling by a power of
# two is exactly an integer add there, which is how the E8M0 group scale is
# applied without a transcendental.
_F32_MANT_BITS = tl.constexpr(23)


@triton.jit
def _e2m1_codes_to_f32(codes):
    """4-bit E2M1 codes (as integers) -> signed fp32 values.

    Avoids ``tl.exp2``: the exponent only takes values 0..3, so a two-select
    chain gives the power-of-two term.
    """
    magnitude_index = codes & 0x07
    exponent = (magnitude_index >> 1) & 0x03
    fraction = (magnitude_index & 0x01).to(tl.float32)
    # 2**(e-1) for e in {1,2,3} is {1,2,4}; e == 0 is the subnormal range.
    power = tl.where(exponent == 1, 1.0, tl.where(exponent == 2, 2.0, 4.0))
    value = tl.where(exponent == 0, 0.5 * fraction, (1.0 + 0.5 * fraction) * power)
    return tl.where((codes & 0x08) != 0, -value, value)


@triton.jit
def _apply_e8m0_scale(values, encoded_scales):
    """Multiply fp32 ``values`` by ``2**(X - 127)`` via an integer exponent add.

    Exact, because scaling by a power of two only shifts the exponent. Zero must
    be special-cased: adding to a zero's exponent field manufactures a denormal
    instead of leaving it zero.
    """
    delta = (encoded_scales.to(tl.int32) - 127) << _F32_MANT_BITS
    bits = values.to(tl.int32, bitcast=True) + delta
    scaled = bits.to(tl.float32, bitcast=True)
    return tl.where(values == 0.0, 0.0, scaled)


# --------------------------------------------------------------------------
# Hardware unpack: v_cvt_scalef32_pk_bf16_fp4 via inline assembly
#
# The software path below costs ~10 VALU ops per value, which measured 1.8x
# SLOWER than bf16 despite moving 3.76x fewer bytes -- bf16 reaches ~7.8 TB/s
# (essentially peak) while the software path plateaus at 8% of peak. The
# hardware instruction does nibble extract, magnitude and scale for two values
# in ONE op, which is the only measured route to the bandwidth win.
#
# Contract, established empirically:
#   * inputs must be int32 -- a uint8 tensor cannot bind to a 'v' constraint
#   * the output must be int32 too -- a bf16 output only ever receives the
#     instruction's low result, so the odd nibbles would be lost
#   * op_sel:[s0, s1, 0] selects source nibble pair 2*s0 + 4*s1; the third
#     field does not affect source selection
#   * the scale is an inline constant 1.0; the E8M0 group scale is applied
#     afterwards by _apply_e8m0_scale, which costs ~1 op per value
# --------------------------------------------------------------------------
# Must be tl.constexpr *instances*: the JIT refuses plain module globals, and
# annotating is not the same thing.
_ASM_P01 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, 1.0 op_sel:[0,0,0]")
_ASM_P23 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, 1.0 op_sel:[1,0,0]")
_ASM_P45 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, 1.0 op_sel:[0,1,0]")
_ASM_P67 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, 1.0 op_sel:[1,1,0]")


@triton.jit
def _cvt_pair(packed_i32, ASM: tl.constexpr):
    """One int32 word -> one int32 holding two packed bf16 (a nibble pair)."""
    return tl.inline_asm_elementwise(
        asm=ASM,
        constraints="=v,v",
        args=[packed_i32],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _split_pair(pair):
    """int32 of two packed bf16 -> (low fp32, high fp32).

    A bf16 occupies the *top* 16 bits of the matching fp32, so the low bf16 is
    recovered by shifting it up and the high one by masking in place.
    """
    lo = (pair << 16).to(tl.float32, bitcast=True)
    # -65536 is 0xFFFF0000 as a signed int32; the unsigned literal is out of
    # range for the type Triton infers here.
    hi = (pair & -65536).to(tl.float32, bitcast=True)
    return lo, hi


@triton.jit
def unpack_mxfp4_tile_hw(
    packed_i32,  # [BLOCK_K, LATENT // 8] int32 view of the row's data region
    encoded_scales,  # [BLOCK_K, LATENT // GROUP] uint8
    BLOCK_K: tl.constexpr,
    LATENT: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Packed MXFP4 -> bf16 [BLOCK_K, LATENT], using the hardware converter.

    Four instructions cover all eight nibbles of each int32 word, in natural
    order: value ``j`` of word ``w`` is latent dim ``8w + j``.
    """
    v0, v1 = _split_pair(_cvt_pair(packed_i32, _ASM_P01))
    v2, v3 = _split_pair(_cvt_pair(packed_i32, _ASM_P23))
    v4, v5 = _split_pair(_cvt_pair(packed_i32, _ASM_P45))
    v6, v7 = _split_pair(_cvt_pair(packed_i32, _ASM_P67))

    # Assemble so that out[8w + j] == vj[w]. Splitting the target index by the
    # bits of j gives a bit-reversed interleave tree: pairing stride-4 tensors
    # first, then stride-2, then stride-1.
    even = tl.interleave(tl.interleave(v0, v4), tl.interleave(v2, v6))
    odd = tl.interleave(tl.interleave(v1, v5), tl.interleave(v3, v7))
    values = tl.interleave(even, odd)  # [BLOCK_K, LATENT]

    scales = tl.broadcast_to(
        encoded_scales[:, :, None], (BLOCK_K, LATENT // GROUP, GROUP)
    ).reshape(BLOCK_K, LATENT)
    return _apply_e8m0_scale(values, scales).to(tl.bfloat16)


@triton.jit
def unpack_mxfp4_tile(
    packed,  # [BLOCK_K, LATENT // 2] uint8
    encoded_scales,  # [BLOCK_K, LATENT // GROUP] uint8
    BLOCK_K: tl.constexpr,
    LATENT: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Packed MXFP4 tile -> bf16 ``[BLOCK_K, LATENT]`` in latent order.

    The low nibble holds the lower element index, so the two nibble streams are
    interleaved back into value order rather than concatenated.
    """
    low = _e2m1_codes_to_f32((packed & 0x0F).to(tl.int32))
    high = _e2m1_codes_to_f32(((packed >> 4) & 0x0F).to(tl.int32))
    values = tl.interleave(low, high)  # [BLOCK_K, LATENT]

    # Broadcast the per-group scale byte across its 32 values, then apply it as
    # an exponent add rather than an exp2 and a multiply.
    scales = tl.broadcast_to(
        encoded_scales[:, :, None], (BLOCK_K, LATENT // GROUP, GROUP)
    ).reshape(BLOCK_K, LATENT)
    return _apply_e8m0_scale(values, scales).to(tl.bfloat16)


@triton.jit
def load_mxfp4_rows(
    cache_ptr, slots, valid, row_pitch, BLOCK_K: tl.constexpr, LATENT: tl.constexpr
):
    """Gather BLOCK_K packed rows as a bf16 [BLOCK_K, LATENT] tile.

    ``cache_ptr`` is the uint8 cache, ``row_pitch`` its row size in bytes.
    Invalid slots come back as zeros.
    """
    base = slots[:, None].to(tl.int64) * row_pitch
    encoded_scales = tl.load(
        cache_ptr + base + LATENT // 2 + tl.arange(0, LATENT // _GROUP)[None, :],
        mask=valid[:, None],
        other=127,
    )
    if _HW_UNPACK:
        words = tl.load(
            cache_ptr.to(tl.pointer_type(tl.int32))
            + base // 4
            + tl.arange(0, LATENT // 8)[None, :],
            mask=valid[:, None],
            other=0,
        )
        tile = unpack_mxfp4_tile_hw(words, encoded_scales, BLOCK_K, LATENT, _GROUP)
    else:
        packed = tl.load(
            cache_ptr + base + tl.arange(0, LATENT // 2)[None, :],
            mask=valid[:, None],
            other=0,
        )
        tile = unpack_mxfp4_tile(packed, encoded_scales, BLOCK_K, LATENT, _GROUP)
    return tl.where(valid[:, None], tile, 0.0)


@triton.jit
def _gather_mxfp4_kernel(
    cache_ptr,
    slots_ptr,
    out_ptr,
    num_slots,
    row_pitch,
    BLOCK_K: tl.constexpr,
    LATENT: tl.constexpr,
):
    """Standalone driver for :func:`load_mxfp4_rows`, used by the tests."""
    offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    slots = tl.load(slots_ptr + offsets).to(tl.int64)
    valid = (slots >= 0) & (slots < num_slots)
    values = load_mxfp4_rows(
        cache_ptr, tl.where(valid, slots, 0), valid, row_pitch, BLOCK_K, LATENT
    )
    tl.store(
        out_ptr + offsets[:, None] * LATENT + tl.arange(0, LATENT)[None, :],
        values,
    )


def gather_mxfp4_rows(cache, slots, latent: int, block_k: int = 16):
    """Host-side driver for the gather kernel. Test and reference harness."""
    import torch

    from vllm.v1.attention.ops.mxfp4_mla import row_bytes

    if cache.dtype != torch.uint8:
        raise ValueError(f"expected a uint8 cache, got {cache.dtype}")
    pitch = row_bytes(latent)
    flat = cache.reshape(-1)
    n = slots.numel()
    if n % block_k:
        raise ValueError(f"{n} slots is not a multiple of block_k={block_k}")
    out = torch.empty(n, latent, dtype=torch.bfloat16, device=cache.device)
    _gather_mxfp4_kernel[(n // block_k,)](
        flat,
        slots,
        out,
        flat.numel() // pitch,
        pitch,
        BLOCK_K=block_k,
        LATENT=latent,
        num_warps=4,
    )
    return out
