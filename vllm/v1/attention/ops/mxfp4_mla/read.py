# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton read-side unpack for the mxfp4_mla KV cache.

A row is packed data then scales, ``row_bytes(d) == d // 2 + d // 32``.
Unpacks a gathered tile to bf16 once and hands it to the existing ``tl.dot``
calls. On gfx950 ``tl.dot_scaled`` lowers to the same convert plus a bf16
MFMA, so it would only convert twice.
"""

from __future__ import annotations

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mxfp4_mla import E8M0_BIAS, GROUP_SIZE

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX950
else:
    _ON_GFX950 = False

# v_cvt_scalef32_pk_bf16_fp4 exists only on gfx950.
_HW_UNPACK = tl.constexpr(_ON_GFX950)
_GROUP = tl.constexpr(GROUP_SIZE)

# fp32 is 1-8-23, so the exponent field starts at bit 23. Scaling by a power of
# two is exactly an integer add there, which is how the E8M0 group scale is
# applied without a transcendental.
_F32_MANT_BITS = tl.constexpr(23)


@triton.jit
def _apply_e8m0_scale(values, encoded_scales):
    """Multiply fp32 ``values`` by ``2**(X - 127)`` via an integer exponent add.

    Exact, because scaling by a power of two only shifts the exponent. Zero must
    be special-cased: adding to a zero's exponent field manufactures a denormal
    instead of leaving it zero.
    """
    delta = (encoded_scales.to(tl.int32) - E8M0_BIAS) << _F32_MANT_BITS
    bits = values.to(tl.int32, bitcast=True) + delta
    scaled = bits.to(tl.float32, bitcast=True)
    return tl.where(values == 0.0, 0.0, scaled)


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
    low = _e2m1_inline((packed & 0x0F).to(tl.int32))
    high = _e2m1_inline(((packed >> 4) & 0x0F).to(tl.int32))
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
    """Gather BLOCK_K packed rows as a bf16 [BLOCK_K, LATENT] tile, using the
    gfx950 hardware unpack where available."""
    return _load_mxfp4_rows(
        cache_ptr, slots, valid, row_pitch, BLOCK_K, LATENT, _HW_UNPACK
    )


@triton.jit
def _load_mxfp4_rows(
    cache_ptr,
    slots,
    valid,
    row_pitch,
    BLOCK_K: tl.constexpr,
    LATENT: tl.constexpr,
    HW_UNPACK: tl.constexpr,
):
    """Gather BLOCK_K packed rows as a bf16 [BLOCK_K, LATENT] tile.

    ``cache_ptr`` is the uint8 cache, ``row_pitch`` its row size in bytes.
    Invalid slots come back as zeros. Each row, with no padding::

        | packed E2M1 codes        | E8M0 scales        |
        |<----- LATENT // 2 ------>|<- LATENT // GROUP ->|
        |<------- row_pitch == row_bytes(LATENT) -------->|

    Code byte ``b`` holds values ``2b`` (low nibble) and ``2b + 1`` (high
    nibble); scale byte ``g`` covers values ``GROUP * g`` to
    ``GROUP * g + GROUP - 1``.
    """
    # The gfx950 path loads the codes as int32 words into one power-of-two tile.
    tl.static_assert(
        LATENT >= 128 and (LATENT & (LATENT - 1)) == 0,
        "mxfp4_mla needs a power-of-two latent of at least 128",
    )
    base = slots[:, None].to(tl.int64) * row_pitch
    encoded_scales = tl.load(
        cache_ptr + base + LATENT // 2 + tl.arange(0, LATENT // _GROUP)[None, :],
        mask=valid[:, None],
        other=E8M0_BIAS,
    )
    if HW_UNPACK:
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
