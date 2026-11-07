# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm NVFP4 K / V / index-K pages for the mono kernels
(``config.KV_FORMAT == "nvfp4"``): the insert's quantization and the decode's
dequantization into the e4m3 bytes the FP8 MFMAs take.

The pages are the ones ``sparse_attention_nvfp4`` / ``indexer_nvfp4`` bind and
``ops/norm_rope_nvfp4_insert`` writes (stock prefill reads and writes the same
caches): per 128-token page and 16-value group, e2m1 codes (two a byte, the
lower dim / token in the low nibble) and an e4m3 group scale; a value is
e2m1 x scale x the layer's fp32 global scale (K / V: ``_k_scale`` /
``_v_scale``; index-K: 1).

- K side of a main page (``NVFP4_SIDE`` bytes): token t's 128 codes at
  ``t * 64``, its 8 scales at ``NVFP4_CODES * 128 + k_scale_off(t, group)``.
- V side (``NVFP4_SIDE`` on): dim d of octet o (tokens 8 o .. 8 o + 7) is the
  dword at ``o * 512 + d * 4`` (nibble t % 8 = token t); the scale of (token t,
  group g) at ``NVFP4_CODES * 128 + g * 128 + t``.
- Index page: token t's codes at ``t * 64``, scales at ``NVFP4_CODES * 128 + t * 8
  + g``.

Quantization is the stock insert's, bit for bit. Dequantization: the exact fp16
products e2m1 x scale (|.| <= 6 x 448) for the fp16 MFMAs (QK, index scores;
``f16x8``), or the stock decode's values ``e4m3_rn_satfinite(e2m1 x scale)`` for
the FP8 PV MFMA (``e4m3x8``); the global scale applied by the caller.
"""

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.mono.kernels.common import xred
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    as_f32,
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
)

RCP6 = 0x3E2AAAAB  # RN(1 / 6)
E4M3_MAX = 448.0


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def e4m3_byte(x):
    """RNE e4m3 byte of ``x`` (in range)."""
    pk = fx.rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        fx.as_ir_value(Vec.filled(2, 0, fx.Int16)),
        fx.as_ir_value(x),
        fx.as_ir_value(x),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return Vec(pk).bitcast(fx.Int32)[0] & 0xFF


def e4m3_value(byte):
    pk = fx.rocdl.cvt_scalef32_pk_f32_fp8(
        T.vec(2, T.f32),
        fx.as_ir_value(fx.Int32(byte)),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return fx.Float32(Vec(pk)[0])


def _e2m1x2(a, b, old, sel):
    """E2M1 codes of ``a``, ``b`` into byte ``sel`` of ``old``, ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def quant4(xs, gs, inv_gs):
    """This lane's 4 values, a quarter of the 16-value group lanes l ^ 0..3 hold
    (in a 32-lane half: the lane's dims 4 (l % 32) .. + 4) -> (their e2m1 codes,
    16 bits, value 0 lowest; the group's e4m3 scale byte), under global scale
    ``gs`` (``inv_gs`` = 1 / gs): the stock insert's rule."""
    a = [fmath.absf(x) for x in xs]
    m = fx.max(fx.max(a[0], a[1]), fx.max(a[2], a[3]))
    m = xred(xred(m, 1, fx.max), 2, fx.max)
    sf = fx.min((m * as_f32(RCP6)) * inv_gs, fx.Float32(E4M3_MAX))
    byte = e4m3_byte(sf)
    sv = e4m3_value(byte)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


def k_scale_off(tok, grp):
    """Byte of (token, group) in the K scale region (swizzled per 32 tokens):
    groups 2 h and 2 h + 1 of a token are adjacent (an even offset, then + 1)."""
    r = tok % 32
    n = (r // 8) * 4 + r % 4
    return ((((tok // 32) * 4 + grp // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + grp % 2


def scale_pair(byte):
    """One e4m3 scale byte as an fp16 pair (s, s): the K / index-K operands, whose
    8 values share a group."""
    return cvt_fp8x2_f16(fx.Int32(byte) * 0x101, False)


def scale_pairs8(w0, w1):
    """Eight e4m3 scale bytes (dwords w0, w1, byte i = value i) as four fp16
    pairs (pair j: values 2 j, 2 j + 1): a V operand's 8 tokens."""
    return [cvt_fp8x2_f16(w, hi) for w in (w0, w1) for hi in (False, True)]


def f16x8(codes, pairs):
    """Eight e2m1 codes (one dword, nibble i = value i) times their scales
    (``pairs``: four fp16 pairs, pair j for values 2 j, 2 j + 1) -> an 8 x fp16 MFMA
    operand, value i in lane element i: exact, 8 VALU (no clamp, no e4m3)."""
    words = [
        (cvt_fp4x2_f16(codes, j) * pairs[j]).bitcast(fx.Int32)[0]
        for j in range_constexpr(4)
    ]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float16)


def f16_pair(a, b):
    """Two f32 -> one f32-typed word holding (fp16(a), fp16(b)), as ``bf16_pair``."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Float32)[0]
    )


def e4m3x8_words(codes, pairs):
    """Eight e2m1 codes (one dword, nibble i = value i) times their scales
    (``pairs``: four fp16 pairs, pair j for values 2 j, 2 j + 1) -> the e4m3 bytes
    in two dwords, byte i of the pair = value i: the fp16 products are exact,
    clamped to +-448 and rounded once (``e4m3_rn_satfinite``)."""
    big = Vec.filled(2, E4M3_MAX, fx.Float16)
    small = Vec.filled(2, -E4M3_MAX, fx.Float16)
    words = []
    for h in range_constexpr(2):
        pk = Vec.filled(2, 0, fx.Int16)
        for j in (2 * h, 2 * h + 1):
            v = cvt_fp4x2_f16(codes, j) * pairs[j]
            v = _arith.MinNumFOp(
                fx.as_ir_value(v), fx.as_ir_value(big), fastmath=_nnan()
            ).result
            v = _arith.MaxNumFOp(v, fx.as_ir_value(small), fastmath=_nnan()).result
            pk = Vec(
                fx.rocdl.cvt_scalef32_pk_fp8_f16(
                    Vec.make_type(2, fx.Int16),
                    fx.as_ir_value(pk),
                    v,
                    fx.as_ir_value(fx.Float32(1.0)),
                    j % 2 == 1,
                )
            )
        words.append(pk.bitcast(fx.Int32)[0])
    return words


def e4m3x8(codes, pairs):
    """``e4m3x8_words`` as one i64 (an 8-byte FP8 MFMA operand)."""
    return fx.Vector.from_elements(e4m3x8_words(codes, pairs), fx.Int32).bitcast(
        fx.Int64
    )[0]
