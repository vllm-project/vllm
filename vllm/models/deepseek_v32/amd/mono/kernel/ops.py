# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/ops.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   unused primitives removed.

"""Common AMD expression primitives for fused model-layer kernels."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

_UMAX_DPP_ASM = "\n".join(
    [
        "s_nop 1\nv_max_u32_dpp $0, $0, $0 " + control
        for control in (
            "row_shr:1 bound_ctrl:0",
            "row_shr:2 bound_ctrl:0",
            "row_shr:4 bound_ctrl:0",
            "row_shr:8 bound_ctrl:0",
            "row_bcast:15 row_mask:0xa",
            "row_bcast:31 row_mask:0xc",
        )
    ]
)


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def spin_pause() -> None:
    """Keep tagged-mailbox polling loads inside the retry loop."""

    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


def read_lane_i32(value, lane):
    """Read one i32 from ``lane`` while keeping IR conversion local."""

    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(value), fx.Int32(lane)))


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32, "llvm.amdgcn.writelane.i32", [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)], [], []
        )
    )


def bpermute_i32(byte_offset, value):
    """Read an i32 VGPR value from the lane selected by a byte offset."""

    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(byte_offset), fx.Int32(value)))


def mem_realtime():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _hardware_f32(name, value):
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], []))


def rsq(value):
    return _hardware_f32("llvm.amdgcn.rsq.f32", value)


def rcp(value):
    return _hardware_f32("llvm.amdgcn.rcp.f32", value)


def exp(value):
    return _hardware_f32("llvm.amdgcn.exp2.f32", fx.Float32(value) * 1.4426950408889634)


def xshfl(value, offset):
    """Return the value from ``lane ^ offset`` using VALU/DPP operations."""

    if offset >= 16:
        return value.shuffle_xor(offset, 64)
    is_float = isinstance(value, fx.Float32)
    source = value.bitcast(fx.Int32) if is_float else fx.Int32(value)
    if offset == 8:
        result = fx.Int32(update_dpp_i32(source, source, 0x118, 0xF, 0xC, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x108, 0xF, 0x3, False))
    elif offset == 4:
        result = fx.Int32(update_dpp_i32(source, source, 0x114, 0xF, 0xA, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x104, 0xF, 0x5, False))
    elif offset == 2:
        result = fx.Int32(update_dpp_i32(source, source, 0x4E, 0xF, 0xF, False))
    else:
        result = fx.Int32(update_dpp_i32(source, source, 0xB1, 0xF, 0xF, False))
    return result.bitcast(fx.Float32) if is_float else result


def wave_umax_dpp(value):
    """Return a wave maximum through the tuned GLM/TileRT DPP schedule."""

    result = llvm.InlineAsmOp(T.i32, [as_ir_value(fx.Int32(value))], _UMAX_DPP_ASM, "=v,0").result
    return read_lane_i32(result, 63)


def xred(value, offset, op):
    """Combine a value with ``lane ^ offset`` using a symmetric operation."""

    if offset < 16:
        return op(value, xshfl(value, offset))
    is_float = isinstance(value, fx.Float32)
    source = as_ir_value(value.bitcast(fx.Int32) if is_float else fx.Int32(value))
    swap = rocdl.permlane32_swap if offset == 32 else rocdl.permlane16_swap
    pair = swap(llvm.StructType.get_literal([T.i32, T.i32]), source, source, False, False)
    lhs, rhs = (fx.Int32(llvm.extractvalue(T.i32, pair, [index])) for index in range(2))
    if is_float:
        return op(lhs.bitcast(fx.Float32), rhs.bitcast(fx.Float32))
    return op(type(value)(lhs), type(value)(rhs))


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def div_rn(value, divisor, reciprocal):
    """IEEE-rounded quotient from one exact reciprocal using Markstein refinement."""

    quotient = value * reciprocal
    residual = fmath.fma(-quotient, divisor, value)
    return fx.Float32(fmath.fma(fx.Float32(residual), reciprocal, quotient))


def fp8_pack4(a, b, c, d):
    """Pack four in-range FP32 values as E4M3 bytes."""

    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def f8_word(k):
    """Map an FP8 activation index to the packed LDS word order."""

    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


def fp8_to_bf16x8(word0, word1):
    """Convert two dwords of eight FP8 values to a BF16 vector."""

    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), as_ir_value(word), one, bool(half)))
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), as_ir_value(word), as_ir_value(scale), select)
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)
