# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared constants and helpers for RDNA4 SplitKV kernels."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

HEAD_DIM = 256
WAVE_SIZE = 32
LOG2E = 1.4426950408889634


def _decode_fp8(raw, byte_index, *, is_fp8fnuz: bool):
    value = fx.Float32(fx.rocdl.cvt_f32_fp8(raw, byte_index))
    if const_expr(is_fp8fnuz):
        byte = (raw >> (byte_index * 8)) & 255
        # FNUZ uses these encodings for +/-240; FN reserves them for NaN.
        value = ((byte & 127) == 127).select(
            ((byte & 128) != 0).select(fx.Float32(-480.0), fx.Float32(480.0)),
            value,
        )
        value = (byte == 128).select(fx.Float32(float("nan")), value)
        value = value * 0.5
    return value


def _dequant_fp8x8(raw, scale, output_type, *, is_fp8fnuz: bool):
    """Decode eight packed FP8 values with scalar conversions."""
    words = raw.bitcast(fx.Int32)
    values = []
    for word_index in range_constexpr(2):
        for byte_index in range_constexpr(4):
            value = _decode_fp8(words[word_index], byte_index, is_fp8fnuz=is_fp8fnuz)
            values.append(value * scale)
    return fx.Vector.from_elements(values, dtype=fx.Float32).to(output_type)


def _flat_view(tensor: fx.Tensor) -> fx.Tensor:
    return fx.make_view(fx.get_iter(tensor), fx.make_layout(1 << 30, 1))


def _wave_reduce(value, mode: str):
    result = value
    for offset in (16, 8, 4, 2, 1):
        peer = fx.gpu.shuffle_xor(result, offset, WAVE_SIZE)
        result = fx.max(result, peer) if const_expr(mode == "max") else result + peer
    return result
