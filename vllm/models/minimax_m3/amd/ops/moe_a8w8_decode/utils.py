# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 weight unpack shared by the a16w8 decode gemm1 / gemm2 kernels."""

import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T


def _fp8x8_to_bf16(dw_lo, dw_hi, scale_f32):
    """8 fp8 (e4m3, two i32 dwords) -> ``vector<8xbf16>`` scaled by ``scale_f32``:
    four ``v_cvt_scalef32_pk_bf16_fp8`` (2 elements each, low/high word of a dword).
    The ODS op is called directly; fx only wraps the fp4 variant."""
    v2bf16 = T.vec(2, T.bf16)
    halves = []
    for dw in (dw_lo, dw_hi):
        for hi in range_constexpr(2):
            halves.append(
                _rocdl.cvt_scalef32_pk_bf16_fp8(
                    v2bf16,
                    fx.Int32(dw).ir_value(),
                    fx.Float32(scale_f32).ir_value(),
                    bool(hi),
                )
            )
    return fx.Vector.from_elements(
        [fx.Vector(h).bitcast(fx.Int32)[0] for h in halves], fx.Int32
    ).bitcast(fx.BFloat16)
