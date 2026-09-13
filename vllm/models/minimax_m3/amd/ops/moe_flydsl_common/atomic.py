# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""MiniMax-M3 FlyDSL MoE atomic helpers."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import (
    _gep1,
    _gep3,
    _global_base_ptr1,
    _lds_ptr3,
    _raw,
)

BM = 16


@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    n_block_idx,
    wave,
    lane,
    i32_M,
    N_OUT,
    BN,
    packed,
    weight,
    bm=BM,
):
    """accm[rt][ni] (f32[4] per lane, MFMA C layout, rt = 16-row tile of the
    ``bm``-row block) -> LDS [bm, BN] f32 -> per row: 2 columns per lane x weight
    -> packed bf16 atomic add at out[token, col]."""
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    row_base = lane_div_16 * fx.Int32(4)
    for rt in range_constexpr(len(accm)):
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[rt][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(rt * 16 + v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(bm // 8):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, off),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
