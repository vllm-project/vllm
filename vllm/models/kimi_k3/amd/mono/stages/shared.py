# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert stages (bf16, unquantized): h = situ(x Wg^T, x Wu^T), y = h Wd^T,
rows < M <= BM, as vLLM's KimiMLP computes it.

Both GEMMs use mfma_f32_16x16x32_bf16 straight from global memory: lane l feeds
row l % 16 and K elements (l / 16) * 8 .. + 8 of A and of B (B row = weight
row), and gets back rows (l / 16) * 4 + r, column l % 16. Buffer resources
sized to M rows return zeros for the padding rows.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.act import sigmoid_f32, tanh_f32
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _raw, global_typed_ptr
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import l1_invalidate, st_wt
from vllm.models.kimi_k3.amd.mono.common.plan import BM, N_WAVES, sh_pairs
from vllm.models.kimi_k3.amd.mono.common.sync import bump, count, wait_ge


def _bf16x8(rsrc, off_dw):
    v = buffer_ops.buffer_load(rsrc, _raw(off_dw), vec_width=4, dtype=T.i32)
    return fx.Vector(v).bitcast(fx.BFloat16)


def _mfma_bf16(a8, b8, acc):
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, [_raw(a8), _raw(b8), _raw(acc), 0, 0, 0]
    )


def _zero_f32x4():
    return fx.Vector.from_elements([_raw(fx.Float32(0.0))] * 4, fx.Float32)


@comm_ops.traced
def shared_gate_up(
    u,
    tid,
    lane,
    wave,
    lds_base,
    i32_M,
    arg_x,
    arg_w,
    arg_part,
    arg_h,
    a_pairs,
    a_pairs_done,
    *,
    SH_HIDDEN,
    SH_INTER,
    SH_KS,
    PAIR_STRIDE,
    SLOT,
    beta,
    linear_beta,
):
    """Ticket u: K split u % SH_KS of gate/up pair u / SH_KS, one K quarter per wave.

    Each wave writes its fp32 partial through to its own slice; the last split
    of a pair sums the slices, rounds to bf16 (vLLM's gate_up GEMM output),
    applies SiTU (soft tanh clip on up, no hard clamp) and stores h.

    Partials are [slice][row][pair][16 gate | 16 up]: one 128-byte line per
    slice, row and pair, so no line holds two pairs, and the last split only
    drops its L1: nothing in this launch read the pair's lines before. Every
    load of the ticket is issued before the first MFMA.
    """
    KC = SH_HIDDEN // SH_KS
    KW = KC // N_WAVES
    N_SLICES = SH_KS * N_WAVES
    NP = sh_pairs(SH_INTER)
    p = u // fx.Int32(SH_KS)
    ks = u - p * fx.Int32(SH_KS)
    row = lane % fx.Int32(16)
    kq = lane // fx.Int32(16)
    x_rsrc = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(arg_x), num_records_bytes=fx.Int64(i32_M) * fx.Int64(SH_HIDDEN * 2)
    )
    w_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_w))
    k_dw = ks * fx.Int32(KC // 2) + wave * fx.Int32(KW // 2) + kq * fx.Int32(4)
    x_dw = row * fx.Int32(SH_HIDDEN // 2) + k_dw
    g_dw = (p * fx.Int32(16) + row) * fx.Int32(SH_HIDDEN // 2) + k_dw
    u_dw = g_dw + fx.Int32(SH_INTER * SH_HIDDEN // 2)
    steps = KW // 32
    acc_g = _zero_f32x4()
    acc_u = _zero_f32x4()
    av = [_bf16x8(x_rsrc, x_dw + fx.Int32(s * 16)) for s in range_constexpr(steps)]
    gv = [_bf16x8(w_rsrc, g_dw + fx.Int32(s * 16)) for s in range_constexpr(steps)]
    uv = [_bf16x8(w_rsrc, u_dw + fx.Int32(s * 16)) for s in range_constexpr(steps)]
    for s in range_constexpr(steps):
        acc_g = _mfma_bf16(av[s], gv[s], acc_g)
        acc_u = _mfma_bf16(av[s], uv[s], acc_u)

    slice_row0 = (ks * fx.Int32(N_WAVES) + wave) * fx.Int32(BM)
    for r in range_constexpr(4):
        m = kq * fx.Int32(4) + fx.Int32(r)
        off = ((slice_row0 + m) * fx.Int32(NP) + p) * fx.Int32(32) + row
        vg = fx.Float32(fx.Vector(acc_g)[r])
        vu = fx.Float32(fx.Vector(acc_u)[r])
        if m < i32_M:
            st_wt(arg_part, off, vg, 4)
            st_wt(arg_part, off + fx.Int32(16), vu, 4)
    rocdl.s_waitcnt(vmcnt=0)
    n = count(
        wave,
        lane,
        lds_base,
        a_pairs + fx.Int64(p) * fx.Int64(4 * PAIR_STRIDE),
        SLOT,
        False,
    )
    if n == fx.Int32(SH_KS - 1):
        if wave == fx.Int32(0):
            l1_invalidate()
        gpu.barrier()
        m = tid // fx.Int32(16)
        c = p * fx.Int32(16) + tid % fx.Int32(16)
        if m < i32_M:
            part = global_typed_ptr(arg_part, T.f32)
            g = fx.Float32(0.0)
            up = fx.Float32(0.0)
            for sl in range_constexpr(N_SLICES):
                off = ((fx.Int32(sl * BM) + m) * fx.Int32(NP) + p) * fx.Int32(
                    32
                ) + tid % fx.Int32(16)
                g = g + fx.Float32(part[off])
                up = up + fx.Float32(part[off + fx.Int32(16)])
            g = g.to(fx.BFloat16).to(fx.Float32)
            up = up.to(fx.BFloat16).to(fx.Float32)
            gate = (
                fx.Float32(beta) * tanh_f32(g * fx.Float32(1.0 / beta)) * sigmoid_f32(g)
            )
            if const_expr(linear_beta > 0):
                up = fx.Float32(linear_beta) * tanh_f32(
                    up * fx.Float32(1.0 / linear_beta)
                )
            h_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(arg_h), num_records_bytes=BM * SH_INTER * 2
            )
            buffer_ops.buffer_store(
                _raw((gate * up).to(fx.BFloat16)),
                h_rsrc,
                _raw(m * fx.Int32(SH_INTER) + c),
            )
        rocdl.s_waitcnt(vmcnt=0)
        bump(wave, lane, a_pairs_done)


@comm_ops.traced
def shared_down(
    nb,
    lane,
    wave,
    i32_M,
    arg_h,
    arg_w,
    arg_out,
    a_pairs_done,
    *,
    SH_HIDDEN,
    SH_INTER,
    SH_DN_BN,
    SPIN,
):
    """Ticket nb: output columns nb * SH_DN_BN .. + SH_DN_BN, 16 per wave per
    pass, full K, once every gate/up pair is done."""
    wait_ge(wave, a_pairs_done, sh_pairs(SH_INTER), False, SPIN)
    row = lane % fx.Int32(16)
    kq = lane // fx.Int32(16)
    h_rsrc = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(arg_h), num_records_bytes=fx.Int64(i32_M) * fx.Int64(SH_INTER * 2)
    )
    w_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_w))
    # Masked buffer stores move the offset to 0x7FFFFFFF: the store resource
    # needs its real extent so the hardware drops them.
    o_rsrc = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(arg_out), num_records_bytes=fx.Int64(i32_M) * fx.Int64(SH_HIDDEN * 2)
    )
    h_dw = row * fx.Int32(SH_INTER // 2) + kq * fx.Int32(4)
    for sub in range_constexpr(SH_DN_BN // (16 * N_WAVES)):
        n0 = nb * fx.Int32(SH_DN_BN) + (fx.Int32(sub * N_WAVES) + wave) * fx.Int32(16)
        w_dw = (n0 + row) * fx.Int32(SH_INTER // 2) + kq * fx.Int32(4)
        acc = _zero_f32x4()
        for s in range_constexpr(SH_INTER // 32):
            acc = _mfma_bf16(
                _bf16x8(h_rsrc, h_dw + fx.Int32(s * 16)),
                _bf16x8(w_rsrc, w_dw + fx.Int32(s * 16)),
                acc,
            )
        for r in range_constexpr(4):
            m = kq * fx.Int32(4) + fx.Int32(r)
            buffer_ops.buffer_store(
                _raw(fx.Float32(fx.Vector(acc)[r]).to(fx.BFloat16)),
                o_rsrc,
                _raw(m * fx.Int32(SH_HIDDEN) + n0 + row),
                mask=_raw(m < i32_M),
            )
