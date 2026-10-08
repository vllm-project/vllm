# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter (https://github.com/ROCm/aiter) v0.1.24.post1,
# Apache-2.0, Copyright (C) 2025-2026 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/mxfp4_gemm1.py (_gemm1_body)
# ruff: noqa: E501, SIM108
"""gemm1 tile of the mono MoE launch: AITER's a4w4 stage-1 body (MXFP4 x MXFP4,
inline quant of x, SiTUv2, the MXFP4 intermediate and its scales out).

The one change from AITER: ``wt_out`` stores the intermediate and its scales at
device scope, so they write through the XCD's L2 and the m-block counter needs
no L2 writeback before a gemm2 tile on another XCD reads them. The helpers that
size the tile (``_bm_constants``, ``default_k_stages``, ``default_epi_splits``)
are AITER's, unchanged.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import dpp_utils
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import ACC_LDS_PAD_DW, _bm_constants
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    _activation_mul_batch,
    _e8m0_from_amax,
    _fabs_f32,
    _global_f32_at,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _global_i32_load,
    _global_scalar_tiles,
    _inline_dpp_quad_amax,
    _inline_e8m0,
    _layout_idx,
    _lds_swizzle_mask,
    _pkmax_u16,
    _scalar_store,
    _scale_mma_atoms,
    _umax_i32,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kStages,
)
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value


def _out_store(wt, tiles, base_i64, idx, value, numeric_cls):
    """Epilogue store; ``wt`` makes it device scope so it writes through the XCD's L2."""
    if const_expr(wt):
        nbytes = numeric_cls.width // 8
        _llvm.StoreOp(
            numeric_cls(value).ir_value(),
            comm_ops._ptr_plus(base_i64, idx, nbytes),
            alignment=nbytes,
            ordering=_llvm.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        )
    else:
        _scalar_store(tiles, idx, value, numeric_cls)


@flyc.jit
def _gemm1_body(
    lds_raw_ptr,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    arg_bias,
    bx_i32,
    lane,
    wave,
    use_nt,
    i32_ntok,
    i32_total_m_blocks,
    *,
    BM,
    BN,
    BK,
    inline_quant=False,
    prefetch_hidden=False,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
    K,
    N_OUT,
    NE,
    interleave=False,
    native_scale_layout=False,
    num_waves=4,
    k_wave=1,
    epi_splits=1,
    k_stages=kStages,
    wt_out=False,
):
    # A-code tile bytes/row: fp4 packs 2 codes/byte (BK/2); fp8 is 1 B/elem (BK).
    KH_TILE = BK if a_dtype == "fp8" else BK // 2
    K_HALF = k_half_for(K)
    # A row bytes: fp4 = K/2, fp8 = K (B is always mxfp4 -> keeps K_HALF).
    A_ROW_BYTES = K if a_dtype == "fp8" else K_HALF
    K_TILES_TOTAL = K // BK
    K_TILES_PER_WAVE = K_TILES_TOTAL // k_wave
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    ASCALE_CHUNK_BYTES = kAS_per_chunk_dw * 4
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = N_OUT // BN
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    OUT_ROW_BYTES = inter if out_dtype == "fp8" else k_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(
        BM, BN, KH_TILE, K_TILES_TOTAL, k_wave, epi_splits, k_stages
    )
    kUnroll = K_TILES_PER_WAVE - k_stages

    k_split = (
        a_dtype == "fp4"
        and not inline_quant
        and BM == 128
        and BN == 256
        and num_waves == 4
        and kMChunks > 1
        and kUnroll >= 1
    )
    a_reg_depth = min(2, kUnroll) if k_split else 0

    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16
    N_REPS = BN // (num_waves * 16)
    EPILOGUE_M_REPS = 1 if num_waves == 2 else M_REPS
    B_SCALE_REPS = max(1, N_REPS // 2) if interleave else 2
    wave_n = wave % fx.Int32(num_waves)
    wave_k = wave // fx.Int32(num_waves)
    aq_group_base = wave_k * fx.Int32(kAStages * BM * KH_TILE)

    mem_1x1 = fx.make_layout(1, 1)
    mem_4x1 = fx.make_layout(4, 1)
    mem_8x1 = fx.make_layout(8, 1)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(A_ROW_BYTES))
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = fx.Int64(i32_total_m_blocks) * fx.Int64(_asc_per_mb)

    bq_tiles = _global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, mem_4x1)
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    aq_reg_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, mem_4x1)
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, mem_1x1)
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)

    if const_expr(inline_quant):
        hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
        hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
        hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in the
    # epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    if const_expr(inline_quant):
        rcls = wave_n * fx.Int32(4) + lane_div_16
        cached_row_inline = _global_i32_at(arg_mind, m_row + rcls)
    elif const_expr(num_waves == 2):
        for load_group in range_constexpr(2):
            idx = (
                m_row
                + wave_n * fx.Int32(BM // 2)
                + fx.Int32(load_group * 8)
                + lane_div_8
            )
            cached_actual_row.append(_global_i32_at(arg_mind, idx))
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            actual_row = _global_i32_at(arg_mind, idx)
            if const_expr(a_dtype == "fp8" and BM >= 64):
                cached_actual_row.append(
                    [
                        fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32,
                                (lane_div_16 + fx.Int32(row_group * 4)) * fx.Int32(32),
                                as_ir_value(actual_row),
                            )
                        )
                        for row_group in range_constexpr(2)
                    ]
                )
            else:
                cached_actual_row.append(actual_row)

    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(N_REPS):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN)
                + wave_n * fx.Int32(N_REPS * 16)
                + fx.Int32(j * 16)
            )
        else:
            tile_il = (
                n_block_idx * fx.Int32(BN // 16)
                + wave_n * fx.Int32(N_REPS)
                + fx.Int32(j)
            )
            g = tile_il & fx.Int32(1)
            n0 = tile_il >> fx.Int32(1)
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    if const_expr(interleave):
        # Each scale word covers two consecutive 16-column weight tiles.
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave_n * fx.Int32(
            N_REPS
        ) // fx.Int32(2)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        if const_expr(BN == 64):
            np_gate = n_block_idx
        elif const_expr(BN == 128):
            np_gate = n_block_idx * fx.Int32(2) + wave_n // fx.Int32(2)
        else:
            np_gate = n_block_idx * fx.Int32(BN // 64) + wave_n
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(B_SCALE_REPS):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    scale_atoms = _scale_mma_atoms(a_dtype)
    accm = [
        [fx.make_rmem_tensor(mem_4x1, fx.Float32) for _ in range(N_REPS)]
        for _ in range(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(N_REPS):
            accm[i][J].store(fx.Vector.filled(4, 0.0, fx.Float32))
    b = [[[None, None] for _ in range(N_REPS)] for _ in range(k_stages)]
    b_scale_v = [[None, None] for _ in range(k_stages)]

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(k_wave * kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, mem_4x1)
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, mem_1x1)
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def issue_a_load_lds(slot, kt):
        # A wave-wide BufferCopyLDS128b writes 1024 contiguous bytes. That is
        # eight FP4 rows or four FP8 rows. Larger FP8 tiles use two four-row
        # passes; BM32 keeps register staging, which is faster at low occupancy.
        if const_expr(num_waves == 2):
            for load_group in range_constexpr(2):
                lds_row = wave_n * fx.Int32(BM // 2) + fx.Int32(load_group * 8)
                mask = _lds_swizzle_mask(lds_row + lane_div_8)
                voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                    load_group
                ] * fx.Int32(A_ROW_BYTES)
                off = (
                    aq_group_base
                    + fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                )
                fx.copy(
                    aq_dma_atom,
                    fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                    fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                    soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                )
            return
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            if const_expr(a_dtype == "fp8" and BM >= 64):
                for row_group in range_constexpr(2):
                    dst_row_base = lds_row + fx.Int32(row_group * 4)
                    mask = _lds_swizzle_mask(dst_row_base + lane_div_16)
                    voffset = ((lane_mod_16 * fx.Int32(16)) ^ mask) + cached_actual_row[
                        sub
                    ][row_group] * fx.Int32(A_ROW_BYTES)
                    off = (
                        aq_group_base
                        + fx.Int32(slot * (BM * KH_TILE))
                        + dst_row_base * fx.Int32(KH_TILE)
                    )
                    fx.copy(
                        aq_dma_atom,
                        fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                        fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                        soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                    )
            elif const_expr(a_dtype == "fp8"):
                actual_row = cached_actual_row[sub]
                dst_row = lds_row + lane_div_8
                mask = _lds_swizzle_mask(dst_row)
                for half in range_constexpr(2):
                    half_off = fx.Int32(half * 128)
                    src_off = (
                        actual_row * fx.Int32(A_ROW_BYTES)
                        + half_off
                        + lane_mod_8 * fx.Int32(16)
                    )
                    dst_off = (
                        aq_group_base
                        + fx.Int32(slot * (BM * KH_TILE))
                        + dst_row * fx.Int32(KH_TILE)
                        + half_off
                        + ((lane_mod_8 * fx.Int32(16)) ^ mask)
                    )
                    r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
                    fx.copy(
                        aq_reg_atom,
                        fx.slice(aq_dma_tiles4, (None, src_off // fx.Int32(16))),
                        r,
                        soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                    )
                    fx.copy(
                        i32x4_copy_atom,
                        r,
                        fx.slice(s_aq_i32x4_tiles, (None, dst_off // fx.Int32(16))),
                    )
            else:
                mask = _lds_swizzle_mask(lds_row + lane_div_8)
                voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                    sub
                ] * fx.Int32(A_ROW_BYTES)
                off = (
                    aq_group_base
                    + fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                )
                fx.copy(
                    aq_dma_atom,
                    fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                    fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                    soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                )

    def issue_a_load_reg(kt):
        regs = []
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(A_ROW_BYTES)
            r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
            fx.copy(
                aq_reg_atom,
                fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                r,
                soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
            )
            regs.append(r)
        return regs

    def a_reg_write_lds(slot, regs):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave_n * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            dst = (
                aq_group_base
                + fx.Int32(slot * (BM * KH_TILE))
                + (lds_row + lane_div_8) * fx.Int32(KH_TILE)
                + lane_mod_8 * fx.Int32(16)
            )
            fx.copy(
                i32x4_copy_atom,
                regs[sub],
                fx.slice(s_aq_i32x4_tiles, (None, dst // fx.Int32(16))),
            )

    def _lds_i32x4_frag(tile_idx):
        # ds_read_b128 straight into an i32[4] register fragment (kept as a tensor
        # so it can feed fx.gemm directly).
        r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
        fx.copy(i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r)
        return r

    def issue_a_ds_read(slot, khalf=None):
        mask = _lds_swizzle_mask(lane_mod_16)
        a = [[None, None] for _ in range(kMChunks)]
        halves = range_constexpr(2) if khalf is None else (khalf,)
        if const_expr(kMChunks == 1):
            order = [(0, k) for k in halves]
        elif const_expr(not k_split):
            order = [(i, k) for k in halves for i in range_constexpr(kMChunks)]
        else:
            order = [
                (sub * 2 + i, k)
                for sub in range_constexpr(kSubBlocks)
                for k in halves
                for i in range_constexpr(2)
            ]
        for i, k in order:
            if const_expr(a_dtype == "fp8"):
                # fp8 128-K operand (v8i32) = two 16 B halves 64 B apart in the row
                # (f8f6f4 ABI), packed lo++hi. k selects the 128-K half (upper = +128 B).
                kbase = fx.Int32(k * 128)
                lo_col = (lane_div_16 * fx.Int32(16) + kbase) ^ mask
                hi_col = (lane_div_16 * fx.Int32(16) + kbase + fx.Int32(64)) ^ mask
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                boff = (
                    aq_group_base
                    + fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                )
                lo = fx.Vector(
                    fx.memref_load_vec(_lds_i32x4_frag((boff + lo_col) // fx.Int32(16)))
                )
                hi = fx.Vector(
                    fx.memref_load_vec(_lds_i32x4_frag((boff + hi_col) // fx.Int32(16)))
                )
                t = fx.make_rmem_tensor(mem_8x1, fx.Int32)
                t.store(lo.shuffle(hi, list(range(8))))
                a[i][k] = t
            else:
                lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    aq_group_base
                    + fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][k] = _lds_i32x4_frag(off // fx.Int32(16))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave_n * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave_n * fx.Int32(64) + lane) * fx.Int32(4)
        thread_linear = wave_n * fx.Int32(64) + lane
        if const_expr(num_waves == 2):
            bytes_per_16b_pass = num_waves * 64 * 16
            bytes_per_4b_pass = num_waves * 64 * 4
            for sub in range_constexpr(kSubBlocks):
                s_chunk = rocdl.readfirstlane(
                    T.i32,
                    (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4),
                )
                lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
                for block4k in range_constexpr(ASCALE_CHUNK_BYTES // 4096):
                    byte_off = block4k * 4096
                    for load_pass in range_constexpr(4096 // bytes_per_16b_pass):
                        pass_off = load_pass * bytes_per_16b_pass
                        s_off = rocdl.readfirstlane(
                            T.i32, s_chunk + fx.Int32(byte_off + pass_off)
                        )
                        fx.copy(
                            ascale_dma_atom16,
                            fx.slice(
                                ascale_dma_tiles4,
                                (None, v16 // fx.Int32(16)),
                            ),
                            fx.slice(
                                asc_i32x4_tiles,
                                (
                                    None,
                                    (
                                        lds_sub
                                        + fx.Int32(byte_off + pass_off)
                                        + wave_n * fx.Int32(1024)
                                    )
                                    // fx.Int32(16),
                                ),
                            ),
                            soffset=s_off // fx.Int32(4),
                        )
                full4k_bytes = (ASCALE_CHUNK_BYTES // 4096) * 4096
                remainder_bytes = ASCALE_CHUNK_BYTES - full4k_bytes
                for block1k in range_constexpr(remainder_bytes // 1024):
                    byte_off = full4k_bytes + block1k * 1024
                    for load_pass in range_constexpr(1024 // bytes_per_4b_pass):
                        pass_off = load_pass * bytes_per_4b_pass
                        s_off = rocdl.readfirstlane(
                            T.i32, s_chunk + fx.Int32(byte_off + pass_off)
                        )
                        fx.copy(
                            ascale_dma_atom4,
                            fx.slice(
                                ascale_dma_tiles1,
                                (None, v4 // fx.Int32(4)),
                            ),
                            fx.slice(
                                asc_i32_tiles,
                                (
                                    None,
                                    (
                                        lds_sub
                                        + fx.Int32(byte_off + pass_off)
                                        + wave_n * fx.Int32(256)
                                    )
                                    // fx.Int32(4),
                                ),
                            ),
                            soffset=s_off // fx.Int32(4),
                        )
                tail_bytes = remainder_bytes % 1024
                for load_pass in range_constexpr(
                    (tail_bytes + bytes_per_4b_pass - 1) // bytes_per_4b_pass
                ):
                    pass_off = load_pass * bytes_per_4b_pass
                    remaining = tail_bytes - pass_off
                    if thread_linear < fx.Int32((remaining + 3) // 4):
                        byte_off = (
                            full4k_bytes + (remainder_bytes // 1024) * 1024 + pass_off
                        )
                        s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                        fx.copy(
                            ascale_dma_atom4,
                            fx.slice(
                                ascale_dma_tiles1,
                                (None, v4 // fx.Int32(4)),
                            ),
                            fx.slice(
                                asc_i32_tiles,
                                (
                                    None,
                                    (
                                        lds_sub
                                        + fx.Int32(byte_off)
                                        + wave_n * fx.Int32(256)
                                    )
                                    // fx.Int32(4),
                                ),
                            ),
                            soffset=s_off // fx.Int32(4),
                        )
            return
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            for block4k in range_constexpr(ASCALE_CHUNK_BYTES // 4096):
                byte_off = block4k * 4096
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom16,
                    fx.slice(ascale_dma_tiles4, (None, v16 // fx.Int32(16))),
                    fx.slice(
                        asc_i32x4_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(1024))
                            // fx.Int32(16),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )
            full4k_bytes = (ASCALE_CHUNK_BYTES // 4096) * 4096
            remainder_bytes = ASCALE_CHUNK_BYTES - full4k_bytes
            for block1k in range_constexpr(remainder_bytes // 1024):
                byte_off = full4k_bytes + block1k * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom4,
                    fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                    fx.slice(
                        asc_i32_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(256))
                            // fx.Int32(4),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )
            tail_bytes = remainder_bytes % 1024
            if const_expr(tail_bytes > 0):
                byte_off = full4k_bytes + (remainder_bytes // 1024) * 1024
                if thread_linear < fx.Int32(tail_bytes // 4):
                    s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                    fx.copy(
                        ascale_dma_atom4,
                        fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                        fx.slice(
                            asc_i32_tiles,
                            (
                                None,
                                (lds_sub + fx.Int32(byte_off) + wave_n * fx.Int32(256))
                                // fx.Int32(4),
                            ),
                        ),
                        soffset=s_off // fx.Int32(4),
                    )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(
            fx.Int32,
            fx.add_offset(lds_raw_ptr, k_wave * kAStages * BM * KH_TILE),
        ),
        fx.make_layout(kSubBlocks * K_TILES_TOTAL * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, mem_1x1)
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, mem_4x1)

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(as_ir_value(_global_i32_load(asc_i32_tiles, lds_dw)))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave_n * fx.Int32(4) + lane_div_16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * (BK * 2) + B128_IDX * 256))
        r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
        fx.copy(
            hidden_copy_atom,
            fx.slice(hidden_tiles, (None, v_voff // fx.Int32(16))),
            r,
            soffset=s_soff // fx.Int32(4),
        )
        return r.load()

    def _bf16x2(dw):
        return as_ir_value(fx.Vector.from_elements([dw], fx.Int32).bitcast(fx.BFloat16))

    def _iq_block_amax(h_dw_i):
        # Per-32 amax over the 8 bf16 in this lane's block (4 bf16x2 dwords).
        hm = [h_dw_i[j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
        m01 = _pkmax_u16(hm[0], hm[1])
        m23 = _pkmax_u16(hm[2], hm[3])
        m0123 = _pkmax_u16(m01, m23)
        lo = m0123 & fx.Int32(0xFFFF)
        hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
        return _umax_i32(lo, hi)

    def _iq_pack_store(h_dw_i, qs_raw, B128_IDX, SUB, slot):
        # Quantize this lane's 8 elements (4 bf16x2 dwords) and store into the A-LDS
        # slot at the position issue_a_ds_read expects. fp4: 8 fp4 -> one i32 in a
        # 16 B swizzled block. fp8: 8 fp8 -> two i32; the 16 B block is
        # B128_IDX*8 + lsa3*2 + lib//2 and the byte within it is (lib%2)*8 (matches
        # the split-64 fp8 read).
        r = fx.Int32(SUB * 16) + r_in_chunk
        mask_r = _lds_swizzle_mask(r)
        row_off = fx.Int32(slot * (BM * KH_TILE)) + r * fx.Int32(KH_TILE)
        if const_expr(a_dtype == "fp8"):
            blk_byte = (
                fx.Int32(B128_IDX * 128)
                + lane_shr2_and3 * fx.Int32(32)
                + (lib >> fx.Int32(1)) * fx.Int32(16)
            )
            off = row_off + (blk_byte ^ mask_r) + (lib & fx.Int32(1)) * fx.Int32(8)
            # cvt.pk.fp8 packs 2 fp8 into a 16b lane of a vector<2xi16> accumulator
            # (dstLoHiSel = lo/hi); two lanes -> 4 fp8 = one i32 stored to LDS.
            i16x2 = fx.Vector.make_type(2, fx.Int16)
            zero16 = fx.Vector.filled(2, 0, fx.Int16)
            pk0 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, zero16, _bf16x2(h_dw_i[0]), qs_raw, 0
            )
            pk0 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, pk0, _bf16x2(h_dw_i[1]), qs_raw, 1
            )
            pk1 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, zero16, _bf16x2(h_dw_i[2]), qs_raw, 0
            )
            pk1 = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, pk1, _bf16x2(h_dw_i[3]), qs_raw, 1
            )
            _scalar_store(
                s_aq_i32x1_tiles,
                off // fx.Int32(4),
                fx.Vector(pk0).bitcast(fx.Int32)[0],
                fx.Int32,
            )
            _scalar_store(
                s_aq_i32x1_tiles,
                (off + fx.Int32(4)) // fx.Int32(4),
                fx.Vector(pk1).bitcast(fx.Int32)[0],
                fx.Int32,
            )
        else:
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            off = row_off + ((kb_in_kt * fx.Int32(16)) ^ mask_r) + lib * fx.Int32(4)
            pk = as_ir_value(fx.Int32(0))
            for j in range_constexpr(4):
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(
                    T.i32, pk, _bf16x2(h_dw_i[j]), qs_raw, j
                )
            _scalar_store(s_aq_i32x1_tiles, off // fx.Int32(4), fx.Int32(pk), fx.Int32)

    def _inline_quant_core_batch(specs, slot, scale_accum):
        n = len(specs)
        h_dw = [[h_v[j] for j in range_constexpr(4)] for (_b, _s, h_v) in specs]
        a = [_iq_block_amax(h_dw[i]) for i in range_constexpr(n)]
        s1 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(
                    as_ir_value(a[i]),
                    as_ir_value(a[i]),
                    0xB1,
                    0xF,
                    0xF,
                    True,
                )
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
        s2 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(
                    as_ir_value(a[i]),
                    as_ir_value(a[i]),
                    0x4E,
                    0xF,
                    0xF,
                    True,
                )
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
        e8 = [
            _inline_e8m0(a[i], max_norm=448.0 if a_dtype == "fp8" else 6.0)
            for i in range_constexpr(n)
        ]
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = as_ir_value((e8[i] << fx.Int32(23)).bitcast(fx.Float32))
            _iq_pack_store(h_dw[i], qs_raw, B128_IDX, SUB, slot)
            pack_byte = B128_IDX * 2 + SUB
            scale_accum = scale_accum | (e8[i] << fx.Int32(pack_byte * 8))
        return scale_accum

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        return _inline_quant_core_batch([(B128_IDX, SUB, h_v)], slot, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=b_load_s_base[j] // fx.Int32(4),
            )
            b_slot[j][half] = r

    def issue_b_load_half(b1, K_C, j, khalf):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048 + khalf * 1024)
        )
        r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
        fx.copy(
            bq_copy_atom,
            fx.slice(bq_tiles, (None, v // fx.Int32(16))),
            r,
            soffset=b_load_s_base[j] // fx.Int32(4),
        )
        b1[j] = r

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        # Interleaved BN64/BN128 have only one scale slot.
        for mw in range_constexpr(B_SCALE_REPS):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(mem_1x1, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            bs_slot[mw] = r.load()[0]

    def _mma(ci, opsel_a, opsel_b, a_frag, b_frag, sa, sb):
        # opsel_a/opsel_b select the e8m0 scale byte in the shared 256-K word and are
        # baked into the atom.
        fx.gemm(
            scale_atoms[(opsel_a, opsel_b)],
            ci,
            a_frag,
            b_frag,
            ci,
            scale_a=sa,
            scale_b=sb,
        )

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, khalf=None):
        if const_expr(interleave):
            mni = J // 2
        else:
            mni = J % 2
        if const_expr(khalf is None):
            bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        else:
            bJ0 = bJ1 = b_slot[J]

        def issue_cluster(scale_slot, in_b):
            sb = bs_slot[scale_slot]
            if const_expr(kMChunks == 1):
                sa = a_scale[0]
                if const_expr(khalf is None or khalf == 0):
                    _mma(accm[0][J], 0, 0 + in_b, a[0][0], bJ0, sa, sb)
                if const_expr(khalf is None or khalf == 1):
                    _mma(accm[0][J], 2, 2 + in_b, a[0][1], bJ1, sa, sb)
            else:
                for sub in range_constexpr(kSubBlocks):
                    i0 = sub * 2 + 0
                    i1 = sub * 2 + 1
                    sa = a_scale[sub]
                    if const_expr(khalf is None or khalf == 0):
                        _mma(accm[i0][J], 0, 0 + in_b, a[i0][0], bJ0, sa, sb)
                        _mma(accm[i1][J], 1, 0 + in_b, a[i1][0], bJ0, sa, sb)
                    if const_expr(khalf is None or khalf == 1):
                        _mma(accm[i0][J], 2, 2 + in_b, a[i0][1], bJ1, sa, sb)
                        _mma(accm[i1][J], 3, 2 + in_b, a[i1][1], bJ1, sa, sb)

        if const_expr(interleave and N_REPS == 1):
            # Adjacent waves consume the low/high N half of one scale word.
            if (wave_n & fx.Int32(1)) == fx.Int32(0):
                issue_cluster(0, 0)
            else:
                issue_cluster(0, 1)
        elif const_expr(BN == 64 and not interleave):
            if const_expr(num_waves == 2):
                if wave_n == fx.Int32(0):
                    issue_cluster(mni, 0)
                else:
                    issue_cluster(mni, 1)
            else:
                if wave_n == fx.Int32(0):
                    issue_cluster(0, 0)
                elif wave_n == fx.Int32(1):
                    issue_cluster(1, 0)
                elif wave_n == fx.Int32(2):
                    issue_cluster(0, 1)
                else:
                    issue_cluster(1, 1)
        elif const_expr(BN == 128 and not interleave):
            if (wave_n & fx.Int32(1)) == fx.Int32(0):
                issue_cluster(mni, 0)
            else:
                issue_cluster(mni, 1)
        else:
            issue_cluster(mni, J % 2 if const_expr(interleave) else J // 2)

    def run_k_slice_ksplit(global_k_tile):
        bh = [[None] * N_REPS for _ in range(2)]
        a_regs = [None] * a_reg_depth
        issue_a_scale_load()
        for S in range_constexpr(k_stages):
            issue_a_load_lds(S, global_k_tile(S))
        rocdl.sched_barrier(0)
        for S in range_constexpr(k_stages):
            issue_b_scale_load(b_scale_v[S], global_k_tile(S))
        for h in range_constexpr(2):
            for j in range_constexpr(N_REPS):
                issue_b_load_half(bh[h], global_k_tile(0), j, h)
        for d in range_constexpr(a_reg_depth):
            if const_expr(k_stages + d < K_TILES_PER_WAVE):
                a_regs[d] = issue_a_load_reg(global_k_tile(k_stages + d))

        for KT in range_constexpr(K_TILES_PER_WAVE):
            read_slot = KT % kAStages
            slot_b = KT % k_stages
            wt = KT + k_stages
            gpu.barrier()
            asc_cur = issue_a_scale_ds_read(global_k_tile(KT))
            if const_expr(wt < K_TILES_PER_WAVE):
                rocdl.sched_barrier(0)
                a_reg_write_lds(wt % kAStages, a_regs[KT % a_reg_depth])
                nxt = wt + a_reg_depth
                if const_expr(nxt < K_TILES_PER_WAVE):
                    a_regs[KT % a_reg_depth] = issue_a_load_reg(global_k_tile(nxt))
                rocdl.sched_barrier(0)
            for h in range_constexpr(2):
                a_h = issue_a_ds_read(read_slot, khalf=h)
                for J in range_constexpr(N_REPS):
                    mfma_cluster(bh[h], a_h, asc_cur, b_scale_v[slot_b], J, khalf=h)
                    rocdl.sched_barrier(0)
                    if const_expr(KT + 1 < K_TILES_PER_WAVE):
                        issue_b_load_half(bh[h], global_k_tile(KT + 1), J, h)
                    rocdl.sched_barrier(0)
            if const_expr(wt < K_TILES_PER_WAVE):
                issue_b_scale_load(b_scale_v[slot_b], global_k_tile(wt))

    def run_k_slice(k_tile_base_const):
        def global_k_tile(local_tile):
            return k_tile_base_const + local_tile

        if const_expr(k_split):
            run_k_slice_ksplit(global_k_tile)
            return

        _relax_prologue = (BM == 128) and not inline_quant
        if const_expr(not inline_quant):
            issue_a_scale_load()
        hidden_prefetch = None
        if const_expr(inline_quant and prefetch_hidden):
            hidden_prefetch = (
                inline_quant_load_kt(0, global_k_tile(0), cached_row_inline),
                inline_quant_load_kt(1, global_k_tile(0), cached_row_inline),
            )
        for K_C_LOCAL in range_constexpr(k_stages):
            K_C = global_k_tile(K_C_LOCAL)
            if const_expr(inline_quant):
                scale_accum = fx.Int32(0)
                if const_expr(prefetch_hidden):
                    h_v0, h_v1 = hidden_prefetch
                    if const_expr(K_C_LOCAL + 1 < K_TILES_PER_WAVE):
                        next_k = global_k_tile(K_C_LOCAL + 1)
                        hidden_prefetch = (
                            inline_quant_load_kt(0, next_k, cached_row_inline),
                            inline_quant_load_kt(1, next_k, cached_row_inline),
                        )
                    scale_accum = _inline_quant_core_batch(
                        [(0, 0, h_v0)], K_C_LOCAL, scale_accum
                    )
                else:
                    scale_accum = inline_quant_kt(
                        0, 0, K_C_LOCAL, K_C, cached_row_inline, scale_accum
                    )

                issue_b_load_j(b[K_C_LOCAL], K_C, 0)
                issue_b_load_j(b[K_C_LOCAL], K_C, 1)
                if const_expr(prefetch_hidden):
                    scale_accum = _inline_quant_core_batch(
                        [(1, 0, h_v1)], K_C_LOCAL, scale_accum
                    )
                else:
                    scale_accum = inline_quant_kt(
                        1, 0, K_C_LOCAL, K_C, cached_row_inline, scale_accum
                    )
                if const_expr(N_REPS > 2):
                    issue_b_load_j(b[K_C_LOCAL], K_C, 2)
                    issue_b_load_j(b[K_C_LOCAL], K_C, 3)
                inline_quant_pack_write(K_C_LOCAL, scale_accum)
            else:
                issue_a_load_lds(K_C_LOCAL, K_C)
                if const_expr(not _relax_prologue):
                    for j in range_constexpr(N_REPS):
                        issue_b_load_j(b[K_C_LOCAL], K_C, j)
            if const_expr(not _relax_prologue):
                issue_b_scale_load(b_scale_v[K_C_LOCAL], K_C)
        if const_expr(_relax_prologue):
            rocdl.sched_barrier(0)
            for K_C_LOCAL in range_constexpr(k_stages):
                K_C = global_k_tile(K_C_LOCAL)
                for j in range_constexpr(N_REPS):
                    issue_b_load_j(b[K_C_LOCAL], K_C, j)
                issue_b_scale_load(b_scale_v[K_C_LOCAL], K_C)

        for OFFSET in range_constexpr(kUnroll):
            K_C_LOCAL = k_stages + OFFSET
            K_C = global_k_tile(K_C_LOCAL)
            read_slot = OFFSET % kAStages
            write_slot = K_C_LOCAL % kAStages
            slot_b = OFFSET % k_stages
            if const_expr(inline_quant and prefetch_hidden):
                h_v0, h_v1 = hidden_prefetch
                if const_expr(kUnroll > OFFSET + 1):
                    hidden_prefetch = (
                        inline_quant_load_kt(0, K_C + 1, cached_row_inline),
                        inline_quant_load_kt(1, K_C + 1, cached_row_inline),
                    )
            gpu.barrier()
            if const_expr(BM == 128):
                asc_cur = issue_a_scale_ds_read(K_C - k_stages)
                a_cur = issue_a_ds_read(read_slot)
            else:
                a_cur = issue_a_ds_read(read_slot)
                asc_cur = issue_a_scale_ds_read(K_C - k_stages)
            if const_expr(not inline_quant):
                # Keep the LDS reads ahead of the next DMA into LDS: if the DMA is
                # hoisted above them, the waitcnt pass cannot tell the slots apart
                # and drains every in-flight load (vmcnt(0)) before the reads.
                rocdl.sched_barrier(0)
                issue_a_load_lds(write_slot, K_C)
            if const_expr(inline_quant and not prefetch_hidden):
                h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline)
                h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline)
                rocdl.sched_barrier(0)
            for J in range_constexpr(N_REPS):
                if const_expr(BM != 128):
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                mfma_cluster(b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J)
                if const_expr(BM != 128):
                    rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                issue_b_load_j(b[slot_b], K_C, J)
                rocdl.sched_barrier(0)
            issue_b_scale_load(b_scale_v[slot_b], K_C)
            if const_expr(inline_quant):
                scale_accum = _inline_quant_core_batch(
                    [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
                )
                inline_quant_pack_write(K_C, scale_accum)

        for S in range_constexpr(k_stages):
            kt_local = K_TILES_PER_WAVE - k_stages + S
            kt = global_k_tile(kt_local)
            gpu.barrier()
            if const_expr(BM == 128):
                asc_cur = issue_a_scale_ds_read(kt)
                a_cur = issue_a_ds_read(kt_local % kAStages)
            else:
                a_cur = issue_a_ds_read(kt_local % kAStages)
                asc_cur = issue_a_scale_ds_read(kt)
            for J in range_constexpr(N_REPS):
                mfma_cluster(
                    b[kt_local % k_stages],
                    a_cur,
                    asc_cur,
                    b_scale_v[kt_local % k_stages],
                    J,
                )

    if const_expr(k_wave == 1):
        run_k_slice(0)
    elif const_expr(k_wave == 2):
        if wave_k == fx.Int32(0):
            run_k_slice(0)
        else:
            run_k_slice(K_TILES_PER_WAVE)
    else:
        if wave_k == fx.Int32(0):
            run_k_slice(0)
        elif wave_k == fx.Int32(1):
            run_k_slice(K_TILES_PER_WAVE)
        elif wave_k == fx.Int32(2):
            run_k_slice(2 * K_TILES_PER_WAVE)
        else:
            run_k_slice(3 * K_TILES_PER_WAVE)

    gpu.barrier()

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    # Four dwords of row padding rotate each four-row store group by 16 banks
    # on gfx950, avoiding the 4-way conflict from a BN-aligned row stride.
    ACC_LDS_STRIDE = BN + ACC_LDS_PAD_DW
    ACC_ROWS = BM // epi_splits
    acc_layout = fx.make_layout((ACC_ROWS, BN), (ACC_LDS_STRIDE, 1))
    ACC_GROUP_STRIDE = ACC_ROWS * ACC_LDS_STRIDE

    def acc_idx(group, row, col):
        return group * fx.Int32(ACC_GROUP_STRIDE) + _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr),
        fx.make_layout(k_wave * ACC_GROUP_STRIDE, 1),
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, mem_1x1)

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
        r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
        fx.copy(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
        fx.copy(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    def acc_load_sum(row, col):
        value = acc_load(acc_idx(fx.Int32(0), row, col))
        for group in range_constexpr(1, k_wave):
            value = value + acc_load(acc_idx(fx.Int32(group), row, col))
        return value

    tx_i32 = wave_n * fx.Int32(64) + lane
    if const_expr(num_waves == 2):
        m_lane = tx_i32 // fx.Int32(4)
        n_lane = tx_i32 % fx.Int32(4)
        wave_grp = fx.Int32(0)
        kk = n_lane
    else:
        m_lane = tx_i32 // fx.Int32(16)
        n_lane = tx_i32 % fx.Int32(16)
        wave_grp = n_lane // fx.Int32(4)
        kk = n_lane % fx.Int32(4)

    CHUNKS_PER_SPLIT = kMChunks // epi_splits

    def store_acc_split(split):
        for i_local in range_constexpr(CHUNKS_PER_SPLIT):
            i = split * CHUNKS_PER_SPLIT + i_local
            row_base = fx.Int32(i_local * 16) + lane_div_16 * fx.Int32(4)
            for J in range_constexpr(N_REPS):
                # Both weight layouts assign alternating gate/up tiles to the
                # same accumulators, including BN64's pairs of adjacent waves.
                if const_expr(BN == 64):
                    if const_expr(num_waves == 2):
                        gate_up = fx.Int32(J % 2)
                        n16 = wave_n
                        lds_col = (
                            gate_up * fx.Int32(BN_INT)
                            + n16 * fx.Int32(16)
                            + lane_mod_16
                        )
                    else:
                        gate_up = wave_n & fx.Int32(1)
                        n16 = wave_n >> fx.Int32(1)
                        lds_col = (
                            gate_up * fx.Int32(BN_INT)
                            + n16 * fx.Int32(16)
                            + lane_mod_16
                        )
                else:
                    is_up = (J % 2) == 1
                    J_local = J // 2
                    col_local = (
                        wave_n * fx.Int32(BN // 8)
                        + fx.Int32(J_local * 16)
                        + lane_mod_16
                    )
                    lds_col = fx.Int32(BN_INT) + col_local if is_up else col_local
                vec = fx.Vector(fx.memref_load_vec(accm[i][J]))
                for v in range_constexpr(4):
                    idx = acc_idx(wave_k, row_base + fx.Int32(v), lds_col)
                    acc_store(idx, vec[v])

    MR_PER_SPLIT = EPILOGUE_M_REPS // epi_splits
    scales_per_mr = [None] * EPILOGUE_M_REPS

    def run_epilogue(split):
        aqout_layout = fx.make_layout((BM, OUT_ROW_BYTES), (OUT_ROW_BYTES, 1))
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)

        def store_payload(payload_row, packed0, packed1):
            if const_expr(out_dtype == "fp8"):
                byte_pos = (
                    n_block_idx * fx.Int32(BN_INT)
                    + wave_grp * fx.Int32(32)
                    + kk * fx.Int32(8)
                )
                store_off = _layout_idx(aqout_layout, payload_row, byte_pos)
                _out_store(
                    wt_out,
                    aqout_tiles,
                    arg_aqout,
                    store_off // fx.Int32(4),
                    packed0,
                    fx.Int32,
                )
                _out_store(
                    wt_out,
                    aqout_tiles,
                    arg_aqout,
                    (store_off + fx.Int32(4)) // fx.Int32(4),
                    packed1,
                    fx.Int32,
                )
            else:
                byte_pos = (
                    n_block_idx * fx.Int32(BN_INT // 2)
                    + wave_grp * fx.Int32(16)
                    + kk * fx.Int32(4)
                )
                store_off = _layout_idx(aqout_layout, payload_row, byte_pos)
                _out_store(
                    wt_out,
                    aqout_tiles,
                    arg_aqout,
                    store_off // fx.Int32(4),
                    packed0,
                    fx.Int32,
                )

        if const_expr(enable_bias):
            bias_gate = [None] * 8
            bias_up = [None] * 8
            bias_base = e * fx.Int32(N_OUT)
            for ee in range_constexpr(8):
                out_col = n_block_idx * fx.Int32(BN_INT) + (
                    wave_grp * fx.Int32(32) + fx.Int32(8) * kk + fx.Int32(ee)
                )
                bias_gate[ee] = _global_f32_at(arg_bias, bias_base + out_col)
                bias_up[ee] = _global_f32_at(
                    arg_bias, bias_base + fx.Int32(inter) + out_col
                )

        for mr in range_constexpr(split * MR_PER_SPLIT, (split + 1) * MR_PER_SPLIT):
            row_lds = fx.Int32((mr - split * MR_PER_SPLIT) * 16) + m_lane
            row_local = fx.Int32(mr * 16) + m_lane

            gate_vs = [None] * 8
            up_vs = [None] * 8
            for ee in range_constexpr(8):
                col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
                gate_col = wave_grp * fx.Int32(32) + col_in_grp
                up_col = fx.Int32(BN_INT) + gate_col
                gate_vs[ee] = acc_load_sum(row_lds, gate_col)
                up_vs[ee] = acc_load_sum(row_lds, up_col)
                if const_expr(enable_bias):
                    gate_vs[ee] = gate_vs[ee] + bias_gate[ee]
                    up_vs[ee] = up_vs[ee] + bias_up[ee]
            result = _activation_mul_batch(
                gate_vs,
                up_vs,
                act=act,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                swiglu_limit=swiglu_limit,
            )
            if const_expr(out_dtype == "fp8"):
                # Match the existing two-stage path: activation is materialized
                # as BF16 before the inter-stage MXFP8 quantization.
                result = [value.to(fx.BFloat16).to(fx.Float32) for value in result]

            local_max = _fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_fabs_f32(result[ee]))
            lm_i = _inline_dpp_quad_amax(local_max.bitcast(fx.Int32))
            local_max = lm_i.bitcast(fx.Float32)

            e8m0, qscale = _e8m0_from_amax(
                local_max, max_norm=448.0 if out_dtype == "fp8" else 6.0
            )
            scales_per_mr[mr] = e8m0

            qscale_raw = as_ir_value(qscale)
            sorted_out_row = m_row + row_local
            if const_expr(out_dtype == "fp8"):
                i16x2 = fx.Vector.make_type(2, fx.Int16)
                zero16 = fx.Vector.filled(2, 0, fx.Int16)
                packed0 = rocdl.cvt_scalef32_pk_fp8_f32(
                    i16x2,
                    zero16,
                    as_ir_value(result[0]),
                    as_ir_value(result[1]),
                    qscale_raw,
                    0,
                )
                packed0 = rocdl.cvt_scalef32_pk_fp8_f32(
                    i16x2,
                    packed0,
                    as_ir_value(result[2]),
                    as_ir_value(result[3]),
                    qscale_raw,
                    1,
                )
                packed1 = rocdl.cvt_scalef32_pk_fp8_f32(
                    i16x2,
                    zero16,
                    as_ir_value(result[4]),
                    as_ir_value(result[5]),
                    qscale_raw,
                    0,
                )
                packed1 = rocdl.cvt_scalef32_pk_fp8_f32(
                    i16x2,
                    packed1,
                    as_ir_value(result[6]),
                    as_ir_value(result[7]),
                    qscale_raw,
                    1,
                )
                store_payload(
                    sorted_out_row,
                    fx.Vector(packed0).bitcast(fx.Int32)[0],
                    fx.Vector(packed1).bitcast(fx.Int32)[0],
                )
            else:
                packed_i32 = as_ir_value(fx.Int32(0))
                for w in range_constexpr(4):
                    packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        packed_i32,
                        as_ir_value(result[2 * w]),
                        as_ir_value(result[2 * w + 1]),
                        qscale_raw,
                        w,
                    )
                store_payload(
                    sorted_out_row,
                    fx.Int32(packed_i32),
                    fx.Int32(0),
                )

    def store_scales():
        # Generic scale chunks cover 32 sorted rows; native BM16 GEMM2 consumes
        # one padded chunk per M block. Derive the active runtime extent.
        num_scale_chunks = (
            i32_total_m_blocks
            if native_scale_layout
            else (i32_total_m_blocks * fx.Int32(BM) + fx.Int32(31)) // fx.Int32(32)
        )
        ascaleout_layout = fx.make_layout(
            (num_scale_chunks, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i8_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
        ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        if kk == fx.Int32(0):
            if const_expr(BN == 64):
                ku = n_block_idx >> fx.Int32(3)
                scale_wave_grp = n_block_idx & fx.Int32(3)
                ikxdl = (n_block_idx >> fx.Int32(2)) & fx.Int32(1)
            elif const_expr(BN == 128):
                # Match shuffle_scale's physical layout for logical scale column
                # c = 2*n_block_idx + wave_grp:
                #   (ku, n4, n2) = (c//8, c%4, (c//4)%2).
                ku = n_block_idx >> fx.Int32(2)
                scale_wave_grp = (n_block_idx * fx.Int32(2) + wave_grp) & fx.Int32(3)
                ikxdl = (n_block_idx >> fx.Int32(1)) & fx.Int32(1)
            else:
                ku = n_block_idx >> fx.Int32(1)
                scale_wave_grp = wave_grp
                ikxdl = n_block_idx & fx.Int32(1)
            if const_expr(BM == 16):
                if const_expr(native_scale_layout):
                    # Native GEMM2 addresses one padded scale chunk per BM16
                    # block. Each A scale is duplicated for GEMM2's two N16
                    # MFMA selectors.
                    chunk = m_block_idx
                    dword_off = _layout_idx(
                        ascaleout_layout, chunk, ku, scale_wave_grp, m_lane
                    )
                    duplicated_scale = scales_per_mr[0] | (
                        scales_per_mr[0] << fx.Int32(8)
                    )
                    # ikxdl selects one of the two disjoint 16-bit halves of
                    # the dword, so a narrow store replaces the atomic merge.
                    _out_store(
                        wt_out,
                        ascaleout_i16_tiles,
                        arg_ascaleout,
                        dword_off * fx.Int32(2) + ikxdl,
                        duplicated_scale,
                        fx.Int16,
                    )
                else:
                    # Generic v2 GEMM2 packs two BM16 blocks into one 32-row
                    # scale chunk.
                    chunk = m_block_idx >> fx.Int32(1)
                    row_half = m_block_idx & fx.Int32(1)
                    dword_off = _layout_idx(
                        ascaleout_layout, chunk, ku, scale_wave_grp, m_lane
                    )
                    byte_pos = ikxdl * fx.Int32(2) + row_half
                    _out_store(
                        wt_out,
                        ascaleout_i8_tiles,
                        arg_ascaleout,
                        dword_off * fx.Int32(4) + byte_pos,
                        scales_per_mr[0],
                        fx.Int8,
                    )
            elif const_expr(num_waves == 2):
                chunk = m_block_idx
                row_half = m_lane >> fx.Int32(4)
                row_in_16 = m_lane & fx.Int32(15)
                dword_off = _layout_idx(
                    ascaleout_layout, chunk, ku, scale_wave_grp, row_in_16
                )
                byte_pos = ikxdl * fx.Int32(2) + row_half
                _out_store(
                    wt_out,
                    ascaleout_i8_tiles,
                    arg_ascaleout,
                    dword_off * fx.Int32(4) + byte_pos,
                    scales_per_mr[0],
                    fx.Int8,
                )
            else:
                for sub in range_constexpr(kSubBlocks):
                    chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                    dword_off = _layout_idx(
                        ascaleout_layout, chunk, ku, scale_wave_grp, m_lane
                    )
                    pair_i32 = scales_per_mr[sub * 2 + 0] | (
                        scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                    )
                    addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                    _out_store(
                        wt_out,
                        ascaleout_i16_tiles,
                        arg_ascaleout,
                        addr // fx.Int32(2),
                        pair_i32,
                        fx.Int16,
                    )

    if const_expr(epi_splits > 1):
        for split in range_constexpr(epi_splits):
            if const_expr(split > 0):
                gpu.barrier()
            store_acc_split(split)
            gpu.barrier()
            run_epilogue(split)
        store_scales()
    else:
        store_acc_split(0)
        gpu.barrier()
        if wave_k == fx.Int32(0):
            if const_expr(num_waves == 2):
                run_epilogue(0)
                store_scales()
            elif const_expr(BN == 64):
                if wave_grp == fx.Int32(0):
                    run_epilogue(0)
                    store_scales()
            elif const_expr(BN == 128):
                if wave_grp < fx.Int32(2):
                    run_epilogue(0)
                    store_scales()
            else:
                run_epilogue(0)
                store_scales()
