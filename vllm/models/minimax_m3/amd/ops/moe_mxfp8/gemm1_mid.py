# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-batch gate/up fp8 GEMM (256 < M < 3072) with swiglu-OAI + MXFP8 quant.

    h[row, :] = swiglu_oai( x[tok] @ W_gate[e]^T , x[tok] @ W_up[e]^T )
    out_q[row, :], out_scale[row, :] = mxfp8_quant(h[row, :])      (per 32 cols)

Why not ``gemm1_prefill.py`` here: with random routing every routed
expert holds 16..64 rows at these batch sizes, i.e. one 128-row block, so the
prefill tile (128 rows x 256 W columns, one CTA per CU) runs 792..864
workgroups = three full rounds plus a tail round almost as long: 257 us at
512, 1024 and 2048 tokens against aiter's 206 / 229 / 273 (MI355X, 09-10).
This kernel keeps the same 1.57 MB W13 slice per workgroup but takes the
sort's ``BM``-row blocks (32 / 64 / 128), reads the A tile once per workgroup
through LDS and fits two workgroups per CU (LDS 33..68 KB, ~230 registers).

Per workgroup: ``BM`` sorted rows x ``TN`` output columns (``TN`` gate + ``TN``
up columns of the gate/up-interleaved W13); four N-waves of ``TN/4`` output
columns (``NI`` 16-column tiles of gate and of up each).

* A: ``KB`` K-steps of 128 B per row are gathered once per workgroup
  (``sorted_ids`` -> token row, 16 B per lane) into an LDS slot with a 16 B
  row pad (conflict-free 16 B reads); two slots, one barrier per batch, the
  batch loads issued before the W loads of the same step so the in-order
  vmcnt waits cover them. Padding rows (token == n_tokens) read as zeros
  through the OOB-clamped resource.
* W: per K-step and 16-column tile the two 1 KB preshuffled blocks ``2kt``,
  ``2kt+1`` (16 B per lane each), ``PREFETCH`` steps in flight, non-temporal.
* MFMA: ``v_mfma_scale_f32_16x16x128_f8f6f4`` with fp8 operands, the per-lane
  operand and scale layout of ``gemm1_prefill.py`` (operands fed
  swapped so a lane holds a row's 4 consecutive columns; accumulators in
  AGPR; scale bytes row-half + 2 x step parity / gate-up + 2 x step parity).
* epilogue: per (row, 32-column group) amax over the 4 lanes of the row,
  e8m0 = ceil_pow2(amax / 448), ``v_cvt_scalef32_pk_fp8_f32``, permlane16 swap
  -> 8 B per lane; scale pairs in the e8m0-shuffled sorted layout gemm2 reads.

Layouts (bytes), those of ``gemm1_prefill.py``:
  A         [n_tokens, H]                   per-token fp8 (aiter per_1x32 quant)
  A_scale   [pad32(max_sorted), H/32]       sorted rows, e8m0-shuffled
  W13       [E, I/16, 2, H/64, 4, 16, 16]   shuffle_weight(is_guinterleave, gate_up)
  W13_sc    [E*I/16, H/256, 4, 16] dwords   shuffle_scale(..., True, True)
  OUT_Q     [num_m_blocks*BM, I]            sorted rows, fp8 e4m3
  OUT_sc    [num_m_blocks*BM, I/32]         sorted rows, e8m0-shuffled
  OUT       [n_tokens, H] bf16              zeroed here for gemm2's atomics
"""

from typing import Any  # noqa: F401 (used in a type comment)

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.epilogue import (
    _cvt_pk_fp8,
    _maxf_nn,
    _undef_i32,
    _v2i32,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.loaders import (
    _as_f32,
    _permlane16_swap,
    _swiglu_oai,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import _lds_ptr3, _raw
from vllm.models.minimax_m3.amd.ops.moe_mxfp8.gemm1_prefill import (
    Mfma16x16x128Fp8,
    _e8m0_roundup_fp8,
    _fmax,
    _pack8,
)

# output columns per workgroup (64: a 0.79 MB W13 slice; 128: the prefill tile's)
TN = 64
PREFETCH = 3  # W K-steps in flight per wave (128-row blocks: 2)
W_CACHE_MOD = 2  # non-temporal W loads: no W reuse below 3072 tokens
BLOCK_K = 128  # K per step (128 B per row, one fp8 MFMA)
LDS_PAD = 16  # bytes of padding per staged A row: conflict-free 16 B reads


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that outputs them again
    (tied ``=a`` / ``0``) with the XDL wait states inside: the epilogue's
    ``v_accvgpr_read`` copies then depend on this fence and cannot be hoisted
    right behind the last inline-asm MFMA (the compiler sees no hazard through
    the asm; seen as random wrong rows at BM 64 in the a4w4 lab kernel)."""
    ty = ir.Type.parse(
        "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>"
    )
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(
        ty,
        [_raw(v) for v in values],
        "s_nop 15\ns_nop 15",
        constraints,
        has_side_effects=True,
    )
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


def compile_moe_gemm1_mid(*, H: int, I: int, E: int, BM: int):  # noqa: E741
    """Grouped fp8 gemm1 for one (H, I, E, BM); ``BM`` (32 / 64 / 128) is the sort
    block of the inputs. Grid: ``num_m_blocks * launch.n_tiles`` workgroups of
    256 threads, workgroup ``(mb0, nb)`` = ``divmod(bx, n_tiles)`` with the
    last expert's blocks (``num_valid_ids[1]`` on) mapped first; blocks at or
    past ``num_valid_ids[0]`` exit at once."""
    assert BM in (32, 64, 128), BM
    # A batch (K-steps per LDS slot, 2 slots) and W prefetch depth per block size:
    # 128-row blocks go to 1-step batches and a 2-deep ring to fit 2 waves per
    # SIMD (280 -> 232 registers, LDS 70 -> 38 KB; 2048 tokens: chain -2%, 3 x A/B)
    KB = {32: 4, 64: 4, 128: 1}[BM]
    PF = {32: PREFETCH, 64: PREFETCH, 128: 2}[BM]
    KT = H // BLOCK_K
    NI = TN // 4 // 16  # 16-col tiles per wave, of gate and of up each
    MR = BM // 16  # 16-row tiles
    N_TILES = I // TN  # n-tiles per m-block
    assert H % (BLOCK_K * KB) == 0 and I % TN == 0 and MR % 2 == 0 and NI in (1, 2)
    ROWB = KB * BLOCK_K  # A bytes per row per batch
    CH = ROWB // 16  # 16 B chunks per row per batch
    ROWS_PER_LD = 64 // CH  # rows covered by one dwordx4 wave load
    RPW = BM // 4  # rows staged by each of the 4 waves
    NLD = RPW // ROWS_PER_LD  # loads per wave per batch
    assert 64 % CH == 0 and RPW % ROWS_PER_LD == 0
    RS = ROWB + LDS_PAD  # LDS row stride
    SLOT = BM * RS
    LDS_BYTES = 2 * SLOT
    N16 = I // 16  # 16-column groups of gate (= of up)
    W_BYTES = E * 2 * I * H
    WS_BYTES = E * 2 * I * (H // 32)
    SC_BLOCKS_PER_ROW32 = H // 256  # A / W scale blocks (256 B) per 32-row group
    OUT_SC_BLOCKS_PER_ROW32 = I // 256
    assert W_BYTES <= 0xFFFFFFFF

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]
        amax: fx.Array[fx.Float32, 4 * MR * 16]  # NI 1: per-wave, per-row 16-col amax

    @flyc.kernel(
        name=f"m3_gemm1_a8w8_mid_h{H}_i{I}_e{E}_bm{BM}_tn{TN}_kb{KB}_pf{PF}",
        known_block_size=[256, 1, 1],
    )
    def kernel_gemm1(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        OUT: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_wgs: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int32(fx.ptrtoint(smem.a.ptr))
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        # zero the [n_tokens, H] bf16 output gemm2 accumulates into: every
        # workgroup of the grid (the early-exit ones too) clears its share, 16 B
        # per thread per round, the OOB-clamped resource cutting the last share
        zero_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=n_tokens * (H * 2)
        )
        total_dw = n_tokens * (H // 2)
        per_wg = ((total_dw + n_wgs - 1) // n_wgs + 3) & ~3
        zero4 = Vec.filled(4, 0, fx.Int32)
        for i in range(bx * per_wg + tx * 4, bx * per_wg + per_wg, 256 * 4):
            buffer_ops.buffer_store(zero4, zero_rsrc, fx.Int32(i))
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=8
        )
        valid_rows = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        last_start = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(1), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        # block order: the last expert's blocks first (the fused shared expert,
        # M / BM blocks re-reading one W13 slice per tile: dispatched together
        # L2 / MALL serve the re-reads; as the grid's tail they ran latency-bound
        # -> gemm1 -8 us at 512..2048 tokens), then the routed blocks, then the
        # blocks past num_valid_ids[0], which exit
        valid_blocks = valid_rows // BM
        last_blocks = valid_blocks - last_start // BM
        mb0, nb = bx // N_TILES, bx % N_TILES
        mb = mb0 - last_blocks
        if mb < 0:
            mb = mb + valid_blocks
        if mb0 >= valid_blocks:
            mb = mb0
        if mb * BM < valid_rows:
            eid_rsrc = buffer_ops.create_buffer_resource(
                sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
            )
            expert = fx.Int32(
                buffer_ops.buffer_load(
                    eid_rsrc, mb, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=n_tokens * H
            )
            w_rsrc = buffer_ops.create_buffer_resource(
                W13, max_size=False, num_records_bytes=W_BYTES
            )
            as_rsrc = buffer_ops.create_buffer_resource(
                A_scale, max_size=False, num_records_bytes=a_scale_bytes
            )
            ws_rsrc = buffer_ops.create_buffer_resource(
                W13_scale, max_size=False, num_records_bytes=WS_BYTES
            )
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT_Q, max_size=False, num_records_bytes=num_m_blocks * (BM * I)
            )
            osc_rsrc = buffer_ops.create_buffer_resource(
                OUT_scale,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * (I // 32)),
            )
            m_base = mb * BM
            col_base = nb * TN + wave * (NI * 16)  # this wave's first gate column
            n16_base = col_base // 16

            # ---- A staging: wave w stages rows w*RPW + j*ROWS_PER_LD + lane//CH,
            #      16 B chunk lane%CH of the batch's ROWB bytes ----
            ld_row = [
                wave * RPW + (j * ROWS_PER_LD) + lane // CH
                for j in range_constexpr(NLD)
            ]
            ld_chunk = (lane % CH) * 16
            ld_tok = [
                fx.Int32(
                    buffer_ops.buffer_load(
                        ids_rsrc, m_base + ld_row[j], vec_width=1, dtype=fx.Int32
                    )
                )
                & 0x00FFFFFF
                for j in range_constexpr(NLD)
            ]
            ld_gdw = [(ld_tok[j] * H + ld_chunk) // 4 for j in range_constexpr(NLD)]
            ld_lbyte = [ld_row[j] * RS + ld_chunk for j in range_constexpr(NLD)]

            def load_a_batch(b):
                return [
                    buffer_ops.buffer_load(
                        a_rsrc,
                        ld_gdw[j],
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=b * ROWB,
                    )
                    for j in range_constexpr(NLD)
                ]

            def stage_a_batch(regs, slot):
                for j in range_constexpr(NLD):
                    llvm.StoreOp(
                        _raw(regs[j]),
                        _lds_ptr3(lds_base, slot * SLOT + ld_lbyte[j]),
                        alignment=16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

            # lane (q16, l16) reads row l16 of tile mi, K bytes [16*q16, +16) and
            # [64 + 16*q16, +16) of the step
            rd_base = [(mi * 16 + l16) * RS + q16 * 16 for mi in range_constexpr(MR)]

            def read_a_tile(kt):
                slot, j = (kt // KB) % 2, kt % KB
                out = []
                for mi in range_constexpr(MR):
                    off = rd_base[mi] + (slot * SLOT + j * BLOCK_K)
                    h0 = llvm.load(
                        T.vec(4, T.i32), _lds_ptr3(lds_base, off), alignment=16
                    )
                    h1 = llvm.load(
                        T.vec(4, T.i32), _lds_ptr3(lds_base, off + 64), alignment=16
                    )
                    out.append(_pack8(h0, h1))
                return out

            # ---- W: block ((e*N16 + n16)*2 + gu) of 16 K-blocks of 1 KB; lane piece
            #      q16*256 + l16*16; step kt = blocks 2kt, 2kt+1 ----
            w_lane = q16 * 256 + l16 * 16
            w_dw = [
                [
                    (((expert * N16 + n16_base + ni) * 2 + gu) * (H * 16) + w_lane) // 4
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]
            # scale dwords: A (32-row group, 256-K block); W13 (16-col group, 256-K
            # block)
            as_dw = [
                ((m_base // 32 + mp) * SC_BLOCKS_PER_ROW32) * 64 + q16 * 16 + l16
                for mp in range_constexpr(MR // 2)
            ]
            ws_dw = [
                ((expert * N16 + n16_base + ni) * SC_BLOCKS_PER_ROW32) * 64
                + q16 * 16
                + l16
                for ni in range_constexpr(NI)
            ]

            def load_b_step(kt):
                bb = [
                    [
                        _pack8(
                            buffer_ops.buffer_load(
                                w_rsrc,
                                w_dw[gu][ni],
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=W_CACHE_MOD,
                                soffset_bytes=kt * 2048,
                            ),
                            buffer_ops.buffer_load(
                                w_rsrc,
                                w_dw[gu][ni],
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=W_CACHE_MOD,
                                soffset_bytes=kt * 2048 + 1024,
                            ),
                        )
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if const_expr(kt % 2 == 0):
                    sa = [
                        buffer_ops.buffer_load(
                            as_rsrc,
                            as_dw[mp],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(kt // 2) * 256,
                        )
                        for mp in range_constexpr(MR // 2)
                    ]
                    sb = [
                        buffer_ops.buffer_load(
                            ws_rsrc,
                            ws_dw[ni],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(kt // 2) * 256,
                        )
                        for ni in range_constexpr(NI)
                    ]
                    return bb, (sa, sb)
                return bb, None

            mfma = Mfma16x16x128Fp8(MR, NI)
            acc = [[[None] * 2 for _ in range(NI)] for _ in range(MR)]
            abuf = load_a_batch(0)
            ring = [load_b_step(kt) for kt in range_constexpr(PF)]
            stage_a_batch(abuf, 0)
            sa = sb = None  # type: Any
            for kt in range_constexpr(KT):
                if const_expr(kt % KB == 0 and kt + KB < KT):
                    abuf = load_a_batch(kt // KB + 1)  # before this step's W loads
                if const_expr(kt + PF < KT):
                    ring.append(load_b_step(kt + PF))
                bb, sc = ring.pop(0)
                if const_expr(kt % 2 == 0):
                    sa, sb = sc
                aa = read_a_tile(kt)
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(NI):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = mfma._mfma_agpr(
                                aa[mi],
                                bb[gu][ni],
                                acc[mi][ni][gu],
                                sa[mi // 2],
                                sb[ni],
                                mi % 2,
                                gu,
                                kt % 2,
                            )
                if const_expr(kt % KB == KB - 1 and kt + 1 < KT):
                    stage_a_batch(
                        abuf, (kt // KB + 1) % 2
                    )  # this batch's reads are done

            # ---- epilogue: swiglu-OAI + MXFP8 quant ----
            flat = _pin_accumulators(
                [
                    acc[mi][ni][gu]
                    for mi in range(MR)
                    for ni in range(NI)
                    for gu in range(2)
                ]
            )
            colgrp = col_base // 32
            sc_in_block = (colgrp % 4) * 64 + l16 * 4 + ((colgrp % 8) // 4) * 2
            e8m0_of_mi = []
            if const_expr(NI == 1):
                # the wave's 16 columns are half of a 32-column scale group: waves 2j
                # and 2j+1 exchange their per-row amax through LDS; the even wave
                # stores the group's scales
                amax_x = smem.amax.ptr
                partner = (wave // 2) * 2 + (1 - wave % 2)
                h_of_mi = []
                amax16_of_mi = []
                for mi in range_constexpr(MR):
                    gv = Vec(flat[mi * 2 + 0])
                    uv = Vec(flat[mi * 2 + 1])
                    h = [
                        _swiglu_oai(fx.Float32(gv[v]), fx.Float32(uv[v]))
                        for v in range_constexpr(4)
                    ]
                    amax = _maxf_nn(fx.math.absf(h[0]), fx.math.absf(h[1]))
                    amax = _maxf_nn(amax, fx.math.absf(h[2]))
                    amax = _maxf_nn(amax, fx.math.absf(h[3]))
                    amax = _fmax(amax, amax.shuffle_xor(16, 64))
                    amax = _fmax(amax, amax.shuffle_xor(32, 64))
                    amax_x[(wave * MR + mi) * 16 + l16] = amax
                    h_of_mi.append(h)
                    amax16_of_mi.append(amax)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for mi in range_constexpr(MR):
                    other = fx.Float32(amax_x[(partner * MR + mi) * 16 + l16])
                    e8m0 = _e8m0_roundup_fp8(_fmax(amax16_of_mi[mi], other))
                    e8m0_of_mi.append(e8m0)
                    scale_f = _as_f32(e8m0 << 23)
                    h = h_of_mi[mi]
                    packed = _cvt_pk_fp8(_undef_i32(), h[0], h[1], scale_f, False)
                    packed = _cvt_pk_fp8(packed, h[2], h[3], scale_f, True)
                    row = m_base + mi * 16 + l16
                    # lane group g holds cols 4g .. +4 of the wave's 16-col tile
                    buffer_ops.buffer_store(
                        packed,
                        out_rsrc,
                        row * I + (col_base + q16 * 4),
                        offset_is_bytes=True,
                    )
                for mp in range_constexpr(MR // 2):
                    blk = (m_base // 32 + mp) * OUT_SC_BLOCKS_PER_ROW32 + colgrp // 8
                    pair = e8m0_of_mi[2 * mp] | (e8m0_of_mi[2 * mp + 1] << 8)
                    buffer_ops.buffer_store(
                        fx.Int32(pair).to(fx.Int16),
                        osc_rsrc,
                        blk * 256 + sc_in_block,
                        offset_is_bytes=True,
                        mask=(q16 == 0) & (wave % 2 == 0),
                    )
            for mi in range_constexpr(MR if NI == 2 else 0):
                gv = Vec(flat[(mi * NI + 0) * 2 + 0])
                gw = Vec(flat[(mi * NI + 1) * 2 + 0])
                uv = Vec(flat[(mi * NI + 0) * 2 + 1])
                uw = Vec(flat[(mi * NI + 1) * 2 + 1])
                h = [
                    _swiglu_oai(fx.Float32(gv[v]), fx.Float32(uv[v]))
                    for v in range_constexpr(4)
                ] + [
                    _swiglu_oai(fx.Float32(gw[v]), fx.Float32(uw[v]))
                    for v in range_constexpr(4)
                ]
                amax = _maxf_nn(fx.math.absf(h[0]), fx.math.absf(h[1]))
                for v in range_constexpr(2, 8):
                    amax = _maxf_nn(amax, fx.math.absf(h[v]))
                # the 4 lanes {L, L^16, L^32, L^48} hold the same row
                amax = _fmax(amax, amax.shuffle_xor(16, 64))
                amax = _fmax(amax, amax.shuffle_xor(32, 64))
                e8m0 = _e8m0_roundup_fp8(amax)
                e8m0_of_mi.append(e8m0)
                scale_f = _as_f32(e8m0 << 23)
                da = _cvt_pk_fp8(_undef_i32(), h[0], h[1], scale_f, False)
                da = _cvt_pk_fp8(da, h[2], h[3], scale_f, True)
                db = _cvt_pk_fp8(_undef_i32(), h[4], h[5], scale_f, False)
                db = _cvt_pk_fp8(db, h[6], h[7], scale_f, True)
                da, db = _permlane16_swap(da, db)
                row = m_base + mi * 16 + l16
                # after the swap lane group g holds tile g%2, cols (g//2)*8 .. +8
                col = col_base + (q16 % 2) * 16 + (q16 // 2) * 8
                buffer_ops.buffer_store(
                    _v2i32(da, db), out_rsrc, row * I + col, offset_is_bytes=True
                )
            # scales: rows of tiles 2mp (byte 0) and 2mp+1 (byte 1) of a 32-row group
            for mp in range_constexpr(MR // 2 if NI == 2 else 0):
                blk = (m_base // 32 + mp) * OUT_SC_BLOCKS_PER_ROW32 + colgrp // 8
                pair = e8m0_of_mi[2 * mp] | (e8m0_of_mi[2 * mp + 1] << 8)
                buffer_ops.buffer_store(
                    fx.Int32(pair).to(fx.Int16),
                    osc_rsrc,
                    blk * 256 + sc_in_block,
                    offset_is_bytes=True,
                    mask=q16 == 0,
                )

    @flyc.jit
    def launch_gemm1(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        OUT: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        """``OUT`` is the [n_tokens, H] bf16 output gemm2 accumulates into (zeroed
        here); ``grid`` = ``num_m_blocks * launch_gemm1.n_tiles``."""
        grid_x = fx.Int64(grid)
        kernel_gemm1(
            A,
            W13,
            OUT_Q,
            A_scale,
            W13_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            OUT,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            grid,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm1.n_tiles = N_TILES
    return launch_gemm1
