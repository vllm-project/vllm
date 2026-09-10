# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-batch down fp8 GEMM (256 < M < 3072) with a routing-weighted bf16 atomic
epilogue: no partials, no reduction.

    out[tok, :] += w[row] * (h[row, :] @ W2[e]^T)      for every sorted row

The prefill gemm2 writes ``[M, topk, H]`` partials and reduces them (at 512
tokens 172 + 8 us against aiter's atomic 112). Below 3072 tokens the output
is small enough for the atomics: the accumulated ``[M, H]`` bf16 is 6..38 MB
and every element takes topk adds.

Per workgroup: ``BM`` sorted rows x ``TN`` output columns; four N-waves of
``TN/4`` columns (``NI`` 16-column tiles). The A tile (``BM`` rows x ``I`` fp8
bytes, contiguous in the sorted intermediate) goes through LDS (row pad 16 B):
at once when it fits 64 KB, else in 2-step batches through two slots as in
gemm1; W2 is streamed per wave (two 1 KB preshuffled blocks per K-step and
16-column tile, ``PREFETCH`` steps deep, non-temporal). The scaled MFMA is the
intrinsic form (accumulators in VGPR, MFMA C layout), the epilogue is
``moe_flydsl_common.atomic._atomic_bf16_epilog`` (LDS [BM, TN] f32, aliasing
the A region, then packed bf16 atomic adds).

Layouts (bytes), those of ``moe_a8w8_prefill/gemm2.py``:
  A        [num_m_blocks*BM, I]            gemm1's sorted fp8 intermediate
  A_scale  [num_m_blocks*BM, I/32]         sorted rows, e8m0-shuffled
  W2       [E, H/16, I/64, 4, 16, 16]      shuffled weights, gate_up=False
  W2_sc    [E*H/32, I/256, 4, 16] dwords   scale bytes: N-half + 2 x 128-K half
  OUT      [n_tokens, H] bf16              zeroed by the caller, atomically accumulated
"""

from typing import Any  # noqa: F401 (used in a type comment)

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_a8w8_prefill.gemm1 import _pack8
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.atomic import _atomic_bf16_epilog
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import _lds_ptr3, _raw

TN = 128  # output columns per workgroup (98 KB of W2: twice the CTAs of a 256-col tile)
PREFETCH = (
    3  # W K-steps in flight per wave (K = 768 is 6 steps); 2 at BM 128 (registers)
)
W_CACHE_MOD = 2  # non-temporal W loads
BLOCK_K = 128
LDS_PAD = 16


def compile_moe_gemm2_mid(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    BM: int,
    sort_block_m: int | None = None,
):
    """Grouped fp8 gemm2 for one (H, I, E, BM); ``BM`` (32 / 64 / 128) is the row
    tile, ``sort_block_m`` (default ``BM``) the sort block of the inputs: a
    128-row sort can run in 64-row tiles. Grid: ``num_tiles * launch.n_tiles``
    workgroups of 256 threads, ``num_tiles`` = sort blocks x ``sort_block_m //
    BM``, workgroup ``(mb, nb)`` = ``divmod(bx, n_tiles)``; tiles at or past
    ``num_valid_ids[0]`` exit at once."""
    assert BM in (32, 64, 128), BM
    SBM = BM if sort_block_m is None else sort_block_m
    SUB = SBM // BM  # tiles per sort block
    assert SBM % BM == 0
    NI = TN // 4 // 16  # 16-col tiles per wave
    MR = BM // 16
    KT = I // BLOCK_K
    N_TILES = H // TN
    assert H % TN == 0 and I % 256 == 0 and MR % 2 == 0 and NI % 2 == 0
    # A staging: the whole [BM, I] tile at once when it fits 64 KB of LDS (two
    # CTAs per CU next to the epilogue's [BM, TN] f32), else in KB-step batches
    # through two slots as gemm1 does
    KB = KT if BM * (I + LDS_PAD) <= 64 * 1024 else 2
    PF = 2 if BM == 128 else PREFETCH
    NSLOT = 1 if KB == KT else 2
    assert KT % KB == 0
    ROWB = KB * BLOCK_K  # A bytes per row per batch
    RS = ROWB + LDS_PAD
    CHR = ROWB // 16  # 16 B chunks per row per batch
    NCH = BM * CHR  # chunks per batch
    assert NCH % 256 == 0
    NLD = NCH // 256  # 1 KB wave loads per wave per batch
    SLOT = BM * RS
    A_LDS = NSLOT * SLOT
    EPI_LDS = BM * TN * 4
    LDS_BYTES = max(A_LDS, EPI_LDS)
    N16 = H // 16
    N32 = H // 32
    W2_BYTES = E * H * I
    W2S_BYTES = E * H * (I // 32)
    SC_BLOCKS_PER_ROW32 = I // 256
    assert W2_BYTES <= 0xFFFFFFFF

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(
        name=f"m3_gemm2_a8w8_mid_atomic_h{H}_i{I}_e{E}_bm{BM}_tn{TN}_pf{PF}_kb{KB}"
        + (f"_sbm{SBM}" if SBM != BM else ""),
        known_block_size=[256, 1, 1],
    )
    def kernel_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        arg_out: fx.Int64,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int32(fx.ptrtoint(smem.a.ptr))
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        valid_rows = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        # tiles of BM rows; the sort's blocks hold SUB of them
        mb, nb = bx // N_TILES, bx % N_TILES
        if mb * BM < valid_rows:
            eid_rsrc = buffer_ops.create_buffer_resource(
                sorted_expert_ids,
                max_size=False,
                num_records_bytes=(num_m_blocks // SUB) * 4,
            )
            expert = fx.Int32(
                buffer_ops.buffer_load(
                    eid_rsrc, mb // SUB, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            sw_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * 4),
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=num_m_blocks * (BM * I)
            )
            w_rsrc = buffer_ops.create_buffer_resource(
                W2, max_size=False, num_records_bytes=W2_BYTES
            )
            as_rsrc = buffer_ops.create_buffer_resource(
                A_scale,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * (I // 32)),
            )
            ws_rsrc = buffer_ops.create_buffer_resource(
                W2_scale, max_size=False, num_records_bytes=W2S_BYTES
            )
            m_base = mb * BM
            col_base = nb * TN + wave * (NI * 16)  # this wave's first output column
            n16_base = col_base // 16

            # ---- A: per batch the tile's BM rows x ROWB bytes (rows contiguous in
            #      the sorted intermediate); wave w copies chunks (w*NLD + j)*64 +
            #      lane (16 B each) of the batch ----
            a_gdw = []
            a_lbyte = []
            for j in range_constexpr(NLD):
                c = (wave * NLD + j) * 64 + lane
                a_gdw.append(((m_base + c // CHR) * I + (c % CHR) * 16) // 4)
                a_lbyte.append((c // CHR) * RS + (c % CHR) * 16)

            def load_a_batch(b):
                return [
                    buffer_ops.buffer_load(
                        a_rsrc,
                        a_gdw[j],
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
                        _lds_ptr3(lds_base, slot * SLOT + a_lbyte[j]),
                        alignment=16,
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

            # ---- W2: block (e*N16 + n16) of I/64 K-blocks of 1 KB; lane piece
            #      q16*256 + l16*16; step kt = blocks 2kt, 2kt+1 ----
            w_lane = q16 * 256 + l16 * 16
            w_dw = [
                ((expert * N16 + n16_base + ni) * (I * 16) + w_lane) // 4
                for ni in range_constexpr(NI)
            ]
            as_dw = [
                ((m_base // 32 + mp) * SC_BLOCKS_PER_ROW32) * 64 + q16 * 16 + l16
                for mp in range_constexpr(MR // 2)
            ]
            ws_dw = [
                ((expert * N32 + n16_base // 2 + np) * SC_BLOCKS_PER_ROW32) * 64
                + q16 * 16
                + l16
                for np in range_constexpr(NI // 2)
            ]

            def load_b_step(kt):
                bb = [
                    _pack8(
                        buffer_ops.buffer_load(
                            w_rsrc,
                            w_dw[ni],
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=W_CACHE_MOD,
                            soffset_bytes=kt * 2048,
                        ),
                        buffer_ops.buffer_load(
                            w_rsrc,
                            w_dw[ni],
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=W_CACHE_MOD,
                            soffset_bytes=kt * 2048 + 1024,
                        ),
                    )
                    for ni in range_constexpr(NI)
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
                            ws_dw[np],
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(kt // 2) * 256,
                        )
                        for np in range_constexpr(NI // 2)
                    ]
                    return bb, (sa, sb)
                return bb, None

            rd_base = [(mi * 16 + l16) * RS + q16 * 16 for mi in range_constexpr(MR)]

            def read_a_tile(kt):
                slot, j = (kt // KB) % NSLOT, kt % KB
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

            acc = [
                [Vec.filled(4, 0.0, fx.Float32) for _ in range(NI)] for _ in range(MR)
            ]
            abuf = load_a_batch(0)
            ring = [load_b_step(kt) for kt in range_constexpr(PF)]
            # batch 0 into LDS (its loads went out before the W ring: waiting for
            # them leaves W in flight)
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
                        acc[mi][ni] = Vec(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    fx.as_ir_value(aa[mi]),
                                    fx.as_ir_value(bb[ni]),
                                    fx.as_ir_value(acc[mi][ni]),
                                    0,
                                    0,
                                    mi % 2 + 2 * (kt % 2),
                                    fx.as_ir_value(sa[mi // 2]),
                                    ni % 2 + 2 * (kt % 2),
                                    fx.as_ir_value(sb[ni // 2]),
                                ],
                            )
                        )
                if const_expr(kt % KB == KB - 1 and kt + 1 < KT):
                    stage_a_batch(
                        abuf, (kt // KB + 1) % NSLOT
                    )  # this batch's reads are done
            # epilogue routing (row mr*8 + tx//32 of the tile), loaded here rather
            # than before the K loop: 2 x BM/8 registers fewer live through it
            m_lane = tx // 32
            ep_ids = [
                fx.Int32(
                    buffer_ops.buffer_load(
                        ids_rsrc,
                        m_base + (mr * 8) + m_lane,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                for mr in range_constexpr(BM // 8)
            ]
            ep_weights = [
                fx.Float32(
                    buffer_ops.buffer_load(
                        sw_rsrc,
                        m_base + (mr * 8) + m_lane,
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )
                for mr in range_constexpr(BM // 8)
            ]
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            # every wave is done reading A before the epilogue reuses the LDS
            fx.gpu.barrier()
            _atomic_bf16_epilog(
                lds_base,
                acc,
                arg_out,
                nb,
                wave,
                lane,
                n_tokens,
                H,
                TN,
                ep_ids,
                ep_weights,
                bm=BM,
            )

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        arg_out: fx.Int64,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(grid)
        kernel_gemm2(
            A,
            W2,
            arg_out,
            A_scale,
            W2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm2.n_tiles = N_TILES
    launch_gemm2.tiles_per_block = SUB
    return launch_gemm2
