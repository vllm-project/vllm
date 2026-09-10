# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Decode down GEMM (bf16 intermediate x MXFP8 W2, MFMA 16x16x32) with a
routing-weighted bf16 atomic-add epilogue.

Each workgroup handles an m-block, a 128-column n-block and a split-K index,
with A staged through LDS, W streamed per wave and an LDS-staged atomic epilogue:

* a 256-K tile of a column is four 1 KB preshuffled blocks of 64 K (fp4: two of
  128 K); lane (klane, n) holds K ``klane*16 .. +16`` of block ``k0`` -> two MFMA
  K-steps of 8 per block, unpacked with ``v_cvt_scalef32_pk_bf16_fp8``.
* the A fragment of block ``k0``, K-step ``ku`` is row ``l16``, K ``k0*64 +
  q16*16 + ku*8`` of the tile.
* scales: a lane's 16 K lie in 32-K group ``(k0%2)*2 + klane//2`` of the 128-K
  half ``k0//2``, so per 256-K tile it loads two scale dwords (groups
  ``klane//2`` and ``2 + klane//2``, bytes: 128-K half, N-half).

* a block of ``BM`` sorted rows is ``RT = BM/16`` MFMA row tiles that share every
  unpacked W fragment. With ``wide=True`` (sorted layout of ``sort_decode``'s
  ``wide_first``) the shared expert's rows come first in WIDE_SORT_BM-row
  blocks, which this kernel runs as WIDE_BM-row blocks with the wide body; the
  routed experts follow in ``BM``-row blocks.

Layouts (aiter ``shuffle_weight(is_guinterleave=True, gate_up=False)`` +
``shuffle_scale``, what vLLM's ``shuffle_mxfp8_moe_weights`` stores for w2):
  W2     [E, H/16, K/64, klane 4, nlane 16, 16 B]  fp8 e4m3
  W2_sc  [E*H/32, K/256, klane 4, nlane 16] dwords  e8m0 (bytes: 128-K half, N-half)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.atomic import _atomic_bf16_epilog
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import (
    _e8m0_byte_to_f32,
    _gep1,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_view,
    _raw,
    _swizzle_xor16,
    inline_sort_max_pairs,
    inline_sort_table,
)

from .utils import _fp8x8_to_bf16

BM = 16  # default rows per m-block; compile_gemm2(BM=32) runs two 16-row tiles
# Tiles: 128 output columns x 256 K per workgroup (MI355X 09-09 sweep: twice the
# workgroups of the a16w4 256-column tile, 0.5-1 us faster at every M by better
# balance across CUs). Split-K 3 up to KSPLIT_SMALL_M_TOKENS is worth ~1.5 us of
# latency hiding; at 256 tokens it costs ~5 us of extra atomics, so larger batches
# run unsplit.
TILE_N = 128
TILE_K = 256
BLOCK_K = 64  # K per 1 KB W block (16 columns)
KSPLIT_SMALL_M = 3
KSPLIT_SMALL_M_TOKENS = 64
# Wide blocks of the shared expert: the sort pads it to WIDE_SORT_BM-row blocks
# (gemm1's wide block); this kernel runs them as WIDE_BM-row blocks so its W2
# streams M/WIDE_BM times instead of M/16 (a 32-row A tile keeps the LDS and
# accumulator footprint within the 16-row tile's occupancy).
WIDE_SORT_BM = 128
WIDE_BM = 32


class _Tile:
    """Compile-time constants of one block body (row tiles x the fixed N/K tile)."""

    def __init__(self, RT):
        self.RT = RT
        self.BM = RT * 16
        self.KH_TILE_BYTES = TILE_K * 2  # A bytes per row per tile
        self.KB16 = self.KH_TILE_BYTES // 16
        self.TILE_K_DW = self.KH_TILE_BYTES // 4
        self.NLD = (self.BM * self.KH_TILE_BYTES) // (
            256 * 16
        )  # A copies per lane per tile
        self.A_BYTES = self.BM * self.KH_TILE_BYTES
        self.LDS_BYTES = max(self.A_BYTES, self.BM * TILE_N * 4)  # epilogue reuses A


def compile_gemm2(
    *,
    NE,
    N_OUT,
    D_INTER,
    n_tokens,
    inline_sort=False,
    TOPK=None,
    BM=BM,
    wide=False,
):
    """N_OUT = hidden size (output columns), D_INTER = contraction. Kernel for
    batches of up to ``n_tokens`` tokens: that picks split-K (``launch.ksplit``
    CTAs per tile, each over D_INTER/ksplit) and the inline-sort scan length
    (``launch.kernel_name``); ``inline_sort`` needs ``TOPK``. ``launch.tile_n`` is
    the N tile for the grid. ``BM`` (16 or 32) is the sort's row block of the
    routed experts; ``wide`` (sorted mode) adds the body for the shared expert's
    wide blocks, which come first: ``ceil(n_tokens/WIDE_SORT_BM) *
    (WIDE_SORT_BM/WIDE_BM)`` wide blocks of ``launch.wide_bm`` rows."""
    assert BM in (16, 32), BM
    ksplit = KSPLIT_SMALL_M if n_tokens <= KSPLIT_SMALL_M_TOKENS else 1
    b_cache_mod = 2  # non-temporal W loads
    K = D_INTER
    assert K % TILE_K == 0 and N_OUT % TILE_N == 0
    NNB = N_OUT // TILE_N
    KT_ALL = K // TILE_K
    KT = KT_ALL // ksplit  # TILE_K tiles per CTA
    K0 = TILE_K // BLOCK_K  # 1 KB W blocks per tile
    KB_ALL = K // BLOCK_K  # W blocks per column group
    NPW = TILE_N // 4
    NI = NPW // 16
    narrow = _Tile(BM // 16)
    bodies = [narrow]
    if wide:
        assert not inline_sort and WIDE_SORT_BM % WIDE_BM == 0
        widet = _Tile(WIDE_BM // 16)
        bodies.append(widet)
    LDS_BYTES = max(t.LDS_BYTES for t in bodies)
    tab_off = LDS_BYTES  # routing table (inline sort only): BM rows + sentinel slot
    LDS_BYTES += 256
    if inline_sort:
        assert TOPK, "inline sort needs TOPK"
        assert n_tokens <= BM, "inline sort: every expert's rows fit one m-block"
        max_pairs = inline_sort_max_pairs(n_tokens, TOPK, BM)
    W_BYTES = NE * N_OUT * K
    SC_K1 = K // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"
    # padding rows: A loads pointed here (>= num_records 0xFFFFC000) read zeros
    A_OOB_DW = 0x3FFFF000

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"m3_gemm2_a16w8_ne{NE}_h{N_OUT}_i{K}_tn{TILE_N}_tk{TILE_K}_ks{ksplit}_bcm{b_cache_mod}"
        + (f"_isort{max_pairs}" if inline_sort else "")
        + (f"_bm{BM}" if BM != 16 else "")
        + (f"_wide{WIDE_BM}of{WIDE_SORT_BM}" if wide else "")
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        arg_out: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        if const_expr(ksplit > 1):
            tile, ks = pid // fx.Int32(ksplit), pid % fx.Int32(ksplit)
        else:
            tile, ks = pid, fx.Int32(0)
        m_lane = tx // fx.Int32(32)
        sw_base = _global_base_ptr1(arg_sweights)
        xbuf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFC000))
        x_tiles4 = fx.logical_divide(xbuf, fx.make_layout(4, 1))
        x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
        a_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        wr = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_bq)), num_records_bytes=W_BYTES
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_bscale)), num_records_bytes=SW_BYTES
        )
        c_k_dw = (K * 2) // 4
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        acc_layout = fx.make_layout(4, 1)
        zero4 = Vec.filled(4, 0.0, fx.Float32)

        def _frag(v8):
            t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            t.store(v8)
            return t

        def rows_of(c, stid_at, sweight_at):
            """Routing of a block's rows (token id and weight per row for the
            epilogue, the pad mask of the A copies), issued up front."""
            packed, weight = [], []
            for mr in range_constexpr(c.BM // 8):
                packed.append(stid_at(fx.Int32(mr * 8) + m_lane))
                weight.append(sweight_at(fx.Int32(mr * 8) + m_lane, packed[-1]))
            # A copies: lane covers dwords tx*4 + i*1024 of the [BM, TILE_K] bf16 tile
            row_local = [
                (tx * fx.Int32(4) + fx.Int32(i * 1024)) // fx.Int32(c.TILE_K_DW)
                for i in range_constexpr(c.NLD)
            ]
            col_dw = [
                (tx * fx.Int32(4) + fx.Int32(i * 1024)) % fx.Int32(c.TILE_K_DW)
                for i in range_constexpr(c.NLD)
            ]
            row_valid = [
                (fx.Int32(stid_at(row_local[i])) & fx.Int32(0x00FFFFFF)) < i32_M
                for i in range_constexpr(c.NLD)
            ]
            return packed, weight, row_local, col_dw, row_valid

        def body(c, mrow, nb, e, packed, weight, row_local, col_dw, row_valid):
            """One (block, n-block, K range): A tiles by DMA into LDS, W tiles per
            wave, MFMAs over the block's RT row tiles, atomic epilogue."""
            expert_off = e * fx.Int32(N_OUT)
            by_n = nb * fx.Int32(TILE_N)
            s_x = fx.make_view(
                fx.recast_iter(fx.Int32, smem), fx.make_layout(c.A_BYTES // 4, 1)
            )
            s_x_tiles4 = fx.logical_divide(s_x, fx.make_layout(4, 1))
            row_base_dw = [
                (mrow + row_local[i]) * fx.Int32(c_k_dw) for i in range_constexpr(c.NLD)
            ]

            def dma_a_tile(kt):
                # 16 B per lane straight into LDS; the XOR swizzle is applied to the
                # global column (the LDS destination of a direct load is linear)
                base_dw = (ks * fx.Int32(KT) + fx.Int32(kt)) * fx.Int32(c.TILE_K_DW)
                for i in range_constexpr(c.NLD):
                    col_bytes = col_dw[i] * fx.Int32(4)
                    col_sw = _swizzle_xor16(row_local[i], col_bytes, c.KB16)
                    row_k_dw = row_valid[i].select(
                        row_base_dw[i] + base_dw, fx.Int32(A_OOB_DW)
                    )
                    global_byte = row_k_dw * fx.Int32(4) + col_sw
                    lds_byte = row_local[i] * fx.Int32(c.KH_TILE_BYTES) + col_bytes
                    fx.copy(
                        x_dma_atom,
                        fx.slice(x_tiles4, (None, global_byte // fx.Int32(16))),
                        fx.slice(s_x_tiles4, (None, lds_byte // fx.Int32(16))),
                    )

            def lds_load_a(k0, ku, rt):
                # block k0, K-step ku (8 bf16 per lane): row rt*16 + l16,
                # bytes k0*128 + q16*32 + ku*16
                row = l16 + fx.Int32(rt * 16)
                col = q16 * fx.Int32(32) + fx.Int32(k0 * 128 + ku * 16)
                byte = row * fx.Int32(c.KH_TILE_BYTES) + _swizzle_xor16(
                    row, col, c.KB16
                )
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy_atom_call(
                    a_copy_atom, fx.slice(s_x_tiles4, (None, byte // fx.Int32(16))), r
                )
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)

            # W2 columns of this wave: block (expert_off + col)//16, this CTA's K range
            # starts at 64-K block ks*KT*K0; per tile the block index goes in an SGPR
            col = [
                by_n + wave * fx.Int32(NPW) + fx.Int32(ni * 16)
                for ni in range_constexpr(NI)
            ]
            wvo = [
                lane * fx.Int32(4)
                + ((expert_off + col[ni]) // fx.Int32(16)) * fx.Int32(KB_ALL * 256)
                + ks * fx.Int32(KT * K0 * 256)
                for ni in range_constexpr(NI)
            ]
            # scale dwords of 32-K groups klane//2 (b = 0) and 2 + klane//2 (b = 1)
            lane_sc = [
                (fx.Int32(2 * b) + q16 // fx.Int32(2)) * fx.Int32(16) + l16
                for b in range_constexpr(2)
            ]
            svo = [
                [
                    lane_sc[b]
                    + ((expert_off + col[ni]) // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                    + ks * fx.Int32(KT * 64)
                    for ni in range_constexpr(NI)
                ]
                for b in range_constexpr(2)
            ]
            npk = [
                (col[ni] // fx.Int32(16)) % fx.Int32(2) for ni in range_constexpr(NI)
            ]

            def load_w_tile(kt):
                bb = [
                    [
                        buffer_ops.buffer_load(
                            wr,
                            wvo[ni] + fx.Int32(((kt * K0 + k0) % 4) * 256),
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=b_cache_mod,
                            soffset_bytes=((kt * K0 + k0) // 4) * 4096,
                        )
                        for k0 in range_constexpr(K0)
                    ]
                    for ni in range_constexpr(NI)
                ]
                # two scale dwords per 256-K tile: (kt % 4) dwords of 64 -> imm + SGPR
                sc = [
                    [
                        buffer_ops.buffer_load(
                            sr,
                            svo[b][ni] + fx.Int32((kt % 4) * 64),
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(kt // 4) * 1024,
                        )
                        for ni in range_constexpr(NI)
                    ]
                    for b in range_constexpr(2)
                ]
                return bb, sc

            acc = [
                [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(NI)]
                for _ in range(c.RT)
            ]
            for rt in range_constexpr(c.RT):
                for ni in range_constexpr(NI):
                    acc[rt][ni].store(zero4)

            for kt in range_constexpr(KT):
                dma_a_tile(kt)
                bb, sc = load_w_tile(kt)
                gpu.barrier()
                for ni in range_constexpr(NI):
                    for k0 in range_constexpr(K0):
                        # scale byte: 128-K half k0//2 -> +2, N-half npk; dword k0%2
                        s = _e8m0_byte_to_f32(
                            fx.Int32(sc[k0 % 2][ni]),
                            fx.Int32((k0 // 2) * 2) + npk[ni],
                        )
                        raw4 = fx.Vector(bb[ni][k0])
                        for ku in range_constexpr(2):
                            # one unpacked W fragment feeds every row tile
                            b_t = _frag(
                                _fp8x8_to_bf16(raw4[2 * ku], raw4[2 * ku + 1], s)
                            )
                            for rt in range_constexpr(c.RT):
                                fx.gemm(
                                    mma_atom,
                                    acc[rt][ni],
                                    _frag(lds_load_a(k0, ku, rt)),
                                    b_t,
                                    acc[rt][ni],
                                )
                gpu.barrier()

            # epilogue: the A region is free once every wave passed the last barrier
            gpu.barrier()
            _atomic_bf16_epilog(
                fx.Int32(fx.ptrtoint(smem)),
                [
                    [acc[rt][ni].load().ir_value() for ni in range(NI)]
                    for rt in range(c.RT)
                ],
                arg_out,
                nb,
                wave,
                lane,
                i32_M,
                N_OUT,
                TILE_N,
                packed,
                weight,
                bm=c.BM,
            )

        if const_expr(inline_sort):
            # block = routing pair mb: expert + rows from a ballot over the pairs
            mb, nb = tile // fx.Int32(NNB), tile % fx.Int32(NNB)
            mrow = mb * fx.Int32(BM)
            tab = fx.recast_iter(fx.Int32, smem + tab_off)
            e, owner, _, build_tab = inline_sort_table(
                arg_stids, i32_M, TOPK, mb, lane, tab, max_pairs=max_pairs, bm=BM
            )
            if owner:
                build_tab()
            np_m1 = i32_M * fx.Int32(TOPK) - fx.Int32(1)

            def stid_at(row):  # token | slot<<24 for row of this block (LDS table)
                return tab[row]

            def sweight_at(row, fused):  # topk_weights[token*TOPK + slot], pads clamped
                f = fx.Int32(fused)
                pair = (f & fx.Int32(0x00FFFFFF)) * fx.Int32(TOPK) + (f >> fx.Int32(24))
                pair = fx.Int32(arith.minsi(_raw(pair), _raw(np_m1)))
                return llvm.load(
                    T.f32, _gep1(sw_base, pair * fx.Int32(4)), invariant=True
                )

            if owner:
                rows = rows_of(narrow, stid_at, sweight_at)
                body(narrow, mrow, nb, e, *rows)
        else:
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            stids_base = _global_base_ptr1(arg_stids)

            def sorted_block(c, mb_id, mrow, nb, go):
                e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, mb_id)))

                def stid_at(row):
                    return llvm.load(
                        T.i32,
                        _gep1(stids_base, (mrow + row) * fx.Int32(4)),
                        invariant=True,
                    )

                def sweight_at(row, fused):
                    return llvm.load(
                        T.f32,
                        _gep1(sw_base, (mrow + row) * fx.Int32(4)),
                        invariant=True,
                    )

                rows = rows_of(c, stid_at, sweight_at)
                if go:
                    body(c, mrow, nb, e, *rows)

            if const_expr(wide):
                # the shared expert's sort blocks come first, each WIDE_SORT_BM/WIDE_BM
                # wide blocks of this kernel
                SUB = WIDE_SORT_BM // WIDE_BM
                n_sort_wide = (i32_M + fx.Int32(WIDE_SORT_BM - 1)) // fx.Int32(
                    WIDE_SORT_BM
                )
                n_wide = n_sort_wide * fx.Int32(SUB)
                n_wide_tiles = n_wide * fx.Int32(NNB)
                # (distinct names per branch: the AST rewriter yields variables that
                # both branches of a dynamic if assign)
                if tile < n_wide_tiles:
                    mbw = tile // fx.Int32(NNB)
                    sorted_block(
                        widet,
                        mbw // fx.Int32(SUB),
                        mbw * fx.Int32(WIDE_BM),
                        tile % fx.Int32(NNB),
                        mbw < n_wide,
                    )
                if tile >= n_wide_tiles:
                    qn = tile - n_wide_tiles
                    mbn = qn // fx.Int32(NNB)
                    mrown = n_sort_wide * fx.Int32(WIDE_SORT_BM) + mbn * fx.Int32(BM)
                    sorted_block(
                        narrow,
                        n_sort_wide + mbn,
                        mrown,
                        qn % fx.Int32(NNB),
                        mrown < cumsum0,
                    )
            else:
                mb, nb = tile // fx.Int32(NNB), tile % fx.Int32(NNB)
                mrow = mb * fx.Int32(BM)
                sorted_block(narrow, mb, mrow, nb, mrow < cumsum0)

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            arg_out,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.tile_n = TILE_N
    launch.ksplit = ksplit
    launch.wide_bm = WIDE_BM if wide else None
    launch.wide_sort_bm = WIDE_SORT_BM if wide else None
    return launch
