# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode gate/up GEMM (bf16 x MXFP8 W13, MFMA 16x16x32) fused with swiglu-OAI.

A is staged through LDS in 256 B chunks (128 K of a row); W streams through a
per-wave register ring of 1 KB preshuffled blocks (16 columns x 64 K: lane
(klane, n) holds K ``klane*16 .. +16`` of column n = two MFMA K-steps of 8,
unpacked right before the MFMA with ``v_cvt_scalef32_pk_bf16_fp8``). Per 256 K a
lane loads two scale dwords (32-K groups ``klane//2`` and ``2 + klane//2``; bytes:
128-K tile, gate/up). A block of ``BM`` sorted rows is ``BM/16`` row tiles sharing
every unpacked W fragment; ``wide=True`` adds the ``WIDE_BM``-row body for the
fused shared expert's blocks, which the sort places first.

Layouts (``shuffle_weight(is_guinterleave=True, gate_up=True)``,
``shuffle_scale(..., True, True)``):
  W13     [E, I/16, 2 (gate, up), K/64, klane 4, nlane 16, 16 B]  fp8 e4m3
  W13_sc  [E, I/16, K/256, klane 4, nlane 16] dwords  e8m0 (bytes: kt, gu)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.utils import (
    _e8m0_byte_to_f32,
    _global_i32_ptr,
    _swigluoai_f32,
    inline_sort_max_pairs,
    inline_sort_table,
)


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


BM = 16  # rows per m-block (one MFMA M tile); compile_gemm1(BM=32) runs two
NW = 4  # waves per workgroup
LDS_PAD = 16  # bytes of padding per LDS row: conflict-free 16 B reads
TILE_K = 64  # K per W tile (one 1 KB block per 16 columns)
CHUNK_K = 128  # K per A load chunk (256 B of bf16 per row, one dwordx4 per lane)
PREFETCH = 3  # W tiles in flight per wave
W_CACHE_MOD = 2  # non-temporal W loads
# the fused shared expert's blocks (sorted layout, ``wide=True``): WIDE_BM rows,
# four N-waves of 16 columns, one A chunk per batch; they come first in the grid
WIDE_BM = 128
WIDE_TILE = (64, 1, 1)


def tile_for(n_tokens):
    """(TILE_N, K-waves, A chunks per batch): one N-wave x four K-waves of 16
    columns at every decode batch size; M=4 keeps 32 columns."""
    if n_tokens == 4:
        return (32, 2, 1)
    if 2 <= n_tokens <= 3:
        return (16, 4, 1)
    return (16, 4, 2)


class _Tile:
    """Compile-time constants of one block body (tile shape x row tiles)."""

    def __init__(self, TILE_N, KW, KB, RT, prefetch, K, INTER):
        self.TILE_N, self.KW, self.KB, self.RT, self.prefetch = (
            TILE_N,
            KW,
            KB,
            RT,
            prefetch,
        )
        self.BM = RT * 16
        self.NWN = NW // KW  # N-waves
        self.NPW = TILE_N // self.NWN  # columns per wave (gate and up each)
        self.NI = self.NPW // 16
        self.TPC = CHUNK_K // TILE_K  # W tiles per A chunk
        self.BT = KB * self.TPC  # W tiles per A batch
        assert self.NPW % 16 == 0 and INTER % TILE_N == 0
        assert K % (KW * KB * CHUNK_K) == 0
        self.KT = K // TILE_K  # W tiles
        self.NNB = INTER // TILE_N  # N blocks per m-block
        self.KTW = self.KT // KW  # W tiles per K-wave
        self.KCW = (K // CHUNK_K) // KW  # A chunks per K-wave
        ROWB = KB * 256  # A bytes per row per batch (per K-wave)
        self.RS = ROWB + LDS_PAD  # LDS row stride
        self.KSLOT = self.BM * self.RS  # one K-wave's batch
        self.SLOT = KW * self.KSLOT
        # K-reduce scratch (reuses the A slots)
        self.RED_BYTES = (KW - 1) * RT * self.NWN * 2 * self.NI * 1024
        self.LDS_BYTES = max(2 * self.SLOT, self.RED_BYTES)
        # LDS is addressed in 16 B tiles
        self.RS_T, self.KSLOT_T, self.SLOT_T = (
            self.RS // 16,
            self.KSLOT // 16,
            self.SLOT // 16,
        )


def compile_gemm1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    n_tokens,
    inline_sort=False,
    BM=BM,
    wide=False,
):
    """Kernel for batches of up to ``n_tokens`` tokens (tile choice and inline-sort
    scan length depend on it; ``launch.kernel_name`` / ``launch.tile_n``). ``BM``
    (16 or 32) is the sort's row block; ``wide`` adds the WIDE_BM-row body for the
    shared expert's blocks, which come first in the grid
    (``ceil(n_tokens/WIDE_BM) * launch.wide_n_blocks`` workgroups).
    """
    assert BM in (16, 32), BM
    TILE_N, KW, KB = tile_for(n_tokens)
    K, INTER = D_HIDDEN, D_INTER
    N_OUT = 2 * INTER
    narrow = _Tile(TILE_N, KW, KB, BM // 16, PREFETCH, K, INTER)
    bodies = [narrow]
    if wide:
        assert not inline_sort, "wide blocks belong to the sorted layout"
        widet = _Tile(*WIDE_TILE, WIDE_BM // 16, PREFETCH, K, INTER)
        bodies.append(widet)
    LDS_BYTES = max(t.LDS_BYTES for t in bodies)
    if inline_sort:
        assert n_tokens <= BM, "inline sort: every expert's rows fit one m-block"
        max_pairs = inline_sort_max_pairs(n_tokens, TOPK, BM)
    W_BYTES = NE * N_OUT * K
    SC_K1 = K // 256  # scale dwords per lane per column group
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # A slots / K-reduce scratch
        tab: fx.Array[
            fx.Int32, 64
        ]  # routing table (inline sort only; BM rows + sentinel)

    name = (
        f"m3_gemm1_a16w8_h{K}_i{INTER}_ne{NE}_tn{TILE_N}_kw{KW}_kb{KB}_pf{PREFETCH}"
        f"_bcm{W_CACHE_MOD}"
        + (f"_isort{max_pairs}" if inline_sort else "")
        + (f"_bm{BM}" if BM != 16 else "")
        + (
            f"_wide{WIDE_BM}t{WIDE_TILE[0]}k{WIDE_TILE[1]}b{WIDE_TILE[2]}"
            if wide
            else ""
        )
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_x: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        f32_alpha: fx.Float32,
        f32_limit: fx.Float32,
        arg_out: fx.Int64,
        arg_zero: fx.Int64,
        i32_zero_dw: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16

        # LDS as 16 B tiles; one dwordx4 per lane per copy
        lds16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.a.ptr), fx.make_layout(LDS_BYTES // 4, 1)
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def lds_store16(tile, vec4):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(vec4)
            fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

        def lds_load16(tile):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
            return r.load()

        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=fx.Int64(i32_ntok) * (K * 2)
        )
        wr = buffer_ops.create_buffer_resource_from_addr(
            arg_bq, num_records_bytes=W_BYTES
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_bscale, num_records_bytes=SW_BYTES
        )
        neg_limit = -f32_limit

        def routing_rows(c, mind_at):
            """A-staging row / token per row tile (wave w loads row rt*16 + w*4 +
            lane//16, 16 B chunk j*16 + lane%16) and the epilogue tokens (lane
            (q16, l16) holds rows rt*16 + q16*4 + ii of column l16)."""
            ld_row = [rt * 16 + wave * 4 + q16 for rt in range_constexpr(c.RT)]
            ld_tok = [mind_at(ld_row[rt]) & 0xFFFFFF for rt in range_constexpr(c.RT)]
            ep_tok = [
                [
                    mind_at(rt * 16 + q16 * 4 + ii) & 0xFFFFFF
                    for ii in range_constexpr(4)
                ]
                for rt in range_constexpr(c.RT)
            ]
            return ld_row, ld_tok, ep_tok

        def body(c, mbase, nb, e, ld_row, ld_tok, ep_tok, cumsum0):
            """One block: stream the W tiles of the block's expert through the
            ring, A batches through LDS, swiglu epilogue. ``c`` picks the tile."""
            wave_n, wave_k = wave % c.NWN, wave // c.NWN
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=fx.Int64(cumsum0) * (INTER * 2)
            )
            ld_gdw = [
                (ld_tok[rt] * (K * 2) + l16 * 16) // 4 for rt in range_constexpr(c.RT)
            ]
            ld_tile = [ld_row[rt] * c.RS_T + l16 for rt in range_constexpr(c.RT)]

            def load_a_batch(b):
                # batch b of every K-wave: chunks kw*KCW + b*KB + j of each row tile;
                # base in an SGPR, j*256 in the immediate offset field
                out = []
                for rt in range_constexpr(c.RT):
                    for kw in range_constexpr(c.KW):
                        so = (kw * c.KCW + b * c.KB) * 256
                        out += [
                            fx.Vector(
                                buffer_ops.buffer_load(
                                    xr,
                                    ld_gdw[rt] + j * 64,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    soffset_bytes=so,
                                )
                            )
                            for j in range_constexpr(c.KB)
                        ]
                return out

            def stage_a_batch(regs, slot):
                for rt in range_constexpr(c.RT):
                    for kw in range_constexpr(c.KW):
                        for j in range_constexpr(c.KB):
                            lds_store16(
                                ld_tile[rt]
                                + (slot * c.SLOT_T + kw * c.KSLOT_T + j * 16),
                                regs[(rt * c.KW + kw) * c.KB + j],
                            )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

            # MFMA A fragment of W tile kt, K-step ku, row tile rt: row rt*16 + l16,
            # chunk kt//2, K (kt%2)*64 + q16*16 + ku*8 -> 16 B tile (kt%2)*8 + q16*2
            # + ku of the chunk
            rd_tile = wave_k * c.KSLOT_T + l16 * c.RS_T + q16 * 2

            def read_a(kt, ku, rt):
                ch = kt // c.TPC
                slot, j, b = (ch // c.KB) % 2, ch % c.KB, kt % c.TPC
                return lds_load16(
                    rd_tile + (rt * 16 * c.RS_T + slot * c.SLOT_T + j * 16 + b * 8 + ku)
                ).bitcast(fx.BFloat16)

            # W addressing (gu 0 = gate, 1 = up; gate/up 16-row blocks interleaved), NI
            # 16-column tiles per wave; scale dword packs gate (byte 0/2) and up (1/3)
            nbase = nb * c.TILE_N + wave_n * c.NPW
            n0 = [(nbase + ni * 16) // 16 for ni in range_constexpr(c.NI)]
            nblk = [
                [e * (N_OUT // 16) + n0[ni] * 2 + gu for ni in range_constexpr(c.NI)]
                for gu in range_constexpr(2)
            ]
            mni = [e * (N_OUT // 32) + n0[ni] for ni in range_constexpr(c.NI)]
            # per column tile one vector address (lane*16 B + block base + the K-wave's
            # K start); the K position is (kt//4)*4096 in an SGPR + (kt%4)*1024 imm
            wvo = [
                [
                    lane * 4 + nblk[gu][ni] * (c.KT * 256) + wave_k * (c.KTW * 256)
                    for ni in range_constexpr(c.NI)
                ]
                for gu in range_constexpr(2)
            ]
            # scale dwords of 32-K groups klane//2 (b = 0) and 2 + klane//2 (b = 1)
            lane_sc = [(2 * b + q16 // 2) * 16 + l16 for b in range_constexpr(2)]
            svo = [
                [
                    lane_sc[b] + mni[ni] * SC_STRIDE_N0 + wave_k * (c.KTW // 4 * 64)
                    for ni in range_constexpr(c.NI)
                ]
                for b in range_constexpr(2)
            ]

            def load_b_tile(kt, prev):
                so = (kt // 4) * 4096
                bb = [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                wr,
                                wvo[gu][ni] + (kt % 4) * 256,
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=W_CACHE_MOD,
                                soffset_bytes=so,
                            )
                        )
                        for ni in range_constexpr(c.NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if const_expr(kt % 4 == 0):
                    g1 = kt // 4  # 256-K group: two scale dwords per lane
                    sc = [
                        [
                            fx.Int32(
                                buffer_ops.buffer_load(
                                    sr,
                                    svo[b][ni] + (g1 % 4) * 64,
                                    vec_width=1,
                                    dtype=fx.Int32,
                                    soffset_bytes=(g1 // 4) * 1024,
                                )
                            )
                            for ni in range_constexpr(c.NI)
                        ]
                        for b in range_constexpr(2)
                    ]
                else:
                    sc = prev[1]
                return bb, sc

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            acc_layout = fx.make_layout(4, 1)
            acc = [
                [
                    [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(c.NI)]
                    for _ in range(2)
                ]
                for _ in range(c.RT)
            ]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for rt in range_constexpr(c.RT):
                for gu in range_constexpr(2):
                    for ni in range_constexpr(c.NI):
                        acc[rt][gu][ni].store(zero4)

            def _frag(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def compute_tile(bb, sc, kt):
                # scale byte: 128-K tile (kt//2)%2 of the 256-K group -> +2, gate/up ->
                # +gu
                if const_expr(c.RT == 1):
                    # A fragments first, W unpacked right before each MFMA
                    a_t = [_frag(read_a(kt, ku, 0)) for ku in range_constexpr(c.TPC)]
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(c.NI):
                            s = _e8m0_byte_to_f32(
                                sc[kt % 2][ni], gu + ((kt // 2) % 2) * 2
                            )
                            for ku in range_constexpr(c.TPC):
                                b8 = _fp8x8_to_bf16(
                                    bb[gu][ni][2 * ku], bb[gu][ni][2 * ku + 1], s
                                )
                                fx.gemm(
                                    mma_atom,
                                    acc[0][gu][ni],
                                    a_t[ku],
                                    _frag(b8),
                                    acc[0][gu][ni],
                                )
                else:
                    # the unpacked W fragments feed every row tile: unpack once, then
                    # read each row tile's A fragments in turn (bounded live registers)
                    b_t = []
                    for gu in range_constexpr(2):
                        row = []
                        for ni in range_constexpr(c.NI):
                            s = _e8m0_byte_to_f32(
                                sc[kt % 2][ni], gu + ((kt // 2) % 2) * 2
                            )
                            row.append(
                                [
                                    _frag(
                                        _fp8x8_to_bf16(
                                            bb[gu][ni][2 * ku],
                                            bb[gu][ni][2 * ku + 1],
                                            s,
                                        )
                                    )
                                    for ku in range_constexpr(c.TPC)
                                ]
                            )
                        b_t.append(row)
                    for rt in range_constexpr(c.RT):
                        a_t = [
                            _frag(read_a(kt, ku, rt)) for ku in range_constexpr(c.TPC)
                        ]
                        for gu in range_constexpr(2):
                            for ni in range_constexpr(c.NI):
                                for ku in range_constexpr(c.TPC):
                                    fx.gemm(
                                        mma_atom,
                                        acc[rt][gu][ni],
                                        a_t[ku],
                                        b_t[gu][ni][ku],
                                        acc[rt][gu][ni],
                                    )

            # pipeline: batch 0 -> LDS, W ring; per tile: (batch loads) W load, LDS
            # read, MFMAs
            abuf = load_a_batch(0)
            ring = []  # type: list
            for t in range_constexpr(c.prefetch):
                ring.append(load_b_tile(t, ring[-1] if ring else None))
            stage_a_batch(abuf, 0)
            abuf = None
            for kt in range_constexpr(c.KTW):
                if const_expr(kt % c.BT == 0 and kt + c.BT < c.KTW):
                    abuf = load_a_batch(
                        kt // c.BT + 1
                    )  # before this iteration's W loads
                if const_expr(kt + c.prefetch < c.KTW):
                    ring.append(load_b_tile(kt + c.prefetch, ring[-1]))
                bb, sc = ring.pop(0)
                compute_tile(bb, sc, kt)
                if const_expr(kt % c.BT == c.BT - 1 and kt + 1 < c.KTW):
                    stage_a_batch(
                        abuf, (kt // c.BT + 1) % 2
                    )  # this batch's reads are done
                    abuf = None

            if const_expr(c.KW > 1):
                # K-reduce: K-wave k > 0 parks its partial sums in slot k - 1 of the
                # (now free) A region, K-wave 0 adds them and runs the epilogue alone
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                RED_SLOT = c.RT * c.NWN * 2 * c.NI * 64  # 16 B tiles parked per K-wave
                red = [
                    [
                        [
                            (((rt * c.NWN + wave_n) * 2 + gu) * c.NI + ni) * 64 + lane
                            for ni in range_constexpr(c.NI)
                        ]
                        for gu in range_constexpr(2)
                    ]
                    for rt in range_constexpr(c.RT)
                ]
                if wave_k > 0:
                    park = (wave_k - 1) * RED_SLOT
                    for rt in range_constexpr(c.RT):
                        for gu in range_constexpr(2):
                            for ni in range_constexpr(c.NI):
                                lds_store16(
                                    red[rt][gu][ni] + park,
                                    acc[rt][gu][ni].load().bitcast(fx.Int32),
                                )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                if wave_k == 0:
                    for rt in range_constexpr(c.RT):
                        for gu in range_constexpr(2):
                            for ni in range_constexpr(c.NI):
                                v = acc[rt][gu][ni].load()
                                for kw in range_constexpr(c.KW - 1):
                                    pv = lds_load16(
                                        red[rt][gu][ni] + kw * RED_SLOT
                                    ).bitcast(fx.Float32)
                                    v = fx.Vector.from_elements(
                                        [v[i] + pv[i] for i in range_constexpr(4)],
                                        fx.Float32,
                                    )
                                acc[rt][gu][ni].store(v)

            # epilogue: swigluoai(gate, up) -> bf16 [sorted_row, inter], padding masked
            def epilogue():
                for rt in range_constexpr(c.RT):
                    for ii in range_constexpr(4):
                        sorted_row = mbase + rt * 16 + q16 * 4 + ii
                        valid = ep_tok[rt][ii] < i32_ntok
                        for ni in range_constexpr(c.NI):
                            g = acc[rt][0][ni].load()[ii]
                            u = acc[rt][1][ni].load()[ii]
                            yb = _swigluoai_f32(g, u, f32_alpha, neg_limit).to(
                                fx.BFloat16
                            )
                            out_idx = sorted_row * INTER + nbase + ni * 16 + l16
                            buffer_ops.buffer_store(yb, outr, out_idx, mask=valid)

            if const_expr(c.KW > 1):
                if wave_k == 0:
                    epilogue()
            else:
                epilogue()

        if const_expr(inline_sort):
            # block = routing pair mb: expert + rows from a ballot over the pairs
            mb, nb = pid // narrow.NNB, pid % narrow.NNB
            mbase = mb * BM
            tab = smem.tab.ptr
            e_pair, owner, _nrows, build_tab = inline_sort_table(
                arg_mind, i32_ntok, TOPK, mb, lane, tab, max_pairs=max_pairs, bm=BM
            )
            if owner:
                build_tab()
            cumsum0 = i32_ntok * (TOPK * BM)
            # zero the stage-2 output (gemm2 accumulates with atomics): the NNB blocks
            # of pair 0 stride over it, one dword per thread
            if mb == 0:
                zb = _global_i32_ptr(arg_zero)
                for iv in range(
                    pid * (64 * NW) + tx, i32_zero_dw, narrow.NNB * 64 * NW
                ):
                    zb[fx.Int32(iv)] = fx.Int32(0)

            def tab_at(row):
                return fx.Int32(tab[row])

            if owner:
                ld_row, ld_tok, ep_tok = routing_rows(narrow, tab_at)
                body(narrow, mbase, nb, e_pair, ld_row, ld_tok, ep_tok, cumsum0)
        else:
            mind = _global_i32_ptr(arg_mind)
            eids = _global_i32_ptr(arg_eids)
            cumsum0 = fx.Int32(_global_i32_ptr(arg_cumsum)[0])

            def sorted_block(c, mb_id, mbase, nb, go):
                # the expert id and the routing rows (in bounds for every block of
                # the grid) are loaded together with the row count, not after its
                # branch: one round trip less before the first W load
                e = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(eids[mb_id])))

                def mind_at(row):
                    return fx.Int32(mind[mbase + row])

                ld_row, ld_tok, ep_tok = routing_rows(c, mind_at)
                if go:
                    body(c, mbase, nb, e, ld_row, ld_tok, ep_tok, cumsum0)

            if const_expr(wide):
                # the shared expert's ceil(ntok/WIDE_BM) wide blocks come first
                n_wide = (i32_ntok + (WIDE_BM - 1)) // WIDE_BM
                n_wide_pid = n_wide * widet.NNB
                # (distinct names per branch: the AST rewriter yields variables that
                # both branches of a dynamic if assign)
                if pid < n_wide_pid:
                    mbw = pid // widet.NNB
                    sorted_block(
                        widet, mbw, mbw * WIDE_BM, pid % widet.NNB, mbw < n_wide
                    )
                if pid >= n_wide_pid:
                    qn = pid - n_wide_pid
                    mbn = qn // narrow.NNB
                    mbasen = n_wide * WIDE_BM + mbn * BM
                    sorted_block(
                        narrow, n_wide + mbn, mbasen, qn % narrow.NNB, mbasen < cumsum0
                    )
            else:
                mb, nb = pid // narrow.NNB, pid % narrow.NNB
                mbase = mb * BM
                sorted_block(narrow, mb, mbase, nb, mbase < cumsum0)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        f32_alpha: fx.Float32,
        f32_limit: fx.Float32,
        arg_out: fx.Int64,
        arg_zero: fx.Int64,
        i32_zero_dw: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_x,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            f32_alpha,
            f32_limit,
            arg_out,
            arg_zero,
            i32_zero_dw,
        ).launch(grid=(grid_x, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.tile_n = TILE_N
    launch.wide_bm = WIDE_BM if wide else None
    launch.wide_n_blocks = widet.NNB if wide else 0
    return launch
