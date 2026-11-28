# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-MoE stage 1 on gfx942: gate and up of every pick, then SwiGLU, as
a grouped GEMM. A unit is one pick group (16 picks of one expert) and 16
intermediate columns, so it reads the expert's 32 weight rows once for 16
picks. A CTA has 8 waves, each a 640-wide K slice, and the partial sums
meet in LDS. Output `h` bf16 [P, I], as vLLM's AITER stage 1 stores it.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.gfx942 import (
    fp4x8_fnuz,
    mfma_fp8,
    pair_i64,
    pin_acc,
)
from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_NT,
    bf16_round,
    fresh,
    hw_exp2,
    hw_rcp,
    rsrc,
)
from vllm.models.deepseek_v41.amd.mono.stages.dims import MOE_INTER

from .shapes import EXPERTS, GROUP_WORDS, HIDDEN, LIMIT, TOPK

# The default stream of a launch: the current one (`mono/layer.py`).
_STREAM = fx.Stream(None)
THREADS = 512
# One CTA a CU of MI300X and MI325X.
BLOCKS = 304
WAVES = THREADS // 64
KB = HIDDEN // 32  # 160 K blocks of 32
KB_WAVE = KB // WAVES  # 20 a wave
# K blocks a load round of a wave: 4 (one 16-byte load a lane of the
# tile-major weights covers 4 blocks), and rounds in flight ahead of the one
# computing.
ROUND = 4
AHEAD = 2
LOG2E = 1.4426950408889634


def _w_round_loads(w, ws, e, n_rows, tile, kb0, lane):
    """One round of a wave's weight loads for one 16-row tile of expert e:
    ROUND K blocks from block kb0 (a multiple of 4). Byte order
    `fp4_tile_major`: a 64-byte K step of a tile's 16 rows is 1 KB, lane l's
    16 bytes at 16 l, which are its 4 blocks' dwords (row l % 16, K chunk
    l / 16). Codes `codes_tile_rounds` with 8-block rounds: 16 rows x 8
    bytes a round, lane group j's rows 4 j .. 4 j + 3 at 32 j. Returns the 4
    weight dwords and, for each of the lane's 4 rows, the round's 4 code
    bytes as one dword."""
    j = lane // 16
    kd = HIDDEN // 2 // 4  # dwords a weight row
    tile_row = e * n_rows + tile * 16
    base = fresh(tile_row * kd + 4 * lane) + kb0 * 64
    v = fx.Vector(
        bo.buffer_load(rsrc(w), base, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wd = [fx.Int32(v[b]) for b in range(4)]
    # Codes: 8-block rounds, so a 4-block round is the low or high half of
    # each row's 8 bytes. Row 4 j + q's 8 bytes start at byte
    # 8 (4 j + q) of the 128-byte round.
    rnd8 = kb0 // 8
    half = (kb0 // 4) % 2
    kc = KB  # code bytes a row
    sbase = fresh(tile_row * (kc // 4) + 8 * j) + rnd8 * 32
    # The lane's 4 rows are 32 bytes: two 16-byte loads (a buffer load takes
    # 16 bytes at most).
    sv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(ws), sbase + 4 * i, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
            )
        )
        for i in range(2)
    ]
    codes = [fx.Int32(sv[q // 2][2 * (q % 2) + half]) for q in range(4)]
    return wd, codes


def _x_loads(xb, xs, picks_lane, kb0, lane):
    """A wave's slice of one group's activations: K blocks kb0 .. kb0 +
    KB_WAVE of the token of the lane's pick (column lane % 16 of the MFMA's
    B), 8 bytes a block (K 8 (lane / 16) .. + 8 of the block, even K first
    within it, as `quant_rows` stores them) and the block's code byte."""
    j = lane // 16
    tok = picks_lane
    words = HIDDEN // 4
    bx = []
    for b in range_constexpr(KB_WAVE):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(xb), tok * words + (kb0 + b) * 8 + 2 * j, vec_width=2, dtype=T.i32
            )
        )
        bx.append(pair_i64(v[0], v[1]))
    cw = []
    for q in range_constexpr(KB_WAVE // 4):
        cw.append(
            fx.Int32(
                bo.buffer_load(
                    rsrc(xs), tok * (KB // 4) + (kb0 // 4) + q, vec_width=1, dtype=T.i32
                )
            )
        )
    return bx, cw


def _byte(words, i):
    return (words[i // 4] >> (8 * (i % 4))) & 0xFF


# One launcher a shape for the process: the 40 layers share it, and it stays
# alive as long as the CUDA graphs that launch it (FlyDSL unloads a kernel's
# code object when its launcher is collected).
@functools.cache
def build_moe_ug(tp: int, experts: int = EXPERTS, inter: int | None = None):
    """The stage 1 launch for one TP rank: intermediate width `inter` (MOE_INTER /
    tp by default), `experts` experts. Launch with one CTA a unit stride:
    `BLOCKS` CTAs, each running units bid, bid + BLOCKS, ..."""
    inter = inter if inter is not None else MOE_INTER // tp
    assert inter % 16 == 0
    n_rows = 2 * inter  # rows of an expert's w13 copy
    col_blocks = inter // 16  # units a group
    name = kernel_symbol("dsv41_ls_moe_ug", tp=tp, e=experts)

    @fx.struct
    class _Lds:
        # each wave's partial sums of the unit's two tiles: [wave][tile][lane][4]
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        w13: Int64,
        w13_s: Int64,
        xb: Int64,
        xs: Int64,
        groups: Int64,
        count: Int64,
        h: Int64,
    ):
        lds = fx.SharedAllocator().allocate(_Lds).peek()
        red = lds.red.ptr
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        kb0 = wave * KB_WAVE
        # The group count from `moe_sort` on the device, so the launch needs
        # no host value and runs inside a CUDA graph.
        n_groups = fx.Int32(bo.buffer_load(rsrc(count), 0, vec_width=1, dtype=T.i32))
        n_units = n_groups * col_blocks
        # A contiguous range of units a CTA, so a group's activations are
        # loaded once and reused for its column blocks.
        per = (n_units + BLOCKS - 1) // BLOCKS
        u0 = bid * per
        u1 = fx.min(u0 + per, n_units)
        g0 = u0 // col_blocks
        g1 = (u1 + col_blocks - 1) // col_blocks
        for g in range(g0, fx.min(g1, n_groups), 1):
            e = fx.Int32(
                bo.buffer_load(rsrc(groups), g * GROUP_WORDS, vec_width=1, dtype=T.i32)
            )
            pick = fx.Int32(
                bo.buffer_load(
                    rsrc(groups),
                    g * GROUP_WORDS + 4 + lane % 16,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            # A missing pick (-1, the group's padding) reads token 0 and is
            # not stored.
            tok = fx.max(pick, fx.Int32(0)) // TOPK
            bx, cw = _x_loads(xb, xs, tok, kb0, lane)
            cb0 = fx.max(u0 - g * col_blocks, fx.Int32(0))
            cb1 = fx.min(u1 - g * col_blocks, fx.Int32(col_blocks))
            for cb in range(cb0, cb1, 1):
                # The unit's two tiles: tile 2 cb holds columns 16 cb .. + 7 (gate
                # and up interleaved a row), tile 2 cb + 1 columns 16 cb + 8 .. + 15.
                accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
                # Weight loads run AHEAD rounds before their MFMAs.
                rounds = KB_WAVE // ROUND

                def round_ops(r, e=e, cb=cb):
                    return [
                        _w_round_loads(
                            w13, w13_s, e, n_rows, 2 * cb + t, kb0 + ROUND * r, lane
                        )
                        for t in range(2)
                    ]

                pending = [round_ops(r) for r in range(min(AHEAD, rounds))]
                for r in range_constexpr(rounds):
                    if r + AHEAD < rounds:
                        pending.append(round_ops(r + AHEAD))
                    ops = pending[r]
                    for b in range_constexpr(ROUND):
                        kb = ROUND * r + b
                        sx = _byte(cw, kb)
                        for t in range_constexpr(2):
                            wd, codes = ops[t]
                            even, odd = fp4x8_fnuz(wd[b])
                            d = mfma_fp8(
                                pair_i64(even, odd),
                                bx[kb],
                                fx.Vector.filled(4, 0.0, fx.Float32),
                            )
                            # 2^(code_w + 7 + code_x - 254) as f32 bits.
                            vals = []
                            for q in range_constexpr(4):
                                cq = (codes[q] >> (8 * b)) & 0xFF
                                f = ((cq + sx - 120) << 23).bitcast(fx.Float32)
                                vals.append(fx.Float32(fmath.fma(d[q], f, accs[t][q])))
                            accs[t] = fx.Vector.from_elements(vals, fx.Float32)
                    accs = [pin_acc(a) for a in accs]
                    rocdl.sched_barrier(0)
                for t in range_constexpr(2):
                    fx.ptr_store(accs[t], red + ((wave * 2 + t) * 64 + lane) * 4)
                gpu.barrier()
                # Epilogue: thread i < 256 owns pick i % 16 of the group and column
                # 16 cb + i / 16. The gate of column c sits in tile row 2 (c % 8)
                # and its up in the next row, of tile c / 8.
                if tid < 256:
                    pk = tid % 16
                    col = tid // 16
                    tile = col // 8
                    rg = 2 * (col % 8)
                    gu = []
                    for row in range_constexpr(2):
                        rr = rg + row
                        # A 16x16 MFMA result: lane 16 (rr / 4) + pk holds rows
                        # 4 (lane / 16) .. + 3 of column pk.
                        at = (16 * (rr // 4) + pk) * 4 + rr % 4
                        acc = fx.Float32(0.0)
                        for wv in range_constexpr(WAVES):
                            acc = acc + fx.ptr_load(red + (wv * 2 + tile) * 256 + at)
                        gu.append(acc)
                    gc = fx.min(gu[0], fx.Float32(LIMIT))
                    uc = fx.max(fx.min(gu[1], fx.Float32(LIMIT)), fx.Float32(-LIMIT))
                    sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
                    v = bf16_round(gc * sig * uc)
                    p = fx.Int32(
                        bo.buffer_load(
                            rsrc(groups),
                            g * GROUP_WORDS + 4 + pk,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    if p >= 0:
                        bo.buffer_store(
                            v.to(fx.BFloat16), rsrc(h), p * inter + 16 * cb + col
                        )
                gpu.barrier()

    @flyc.jit
    def launch(
        w13: Int64,
        w13_s: Int64,
        xb: Int64,
        xs: Int64,
        groups: Int64,
        count: Int64,
        h: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(w13, w13_s, xb, xs, groups, count, h).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
