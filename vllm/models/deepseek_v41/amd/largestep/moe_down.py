# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-MoE stage 2 on gfx942: each pick's down projection times its
routing weight, as a grouped GEMM. A unit is one pick group and 256 output
rows. Each of the 8 waves takes 2 tiles of 16 rows over the whole K, so no
partial sums cross waves. Output `y` bf16 [P, 5120], combined per token by
`moe_small.build_combine`.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.gfx942 import (
    fp4x8_fnuz,
    mfma_fp8,
    pair_i64,
    pin_acc,
)
from vllm.models.deepseek_v41.amd.mono.common.ops import CM_NT, bf16_pair, fresh, rsrc
from vllm.models.deepseek_v41.amd.mono.stages.dims import MOE_INTER

from .shapes import EXPERTS, GROUP_WORDS, HIDDEN

# The default stream of a launch: the current one (`mono/layer.py`).
_STREAM = fx.Stream(None)
THREADS = 512
# One CTA a CU of MI300X and MI325X.
BLOCKS = 304
WAVES = THREADS // 64
CHUNK_TILES = 16  # 16-row tiles a unit: 256 output rows
CHUNKS = HIDDEN // (16 * CHUNK_TILES)  # 20
TILES_WAVE = CHUNK_TILES // WAVES  # 2


def _w2_tile_loads(w2, w2_s, e, tile, inter, lane):
    """The lane's w2 operands of rows 16 tile .. + 15 of expert e over the
    whole K (inter): a dword of 8 e2m1 a K block (row lane % 16, K chunk
    lane / 16), in `fp4_tile_major` order: 64-byte steps of 4 blocks, 1 KB a
    step for the 16 rows, and a last step of 2 blocks (TP4) or 1 block (TP8)
    with 8 or 4 bytes a lane. And the codes of rows 4 (lane / 16) .. + 3:
    4 rows x inter / 32 bytes, contiguous because the codes are row-major."""
    j = lane // 16
    kd, kc = inter // 8, inter // 32
    tile0 = (e * HIDDEN + tile * 16) * kd
    base = fresh(tile0 + 4 * lane)
    wd = []
    for m in range_constexpr(kc // 4):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(w2), base + 256 * m, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
            )
        )
        wd += [fx.Int32(v[i]) for i in range(4)]
    tail = kc % 4
    if tail == 2:
        v = fx.Vector(
            bo.buffer_load(
                rsrc(w2),
                fresh(tile0 + 2 * lane) + 256 * (kc // 4),
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        wd += [fx.Int32(v[i]) for i in range(2)]
    elif tail == 1:
        wd.append(
            fx.Int32(
                bo.buffer_load(
                    rsrc(w2),
                    fresh(tile0 + lane) + 256 * (kc // 4),
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
        )
    # 4 rows' codes, 4 kc bytes, kc dwords from row 4 j.
    sw0 = (e * HIDDEN + tile * 16 + 4 * j) * kc // 4
    sc = []
    for w0 in range_constexpr(0, kc, 4):
        n = min(4, kc - w0)
        v = bo.buffer_load(
            rsrc(w2_s), sw0 + w0, vec_width=n, dtype=T.i32, cache_modifier=CM_NT
        )
        if n == 1:
            sc.append(fx.Int32(v))
        else:
            v = fx.Vector(v)
            sc += [fx.Int32(v[q]) for q in range(n)]
    return wd, sc


def _h_loads(hq, hqs, pick, inter, lane):
    """The lane's B operand: the intermediate row of pick `pick` (column lane
    % 16 of the MFMA), 8 bytes a K block (K 8 (lane / 16) .. + 8 of the
    block, even K first as `quant_rows` stores them), and the row's codes,
    inter / 32 bytes."""
    j = lane // 16
    kc = inter // 32
    words = inter // 4
    bx = []
    for b in range_constexpr(kc):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(hq), pick * words + b * 8 + 2 * j, vec_width=2, dtype=T.i32
            )
        )
        bx.append(pair_i64(v[0], v[1]))
    # codes: kc bytes from byte pick * kc; kc is 9 or 18, so read the dwords
    # that cover them and pick bytes on the host side of the index math.
    cw = []
    first = pick * kc
    for b in range_constexpr(kc):
        at = first + b
        word = fx.Int32(bo.buffer_load(rsrc(hqs), at // 4, vec_width=1, dtype=T.i32))
        cw.append((word >> (8 * (at % 4))) & 0xFF)
    return bx, cw


# One launcher a shape for the process: the 40 layers share it, and it stays
# alive as long as the CUDA graphs that launch it (FlyDSL unloads a kernel's
# code object when its launcher is collected).
@functools.cache
def build_moe_down(tp: int, experts: int = EXPERTS, inter: int | None = None):
    """The stage 2 launch for one TP rank: `BLOCKS` CTAs, each running units
    bid, bid + BLOCKS, ... of n_groups x CHUNKS units."""
    inter = inter if inter is not None else MOE_INTER // tp
    kc = inter // 32
    assert inter % 32 == 0 and kc % 4 in (0, 1, 2)
    name = kernel_symbol("dsv41_ls_moe_down", tp=tp, e=experts)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        w2: Int64,
        w2_s: Int64,
        hq: Int64,
        hqs: Int64,
        wts: Int64,
        groups: Int64,
        count: Int64,
        y: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        # The group count from `moe_sort` on the device, so the launch needs
        # no host value and runs inside a CUDA graph.
        n_groups = fx.Int32(bo.buffer_load(rsrc(count), 0, vec_width=1, dtype=T.i32))
        n_units = n_groups * CHUNKS
        for unit in range(bid, n_units, BLOCKS):
            g = unit // CHUNKS
            ch = unit % CHUNKS
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
            # A missing pick (-1) reads row 0 and is not stored.
            pr = fx.max(pick, fx.Int32(0))
            bx, cw = _h_loads(hq, hqs, pr, inter, lane)
            wgt = fx.Float32(bo.buffer_load(rsrc(wts), pr, vec_width=1, dtype=T.f32))
            # Both of the wave's tiles' weights are loaded before either
            # computes, so the second tile's loads overlap the first's MFMAs.
            tiles = [
                ch * CHUNK_TILES + wave * TILES_WAVE + tw for tw in range(TILES_WAVE)
            ]
            wops = [_w2_tile_loads(w2, w2_s, e, tile, inter, lane) for tile in tiles]
            for tw in range_constexpr(TILES_WAVE):
                tile = tiles[tw]
                wd, sc = wops[tw]
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for b in range_constexpr(kc):
                    even, odd = fp4x8_fnuz(wd[b])
                    d = mfma_fp8(
                        pair_i64(even, odd), bx[b], fx.Vector.filled(4, 0.0, fx.Float32)
                    )
                    vals = []
                    for q in range_constexpr(4):
                        # row 4 (lane / 16) + q's code of block b: byte
                        # q * kc + b of the lane's 4 rows of codes
                        i = q * kc + b
                        cq = (sc[i // 4] >> (8 * (i % 4))) & 0xFF
                        # 2^(code_h + 7 + code_w - 254) as f32 bits, as in
                        # moe_ug: the f32 exponent field is code_h + code_w
                        # - 120, a normal f32 while code_h + code_w lies in
                        # 121..374. Outside it the bits wrap.
                        f = ((cq + cw[b] - 120) << 23).bitcast(fx.Float32)
                        vals.append(fx.Float32(fmath.fma(d[q], f, acc[q])))
                    acc = fx.Vector.from_elements(vals, fx.Float32)
                    if b % 6 == 5:
                        acc = pin_acc(acc)
                # D: lane l holds rows 4 (l / 16) .. + 3 of column (pick)
                # l % 16: the 4 rows' weighted values as bf16 in one 8-byte
                # store (two words), not four 2-byte stores.
                if pick >= 0:
                    row0 = tile * 16 + 4 * (lane // 16)
                    words = [
                        bf16_pair(acc[2 * h] * wgt, acc[2 * h + 1] * wgt).bitcast(
                            fx.Int32
                        )
                        for h in range(2)
                    ]
                    bo.buffer_store(
                        fx.Vector.from_elements(words, fx.Int32),
                        rsrc(y),
                        (pick * HIDDEN + row0) // 2,
                    )
        return

    @flyc.jit
    def launch(
        w2: Int64,
        w2_s: Int64,
        hq: Int64,
        hqs: Int64,
        wts: Int64,
        groups: Int64,
        count: Int64,
        y: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(w2, w2_s, hq, hqs, wts, groups, count, y).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
