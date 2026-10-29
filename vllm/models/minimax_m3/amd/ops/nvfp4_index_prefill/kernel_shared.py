# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 prefill index scorer for small work (FlyDSL, gfx950).

The scores of ``kernel.py`` with one dequantized block per workgroup: a
workgroup is 4 waves x 32 MQ query rows x a segment of the causal window. Per
block, wave w dequantizes token tile w into e4m3 A fragments in LDS (double
buffered, one barrier per block) while the MFMAs of the previous block run;
every wave multiplies all four token tiles against its query rows with
v_mfma_f32_32x32x64_f8f6f4. Blocks before the wave's first query position skip
the causal mask in a loop of their own.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    as_f32,
    bits,
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
    maxf_nn,
    permlane_swap,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.kernel import (
    BLK,
    PAGE_BYTES,
    PAGE_CODES,
    D,
    page_lookup,
    score_resources,
    segment_blocks,
    tile_rows,
    with_sentinels,
)

NW = 4
MQ_CHOICES = (1, 2)  # 32-row query tiles per wave
TILES = BLK // 32  # 32-token tiles per block
KS = 2  # MFMA k-steps per key row
FL = 32  # fragment bytes per lane
FRAG_BYTES = TILES * KS * 64 * FL  # one block's A fragments
RING = 4  # fragment reads in flight ahead of the MFMA


def _sat448_fp8x2(x):
    """Clamp 2 x f16 to +-448; the hardware fp8 convert does not saturate."""
    fm = _mlir_ir.Attribute.parse("#arith.fastmath<nnan>")
    hi = fx.as_ir_value(fx.Vector.filled(2, 448.0, fx.Float16))
    lo = fx.as_ir_value(fx.Vector.filled(2, -448.0, fx.Float16))
    v = _arith.MaxNumFOp(fx.as_ir_value(x), lo, fastmath=fm).result
    return _arith.MinNumFOp(v, hi, fastmath=fm).result


def _fp8_pair(w, sv):
    """Code dword w x splat scale sv -> 2 dwords of e4m3."""
    out = []
    for h in range_constexpr(2):
        p = fx.as_ir_value(fx.Vector.filled(2, 0, fx.Int16))
        for k in range_constexpr(2):
            x = _sat448_fp8x2(cvt_fp4x2_f16(w, 2 * h + k) * sv)
            p = fx.rocdl.cvt_scalef32_pk_fp8_f16(
                Vec.make_type(2, fx.Int16),
                p,
                x,
                fx.as_ir_value(fx.Float32(1.0)),
                k == 1,
            )
        out.append(fx.Int32(Vec(p).bitcast(fx.Int32)[0]))
    return out


def _mfma_fp8_32x32x64(a8, b8, acc16):
    """v_mfma_f32_32x32x64_f8f6f4 on e4m3; lane (r, h) gets
    D[8 (i // 4) + 4 h + i % 4][r]."""
    return Vec(
        fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, fx.Float32), [a8, b8, acc16, 0, 0, 0, 0, 0, 0]
        )
    )


def _xlane_max32(x):
    """max(x[L], x[L ^ 32])."""
    a, b = permlane_swap(_rocdl.permlane32_swap, bits(x), bits(x))
    return maxf_nn(as_f32(a), as_f32(b))


def tile_q(mq: int) -> int:
    """Query rows per workgroup."""
    return NW * 32 * mq


def compile_score_prefill_shared(heads: int = 1, mq: int = 2):
    """Build the scorer for 1, 2 or 4 index heads and ``mq`` 32-row query tiles
    per wave."""
    assert heads in (1, 2, 4) and mq in MQ_CHOICES, (heads, mq)
    H, MQ = heads, mq
    ROWS_PER_WAVE = 32 * MQ
    TILE_Q = tile_q(MQ)
    name = f"m3_nvfp4_index_score_prefill_fp8_shared_tq{TILE_Q}_h{H}"

    @fx.struct
    class Shared:
        frag: fx.Array[fx.Uint8, 2 * FRAG_BYTES, 16]  # A fragments of 2 blocks

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [total_q, H, 128] e4m3
        arg_kv: fx.Int64,  # [num_blocks, 128, 72] uint8
        arg_score: fx.Int64,  # [H, total_q, S] fp32
        arg_bt: fx.Int64,  # [batch, >= max_blocks] int32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] int32
        arg_seq: fx.Int64,  # seq_lens [batch] int32
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,  # score head stride
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        r32, h = lane % 32, lane // 32
        live, b, seg, seq_start, seq_len, prefix, q_rows, row0 = tile_rows(
            pid, i32_qt, i32_nseg, i32_nb, arg_cu, arg_seq, H, TILE_Q
        )
        if live:
            nblk, blk0, blk1 = segment_blocks(
                prefix, row0, seq_len, seg, i32_nseg, H, TILE_Q, 1
            )
            if blk0 < blk1:
                qr, sr = score_resources(
                    arg_q, arg_score, i32_total_q, i32_hs, i32_S, H
                )
                page_of = page_lookup(arg_bt, b * i32_bt_stride, blk1)

                tok = wave * 32 + r32

                def load_raw(blk):
                    """Token tok's codes and scales of dims 64 h .. +63; zeros past
                    the segment."""
                    ok = blk < blk1
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page_of(blk)) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    w = []
                    for i in range_constexpr(2):
                        v = Vec(
                            buffer_ops.buffer_load(
                                kr,
                                (tok * 64 + h * 32 + i * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=ok,
                            )
                        )
                        w += [fx.Int32(v[e]) for e in range_constexpr(4)]
                    sc = fx.Int32(
                        buffer_ops.buffer_load(
                            kr,
                            (PAGE_CODES + tok * 8 + h * 4) // 4,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=ok,
                        )
                    )
                    return w + [sc]

                frag128 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.frag.ptr),
                        fx.make_layout(2 * FRAG_BYTES // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

                def frag_slot(buf, n, s, half=0):
                    return (
                        buf * (FRAG_BYTES // 16)
                        + ((n * KS + s) * 64 + lane) * (FL // 16)
                        + half
                    )

                def store16(words, slot):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(Vec.from_elements(words, fx.Int32))
                    fx.copy(atom16, r, fx.slice(frag128, (None, slot)))

                def load16(slot):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom16, fx.slice(frag128, (None, slot)), r)
                    return [fx.Int32(x) for x in Vec(r.load())]

                def dequant_step(raw, buf, s):
                    """K-step s (dims 64 h + 32 s .. +31) of this wave's tile."""
                    scl = cvt_fp8x2_f16(raw[8], s == 1)
                    sv = [
                        Vec.from_elements([scl[g], scl[g]], fx.Float16) for g in (0, 1)
                    ]
                    k = []
                    for e in range_constexpr(4):
                        k += _fp8_pair(raw[4 * s + e], sv[e // 2])
                    store16(k[:4], frag_slot(buf, wave, s, 0))
                    store16(k[4:], frag_slot(buf, wave, s, 1))

                def read_frag(buf, n, s):
                    return Vec.from_elements(
                        load16(frag_slot(buf, n, s, 0))
                        + load16(frag_slot(buf, n, s, 1)),
                        fx.Int32,
                    )

                # Rows past q_len * H are zero and never stored.
                rows = [row0 + wave * ROWS_PER_WAVE + 32 * m + r32 for m in range(MQ)]
                qpos = [prefix + rw // H for rw in rows]
                qf = []
                for m in range_constexpr(MQ):
                    qrow = (seq_start * H + rows[m]) * D
                    d = []
                    for i in range_constexpr(4):
                        v = Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + h * 64 + i * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=rows[m] < q_rows,
                            )
                        )
                        d += [fx.Int32(v[e]) for e in range_constexpr(4)]
                    qf.append(
                        [
                            Vec.from_elements(d[8 * s : 8 * s + 8], fx.Int32)
                            for s in range(KS)
                        ]
                    )

                zero16 = fx.Vector.filled(16, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                st_tok = [rw // H for rw in rows]
                st_out = [
                    (rw % H) * i32_hs + (seq_start + t) * i32_S
                    for rw, t in zip(rows, st_tok)
                ]
                st_nb = [qp // BLK + 1 for qp in qpos]
                qpos_min = prefix + (row0 + wave * ROWS_PER_WAVE) // H

                def max16(acc):
                    v = acc[0]
                    for i in range_constexpr(1, 16):
                        v = maxf_nn(v, acc[i])
                    return v

                def max16_masked(acc, lim):
                    v = neg_inf
                    for i in range_constexpr(16):
                        x = (lim >= 8 * (i // 4) + i % 4).select(acc[i], neg_inf)
                        v = maxf_nn(v, x)
                    return v

                def block(blk, buf, raw_next, masked):
                    """Scores of block blk for this wave's rows; dequantizes
                    raw_next into the other buffer meanwhile."""
                    runs = [neg_inf] * MQ
                    jobs = [(n, s) for n in range(TILES) for s in range(KS)]
                    ring = [None] * len(jobs)
                    for j in range_constexpr(RING - 1):
                        ring[j] = read_frag(buf, *jobs[j])
                    for n in range_constexpr(TILES):
                        accs = [zero16] * MQ
                        for s in range_constexpr(KS):
                            j = n * KS + s
                            if const_expr(j + RING - 1 < len(jobs)):
                                ring[j + RING - 1] = read_frag(buf, *jobs[j + RING - 1])
                            if const_expr(j % TILES == 0):
                                dequant_step(raw_next, 1 - buf, j // TILES)
                            for m in range_constexpr(MQ):
                                accs[m] = _mfma_fp8_32x32x64(ring[j], qf[m][s], accs[m])
                        base = blk * BLK + 32 * n + 4 * h
                        for m in range_constexpr(MQ):
                            if const_expr(masked):
                                mx = max16_masked(accs[m], qpos[m] - base)
                            else:
                                mx = max16(accs[m])
                            runs[m] = maxf_nn(runs[m], mx)
                    for m in range_constexpr(MQ):
                        v = _xlane_max32(runs[m])
                        v = with_sentinels(v, blk, st_nb[m], i32_init, i32_local)
                        buffer_ops.buffer_store(
                            v,
                            sr,
                            st_out[m] + blk,
                            mask=(rows[m] < q_rows) & (h == 0),
                        )

                raw0 = load_raw(blk0)
                for s in range_constexpr(KS):
                    dequant_step(raw0, 0, s)
                raw = load_raw(blk0 + 1)
                fx.gpu.barrier()

                # Blocks below dsplit need no causal mask. Two loops rather than a
                # branch per block keep the wave within 256 registers; waves split
                # at different blocks but pass the same barriers.
                dfirst = (qpos_min + 1) // BLK
                dsplit = (dfirst < blk0).select(blk0, dfirst)
                dsplit = (dsplit > blk1).select(blk1, dsplit)
                blk = blk0
                buf = fx.Int32(0)
                while blk < dsplit:
                    nxt = load_raw(blk + 2)
                    block(blk, buf, raw, False)
                    raw = nxt
                    fx.gpu.barrier()
                    blk = blk + 1
                    buf = 1 - buf
                while blk < blk1:
                    nxt = load_raw(blk + 2)
                    block(blk, buf, raw, True)
                    raw = nxt
                    fx.gpu.barrier()
                    blk = blk + 1
                    buf = 1 - buf

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            i64_page_stride,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i32_init,
            i32_local,
            i32_hs,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
