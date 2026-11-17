# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""FlyDSL sliding-window GQA decode/verify attention, gfx950.

Serves the Kimi-K3.1 DSpark draft layers (head_dim 64, GQA group 6, window
2048, causal, a few query rows per request) over the paged cache that
TRITON_ATTN writes: logical [num_blocks, num_kv_heads, block_size, 2 * 64]
with K in content[0:64] and V in content[64:128], at any physical strides
(the DSA indexer pins the worker to the block-outermost BLHNC layout, so a
layer's block stride spans every layer's page). bf16 or fp8 e4m3 (per-tensor
scales) cache.

Workgroup = (request, kv head, split): one wave that owns ``TPC`` 16-token
tiles of the request's window and every query (row, head) pair of the kv
head. The window of the request's rows spans at most ``window + rows - 1``
keys, cut into consecutive tile ranges, one per split; splits past the end
exit at once, so a short context or a CUDA-graph capture (seq_len 1) runs
one or two splits. Unused splits write nothing: the combine kernel derives
each request's split count from the same metadata.

  * GQA packing: the (row, head) pairs of the kv head, pair n = row * 6 + h,
    are the N columns of 16-wide MFMA tiles (``NT`` tiles: 7 or 8 rows -> 3,
    16 rows -> 6), so a K/V tile is read once for all of them.
  * QK: mfma_f32_16x16x32_bf16 with A = K (16 tokens), B = Q^T. K and Q are
    read straight from global into the operand registers; lane group g holds
    dims [16g, 16g + 16) of both, which the K-dim sum does not care about.
    fp8 K is widened to bf16 exactly (e4m3 fits bf16); its scale folds into
    the softmax scale.
  * Softmax: every tile of the workgroup is scored before any exp, so the
    column max is taken once (no online rescale): P = exp2(s - m).
  * PV: mfma_f32_16x16x16_bf16 with A = V^T read from LDS by the
    ds_read_tr16_b64 transpose (V tiles stored row-major), B = P (the QK
    accumulator layout is the PV B layout).
  * Partials: O (unnormalised, fp32), m and l per (pair, split); the combine
    kernel (Triton, in flydsl_swa_gqa) weighs splits, applies the V scale and
    writes the rows of each request.

Block-table entries are read per tile and made uniform (readfirstlane); the
tile's block base folds into the 64-bit buffer base, so the cache can span
more than 4 GiB. Tiles past the request's last key re-read its last tile and
are fully masked.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.v1.attention.ops.flydsl_kernels.buffer_utils import (
    buffer_view,
    load_vec,
    store_vec,
)
from vllm.v1.attention.ops.flydsl_kernels.mma_utils import mma

HEAD_DIM = 64
TILE = 16
WARP = 64
NEG_INF = float("-inf")


def _rfl(v):
    """Make a wave-uniform i32 scalar (an SGPR operand for buffer bases)."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))


def _fp8x4_to_f32(word):
    w = fx.Int32(word)
    a = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), w, False))
    b = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), w, True))
    return [a[0], a[1], b[0], b[1]]


def _fp8x16_to_bf16(words):
    """16 e4m3 values (4 i32) -> two 8-element bf16 vectors (exact)."""
    vals = []
    for i in range(4):
        vals += _fp8x4_to_f32(words[i])
    v = fx.Vector.from_elements(vals, dtype=fx.Float32).to(fx.BFloat16)
    lo = v.shuffle(v, list(range(8)))
    hi = v.shuffle(v, list(range(8, 16)))
    return lo, hi


def num_splits(window: int, max_rows: int, tiles_per_cta: int) -> int:
    """Splits that cover the keys of ``max_rows`` rows at any alignment."""
    max_tiles = (window - 1 + max_rows + TILE - 1) // TILE + 1
    return (max_tiles + tiles_per_cta - 1) // tiles_per_cta


def build_swa_gqa_module(
    *,
    qg: int,
    num_kv_heads: int,
    n_qtiles: int,
    tiles_per_cta: int,
    window: int,
    kv_fp8: bool,
):
    QG = int(qg)
    KVH = int(num_kv_heads)
    NT = int(n_qtiles)
    TPC = int(tiles_per_cta)
    W = int(window)
    FP8 = bool(kv_fp8)
    NP = NT * 16
    D = HEAD_DIM
    ESZ = 1 if FP8 else 2
    TILE_LDS_I64 = TILE * D * 2 // 8

    @fx.struct
    class SharedStorage:
        v: fx.Array[fx.Int64, TPC * TILE_LDS_I64, 16]

    name = f"swa_gqa_qg{QG}_nt{NT}_tpc{TPC}_w{W}_{'fp8' if FP8 else 'bf16'}"

    @flyc.kernel(name=name, known_block_size=[WARP, 1, 1])
    def swa_gqa_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        BT: fx.Tensor,
        SL: fx.Tensor,
        QSL: fx.Tensor,
        KSC: fx.Tensor,
        PO: fx.Tensor,
        PML: fx.Tensor,
        stride_qt: fx.Int32,
        stride_qh: fx.Int32,
        stride_blk: fx.Int32,
        stride_head: fx.Int32,
        stride_slot: fx.Int32,
        v_off: fx.Int32,
        stride_bt: fx.Int32,
        block_size: fx.Int32,
        c_log2: fx.Float32,
        n_split: fx.Int32,
    ):
        lane = gpu.thread_idx.x
        r = gpu.block_idx.x
        kvh = gpu.block_idx.y
        sp = gpu.block_idx.z
        col = lane & 15
        g = lane >> 4

        qsl_buf = buffer_view(QSL, fx.Int32)
        sl_buf = buffer_view(SL, fx.Int32)
        q_start = _rfl(qsl_buf[r])
        q_len = _rfl(qsl_buf[r + 1]) - q_start
        seq_len = _rfl(sl_buf[r])
        p0 = seq_len - q_len
        kfirst = p0 - (W - 1)
        kfirst = (kfirst > 0).select(kfirst, 0)
        t_hi = (seq_len - 1) >> 4
        t0 = (kfirst >> 4) + sp * TPC
        # q_len > 0 and seq_len > 0 and t0 <= t_hi, as one wave-uniform test.
        lim = (q_len > 0).select(t_hi, -1)
        lim = (seq_len > 0).select(lim, -1)

        if t0 <= lim:
            q_buf = buffer_view(Q, fx.Int32, 4)
            bt_buf = buffer_view(BT, fx.Int32)
            qk_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            pv_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            v_lds_i64 = lds.v.ptr
            v_lds_u8 = fx.recast_iter(fx.Uint8, v_lds_i64)
            neg_inf = fx.Float32(NEG_INF)
            zero_f = fx.Float32(0.0)
            zero_v4 = fx.Vector.filled(4, 0.0, fx.Float32)
            c = c_log2
            if const_expr(FP8):
                c = c * fx.Float32(buffer_view(KSC, fx.Float32)[0])

            # ---- K / V tiles: issue every load up front ------------------
            k_ops = []
            tok_base = []
            v_tok = lane >> 2
            v_part = lane & 3
            for i in range_constexpr(TPC):
                ti = t0 + i
                tok_base.append(ti * TILE)
                ti_c = (ti <= t_hi).select(ti, t_hi)
                tb = ti_c * TILE
                lb = tb // block_size
                slot0 = tb - lb * block_size
                phys = _rfl(bt_buf[r * stride_bt + lb])
                base = fx.Int64(fx.Uint32(phys)) * fx.Int64(stride_blk) + fx.Int64(
                    kvh * stride_head
                )
                blk = buffer_view(KV, fx.Int32, 4, base)
                k_off = (slot0 + col) * stride_slot + g * (16 * ESZ)
                v_off_b = (slot0 + v_tok) * stride_slot + v_off + v_part * (16 * ESZ)
                if const_expr(FP8):
                    kw = load_vec(blk, k_off // 4, 4)
                    vw = load_vec(blk, v_off_b // 4, 4)
                    ka, kb = _fp8x16_to_bf16([kw[j] for j in range(4)])
                    va, vb = _fp8x16_to_bf16([vw[j] for j in range(4)])
                    va = va.bitcast(fx.Int64)
                    vb = vb.bitcast(fx.Int64)
                else:
                    ka = load_vec(blk, k_off // 4, 4).bitcast(fx.BFloat16)
                    kb = load_vec(blk, k_off // 4 + 4, 4).bitcast(fx.BFloat16)
                    va = load_vec(blk, v_off_b // 4, 4).bitcast(fx.Int64)
                    vb = load_vec(blk, v_off_b // 4 + 4, 4).bitcast(fx.Int64)
                k_ops.append((ka, kb))
                # V row-major [tile][token][64] bf16.
                v_idx = i * TILE_LDS_I64 + v_tok * (D // 4) + v_part * 4
                (v_lds_i64 + v_idx).store(va)
                (v_lds_i64 + (v_idx + 2)).store(vb)

            # ---- Q operands and per-column positions ---------------------
            q_ops = []
            pos = []
            for nt in range_constexpr(NT):
                n = nt * 16 + col
                row = n // QG
                hh = n - row * QG
                row_c = (row < q_len).select(row, 0)
                q_off = (q_start + row_c) * stride_qt + (kvh * QG + hh) * stride_qh
                q_off = q_off + g * 16
                qa = load_vec(q_buf, q_off // 2, 4).bitcast(fx.BFloat16)
                qb = load_vec(q_buf, q_off // 2 + 4, 4).bitcast(fx.BFloat16)
                q_ops.append((qa, qb))
                pos.append(p0 + row)

            # ---- QK + mask, then the column max over all tiles ------------
            s_all = []
            mx = [neg_inf for _ in range(NT)]
            for i in range_constexpr(TPC):
                s_i = []
                for nt in range_constexpr(NT):
                    acc = mma(qk_atom, zero_v4, k_ops[i][0], q_ops[nt][0])
                    acc = mma(qk_atom, acc, k_ops[i][1], q_ops[nt][1])
                    acc = acc * c
                    els = []
                    for e in range_constexpr(4):
                        tok = tok_base[i] + g * 4 + e
                        ok = (tok <= pos[nt]) & (tok > pos[nt] - W)
                        els.append(ok.select(acc[e], neg_inf))
                    v = fx.Vector.from_elements(els, dtype=fx.Float32)
                    s_i.append(v)
                    mx[nt] = fx.max(mx[nt], v.reduce(fx.ReductionOp.MAX))
                s_all.append(s_i)
            m_safe = []
            for nt in range_constexpr(NT):
                m = mx[nt]
                m = fx.max(m, m.shuffle_xor(16, WARP))
                m = fx.max(m, m.shuffle_xor(32, WARP))
                mx[nt] = m
                m_safe.append((m > neg_inf).select(m, zero_f))

            # V tiles were written by other lanes: drain the LDS writes.
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

            # ---- P = exp2(s - m), PV --------------------------------------
            acc_o = [[zero_v4 for _ in range(4)] for _ in range(NT)]
            lsum = [zero_f for _ in range(NT)]
            v_lane_byte = v_tok * (D * 2) + v_part * 8
            for i in range_constexpr(TPC):
                v_ops = []
                for h in range_constexpr(4):
                    ptr = fx.to_llvm_ptr(
                        v_lds_u8 + (v_lane_byte + i * (TILE * D * 2) + h * 32)
                    )
                    raw = rocdl.ds_read_tr16_b64(T.vec(4, T.i16), ptr).result
                    v_ops.append(fx.Vector(raw).bitcast(fx.BFloat16))
                for nt in range_constexpr(NT):
                    ps = []
                    for e in range_constexpr(4):
                        d = s_all[i][nt][e] - m_safe[nt]
                        ps.append(d.exp2(fastmath=arith.FastMathFlags.fast))
                    p = fx.Vector.from_elements(ps, dtype=fx.Float32)
                    lsum[nt] = lsum[nt] + p.reduce(fx.ReductionOp.ADD)
                    p_op = p.to(fx.BFloat16)
                    for h in range_constexpr(4):
                        acc_o[nt][h] = mma(pv_atom, acc_o[nt][h], v_ops[h], p_op)

            # ---- partials --------------------------------------------------
            po_buf = buffer_view(PO, fx.Float32, 4)
            pml_buf = buffer_view(PML, fx.Float32)
            cta = (r * KVH + kvh) * n_split + sp
            for nt in range_constexpr(NT):
                n = nt * 16 + col
                for h in range_constexpr(4):
                    off = (cta * NP + n) * D + h * 16 + g * 4
                    store_vec(po_buf, off, acc_o[nt][h], 4)
                l = lsum[nt]
                l = l + l.shuffle_xor(16, WARP)
                l = l + l.shuffle_xor(32, WARP)
                lsum[nt] = l
            if g == 0:
                for nt in range_constexpr(NT):
                    n = nt * 16 + col
                    pml_buf[(cta * 2) * NP + n] = mx[nt]
                    pml_buf[(cta * 2 + 1) * NP + n] = lsum[nt]

    @flyc.jit
    def launch_swa_gqa(
        Q,
        KV,
        BT,
        SL,
        QSL,
        KSC,
        PO,
        PML,
        stride_qt,
        stride_qh,
        stride_blk,
        stride_head,
        stride_slot,
        v_off,
        stride_bt,
        block_size,
        c_log2,
        n_split,
        n_req,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        swa_gqa_kernel(
            Q,
            KV,
            BT,
            SL,
            QSL,
            KSC,
            PO,
            PML,
            stride_qt,
            stride_qh,
            stride_blk,
            stride_head,
            stride_slot,
            v_off,
            stride_bt,
            block_size,
            c_log2,
            n_split,
        ).launch(
            grid=(fx.Int64(n_req), KVH, fx.Int64(n_split)),
            block=(WARP, 1, 1),
            stream=stream,
        )

    return launch_swa_gqa
