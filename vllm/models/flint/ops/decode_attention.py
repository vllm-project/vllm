# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV paged decode attention for Flint (GQA 8, head_dim 128, bf16 KV).

Numerics mirror FlashInfer's trtllm-gen decode kernel: ``S = K q`` with bf16
inputs and fp32 accumulation; softmax in fp32 as ``exp2(S * c - m * c)`` with
``c = scale * log2(e)`` and the running max ``m`` in unscaled logits; the row
sum accumulates the fp32 probabilities while ``P`` is rounded to bf16 for
``P V`` (fp32 accumulation); the sink adds ``exp2(sink * log2(e) - m * c)`` to
the final denominator. Split partials (unnormalized O, m, l) stay fp32
(trtllm-gen stores partial O in bf16) and are merged either by the last CTA of
each (request, kv head), found with a self-resetting arrival counter, or, for
the short sliding-window layers, inside a thread-block cluster through
distributed shared memory (every rank merges a slice of the head dims).

Each warp streams 8 KB pages (16 tokens x [K | V], HND layout) through a
private ring of bulk-copy (TMA) stages and runs ``mma.sync`` on them. The
kernel is launched with programmatic dependent launch: it prefetches its
leading pages before waiting on the grid dependency (all but the page holding
the current token, whose K/V the previous kernel writes), touches q and its
workspace only after the wait, and triggers its dependents once K/V streaming
is done.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op
from quack.compile_utils import make_fake_tensor

from .cute_primitives import (
    bf16x2,
    fma,
    lds_f32,
    ptx,
    smem_addr,
    stg_bf16,
    stg_u32x4,
    sts_f32,
)

_PAGE = 16
_GQA = 8
_HEAD_DIM = 128
_PAGE_BYTES = _PAGE * 2 * _HEAD_DIM * 2
_ROW_BYTES = 2 * _HEAD_DIM * 2
# Per-warp CTA-combine buffer (fp32): O[8][128], m[8], l[8].
_WBUF = _GQA * _HEAD_DIM + 2 * _GQA
_LOG2E = 1.4426950408889634
_SMEM_PER_SM = 228 * 1024


@dsl_user_op
def _bulk_copy(dst, src, mbar, nbytes: int, *, loc=None, ip=None):
    ptx(
        "{ .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p, 1.0; "
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        f".L2::cache_hint [$0], [$1], {nbytes}, [$2], p; }}",
        "r,l,r,~{memory}",
        [Uint32(dst), Int64(src), Uint32(mbar)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds128(addr, *, loc=None, ip=None):
    return ptx(
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        [Uint32(addr)],
        (Uint32,) * 4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg128(addr, *, loc=None, ip=None):
    return ptx(
        "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        [Int64(addr)],
        (Uint32,) * 4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_cg_f32x4(addr, *, loc=None, ip=None):
    return ptx(
        "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,l,~{memory}",
        [Int64(addr)],
        (Float32,) * 4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_i32_pred(addr, pred, *, loc=None, ip=None):
    return ptx(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; "
        "@p ld.global.nc.b32 $0, [$1]; }",
        "=r,l,r",
        [Int64(addr), Int32(pred)],
        Int32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    ptx(
        "st.global.v4.f32 [$0], {$1,$2,$3,$4};",
        "l,f,f,f,f,~{memory}",
        [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32(addr, a, *, loc=None, ip=None):
    ptx(
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        [Int64(addr), Float32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u32x2(addr, a, b, *, loc=None, ip=None):
    ptx(
        "st.global.v2.b32 [$0], {$1,$2};",
        "l,r,r,~{memory}",
        [Int64(addr), Uint32(a), Uint32(b)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_i32(addr, a, *, loc=None, ip=None):
    ptx(
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        [Uint32(addr), Int32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_i32(addr, *, loc=None, ip=None):
    return ptx(
        "ld.shared.b32 $0, [$1];",
        "=r,r,~{memory}",
        [Uint32(addr)],
        Int32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_add_acq_rel(addr, val, *, loc=None, ip=None):
    return ptx(
        "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
        "=r,l,r,~{memory}",
        [Int64(addr), Int32(val)],
        Int32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_i32_relaxed(addr, val, *, loc=None, ip=None):
    ptx(
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r,~{memory}",
        [Int64(addr), Int32(val)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    return ptx(
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        [Uint32(addr), Int32(rank)],
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    ptx(
        "st.shared::cluster.v4.f32 [$0], {$1,$2,$3,$4};",
        "r,f,f,f,f,~{memory}",
        [Uint32(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u32(addr, a, *, loc=None, ip=None):
    ptx(
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        [Int64(addr), Uint32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster_f32(addr, a, *, loc=None, ip=None):
    ptx(
        "st.shared::cluster.f32 [$0], $1;",
        "r,f,~{memory}",
        [Uint32(addr), Float32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    return ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        [
            Uint32(a0),
            Uint32(a1),
            Uint32(a2),
            Uint32(a3),
            Uint32(b0),
            Uint32(b1),
            Float32(c0),
            Float32(c1),
            Float32(c2),
            Float32(c3),
        ],
        (Float32,) * 4,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prmt(a, b, sel, *, loc=None, ip=None):
    return ptx(
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [Uint32(a), Uint32(b), Uint32(sel)],
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return ptx(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [Float32(x)],
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


def _neg_inf():
    return Float32(float("-inf"))


class _FlintDecodeAttention:
    """Grid (splits, kv heads, batch). The attended pages of a (request, kv
    head) are interleaved over the warps of its active splits; a split is
    active when every one of its warps gets at least ``min_pages`` pages.
    With ``cluster`` the splits of a (request, kv head) form a thread-block
    cluster and merge through distributed shared memory instead of global
    partials."""

    def __init__(
        self,
        num_kv_heads,
        num_splits,
        num_warps,
        num_stages,
        sliding,
        min_pages,
        cluster,
    ):
        self.num_kv_heads = num_kv_heads
        self.num_splits = num_splits
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.sliding = sliding
        self.min_pages = min_pages
        self.cluster = cluster
        self.ctas_per_sm = 0 if sliding else max(1, 2 * 4 // num_warps)

    @cute.jit
    def __call__(
        self,
        q,
        kv,
        bt,
        seq_lens,
        sinks,
        out,
        part_o,
        part_ml,
        counters,
        scale_log2: Float32,
        window: Int32,
        stream,
    ):
        batch = seq_lens.shape[0]
        self.kernel(
            q,
            kv,
            bt,
            seq_lens,
            sinks,
            out,
            part_o,
            part_ml,
            counters,
            scale_log2,
            window,
        ).launch(
            grid=(self.num_splits, self.num_kv_heads, batch),
            block=(32 * self.num_warps, 1, 1),
            stream=stream,
            cluster=(self.num_splits, 1, 1) if self.cluster else None,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q,
        kv,
        bt,
        seq_lens,
        sinks,
        out,
        part_o,
        part_ml,
        counters,
        scale_log2: Float32,
        window: Int32,
    ):
        NW = self.num_warps
        NS = self.num_stages
        NSPLIT = self.num_splits
        H = self.num_kv_heads
        MINP = self.min_pages
        tid, _, _ = cute.arch.thread_idx()
        split, kvh, b = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane >> 2
        t = lane & 3

        smem = cutlass.memory.SmemAllocator()
        ring = smem.allocate_array(
            cutlass.Uint8, NW * NS * _PAGE_BYTES, byte_alignment=128
        )
        if cutlass.const_expr(self.ctas_per_sm > 0):
            # Pad shared memory so at most ``ctas_per_sm`` CTAs share an SM:
            # with PDL the next grid's CTAs launch while this one drains and
            # could otherwise pile up on the first SMs to free up.
            pad = _SMEM_PER_SM // (self.ctas_per_sm + 1) + 1 - NW * NS * _PAGE_BYTES
            if cutlass.const_expr(pad > 0):
                smem.allocate_array(cutlass.Uint8, pad, byte_alignment=128)
        ring_base = smem_addr(ring)
        mbars = smem.allocate_array(Int64, NW * NS, byte_alignment=8)
        flag = smem.allocate_array(Int32, 4, byte_alignment=16)
        flag_addr = smem_addr(flag)
        cbuf_addr = flag_addr
        if cutlass.const_expr(self.cluster):
            # Per rank: this rank's 128 / NSPLIT dims of every split's partial
            # O ([split][head][dim]), then every split's m and l per head.
            cbuf = smem.allocate_array(
                Float32,
                NSPLIT * 8 * (_HEAD_DIM // NSPLIT) + 16 * NSPLIT,
                byte_alignment=16,
            )
            cbuf_addr = smem_addr(cbuf)

        # seq_lens / block_tables are read before griddepcontrol.wait by design:
        # they are written by host-to-device copies ahead of the forward, not by
        # the preceding kernel (only q, the current K/V and the workspace are).
        seq_len = seq_lens[b]
        start_tok = Int32(0)
        if cutlass.const_expr(self.sliding):
            start_tok = cutlass.max(seq_len - window, Int32(0))
        start_page = start_tok // _PAGE
        num_pages = (seq_len + _PAGE - 1) // _PAGE - start_page
        num_splits = cutlass.min(
            Int32(NSPLIT), (num_pages + NW * MINP - 1) // (NW * MINP)
        )
        if cutlass.const_expr(self.cluster):
            # Every rank of the cluster merges a slice of the output.
            num_splits = Int32(NSPLIT)

        bh = b * H + kvh
        q_base = q.iterator.toint()
        out_base = out.iterator.toint()
        kv_base = kv.iterator.toint()
        cnt_base = counters.iterator.toint()
        arrive_addr = cnt_base + Int64(bh) * 4
        sq0 = Int64(q.stride[0])
        sq1 = Int64(q.stride[1])
        so0 = Int64(out.stride[0])
        so1 = Int64(out.stride[1])
        skb = Int64(kv.stride[0])
        skh = Int64(kv.stride[1])
        sbt = Int64(bt.stride[0])

        # CTA epilogue mapping: E consecutive dims of one head per thread.
        E = 1024 // (32 * NW)
        TPH = 128 // E
        eh = tid // TPH
        ed = (tid % TPH) * E
        out_row = out_base + (Int64(b) * so0 + Int64(kvh * _GQA + eh) * so1) * 2

        if num_pages <= 0:
            cute.arch.griddepcontrol_wait()
            if split == 0:
                if cutlass.const_expr(E == 4):
                    _stg_u32x2(out_row + ed * 2, 0, 0)
                for i in cutlass.range_constexpr(E // 8):
                    stg_u32x4(out_row + (ed + 8 * i) * 2, 0, 0, 0, 0)
        elif split < num_splits:
            # Pages are interleaved over all warps of this (request, kv head).
            WH = num_splits * NW
            gw = split * NW + warp
            n_w = (cutlass.max(num_pages - gw, Int32(0)) + WH - 1) // WH
            bt_head = (
                bt.iterator.toint() + (Int64(b) * sbt + Int64(start_page + gw)) * 4
            )
            kv_head = kv_base + Int64(kvh) * skh * 2
            win_cur = _ldg_i32_pred(bt_head + Int64(WH * lane) * 4, Int32(lane < n_w))
            win_nxt = _ldg_i32_pred(
                bt_head + Int64(WH * (lane + 32)) * 4, Int32(lane + 32 < n_w)
            )
            warp_ring = ring_base + Uint32(warp * NS * _PAGE_BYTES)
            warp_mbar = mbars + warp * NS
            if lane == 0:
                for s in cutlass.range_constexpr(NS):
                    cute.arch.mbarrier_init(warp_mbar + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_warp()
            if cutlass.const_expr(self.cluster):
                # Peers must be running before DSMEM stores (waited on there).
                cute.arch.cluster_arrive_relaxed()

            # Only the page holding the current token (its K/V write), q and
            # the workspace depend on the previous kernel: prefetch the other
            # leading pages before waiting on it.
            p_last = (seq_len - 1) // _PAGE
            for s in cutlass.range_constexpr(NS - 1):
                if (s < n_w) & (start_page + gw + WH * s != p_last):
                    self.issue(
                        kv_head,
                        skb,
                        cute.arch.shuffle_sync(win_cur, s),
                        warp_ring + Uint32(s * _PAGE_BYTES),
                        warp_mbar + s,
                        lane,
                    )
            cute.arch.griddepcontrol_wait()

            # Q as the B operand of S^T = K Q^T: lane (g, t) holds head g, dims
            # 32i + 8t .. +7.
            qf = cute.make_rmem_tensor((16,), Uint32)
            q_row = q_base + (Int64(b) * sq0 + Int64(kvh * _GQA + g) * sq1) * 2
            for i in cutlass.range_constexpr(4):
                w = _ldg128(q_row + (32 * i + 8 * t) * 2)
                for j in cutlass.range_constexpr(4):
                    qf[4 * i + j] = w[j]

            m0 = _neg_inf()
            m1 = _neg_inf()
            l0 = Float32(0.0)
            l1 = Float32(0.0)
            acc = cute.make_rmem_tensor((32,), Float32)
            for i in cutlass.range_constexpr(32):
                acc[i] = Float32(0.0)
            kv_regs = cute.make_rmem_tensor((64,), Uint32)

            for s in cutlass.range_constexpr(NS - 1):
                if (s < n_w) & (start_page + gw + WH * s == p_last):
                    self.issue(
                        kv_head,
                        skb,
                        cute.arch.shuffle_sync(win_cur, s),
                        warp_ring + Uint32(s * _PAGE_BYTES),
                        warp_mbar + s,
                        lane,
                    )
            for k in cutlass.range(n_w, unroll=1):
                # Refill the slot consumed in the previous iteration.
                kn = k + NS - 1
                if (kn & 31) == 0:
                    win_cur = win_nxt
                    win_nxt = _ldg_i32_pred(
                        bt_head + Int64(WH * (kn + 32 + lane)) * 4,
                        Int32(kn + 32 + lane < n_w),
                    )
                # Every lane's shared-memory reads of this slot (previous
                # iteration) are done before lane 0 refills it.
                cute.arch.sync_warp()
                if kn < n_w:
                    self.issue(
                        kv_head,
                        skb,
                        cute.arch.shuffle_sync(win_cur, kn & 31),
                        warp_ring + Uint32((kn % NS) * _PAGE_BYTES),
                        warp_mbar + kn % NS,
                        lane,
                    )
                slot = k % NS
                cute.arch.mbarrier_wait(warp_mbar + slot, (k // NS) & 1)
                page = start_page + gw + WH * k
                self.load_page(
                    kv_regs,
                    warp_ring + Uint32(slot * _PAGE_BYTES),
                    page,
                    start_tok,
                    seq_len,
                    g,
                    t,
                )
                m0, m1, l0, l1 = self.compute(
                    kv_regs,
                    qf,
                    acc,
                    page,
                    start_tok,
                    seq_len,
                    scale_log2,
                    m0,
                    m1,
                    l0,
                    l1,
                    g,
                    t,
                    lane,
                )

            for off in cutlass.range_constexpr(3):
                l0 = l0 + cute.arch.shuffle_sync_bfly(l0, 4 << off)
                l1 = l1 + cute.arch.shuffle_sync_bfly(l1, 4 << off)
            # K/V streaming is done: let a PDL-launched successor start its
            # prologue (e.g. o_proj weight loads) during the combine.
            cute.arch.griddepcontrol_launch_dependents()

            # CTA combine through shared memory (reuses the ring).
            cute.arch.sync_threads()
            wbuf = ring_base + Uint32(warp * _WBUF * 4)
            for j in cutlass.range_constexpr(8):
                d0 = 8 * g + 2 * j if j < 4 else 64 + 8 * g + 2 * (j - 4)
                sts_f32(wbuf + ((2 * t) * 128 + d0) * 4, acc[4 * j])
                sts_f32(wbuf + ((2 * t + 1) * 128 + d0) * 4, acc[4 * j + 1])
                sts_f32(wbuf + ((2 * t) * 128 + d0 + 1) * 4, acc[4 * j + 2])
                sts_f32(wbuf + ((2 * t + 1) * 128 + d0 + 1) * 4, acc[4 * j + 3])
            if g == 0:
                sts_f32(wbuf + (1024 + 2 * t) * 4, m0)
                sts_f32(wbuf + (1024 + 2 * t + 1) * 4, m1)
                sts_f32(wbuf + (1032 + 2 * t) * 4, l0)
                sts_f32(wbuf + (1032 + 2 * t + 1) * 4, l1)
            cute.arch.sync_threads()

            m_cta = _neg_inf()
            for w in cutlass.range_constexpr(NW):
                m_cta = cute.arch.fmax(
                    m_cta, lds_f32(ring_base + (w * _WBUF + 1024 + eh) * 4)
                )
            m_cta_s = m_cta * scale_log2
            l_cta = Float32(0.0)
            o = cute.make_rmem_tensor((E,), Float32)
            for e in cutlass.range_constexpr(E):
                o[e] = Float32(0.0)
            for w in cutlass.range_constexpr(NW):
                mw = lds_f32(ring_base + (w * _WBUF + 1024 + eh) * 4)
                corr = Float32(0.0)
                if mw != _neg_inf():
                    corr = _ex2(fma(mw, scale_log2, -m_cta_s))
                l_cta = l_cta + corr * lds_f32(ring_base + (w * _WBUF + 1032 + eh) * 4)
                for e in cutlass.range_constexpr(E):
                    o[e] = o[e] + corr * lds_f32(
                        ring_base + (w * _WBUF + eh * 128 + ed + e) * 4
                    )

            sink_s = Float32(sinks[kvh * _GQA + eh]) * _LOG2E
            if cutlass.const_expr(self.cluster):
                self.cluster_reduce(
                    cbuf_addr, split, tid, eh, ed, o, m_cta, l_cta, sinks, kvh,
                    scale_log2, out_base + Int64(b) * so0 * 2, so1, E,
                )  # fmt: skip
            elif num_splits == 1:
                inv = Float32(1.0) / (l_cta + _ex2(sink_s - m_cta_s))
                self.store_out(out_row, ed, o, inv, E)
            else:
                prow = Int64(bh * NSPLIT + split)
                po = part_o.iterator.toint() + (prow * 1024 + eh * 128 + ed) * 4
                for i in cutlass.range_constexpr(E // 4):
                    _stg_f32x4(
                        po + 16 * i, o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]
                    )
                pml = part_ml.iterator.toint() + prow * 64
                if ed == 0:
                    _stg_f32(pml + eh * 4, m_cta)
                    _stg_f32(pml + (8 + eh) * 4, l_cta)
                cute.arch.sync_threads()
                if tid == 0:
                    _sts_i32(flag_addr, _atomic_add_acq_rel(arrive_addr, 1))
                cute.arch.sync_threads()
                if _lds_i32(flag_addr) == num_splits - 1:
                    self.combine_splits(
                        part_o,
                        part_ml,
                        bh,
                        num_splits,
                        eh,
                        ed,
                        tid,
                        ring_base + Uint32(NW * _WBUF * 4),
                        scale_log2,
                        sink_s,
                        out_row,
                        o,
                        E,
                    )
                    if tid == 0:
                        _stg_i32_relaxed(arrive_addr, 0)

    @cute.jit
    def cluster_reduce(
        self, cbuf_addr, split, tid, eh, ed, o, m_cta, l_cta, sinks, kvh,
        scale_log2, out_b, so1, E: cutlass.Constexpr,
    ):  # fmt: skip
        """Distributed cluster merge: each CTA sends rank r the dims
        [r * DPR, (r + 1) * DPR) of its partial O (plus m, l); after one
        cluster barrier every rank merges all splits for its dims."""
        NSPLIT = self.num_splits
        NW = self.num_warps
        DPR = _HEAD_DIM // NSPLIT
        M_OFF = NSPLIT * 8 * DPR
        L_OFF = M_OFF + 8 * NSPLIT
        rank = ed // DPR
        peer = _mapa(cbuf_addr, rank)
        cute.arch.cluster_wait()
        dst = peer + ((split * 8 + eh) * DPR + ed % DPR) * 4
        for i in cutlass.range_constexpr(E // 4):
            _st_cluster_f32x4(
                dst + 16 * i, o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]
            )
        if ed % DPR == 0:
            _st_cluster_f32(peer + (M_OFF + split * 8 + eh) * 4, m_cta)
            _st_cluster_f32(peer + (L_OFF + split * 8 + eh) * 4, l_cta)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # This rank owns dims [split * DPR, (split + 1) * DPR) of all heads:
        # OPT consecutive dims of one head per thread.
        OPT = 8 * DPR // (32 * NW)
        h = tid * OPT // DPR
        dl = tid * OPT % DPR
        m_all = _neg_inf()
        for r in cutlass.range_constexpr(NSPLIT):
            m_all = cute.arch.fmax(m_all, lds_f32(cbuf_addr + (M_OFF + r * 8 + h) * 4))
        m_all_s = m_all * scale_log2
        l_all = Float32(0.0)
        oo = [Float32(0.0)] * OPT
        for r in cutlass.range_constexpr(NSPLIT):
            mr = lds_f32(cbuf_addr + (M_OFF + r * 8 + h) * 4)
            corr = Float32(0.0)
            if mr != _neg_inf():
                corr = _ex2(fma(mr, scale_log2, -m_all_s))
            l_all = l_all + corr * lds_f32(cbuf_addr + (L_OFF + r * 8 + h) * 4)
            for e in cutlass.range_constexpr(OPT):
                oo[e] = oo[e] + corr * lds_f32(
                    cbuf_addr + ((r * 8 + h) * DPR + dl + e) * 4
                )
        sink_s = Float32(sinks[kvh * _GQA + h]) * _LOG2E
        inv = Float32(1.0) / (l_all + _ex2(sink_s - m_all_s))
        dst_g = out_b + (Int64(kvh * _GQA + h) * so1 + split * DPR + dl) * 2
        for e in cutlass.range_constexpr(0, OPT, 2):
            if cutlass.const_expr(e + 1 < OPT):
                _stg_u32(dst_g + e * 2, bf16x2(oo[e] * inv, oo[e + 1] * inv))
            else:
                stg_bf16(dst_g + e * 2, oo[e] * inv)

    @cute.jit
    def combine_splits(
        self,
        part_o,
        part_ml,
        bh,
        num_splits,
        eh,
        ed,
        tid,
        ml_smem,
        scale_log2,
        sink_s,
        out_row,
        o,
        E: cutlass.Constexpr,
    ):
        """Last CTA of a (request, kv head): merge the fp32 split partials."""
        NSPLIT = self.num_splits
        ml_base = part_ml.iterator.toint() + Int64(bh * NSPLIT) * 64
        o_base = (
            part_o.iterator.toint() + (Int64(bh * NSPLIT) * 1024 + eh * 128 + ed) * 4
        )
        # Rounds of RCH splits keep at most ~40 vector loads per thread in flight.
        RCH = min(NSPLIT, max(1, 160 // E))
        vs = [None] * RCH
        for r in cutlass.range_constexpr(RCH):
            sr = Int64(cutlass.min(Int32(r), num_splits - 1))
            vs[r] = [_ldg_cg_f32x4(o_base + sr * 4096 + 16 * i) for i in range(E // 4)]
        for i in cutlass.range_constexpr(
            (NSPLIT * 4 + 32 * self.num_warps - 1) // (32 * self.num_warps)
        ):
            idx = tid + i * 32 * self.num_warps
            if idx < num_splits * 4:
                v = _ldg_cg_f32x4(ml_base + Int64(idx) * 16)
                for j in cutlass.range_constexpr(4):
                    sts_f32(ml_smem + (idx * 4 + j) * 4, v[j])
        cute.arch.sync_threads()
        # Warp w owns heads w, w + NW, ..; its lanes cover the splits. The split
        # weights replace l in ml_smem, the per-head scaled max and sum follow.
        warp = tid // 32
        lane = tid % 32
        for hh in cutlass.range_constexpr(
            _GQA // self.num_warps if self.num_warps < _GQA else 1
        ):
            h = warp + hh * self.num_warps
            if h < _GQA:
                NL = (NSPLIT + 31) // 32
                mloc = [_neg_inf()] * NL
                for r in cutlass.range_constexpr(NL):
                    sidx = lane + 32 * r
                    if sidx < num_splits:
                        mloc[r] = lds_f32(ml_smem + (sidx * 16 + h) * 4)
                mh = mloc[0]
                for r in cutlass.range_constexpr(1, NL):
                    mh = cute.arch.fmax(mh, mloc[r])
                for off in cutlass.range_constexpr(5):
                    mh = cute.arch.fmax(mh, cute.arch.shuffle_sync_bfly(mh, 1 << off))
                mh_s = mh * scale_log2
                lh = Float32(0.0)
                for r in cutlass.range_constexpr(NL):
                    sidx = lane + 32 * r
                    if sidx < num_splits:
                        corr = _ex2(fma(mloc[r], scale_log2, -mh_s))
                        lh = lh + corr * lds_f32(ml_smem + (sidx * 16 + 8 + h) * 4)
                        sts_f32(ml_smem + (sidx * 16 + 8 + h) * 4, corr)
                for off in cutlass.range_constexpr(5):
                    lh = lh + cute.arch.shuffle_sync_bfly(lh, 1 << off)
                if lane == 0:
                    sts_f32(ml_smem + (NSPLIT * 16 + h) * 4, mh_s)
                    sts_f32(ml_smem + (NSPLIT * 16 + 8 + h) * 4, lh)
        cute.arch.sync_threads()
        m_all_s = lds_f32(ml_smem + (NSPLIT * 16 + eh) * 4)
        l_all = lds_f32(ml_smem + (NSPLIT * 16 + 8 + eh) * 4)
        for e in cutlass.range_constexpr(E):
            o[e] = Float32(0.0)
        for r0 in cutlass.range_constexpr(0, NSPLIT, RCH):
            if cutlass.const_expr(r0 > 0):  # noqa: SIM102 (static vs dynamic)
                if r0 < num_splits:
                    for r in cutlass.range_constexpr(RCH):
                        sr = Int64(cutlass.min(Int32(r0 + r), num_splits - 1))
                        vs[r] = [
                            _ldg_cg_f32x4(o_base + sr * 4096 + 16 * i)
                            for i in range(E // 4)
                        ]
            for r in cutlass.range_constexpr(min(RCH, NSPLIT - r0)):
                corr = Float32(0.0)
                if r0 + r < num_splits:
                    corr = lds_f32(ml_smem + ((r0 + r) * 16 + 8 + eh) * 4)
                for i in cutlass.range_constexpr(E // 4):
                    for j in cutlass.range_constexpr(4):
                        o[4 * i + j] = o[4 * i + j] + corr * vs[r][i][j]
        inv = Float32(1.0) / (l_all + _ex2(sink_s - m_all_s))
        self.store_out(out_row, ed, o, inv, E)

    @cute.jit
    def store_out(self, out_row, ed, o, inv, E: cutlass.Constexpr):
        if cutlass.const_expr(E == 4):
            _stg_u32x2(
                out_row + ed * 2,
                bf16x2(o[0] * inv, o[1] * inv),
                bf16x2(o[2] * inv, o[3] * inv),
            )
        for i in cutlass.range_constexpr(E // 8):
            w = [
                bf16x2(o[8 * i + 2 * j] * inv, o[8 * i + 2 * j + 1] * inv)
                for j in range(4)
            ]
            stg_u32x4(out_row + (ed + 8 * i) * 2, w[0], w[1], w[2], w[3])

    @cute.jit
    def issue(self, kv_head, skb, blk, dst, mbar, lane):
        """Bulk-copy one 8 KB page (16 tokens x [K | V]) into a ring slot."""
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, _PAGE_BYTES)
            _bulk_copy(
                dst, kv_head + Int64(blk) * skb * 2, smem_addr(mbar), _PAGE_BYTES
            )

    @cute.jit
    def load_page(self, kv, slot, page, start_tok, seq_len, g, t):
        """Lane fragments: K tokens g, g+8 (dims 32i + 8t ..) and V tokens 2t,
        2t+1, 2t+8, 2t+9 (dims 64h + 8g ..). V rows outside the attended range
        are zeroed so stale cache contents never reach P V."""
        for r in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(4):
                w = _lds128(slot + (g + 8 * r) * _ROW_BYTES + (32 * i + 8 * t) * 2)
                for j in cutlass.range_constexpr(4):
                    kv[16 * r + 4 * i + j] = w[j]
        vbase = slot + _HEAD_DIM * 2
        for r in cutlass.range_constexpr(4):
            tok = 2 * t + (r % 2) + 8 * (r // 2)
            for h in cutlass.range_constexpr(2):
                w = _lds128(vbase + tok * _ROW_BYTES + (64 * h + 8 * g) * 2)
                for j in cutlass.range_constexpr(4):
                    kv[32 + 4 * (2 * r + h) + j] = w[j]
        p0 = page * _PAGE
        if (p0 < start_tok) | (p0 + _PAGE > seq_len):
            for r in cutlass.range_constexpr(4):
                pos = p0 + 2 * t + (r % 2) + 8 * (r // 2)
                ok = (pos >= start_tok) & (pos < seq_len)
                for j in cutlass.range_constexpr(8):
                    kv[32 + 8 * r + j] = cutlass.select_(
                        ok, kv[32 + 8 * r + j], Uint32(0)
                    )

    @cute.jit
    def compute(
        self,
        kv,
        qf,
        acc,
        page,
        start_tok,
        seq_len,
        scale_log2,
        m0,
        m1,
        l0,
        l1,
        g,
        t,
        lane,
    ):
        # S^T[tok, head] = K Q^T. Chunk c covers dims 32(c//2) + 8t + 4(c%2) + 0..3.
        sa = [Float32(0.0)] * 4
        sb = [Float32(0.0)] * 4
        for c in cutlass.range_constexpr(8):
            i = c // 2
            h = c % 2
            frag = (
                kv[4 * i + 2 * h],
                kv[16 + 4 * i + 2 * h],
                kv[4 * i + 2 * h + 1],
                kv[16 + 4 * i + 2 * h + 1],
                qf[4 * i + 2 * h],
                qf[4 * i + 2 * h + 1],
            )
            if cutlass.const_expr(c % 2 == 0):
                sa = list(_mma(*frag, *sa))
            else:
                sb = list(_mma(*frag, *sb))
        s0 = sa[0] + sb[0]
        s1 = sa[1] + sb[1]
        s2 = sa[2] + sb[2]
        s3 = sa[3] + sb[3]
        pos = page * _PAGE + g
        ninf = _neg_inf()
        if (pos < start_tok) | (pos >= seq_len):
            s0 = ninf
            s1 = ninf
        if (pos + 8 < start_tok) | (pos + 8 >= seq_len):
            s2 = ninf
            s3 = ninf
        mx0 = cute.arch.fmax(s0, s2)
        mx1 = cute.arch.fmax(s1, s3)
        for off in cutlass.range_constexpr(3):
            mx0 = cute.arch.fmax(mx0, cute.arch.shuffle_sync_bfly(mx0, 4 << off))
            mx1 = cute.arch.fmax(mx1, cute.arch.shuffle_sync_bfly(mx1, 4 << off))
        mn0 = cute.arch.fmax(m0, mx0)
        mn1 = cute.arch.fmax(m1, mx1)
        ms0 = mn0 * scale_log2
        ms1 = mn1 * scale_log2
        alpha0 = _ex2(fma(m0, scale_log2, -ms0))
        alpha1 = _ex2(fma(m1, scale_log2, -ms1))
        p0 = _ex2(fma(s0, scale_log2, -ms0))
        p1 = _ex2(fma(s1, scale_log2, -ms1))
        p2 = _ex2(fma(s2, scale_log2, -ms0))
        p3 = _ex2(fma(s3, scale_log2, -ms1))
        l0 = l0 * alpha0 + (p0 + p2)
        l1 = l1 * alpha1 + (p1 + p3)
        for j in cutlass.range_constexpr(8):
            acc[4 * j] = acc[4 * j] * alpha0
            acc[4 * j + 1] = acc[4 * j + 1] * alpha1
            acc[4 * j + 2] = acc[4 * j + 2] * alpha0
            acc[4 * j + 3] = acc[4 * j + 3] * alpha1

        # P^T as the B operand: lane (g, t) needs P[tok 2t, 2t+1, 2t+8, 2t+9][head g].
        x = bf16x2(p0, p1)
        y = bf16x2(p2, p3)
        src0 = 8 * t + (g >> 1)
        x0 = cute.arch.shuffle_sync(x, src0)
        x1 = cute.arch.shuffle_sync(x, src0 + 4)
        y0 = cute.arch.shuffle_sync(y, src0)
        y1 = cute.arch.shuffle_sync(y, src0 + 4)
        sel = cutlass.select_((g & 1) == 0, Uint32(0x5410), Uint32(0x7632))
        b0 = _prmt(x0, x1, sel)
        b1 = _prmt(y0, y1, sel)

        # O^T[dim, head] += V^T P^T. m-tile j rows g / g+8 are dims d0 / d0+1.
        for j in cutlass.range_constexpr(8):
            hh = j // 4
            wi = j % 4
            va = kv[32 + 4 * (0 * 2 + hh) + wi]
            vb = kv[32 + 4 * (1 * 2 + hh) + wi]
            vc = kv[32 + 4 * (2 * 2 + hh) + wi]
            vd = kv[32 + 4 * (3 * 2 + hh) + wi]
            acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3] = _mma(
                _prmt(va, vb, 0x5410),
                _prmt(va, vb, 0x7632),
                _prmt(vc, vd, 0x5410),
                _prmt(vc, vd, 0x7632),
                b0,
                b1,
                acc[4 * j],
                acc[4 * j + 1],
                acc[4 * j + 2],
                acc[4 * j + 3],
            )
        return mn0, mn1, l0, l1


@cache
def _compile(
    num_kv_heads, num_splits, num_warps, num_stages, sliding, min_pages, cluster
):
    bf16 = cutlass.BFloat16
    batch = cute.sym_int()
    q = make_fake_tensor(
        bf16, (cute.sym_int(), num_kv_heads * _GQA, _HEAD_DIM), divisibility=8
    )
    kv = make_fake_tensor(
        bf16, (cute.sym_int(), num_kv_heads, _PAGE, 2 * _HEAD_DIM), divisibility=8
    )
    bt = make_fake_tensor(Int32, (batch, cute.sym_int()), divisibility=1)
    seq_lens = make_fake_tensor(Int32, (batch,), divisibility=1)
    sinks = make_fake_tensor(Float32, (num_kv_heads * _GQA,), divisibility=1)
    out = make_fake_tensor(
        bf16, (cute.sym_int(), num_kv_heads * _GQA, _HEAD_DIM), divisibility=8
    )
    part_o = make_fake_tensor(Float32, (cute.sym_int(),), divisibility=4)
    part_ml = make_fake_tensor(Float32, (cute.sym_int(),), divisibility=4)
    counters = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _FlintDecodeAttention(
        num_kv_heads, num_splits, num_warps, num_stages, sliding, min_pages, cluster
    )
    return cute.compile(
        kernel,
        q,
        kv,
        bt,
        seq_lens,
        sinks,
        out,
        part_o,
        part_ml,
        counters,
        Float32(1.0),
        Int32(0),
        stream,
        options="--enable-tvm-ffi",
    )


# Upper bounds for the persistent workspace. Partials are only used with more
# than one (non-cluster) split, where num_bh * splits <= 2 * num_sms.
_MAX_PARTIALS_PER_SM = 4
_MAX_BH = 1 << 16

_workspaces: dict = {}
_last_stream: dict = {}


def _workspace(device, num_partials, num_bh):
    """Persistent split partials and arrival counters, allocated once per
    device at a fixed maximum (never replaced, so captured CUDA graphs keep
    valid pointers). Counters reset themselves and are zeroed only at
    allocation; launches sharing them must be ordered on one stream, as within
    a model forward; a launch from a different stream than the previous one
    first waits for that stream."""
    if not torch.cuda.is_current_stream_capturing():
        stream = torch.cuda.current_stream(device)
        last = _last_stream.get(device)
        if last is not None and last != stream:
            stream.wait_stream(last)
        _last_stream[device] = stream
    ws = _workspaces.get(device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Flint decode attention must run eagerly once before CUDA graph "
                "capture."
            )
        max_partials = _MAX_PARTIALS_PER_SM * _num_sms(device.index)
        ws = (
            torch.empty(
                max_partials * _GQA * _HEAD_DIM, dtype=torch.float32, device=device
            ),
            torch.empty(max_partials * 2 * _GQA, dtype=torch.float32, device=device),
            torch.zeros(_MAX_BH, dtype=torch.int32, device=device),
        )
        _workspaces[device] = ws
    if ws[1].numel() < num_partials * 2 * _GQA or ws[2].numel() < num_bh:
        raise ValueError(
            f"Flint decode attention workspace too small for {num_partials} "
            f"partials / {num_bh} (request, kv head) pairs"
        )
    return ws


@cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _launch_config(batch, num_kv_heads, sliding, num_sms):
    """(num_splits, num_warps, num_stages, min_pages, cluster) for a static
    batch size."""
    num_bh = batch * num_kv_heads
    if sliding:
        if batch <= 2:
            return 8, 4, 2, 1, True
        return (4, 4, 3, 1, True) if batch <= 16 else (1, 4, 3, 1, False)
    if batch <= 2:
        return max(1, num_sms // num_bh), 8, 2, 2, False
    return max(1, 2 * num_sms // num_bh), 4, 2 if batch <= 4 else 3, 2, False


def _supports_kv_cache(kv_cache: torch.Tensor) -> bool:
    """Whether ``kv_cache`` has the HND bf16 page layout the kernel assumes:
    [blocks, kv_heads, 16, 2 * 128], contiguous pages, 16 B aligned."""
    return (
        kv_cache.dtype == torch.bfloat16
        and kv_cache.dim() == 4
        and kv_cache.shape[2] == _PAGE
        and kv_cache.shape[3] == 2 * _HEAD_DIM
        and kv_cache.stride(3) == 1
        and kv_cache.stride(2) == 2 * _HEAD_DIM
        and kv_cache.stride(0) % 8 == 0
        and kv_cache.stride(1) % 8 == 0
        and kv_cache.data_ptr() % 16 == 0
    )


def flint_decode_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    window: int | None,
    scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Paged decode attention with sinks, one query token per request.

    CUDA-graph safe: the launch shape depends only on the batch size; sequence
    lengths and block tables are read on the device.

    Args:
        q: [B, num_heads, 128] bf16 queries.
        kv_cache: [num_blocks, num_kv_heads, 16, 256] bf16 HND cache with K in
            ``[..., :128]`` and V in ``[..., 128:]``.
        block_tables: [B, max_blocks] int32.
        seq_lens: [B] int32 KV lengths including the current token (0 writes
            zeros).
        sinks: [num_heads] fp32 sink logits.
        window: Attend to the last ``window`` positions, or None for all.
        scale: Softmax scale.
        out: [B, num_heads, 128] bf16 output.

    Returns:
        ``out``.

    """
    batch, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[1]
    assert head_dim == _HEAD_DIM and num_heads == num_kv_heads * _GQA
    assert _supports_kv_cache(kv_cache), (
        "Flint decode attention needs FlashInfer's HND bf16 KV cache with "
        "16-token blocks"
    )
    assert q.stride(-1) == 1 and out.stride(-1) == 1
    assert seq_lens.shape[0] == batch == out.shape[0] == block_tables.shape[0]
    if batch == 0:
        return out
    sliding = window is not None
    splits, num_warps, num_stages, min_pages, cluster = _launch_config(
        batch, num_kv_heads, sliding, _num_sms(q.device.index)
    )
    num_bh = batch * num_kv_heads
    part_o, part_ml, counters = _workspace(
        q.device, num_bh * splits if splits > 1 and not cluster else 0, num_bh
    )
    kernel = _compile(
        num_kv_heads, splits, num_warps, num_stages, sliding, min_pages, cluster
    )
    kernel(
        q,
        kv_cache,
        block_tables,
        seq_lens,
        sinks,
        out,
        part_o,
        part_ml,
        counters,
        scale * _LOG2E,
        window if sliding else 0,
    )
    return out
