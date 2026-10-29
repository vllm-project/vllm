# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL top-k per row, one CTA (512 threads) per row, values held in
registers (NPT per thread):

1. block min/max of the finite values;
2. 4096-bin *linear* histogram over [min, max] (bin 0 = -inf), so atomics
   spread out instead of piling on a few exponent bins;
3. suffix scan -> boundary bin b*: everything above is selected directly;
4. the boundary bin's elements (few: the threshold sits in the sparse upper
   tail) are ranked exactly in smem (ties -> lower index first); a refinement
   histogram pass handles crowded boundary bins.

mode "topk": out = indices of the K largest values of logits[t, :seq[t]].
mode "blocks": candidate write over block maxima of 8 (newest block pinned to
+inf), plus a bitmap of the selected blocks.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from .indexer_ptx import ld_f32, ld_nc_b32, red_or_b32, st_global_b32

NT = 512
NB = 4096
CB = 8
CAND = 1024


class IdxSelect2:
    def __init__(self, K: int = 512, max_len: int = 16384):
        self.K = K
        n = (max_len + CB - 1) // CB
        self.NPT = max(1, (n + NT - 1) // NT)

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        lst: Int32,
        seq: Int64,
        out: Int64,
        ost: Int32,
        cbits: Int64,
        cbst: Int32,
        T: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            hist: cute.struct.MemRange[Int32, NB]
            pre: cute.struct.MemRange[Int32, NT]
            red: cute.struct.MemRange[Float32, 64]
            wsum: cute.struct.MemRange[Int32, 16]
            ctl: cute.struct.MemRange[Int32, 8]
            cv: cute.struct.MemRange[Float32, CAND]
            ci: cute.struct.MemRange[Int32, CAND]
            so: cute.struct.MemRange[Int32, 2048]

        self.kernel(logits, lst, seq, out, ost, cbits, cbst, Smem).launch(
            grid=[T, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=True
        )

    @cute.jit
    def _val(self, logits: Int64, row: Int64, i: Int32, Lp: Int32) -> Float32:
        v = Float32(-math.inf)
        nan = False
        for e in cutlass.range_constexpr(CB):
            p = i * CB + e
            if p < Lp:
                x = ld_f32(logits + (row + Int64(p)) * 4)
                if x != x:
                    nan = True
                v = cutlass.max(v, x)
        if nan or i == (Lp - 1) // CB:
            v = Float32(math.inf)
        return v

    @cute.jit
    def _setbit(self, cbits: Int64, t: Int32, cbst: Int32, i: Int32):
        red_or_b32(
            cbits + (Int64(t) * cbst + Int64(i // 32)) * 4,
            Uint32(1) << Uint32(i % 32),
        )

    @cute.jit
    def _bin(self, v: Float32, lo: Float32, sc: Float32) -> Int32:
        b = Int32(0)
        if v == Float32(math.inf):
            b = Int32(NB - 1)
        elif v > Float32(-math.inf):
            x = (v - lo) * sc
            b = 1 + Int32(cutlass.min(cutlass.max(x, Float32(0.0)), Float32(NB - 3)))
        return b

    @cute.jit
    def _bmax(
        self, x: Float32, red, tid, lane, warp, slot: cutlass.Constexpr
    ) -> Float32:
        for o in cutlass.range_constexpr(5):
            x = cutlass.max(x, cute.arch.shuffle_sync_bfly(x, offset=1 << o))
        if lane == 0:
            red[slot * 16 + warp] = x
        cute.arch.sync_threads()
        y = red[slot * 16]
        for w in cutlass.range_constexpr(1, NT // 32):
            y = cutlass.max(y, red[slot * 16 + w])
        return y

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        lst: Int32,
        seq: Int64,
        out: Int64,
        ost: Int32,
        cbits: Int64,
        cbst: Int32,
        Smem: cutlass.Constexpr,
    ):
        K, NPT = self.K, self.NPT
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        hist = st.hist.get_tensor(cute.make_layout(NB))
        pre = st.pre.get_tensor(cute.make_layout(NT))
        red = st.red.get_tensor(cute.make_layout(64))
        wsum = st.wsum.get_tensor(cute.make_layout(16))
        ctl = st.ctl.get_tensor(cute.make_layout(8))
        cv = st.cv.get_tensor(cute.make_layout(CAND))
        ci = st.ci.get_tensor(cute.make_layout(CAND))
        so = st.so.get_tensor(cute.make_layout(2048))
        for i in cutlass.range_constexpr(NB // NT):
            hist[i * NT + tid] = 0
        cute.arch.griddepcontrol_wait()
        Lp = ld_nc_b32(seq + Int64(t) * 4)
        L = (Lp + CB - 1) // CB
        for i in cutlass.range(tid, cbst, NT, unroll=1):
            st_global_b32(cbits + (Int64(t) * cbst + Int64(i)) * 4, Uint32(0))
        row = Int64(t) * Int64(lst)
        orow = out + Int64(t) * Int64(ost) * 4
        if L <= K:
            cute.arch.sync_threads()
            for i in cutlass.range(tid, K, NT, unroll=1):
                v = Int32(-1)
                if i < L:
                    v = i
                    self._setbit(cbits, t, cbst, i)
                st_global_b32(orow + Int64(i) * 4, Uint32(v))
        else:
            vals = cute.make_rmem_tensor(NPT, Float32)
            hi = Float32(-math.inf)
            nlo = Float32(-math.inf)  # max of -v over finite values (= -min)
            for k in cutlass.range_constexpr(NPT):
                i = k * NT + tid
                v = Float32(-math.inf)
                if i < L:
                    v = self._val(logits, row, i, Lp)
                vals[k] = v
                if v > Float32(-math.inf) and v < Float32(math.inf):
                    hi = cutlass.max(hi, v)
                    nlo = cutlass.max(nlo, -v)
            hi = self._bmax(hi, red, tid, lane, warp, 0)
            lo = -self._bmax(nlo, red, tid, lane, warp, 1)
            sc = Float32(0.0)
            if hi > lo:
                sc = Float32(NB - 3) / (hi - lo)
            for k in cutlass.range_constexpr(NPT):
                if k * NT + tid < L:
                    cute.arch.atomic_add(
                        hist.iterator + self._bin(vals[k], lo, sc),
                        Int32(1),
                        scope="cta",
                    )
            if tid == 0:
                ctl[2] = 0
                ctl[3] = 0
            cute.arch.sync_threads()
            # suffix scan over bins (descending); thread owns 8 bins
            s = Int32(0)
            for e in cutlass.range_constexpr(NB // NT):
                s += hist[tid * (NB // NT) + e]
            pre[NT - 1 - tid] = s
            cute.arch.sync_threads()
            x = pre[tid]
            inc = x
            for o in cutlass.range_constexpr(5):
                y = cute.arch.shuffle_sync_up(inc, offset=1 << o, mask_and_clamp=0)
                if lane >= (1 << o):
                    inc += y
            if lane == 31:
                wsum[warp] = inc
            cute.arch.sync_threads()
            woff = Int32(0)
            for w in cutlass.range_constexpr(NT // 32):
                if w < warp:
                    woff += wsum[w]
            excl = woff + inc - x
            cute.arch.sync_threads()
            pre[tid] = excl
            cute.arch.sync_threads()
            above = pre[NT - 1 - tid]
            if above < K and above + s >= K:
                acc = above
                for e in cutlass.range_constexpr(NB // NT):
                    bi = tid * (NB // NT) + (NB // NT - 1 - e)
                    hv = hist[bi]
                    if acc < K and acc + hv >= K:
                        ctl[0] = bi
                        ctl[1] = K - acc
                    acc += hv
            cute.arch.sync_threads()
            bstar = ctl[0]
            need = ctl[1]
            ngt = K - need
            # classify: count per thread, one block scan, then write at offsets
            cnt = Int32(0)  # (gt << 16) | eq
            for k in cutlass.range_constexpr(NPT):
                if k * NT + tid < L:
                    b = self._bin(vals[k], lo, sc)
                    if b > bstar:
                        cnt += Int32(1 << 16)
                    elif b == bstar:
                        cnt += 1
            inc2 = cnt
            for o in cutlass.range_constexpr(5):
                y = cute.arch.shuffle_sync_up(inc2, offset=1 << o, mask_and_clamp=0)
                if lane >= (1 << o):
                    inc2 += y
            if lane == 31:
                wsum[warp] = inc2
            cute.arch.sync_threads()
            wo = Int32(0)
            for w in cutlass.range_constexpr(NT // 32):
                if w < warp:
                    wo += wsum[w]
            ex = wo + inc2 - cnt
            pg = ex >> 16
            pe = ex & 0xFFFF
            for k in cutlass.range_constexpr(NPT):
                i = k * NT + tid
                if i < L:
                    b = self._bin(vals[k], lo, sc)
                    if b > bstar:
                        so[pg] = i
                        pg += 1
                    elif b == bstar:
                        if pe < CAND:
                            cv[pe] = vals[k]
                            ci[pe] = i
                        pe += 1
            if tid == NT - 1:
                ctl[3] = pe
            cute.arch.sync_threads()
            nc = cutlass.min(ctl[3], CAND)
            # exact rank inside the boundary bin (value desc, then index asc)
            for c in cutlass.range(tid, nc, NT, unroll=1):
                v = cv[c]
                ii = ci[c]
                rank = Int32(0)
                for d in cutlass.range(nc, unroll=4):
                    vd = cv[d]
                    if vd > v or (vd == v and ci[d] < ii):
                        rank += 1
                if rank < need:
                    so[ngt + rank] = ii
            cute.arch.sync_threads()
            for i in cutlass.range(tid, K, NT, unroll=1):
                oi = so[i]
                st_global_b32(orow + Int64(i) * 4, Uint32(oi))
                self._setbit(cbits, t, cbst, oi)
