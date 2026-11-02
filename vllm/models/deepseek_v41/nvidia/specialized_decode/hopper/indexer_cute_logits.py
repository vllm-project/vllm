# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL SM90 FP8 indexer logits, all verify rows of a request sharing each
index-K tile (native next_n).

logits[t, p] = sum_h relu((q[t,h] . k[p]) * kscale[p]) * w[t,h]   for p < seq[t]
             = -inf                                               otherwise

One FP8 WGMMA per 64-position tile: A = K tile (64 x 128 e4m3, SW128 K-major,
cp.async-gathered from the paged 132 B-record cache), B = the request's
R x 32 query heads (R*32 x 128 e4m3). The head reduction is a register +
quad-shuffle sum in the epilogue. Optional candidate mask (per-row bitmap of
8-position candidate blocks): non-candidate positions get -inf and tiles with
no candidate for any row are skipped. Grid (chunks, requests), 128 threads,
PDL (early trigger)."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum
from quack import layout_utils, sm90_utils

from .indexer_ptx import (
    cp_async4,
    cp_async16,
    cp_async_commit,
    cp_async_wait,
    ld_f32,
    ld_nc_b32,
    ld_shared_f32,
    st_global_f32,
)

TM = 64  # positions per tile
D = 128
HQ = 32


class IdxLogits:
    def __init__(
        self, R: int = 6, tiles_per_cta: int = 1, stages: int = 3, cand: bool = False
    ):
        assert 1 <= R <= 8
        self.R = R
        self.N = R * HQ
        self.TPC = tiles_per_cta
        self.ST = stages
        self.cand = cand
        self.dt = cutlass.Float8E4M3FN

    def _layout(self, rows, stages=None):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dt, D),
            self.dt,
        )
        shape = (rows, D) if stages is None else (rows, D, stages)
        return cute.tile_to_shape(
            atom, shape, order=(0, 1) if stages is None else (0, 1, 2)
        )

    @cute.jit
    def __call__(
        self,
        q: Int64,
        w: Int64,
        kc: Int64,
        kc_bs: Int64,
        pshift: Int32,
        btab: Int64,
        btab_st: Int32,
        seq: Int64,
        out: Int64,
        out_st: Int32,
        cbits: Int64,
        cbits_st: Int32,
        nchunks: Int32,
        B: Int32,
        stream: cuda.CUstream,
    ):
        lA = self._layout(TM, self.ST)
        lB = self._layout(self.N)
        mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dt,
            self.dt,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(TM, self.N),
        )

        @cute.struct
        class Smem:
            sA: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(lA)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(lB)], 1024]
            sS: cute.struct.MemRange[Float32, TM * self.ST]
            sSeq: cute.struct.MemRange[Int32, 8]

        self.kernel(
            q,
            w,
            kc,
            kc_bs,
            pshift,
            btab,
            btab_st,
            seq,
            out,
            out_st,
            cbits,
            cbits_st,
            nchunks,
            lA,
            lB,
            mma,
            Smem,
        ).launch(grid=[nchunks, B, 1], block=[128, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        w: Int64,
        kc: Int64,
        kc_bs: Int64,
        pshift: Int32,
        btab: Int64,
        btab_st: Int32,
        seq: Int64,
        out: Int64,
        out_st: Int32,
        cbits: Int64,
        cbits_st: Int32,
        nchunks: Int32,
        sA_l: cute.ComposedLayout,
        sB_l: cute.ComposedLayout,
        mma: cute.TiledMma,
        Smem: cutlass.Constexpr,
    ):
        R, N, ST = self.R, self.N, self.ST
        tid, _, _ = cute.arch.thread_idx()
        ck, b, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32  # noqa: F841
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)
        sB = st.sB.get_tensor(sB_l.outer, swizzle=sB_l.inner)
        a_addr = st.sA.data_ptr().toint()
        b_addr = st.sB.data_ptr().toint()
        s_addr = st.sS.data_ptr().toint()
        sSeq = st.sSeq.get_tensor(cute.make_layout(8))
        cute.arch.griddepcontrol_wait()

        t0 = b * R
        lmax = ld_nc_b32(seq + Int64(t0 + R - 1) * 4)  # last verify row sees the most
        ntile = (lmax + TM - 1) // TM
        # Tiles strided over the request's CTAs: tile ck + i * nchunks.
        nt = cutlass.max((ntile - ck + nchunks - 1) // nchunks, 0)
        if nt > 0:
            if tid < R:
                sSeq[tid] = ld_nc_b32(seq + Int64(t0 + tid) * 4)
            # ---- Q: R*32 rows x 8 chunks of 16 B, SW128 K-major ----
            for it in cutlass.range_constexpr((N * 8 + 127) // 128):
                cid = it * 128 + tid
                if cid < N * 8:
                    n = cid // 8
                    c = cid % 8
                    cp_async16(
                        b_addr + self._swz(n, c),
                        q + Int64(t0 * HQ * D + n * D + c * 16),
                    )
            cp_async_commit()
            page = 1 << pshift
            prow = btab + Int64(t0) * btab_st * 4
            for s in cutlass.range_constexpr(ST - 1):
                if s < nt:
                    self._issue(
                        s,
                        ck + s * nchunks,
                        kc,
                        kc_bs,
                        pshift,
                        page,
                        prow,
                        a_addr,
                        s_addr,
                        tid,
                        lmax,
                    )
                cp_async_commit()
            # per-thread head weights for its accumulator columns
            thr = mma.get_slice(tid)
            cC = cute.make_identity_tensor((TM, N))
            tC = layout_utils.reshape_acc_to_mn(thr.partition_C(cC))
            ncol = cute.size(tC, mode=[1])
            wr = cute.make_rmem_tensor(ncol, Float32)
            for c in cutlass.range_constexpr(ncol):
                n = tC[0, c][1]
                wr[c] = ld_f32(w + Int64(t0 * HQ + n) * 4)
            tA = thr.make_fragment_A(thr.partition_A(sA))
            tB = thr.make_fragment_B(thr.partition_B(sB))
            for i in cutlass.range(nt, unroll=1):
                stg = i % ST
                if i + ST - 1 < nt:
                    self._issue(
                        (i + ST - 1) % ST,
                        ck + (i + ST - 1) * nchunks,
                        kc,
                        kc_bs,
                        pshift,
                        page,
                        prow,
                        a_addr,
                        s_addr,
                        tid,
                        lmax,
                    )
                cp_async_commit()
                cp_async_wait(ST - 1)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                acc = cute.make_rmem_tensor(thr.partition_shape_C((TM, N)), Float32)
                sm90_utils.gemm(
                    mma, acc, tA[None, None, None, stg], tB, zero_init=True, wg_wait=0
                )
                a_mn = layout_utils.reshape_acc_to_mn(acc)
                tile = ck + i * nchunks
                for r in cutlass.range_constexpr(2):
                    ml = tC[r, 0][0]
                    p = tile * TM + ml
                    ks = ld_shared_f32(s_addr + (stg * TM + ml) * 4)
                    rs = cute.make_rmem_tensor(R, Float32)
                    rs.fill(0.0)
                    for c in cutlass.range_constexpr(ncol):
                        j = (tC[r, c][1]) // HQ  # noqa: F841
                        v = a_mn[r, c] * ks
                        v = cutlass.max(v, Float32(0.0))
                        rs[c // (ncol // R)] = rs[c // (ncol // R)] + v * wr[c]
                    for jj in cutlass.range_constexpr(R):
                        x = rs[jj]
                        x = x + cute.arch.shuffle_sync_bfly(x, offset=1)
                        x = x + cute.arch.shuffle_sync_bfly(x, offset=2)
                        rs[jj] = x
                    if lane % 4 == 0:
                        for jj in cutlass.range_constexpr(R):
                            x = rs[jj]
                            if p >= sSeq[jj]:
                                x = Float32(-math.inf)
                            if const_expr(self.cand):
                                bw = ld_nc_b32(
                                    cbits
                                    + (Int64(t0 + jj) * cbits_st + Int64(p // 256)) * 4
                                )
                                if ((bw >> ((p // 8) % 32)) & 1) == 0:
                                    x = Float32(-math.inf)
                            st_global_f32(
                                out + (Int64(t0 + jj) * out_st + Int64(p)) * 4, x
                            )
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def _swz(self, n: Int32, c: Int32) -> Int32:
        """Byte offset of 16 B chunk c of row n in a SW128 K-major fp8 tile (128 B rows)."""  # noqa: E501
        return (n // 8) * 1024 + (n % 8) * 128 + ((c ^ (n % 8)) * 16)

    @cute.jit
    def _issue(
        self,
        stg,
        tile,
        kc: Int64,
        kc_bs: Int64,
        pshift: Int32,
        page: Int32,
        prow: Int64,
        a_addr: Int32,
        s_addr: Int32,
        tid: Int32,
        lmax: Int32,
    ):
        base = a_addr + stg * (TM * D)
        for it in cutlass.range_constexpr(TM * 8 // 128):
            cid = it * 128 + tid
            m = cid // 8
            c = cid % 8
            p = cutlass.min(tile * TM + m, lmax - 1)
            blk = ld_nc_b32(prow + Int64(p >> pshift) * 4)
            rec = kc + Int64(blk) * kc_bs + Int64((p & (page - 1)) * D + c * 16)
            cp_async16(base + self._swz(m, c), rec)
        if tid < TM:
            p = cutlass.min(tile * TM + tid, lmax - 1)
            blk = ld_nc_b32(prow + Int64(p >> pshift) * 4)
            cp_async4(
                s_addr + (stg * TM + tid) * 4,
                kc + Int64(blk) * kc_bs + Int64(page * D + (p & (page - 1)) * 4),
            )
