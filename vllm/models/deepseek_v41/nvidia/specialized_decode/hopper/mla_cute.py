# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels of the Hopper small-batch sparse-MLA decode
(``mla_small``), on the FlashMLA FP8 caches (584-byte V4 records). A row group
is up to 64 // H verify rows of one request, whose H heads each share one
64-row WGMMA tile.

- ``UnionBuild``: per row group, the union of its rows' SWA slots (up to
  ``NSWB`` 64-key blocks) and a per-key row mask.
- ``CuPrep3``: per row group, the union of its rows' compressed top-k slots (up
  to ``NBC`` blocks), optionally mapped from the indexer's local indices
  through the compressed block table, dequantized into fp16 tile images, with
  a per-key visibility byte.
- ``SmallDecode``: one CTA per (row group, block); a normalized BF16 partial
  and its log-sum-exp terms per row.
- ``Combine``: the partials of each row merged with the attention sink.

``CuPrep3``, ``SmallDecode`` and ``Combine`` launch with programmatic
dependent launch: each releases its dependents at entry and waits for its
predecessor before reading its outputs."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum
from quack import layout_utils, sm90_utils

from .mla_ptx import (
    bf16x2_to_f16x2,
    bf16x2_unpack,
    bulk_g2s,
    bulk_s2g,
    bulk_wait_read0,
    cvt_bf16x2,
    cvt_f16x2,
    dq_e4m3x4_f16,
    dq_e4m3x4_f16_2,
    ld_cg_b32,
    ld_cg_v2,
    ld_cg_v4,
    ld_f32,
    ld_nc_b32,
    ld_nc_u8,
    ld_nc_v2,
    ld_nc_v4,
    ld_shared_v2,
    st_global_b32,
    st_global_f32,
    st_global_u8,
    st_global_v2,
    st_global_v4,
    st_i32,
    st_shared_b32,
    st_shared_u8,
    st_shared_v4,
    st_u8,
    u32_as_f32,
)

BM = 64
BN = 64
D = 512
DV = 512
D_NOPE = 448
DATA_TOK = 576
LOG2E = 1.4426950408889634
BAR_P_READY = 1
BAR_P_FREE = 2
BAR_O_READY = 3
NBC = 32  # max compressed blocks per group (4 rows x 512 / 64)
NSWB = 8  # max SWA-union blocks per group (4 rows x 128 / 64)
IMG = BN * D  # fp16 elements per 64-key tile image
EMPTY = 0x7FFFFFFF
TBL = 4096
UNION_TBL = 8192
UNION_NT = 512
EMPTY = 0x7FFFFFFF


class SparseMlaDecodeSm90:
    def __init__(self, H: int = 16, num_stages: int = 2, num_splits: int = 1):
        assert BM % H == 0
        self.H = H
        self.rpg = BM // H
        self.num_stages = num_stages
        self.num_splits = num_splits
        self.dtype = cutlass.Float16

    def _smem_layout(self, rows, cols, stages=None):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, cols
            ),
            self.dtype,
        )
        shape = (rows, cols) if stages is None else (rows, cols, stages)
        order = (0, 1) if stages is None else (0, 1, 2)
        return cute.tile_to_shape(atom, shape, order=order)

    @cute.jit
    def _swz(self, m: Int32, c: Int32) -> Int32:
        """Element offset of 16 B chunk c (8 dims) of row m in a SW128 K-major
        bf16 tile tiled rows-first: atoms of 8 rows x 64 dims (512 elements)."""
        return (
            ((m // 8) + (c // 8) * (BM // 8)) * 512
            + (m % 8) * 64
            + (((c % 8) ^ (m % 8)) * 8)
        )

    @cute.jit
    def _load_token(
        self,
        v: Int32,
        ch: Int32,
        raw: cute.Tensor,
        scl: cute.Tensor,
        rnd: cutlass.Constexpr,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_page: Int32,
        ex_ptr: Int64,
        ex_bs: Int64,
        ex_page: Int32,
    ):
        """Issue the 16 B loads of one key's record (4 threads per key, `ch` =
        16 B lane within each 64-dim tile) into registers."""
        valid = v != -1
        slot = v
        base = swa_ptr
        bs = swa_bs
        page = swa_page
        if v < -1:
            slot = -2 - v
            base = ex_ptr
            bs = ex_bs
            page = ex_page
        if not valid:
            slot = 0
        blkno = slot >> page
        rel = slot & ((1 << page) - 1)
        blk_base = base + Int64(blkno) * bs
        data = blk_base + Int64(rel * DATA_TOK)
        lo, hi = ld_nc_v2(blk_base + Int64((DATA_TOK << page) + rel * 8))
        scl[rnd * 2] = lo
        scl[rnd * 2 + 1] = hi
        o = rnd * 36
        for t in cutlass.range_constexpr(7):
            w0, w1, w2, w3 = ld_nc_v4(data + (t * 64 + ch * 16))
            raw[o + t * 4 + 0] = w0
            raw[o + t * 4 + 1] = w1
            raw[o + t * 4 + 2] = w2
            raw[o + t * 4 + 3] = w3
        for t in cutlass.range_constexpr(2):
            w0, w1, w2, w3 = ld_nc_v4(data + (D_NOPE + ch * 16 + t * 64))
            raw[o + 28 + t * 4 + 0] = w0
            raw[o + 28 + t * 4 + 1] = w1
            raw[o + 28 + t * 4 + 2] = w2
            raw[o + 28 + t * 4 + 3] = w3


class _Common(SparseMlaDecodeSm90):
    @cute.jit
    def _store_token_any(
        self,
        valid,
        tk: Int32,
        ch: Int32,
        base,
        raw: cute.Tensor,
        scl: cute.Tensor,
        rnd: cutlass.Constexpr,
        glob: cutlass.Constexpr,
    ):
        """Dequantise one key into row tk of a SW128 K-major fp16 tile at
        `base` (smem address, or global address if `glob`)."""
        o = rnd * 36
        if not valid:
            for k in cutlass.range_constexpr(36):
                raw[o + k] = Uint32(0)
        sc_lo = scl[rnd * 2]
        sc_hi = scl[rnd * 2 + 1]
        rowo = ((tk // 8) * 512 + (tk % 8) * 64) * 2
        x0 = rowo + (((2 * ch) ^ (tk % 8)) * 8) * 2
        x1 = rowo + (((2 * ch + 1) ^ (tk % 8)) * 8) * 2
        kk = cute.make_rmem_tensor(7, Int32)
        kmin = Int32(127)
        for t in cutlass.range_constexpr(7):
            e = Int32(0)
            if const_expr(t < 4):
                e = Int32((sc_lo >> (8 * t)) & 0xFF)
            else:
                e = Int32((sc_hi >> (8 * (t - 4))) & 0xFF)
            k = cutlass.min(e - 127, 15)
            kk[t] = k
            kmin = cutlass.min(kmin, k)
        two = cute.arch.vote_any_sync(kmin < -14)
        for t in cutlass.range_constexpr(7):
            k = kk[t]
            s1 = Uint32(cutlass.max(k + 15, 1) << 10)
            s1 = s1 | (s1 << 16)
            ob = cute.make_rmem_tensor(8, Uint32)
            if two:
                k2 = k + 14
                s2 = Uint32(0x3C00)
                if k < -14:
                    s2 = Uint32(0)
                    if k2 >= -14:
                        s2 = Uint32((k2 + 15) << 10)
                    elif k2 >= -24:
                        s2 = Uint32(1 << (k2 + 24))
                s2 = s2 | (s2 << 16)
                for j in cutlass.range_constexpr(4):
                    lo, hi = dq_e4m3x4_f16_2(raw[o + t * 4 + j], s1, s2)
                    ob[2 * j] = lo
                    ob[2 * j + 1] = hi
            else:
                for j in cutlass.range_constexpr(4):
                    lo, hi = dq_e4m3x4_f16(raw[o + t * 4 + j], s1)
                    ob[2 * j] = lo
                    ob[2 * j + 1] = hi
            self._st16(base, x0 + t * 8192, ob[0], ob[1], ob[2], ob[3], glob)
            self._st16(base, x1 + t * 8192, ob[4], ob[5], ob[6], ob[7], glob)
        for t in cutlass.range_constexpr(2):
            c0 = D_NOPE // 8 + ch + t * 4
            self._st16(
                base,
                self._swz(tk, c0) * 2,
                bf16x2_to_f16x2(raw[o + 28 + t * 4]),
                bf16x2_to_f16x2(raw[o + 29 + t * 4]),
                bf16x2_to_f16x2(raw[o + 30 + t * 4]),
                bf16x2_to_f16x2(raw[o + 31 + t * 4]),
                glob,
            )

    @cute.jit
    def _st16(self, base, off: Int32, a, b, c, d, glob: cutlass.Constexpr):
        if const_expr(glob):
            st_global_v4(base + Int64(off), a, b, c, d)
        else:
            st_shared_v4(base + off, a, b, c, d)


class SmallDecode(_Common):
    def __init__(self, H: int = 16, has_cu: bool = True):
        super().__init__(H, 1, 1)
        self.S = NSWB + (NBC if has_cu else 0)
        self.has_cu = has_cu

    @cute.jit
    def __call__(
        self,
        q_ptr: Int64,
        q_st: Int32,
        q_sh: Int32,
        o_ptr: Int64,
        o_st: Int32,
        o_sh: Int32,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_pshift: Int32,
        kidx_ptr: Int64,
        kmask_ptr: Int64,
        klen_ptr: Int64,
        cap: Int32,
        cu_img: Int64,
        cu_mbits: Int64,
        cu_cnt: Int64,
        sink_ptr: Int64,
        scale_log2: Float32,
        work_o: Int64,
        work_ml: Int64,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        sQ_layout = self._smem_layout(BM, D)
        sK_layout = self._smem_layout(BN, D, 1)
        sP_layout = self._smem_layout(BM, BN)
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(BM, BN),
        )
        tiled_mma_pv_rs = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(BM, DV // 2),
            a_source=warpgroup.OperandSource.RMEM,
        )
        tiled_mma_pv_ss = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(BM, DV // 2),
        )

        @cute.struct
        class SharedStorage:
            mbar_full: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_empty: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            sScale: cute.struct.MemRange[Float32, BM]
            sOScale: cute.struct.MemRange[Float32, BM]
            sMask: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 8 * 8], 16]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(sP_layout)], 1024
            ]

        self.kernel(
            q_ptr,
            q_st,
            q_sh,
            o_ptr,
            o_st,
            o_sh,
            swa_ptr,
            swa_bs,
            swa_pshift,
            kidx_ptr,
            kmask_ptr,
            klen_ptr,
            cap,
            cu_img,
            cu_mbits,
            cu_cnt,
            sink_ptr,
            scale_log2,
            work_o,
            work_ml,
            R,
            ngr,
            sQ_layout,
            sK_layout,
            sP_layout,
            tiled_mma_qk,
            tiled_mma_pv_rs,
            tiled_mma_pv_ss,
            SharedStorage,
        ).launch(
            grid=[self.S, G, 1],
            block=[384, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q_ptr: Int64,
        q_st: Int32,
        q_sh: Int32,
        o_ptr: Int64,
        o_st: Int32,
        o_sh: Int32,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_pshift: Int32,
        kidx_ptr: Int64,
        kmask_ptr: Int64,
        klen_ptr: Int64,
        cap: Int32,
        cu_img: Int64,
        cu_mbits: Int64,
        cu_cnt: Int64,
        sink_ptr: Int64,
        scale_log2: Float32,
        work_o: Int64,
        work_ml: Int64,
        R: Int32,
        ngr: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tiled_mma_pv_ss: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        H = self.H
        rpg = self.rpg
        SPL = self.S
        tidx, _, _ = cute.arch.thread_idx()
        split, g, _ = cute.arch.block_idx()
        wg = cute.arch.make_warp_uniform(tidx // 128)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        tw = tidx % 128
        cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        mbar_full = st.mbar_full.data_ptr()
        mbar_empty = st.mbar_empty.data_ptr()
        mbar_q = st.mbar_q.data_ptr()
        sQ = st.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = st.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sP = st.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sScale = st.sScale.get_tensor(cute.make_layout(BM))  # noqa: F841
        sOScale = st.sOScale.get_tensor(cute.make_layout(BM))
        sQ_addr = st.sQ.data_ptr().toint()
        sK_addr = st.sK.data_ptr().toint()
        sMask_addr = st.sMask.data_ptr().toint()
        full_addr = st.mbar_full.data_ptr().toint()

        if warp == 0:
            if lane == 0:
                cute.arch.mbarrier_init(mbar_full, 256)
                cute.arch.mbarrier_init(mbar_empty, 256)
                cute.arch.mbarrier_init(mbar_q, 256)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

        b = g // ngr
        kg = g - b * ngr
        tok0 = b * R + kg * rpg
        nrows = cutlass.min(rpg, R - kg * rpg)
        is_sw = split < NSWB
        cb = split - NSWB
        klen = ld_cg_b32(klen_ptr + Int64(g) * 4)
        cnt_g = Int32(0)
        if const_expr(self.has_cu):
            cnt_g = ld_cg_b32(cu_cnt + Int64(g) * 4)
        # SWA gather: WG2 owns keys 0..31 of the block, WG1 keys 32..63.
        wi = tw // 32
        ch = lane // 8
        tk = wi * 8 + lane % 8 + (wg % 2) * 32
        kbase = kidx_ptr + Int64(g) * Int64(cap) * 4
        pos = split * BN + tk
        vraw = Int32(-1)
        if is_sw and wg >= 1:
            vraw = ld_cg_b32(kbase + Int64(pos) * 4)
        if const_expr(self.has_cu):  # noqa: SIM102
            if not is_sw and tidx == 256:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_full, BN * D * 2)
                bulk_g2s(
                    sK_addr,
                    cu_img + Int64(g * NBC + cb) * Int64(IMG * 2),
                    Int32(BN * D * 2),
                    full_addr,
                )
        if tidx < 256:
            for it in cutlass.range_constexpr(BM * (D // 8) // 256):
                cid = it * 256 + tidx
                m = cid // (D // 8)
                c = cid % (D // 8)
                r = cutlass.min(m // H, nrows - 1)
                h = m % H
                ga = q_ptr + (Int64(tok0 + r) * q_st + Int64(h * q_sh + c * 8)) * 2
                w0, w1, w2, w3 = ld_cg_v4(ga)
                st_shared_v4(
                    sQ_addr + self._swz(m, c) * 2,
                    bf16x2_to_f16x2(w0),
                    bf16x2_to_f16x2(w1),
                    bf16x2_to_f16x2(w2),
                    bf16x2_to_f16x2(w3),
                )
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(mbar_q)
        valid = split * BN < klen
        if const_expr(self.has_cu):  # noqa: SIM102
            if not is_sw:
                valid = cb * BN < cnt_g

        if wg == 2:
            cute.arch.setmaxregister_decrease(152)
            wi = tw // 32
            tkb = wi * 8 + lane % 8  # noqa: F841
            ch = lane // 8
            if is_sw:
                if valid:
                    self._gather_half(
                        vraw,
                        pos,
                        klen,
                        tk,
                        ch,
                        wi,
                        0,
                        swa_ptr,
                        swa_bs,
                        swa_pshift,
                        kmask_ptr,
                        g,
                        cap,
                        sMask_addr,
                        sK_addr,
                        mbar_full,
                    )
            else:
                if const_expr(self.has_cu):
                    # cu_mbits: per union key a row-visibility byte [G, NBC*64]
                    if tw < BN:
                        u = cb * BN + tw
                        mb = Int32(0)
                        if u < cnt_g:
                            mb = ld_nc_u8(cu_mbits + Int64(g * NBC * BN + u))
                        for r in cutlass.range_constexpr(rpg):
                            bal = cute.arch.vote_ballot_sync(((mb >> r) & 1) != 0)
                            if lane == 0:
                                st_shared_b32(
                                    sMask_addr + (r * 2 + tw // 32) * 4, Uint32(bal)
                                )
                    if tw != 0:
                        cute.arch.mbarrier_arrive(mbar_full)
                    if not valid:
                        cute.arch.mbarrier_wait(mbar_full, 0)
        elif wg == 0:
            cute.arch.setmaxregister_increase(192)
            thr_qk = tiled_mma_qk.get_slice(tw)
            thr_pv = tiled_mma_pv_rs.get_slice(tw)
            cS = cute.make_identity_tensor((BM, BN))
            tScS = layout_utils.reshape_acc_to_mn(thr_qk.partition_C(cS))
            row_max = cute.make_rmem_tensor(2, Float32)
            row_sum = cute.make_rmem_tensor(2, Float32)
            row_max.fill(-1e30)
            row_sum.fill(0.0)
            acc_O = cute.make_rmem_tensor(
                thr_pv.partition_shape_C((BM, DV // 2)), Float32
            )
            acc_O.fill(0.0)
            if valid:
                tSrQ = thr_qk.make_fragment_A(thr_qk.partition_A(sQ))
                tSrK = thr_qk.make_fragment_B(thr_qk.partition_B(sK))
                sVt = layout_utils.transpose_view(sK)
                sVt0 = cute.local_tile(sVt, (DV // 2, BN, 1), (0, 0, 0))
                tOrV = thr_pv.make_fragment_B(thr_pv.partition_B(sVt0))
                stmatrix = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    self.dtype,
                )
                thr_cp_P = cute.make_tiled_copy_C(stmatrix, tiled_mma_qk).get_slice(tw)
                tPsP = thr_cp_P.partition_D(sP)
                cute.arch.mbarrier_wait(mbar_q, 0)
                cute.arch.mbarrier_wait(mbar_full, 0)
                acc_S = cute.make_rmem_tensor(
                    thr_qk.partition_shape_C((BM, BN)), Float32
                )
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK[None, None, None, 0],
                    zero_init=True,
                    wg_wait=0,
                )
                s_mn = layout_utils.reshape_acc_to_mn(acc_S)
                for r in cutlass.range_constexpr(2):
                    mx = Float32(-math.inf)
                    mw2 = ld_shared_v2(sMask_addr + (tScS[r, 0][0] // H) * 8)
                    sh = Uint32((lane % 4) * 2)
                    mlo = mw2[0] >> sh
                    mhi = mw2[1] >> sh
                    for c in cutlass.range_constexpr(cute.size(s_mn, mode=[1])):
                        sv = s_mn[r, c] * scale_log2
                        jb = (c % 2) + 8 * ((c // 2) % 4)
                        mword = mlo if const_expr(c // 2 < 4) else mhi
                        if (mword & Uint32(1 << jb)) == Uint32(0):
                            sv = -math.inf
                        s_mn[r, c] = sv
                        mx = cutlass.max(mx, sv)
                    mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=1))
                    mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=2))
                    new_max = cutlass.max(mx, Float32(-1e30))
                    row_max[r] = new_max
                    acc = Float32(0.0)
                    for c in cutlass.range_constexpr(cute.size(s_mn, mode=[1])):
                        p = cute.arch.exp2(s_mn[r, c] - new_max)
                        s_mn[r, c] = p
                        acc += p
                    acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1)
                    acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=2)
                    row_sum[r] = acc
                tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
                tOrP = cute.make_rmem_tensor(tOrP_acc.layout, self.dtype)
                tOrP_u32 = cute.recast_tensor(tOrP, Uint32)
                for k in cutlass.range_constexpr(cute.size(tOrP_u32)):
                    tOrP_u32[k] = cvt_f16x2(tOrP_acc[2 * k], tOrP_acc[2 * k + 1])
                cute.copy(thr_cp_P, thr_cp_P.retile(tOrP), tPsP)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(barrier_id=BAR_P_READY, number_of_threads=256)
                sm90_utils.gemm(
                    tiled_mma_pv_rs, acc_O, tOrP, tOrV[None, None, None, 0], wg_wait=0
                )
            osc = cute.make_rmem_tensor(2, Float32)
            for r in cutlass.range_constexpr(2):
                m = tScS[r, 0][0]
                o = Float32(0.0)
                if row_sum[r] > 0.0:
                    o = 1.0 / row_sum[r]
                osc[r] = o
                if lane % 4 == 0:
                    wa = work_ml + (Int64((g * SPL + split) * BM + m) * 2) * 4
                    st_global_f32(wa, row_max[r])
                    st_global_f32(wa + 4, row_sum[r])
                    sOScale[m] = o
            if valid:
                cute.arch.barrier_arrive(barrier_id=BAR_O_READY, number_of_threads=256)
                self._stage_p(acc_O, osc, thr_pv, 0, sQ_addr)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=BAR_P_FREE, number_of_threads=256)
                if tw == 0:
                    bulk_s2g(
                        work_o + Int64(g * SPL + split) * Int64(BM * DV * 2),
                        sQ_addr,
                        nrows * H * (DV * 2),
                    )
                    bulk_wait_read0()
        else:
            cute.arch.setmaxregister_decrease(160)
            if const_expr(self.has_cu):  # noqa: SIM102
                if not valid and not is_sw:
                    cute.arch.mbarrier_arrive(mbar_full)
            if valid:
                if is_sw:
                    self._gather_half(
                        vraw,
                        pos,
                        klen,
                        tk,
                        ch,
                        wi,
                        4,
                        swa_ptr,
                        swa_bs,
                        swa_pshift,
                        kmask_ptr,
                        g,
                        cap,
                        sMask_addr,
                        sK_addr,
                        mbar_full,
                    )
                else:
                    cute.arch.mbarrier_arrive(mbar_full)
                thr_pv = tiled_mma_pv_ss.get_slice(tw)
                tOrP = thr_pv.make_fragment_A(thr_pv.partition_A(sP))
                sVt = layout_utils.transpose_view(sK)
                sVt1 = cute.local_tile(sVt, (DV // 2, BN, 1), (1, 0, 0))
                tOrV = thr_pv.make_fragment_B(thr_pv.partition_B(sVt1))
                acc_O = cute.make_rmem_tensor(
                    thr_pv.partition_shape_C((BM, DV // 2)), Float32
                )
                acc_O.fill(0.0)
                cO = cute.make_identity_tensor((BM, DV // 2))
                tOcO = layout_utils.reshape_acc_to_mn(thr_pv.partition_C(cO))
                cute.arch.mbarrier_wait(mbar_full, 0)
                cute.arch.barrier(barrier_id=BAR_P_READY, number_of_threads=256)
                sm90_utils.gemm(
                    tiled_mma_pv_ss, acc_O, tOrP, tOrV[None, None, None, 0], wg_wait=0
                )
                cute.arch.barrier(barrier_id=BAR_O_READY, number_of_threads=256)
                osc = cute.make_rmem_tensor(2, Float32)
                for r in cutlass.range_constexpr(2):
                    osc[r] = sOScale[tOcO[r, 0][0]]
                self._stage_p(acc_O, osc, thr_pv, DV // 2, sQ_addr)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(barrier_id=BAR_P_FREE, number_of_threads=256)

    @cute.jit
    def _stage_p(
        self,
        acc_O: cute.Tensor,
        osc: cute.Tensor,
        thr_pv: cute.ThrMma,
        col0: cutlass.Constexpr,
        base: Int32,
    ):
        """Normalised bf16 O half -> row-contiguous [64, 512] smem tile, 16 B chunks XOR-swizzled by row."""  # noqa: E501
        cO = cute.make_identity_tensor((BM, DV // 2))
        tOcO = layout_utils.reshape_acc_to_mn(thr_pv.partition_C(cO))
        o_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(2):
            m = tOcO[r, 0][0]
            for c in cutlass.range_constexpr(cute.size(o_mn, mode=[1]) // 2):
                n = tOcO[r, 2 * c][1] + col0
                st_shared_b32(
                    base + (m * DV + (((n // 8) ^ (m % 8)) * 8) + n % 8) * 2,
                    cvt_bf16x2(o_mn[r, 2 * c] * osc[r], o_mn[r, 2 * c + 1] * osc[r]),
                )

    @cute.jit
    def _gather_half(
        self,
        vraw: Int32,
        pos: Int32,
        klen: Int32,
        tk: Int32,
        ch: Int32,
        wi: Int32,
        boff: cutlass.Constexpr,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_pshift: Int32,
        kmask_ptr: Int64,
        g: Int32,
        cap: Int32,
        sMask_addr: Int32,
        sK_addr: Int32,
        mbar_full,
    ):
        """One key per 4 threads: 32 keys of the SWA block (WG2: 0..31, WG1: 32..63)."""
        rpg = self.rpg
        lane = cute.arch.lane_idx()
        v = vraw
        if pos >= klen:
            v = Int32(-1)
        raw = cute.make_rmem_tensor(36, Uint32)
        scl = cute.make_rmem_tensor(2, Uint32)
        self._load_token(
            v, ch, raw, scl, 0, swa_ptr, swa_bs, swa_pshift, swa_ptr, swa_bs, swa_pshift
        )
        mk = Int32(0)
        if ch == 0 and v != -1:
            mk = ld_nc_u8(kmask_ptr + Int64(g) * Int64(cap) + Int64(pos))
        self._store_token_any(v != -1, tk, ch, sK_addr, raw, scl, 0, False)
        for r in cutlass.range_constexpr(rpg):
            bal = cute.arch.vote_ballot_sync(((mk >> r) & 1) != 0)
            if lane == 0:
                st_shared_u8(sMask_addr + r * 8 + wi + boff, bal & 0xFF)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(mbar_full)


class Combine:
    """4 warps per output row (128 columns each); partial slots with l == 0
    (invalid / empty blocks) get weight 0 and are not read."""

    def __init__(self, H: int = 16, has_cu: bool = True):
        self.H = H
        self.S = NSWB + (NBC if has_cu else 0)
        self.rpg = BM // H

    @cute.jit
    def __call__(
        self,
        work_o: Int64,
        work_ml: Int64,
        sink_ptr: Int64,
        o_ptr: Int64,
        o_st: Int32,
        o_sh: Int32,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            w: cute.struct.MemRange[Float32, 8 * 64]

        self.kernel(work_o, work_ml, sink_ptr, o_ptr, o_st, o_sh, R, ngr, Smem).launch(
            grid=[G * BM * 4 // 8, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        work_o: Int64,
        work_ml: Int64,
        sink_ptr: Int64,
        o_ptr: Int64,
        o_st: Int32,
        o_sh: Int32,
        R: Int32,
        ngr: Int32,
        Smem: cutlass.Constexpr,
    ):
        H, SPL, rpg = self.H, self.S, self.rpg
        tid, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        sw = st.w.get_tensor(cute.make_layout(8 * 64))
        gw = bx * 8 + warp
        q4 = gw % 4
        gr = gw // 4
        g = gr // BM
        m = gr % BM
        b = g // ngr
        kg = g - b * ngr
        nrows = cutlass.min(rpg, R - kg * rpg)
        rr = m // H
        sk = ld_f32(sink_ptr + Int64(m % H) * 4) * LOG2E
        cute.arch.griddepcontrol_wait()
        if rr < nrows:
            ms = cute.make_rmem_tensor(2, Float32)
            ls = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(2):
                s = lane + 32 * k
                mv = Float32(-1e30)
                lv = Float32(0.0)
                if s < SPL:
                    x0, x1 = ld_cg_v2(work_ml + (Int64((g * SPL + s) * BM + m) * 2) * 4)
                    mv = u32_as_f32(x0)
                    lv = u32_as_f32(x1)
                ms[k] = mv
                ls[k] = lv
            mx = Float32(-1e30)
            for k in cutlass.range_constexpr(2):
                if ls[k] > 0.0:
                    mx = cutlass.max(mx, ms[k])
            for o in cutlass.range_constexpr(5):
                mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=1 << o))
            den = Float32(0.0)
            for k in cutlass.range_constexpr(2):
                w = Float32(0.0)
                if ls[k] > 0.0:
                    w = cute.arch.exp2(ms[k] - mx) * ls[k]
                ms[k] = w
                den += w
            for o in cutlass.range_constexpr(5):
                den = den + cute.arch.shuffle_sync_bfly(den, offset=1 << o)
            inv = Float32(0.0)
            if den > 0.0:
                inv = 1.0 / (den + cute.arch.exp2(sk - mx))
            for k in cutlass.range_constexpr(2):
                sw[warp * 64 + lane + 32 * k] = ms[k] * inv
            cute.arch.sync_warp()
            col = q4 * 128 + lane * 4
            off = (m * DV + (((col // 8) ^ (m % 8)) * 8) + col % 8) * 2
            xs = cute.make_rmem_tensor(SPL * 2, Uint32)
            for s in cutlass.range_constexpr(SPL):
                a0 = Uint32(0)
                a1 = Uint32(0)
                if sw[warp * 64 + s] > 0.0:
                    a0, a1 = ld_cg_v2(
                        work_o + Int64(g * SPL + s) * Int64(BM * DV * 2) + Int64(off)
                    )
                xs[2 * s] = a0
                xs[2 * s + 1] = a1
            acc = cute.make_rmem_tensor(4, Float32)
            acc.fill(0.0)
            for s in cutlass.range_constexpr(SPL):
                w = sw[warp * 64 + s]
                for j in cutlass.range_constexpr(2):
                    lo, hi = bf16x2_unpack(xs[2 * s + j])
                    acc[2 * j] = acc[2 * j] + w * lo
                    acc[2 * j + 1] = acc[2 * j + 1] + w * hi
            orow = (
                o_ptr
                + (Int64(b * R + kg * rpg + rr) * o_st + Int64((m % H) * o_sh + col))
                * 2
            )
            st_global_v2(orow, cvt_bf16x2(acc[0], acc[1]), cvt_bf16x2(acc[2], acc[3]))

    @cute.jit
    def _swz(self, m: Int32, c: Int32) -> Int32:
        return (
            ((m // 8) + (c // 8) * (BM // 8)) * 512
            + (m % 8) * 64
            + (((c % 8) ^ (m % 8)) * 8)
        )


class CuPrep3(_Common):
    def __init__(self, H: int = 16):
        super().__init__(H, 1, 1)

    @cute.jit
    def __call__(
        self,
        topk: Int64,
        topk_st: Int32,
        topk_w: Int32,
        lens: Int64,
        tok2req: Int64,
        valid_tok: Int64,
        btab: Int64,
        btab_st: Int32,
        cps: Int32,
        cache: Int64,
        cache_bs: Int64,
        pshift: Int32,
        img: Int64,
        cmask: Int64,
        cnt: Int64,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            keys: cute.struct.MemRange[Int32, TBL]
            rb: cute.struct.MemRange[cutlass.Uint8, TBL * 4]
            wsum: cute.struct.MemRange[Int32, 16]
            upos: cute.struct.MemRange[Int32, BN]
            umask: cute.struct.MemRange[Int32, BN]

        self.kernel(
            topk,
            topk_st,
            topk_w,
            lens,
            tok2req,
            valid_tok,
            btab,
            btab_st,
            cps,
            cache,
            cache_bs,
            pshift,
            img,
            cmask,
            cnt,
            R,
            ngr,
            Smem,
        ).launch(
            grid=[self.rpg * 512 // BN, G, 1],
            block=[512, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        topk: Int64,
        topk_st: Int32,
        topk_w: Int32,
        lens: Int64,
        tok2req: Int64,
        valid_tok: Int64,
        btab: Int64,
        btab_st: Int32,
        cps: Int32,
        cache: Int64,
        cache_bs: Int64,
        pshift: Int32,
        img: Int64,
        cmask: Int64,
        cnt: Int64,
        R: Int32,
        ngr: Int32,
        Smem: cutlass.Constexpr,
    ):
        rpg = self.rpg
        CPT = rpg
        tid, _, _ = cute.arch.thread_idx()
        cc, g, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        keys = st.keys.get_tensor(cute.make_layout(TBL))
        rb = st.rb.get_tensor(cute.make_layout(TBL * 4))
        rbw = cute.recast_tensor(rb, Int32)
        wsum = st.wsum.get_tensor(cute.make_layout(16))
        upos = st.upos.get_tensor(cute.make_layout(BN))
        umask = st.umask.get_tensor(cute.make_layout(BN))
        for i in cutlass.range_constexpr(TBL // 512):
            keys[i * 512 + tid] = Int32(EMPTY)
            rbw[i * 512 + tid] = Int32(0)
        if tid < BN:
            upos[tid] = Int32(-1)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

        b = g // ngr
        kg = g - b * ngr
        # ---- candidate slots: thread owns candidates tid*CPT .. +CPT-1 ----
        r = (tid * CPT) // 512
        j = kg * rpg + r
        t = b * R + cutlass.min(j, R - 1)
        vv = self._cands(
            topk,
            topk_st,
            topk_w,
            lens,
            tok2req,
            valid_tok,
            btab,
            btab_st,
            cps,
            t,
            j < R,
            tid * CPT,
        )
        # ---- this CTA's 64 own candidates: c = cc*64 + key, 4 threads per key (tid < 256) ----  # noqa: E501
        wi = tid // 32
        tk = wi * 8 + lane % 8
        ch = lane // 8
        oc = cc * BN + tk
        orow = oc // 512
        oj = kg * rpg + orow
        out_tok = b * R + cutlass.min(oj, R - 1)
        ov = Int32(-1)
        raw = cute.make_rmem_tensor(72, Uint32)
        scl = cute.make_rmem_tensor(4, Uint32)
        if tid < 256:
            ovr = self._cands(
                topk,
                topk_st,
                topk_w,
                lens,
                tok2req,
                valid_tok,
                btab,
                btab_st,
                cps,
                out_tok,
                oj < R,
                oc,
            )
            ov = ovr[0]
            self._load_token(
                ov, ch, raw, scl, 0, cache, cache_bs, pshift, cache, cache_bs, pshift
            )
        # ---- union (redundant per CTA, deterministic) ----
        hh = cute.make_rmem_tensor(CPT, Int32)
        for k in cutlass.range_constexpr(CPT):
            h = Int32(0)
            if vv[k] >= 0:
                h = self._insert3(keys, vv[k])
                rb[h * 4 + r] = cutlass.Uint8(1)
            hh[k] = h
        cute.arch.sync_threads()
        rep = cute.make_rmem_tensor(CPT, Int32)
        c = Int32(0)
        for k in cutlass.range_constexpr(CPT):
            x = Int32(0)
            if vv[k] >= 0:  # noqa: SIM102
                if (rbw[hh[k]] & ((1 << (8 * r)) - 1)) == 0:
                    x = Int32(1)
            rep[k] = x
            c += x
        incl = c
        for o in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_up(incl, offset=1 << o, mask_and_clamp=0)
            if lane >= (1 << o):
                incl += y
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        woff = Int32(0)
        tot = Int32(0)
        for w in cutlass.range_constexpr(16):
            s = wsum[w]
            if w < warp:
                woff += s
            tot += s
        pos = woff + incl - c
        for k in cutlass.range_constexpr(CPT):
            if rep[k] == 1:
                lc = tid * CPT + k - cc * BN
                if lc >= 0 and lc < BN:
                    w4b = rbw[hh[k]]
                    mbit = Int32(0)
                    for rr in cutlass.range_constexpr(rpg):
                        mbit = mbit | (((w4b >> (8 * rr)) & 1) << rr)
                    upos[lc] = pos
                    umask[lc] = mbit
                pos += 1
        if cc == 0 and tid == 0:
            st_global_b32(cnt + Int64(g) * 4, Uint32(tot))
        cute.arch.sync_threads()
        if tid < 256:
            u = upos[tk]
            if u >= 0:
                base = img + Int64(g * NBC + u // BN) * Int64(IMG * 2)
                self._store_token_any(True, u % BN, ch, base, raw, scl, 0, True)
                if ch == 0:
                    st_global_u8(cmask + Int64(g * NBC * BN + u), umask[tk])

    @cute.jit
    def _cands(
        self,
        topk,
        topk_st,
        topk_w,
        lens,
        tok2req,
        valid_tok,
        btab,
        btab_st,
        cps,
        t,
        okrow,
        c0,
    ):
        """Global slot ids of candidates c0 .. c0+CPT-1 of token t (-1 if none)."""
        CPT = self.rpg
        vv = cute.make_rmem_tensor(CPT, Int32)
        for k in cutlass.range_constexpr(CPT):
            p = (c0 + k) % 512
            v = Int32(-1)
            if p < topk_w:
                v = ld_nc_b32(topk + (Int64(t) * topk_st + Int64(p)) * 4)
            vv[k] = v
        req = ld_nc_b32(tok2req + Int64(t) * 4)
        okt = ld_nc_b32(valid_tok + Int64(t // 4) * 4)
        okb = (okt >> ((t % 4) * 8)) & 0xFF
        for k in cutlass.range_constexpr(CPT):
            v = vv[k]
            if v >= 0 and okrow and okb != 0:
                blk = ld_nc_b32(btab + (Int64(req) * btab_st + Int64(v >> cps)) * 4)
                v = (blk << cps) | (v & ((1 << cps) - 1))
            else:
                v = Int32(-1)
            vv[k] = v
        return vv

    @cute.jit
    def _insert3(self, keys, v: Int32) -> Int32:
        h = Int32((cutlass.Uint32(v) * cutlass.Uint32(2654435761)) >> (32 - 12))
        done = False
        while not done:
            old = cute.arch.atomic_cas(
                keys.iterator + h, cmp=Int32(EMPTY), val=v, scope="cta"
            )
            if old == EMPTY or old == v:  # noqa: SIM109
                done = True
            else:
                h = (h + 1) % TBL
        return h


class UnionBuild:
    def __init__(self, rpg: int):
        self.rpg = rpg

    @cute.jit
    def __call__(
        self,
        swa_idx: Int64,
        swa_st: Int32,
        swa_w: Int32,
        swa_len: Int64,
        ex_idx: Int64,
        ex_st: Int32,
        ex_w: Int32,
        ex_len: Int64,
        kidx: Int64,
        kmask: Int64,
        klen: Int64,
        cap: Int32,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            keys: cute.struct.MemRange[Int32, UNION_TBL]
            bits: cute.struct.MemRange[Int32, UNION_TBL]
            wsum: cute.struct.MemRange[Int32, UNION_NT // 32]

        self.kernel(
            swa_idx,
            swa_st,
            swa_w,
            swa_len,
            ex_idx,
            ex_st,
            ex_w,
            ex_len,
            kidx,
            kmask,
            klen,
            cap,
            R,
            ngr,
            Smem,
        ).launch(grid=[G, 1, 1], block=[UNION_NT, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        swa_idx: Int64,
        swa_st: Int32,
        swa_w: Int32,
        swa_len: Int64,
        ex_idx: Int64,
        ex_st: Int32,
        ex_w: Int32,
        ex_len: Int64,
        kidx: Int64,
        kmask: Int64,
        klen: Int64,
        cap: Int32,
        R: Int32,
        ngr: Int32,
        Smem: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        g, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        keys = st.keys.get_tensor(cute.make_layout(UNION_TBL))
        bits = st.bits.get_tensor(cute.make_layout(UNION_TBL))
        wsum = st.wsum.get_tensor(cute.make_layout(UNION_NT // 32))
        for i in cutlass.range_constexpr(UNION_TBL // UNION_NT):
            keys[i * UNION_NT + tid] = Int32(EMPTY)
            bits[i * UNION_NT + tid] = Int32(0)
        cute.arch.sync_threads()

        b = g // ngr
        kg = g - b * ngr
        # All loads of the group's rows issued up front (one latency): each
        # thread owns entry `tid` of every row's SWA list and entry `tid` of
        # every row's compressed list (widths <= UNION_NT).
        vs = cute.make_rmem_tensor(self.rpg, Int32)
        for r in cutlass.range_constexpr(self.rpg):
            j = kg * self.rpg + r
            t = b * R + cutlass.min(j, R - 1)
            n = ld_nc_b32(swa_len + Int64(t) * 4)
            v = Int32(-1)
            if j < R and tid < n and tid < swa_w:
                v = ld_nc_b32(swa_idx + (Int64(t) * swa_st + Int64(tid)) * 4)
            vs[r] = v
        for r in cutlass.range_constexpr(self.rpg):
            if vs[r] >= 0:
                self._insert(keys, bits, vs[r], 1 << r)
        cute.arch.sync_threads()

        # Compaction: each thread owns UNION_TBL // UNION_NT consecutive slots.
        per = UNION_TBL // UNION_NT
        cnt = Int32(0)
        for i in cutlass.range_constexpr(per):
            if keys[tid * per + i] != EMPTY:
                cnt += 1
        incl = cnt
        for o in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_up(incl, offset=1 << o, mask_and_clamp=0)
            if lane >= (1 << o):
                incl += y
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        woff = Int32(0)
        tot = Int32(0)
        for w in cutlass.range_constexpr(UNION_NT // 32):
            s = wsum[w]
            if w < warp:
                woff += s
            tot += s
        pos = woff + incl - cnt
        for i in cutlass.range_constexpr(per):
            k = keys[tid * per + i]
            if k != EMPTY:
                if pos < cap:
                    st_i32(kidx + (Int64(g) * cap + Int64(pos)) * 4, k)
                    st_u8(kmask + Int64(g) * cap + Int64(pos), bits[tid * per + i])
                pos += 1
        if tid == 0:
            st_i32(klen + Int64(g) * 4, cutlass.min(tot, cap))

    @cute.jit
    def _insert(self, keys: cute.Tensor, bits: cute.Tensor, v: Int32, bit: Int32):
        h = Int32((cutlass.Uint32(v) * cutlass.Uint32(2654435761)) >> (32 - 13))
        done = False
        while not done:
            old = cute.arch.atomic_cas(
                keys.iterator + h, cmp=Int32(EMPTY), val=v, scope="cta"
            )
            if old == EMPTY or old == v:  # noqa: SIM109
                cute.arch.atomic_or(bits.iterator + h, Int32(bit), scope="cta")
                done = True
            else:
                h = (h + 1) % UNION_TBL
