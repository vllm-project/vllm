# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels of the Hopper small-batch sparse-MLA decode
(``mla_small``), on the FlashMLA FP8 caches (584-byte V4 records). A row group
is up to 64 // H verify rows of one request, whose H heads each share one
64-row WGMMA tile.

- ``UnionBuild``: per row group, the union of its rows' SWA slots (up to
  ``NSWB`` 64-key blocks) and a per-key row mask.
- ``SwaImage``: the SWA union's keys dequantized into fp16 tile images.
- ``CuPrep3``: per row group, the union of its rows' compressed top-k slots (up
  to ``NBC`` blocks), optionally mapped from the indexer's local indices
  through the compressed block table, dequantized into fp16 tile images, with
  a per-key visibility byte.
- ``ChainDecode``: one cluster per row group over the tile images, from the
  ``wq_b`` output to the inverse-rotated ``wo_a`` input.

They launch with programmatic dependent launch: each releases its dependents
at entry and waits for its predecessor before reading its outputs."""

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
    cvt_bf16x2,
    cvt_f16x2,
    dq_e4m3x4_f16,
    dq_e4m3x4_f16_2,
    ld_cg_b32,
    ld_cg_v4,
    ld_dsmem_v2f,
    ld_dsmem_v4,
    ld_f32,
    ld_nc_b32,
    ld_nc_i64,
    ld_nc_u8,
    ld_nc_v2,
    ld_nc_v4,
    ld_nc_v4f,
    ld_shared_v2,
    mapa,
    st_global_b32,
    st_global_u8,
    st_global_v4,
    st_i32,
    st_shared_b32,
    st_shared_v2f,
    st_shared_v4,
    st_u8,
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
BAR_Q_FREE = 4
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
        keep=True,
    ):
        """Dequantise one key into row tk of a SW128 K-major fp16 tile at
        `base` (smem address, or global address if `glob`); with `keep`
        false the row is left unwritten."""
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
            if keep:
                self._st16(base, x0 + t * 8192, ob[0], ob[1], ob[2], ob[3], glob)
                self._st16(base, x1 + t * 8192, ob[4], ob[5], ob[6], ob[7], glob)
        for t in cutlass.range_constexpr(2):
            c0 = D_NOPE // 8 + ch + t * 4
            if keep:
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


class ChainDecode(_Common):
    """The sparse MLA of an ``attn_chain`` step in one kernel, writing the
    ``wo_a`` input rot[t * rot_t + (h // 8) * rot_g + (h % 8) * 512 + col].

    Every key block is a tile image loaded with one bulk copy: the SWA
    union's (``SwaImage``, plus the step's own keys from
    ``attn_cute.QkvNorm``) and the compressed stage's (``CuPrep3``). The
    query, the ``wq_b`` output [T, H, 512], is rotated (GPT-J RoPE on dims
    448..511, rounded to BF16) as it is staged.

    A cluster of C CTAs per row group; CTA c takes the group's tasks c, c + C,
    ... (its SWA blocks, then its compressed blocks), keeping one online
    softmax over them, the next task's tile landing during the current one.
    The CTAs then exchange their per-row (max, sum) over DSMEM, stage their
    output scaled to the merged softmax (sink included) in shared memory, and
    each sums a 512 / C column slice over the cluster, rounds it to BF16 and
    applies the inverse RoPE. With ``early`` set the compressed stage was
    written before this layer, so its count is read before the wait."""

    def __init__(self, H: int = 16, has_cu: bool = True, C: int = 8):
        super().__init__(H, 1, 1)
        self.has_cu = has_cu
        self.C = C

    @cute.jit
    def __call__(
        self,
        q_ptr: Int64,
        q_st: Int32,
        pos_ptr: Int64,
        cs_ptr: Int64,
        sw_img: Int64,
        kmask_ptr: Int64,
        klen_ptr: Int64,
        cap: Int32,
        cu_img: Int64,
        cu_mbits: Int64,
        cu_cnt: Int64,
        scale_log2: Float32,
        sink_ptr: Int64,
        rot: Int64,
        rot_t: Int32,
        rot_g: Int32,
        R: Int32,
        ngr: Int32,
        G: Int32,
        early: Int32,
        stream: cuda.CUstream,
    ):
        sQ_layout = self._smem_layout(BM, D)
        sK_layout = self._smem_layout(BN, D, 2)
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
            mbar_full: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_empty: cute.struct.MemRange[cutlass.Int64, 2]
            sScale: cute.struct.MemRange[Float32, BM]
            sML: cute.struct.Align[cute.struct.MemRange[Float32, BM * 2], 16]
            sCS: cute.struct.Align[cute.struct.MemRange[Float32, self.rpg * 64], 16]
            sSink: cute.struct.MemRange[Float32, self.H]
            sMask: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 8 * 8 * 2], 16]
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
            pos_ptr,
            cs_ptr,
            sw_img,
            kmask_ptr,
            klen_ptr,
            cap,
            cu_img,
            cu_mbits,
            cu_cnt,
            scale_log2,
            sink_ptr,
            rot,
            rot_t,
            rot_g,
            R,
            ngr,
            early,
            sQ_layout,
            sK_layout,
            sP_layout,
            tiled_mma_qk,
            tiled_mma_pv_rs,
            tiled_mma_pv_ss,
            SharedStorage,
        ).launch(
            grid=[self.C, G, 1],
            block=[384, 1, 1],
            cluster=[self.C, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q_ptr: Int64,
        q_st: Int32,
        pos_ptr: Int64,
        cs_ptr: Int64,
        sw_img: Int64,
        kmask_ptr: Int64,
        klen_ptr: Int64,
        cap: Int32,
        cu_img: Int64,
        cu_mbits: Int64,
        cu_cnt: Int64,
        scale_log2: Float32,
        sink_ptr: Int64,
        rot: Int64,
        rot_t: Int32,
        rot_g: Int32,
        R: Int32,
        ngr: Int32,
        early: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tiled_mma_pv_ss: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        rpg, C = self.rpg, self.C
        tidx, _, _ = cute.arch.thread_idx()
        cid, g, _ = cute.arch.block_idx()
        wg = cute.arch.make_warp_uniform(tidx // 128)
        lane = cute.arch.lane_idx()
        tw = tidx % 128
        cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(SharedStorage)
        mbar_full = st.mbar_full.data_ptr()
        mbar_q = st.mbar_q.data_ptr()
        mbar_empty = st.mbar_empty.data_ptr()
        sQ = st.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = st.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sP = st.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sScale = st.sScale.get_tensor(cute.make_layout(BM))
        sQ_addr = st.sQ.data_ptr().toint()
        sK_addr = st.sK.data_ptr().toint()
        sMask_addr = st.sMask.data_ptr().toint()
        sML_addr = st.sML.data_ptr().toint()
        sCS = st.sCS.get_tensor(cute.make_layout(rpg * 64))
        sSink = st.sSink.get_tensor(cute.make_layout(self.H))
        full_addr = st.mbar_full.data_ptr().toint()

        if tidx == 0:
            for b in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_full + b, 3)
                cute.arch.mbarrier_init(mbar_empty + b, 256)
            cute.arch.mbarrier_init(mbar_q, 256)
            cute.arch.mbarrier_init_fence()
        c = tidx % (D // 8)
        rope = Int32(0)
        if tidx < 256:  # noqa: SIM102
            if c >= D_NOPE // 8:
                rope = Int32(1)
        cv = cute.make_rmem_tensor(rpg * 8, Float32)
        # The group's task counts are static, the compressed one with early.
        kl = ld_cg_b32(klen_ptr + Int64(g) * 4)
        nsw = (kl + (BN - 1)) // BN
        cn = Int32(0)
        if const_expr(self.has_cu) and early != 0:
            cn = ld_cg_b32(cu_cnt + Int64(g) * 4)
        if rope != 0:
            self._rope_coef(cv, g, pos_ptr, cs_ptr, c, R, ngr)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        if const_expr(self.has_cu) and early == 0:
            cn = ld_cg_b32(cu_cnt + Int64(g) * 4)
        ntask = nsw + (cn + (BN - 1)) // BN
        mine = Int32(0)
        if cid < ntask:
            mine = (ntask - cid + C - 1) // C
        b_ = g // ngr
        nrows = cutlass.min(rpg, R - (g - b_ * ngr) * rpg)
        if wg == 2:
            cute.arch.setmaxregister_decrease(152)
            # The merge's cos / sin rows and sinks, off the critical path.
            if tw >= 32:
                b_ = g // ngr
                t0 = b_ * R + (g - b_ * ngr) * rpg
                for k in cutlass.range_constexpr(-(-(rpg * 64) // 96)):
                    e = k * 96 + tw - 32
                    if e < rpg * 64:
                        t = t0 + cutlass.min(e // 64, nrows - 1)
                        pp = ld_nc_i64(pos_ptr + Int64(t) * 8)
                        sCS[e] = ld_f32(cs_ptr + (pp * Int64(64) + Int64(e % 64)) * 4)
                if tw - 32 < self.H:
                    sSink[tw - 32] = ld_f32(sink_ptr + Int64(tw - 32) * 4) * LOG2E
            for i in cutlass.range(mine, unroll=1):
                b = i % 2
                j = cid + i * C
                issw = Int32(1)
                lim = kl
                if j >= nsw:
                    issw = Int32(0)
                    j = j - nsw
                    lim = cn
                if i >= 2:
                    cute.arch.mbarrier_wait(mbar_empty + b, (i // 2 - 1) % 2)
                if tidx == 256:
                    self._issue(
                        g,
                        j,
                        issw,
                        sw_img,
                        cu_img,
                        sK_addr + b * (BN * D * 2),
                        full_addr + b * 8,
                        mbar_full + b,
                    )
                if tw < BN:
                    u = j * BN + tw
                    mb = Int32(0)
                    if u < lim:
                        if issw != 0:
                            mb = ld_nc_u8(kmask_ptr + Int64(g) * Int64(cap) + Int64(u))
                        else:
                            mb = ld_nc_u8(cu_mbits + Int64(g * NBC * BN + u))
                    for r in cutlass.range_constexpr(rpg):
                        bal = cute.arch.vote_ballot_sync(((mb >> r) & 1) != 0)
                        if lane == 0:
                            st_shared_b32(
                                sMask_addr + b * 64 + (r * 2 + tw // 32) * 4,
                                Uint32(bal),
                            )
                    if lane == 0:
                        cute.arch.mbarrier_arrive(mbar_full + b)
            self._cluster_sync()
        elif wg == 0:
            cute.arch.setmaxregister_increase(192)
            self._stage_q(g, q_ptr, q_st, cv, rope, c, R, ngr, sQ_addr, mbar_q)
            self._wg0(
                mine,
                scale_log2,
                sQ,
                sK,
                sP,
                sScale,
                sMask_addr,
                sML_addr,
                mbar_q,
                mbar_full,
                mbar_empty,
                tiled_mma_qk,
                tiled_mma_pv_rs,
                sK_addr,
            )
        else:
            cute.arch.setmaxregister_decrease(160)
            self._stage_q(g, q_ptr, q_st, cv, rope, c, R, ngr, sQ_addr, mbar_q)
            thr_pv = tiled_mma_pv_ss.get_slice(tw)
            tOrP = thr_pv.make_fragment_A(thr_pv.partition_A(sP))
            sVt = layout_utils.transpose_view(sK)
            sVt1 = cute.local_tile(sVt, (DV // 2, BN, 2), (1, 0, 0))
            tOrV = thr_pv.make_fragment_B(thr_pv.partition_B(sVt1))
            acc_O = cute.make_rmem_tensor(
                thr_pv.partition_shape_C((BM, DV // 2)), Float32
            )
            acc_O.fill(0.0)
            cO = cute.make_identity_tensor((BM, DV // 2))
            tOcO = layout_utils.reshape_acc_to_mn(thr_pv.partition_C(cO))
            o_mn = layout_utils.reshape_acc_to_mn(acc_O)
            for i in cutlass.range(mine, unroll=1):
                b = i % 2
                cute.arch.mbarrier_wait(mbar_full + b, (i // 2) % 2)
                cute.arch.barrier(barrier_id=BAR_P_READY, number_of_threads=256)
                for r in cutlass.range_constexpr(2):
                    al = sScale[tOcO[r, 0][0]]
                    for cc in cutlass.range_constexpr(cute.size(o_mn, mode=[1])):
                        o_mn[r, cc] = o_mn[r, cc] * al
                sm90_utils.gemm(
                    tiled_mma_pv_ss, acc_O, tOrP, tOrV[None, None, None, b], wg_wait=0
                )
                cute.arch.mbarrier_arrive(mbar_empty + b)
                if i + 1 < mine:
                    cute.arch.barrier_arrive(
                        barrier_id=BAR_P_FREE, number_of_threads=256
                    )
            # WG0's 1 / sum of this CTA's rows.
            cute.arch.barrier(barrier_id=BAR_O_READY, number_of_threads=256)
            f = cute.make_rmem_tensor(2, Float32)
            for r in cutlass.range_constexpr(2):
                f[r] = sScale[tOcO[r, 0][0]]
            self._stage_out(acc_O, thr_pv, DV // 2, f, sK_addr)
            self._cluster_sync()
        # Every CTA's output is staged: merge the column slices.
        self._merge(
            cid,
            g,
            nrows,
            sK_addr,
            sML_addr,
            sSink,
            sCS,
            rot,
            rot_t,
            rot_g,
            R,
            ngr,
        )
        # Peers read this CTA's shared memory until they pass here.
        self._cluster_sync()

    @cute.jit
    def _cluster_sync(self):
        """A cluster barrier (release / acquire), reached by every thread,
        from any role."""
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _wg0(
        self,
        mine,
        scale_log2,
        sQ,
        sK,
        sP,
        sScale,
        sMask_addr,
        sML_addr,
        mbar_q,
        mbar_full,
        mbar_empty,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        sK_addr,
    ):
        """QK, the online masked softmax and the left PV half over this CTA's
        tasks; then this CTA's (max, sum) per row and its locally normalized
        left output half staged for the cluster merge."""
        H = self.H
        tw = cute.arch.thread_idx()[0] % 128
        lane = cute.arch.lane_idx()
        thr_qk = tiled_mma_qk.get_slice(tw)
        thr_pv = tiled_mma_pv_rs.get_slice(tw)
        cS = cute.make_identity_tensor((BM, BN))
        tScS = layout_utils.reshape_acc_to_mn(thr_qk.partition_C(cS))
        row_max = cute.make_rmem_tensor(2, Float32)
        row_sum = cute.make_rmem_tensor(2, Float32)
        row_max.fill(-1e30)
        row_sum.fill(0.0)
        acc_O = cute.make_rmem_tensor(thr_pv.partition_shape_C((BM, DV // 2)), Float32)
        acc_O.fill(0.0)
        o_mn = layout_utils.reshape_acc_to_mn(acc_O)
        tSrQ = thr_qk.make_fragment_A(thr_qk.partition_A(sQ))
        tSrK = thr_qk.make_fragment_B(thr_qk.partition_B(sK))
        sVt = layout_utils.transpose_view(sK)
        sVt0 = cute.local_tile(sVt, (DV // 2, BN, 2), (0, 0, 0))
        tOrV = thr_pv.make_fragment_B(thr_pv.partition_B(sVt0))
        stmatrix = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        thr_cp_P = cute.make_tiled_copy_C(stmatrix, tiled_mma_qk).get_slice(tw)
        tPsP = thr_cp_P.partition_D(sP)
        alpha = cute.make_rmem_tensor(2, Float32)
        cute.arch.mbarrier_wait(mbar_q, 0)
        for i in cutlass.range(mine, unroll=1):
            b = i % 2
            cute.arch.mbarrier_wait(mbar_full + b, (i // 2) % 2)
            acc_S = cute.make_rmem_tensor(thr_qk.partition_shape_C((BM, BN)), Float32)
            sm90_utils.gemm(
                tiled_mma_qk,
                acc_S,
                tSrQ,
                tSrK[None, None, None, b],
                zero_init=True,
                wg_wait=0,
            )
            s_mn = layout_utils.reshape_acc_to_mn(acc_S)
            for r in cutlass.range_constexpr(2):
                mx = Float32(-math.inf)
                mw2 = ld_shared_v2(sMask_addr + b * 64 + (tScS[r, 0][0] // H) * 8)
                sh = Uint32((lane % 4) * 2)
                mlo = mw2[0] >> sh
                mhi = mw2[1] >> sh
                for cc in cutlass.range_constexpr(cute.size(s_mn, mode=[1])):
                    sv = s_mn[r, cc] * scale_log2
                    jb = (cc % 2) + 8 * ((cc // 2) % 4)
                    mword = mlo if const_expr(cc // 2 < 4) else mhi
                    if (mword & Uint32(1 << jb)) == Uint32(0):
                        sv = -math.inf
                    s_mn[r, cc] = sv
                    mx = cutlass.max(mx, sv)
                mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=1))
                mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=2))
                new_max = cutlass.max(row_max[r], mx)
                alpha[r] = cute.arch.exp2(row_max[r] - new_max)
                row_max[r] = new_max
                acc = Float32(0.0)
                for cc in cutlass.range_constexpr(cute.size(s_mn, mode=[1])):
                    p = cute.arch.exp2(s_mn[r, cc] - new_max)
                    s_mn[r, cc] = p
                    acc += p
                acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1)
                acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=2)
                row_sum[r] = row_sum[r] * alpha[r] + acc
            tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
            tOrP = cute.make_rmem_tensor(tOrP_acc.layout, self.dtype)
            tOrP_u32 = cute.recast_tensor(tOrP, Uint32)
            for k in cutlass.range_constexpr(cute.size(tOrP_u32)):
                tOrP_u32[k] = cvt_f16x2(tOrP_acc[2 * k], tOrP_acc[2 * k + 1])
            if i > 0:
                cute.arch.barrier(barrier_id=BAR_P_FREE, number_of_threads=256)
            for r in cutlass.range_constexpr(2):
                if lane % 4 == 0:
                    sScale[tScS[r, 0][0]] = alpha[r]
            cute.copy(thr_cp_P, thr_cp_P.retile(tOrP), tPsP)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(barrier_id=BAR_P_READY, number_of_threads=256)
            for r in cutlass.range_constexpr(2):
                for cc in cutlass.range_constexpr(cute.size(o_mn, mode=[1])):
                    o_mn[r, cc] = o_mn[r, cc] * alpha[r]
            sm90_utils.gemm(
                tiled_mma_pv_rs, acc_O, tOrP, tOrV[None, None, None, b], wg_wait=0
            )
            cute.arch.mbarrier_arrive(mbar_empty + b)
        # This CTA's (max, sum) and normalized output per row, for the merge.
        for r in cutlass.range_constexpr(2):
            o = Float32(0.0)
            if row_sum[r] > 0.0:
                o = 1.0 / row_sum[r]
            alpha[r] = o
            if lane % 4 == 0:
                m = tScS[r, 0][0]
                st_shared_v2f(sML_addr + m * 8, row_max[r], row_sum[r])
                sScale[m] = o
        # With WG1 done reading the key tiles, both stage into them.
        cute.arch.barrier(barrier_id=BAR_O_READY, number_of_threads=256)
        self._stage_out(acc_O, thr_pv, 0, alpha, sK_addr)
        self._cluster_sync()

    @cute.jit
    def _stage_out(self, acc_O, thr_pv: cute.ThrMma, col0: cutlass.Constexpr, f, base):
        """This warpgroup's output half scaled by the merged factor, BF16,
        row-major [64, 512] with 16 B chunks XOR-swizzled by row, at ``base``
        (the key tiles, free once the PV is done)."""
        cO = cute.make_identity_tensor((BM, DV // 2))
        tOcO = layout_utils.reshape_acc_to_mn(thr_pv.partition_C(cO))
        o_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(2):
            m = tOcO[r, 0][0]
            fr = f[r]
            for cc in cutlass.range_constexpr(cute.size(o_mn, mode=[1]) // 2):
                n = tOcO[r, 2 * cc][1] + col0
                st_shared_b32(
                    base + (m * DV + (((n // 8) ^ (m % 8)) * 8) + n % 8) * 2,
                    cvt_bf16x2(o_mn[r, 2 * cc] * fr, o_mn[r, 2 * cc + 1] * fr),
                )

    @cute.jit
    def _merge(
        self,
        cid,
        g,
        nrows,
        base,
        ml,
        sSink,
        sCS,
        rot,
        rot_t,
        rot_g,
        R,
        ngr,
    ):
        """Column slice ``cid`` of the group's output: the cluster's staged
        normalized outputs weighted by their share of the merged softmax (the
        sink included), rounded to BF16, inverse-rotated on dims 448..511,
        into the wo_a input."""
        H, rpg, C = self.H, self.rpg, self.C
        W = DV // C
        NCH = W // 8
        tidx = cute.arch.thread_idx()[0]
        b_ = g // ngr
        tok0 = b_ * R + (g - b_ * ngr) * rpg
        for it in cutlass.range_constexpr(-(-(BM * NCH) // 384)):
            item = it * 384 + tidx
            m = item // NCH
            if item < BM * NCH and m < nrows * H:
                col = cid * W + (item % NCH) * 8
                off = (m * DV + (((col // 8) ^ (m % 8)) * 8)) * 2
                # Every peer's (max, sum) and slice in one DSMEM round trip.
                ms = cute.make_rmem_tensor(C, Float32)
                ls = cute.make_rmem_tensor(C, Float32)
                xs = cute.make_rmem_tensor(C * 4, Uint32)
                for p in cutlass.range_constexpr(C):
                    ms[p], ls[p] = ld_dsmem_v2f(mapa(ml + m * 8, Int32(p)))
                    w0, w1, w2, w3 = ld_dsmem_v4(mapa(base + off, Int32(p)))
                    xs[4 * p], xs[4 * p + 1], xs[4 * p + 2], xs[4 * p + 3] = (
                        w0,
                        w1,
                        w2,
                        w3,
                    )
                sk = sSink[m % H]
                mx = sk
                for p in cutlass.range_constexpr(C):
                    if ls[p] > 0.0:
                        mx = cutlass.max(mx, ms[p])
                den = cute.arch.exp2(sk - mx)
                for p in cutlass.range_constexpr(C):
                    w = Float32(0.0)
                    if ls[p] > 0.0:
                        w = cute.arch.exp2(ms[p] - mx) * ls[p]
                    ms[p] = w
                    den = den + w
                inv = 1.0 / den
                acc = cute.make_rmem_tensor(8, Float32)
                acc.fill(0.0)
                for p in cutlass.range_constexpr(C):
                    w = ms[p] * inv
                    for k in cutlass.range_constexpr(4):
                        lo, hi = bf16x2_unpack(xs[4 * p + k])
                        acc[2 * k] = acc[2 * k] + w * lo
                        acc[2 * k + 1] = acc[2 * k + 1] + w * hi
                t = tok0 + m // H
                hh = m % H
                out = cute.make_rmem_tensor(4, Uint32)
                if col >= D_NOPE:
                    cb = (m // H) * 64 + (col - D_NOPE) // 2
                    co = [sCS[cb + k] for k in range(4)]
                    sn = [sCS[cb + 32 + k] for k in range(4)]
                    for k in cutlass.range_constexpr(4):
                        xe = self._rnd(acc[2 * k])
                        xo = self._rnd(acc[2 * k + 1])
                        out[k] = cvt_bf16x2(
                            xe * co[k] + xo * sn[k], xo * co[k] - xe * sn[k]
                        )
                else:
                    for k in cutlass.range_constexpr(4):
                        out[k] = cvt_bf16x2(acc[2 * k], acc[2 * k + 1])
                dst = (
                    rot
                    + (
                        Int64(t) * Int64(rot_t)
                        + Int64((hh // 8) * rot_g + (hh % 8) * 512 + col)
                    )
                    * 2
                )
                st_global_v4(dst, out[0], out[1], out[2], out[3])

    @cute.jit
    def _rnd(self, x: Float32) -> Float32:
        """X rounded to BF16."""
        lo, _ = bf16x2_unpack(cvt_bf16x2(x, x))
        return lo

    @cute.jit
    def _stage_q(self, g, q_ptr, q_st, cv, rope, c, R, ngr, sQ_addr, mbar_q):
        """Group g's query rows, rotated, into sQ (WG0 and WG1); its row count."""
        H = self.H
        rpg = self.rpg
        tidx = cute.arch.thread_idx()[0]
        b = g // ngr
        kg = g - b * ngr
        tok0 = b * R + kg * rpg
        nrows = cutlass.min(rpg, R - kg * rpg)
        NIT = BM * (D // 8) // 256
        xs = cute.make_rmem_tensor(NIT * 4, Uint32)
        # Every load before the first use: one L2 round trip.
        xs.fill(0)
        for it in cutlass.range_constexpr(NIT):
            m = it * 256 // (D // 8) + tidx // (D // 8)
            # Rows past the group's stay zero; their outputs are not stored.
            if it * 256 // (D // 8) // H < nrows:
                h = m % H
                r = m // H
                ga = q_ptr + (Int64(tok0 + r) * q_st + Int64(h * D + c * 8)) * 2
                w0, w1, w2, w3 = ld_cg_v4(ga)
                xs[4 * it], xs[4 * it + 1], xs[4 * it + 2], xs[4 * it + 3] = (
                    w0,
                    w1,
                    w2,
                    w3,
                )
        for it in cutlass.range_constexpr(NIT):
            m = it * 256 // (D // 8) + tidx // (D // 8)
            w0, w1, w2, w3 = xs[4 * it], xs[4 * it + 1], xs[4 * it + 2], xs[4 * it + 3]
            if rope != 0:
                rr = it * 256 // (D // 8) // H
                w0 = self._rot(w0, cv[rr * 8 + 0], cv[rr * 8 + 4])
                w1 = self._rot(w1, cv[rr * 8 + 1], cv[rr * 8 + 5])
                w2 = self._rot(w2, cv[rr * 8 + 2], cv[rr * 8 + 6])
                w3 = self._rot(w3, cv[rr * 8 + 3], cv[rr * 8 + 7])
            st_shared_v4(
                sQ_addr + self._swz(m, c) * 2,
                bf16x2_to_f16x2(w0),
                bf16x2_to_f16x2(w1),
                bf16x2_to_f16x2(w2),
                bf16x2_to_f16x2(w3),
            )
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(mbar_q)
        return nrows

    @cute.jit
    def _rot(self, w: Uint32, co: Float32, sn: Float32) -> Uint32:
        """GPT-J rotation of one BF16 pair, rounded to BF16."""
        xe, xo = bf16x2_unpack(w)
        return cvt_bf16x2(xe * co - xo * sn, xe * sn + xo * co)

    @cute.jit
    def _issue(self, g, j, issw, sw_img, cu_img, sK_addr, full_addr, mbar_full):
        """The task's tile image into sK (thread 256)."""
        src = sw_img + Int64(g * NSWB + j) * Int64(IMG * 2)
        if issw == 0:
            src = cu_img + Int64(g * NBC + j) * Int64(IMG * 2)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_full, BN * D * 2)
        bulk_g2s(sK_addr, src, Int32(BN * D * 2), full_addr)

    @cute.jit
    def _rope_coef(self, cv, g, pos_ptr, cs_ptr, c, R, ngr):
        """Cos / sin of this thread's 4 RoPE pairs (chunk c) for each row of
        group g."""
        rpg = self.rpg
        b = g // ngr
        kg = g - b * ngr
        nrows = cutlass.min(rpg, R - kg * rpg)
        for r in cutlass.range_constexpr(rpg):
            t = b * R + kg * rpg + cutlass.min(r, nrows - 1)
            p = ld_nc_i64(pos_ptr + Int64(t) * 8)
            cb = cs_ptr + (p * Int64(64) + Int64((c - D_NOPE // 8) * 4)) * 4
            c0, c1, c2, c3 = ld_nc_v4f(cb)
            s0, s1, s2, s3 = ld_nc_v4f(cb + 128)
            cv[r * 8 + 0] = c0
            cv[r * 8 + 1] = c1
            cv[r * 8 + 2] = c2
            cv[r * 8 + 3] = c3
            cv[r * 8 + 4] = s0
            cv[r * 8 + 5] = s1
            cv[r * 8 + 6] = s2
            cv[r * 8 + 7] = s3


class SwaImage(_Common):
    """Per row group, the keys of its SWA union (``UnionBuild``) dequantized
    into fp16 tile images [G, NSWB, 64 x 512]; keys past the union's length
    are zero. The step's own keys (the
    request's ``slot_mapping``) are left to ``attn_cute.QkvNorm``, which
    inserts them. One CTA per (block, group), 4 threads per key."""

    def __init__(self, H: int = 16):
        super().__init__(H, 1, 1)

    @cute.jit
    def __call__(
        self,
        kidx: Int64,
        klen: Int64,
        cap: Int32,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_pshift: Int32,
        slots: Int64,
        img: Int64,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            kidx, klen, cap, swa_ptr, swa_bs, swa_pshift, slots, img, R, ngr
        ).launch(grid=[NSWB, G, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        kidx: Int64,
        klen: Int64,
        cap: Int32,
        swa_ptr: Int64,
        swa_bs: Int64,
        swa_pshift: Int32,
        slots: Int64,
        img: Int64,
        R: Int32,
        ngr: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        blk, g, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        kl = ld_nc_b32(klen + Int64(g) * 4)
        if blk * BN < kl:
            tk = (tid // 32) * 8 + lane % 8
            ch = lane // 8
            u = blk * BN + tk
            v = Int32(-1)
            if u < kl:
                v = ld_nc_b32(kidx + (Int64(g) * Int64(cap) + Int64(u)) * 4)
            b = g // ngr
            fresh = Int32(0)
            for r in cutlass.range_constexpr(8):
                if r < R:
                    s = ld_nc_i64(slots + Int64(b * R + r) * 8)
                    if v >= 0 and Int64(v) == s:
                        fresh = Int32(1)
            raw = cute.make_rmem_tensor(36, Uint32)
            scl = cute.make_rmem_tensor(2, Uint32)
            self._load_token(
                v,
                ch,
                raw,
                scl,
                0,
                swa_ptr,
                swa_bs,
                swa_pshift,
                swa_ptr,
                swa_bs,
                swa_pshift,
            )
            base = img + Int64(g * NSWB + blk) * Int64(IMG * 2)
            self._store_token_any(v != -1, tk, ch, base, raw, scl, 0, True, fresh == 0)


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
