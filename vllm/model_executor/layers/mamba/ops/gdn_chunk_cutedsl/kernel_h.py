# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import (
    EVICT_FIRST,
    _tcgen05,
    cvt,
    fence_before_tma_store,
    simple_tma_copy,
)


class Sm100ChunkHKernel:
    """For each sequence, compute the chunk recurrent update.

    The input V tile is the U output from the KKT/UW kernel. For each chunk:
        V_new = U - W @ H.T
        (we actually do V_new.T = U.T - H @ W.T instead)

        H_scaled = H * exp(g_last)
        V_scaled = V_new * exp(g_last - g)
        H_new = H_scaled + V_scaled.T @ K
    """

    def __init__(
        self,
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        h_dtype: cutlass.Numeric = Float32,
        BT: int = 64,
        num_stages: int = 2,
    ) -> None:
        assert Hv % H == 0
        assert K_dim == V_dim == 128
        assert BT == 64
        self.H = H
        self.Hv = Hv
        self.K_dim = K_dim
        self.V_dim = V_dim
        self.h_dtype = h_dtype
        self.BT = BT
        self.num_stages = num_stages
        self.num_warps = 10

    @cute.jit
    def _make_bf16_tma_args(
        self,
        tensor: cute.Tensor,
        dim: cutlass.Constexpr[int],
        op: cpasync.TmaCopyOp,
        stages: cutlass.Constexpr[int],
    ):
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (self.BT, 1, (64, dim // 64), stages),
            stride=(64, 0, (1, self.BT * 64), self.BT * dim),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            slayout,
            cta_tiler=(self.BT, 1, dim),
        )

    @cute.jit
    def _make_h_tma_args(self, tensor: cute.Tensor, op: cpasync.TmaCopyOp):
        # number of elements to fill 128B
        num_elems = 128 // (tensor.element_type.width // 8)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (1, 1, self.V_dim, (num_elems, self.K_dim // num_elems)),
            stride=(0, 0, num_elems, (1, self.V_dim * num_elems)),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, None, num_elems)),
            slayout,
            cta_tiler=(1, 1, self.V_dim, self.K_dim),
        )

    @cute.jit
    def __call__(
        self,
        K: cute.Tensor,
        V: cute.Tensor,
        W: cute.Tensor,
        V_new: cute.Tensor,
        g_cu: cute.Tensor,
        h: cute.Tensor,
        h0: cute.Tensor,
        ht: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        stream: CUstream,
    ):
        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()

        K_tma = self._make_bf16_tma_args(K, self.K_dim, tma_g2s, self.num_stages)
        V_tma = self._make_bf16_tma_args(V, self.V_dim, tma_g2s, self.num_stages)
        W_tma = self._make_bf16_tma_args(W, self.K_dim, tma_g2s, self.num_stages)
        V_new_tma = self._make_bf16_tma_args(V_new, self.V_dim, tma_s2g, 1)
        H0_tma = self._make_h_tma_args(h0, tma_g2s)
        HT_tma = self._make_h_tma_args(ht, tma_s2g)
        H_tma = self._make_h_tma_args(h, tma_s2g)

        grid = (self.Hv, h0.shape[0], 1)
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            K_tma,
            V_tma,
            W_tma,
            V_new_tma,
            H0_tma,
            HT_tma,
            H_tma,
            g_cu,
            cu_seqlens,
            chunk_offsets,
        ).launch(grid=grid, block=block, min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(
        self,
        K_tma: cpasync.TmaInfo,
        V_tma: cpasync.TmaInfo,
        W_tma: cpasync.TmaInfo,
        V_new_tma: cpasync.TmaInfo,
        H0_tma: cpasync.TmaInfo,
        HT_tma: cpasync.TmaInfo,
        H_tma: cpasync.TmaInfo,
        g_cu: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        head_id, seq_id, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BT = self.BT
        V_dim = self.V_dim
        K_dim = self.K_dim
        num_stages = self.num_stages
        is_f32 = self.h_dtype == Float32

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()

        # remove size=1 modes
        sW = allocate_tensor(smem, BFloat16, W_tma.smem_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, V_tma.smem_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, K_tma.smem_layout)[None, 0, None, None]
        sH0 = allocate_tensor(smem, self.h_dtype, H0_tma.smem_layout)[0, 0, None, None]
        sH = allocate_tensor(smem, BFloat16, H_tma.smem_layout)[0, 0, None, None]
        sV_new = allocate_tensor(smem, BFloat16, V_new_tma.smem_layout)[
            None, 0, None, 0
        ]

        s_v_scale = smem.allocate_array(Float32, BT)
        tma_mbar = smem.allocate_array(Int64, num_stages)
        wh_in_mbar = smem.allocate_array(Int64, num_stages)
        wh_done_mbar = smem.allocate_array(Int64, num_stages)
        vk_in_mbar = smem.allocate_array(Int64, num_stages)
        vk_done_mbar = smem.allocate_array(Int64, num_stages)
        h0_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        wh_tmem = 0
        vk_tmem = wh_tmem + BT
        h_tmem_base = vk_tmem + K_dim
        v_tmem_base = h_tmem_base + K_dim // 2

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_mbar + i, 1)
                    cute.arch.mbarrier_init(wh_in_mbar + i, 256)
                    cute.arch.mbarrier_init(wh_done_mbar + i, 1)
                    cute.arch.mbarrier_init(vk_in_mbar + i, 256)
                    cute.arch.mbarrier_init(vk_done_mbar + i, 1)
                cute.arch.mbarrier_init(h0_mbar, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(H0_tma.atom)
            cpasync.prefetch_descriptor(W_tma.atom)
            cpasync.prefetch_descriptor(V_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
            cpasync.prefetch_descriptor(HT_tma.atom)
            cpasync.prefetch_descriptor(H_tma.atom)
            cpasync.prefetch_descriptor(V_new_tma.atom)
        cute.arch.sync_threads()

        bos = cu_seqlens[seq_id]
        eos = cu_seqlens[seq_id + 1]
        seqlen = eos - bos
        num_chunks = cute.ceil_div(seqlen, BT)

        if warp_id == 9:
            # TMA warp
            stage_id = 0
            parity = 1

            k_head_id = head_id // (self.Hv // self.H)
            chunk_offset = chunk_offsets[seq_id]

            # load H0
            with cute.arch.elect_one():
                H0_size = V_dim * K_dim * self.h_dtype.width // 8
                cute.arch.mbarrier_arrive_and_expect_tx(h0_mbar, H0_size)
            simple_tma_copy(
                H0_tma.atom,
                H0_tma.tma_tensor[seq_id, head_id, None, None],
                sH0,
                h0_mbar,
            )

            # shape: ((BT, num_BT_tiles), (64, 2))
            gW_tiles = cute.logical_divide(
                W_tma.tma_tensor[None, head_id, None], (BT, None)
            )
            gV_tiles = cute.logical_divide(
                V_tma.tma_tensor[None, head_id, None], (BT, None)
            )
            gK_tiles = cute.logical_divide(
                cute.domain_offset((bos, 0), K_tma.tma_tensor[None, k_head_id, None]),
                (BT, None),
            )

            for chunk_id in range(num_chunks):
                mbar = tma_mbar + stage_id
                gW = gW_tiles[(None, chunk_offset + chunk_id), None]
                gV = gV_tiles[(None, chunk_offset + chunk_id), None]
                gK = gK_tiles[(None, chunk_id), None]

                # wait for MMA to release the buffer
                cute.arch.mbarrier_wait(vk_done_mbar + stage_id, parity)

                # load W, V (i.e. U), and K
                with cute.arch.elect_one():
                    STAGE_SIZE = BT * (K_dim + V_dim + K_dim) * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(
                    W_tma.atom, gW, sW[None, None, stage_id], mbar, EVICT_FIRST
                )
                simple_tma_copy(
                    V_tma.atom, gV, sV[None, None, stage_id], mbar, EVICT_FIRST
                )
                simple_tma_copy(K_tma.atom, gK, sK[None, None, stage_id], mbar)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id == 8:
            # MMA warp
            _tcgen05.alloc(taddr)
            stage_id = 0
            parity = 0

            wh_idesc = _tcgen05.make_bf16_idesc(V_dim, BT, negate_A=True)
            vk_idesc = _tcgen05.make_bf16_idesc(V_dim, K_dim, transpose_B=True)

            # LBO=BT*128 is ignored for K-major
            sdesc_template = _tcgen05.make_sdesc_128B_swizzle(BT * 128)

            # when using BF16 state, H is read from smem for the 1st iteration
            # variable names in this conditional branch can't be the same as those
            # in the mainloop below due to CuteDSL restrictions.
            if cutlass.const_expr(not is_f32):
                ##### 1st MMA: V_new.T = V.T - H @ W.T #####
                Haddr0 = sH0[None, None].iterator.toint()
                Waddr0 = sW[None, None, stage_id].iterator.toint()
                hdesc0_base = sdesc_template | (Haddr0 >> 4)
                wdesc0_base = sdesc_template | (Waddr0 >> 4)

                cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                cute.arch.mbarrier_wait(wh_in_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(K_dim // 64):
                    for j in cutlass.range_constexpr(64 // 16):
                        hdesc0 = hdesc0_base | ((i * V_dim * 128 + j * 32) >> 4)
                        wdesc0 = wdesc0_base | ((i * BT * 128 + j * 32) >> 4)
                        _tcgen05.mma_f16(wh_tmem, hdesc0, wdesc0, wh_idesc, True)
                _tcgen05.commit(wh_done_mbar + stage_id)

                ##### 2nd MMA: H_new = H + V_new.T @ K #####
                Kaddr0 = sK[None, None, stage_id].iterator.toint()
                kdesc0_base = sdesc_template | (Kaddr0 >> 4)

                cute.arch.mbarrier_wait(vk_in_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for k in cutlass.range_constexpr(BT // 16):
                    vtmem0 = v_tmem_base + k * 8
                    kdesc0 = kdesc0_base | ((k * 16 * 128) >> 4)
                    _tcgen05.mma_ts_f16(vk_tmem, vtmem0, kdesc0, vk_idesc, True)
                _tcgen05.commit(vk_done_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

            num_iters = num_chunks - int(not is_f32)
            for _ in range(num_iters):
                ##### 1st MMA: V_new.T = V.T - H @ W.T #####
                Waddr = sW[None, None, stage_id].iterator.toint()
                wdesc_base = sdesc_template | (Waddr >> 4)

                cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                cute.arch.mbarrier_wait(wh_in_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(K_dim // 64):
                    for j in cutlass.range_constexpr(64 // 16):
                        htmem = h_tmem_base + i * 32 + j * 8
                        wdesc = wdesc_base | ((i * BT * 128 + j * 32) >> 4)
                        _tcgen05.mma_ts_f16(wh_tmem, htmem, wdesc, wh_idesc, True)
                _tcgen05.commit(wh_done_mbar + stage_id)

                ##### 2nd MMA: H_new = H + V_new.T @ K #####
                Kaddr = sK[None, None, stage_id].iterator.toint()
                kdesc_base = sdesc_template | (Kaddr >> 4)

                cute.arch.mbarrier_wait(vk_in_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for k in cutlass.range_constexpr(BT // 16):
                    vtmem = v_tmem_base + k * 8
                    kdesc = kdesc_base | ((k * 16 * 128) >> 4)
                    _tcgen05.mma_ts_f16(vk_tmem, vtmem, kdesc, vk_idesc, True)
                _tcgen05.commit(vk_done_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id >= 4:
            # H warps
            tid_ = tid % 128
            warp_id_ = warp_id % 4
            chunk_offset = chunk_offsets[seq_id]

            stage_id = 0
            vk_stage_id = 0
            vk_parity = 0

            op = cute.nvgpu.CopyUniversalOp()
            cp_16B = cute.make_copy_atom(op, Float32, num_bits_per_copy=128)

            ##### chunk_id = 0 #####
            if True:
                chunk_id = 0
                end_t = min(bos + (chunk_id + 1) * BT, eos)
                last_idx = end_t - 1
                h_scale = cute.math.exp(g_cu[last_idx, head_id], fastmath=True)

                # for 1st chunk, wait for H0 transfer from gmem
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(h0_mbar, 0)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                # when H0 is FP32, we need to pack it to BF16
                # also store to smem for TMA store later.
                if cutlass.const_expr(is_f32):
                    for i in cutlass.range_constexpr(K_dim // 32):
                        # H0 smem layout: (V_dim, (32, K_dim/32))
                        h_f32 = cute.make_rmem_tensor(32, Float32)
                        cute.copy(cp_16B, sH0[tid_, (None, i)], h_f32)

                        h_bf16 = cute.make_rmem_tensor(32, BFloat16)
                        h_bf16.store(h_f32.load().to(BFloat16))
                        _tcgen05.st(
                            warp_id_ * 32, h_tmem_base + i * 16, "32x32b", 16, h_bf16
                        )

                        # H smem layout: (V_dim, (64, K_dim/64))
                        dst = cute.local_tile(sH[tid_, None], (32,), (i,))
                        cute.copy(cp_16B, h_bf16, dst)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(wh_in_mbar + stage_id)

                # scale H for 2nd MMA
                for i in cutlass.range_constexpr(K_dim // 32):
                    h_f32 = cute.make_rmem_tensor(32, Float32)

                    if cutlass.const_expr(is_f32):
                        cute.copy(cp_16B, sH0[tid_, (None, i)], h_f32)

                    else:
                        h_bf16 = cute.make_rmem_tensor(32, BFloat16)
                        sH_src = cute.local_tile(sH0[tid_, None], (32,), (i,))
                        cute.copy(cp_16B, sH_src, h_bf16)
                        h_f32.store(
                            cvt.bf16x2_to_fp32x2(
                                cute.recast_tensor(h_bf16, Uint32)
                            ).load()
                        )

                    for j in cutlass.range(32, vectorize=True):
                        h_f32[j] *= h_scale
                    _tcgen05.st(warp_id_ * 32, vk_tmem + i * 32, "32x32b", 32, h_f32)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(vk_in_mbar + stage_id)

                # for BF16 H0, we issue TMA store from H0 smem
                # for FP32 H0, we issue TMA store from H smem (after packing)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                fence_before_tma_store()
                if warp_id_ == 3:
                    h_src = sH if cutlass.const_expr(is_f32) else sH0
                    h_dst = H_tma.tma_tensor[
                        chunk_offset + chunk_id, head_id, None, None
                    ]
                    simple_tma_copy(H_tma.atom, h_src, h_dst)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                        # When H0 is BF16, and there is only 1 chunk, storing
                        # the final state to sH0 can race before this store
                        # has finished. hence, we need to wait here.
                        if cutlass.const_expr(not is_f32):
                            cute.arch.cp_async_bulk_wait_group(0, read=True)

                stage_id = (stage_id + 1) % num_stages

            ##### subsequent chunks #####
            for chunk_id in range(1, num_chunks):
                end_t = min(bos + (chunk_id + 1) * BT, eos)
                last_idx = end_t - 1
                h_scale = cute.math.exp(g_cu[last_idx, head_id], fastmath=True)

                # wait for H from previous vk MMA
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(vk_done_mbar + vk_stage_id, vk_parity)
                    vk_stage_id = (vk_stage_id + 1) % num_stages
                    if vk_stage_id == 0:
                        vk_parity ^= 1
                elif warp_id_ == 3:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                # load FP32 H from tmem, convert to BF16, store to tmem for 1st MMA,
                # store to smem for TMA store later.
                for i in cutlass.range_constexpr(K_dim // 32):
                    h_f32 = _tcgen05.ld(warp_id_ * 32, vk_tmem + i * 32, "32x32b", 32)
                    h_bf16 = cute.make_rmem_tensor(32, BFloat16)
                    h_bf16.store(h_f32.to(BFloat16))
                    _tcgen05.st(
                        warp_id_ * 32, h_tmem_base + i * 16, "32x32b", 16, h_bf16
                    )

                    # H smem layout: (V_dim, (64, K_dim/64))
                    dst = cute.local_tile(sH[tid_, None], (32,), (i,))
                    cute.copy(cp_16B, h_bf16, dst)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(wh_in_mbar + stage_id)

                # scale H for 2nd MMA
                for i in cutlass.range_constexpr(K_dim // 32):
                    h_f32 = cute.make_rmem_tensor(32, Float32)
                    h_f32.store(
                        _tcgen05.ld(warp_id_ * 32, vk_tmem + i * 32, "32x32b", 32)
                    )
                    for j in cutlass.range(32, vectorize=True):
                        h_f32[j] *= h_scale
                    _tcgen05.st(warp_id_ * 32, vk_tmem + i * 32, "32x32b", 32, h_f32)
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(vk_in_mbar + stage_id)

                # issue TMA store for O kernel
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                fence_before_tma_store()
                if warp_id_ == 3:
                    h_dst = H_tma.tma_tensor[
                        chunk_offset + chunk_id, head_id, None, None
                    ]
                    simple_tma_copy(H_tma.atom, sH, h_dst)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                stage_id = (stage_id + 1) % num_stages

            # handle final state. reuse H0 smem.
            if warp_id_ == 0:
                cute.arch.mbarrier_wait(vk_done_mbar + vk_stage_id, vk_parity)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()

            for i in cutlass.range_constexpr(K_dim // 32):
                h_f32 = cute.make_rmem_tensor(32, Float32)
                h_f32.store(_tcgen05.ld(warp_id_ * 32, vk_tmem + i * 32, "32x32b", 32))

                if cutlass.const_expr(is_f32):
                    cute.copy(cp_16B, h_f32, sH0[tid_, (None, i)])

                else:
                    h_bf16 = cute.make_rmem_tensor(32, BFloat16)
                    h_bf16.store(h_f32.load().to(BFloat16))
                    sH0_dst = cute.local_tile(sH0[tid_, None], (32,), (i,))
                    cute.copy(cp_16B, h_bf16, sH0_dst)

            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            if warp_id_ == 0:
                ht_dst = HT_tma.tma_tensor[seq_id, head_id, None, None]
                simple_tma_copy(HT_tma.atom, sH0, ht_dst)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
            if warp_id_ == 1:
                _tcgen05.dealloc()

        else:
            # V warps
            stage_id = 0
            parity = 0

            chunk_offset = chunk_offsets[seq_id]

            ldsm_trans_op = warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True)
            stsm_trans_op = warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True)
            ldsm_trans_atom = cute.make_copy_atom(ldsm_trans_op, BFloat16)
            stsm_trans_atom = cute.make_copy_atom(stsm_trans_op, BFloat16)

            # ((BT, num_BT_tiles), V_dim)
            gV_new_tiles = cute.logical_divide(
                V_new_tma.tma_tensor[None, head_id, None], (BT, None)
            )

            # sV shape: [BT, (64, V_dim/64), num_stages]
            # sV_view shape: [BT, (8, (8,2)), num_stages]
            sV_view = cute.logical_divide(sV, (None, 8, None))
            sV_new_view = cute.logical_divide(sV_new, (None, 8))

            # [BT, 8, num_stages]
            s_col = warp_id * 4 + (lane_id // 8)
            sV_view = sV_view[None, (None, s_col), None]
            sV_new_view = sV_new_view[None, (None, s_col)]

            for chunk_id in range(num_chunks):
                # wait for V to arrive
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)

                # unpack V BF16->FP32, then store to tmem for 1st MMA
                # V smem layout: [BT, (64, V_dim/64)] / [BT, V_dim]
                # each iteration, CTA loads [8, V_dim] tile
                # (warp loads [8, 32] tile)
                for i in cutlass.range_constexpr(BT // 8):
                    s_row = i * 8 + (lane_id % 8)
                    v_bf16 = cute.make_rmem_tensor(8, BFloat16)
                    cute.copy(ldsm_trans_atom, sV_view[s_row, None, stage_id], v_bf16)
                    v_fp32 = cvt.bf16x2_to_fp32x2(cute.recast_tensor(v_bf16, Uint32))
                    v_fp32 = cute.logical_divide(v_fp32, 4)  # (4, 2)

                    tcol = wh_tmem + i * 8
                    _tcgen05.st(warp_id * 32 + 0, tcol, "16x256b", 1, v_fp32[None, 0])
                    _tcgen05.st(warp_id * 32 + 16, tcol, "16x256b", 1, v_fp32[None, 1])

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(wh_in_mbar + stage_id)

                # load g_cu for scaling
                if tid < BT:
                    end_t = min(bos + (chunk_id + 1) * BT, eos)
                    last_idx = end_t - 1
                    t = bos + chunk_id * BT + tid
                    val = Float32(0.0)
                    if t < eos:
                        val = cute.math.exp(
                            g_cu[last_idx, head_id] - g_cu[t, head_id],
                            fastmath=True,
                        )
                    s_v_scale[tid] = val

                # wait for 1st MMA to finish
                if warp_id == 2:
                    cute.arch.mbarrier_wait(wh_done_mbar + stage_id, parity)
                elif warp_id == 3:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(BT // 8):
                    v_new = cute.make_rmem_tensor((4, 2), Float32)
                    tcol = wh_tmem + i * 8
                    v_new[None, 0].store(
                        _tcgen05.ld(warp_id * 32 + 0, tcol, "16x256b", 1)
                    )
                    v_new[None, 1].store(
                        _tcgen05.ld(warp_id * 32 + 16, tcol, "16x256b", 1)
                    )
                    v_new_bf16 = cute.make_rmem_tensor(8, BFloat16)
                    v_new_bf16.store(v_new.load().to(BFloat16))

                    # scale V_new for 2nd MMA
                    scale0 = s_v_scale[i * 8 + (lane_id % 4) * 2 + 0]
                    scale1 = s_v_scale[i * 8 + (lane_id % 4) * 2 + 1]
                    v_scaled = cute.make_rmem_tensor(8, Float32)
                    for k in cutlass.range_constexpr(4):
                        v_scaled[k * 2] = v_new[k * 2] * scale0
                        v_scaled[k * 2 + 1] = v_new[k * 2 + 1] * scale1
                    v_scaled_bf16 = v_scaled.load().to(BFloat16).reshape((4, 2))

                    # store V_new BF16 for O kernel
                    s_row = i * 8 + (lane_id % 8)
                    cute.copy(stsm_trans_atom, v_new_bf16, sV_new_view[s_row, None])

                    # store to tmem
                    tcol = v_tmem_base + i * 4
                    _tcgen05.st(
                        warp_id * 32 + 0, tcol, "16x128b", 1, v_scaled_bf16[None, 0]
                    )
                    _tcgen05.st(
                        warp_id * 32 + 16, tcol, "16x128b", 1, v_scaled_bf16[None, 1]
                    )
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(vk_in_mbar + stage_id)

                # issue TMA store for V_new
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 3:
                    gV = gV_new_tiles[(None, chunk_offset + chunk_id), None]
                    simple_tma_copy(V_new_tma.atom, sV_new, gV)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

    @cache
    @staticmethod
    def compile(
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        h_dtype: cutlass.Numeric = Float32,
        BT: int = 64,
        num_stages: int = 2,
    ):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()
        total_chunks_n = cute.sym_int()
        num_sequences = cute.sym_int()
        cu_entries = cute.sym_int()

        K = make_fake_tensor(BFloat16, (total_t, H, K_dim), divisibility=16)
        V = make_fake_tensor(BFloat16, (pad_t, Hv, V_dim), divisibility=16)
        W = make_fake_tensor(BFloat16, (pad_t, Hv, K_dim), divisibility=16)
        V_new = make_fake_tensor(BFloat16, (pad_t, Hv, V_dim), divisibility=16)
        g_cu = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        h = make_fake_tensor(
            BFloat16, (total_chunks_n, Hv, V_dim, K_dim), divisibility=16
        )
        h0 = make_fake_tensor(
            h_dtype, (num_sequences, Hv, V_dim, K_dim), divisibility=16
        )
        ht = make_fake_tensor(
            h_dtype, (num_sequences, Hv, V_dim, K_dim), divisibility=16
        )
        cu_seqlens = make_fake_tensor(Int32, (cu_entries,), divisibility=1)
        chunk_offsets = make_fake_tensor(Int32, (cu_entries,), divisibility=1)

        kernel = Sm100ChunkHKernel(H, Hv, K_dim, V_dim, h_dtype, BT, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            K,
            V,
            W,
            V_new,
            g_cu,
            h,
            h0,
            ht,
            cu_seqlens,
            chunk_offsets,
            stream,
            options="--enable-tvm-ffi",
        )


def h_cutedsl(
    K: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    V_new: torch.Tensor,
    g_cu: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    ht: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    BT: int = 64,
    num_stages: int = 2,
) -> None:
    """Compute H/V_new with the same argument order as the CUDA wrapper."""

    _, H, K_dim = K.shape
    _, Hv, V_dim = V.shape
    h_dtype = {
        torch.bfloat16: BFloat16,
        torch.float32: Float32,
    }[h0.dtype]
    Sm100ChunkHKernel.compile(H, Hv, K_dim, V_dim, h_dtype, BT, num_stages)(
        K,
        V,
        W,
        V_new,
        g_cu,
        h,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
    )


h_v2b_cutedsl = h_cutedsl
