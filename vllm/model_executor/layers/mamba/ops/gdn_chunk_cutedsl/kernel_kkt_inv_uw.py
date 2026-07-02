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
    _bf16x2_neg,
    _bf16x2_sub,
    _tcgen05,
    cvt,
    fence_before_tma_store,
    mma_bf16,
    simple_tma_copy,
)


class Sm100ChunkUWKernel:
    """Compute per-chunk KKT inverse preprocessing and U/W tiles.

    Gamma[i,j] = exp(g_cu[i] - g_cu[j])
    A = strictLower(beta * (K @ K.T) * Gamma)
    Ai = inverse(I + A)
    U = (Ai * beta) @ V
    W = (Ai * beta * exp(g_cu)) @ K
    """

    def __init__(
        self,
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        num_stages: int = 2,
    ) -> None:
        assert Hv % H == 0
        assert K_dim == V_dim == 128
        self.H = H
        self.Hv = Hv
        self.K_dim = K_dim
        self.V_dim = V_dim
        self.num_stages = num_stages

        # hard-code
        self.BT = 64
        self.num_warps = 4 + 4 + 4

    @cute.jit
    def _make_tma_args(
        self,
        tensor: cute.Tensor,
        dim: cutlass.Constexpr[int],
        num_stages: int,
        op: cpasync.TmaCopyOp,
    ):
        # logical layout: [BT, dim]
        # permute for TMA: [dim/64, BT, 64] with swizzling
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (self.BT, 1, (64, dim // 64), num_stages),
            stride=(64, 0, (1, self.BT * 64), self.BT * dim),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)

        # we need to convert gmem layout to (T, H, (64, D/64)) for make_tiled_tma_atom()
        # to emit a single 4D TMA. otherwise, it will emit (D/64)x 3D TMA.
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            slayout,
            cta_tiler=(self.BT, 1, dim),
        )

    @cute.jit
    def __call__(
        self,
        K: cute.Tensor,
        V: cute.Tensor,
        U: cute.Tensor,
        W: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        g_cu: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        num_sms: Int32,
        stream: CUstream,
    ):
        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()

        K_tma = self._make_tma_args(K, self.K_dim, self.num_stages, tma_g2s)
        V_tma = self._make_tma_args(V, self.V_dim, self.num_stages, tma_g2s)
        U_tma = self._make_tma_args(U, self.V_dim, 1, tma_s2g)
        W_tma = self._make_tma_args(W, self.K_dim, 1, tma_s2g)

        grid = (num_sms // self.Hv, self.Hv, 1)
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            K_tma,
            V_tma,
            U_tma,
            W_tma,
            g,
            beta,
            g_cu,
            cu_seqlens,
            chunk_indices,
            total_chunks,
        ).launch(grid=grid, block=block, min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(
        self,
        K_tma: cpasync.TmaInfo,
        V_tma: cpasync.TmaInfo,
        U_tma: cpasync.TmaInfo,
        W_tma: cpasync.TmaInfo,
        g: cute.Tensor,
        beta: cute.Tensor,
        g_cu: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, head_id, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        k_head_id = head_id // (self.Hv // self.H)

        BT = self.BT
        K_dim = self.K_dim
        V_dim = self.V_dim
        num_stages = self.num_stages

        INV_BAR = 1
        EPI_BAR = 2
        SCAN_BAR = 3
        TMEM_ALLOC_BAR = 4

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sK = allocate_tensor(smem, BFloat16, K_tma.smem_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, V_tma.smem_layout)[None, 0, None, None]
        sU = allocate_tensor(smem, BFloat16, U_tma.smem_layout)[None, 0, None, 0]
        sW = allocate_tensor(smem, BFloat16, W_tma.smem_layout)[None, 0, None, 0]

        sA_ptr = smem.allocate_array(BFloat16, BT * BT, byte_alignment=16)
        sAi_ptr = smem.allocate_array(BFloat16, BT * BT, byte_alignment=16)

        s_beta = smem.allocate_tensor(Float32, cute.make_layout((BT, num_stages)))
        s_g_cu = smem.allocate_tensor(Float32, cute.make_layout((BT, num_stages)))
        s_beta_g = smem.allocate_tensor(Float32, cute.make_layout((BT, num_stages)))

        tma_mbar = smem.allocate_array(Int64, num_stages)
        prep_mbar = smem.allocate_array(Int64, num_stages)
        mma_kkt_mbar = smem.allocate_array(Int64, num_stages)
        inv_mbar = smem.allocate_array(Int64, num_stages)
        mma_u_mbar = smem.allocate_array(Int64, num_stages)
        mma_w_mbar = smem.allocate_array(Int64, num_stages)
        epi_mbar = smem.allocate_array(Int64, num_stages)
        taddr = smem.allocate(Int32, 4)

        kkt_tmem = 0
        U_tmem_base = kkt_tmem + BT
        Ab_tmem_base = U_tmem_base + V_dim * num_stages
        assert Ab_tmem_base + (BT // 2) <= 512

        # prepare ldmatrix/stmatrix ops
        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4)
        ldsm_trans_op = warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True)
        ldsm_atom = cute.make_copy_atom(ldsm_op, BFloat16)
        stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)
        ldsm_trans_atom = cute.make_copy_atom(ldsm_trans_op, BFloat16)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_mbar + i, 1)
                    cute.arch.mbarrier_init(prep_mbar + i, 64)
                    cute.arch.mbarrier_init(mma_kkt_mbar + i, 1)
                    cute.arch.mbarrier_init(inv_mbar + i, 128)
                    cute.arch.mbarrier_init(mma_u_mbar + i, 1)
                    cute.arch.mbarrier_init(mma_w_mbar + i, 1)
                    cute.arch.mbarrier_init(epi_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(K_tma.atom)
            cpasync.prefetch_descriptor(V_tma.atom)
            cpasync.prefetch_descriptor(U_tma.atom)
            cpasync.prefetch_descriptor(W_tma.atom)
        cute.arch.sync_threads()

        num_global_chunks = total_chunks[0]
        if warp_id == 11:
            # TMA warp
            stage_id = 0
            parity = 1

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]

                # since off_t is not a multiple of BT, we need to use
                # domain_offset() to shift the pointer first.
                mbar = tma_mbar + stage_id
                gK = cute.local_tile(
                    cute.domain_offset(
                        (bos, 0), K_tma.tma_tensor[None, k_head_id, None]
                    ),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                gV = cute.local_tile(
                    cute.domain_offset((bos, 0), V_tma.tma_tensor[None, head_id, None]),
                    tiler=(BT, V_dim),
                    coord=(chunk_id, 0),
                )

                # when UW MMA is done, K and V TMA buffers are released
                cute.arch.mbarrier_wait(mma_u_mbar + stage_id, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = BT * (K_dim + V_dim) * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(K_tma.atom, gK, sK[None, None, stage_id], mbar)
                simple_tma_copy(
                    V_tma.atom, gV, sV[None, None, stage_id], mbar, EVICT_FIRST
                )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id == 10:
            # MMA warp
            _tcgen05.alloc(taddr)
            cute.arch.barrier(barrier_id=TMEM_ALLOC_BAR, number_of_threads=160)

            stage_id = 0
            parity = 0

            kkt_idesc = _tcgen05.make_bf16_idesc(BT, BT)
            u_idesc = _tcgen05.make_bf16_idesc(BT, V_dim, transpose_B=True)
            w_idesc = _tcgen05.make_bf16_idesc(BT, K_dim, transpose_B=True)

            # LBO=BT*128 is ignored for K-major
            sdesc_template = _tcgen05.make_sdesc_128B_swizzle(BT * 128)

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                U_tmem = U_tmem_base + V_dim * stage_id
                W_tmem = U_tmem | (16 << 16)
                Ab_tmem = Ab_tmem_base
                Abg_tmem = Ab_tmem_base | (16 << 16)

                ##### KKT MMA: KKT = K @ K.T #####
                kaddr = sK[None, None, stage_id].iterator.toint()
                kdesc_base = sdesc_template | (kaddr >> 4)

                # wait for TMA data to arrive
                # kkt tmem is guaranteed to be free as this is issued
                # after the previous kkt's consumer (inv warps)
                cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(K_dim // 64):
                    for j in cutlass.range_constexpr(64 // 16):
                        kdesc = kdesc_base | ((i * BT * 128 + j * 32) >> 4)
                        enable_d = (i > 0) or (j > 0)
                        _tcgen05.mma_f16(kkt_tmem, kdesc, kdesc, kkt_idesc, enable_d)
                _tcgen05.commit(mma_kkt_mbar + stage_id)

                ##### U/W MMA: U = Ab @ V, W = Abg @ K #####
                vaddr = sV[None, None, stage_id].iterator.toint()
                vdesc = sdesc_template | (vaddr >> 4)
                kdesc = sdesc_template | (kaddr >> 4)

                # wait for epilogue to release tmem buffer
                cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)
                cute.arch.mbarrier_wait(inv_mbar + stage_id, parity)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(BT // 16):
                    _tcgen05.mma_ts_f16(W_tmem, Abg_tmem + i * 8, kdesc, w_idesc, i > 0)
                    kdesc += (16 * 128) >> 4
                _tcgen05.commit(mma_w_mbar + stage_id)

                for i in cutlass.range_constexpr(BT // 16):
                    _tcgen05.mma_ts_f16(U_tmem, Ab_tmem + i * 8, vdesc, u_idesc, i > 0)
                    vdesc += (16 * 128) >> 4
                _tcgen05.commit(mma_u_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

            cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)
            _tcgen05.dealloc()

        elif warp_id >= 8:
            # dedicated prep warps for beta and gate, consumed by INV warps
            stage_id = 0
            parity = 0
            tid_ = tid % 128
            warp_id_ = warp_id % 4

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                eos = cu_seqlens[seq_id + 1]
                off_t = bos + chunk_id * BT
                t = off_t + tid_

                in_bounds = t < eos
                beta_val = beta[t, head_id] if in_bounds else Float32(0.0)
                g_val = g[t, head_id] if in_bounds else Float32(0.0)

                # warp-local prefix scan
                for i in cutlass.range_constexpr(5):
                    offset = cutlass.const_expr(1 << i)
                    lower = cute.arch.shuffle_sync_up(g_val, offset, mask_and_clamp=0)
                    if lane_id >= offset:
                        g_val += lower

                # Delay the stage-reuse wait until just before touching smem:
                # global loads and the warp-local scan do not use staged buffers.
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(inv_mbar + stage_id, parity ^ 1)
                cute.arch.barrier(barrier_id=SCAN_BAR, number_of_threads=BT)

                # Store beta and the per-warp scan totals for the cross-warp fixup.
                s_beta[tid_, stage_id] = beta_val
                if lane_id == 31:
                    s_g_cu[warp_id_, stage_id] = g_val
                cute.arch.barrier(barrier_id=SCAN_BAR, number_of_threads=BT)

                # Add the sum from the lower prep warp.
                if warp_id_ == 1:
                    g_val += s_g_cu[0, stage_id]
                cute.arch.barrier(barrier_id=SCAN_BAR, number_of_threads=BT)

                if in_bounds:
                    g_cu[t, head_id] = g_val

                s_g_cu[tid_, stage_id] = g_val
                s_beta_g[tid_, stage_id] = beta_val * cute.math.exp(g_val)
                cute.arch.mbarrier_arrive(prep_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id >= 4:
            # inv warps
            tid_ = tid % 128
            warp_id_ = warp_id % 4

            def store_ab_abg(
                Ai_f32, s_beta, s_beta_g, warp_id_, lane_id, tile_col, Ab_tmem_base
            ):
                # compute Ab and Abg from Ai, then store to tmem
                beta_col = cute.make_rmem_tensor((2, 2), Float32)
                beta_g_col = cute.make_rmem_tensor((2, 2), Float32)

                for i in cutlass.range_constexpr(2):
                    base = tile_col * 16 + i * 8 + (lane_id % 4) * 2
                    for j in cutlass.range_constexpr(2):
                        beta_col[j, i] = s_beta[base + j]
                        beta_g_col[j, i] = s_beta_g[base + j]

                # without conversion to TensorSSA, cutlass.range(vectorize=True) fails
                beta_col = beta_col.load()
                beta_g_col = beta_g_col.load()

                Ab_f32 = cute.make_rmem_tensor(8, Float32)
                Abg_f32 = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range(8, vectorize=True):
                    scale_idx = (i // 4) * 2 + (i % 2)
                    Ab_f32[i] = Ai_f32[i] * beta_col[scale_idx]
                    Abg_f32[i] = Ai_f32[i] * beta_g_col[scale_idx]

                Ab = Ab_f32.load().to(BFloat16)
                Abg = Abg_f32.load().to(BFloat16)
                Ab_tmem = Ab_tmem_base + tile_col * 8
                _tcgen05.st(warp_id_ * 32, Ab_tmem, "16x128b", 2, Ab)
                _tcgen05.st(warp_id_ * 32 + 16, Ab_tmem, "16x128b", 2, Abg)

            # clear the Ab/Abg tmem buffer once before mainloop
            # this is to keep the upper triangular tiles zeros
            cute.arch.barrier(barrier_id=TMEM_ALLOC_BAR, number_of_threads=160)
            zeros_ = cute.make_rmem_tensor(BT // 2, Float32)
            zeros_.fill(0.0)
            _tcgen05.st(warp_id_ * 32, Ab_tmem_base, "32x32b", BT // 2, zeros_)

            stage_id = 0
            parity = 0

            # for sA, we can avoid bank conflict without using swizzling because
            # this is only used as tmp buffer between rmem<->smem, no gmem interactions.
            # to do so, we can put (8,8) tile contiguous in memory. logically, we are
            # partitioning (64,64) tile into 4x (16,16) tiles. to make indexing easier
            # later, we view (16,16) tile as (32,8) tile here.
            # (4,4) is (row_tile, col_tile).
            sA_layout = cute.make_layout(((8, 32), (4, 4)))
            sA = cute.make_tensor(sA_ptr, sA_layout)
            sAi = cute.make_tensor(sAi_ptr, sA_layout)

            # pre-compute ldmatrix addresses
            sA_ldsm = sA[(None, lane_id), None]
            sAi_ldsm = sAi[(None, lane_id), None]

            # init Ai smem buffer with zeros (only the first 48 rows)
            zeros_bf16 = cute.make_rmem_tensor(8, BFloat16)
            zeros_bf16.fill(0.0)
            for i in cutlass.range_constexpr(3):
                cute.copy(stsm_atom, zeros_bf16, sAi_ldsm[None, (i, warp_id_)])

            # indices for ldmatrix layout later
            row_indices = cute.make_rmem_tensor((1, 2, 1), Int32)
            row_indices[0, 0, 0] = warp_id_ * 16 + (lane_id // 4)
            row_indices[0, 1, 0] = warp_id_ * 16 + (lane_id // 4) + 8
            row_indices = row_indices.load()

            col_indices = cute.make_rmem_tensor((2, 1, 2), Int32)
            col_indices[0, 0, 0] = (lane_id % 4) * 2 + 0
            col_indices[1, 0, 0] = (lane_id % 4) * 2 + 1
            col_indices[0, 0, 1] = (lane_id % 4) * 2 + 8
            col_indices[1, 0, 1] = (lane_id % 4) * 2 + 9
            col_indices = col_indices.load()

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                eos = cu_seqlens[seq_id + 1]
                off_t = bos + chunk_id * BT

                ##### Phase 1: A = strictLower(beta * kkt * Gamma) #####
                # Ab/Abg share one tmem slot across stages. The MMA warp commits
                # tcgen05 groups in program order: KKT_i, W_i, U_i, KKT_{i+1}.
                # Waiting for KKT_i means the previous W/U commits have completed,
                # so the INV warps can safely overwrite Ab/Abg for this iteration.
                # Wait for prep warps to publish beta/gate and for KKT MMA.
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(prep_mbar + stage_id, parity)
                    cute.arch.mbarrier_wait(mma_kkt_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=INV_BAR, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                # tmem 16x256b layout / ldmatrix layout
                beta_row = cute.make_rmem_tensor(2, Float32)
                g_cu_row = cute.make_rmem_tensor(2, Float32)
                for i in cutlass.range_constexpr(2):
                    idx = warp_id_ * 16 + i * 8 + (lane_id // 4)
                    beta_row[i] = s_beta[idx, stage_id]
                    g_cu_row[i] = s_g_cu[idx, stage_id]

                # mode0 is 8 rows together
                # mode1 is top and bottom 8 rows
                # mode2 is groups of 16 rows
                beta_row = beta_row.load().reshape((1, 2, 1))
                g_cu_row = g_cu_row.load().reshape((1, 2, 1))

                # mode0 is 2 consecutive elems
                # mode1 is top and bottom 8 rows
                # mode2 is next 8 columns
                # mode3 is repeating that 16x16 tile pattern
                kkt = _tcgen05.ld(kkt_tmem, 0, "16x256b", BT // 8)
                kkt = kkt.reshape((2, 2, 2, BT // 16))

                for i in cutlass.range_constexpr(BT // 16):
                    # mode0 is 2 elems next to each other
                    # mode1 is 4 pairs of elems on 1 row
                    # mode2 is top and bottom 8 rows
                    # mode3 is next 16 columns
                    col_coord = (None, lane_id % 4, None, i)
                    s_g_cu_view = cute.make_tensor(
                        s_g_cu[None, stage_id].iterator, (2, 4, 2, BT // 16)
                    )
                    g_cu_col = s_g_cu_view[col_coord].load().reshape((2, 1, 2))

                    Gamma = cute.math.exp(g_cu_row - g_cu_col, fastmath=True)
                    A = kkt[None, None, None, i] * beta_row * Gamma

                    # strict lower mask
                    # NOTE: for OOB t position, s_beta is filled with zeros.
                    # hence, we don't need to apply bounds check for columns.
                    A_masked = cute.where(row_indices > col_indices + i * 16, A, 0.0)

                    # pack to BF16
                    # CuteDSL doesn't generate cvt.bf16x2.f32 here for some reasons
                    packed = cute.make_rmem_tensor(4, Uint32)
                    for j in cutlass.range_constexpr(4):
                        packed[j] = cvt.fp32x2_to_bf16x2(
                            A_masked[j * 2], A_masked[j * 2 + 1]
                        )

                    # store to smem
                    cute.copy(
                        stsm_atom,
                        cute.recast_tensor(packed, BFloat16),
                        sA_ldsm[None, (warp_id_, i)],
                    )

                # use sync warp instead of bar.sync because for block-diagonal inverse,
                # each warp reads its own private smem memory.
                cute.arch.sync_warp()

                ##### Phase 2: matrix inverse #####
                # we use Newton-Schulz iterations to compute the inverse
                # of the four 16x16 diagonal blocks.
                #   Ai_new = 2 Ai - Ai @ M @ Ai
                #   where M = I + A
                #
                # we do this with 2 MMAs:
                # 1. -AiM = Ai @ (-M)
                # 2. Ai_new = 2 Ai + (-AiM) @ Ai
                zeros_f32 = cute.make_rmem_tensor(4, Float32)
                zeros_f32.fill(0.0)

                Ai_bf16 = cute.make_rmem_tensor(8, BFloat16)
                mma_B_bf16 = cute.make_rmem_tensor(8, BFloat16)
                M_bf16 = cute.make_rmem_tensor(8, BFloat16)
                acc = cute.make_rmem_tensor((4, 2), Float32)

                # share the same storage
                Ai = cute.recast_tensor(Ai_bf16, Uint32)
                mma_B = cute.logical_divide(cute.recast_tensor(mma_B_bf16, Uint32), 2)
                M = cute.logical_divide(cute.recast_tensor(M_bf16, Uint32), 2)

                # construct rmem-backed identity matrix
                eye = cute.make_rmem_tensor(4, Uint32)
                eye[0] = Uint32(lane_id % 9 == 0) * Uint32(0x00003F80) + Uint32(
                    lane_id % 9 == 4
                ) * Uint32(0x3F800000)
                eye[1] = 0
                eye[2] = 0
                eye[3] = eye[0]

                # initial guess: Ai = I-A
                cute.copy(ldsm_atom, sA_ldsm[None, (warp_id_, warp_id_)], Ai_bf16)
                for i in cutlass.range_constexpr(4):
                    Ai[i] = _bf16x2_sub(eye[i], Ai[i])

                # (4, 2)
                Ai_f32 = cute.logical_divide(cvt.bf16x2_to_fp32x2(Ai), 4)

                # M is holding -(I+A), stay constant throughout the iterations
                cute.copy(ldsm_trans_atom, sA_ldsm[None, (warp_id_, warp_id_)], M_bf16)
                for i in cutlass.range_constexpr(4):
                    M[i] = _bf16x2_sub(_bf16x2_neg(eye[i]), M[i])

                # 3 rounds of Newton-Schulz
                for _ in cutlass.range_constexpr(3):
                    # First MMA: -AiM = Ai @ (-M)
                    cute.copy(stsm_atom, Ai_bf16, sA_ldsm[None, (warp_id_, warp_id_)])
                    cute.arch.sync_warp()
                    acc[None, 0] = mma_bf16(Ai, M[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, M[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))

                    # Second MMA: Ai_new = 2Ai + (-AiM) @ Ai
                    for j in cutlass.range(8, vectorize=True):
                        Ai_f32[j] *= 2.0
                    cute.copy(
                        ldsm_trans_atom,
                        sA_ldsm[None, (warp_id_, warp_id_)],
                        mma_B_bf16,
                    )
                    Ai_f32[None, 0] = mma_bf16(Ai, mma_B[None, 0], Ai_f32[None, 0])
                    Ai_f32[None, 1] = mma_bf16(Ai, mma_B[None, 1], Ai_f32[None, 1])
                    Ai_bf16.store(Ai_f32.load().to(BFloat16))

                cute.copy(stsm_atom, Ai_bf16, sAi_ldsm[None, (warp_id_, warp_id_)])
                store_ab_abg(
                    Ai_f32,
                    s_beta[None, stage_id],
                    s_beta_g[None, stage_id],
                    warp_id_,
                    lane_id,
                    warp_id_,
                    Ab_tmem_base,
                )
                cute.arch.barrier(barrier_id=INV_BAR, number_of_threads=128)

                # off-diagonal by 1
                # given
                # [ Ai00               ]
                # [  A10 Ai11          ]
                # [  A20  A21 Ai22     ]
                # [  A30  A31  A32 Ai33]
                # warp1: Ai10 = -Ai11 @ A10 @ Ai00
                # warp2: Ai21 = -Ai22 @ A21 @ Ai11
                # warp3: Ai32 = -Ai33 @ A32 @ Ai22
                if warp_id_ >= 1:
                    neg_Ai = cute.make_rmem_tensor(4, Uint32)
                    for i in cutlass.range_constexpr(4):
                        neg_Ai[i] = _bf16x2_neg(Ai[i])

                    cute.copy(
                        ldsm_trans_atom,
                        sA_ldsm[None, (warp_id_, warp_id_ - 1)],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(neg_Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(neg_Ai, mma_B[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))

                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[None, (warp_id_ - 1, warp_id_ - 1)],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))
                    store_ab_abg(
                        acc,
                        s_beta[None, stage_id],
                        s_beta_g[None, stage_id],
                        warp_id_,
                        lane_id,
                        warp_id_ - 1,
                        Ab_tmem_base,
                    )
                    cute.copy(
                        stsm_atom,
                        Ai_bf16,
                        sAi_ldsm[None, (warp_id_, warp_id_ - 1)],
                    )
                cute.arch.barrier(barrier_id=INV_BAR, number_of_threads=128)

                # off-diagonal by 2
                # warp2: Ai20 = -Ai22 @ (A20 @ Ai00 + A21 @ Ai10)
                # warp3: Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
                if warp_id_ >= 2:
                    tile_col = warp_id_ - 2
                    cute.copy(
                        ldsm_atom,
                        sA_ldsm[None, (warp_id_, tile_col)],
                        Ai_bf16,
                    )
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[None, (tile_col, tile_col)],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)

                    cute.copy(
                        ldsm_atom, sA_ldsm[None, (warp_id_, tile_col + 1)], Ai_bf16
                    )
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[None, (tile_col + 1, tile_col)],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], acc[None, 0])
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], acc[None, 1])

                    tmp = cute.make_rmem_tensor(8, BFloat16)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[None, (warp_id_, tile_col)])
                    cute.arch.sync_warp()

                    cute.copy(ldsm_atom, sAi_ldsm[None, (warp_id_, warp_id_)], Ai_bf16)
                    for i in cutlass.range_constexpr(4):
                        Ai[i] = _bf16x2_neg(Ai[i])
                    cute.copy(
                        ldsm_trans_atom,
                        sAi_ldsm[None, (warp_id_, tile_col)],
                        mma_B_bf16,
                    )
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[None, (warp_id_, tile_col)])
                    store_ab_abg(
                        acc,
                        s_beta[None, stage_id],
                        s_beta_g[None, stage_id],
                        warp_id_,
                        lane_id,
                        tile_col,
                        Ab_tmem_base,
                    )
                cute.arch.barrier(barrier_id=INV_BAR, number_of_threads=128)

                # off-diagonal by 3
                # warp3: Ai30 = -Ai33 @ (A30 @ Ai00 + A31 @ Ai10 + A32 @ Ai20)
                if warp_id_ == 3:
                    cute.copy(ldsm_atom, sA_ldsm[None, (3, 0)], Ai_bf16)
                    cute.copy(ldsm_trans_atom, sAi_ldsm[None, (0, 0)], mma_B_bf16)
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)

                    for i in cutlass.range_constexpr(1, 3):
                        cute.copy(ldsm_atom, sA_ldsm[None, (3, i)], Ai_bf16)
                        cute.copy(ldsm_trans_atom, sAi_ldsm[None, (i, 0)], mma_B_bf16)
                        acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], acc[None, 0])
                        acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], acc[None, 1])

                    tmp = cute.make_rmem_tensor(8, BFloat16)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[None, (3, 0)])
                    cute.arch.sync_warp()

                    cute.copy(ldsm_atom, sAi_ldsm[None, (3, 3)], Ai_bf16)
                    for i in cutlass.range_constexpr(4):
                        Ai[i] = _bf16x2_neg(Ai[i])
                    cute.copy(ldsm_trans_atom, sAi_ldsm[None, (3, 0)], mma_B_bf16)
                    acc[None, 0] = mma_bf16(Ai, mma_B[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, mma_B[None, 1], zeros_f32)
                    tmp.store(acc.load().to(BFloat16))
                    cute.copy(stsm_atom, tmp, sAi_ldsm[None, (3, 0)])
                    store_ab_abg(
                        acc,
                        s_beta[None, stage_id],
                        s_beta_g[None, stage_id],
                        warp_id_,
                        lane_id,
                        0,
                        Ab_tmem_base,
                    )
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(inv_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        else:
            # epi warps
            stage_id = 0
            parity = 0

            # ((BT, num_global_chunks), V_dim)
            gU_tiles = cute.logical_divide(
                U_tma.tma_tensor[None, head_id, None], (BT, None)
            )
            gW_tiles = cute.logical_divide(
                W_tma.tma_tensor[None, head_id, None], (BT, None)
            )

            # sW shape: [BT, (64, K_dim/64)]
            # sW_view shape: [(8, 2), (4, K_dim/64)]
            s_row = warp_id * 16 + lane_id % 16  # select the rows of [16,16] tile
            sW_view = cute.zipped_divide(
                sW[s_row, None],
                tiler=cute.make_layout((8, 2)),
            )
            sU_view = cute.zipped_divide(
                sU[s_row, None],
                tiler=cute.make_layout((8, 2)),
            )

            # select the 8 columns within [16,16] tile
            sW_view = sW_view[(None, lane_id // 16), None]
            sU_view = sU_view[(None, lane_id // 16), None]

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                # wait for W MMA + previous TMA store to finish
                U_tmem = U_tmem_base + V_dim * stage_id
                if warp_id == 0:
                    cute.arch.mbarrier_wait(mma_w_mbar + stage_id, parity)
                elif warp_id == 1:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                w_f32 = _tcgen05.ld(warp_id * 32 + 16, U_tmem, "16x256b", K_dim // 8)
                _tcgen05.wait_ld()
                w_bf16 = cute.make_rmem_tensor((8, K_dim // 16), BFloat16)
                w_bf16.store(w_f32.to(BFloat16))
                cute.copy(stsm_atom, w_bf16, sW_view)

                # wait for U MMA + issue W TMA store
                cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 0:
                    cute.arch.mbarrier_wait(mma_u_mbar + stage_id, parity)
                elif warp_id == 1:
                    # don't need to commit
                    simple_tma_copy(
                        W_tma.atom, sW, gW_tiles[(None, global_chunk_id), None]
                    )
                cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                u_f32 = _tcgen05.ld(warp_id * 32, U_tmem, "16x256b", V_dim // 8)
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(epi_mbar + stage_id)
                u_bf16 = cute.make_rmem_tensor((8, V_dim // 16), BFloat16)
                u_bf16.store(u_f32.to(BFloat16))
                cute.copy(stsm_atom, u_bf16, sU_view)

                cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 1:
                    simple_tma_copy(
                        U_tma.atom, sU, gU_tiles[(None, global_chunk_id), None]
                    )
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

    @cache
    @staticmethod
    def compile(H: int, Hv: int, K_dim: int, V_dim: int, num_stages: int = 2):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()
        total_chunks_n = cute.sym_int()
        num_sequences = cute.sym_int()

        K = make_fake_tensor(BFloat16, (total_t, H, K_dim), divisibility=16)
        V = make_fake_tensor(BFloat16, (total_t, Hv, V_dim), divisibility=16)
        U = make_fake_tensor(BFloat16, (pad_t, Hv, V_dim), divisibility=16)
        W = make_fake_tensor(BFloat16, (pad_t, Hv, K_dim), divisibility=16)
        g = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        beta = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        g_cu = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        cu_seqlens = make_fake_tensor(Int32, (num_sequences,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (total_chunks_n, 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = Sm100ChunkUWKernel(H, Hv, K_dim, V_dim, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            K,
            V,
            U,
            W,
            g,
            beta,
            g_cu,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            Int32(148),
            stream,
            options="--enable-tvm-ffi",
        )


def kkt_inv_uw_cutedsl(
    K: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    g_cu: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor,
    num_sms: int = 148,
) -> None:
    _, Hv, V_dim = V.shape
    _, H, K_dim = K.shape

    Sm100ChunkUWKernel.compile(H, Hv, K_dim, V_dim)(
        K,
        V,
        U,
        W,
        g,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms,
    )
