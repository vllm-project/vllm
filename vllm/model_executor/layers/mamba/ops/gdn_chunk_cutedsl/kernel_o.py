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


class Sm100ChunkOKernel:
    def __init__(
        self,
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        BT: int = 64,
        num_stages: int = 2,
    ) -> None:
        assert Hv % H == 0
        assert K_dim == 128
        assert V_dim == 128
        assert BT == 64
        self.H = H
        self.Hv = Hv
        self.K_dim = K_dim
        self.V_dim = V_dim
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
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            slayout,
            cta_tiler=(self.BT, 1, dim),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def _make_h_tma_args(
        self,
        tensor: cute.Tensor,
        op: cpasync.TmaCopyOp,
        stages: cutlass.Constexpr[int],
    ):
        num_elems = 128 // (tensor.element_type.width // 8)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (1, self.V_dim, (num_elems, self.K_dim // num_elems), stages),
            stride=(0, num_elems, (1, self.V_dim * num_elems), self.V_dim * self.K_dim),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, num_elems)),
            slayout,
            cta_tiler=(1, self.V_dim, self.K_dim),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v_new_chunks: cute.Tensor,
        h: cute.Tensor,
        g_cu: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        scale: Float32,
        num_sms: Int32,
        stream: CUstream,
    ):
        grid = (num_sms // self.Hv, self.Hv, 1)
        block = (self.num_warps * 32, 1, 1)
        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()
        Q_args = self._make_bf16_tma_args(q, self.K_dim, tma_g2s, self.num_stages)
        K_args = self._make_bf16_tma_args(k, self.K_dim, tma_g2s, self.num_stages)
        V_args = self._make_bf16_tma_args(
            v_new_chunks, self.V_dim, tma_g2s, self.num_stages
        )
        H_args = self._make_h_tma_args(h, tma_g2s, self.num_stages)
        O_args = self._make_bf16_tma_args(o, self.V_dim, tma_s2g, 1)
        self.kernel(
            Q_args,
            K_args,
            V_args,
            H_args,
            O_args,
            g_cu,
            o,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            scale,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        Q_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        K_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        V_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        H_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        O_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        g_cu: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, v_head_id, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BT = self.BT
        K_dim = self.K_dim
        V_dim = self.V_dim
        num_stages = self.num_stages

        heads_per_qk = self.Hv // self.H
        k_head_id = v_head_id // heads_per_qk
        num_global_chunks = total_chunks[0]

        Q_tma_atom, tmaQ, sQ_layout = Q_args
        K_tma_atom, tmaK, sK_layout = K_args
        V_tma_atom, tmaV, sV_layout = V_args
        H_tma_atom, tmaH, sH_layout = H_args
        O_tma_atom, tmaO, sO_layout = O_args

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_tensor(smem, BFloat16, sQ_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, sK_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, sV_layout)[None, 0, None, None]
        sH = allocate_tensor(smem, BFloat16, sH_layout)[0, None, None, None]
        sO = allocate_tensor(smem, BFloat16, sO_layout)[None, 0, None, 0]

        s_g_cu = smem.allocate_array(Float32, BT)
        qk_full_mbar = smem.allocate_array(Int64, num_stages)
        hv_full_mbar = smem.allocate_array(Int64, num_stages)
        qk_empty_mbar = smem.allocate_array(Int64, num_stages)
        pv_mma_mbar = smem.allocate_array(Int64, num_stages)
        qk_mbar = smem.allocate_array(Int64, 1)
        mask_mbar = smem.allocate_array(Int64, 1)
        epi_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        qk_tmem = 0
        p_tmem = 64
        out_tmem = 128
        qh_tmem = 256

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(qk_full_mbar + i, 1)
                    cute.arch.mbarrier_init(qk_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(hv_full_mbar + i, 1)
                    cute.arch.mbarrier_init(pv_mma_mbar + i, 1)
                cute.arch.mbarrier_init(qk_mbar, 1)
                cute.arch.mbarrier_init(mask_mbar, 128)
                cute.arch.mbarrier_init(epi_mbar, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 9:
            cpasync.prefetch_descriptor(Q_tma_atom)
            cpasync.prefetch_descriptor(K_tma_atom)
            cpasync.prefetch_descriptor(V_tma_atom)
            cpasync.prefetch_descriptor(H_tma_atom)
        cute.arch.sync_threads()

        if warp_id == 9:
            # TMA warp
            stage_id = 0
            parity = 1

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]

                # copy Q and K
                q_tile = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaQ[None, k_head_id, None]),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                k_tile = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaK[None, k_head_id, None]),
                    tiler=(BT, K_dim),
                    coord=(chunk_id, 0),
                )
                mbar = qk_full_mbar + stage_id

                cute.arch.mbarrier_wait(qk_empty_mbar + stage_id, parity)

                with cute.arch.elect_one():
                    STAGE_SIZE = BT * (K_dim + K_dim) * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(Q_tma_atom, q_tile, sQ[None, None, stage_id], mbar)
                simple_tma_copy(K_tma_atom, k_tile, sK[None, None, stage_id], mbar)

                # copy H and V
                gH = tmaH[global_chunk_id * self.Hv + v_head_id, None, None]
                gV = cute.local_tile(
                    tmaV[None, v_head_id, None],
                    tiler=(BT, V_dim),
                    coord=(global_chunk_id, 0),
                )
                mbar = hv_full_mbar + stage_id

                cute.arch.mbarrier_wait(pv_mma_mbar + stage_id, parity)

                with cute.arch.elect_one():
                    H_STAGE_SIZE = V_dim * K_dim * 2
                    V_STAGE_SIZE = BT * V_dim * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar, H_STAGE_SIZE + V_STAGE_SIZE
                    )
                simple_tma_copy(
                    H_tma_atom, gH, sH[None, None, stage_id], mbar, EVICT_FIRST
                )
                simple_tma_copy(
                    V_tma_atom, gV, sV[None, None, stage_id], mbar, EVICT_FIRST
                )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id == 8:
            # MMA warp
            _tcgen05.alloc(taddr)

            # LBO=BT*128 is ignored for K-major
            sdesc_template = _tcgen05.make_sdesc_128B_swizzle(BT * 128)
            qk_idesc = _tcgen05.make_bf16_idesc(BT, BT)
            qh_idesc = _tcgen05.make_bf16_idesc(BT, V_dim)
            pv_idesc = _tcgen05.make_bf16_idesc(BT, V_dim, transpose_B=True)

            stage_id = 0
            tma_parity = 0
            mask_parity = 0

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                qaddr = sQ[None, None, stage_id].iterator.toint()
                kaddr = sK[None, None, stage_id].iterator.toint()
                haddr = sH[None, None, stage_id].iterator.toint()
                vaddr = sV[None, None, stage_id].iterator.toint()
                qdesc_base = sdesc_template | (qaddr >> 4)
                kdesc_base = sdesc_template | (kaddr >> 4)
                hdesc_base = sdesc_template | (haddr >> 4)
                vdesc_base = sdesc_template | (vaddr >> 4)

                ##### 1st MMA: Q @ K.T #####
                cute.arch.mbarrier_wait(epi_mbar, mask_parity ^ 1)
                cute.arch.mbarrier_wait(qk_full_mbar + stage_id, tma_parity)
                _tcgen05.fence_after_thread_sync()

                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(K_dim // BT):
                        for j in cutlass.range_constexpr(BT // 16):
                            qdesc = qdesc_base | ((i * BT * 128 + j * 32) >> 4)
                            kdesc = kdesc_base | ((i * BT * 128 + j * 32) >> 4)
                            _tcgen05.mma_f16(
                                qk_tmem, qdesc, kdesc, qk_idesc, (i > 0) or (j > 0)
                            )
                    _tcgen05.commit(qk_mbar)

                ##### 2nd MMA: Q @ H.T #####
                cute.arch.mbarrier_wait(hv_full_mbar + stage_id, tma_parity)
                _tcgen05.fence_after_thread_sync()
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(K_dim // BT):
                        for j in cutlass.range_constexpr(BT // 16):
                            qdesc = qdesc_base | ((i * BT * 128 + j * 32) >> 4)
                            hdesc = hdesc_base | ((i * V_dim * 128 + j * 32) >> 4)
                            _tcgen05.mma_f16(
                                qh_tmem, qdesc, hdesc, qh_idesc, (i > 0) or (j > 0)
                            )
                    _tcgen05.commit(qk_empty_mbar + stage_id)

                ##### 3rd MMA: P @ V #####
                cute.arch.mbarrier_wait(mask_mbar, mask_parity)
                _tcgen05.fence_after_thread_sync()
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(BT // 16):
                        vdesc = vdesc_base | ((i * 16 * 128) >> 4)
                        _tcgen05.mma_ts_f16(
                            out_tmem, p_tmem + i * 8, vdesc, pv_idesc, i > 0
                        )
                    _tcgen05.commit(pv_mma_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    tma_parity ^= 1
                mask_parity ^= 1

            # wait for epilogue to finish for deallocation
            cute.arch.mbarrier_wait(epi_mbar, mask_parity ^ 1)
            _tcgen05.dealloc()

        elif warp_id >= 4:
            # masking warps
            warp_id_ = warp_id % 4
            tid_ = tid % 128
            row0 = warp_id_ * 16 + lane_id // 4
            row1 = row0 + 8

            parity = 0

            # for ldmatrix layout later
            row_indices = cute.make_rmem_tensor(2, Int32)
            row_indices[0] = warp_id_ * 16 + lane_id // 4
            row_indices[1] = warp_id_ * 16 + lane_id // 4 + 8
            row_indices = row_indices.load().reshape((1, 2))

            col_indices = cute.make_rmem_tensor(2, Int32)
            col_indices[0] = (lane_id % 4) * 2
            col_indices[1] = (lane_id % 4) * 2 + 1
            col_indices = col_indices.load().reshape((2, 1))

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                if tid_ < BT:
                    seq_id = chunk_indices[global_chunk_id, 0]
                    chunk_id = chunk_indices[global_chunk_id, 1]
                    bos = cu_seqlens[seq_id]
                    eos = cu_seqlens[seq_id + 1]

                    t_ = bos + chunk_id * BT + tid_
                    s_g_cu[tid_] = g_cu[t_, v_head_id] if t_ < eos else Float32(0.0)

                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(qk_mbar, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()
                qk = _tcgen05.ld(warp_id_ * 32, qk_tmem, "16x256b", BT // 8)
                qk = qk.reshape((2, 2, BT // 8))
                _tcgen05.wait_ld()

                g_cu_rows = cute.make_rmem_tensor(2, Float32)
                g_cu_rows[0] = s_g_cu[row0]
                g_cu_rows[1] = s_g_cu[row1]
                g_cu_rows = g_cu_rows.load().reshape((1, 2))

                for i in cutlass.range_constexpr(BT // 8):
                    col = i * 8 + (lane_id % 4) * 2
                    g_cu_cols = cute.make_rmem_tensor(2, Float32)
                    g_cu_cols[0] = s_g_cu[col]
                    g_cu_cols[1] = s_g_cu[col + 1]
                    g_cu_cols = g_cu_cols.load().reshape((2, 1))

                    # apply causal mask
                    Gamma = cute.math.exp(g_cu_rows - g_cu_cols, fastmath=True)
                    tmp = qk[None, None, i] * Gamma
                    tmp = cute.where(row_indices >= col_indices + i * 8, tmp, 0.0)

                    # CuteDSL can't emit cvt.bf16x2.f32 here
                    attn_lo = cute.make_rmem_tensor(2, Uint32)
                    attn_lo[0] = cvt.fp32x2_to_bf16x2(tmp[0, 0], tmp[1, 0])
                    attn_lo[1] = cvt.fp32x2_to_bf16x2(tmp[0, 1], tmp[1, 1])
                    _tcgen05.st(warp_id_ * 32, p_tmem + i * 4, "16x128b", 1, attn_lo)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(mask_mbar)

                parity ^= 1

        else:
            # epilogue warps
            row0 = warp_id * 16 + lane_id // 4
            row1 = row0 + 8

            stage_id = 0
            mma_parity = 0

            op = cute.nvgpu.CopyUniversalOp()
            cp_4B = cute.make_copy_atom(op, BFloat16, num_bits_per_copy=32)
            stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False)
            stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)

            # ldmatrix layout
            # [total_seq_len, ((2, 4, WIDTH/8), V_DIM/WIDTH)]
            WIDTH = 64
            o_view = cute.logical_divide(
                o[None, v_head_id, None],
                (None, cute.make_layout((2, 4, WIDTH // 8))),
            )
            # select lane: [total_seq_len, 2, WIDTH/8, V_DIM/WIDTH]
            o_view = o_view[None, ((None, lane_id % 4, None), None)]
            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                eos = cu_seqlens[seq_id + 1]
                chunk_start = bos + chunk_id * BT
                full_chunk = chunk_start + BT <= eos

                g_cu_rows = cute.make_rmem_tensor(2, Float32)
                g_cu_rows.fill(0.0)

                if chunk_start + row0 < eos:
                    g_cu_rows[0] = cute.math.exp(
                        g_cu[chunk_start + row0, v_head_id], fastmath=True
                    )
                if chunk_start + row1 < eos:
                    g_cu_rows[1] = cute.math.exp(
                        g_cu[chunk_start + row1, v_head_id], fastmath=True
                    )
                g_cu_rows = g_cu_rows.load().reshape((1, 2, 1))

                if warp_id == 0:
                    cute.arch.mbarrier_wait(pv_mma_mbar + stage_id, mma_parity)
                elif warp_id == 3 and full_chunk:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                if full_chunk:
                    # use TMA store: tmem->rmem->smem->gmem
                    for i in cutlass.range_constexpr(V_dim // WIDTH):
                        qh = _tcgen05.ld(
                            warp_id * 32, qh_tmem + i * WIDTH, "16x256b", WIDTH // 8
                        )
                        pv = _tcgen05.ld(
                            warp_id * 32, out_tmem + i * WIDTH, "16x256b", WIDTH // 8
                        )
                        _tcgen05.wait_ld()
                        if i == V_dim // WIDTH - 1:
                            _tcgen05.fence_before_thread_sync()
                            cute.arch.mbarrier_arrive(epi_mbar)

                        qh = qh.reshape((2, 2, WIDTH // 8))
                        pv = pv.reshape((2, 2, WIDTH // 8))

                        out_f32 = scale * (g_cu_rows * qh + pv)
                        out_bf16 = cute.make_rmem_tensor((8, WIDTH // 16), BFloat16)
                        out_bf16.store(out_f32.to(BFloat16).reshape((8, WIDTH // 16)))

                        # TODO: issue single cute.copy()
                        for j in cutlass.range_constexpr(WIDTH // 16):
                            s_row = warp_id * 16 + lane_id % 16
                            s_col = i * (WIDTH // 8) + j * 2 + lane_id // 16
                            sO_tile = cute.local_tile(sO[s_row, None], (8,), (s_col,))
                            cute.copy(stsm_atom, out_bf16[None, j], sO_tile)

                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    fence_before_tma_store()
                    if warp_id == 3:
                        gO = cute.local_tile(
                            cute.domain_offset((bos, 0), tmaO[None, v_head_id, None]),
                            tiler=(BT, V_dim),
                            coord=(chunk_id, 0),
                        )
                        simple_tma_copy(O_tma_atom, sO, gO)
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_commit_group()

                else:
                    # direct gmem store
                    for i in cutlass.range_constexpr(V_dim // WIDTH):
                        qh = _tcgen05.ld(
                            warp_id * 32, qh_tmem + i * WIDTH, "16x256b", WIDTH // 8
                        )
                        pv = _tcgen05.ld(
                            warp_id * 32, out_tmem + i * WIDTH, "16x256b", WIDTH // 8
                        )
                        _tcgen05.wait_ld()
                        if i == V_dim // WIDTH - 1:
                            _tcgen05.fence_before_thread_sync()
                            cute.arch.mbarrier_arrive(epi_mbar)

                        qh = qh.reshape((2, 2, WIDTH // 8))
                        pv = pv.reshape((2, 2, WIDTH // 8))

                        out_f32 = scale * (g_cu_rows * qh + pv)
                        out_bf16 = cute.make_rmem_tensor((2, 2, WIDTH // 8), BFloat16)
                        out_bf16.store(out_f32.to(BFloat16))

                        if chunk_start + row0 < eos:
                            cute.copy(
                                cp_4B,
                                out_bf16[None, 0, None],
                                o_view[chunk_start + row0, None, None, i],
                            )
                        if chunk_start + row1 < eos:
                            cute.copy(
                                cp_4B,
                                out_bf16[None, 1, None],
                                o_view[chunk_start + row1, None, None, i],
                            )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    mma_parity ^= 1

    @cache
    @staticmethod
    def compile(
        H: int,
        Hv: int,
        K_dim: int,
        V_dim: int,
        BT: int = 64,
        num_stages: int = 2,
    ):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()
        total_chunks_n = cute.sym_int()
        h_outer_n = cute.sym_int()
        cu_entries = cute.sym_int()

        q = make_fake_tensor(BFloat16, (total_t, H, K_dim), divisibility=16)
        k = make_fake_tensor(BFloat16, (total_t, H, K_dim), divisibility=16)
        v_new = make_fake_tensor(BFloat16, (pad_t, Hv, V_dim), divisibility=16)
        h_flat = make_fake_tensor(BFloat16, (h_outer_n, V_dim, K_dim), divisibility=16)
        g_cu = make_fake_tensor(Float32, (total_t, Hv), divisibility=4)
        o = make_fake_tensor(BFloat16, (total_t, Hv, V_dim), divisibility=16)
        cu_seqlens = make_fake_tensor(Int32, (cu_entries,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (total_chunks_n, 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = Sm100ChunkOKernel(
            H,
            Hv,
            K_dim,
            V_dim,
            BT,
            num_stages,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k,
            v_new,
            h_flat,
            g_cu,
            o,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            Float32(1.0),
            Int32(148),
            stream,
            options="--enable-tvm-ffi",
        )


def o_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_chunks: torch.Tensor,
    h: torch.Tensor,
    g_cu: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor,
    scale: float,
    num_sms: int = 148,
) -> None:
    _, H, K_dim = q.shape
    _, Hv, V_dim = o.shape

    Sm100ChunkOKernel.compile(H, Hv, K_dim, V_dim)(
        q,
        k,
        v_new_chunks.view(-1, Hv, V_dim),
        h.view(-1, V_dim, K_dim),
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        float(scale),
        num_sms,
    )
