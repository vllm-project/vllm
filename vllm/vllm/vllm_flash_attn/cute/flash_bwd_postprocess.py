# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_postprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
from collections.abc import Callable
from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum
from quack import layout_utils, sm90_utils

from vllm.vllm_flash_attn.cute import ampere_helpers as sm80_utils
from vllm.vllm_flash_attn.cute import copy_utils, utils
from vllm.vllm_flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPostprocess:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[80, 90, 100],
        tile_m: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ):
        """
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        """
        self.dtype = dtype
        self.tile_m = tile_m
        assert arch in [80, 90, 100], (
            "Only Ampere (80), Hopper (90), and Blackwell (100) are supported"
        )
        self.arch = arch
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.num_threads = num_threads
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.dQ_swapAB = dQ_swapAB

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    def _get_tiled_mma(self):
        if const_expr(self.arch == 80):
            num_mma_warps = self.num_threads // 32
            atom_layout_dQ = (
                (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1)
                if const_expr(not self.dQ_swapAB)
                else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                atom_layout_dQ,
                permutation_mnk=(atom_layout_dQ[0] * 16, atom_layout_dQ[1] * 16, 16),
            )
        elif const_expr(self.arch == 90):
            num_mma_warp_groups = self.num_threads // 128
            atom_layout_dQ = (
                self.AtomLayoutMdQ,
                num_mma_warp_groups // self.AtomLayoutMdQ,
            )
            tiler_mn_dQ = (
                self.tile_m // atom_layout_dQ[0],
                self.tile_hdim // atom_layout_dQ[1],
            )
            tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.K,  # These don't matter, we only care about the accum
                warpgroup.OperandMajorMode.K,
                Float32,
                atom_layout_mnk=(
                    atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1]
                )
                + (1,),
                tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1],
            )
        else:
            cta_group = tcgen05.CtaGroup.ONE
            tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                tcgen05.OperandMajorMode.MN,  # dS_major_mode
                tcgen05.OperandMajorMode.MN,  # Kt_major_mode
                Float32,
                cta_group,
                (self.tile_m, self.tile_hdim),
            )
        if const_expr(self.arch in [80, 90]):
            assert self.num_threads == tiled_mma.size
        return tiled_mma

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (
            self.tile_m * self.tile_hdim // async_copy_elems_accum
        ) % self.num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        num_s2r_copy_elems = 1 if const_expr(self.arch == 80) else 4
        if const_expr(self.arch == 80):
            self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
                Float32, self.num_threads, num_s2r_copy_elems
            )
            self.sdQaccum_layout = cute.make_layout(self.tile_m * self.tile_hdim)
        elif const_expr(self.arch == 90):
            num_threads_per_warp_group = 128
            num_mma_warp_groups = self.num_threads // 128
            self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                ),
                cute.make_layout(
                    (num_threads_per_warp_group, num_mma_warp_groups)
                ),  # thr_layout
                cute.make_layout(128 // Float32.width),  # val_layout
            )
            self.sdQaccum_layout = cute.make_layout(
                (
                    self.tile_m * self.tile_hdim // num_mma_warp_groups,
                    num_mma_warp_groups,
                )
            )
        else:
            self.dQ_reduce_ncol = 32
            dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
            assert self.num_threads == 128  # TODO: currently hard-coded
            self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
                Float32, self.num_threads, num_s2r_copy_elems
            )
            self.sdQaccum_layout = cute.make_layout(
                (
                    self.tile_m * self.tile_hdim // dQaccum_reduce_stage,
                    dQaccum_reduce_stage,
                )
            )

        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, self.tile_hdim, self.num_threads
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQ
        # ///////////////////////////////////////////////////////////////////////////////
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        mma_shape_n = self.tiled_mma.get_tile_size(1)
        if const_expr(self.arch == 80):
            sdQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, mma_shape_n)
            self.sdQ_layout = cute.tile_to_shape(
                sdQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1)
            )
        elif const_expr(self.arch == 90):
            self.sdQ_layout = sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim)
            )
        else:
            # TODO: this is hard-coded for hdim 128
            self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), 1
            )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]

        self.tiled_mma = self._get_tiled_mma()
        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mdQ.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
            num_block = cute.ceil_div(mdQ.shape[0], self.tile_m)
        else:
            TileScheduler = SingleTileScheduler
            num_head = mdQ.shape[2]
            num_batch = mdQ.shape[0]
            num_block = cute.ceil_div(mdQ.shape[1], self.tile_m)

        tile_sched_args = TileSchedulerArguments(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mdQ.shape[2],
            headdim_v=0,
            total_q=mdQ.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # grid_dim: (m_block, num_head, batch_size)
        self.kernel(
            mdQaccum,
            mdQ,
            mCuSeqlensQ,
            mSeqUsedQ,
            scale,
            self.tiled_mma,
            self.dQ_swapAB,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        mCuSeqlensQ: cute.Tensor | None,
        mSeqUsedQ: cute.Tensor | None,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(
            cutlass.Float32, sdQaccum_layout, byte_alignment=1024
        )
        sdQaccum_flat = cute.make_tensor(
            sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum))
        )
        if const_expr(self.arch in [80, 90]):
            sdQ = cute.make_tensor(
                cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout
            )
        else:
            # extra stage dimension
            sdQ = cute.make_tensor(
                cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype),
                sdQ_layout.outer,
            )[None, None, 0]
        sdQt = layout_utils.transpose_view(sdQ)

        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////

            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mdQ.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=None,
            )
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQ_cur = mdQ[batch_idx, None, head_idx, None]
                mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
                head_dim = mdQ.shape[3]
            else:
                padded_offset_q = seqlen.offset_q + batch_idx * self.tile_m
                if cutlass.const_expr(self.arch >= 90):
                    padded_offset_q = padded_offset_q // self.tile_m * self.tile_m
                mdQ_cur = cute.domain_offset(
                    (seqlen.offset_q, 0), mdQ[None, head_idx, None]
                )
                mdQaccum_cur = cute.domain_offset(
                    (padded_offset_q * self.tile_hdim,), mdQaccum[head_idx, None]
                )
                head_dim = mdQ.shape[2]

                # HACK: Compiler doesn't seem to recognize that padding
                # by padded_offset_q * self.tile_hdim keeps alignment
                # since statically divisible by 4

                mdQaccum_cur_ptr = cute.make_ptr(
                    dtype=mdQaccum_cur.element_type,
                    value=mdQaccum_cur.iterator.toint(),
                    mem_space=mdQaccum_cur.iterator.memspace,
                    assumed_align=mdQaccum.iterator.alignment,
                )
                mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)

            gdQaccum = cute.local_tile(
                mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,)
            )
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

            seqlen_q = seqlen.seqlen_q
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)

            # Step 1: load dQaccum from gmem to smem
            g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem
            s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
            tile_shape = (self.tile_m, self.tile_hdim)
            acc = None
            tiled_copy_t2r = None
            if const_expr(self.arch in [80, 90]):
                acc_shape = tiled_mma.partition_shape_C(
                    tile_shape if const_expr(not dQ_swapAB) else tile_shape[::-1]
                )
                acc = cute.make_fragment(acc_shape, cutlass.Float32)
                assert cute.size(acc) == cute.size(tdQsdQaccum)
            else:
                thr_mma = tiled_mma.get_slice(0)  # 1-CTA
                dQacc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim))
                tdQtdQ = tiled_mma.make_fragment_C(dQacc_shape)
                tdQcdQ = thr_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_hdim))
                )
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(
                        tcgen05.copy.Repetition(self.dQ_reduce_ncol)
                    ),
                    Float32,
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
                acc = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
            tdQrdQaccum = cute.make_tensor(
                acc.iterator, cute.make_layout(tdQsdQaccum.shape)
            )
            cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
            # Convert tdQrdQaccum from fp32 to fp16/bf16
            rdQ = cute.make_fragment_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: Copy dQ from register to smem
            cute.arch.barrier()  # make sure all threads have finished loading dQaccum
            if const_expr(self.arch in [80, 90]):
                copy_atom_r2s_dQ = utils.get_smem_store_atom(
                    self.arch, self.dtype, transpose=self.dQ_swapAB
                )
                tiled_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma)
            else:
                # copy_atom_r2s_dQ = sm100_utils_basic.get_smem_store_op(
                #     LayoutEnum.ROW_MAJOR, self.dtype, Float32, tiled_copy_t2r,
                # )
                # tiled_copy_r2s_dQ = cute.make_tiled_copy_D(copy_atom_r2s_dQ, tiled_copy_t2r)
                thr_layout_r2s_dQ = cute.make_layout(
                    (self.num_threads, 1)
                )  # 128 threads
                val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
                copy_atom_r2s_dQ = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=128,
                )
                tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
                    copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
                )
            thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            if const_expr(self.arch in [80, 90]):
                taccdQrdQ = thr_copy_r2s_dQ.retile(rdQ)
            else:
                taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
                taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
            taccdQsdQ = thr_copy_r2s_dQ.partition_D(
                sdQ if const_expr(not self.dQ_swapAB) else sdQt
            )
            cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)

            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
            cute.arch.barrier()  # make sure all smem stores are done
            gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
            tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
            tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
            # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
            cute.autovec_copy(tdQsdQ, tdQrdQ)

            # Step 5: Copy dQ from register to gmem
            tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
            tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
            for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
                if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
                    cute.copy(
                        gmem_tiled_copy_dQ,
                        tdQrdQ[None, rest_m, None],
                        tdQgdQ[None, rest_m, None],
                        pred=tdQpdQ[None, rest_m, None],
                    )
