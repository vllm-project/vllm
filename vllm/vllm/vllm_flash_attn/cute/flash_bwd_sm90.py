import math
from collections.abc import Callable
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils
from quack.sm90_utils import gemm_w_idx, gemm_zero_init

from vllm.vllm_flash_attn.cute import pipeline, utils
from vllm.vllm_flash_attn.cute.block_info import BlockInfo
from vllm.vllm_flash_attn.cute.block_sparse_utils import (
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
)
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensors
from vllm.vllm_flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from vllm.vllm_flash_attn.cute.mask import AttentionMask
from vllm.vllm_flash_attn.cute.named_barrier import NamedBarrierBwd, NamedBarrierFwd
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK
from vllm.vllm_flash_attn.cute.softmax import (
    apply_score_mod_bwd_inner,
    apply_score_mod_inner,
)
from vllm.vllm_flash_attn.cute.tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
    ):
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = False
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_mma_warp_groups = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_mma_warp_groups
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        if qhead_per_kvhead > 1:
            assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
            assert self.num_mma_warp_groups == 2, "GQA backward assumes 2 warp groups"
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        # TODO: impl these for hdim 64
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: type[cutlass.Numeric],
        mK_type: type[cutlass.Numeric],
        mV_type: type[cutlass.Numeric],
        mdO_type: type[cutlass.Numeric],
        mLSE_type: type[cutlass.Numeric],
        mdPsum_type: type[cutlass.Numeric],
        mdQaccum_type: type[cutlass.Numeric],
        mdK_type: type[cutlass.Numeric],
        mdV_type: type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError(
                    "mdK and mdV tensors must have the same data type as mQ"
                )
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError(
                    "mdKaccum and mdVaccum tensors must have the data type Float32"
                )
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        (
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
        ) = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), self.Q_stage),
                ((self.tile_n, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdimv), None),
                ((self.tile_m, self.tile_hdimv), self.dO_stage),
                ((self.tile_m, self.tile_n), self.PdS_stage),
            ]
        ]
        self.sdQaccum_layout = cute.make_layout(
            (
                self.tile_m * self.tile_hdim // self.num_mma_warp_groups,
                self.num_mma_warp_groups,
            )
        )
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            # thr_layout
            cute.make_layout(
                (self.num_threads_per_warp_group, self.num_mma_warp_groups)
            ),
            cute.make_layout(128 // Float32.width),  # val_layout
        )
        # dKVaccum for GQA epilogue - reuses sV+sK memory recast as f32
        self.sdKVaccum_layout = cute.make_layout(
            (
                self.tile_n * self.tile_hdim // self.num_mma_warp_groups,
                self.num_mma_warp_groups,
            )
        )
        # dKVaccum R->S (same pattern as dQaccum but sized for tile_n)
        self.r2s_tiled_copy_dKVaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            cute.make_layout(
                (self.num_threads_per_warp_group, self.num_mma_warp_groups)
            ),
            cute.make_layout(128 // Float32.width),
        )

    def _get_tiled_mma(self):
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (
            self.AtomLayoutMSdP,
            self.num_mma_warp_groups // self.AtomLayoutMSdP,
        )
        tiler_mn_SdP = (
            self.tile_m // atom_layout_SdP[0],
            self.tile_n // atom_layout_SdP[1],
        )
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                atom_layout_SdP if not self.SdP_swapAB else atom_layout_SdP[::-1]
            )
            + (1,),
            tiler_mn=tiler_mn_SdP if not self.SdP_swapAB else tiler_mn_SdP[::-1],
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (
            self.AtomLayoutNdKV,
            self.num_mma_warp_groups // self.AtomLayoutNdKV,
        )
        tiler_mn_dK = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdim // atom_layout_dKV[1],
        )
        tiler_mn_dV = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdimv // atom_layout_dKV[1],
        )
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(
                    atom_layout_dKV if not self.dKV_swapAB else atom_layout_dKV[::-1]
                )
                + (1,),
                tiler_mn=tiler_mn_d if not self.dKV_swapAB else tiler_mn_d[::-1],
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        # dQ = dS @ K
        atom_layout_dQ = (
            self.AtomLayoutMdQ,
            self.num_mma_warp_groups // self.AtomLayoutMdQ,
        )
        tiler_mn_dQ = (
            self.tile_m // atom_layout_dQ[0],
            self.tile_hdim // atom_layout_dQ[1],
        )
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1]
            )
            + (1,),
            tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1],
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_alignment = sK_alignment = sV_alighment = sdQaccum_alignment = (
            sdO_alignment
        ) = 1024

        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[
                cute.struct.MemRange[type, cute.cosize(layout)], alignment
            ]
            for (layout, type, alignment) in [
                (self.sQ_layout, self.dtype, sQ_alignment),
                (self.sK_layout, self.dtype, sK_alignment),
                (self.sV_layout, self.dtype, sV_alighment),
                (self.sdO_layout, self.dtype, sdO_alignment),
                (self.sdQaccum_layout, Float32, sdQaccum_alignment),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = (
            cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        )
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.Q_stage
            ],
            128,
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.dO_stage
            ],
            128,
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: cute.Tensor | None = None,
        mdK_semaphore: cute.Tensor | None = None,
        mdV_semaphore: cute.Tensor | None = None,
        aux_tensors: list | None = None,
        blocksparse_tensors: BlockSparseTensors | None = None,
    ):
        assert (
            mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None
        ), "determinism not supported yet for Sm90"

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t)
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdO = [
            layout_utils.select(t, layout_transpose) for t in (mQ, mK, mV, mdO)
        ]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [layout_utils.select(t, layout_transpose) for t in (mdK, mdV)]
        else:
            accum_transpose = [2, 1, 0]  # (b, n, s*h) -> (s*h, n, b)
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]  # (b, n, s) -> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        self.num_mma_regs = 240
        self.num_producer_regs = 24
        # self.num_mma_regs = 232
        # self.num_producer_regs = 40

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m
            * self.tile_hdim
            * Float32.width
            // 8
            // self.num_mma_warp_groups
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = (
            self.tile_n * self.tile_hdimv * Float32.width // 8
        )

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(self.qhead_per_kvhead == 1):
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=False,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.sdKVaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            self.r2s_tiled_copy_dKVaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKVaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=None,
            window_size_right=None,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                )
            if warp_idx == 1:
                for warp_group_idx in cutlass.range(self.num_mma_warp_groups):
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                )
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                r2s_tiled_copy_dKVaccum,
                sdKVaccum_layout,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_tensors,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )
                mK_cur = mK[None, None, head_idx_kv, batch_idx]
                gK = cute.local_tile(
                    mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
                )
                mV_cur = mV[None, None, head_idx_kv, batch_idx]
                gV = cute.local_tile(
                    mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0)
                )

                mQ_cur = mQ[None, None, head_idx, batch_idx]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                mdO_cur = mdO[None, None, head_idx, batch_idx]
                gdO = cute.local_tile(
                    mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0)
                )
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                mdPsum_cur = mdPsum[None, head_idx, batch_idx]
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(
                    seqlen, n_block
                )

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local) or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(
                            producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                        )
                        load_K(
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q
                            )
                        )
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(
                            producer_state_dO_cur,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_dO_cur
                            )
                        )
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(
                            m_block_min + 1, m_block_max, unroll=1
                        ):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:
                        producer_state_Q, producer_state_dO = (
                            produce_block_sparse_q_loads_bwd_sm90(
                                blocksparse_tensors,
                                batch_idx,
                                head_idx,
                                n_block,
                                producer_state_Q,
                                producer_state_dO,
                                pipeline_Q,
                                pipeline_dO,
                                load_K,
                                load_V,
                                load_Q,
                                load_dO,
                                load_LSE,
                                load_dPsum,
                                self.tma_copy_bytes["K"],
                                self.tma_copy_bytes["V"],
                                Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                                subtile_factor=self.subtile_factor,
                                m_block_max=m_block_max,
                            )
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.core.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        # [NOTE] SdP_swapAB: swapAB transposes the tile, so use (n, m) indexing
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m)
            if self.SdP_swapAB
            else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.core.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m)
            if self.SdP_swapAB
            else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            tScS,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: cute.Tensor | None,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        sdKVaccum_layout: cute.Layout,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: BlockSparseTensors | None = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx))
        # S = Q @ K.T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_S[:2],
            tSrQ,
            tSrK,
            swap_AB=self.SdP_swapAB,
        )
        # dP = dO @ V.T
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_dP[:2],
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )
        # dV += P.T @ dO
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx,
                tiled_mma_dV,
                acc_dV,
                tdVrPt,
                tdVrdOt,
                swap_AB=self.dKV_swapAB,
            )
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        # dK += dS.T @ Q
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(
                gemm_w_idx,
                tiled_mma_dK,
                acc_dK,
                tdKrdSt,
                tdKrQt,
                swap_AB=self.dKV_swapAB,
            )
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
            wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
        )
        mma_dsk_fn = partial(
            gemm_zero_init,
            tiled_mma_dQ,
            shape_mnk_dQ[:2],
            tdQrdS,
            tdQrKt,
            swap_AB=self.dQ_swapAB,
        )

        # Smem copy atom tiling
        smem_copy_atom_PdS = copy_utils.get_smem_store_atom(
            self.arch, self.dtype, transpose=self.SdP_swapAB
        )
        smem_thr_copy_PdS = cute.make_tiled_copy_C(
            smem_copy_atom_PdS, tiled_mma_SdP
        ).get_slice(tidx)
        tPsP = None
        if const_expr(sP is not None):
            tPsP = smem_thr_copy_PdS.partition_D(
                sP if const_expr(not self.SdP_swapAB) else sPt
            )
        tdSsdS = smem_thr_copy_PdS.partition_D(
            sdS if const_expr(not self.SdP_swapAB) else sdSt
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )

        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )
        score_mod_fn = partial(
            self.apply_score_mod,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
        score_mod_bwd_fn = partial(
            self.apply_score_mod_bwd,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tPsP=tPsP,
            tdSsdS=tdSsdS,
            tdQsdQaccum=tdQsdQaccum,
            smem_thr_copy_PdS=smem_thr_copy_PdS,
            smem_thr_copy_dQaccum=smem_thr_copy_dQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local) or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_tensors=aux_tensors,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            score_mod_fn=score_mod_fn_cur,
                            score_mod_bwd_fn=score_mod_bwd_fn_cur,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
                else:
                    consumer_state_Q, consumer_state_dO = (
                        consume_block_sparse_mma_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mma_one_m_block_all,
                            mask,
                            self.mask_mod,
                            is_causal=self.is_causal,
                            is_local=self.is_local,
                            thr_mma_SdP=thr_mma_SdP,
                            score_mod_fn=score_mod_fn_cur,
                            score_mod_bwd_fn=score_mod_bwd_fn_cur,
                            subtile_factor=self.subtile_factor,
                            m_block_max=m_block_max,
                            aux_tensors=aux_tensors,
                            fastdiv_mods=fastdiv_mods,
                        )
                    )

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    r2s_tiled_copy_dKVaccum,
                    sdKVaccum_layout,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                )
            else:
                # Block sparsity: KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(self.use_block_sparsity):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        r2s_tiled_copy_dKVaccum,
                        sdKVaccum_layout,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState
        | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tPsP: cute.Tensor | None,
        tdSsdS: cute.Tensor | None,
        tdQsdQaccum: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        mask_fn: Callable | None = None,
        score_mod_fn: Callable | None = None,
        score_mod_bwd_fn: Callable | None = None,
        dKV_accumulate: Boolean = True,
    ):
        consumer_state_dO_cur = (
            consumer_state_dO
            if const_expr(self.Q_stage == self.dO_stage)
            else consumer_state_Q
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = (
            consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        )
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(
            consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
        )
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        if const_expr(self.score_mod_bwd is not None):
            acc_S_pre = cute.make_fragment_like(acc_S)
            cute.autovec_copy(acc_S, acc_S_pre)

        if const_expr(self.score_mod is not None):
            score_mod_fn(acc_S, m_block=m_block)

        # (3) [Pointwise 1] P = exp(S - LSE)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r], fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            tPrP = smem_thr_copy_PdS.retile(tdVrP)
            cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, smem_idx_PdS])

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])

        if const_expr(self.score_mod_bwd is not None):
            score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

        # Convert dS from f32 -> f16
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(
            not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)
        ):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        # R2S for dS
        tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)
        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, smem_idx_PdS])

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS,
                B_idx=smem_idx_dO,
                zero_init=not dKV_accumulate,
                wg_wait=-1,
            )
        else:
            mma_pdo_fn(
                tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        # (6) [GEMM 4] dQ = dS @ K
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dV)
        pipeline_dO.consumer_release(
            consumer_state_dO_cur
        )  # release dO as dV mma is done

        # (7) [GEMM 5] dK += dS.T @ Q
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn(
                A_idx=smem_idx_PdS,
                B_idx=smem_idx_Q,
                zero_init=not dKV_accumulate,
                wg_wait=1,
            )
        else:
            mma_dsq_fn(
                tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
            )
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dQ)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        tdQrdQaccum_flat = cute.make_tensor(
            acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
        )
        cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )

        warpgroup.wait_group(0)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dK)
        pipeline_Q.consumer_release(consumer_state_Q)
        # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("tidx = {}, m_block = {}, after pipeline_Q consumer release", cute.arch.thread_idx()[0], m_block)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        r2s_tiled_copy_dKVaccum: cute.TiledCopy,
        sdKVaccum_layout: cute.Layout,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: FastDivmodDivisor | None = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(self.qhead_per_kvhead == 1):
            rdV = cute.make_fragment_like(acc_dV, self.dtype)
            rdV.store(acc_dV.load().to(self.dtype))
            rdK = utils.cvt_f16(acc_dK, self.dtype)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )

            smem_copy_atom_dKV = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    transpose=self.dKV_swapAB, num_matrices=4
                ),
                self.dtype,
            )
            smem_thr_copy_dK = cute.make_tiled_copy_C(
                smem_copy_atom_dKV, tiled_mma_dK
            ).get_slice(tidx)
            smem_thr_copy_dV = cute.make_tiled_copy_C(
                smem_copy_atom_dKV, tiled_mma_dV
            ).get_slice(tidx)
            mdV_cur = mdV[None, None, head_idx, batch_idx]
            mdK_cur = mdK[None, None, head_idx, batch_idx]
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )

            taccdVrdV = smem_thr_copy_dV.retile(rdV)
            sdV = (
                sV
                if const_expr(not self.dKV_swapAB)
                else layout_utils.transpose_view(sV)
            )
            taccdVsdV = smem_thr_copy_dV.partition_D(sdV)
            cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )
            if warp_idx == 4:
                store_dV()
            taccdKrdK = smem_thr_copy_dK.retile(rdK)
            sdK = (
                sK
                if const_expr(not self.dKV_swapAB)
                else layout_utils.transpose_view(sK)
            )
            taccdKsdK = smem_thr_copy_dK.partition_D(sdK)
            cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            head_idx_kv = head_idx // qhead_per_kvhead_divmod

            mdKaccum_cur = mdK[None, head_idx_kv, batch_idx]
            gdKaccum_ = cute.local_tile(
                mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,)
            )
            gdKaccum = cute.flat_divide(
                gdKaccum_, (self.tile_n * self.tile_hdim // self.num_mma_warp_groups,)
            )

            mdVaccum_cur = mdV[None, head_idx_kv, batch_idx]
            gdVaccum_ = cute.local_tile(
                mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,)
            )
            gdVaccum = cute.flat_divide(
                gdVaccum_, (self.tile_n * self.tile_hdimv // self.num_mma_warp_groups,)
            )

            sdKVaccum = cute.make_tensor(
                cute.recast_ptr(sV.iterator, dtype=Float32),
                sdKVaccum_layout,
            )

            smem_thr_copy_dKVaccum = r2s_tiled_copy_dKVaccum.get_slice(tidx)
            tdKsdKVaccum = smem_thr_copy_dKVaccum.partition_D(sdKVaccum)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )

            tdKrdKaccum_flat = cute.make_tensor(
                acc_dK.iterator, cute.make_layout(tdKsdKVaccum.shape)
            )
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKVaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )

            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKVaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_mma_warp_groups,
                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )

            tdVrdVaccum_flat = cute.make_tensor(
                acc_dV.iterator, cute.make_layout(tdKsdKVaccum.shape)
            )
            cute.autovec_copy(tdVrdVaccum_flat, tdKsdKVaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_mma_threads,
            )

            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_mma_warp_groups,
                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: BlockSparseTensors | None = None,
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(
                mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,)
            )
            # (M * K / WG, WG, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.num_mma_warp_groups,)
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local) or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block = m_block_min + iter_idx
                        m_block_safe = m_block

                        for warp_group_idx in cutlass.range_constexpr(
                            self.num_mma_warp_groups
                        ):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.dQFullWG0)
                                + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, warp_group_idx].iterator,
                                    gdQaccum[
                                        None, warp_group_idx, m_block_safe
                                    ].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            cute.arch.cp_async_bulk_commit_group()
                        for warp_group_idx in cutlass.range_constexpr(
                            self.num_mma_warp_groups
                        ):
                            cute.arch.cp_async_bulk_wait_group(
                                self.num_mma_warp_groups - 1 - warp_group_idx, read=True
                            )
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0)
                                + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )
                else:
                    dQaccum_store_block_sparse_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        sdQaccum,
                        gdQaccum,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                        num_mma_warp_groups=self.num_mma_warp_groups,
                        num_threads_per_warp_group=self.num_threads_per_warp_group,
                        tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                    )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
