"""Low-latency SM100 BF16 gate/up GEMM with a register-only SiTU epilogue.

Computes ``SiTU(x @ weight[:I].T, x @ weight[I:].T)`` for a K-major
``weight[2*I, K]``. Both GEMMs accumulate in FP32. Their matching output values
occupy two disjoint TMEM column ranges and are combined before the final BF16
store, so no ``[T, 2*I]`` intermediate reaches global memory.

Toggled by ``use_2cta`` in the constructor:
  use_2cta=False -> 1x1 cluster, 1-CTA tcgen05.mma, cta_n in [8, 256] step 8
  use_2cta=True  -> 2x1 cluster, 2-CTA tcgen05.mma, cta_n in [16, 256] step 16

Warp specialization (8 warps, 256 threads/CTA; warp 3 idle):
  Warp 0    DMA_A   TMA-loads gate and up A tiles
  Warp 1    DMA_B   TMA-loads B tiles; PDL griddepcontrol.wait
  Warp 2    MMA     two tcgen05.mma streams into TMEM; owns alloc/dealloc
  Warps 4-7 EPILOG  TMEM -> RMEM -> FP32 SiTU/SiLU -> bf16 -> st.global
"""

from __future__ import annotations

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from vllm.cute_utils import _tcgen05
from vllm.utils.torch_utils import direct_register_custom_op

SITU_CTA_K: int = 128
SITU_DEFAULT_TACTIC: int = 2

# (cta_m, cta_n, num_ab_stage, use_2cta)
CUTEDSL_GATED_TACTICS: list[tuple[int, int, int, bool]] = [
    (64, 8, 4, False),
    (64, 8, 5, False),
    (64, 8, 6, False),
    (64, 16, 4, False),
    (64, 16, 5, False),
    (64, 16, 6, False),
    (64, 16, 4, True),
    (64, 16, 5, True),
    (64, 16, 6, True),
    (64, 32, 4, True),
    (64, 32, 5, True),
    (64, 32, 6, True),
    (64, 64, 4, False),
    (64, 128, 3, False),
    (128, 8, 3, False),
    (128, 16, 3, False),
    (128, 32, 3, False),
    (128, 64, 2, False),
    (128, 128, 2, False),
    (64, 64, 5, True),
    (64, 128, 4, True),
    (128, 16, 3, True),
    (128, 32, 3, True),
    (128, 64, 3, True),
    (128, 128, 2, True),
    (64, 32, 5, False),
]


def get_cutedsl_situ_tactic_num() -> int:
    return len(CUTEDSL_GATED_TACTICS)


def get_cutedsl_situ_default_tactic() -> int:
    return SITU_DEFAULT_TACTIC


class WorkTileInfo(NamedTuple):
    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class Bf16GatedGemm:
    def __init__(
        self,
        activation: str = "situ",
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = SITU_CTA_K,
        num_ab_stage: int = 6,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: bool | None = None,
        pdl_count: int = -1,
        out_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        beta: float = 1.0,
        linear_beta: float | None = 1.0,
    ):
        self.activation = activation
        if activation not in ("silu", "situ"):
            raise ValueError(f"Unsupported gated activation: {activation}")
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count
        self.beta = beta
        self.linear_beta = linear_beta

        # 1-CTA: cta_n ∈ [8, 256] step 8 (bf16 tcgen05.mma atom limit).
        # 2-CTA: cta_n ∈ [16, 256] step 16 (bf16 K-major cluster mma).
        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )

        self.threads_per_cta = 256
        if use_2cta:
            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:
            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"Bf16GatedGemm_{self.activation}_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_out{self.out_dtype.__name__.lower()}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        a_gate: cute.Tensor,  # (Gemm_M=I, Gemm_K, Gemm_L), K-major
        a_up: cute.Tensor,  # (Gemm_M=I, Gemm_K, Gemm_L), K-major
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        c: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        stream: cuda.CUstream,
    ):
        # A 2-CTA launch must not leave an M tile without its cluster peer.
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a_gate, a_up, b, c).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA_gate: cute.Tensor,  # (Gemm_M=I, Gemm_K, Gemm_L), K-major
        mA_up: cute.Tensor,  # (Gemm_M=I, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
    ):
        DMA_Stage = self.num_ab_stage

        a_major = utils.LayoutEnum.from_tensor(mA_gate).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA_gate.element_type  # bf16
        c_dtype = mC.element_type  # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # 1 (1-CTA) or 2 (2-CTA)

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — Sw<3,4,3>
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — Sw<3,4,3>

        sA_gate = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sA_up = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        # ((Mma_N_per_cta, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        # ((MMA, MMA_M, MMA_N), gate/up)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mn)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        acc_layout = tCtAcc_fake.layout
        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_empty = bar_empty_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr = tmem_base_arr.iterator  # Pointer[Int32]

        # Per CTA and stage: one combined gate/up arrival and one B arrival.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA_gate, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA_gate, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        gA_gate_tile = cute.local_tile(
            mA_gate,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        gA_up_tile = cute.local_tile(
            mA_up,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(  # (cta_n//num_mma_ctas, cta_k, Tiles_K)
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(  # (cta_m, cta_n)
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_gate_tile,
                gA_up_tile,
                sA_gate,
                sA_up,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA_gate,
                sA_up,
                sB,
                tmem_base_ptr,
                acc_layout,
                num_tmem_cols,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                epi_tid,
                c_dtype,
                d_layout,
            )

        if cutlass.const_expr(self.use_2cta):
            # Keep both CTAs alive until peer-targeted operations have retired.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    # DMA_A warp
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,  # used for the 2-CTA arrive-peer redirect
        gA_gate_tile: cute.Tensor,
        gA_up_tile: cute.Tensor,
        sA_gate: cute.Tensor,
        sA_up: cute.Tensor,
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            # One arrival accounts for both A transactions.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sA_gate.element_type,
                        cute.slice_(sA_gate.layout, (None, None, None, 0)),
                    )
                    + cute.size_in_bytes(
                        sA_up.element_type,
                        cute.slice_(sA_up.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_gate_tile[None, None, k_tile],
                sA_gate[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            cute_ext.tma_load(
                gA_up_tile[None, None, k_tile],
                sA_up[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            if cutlass.const_expr(self.use_pdl) and k_tile == pdl_count:
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # Keep barrier storage alive for the final MMA commits.
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # DMA_B warp
    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,  # (CTA_N, CTA_K, Tiles_K) — this CTA's B strip
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],
                sB[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        # Keep barrier storage alive for the final MMA commits.
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # MMA warp
    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tiled_mma: cute.TiledMma,
        sA_gate: cute.Tensor,
        sA_up: cute.Tensor,
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,  # TMEM accumulator layout
        num_tmem_cols: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,  # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,  # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage

        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc_gate = tAcc[None, None, None, 0]
        acc_up = tAcc[None, None, None, 1]

        if is_leader:
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                for k_block in range(mma_inst_tile_k):
                    mma_atom.set(
                        tcgen05.Field.ACCUMULATE,
                        k_tile != 0 or k_block != 0,
                    )
                    a_gate_frag = cute.append_ones(
                        sA_gate[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    a_up_frag = cute.append_ones(
                        sA_up[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_gate_frag, b_frag, acc_gate)
                    cute_ext.dot(mma_atom, a_up_frag, b_frag, acc_up)

                # tcgen05.commit is single-thread issued.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    # Epilogue warps
    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot from MMA
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,  # (CTA_M, CTA_N) — this CTA's output tile
        epi_tid: cutlass.Int32,  # 0..127 within the 4 EPILOG warps
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
    ):
        # Publish the TMEM pointer: 128 epilogue threads + 32 MMA threads.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc_gate = tCtAcc[((None, None), 0, 0, 0)]
        acc_up = tCtAcc[((None, None), 0, 0, 1)]

        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_gate)

        # (CTA_M, CTA_N, Rest_M=1, Rest_N=1)
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rGate = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rUp = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(  # bf16, per-thread
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        # Required ordering from the mbarrier wait to asynchronous tcgen05.ld.
        _tcgen05.fence_after_thread_sync()

        cute_ext.partition_and_copy(thr_t2r, acc_gate, rGate)
        cute_ext.partition_and_copy(thr_t2r, acc_up, rUp)

        # Complete both tcgen05.ld operations before releasing TMEM.
        cute.arch.fence_view_async_tmem_load()

        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        gate = rGate.load()
        up = rUp.load()
        result = cute.make_rmem_tensor(rGate.shape, self.acc_dtype)
        log2_e = cutlass.Float32(1.4426950408889634)
        beta = cutlass.Float32(self.beta)
        linear_beta = cutlass.Float32(self.linear_beta or 1.0)

        for i in cutlass.range(cute.size(rGate), unroll_full=True):
            g = gate[i]
            sigmoid_g = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp2(-g * log2_e, fastmath=True)
            )
            if cutlass.const_expr(self.activation == "silu"):
                result[i] = g * sigmoid_g * up[i]
            else:
                gate_softcap = beta * cute.math.tanh(g / beta, fastmath=True)
                up_softcap = up[i]
                if cutlass.const_expr(self.linear_beta is not None):
                    up_softcap = linear_beta * cute.math.tanh(
                        up[i] / linear_beta, fastmath=True
                    )
                result[i] = gate_softcap * sigmoid_g * up_softcap

        rD.store(result.load().to(c_dtype))

        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


@cute.experimental.jit
def _bmm_situ(
    gemm_op: cutlass.Constexpr,
    a_gate: cute.Tensor,  # (L, I, K)
    a_up: cute.Tensor,  # (L, I, K)
    b: cute.Tensor,  # (L, K, T)
    c: cute.Tensor,  # (L, I, T)
    stream: cuda.CUstream,
):
    a_gate = cute.make_tensor(
        a_gate.iterator, cute.select(a_gate.layout, mode=[1, 2, 0])
    )
    a_up = cute.make_tensor(a_up.iterator, cute.select(a_up.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a_gate, a_up, b, c, stream)


SITU_COMPILE_CACHE: dict = {}


def _detect_leading_dim(tensor: torch.Tensor) -> int:
    for dim, stride in enumerate(tensor.stride()):
        if stride == 1:
            return dim
    return tensor.dim() - 1


def _make_layout_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, leading_dim: int
) -> torch.Tensor:
    permutation = [i for i in range(len(shape)) if i != leading_dim]
    permutation.append(leading_dim)
    permuted_shape = tuple(shape[i] for i in permutation)
    tensor = torch.empty(permuted_shape, dtype=dtype, device="cuda")
    inverse = [permutation.index(i) for i in range(len(shape))]
    return tensor.permute(inverse)


def _make_compile_repr_tensors(
    dtype: torch.dtype,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    # Representative for the stricter 2-CTA tile contract.
    batch, intermediate, k_size, tokens = 1, 128, 128, 16
    a_gate_t = _make_layout_tensor((batch, intermediate, k_size), dtype, a_leading)
    a_up_t = _make_layout_tensor((batch, intermediate, k_size), dtype, a_leading)
    b_t = _make_layout_tensor((batch, k_size, tokens), dtype, b_leading)
    c_t = _make_layout_tensor((batch, intermediate, tokens), dtype, c_leading)

    a_gate = from_dlpack(a_gate_t, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    a_up = from_dlpack(a_up_t, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    b = from_dlpack(b_t, assumed_align=32).mark_layout_dynamic(leading_dim=b_leading)
    c = from_dlpack(c_t, assumed_align=32).mark_layout_dynamic(leading_dim=c_leading)
    return a_gate, a_up, b, c


def _resolve_gated_tactic(tactic: int) -> tuple[int, int, int, bool]:
    if tactic < 0:
        tactic = SITU_DEFAULT_TACTIC
    if tactic < 0 or tactic >= len(CUTEDSL_GATED_TACTICS):
        raise ValueError(
            f"fused SiTU tactic {tactic} is outside [0, {len(CUTEDSL_GATED_TACTICS)})"
        )
    return CUTEDSL_GATED_TACTICS[tactic]


def _get_compiled_gated_kernel(
    dtype: torch.dtype,
    activation: str,
    tactic: int,
    use_pdl: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
    beta: float,
    linear_beta: float | None,
):
    cta_m, cta_n, num_ab_stage, use_2cta = _resolve_gated_tactic(tactic)
    key = (
        dtype,
        activation,
        tactic,
        bool(use_pdl),
        a_leading,
        b_leading,
        c_leading,
        float(beta),
        None if linear_beta is None else float(linear_beta),
    )
    compiled = SITU_COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled

    gemm = Bf16GatedGemm(
        activation=activation,
        acc_dtype=cutlass.Float32,
        out_dtype=cutlass.BFloat16,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=SITU_CTA_K,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=use_pdl,
        beta=float(beta),
        linear_beta=None if linear_beta is None else float(linear_beta),
    )
    a_gate, a_up, b, c = _make_compile_repr_tensors(
        dtype, a_leading, b_leading, c_leading
    )
    compiled = cute_ext.compile(
        _bmm_situ,
        gemm,
        a_gate,
        a_up,
        b,
        c,
        make_fake_stream(),
    )
    SITU_COMPILE_CACHE[key] = compiled
    return compiled


def _to_cute_situ(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
):
    intermediate_size = weight.shape[0] // 2
    weight_gate = weight[:intermediate_size].unsqueeze(0)
    weight_up = weight[intermediate_size:].unsqueeze(0)
    b = x.unsqueeze(0).transpose(-2, -1)
    c = out.unsqueeze(0).transpose(-2, -1)

    a_leading = _detect_leading_dim(weight_gate)
    b_leading = _detect_leading_dim(b)
    c_leading = _detect_leading_dim(c)

    a_gate = from_dlpack(weight_gate, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    a_up = from_dlpack(weight_up, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    b = from_dlpack(b, assumed_align=32).mark_layout_dynamic(leading_dim=b_leading)
    c = from_dlpack(c, assumed_align=32).mark_layout_dynamic(leading_dim=c_leading)
    return a_gate, a_up, b, c, (a_leading, b_leading, c_leading)


def _cutedsl_bf16_gated_gemm_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str,
    beta: float,
    linear_beta: float | None,
    tactic: int,
) -> torch.Tensor:
    if activation not in ("silu", "situ"):
        raise ValueError(f"Unsupported gated activation: {activation}")
    if torch.cuda.get_device_capability(x.device) != (10, 0):
        raise RuntimeError("fused SiTU BF16 GEMM requires an SM10x GPU")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must both be BF16")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must both be rank-2")
    if x.device.type != "cuda" or x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x and weight must have the same K dimension")
    if weight.shape[0] % 2 != 0:
        raise ValueError("weight must have shape [2*I, K]")
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("x and weight must be K-major")
    if x.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
        raise ValueError("x and weight must be 16-byte aligned")
    if x.shape[1] % SITU_CTA_K != 0:
        raise ValueError(f"K must be divisible by {SITU_CTA_K}")
    if activation == "situ" and beta <= 0.0:
        raise ValueError("beta must be positive")
    if linear_beta is not None and linear_beta <= 0.0:
        raise ValueError("linear_beta must be positive when specified")

    intermediate_size = weight.shape[0] // 2
    if tactic < 0:
        tactic = 2
        if x.shape[0] > 8 and intermediate_size % 128 == 0:
            tactic = 8
            wide_grid_ctas = (intermediate_size // 64) * -(-x.shape[0] // 32)
            sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
            target_ctas = (4 * sm_count + 4) // 5
            if x.shape[0] > 16 and wide_grid_ctas >= target_ctas:
                tactic = 11
    cta_m, _, _, use_2cta = _resolve_gated_tactic(tactic)
    required_i_multiple = cta_m * (2 if use_2cta else 1)
    if intermediate_size % required_i_multiple != 0:
        raise ValueError(
            f"tactic {tactic} requires I divisible by {required_i_multiple}"
        )

    out = torch.empty(
        (x.shape[0], intermediate_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    if x.shape[0] == 0:
        return out

    a_gate, a_up, b, c, leading = _to_cute_situ(x, weight, out)
    compiled = _get_compiled_gated_kernel(
        dtype=x.dtype,
        activation=activation,
        tactic=tactic,
        use_pdl=True,
        a_leading=leading[0],
        b_leading=leading[1],
        c_leading=leading[2],
        beta=float(beta),
        linear_beta=None if linear_beta is None else float(linear_beta),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(a_gate, a_up, b, c, stream)
    return out


def _cutedsl_bf16_gated_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str,
    beta: float,
    linear_beta: float | None,
    tactic: int,
) -> torch.Tensor:
    del activation, beta, linear_beta, tactic
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="cutedsl_bf16_gated_gemm",
    op_func=_cutedsl_bf16_gated_gemm_run,
    mutates_args=[],
    fake_impl=_cutedsl_bf16_gated_gemm_fake,
)


def cutedsl_bf16_gated_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "situ",
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    tactic: int = -1,
) -> torch.Tensor:
    """Compute packed BF16 gate/up GEMMs with a fused SiLU or SiTU epilogue.

    ``weight`` is K-major [2*I, K], with all gate rows followed by all up rows.
    """
    return torch.ops.vllm.cutedsl_bf16_gated_gemm(
        x,
        weight,
        activation,
        float(beta),
        None if linear_beta is None else float(linear_beta),
        int(tactic),
    )
