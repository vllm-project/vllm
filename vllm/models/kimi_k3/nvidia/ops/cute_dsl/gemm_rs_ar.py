# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM100 BF16 GEMM with fused tensor-parallel reduce-scatter/all-reduce."""

# Based on CUTLASS's Blackwell distributed GEMM-RS example at dcf215a.
# See https://github.com/NVIDIA/cutlass/issues/3117 for memory semantics.

from functools import cache

import cutlass
import torch
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Int64, Uint16, cute, utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor, make_ptr, nullptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import get_smem_capacity_in_bytes

from vllm.cute_utils import _tcgen05, mbarrier, simple_tma_copy, to_cta0_smem
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

logger = init_logger(__name__)


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    nvvm.nanosleep(Int32(ns).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def multimem_ld_reduce_16B(x: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    # The vector instruction assumes x is contiguous and 16-byte aligned.
    assert x.element_type == BFloat16
    vec_type = ".v4.bf16x2"

    ptr = x.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    asm = (
        "multimem.ld_reduce.relaxed.gpu.global.add.acc::f32"
        f"{vec_type} {{$0, $1, $2, $3}}, [$4];"
    )
    struct = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [ptr],
        asm,
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], Int32.mlir_type, loc=loc),
        [
            llvm.extractvalue(Int32.mlir_type, struct, [i], loc=loc, ip=ip)
            for i in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    ssa = cute.TensorSSA(vec, 4, Int32)

    y = cute.make_rmem_tensor(4, Int32)
    y.store(ssa)
    return cute.recast_tensor(y, x.element_type)


@dsl_user_op
def multimem_st_16B(dst: cute.Tensor, value: cute.Tensor, *, loc=None, ip=None) -> None:
    # The vector instruction assumes dst and value are contiguous BF16x8 vectors
    # and dst is 16-byte aligned.
    assert dst.element_type == BFloat16
    assert value.element_type == BFloat16
    ptr = dst.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    regs = cute.recast_tensor(value, Int32, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr] + [regs[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class Sm100GemmRsArBF16:
    def __init__(
        self,
        rank: int,
        num_ranks: int,
        BN: int = 128,
        cta_group: int = 1,
        all_reduce: bool = False,
    ) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        BM, BK = 128, 64
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group
        self.all_reduce = all_reduce

        smem_bytes = get_smem_capacity_in_bytes()
        self.stage_size = (BM + (BN // cta_group)) * BK * 2
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_tma(
        self, tensor: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr
    ) -> cpasync.TmaInfo:
        tma_group = (
            tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tma_group)
        swizzle_128b = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (BM, BK, self.num_stages),
            stride=(BK, 1, BM * BK),
        )
        layout = cute.make_composed_layout(swizzle_128b, 0, layout)
        return cpasync.make_tiled_tma_atom(tma_op, tensor, layout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        partial_uc: cute.Tensor,
        partial_mc_ptr: cute.Pointer,
        output: cute.Tensor,
        flags_uc: cute.Tensor,
        flags_mc_ptr: cute.Pointer,
        peer_flag_ptr: cute.Pointer,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        N = B.shape[0]
        BM, BN, BK = self.cta_tile
        A_tma = self.prepare_tma(A, BM, BK)
        B_tma = self.prepare_tma(B, BN // self.cta_group, BK)
        padded_M = partial_uc.shape[0]
        partial_mc = cute.make_tensor(
            partial_mc_ptr,
            cute.make_layout((padded_M, N), stride=(N, 1)),
        )
        peer_flags = cute.make_tensor(
            peer_flag_ptr,
            cute.make_layout(self.num_ranks),
        )

        grid = (grid_size, 1, 1)
        block = (10 * 32, 1, 1)
        cluster = (self.cta_group, 1, 1)
        self.kernel(
            A_tma,
            B_tma,
            partial_uc,
            partial_mc,
            output,
            flags_uc.iterator,
            flags_mc_ptr,
            peer_flags,
        ).launch(grid=grid, block=block, cluster=cluster, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        partial_uc: cute.Tensor,
        partial_mc: cute.Tensor,
        output: cute.Tensor,
        flags_uc_ptr: cute.Pointer,
        flags_mc_ptr: cute.Pointer,
        peer_flags: cute.Tensor,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM, BN, BK = self.cta_tile
        cta_group = self.cta_group
        num_stages = self.num_stages
        num_ranks = self.num_ranks

        is_2cta = cta_group == 2
        cta_rank = raw_bid % self.cta_group
        num_tmem_stages = 512 // BN

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            BFloat16,
            A_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=A_tma.smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            BFloat16,
            B_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=B_tma.smem_layout.inner,
        )
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, num_tmem_stages)
        tmem_empty_mbar = smem.allocate_array(Int64, num_tmem_stages)
        taddr = smem.allocate(Int32, 4)

        # Named barriers
        BAR_TMEM_ALLOC = 1
        BAR_EPILOGUE = 2
        BAR_COMM = 3

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape
        local_M = (
            partial_uc.shape[0] // num_ranks if self.all_reduce else output.shape[0]
        )
        grid_m = cute.ceil_div(M, BM)
        # Keep 2-CTA clusters within a single N tile.
        grid_m = cute.ceil_div(grid_m, cta_group) * cta_group
        grid_n = cute.ceil_div(N, BN)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(num_tmem_stages):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)

        if cutlass.const_expr(is_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        total_tiles = grid_m * grid_n

        if warp_id == 9:
            # TMA warp
            tma_stage = 0
            parity = 1

            if cutlass.const_expr(is_2cta):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
            else:
                tma_full_mbar_ = tma_full_mbar

            # Select global-memory tiles.
            # [(BM, BK), (M/BM, K/BK)]
            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // cta_group, BK))

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m
                if cutlass.const_expr(cta_group == 2):
                    bid_n = bid_n * cta_group + cta_rank

                for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    simple_tma_copy(
                        A_tma.atom,
                        gA_tiles[None, (bid_m, iter_k)],
                        sA[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        B_tma.atom,
                        gB_tiles[None, (bid_n, iter_k)],
                        sB[None, None, tma_stage],
                        mbar,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

        elif warp_id == 8:
            # MMA warp
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1

                MMA_M = BM * cta_group
                MMA_N = BN
                idesc = _tcgen05.make_bf16_idesc(MMA_M, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                multicast_mask = Uint16((1 << self.cta_group) - 1)

                for bid in range(raw_bid, total_tiles, num_bids):
                    cute.arch.mbarrier_wait(
                        tmem_empty_mbar + tmem_stage, tmem_empty_parity
                    )
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                        d_tmem = BN * tmem_stage
                        a_addr = sA[None, None, tma_stage].iterator.toint()
                        b_addr = sB[None, None, tma_stage].iterator.toint()
                        a_desc = sdesc | (a_addr >> 4)
                        b_desc = sdesc | (b_addr >> 4)

                        cute.arch.mbarrier_wait(
                            tma_full_mbar + tma_stage, tma_full_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        for mma_k in cutlass.range_constexpr(BK // 16):
                            enable_d = iter_k > 0 or mma_k > 0
                            _tcgen05.mma_f16(
                                d_tmem, a_desc, b_desc, idesc, enable_d, cta_group
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                        _tcgen05.commit(
                            tma_empty_mbar + tma_stage, multicast_mask, cta_group
                        )

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(
                        tmem_full_mbar + tmem_stage, multicast_mask, cta_group
                    )

                    tmem_stage = (tmem_stage + 1) % num_tmem_stages
                    if tmem_stage == 0:
                        tmem_empty_parity ^= 1

        elif warp_id >= 4:
            # Communication warps
            # Keep epilogue in warps 0-3 and communication in warps 4-7.
            # Swapping the warpgroups can hang due to warp scheduling.
            tid_ = tid % 128

            # Offset in the [M, N] GEMM result.
            rank_start = self.rank * local_M
            rank_end = min(rank_start + local_M, M)

            st_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            )

            # Each thread issues one BF16x8 multimem reduction per vector.
            vec_width = 8
            vec_cols = BN // vec_width
            max_tile_rows = BM // num_ranks
            vecs_per_tile = max_tile_rows * vec_cols
            partial_vecs = cute.zipped_divide(partial_mc, (1, vec_width))
            if cutlass.const_expr(not self.all_reduce):
                output_vecs = cute.zipped_divide(output, (1, vec_width))

            for tile_id in range(raw_bid, total_tiles, num_bids):
                rs_bid_m = tile_id % grid_m
                bid_n = tile_id // grid_m

                local_row_start = rs_bid_m * local_M // grid_m
                local_row_end = (rs_bid_m + 1) * local_M // grid_m
                global_row_start = rank_start + local_row_start
                global_row_end = min(rank_start + local_row_end, M)

                if global_row_start < M:
                    # Map the current RS tile to the required GEMM tiles.
                    # Since an RS tile is smaller than a GEMM tile, each RS tile
                    # can overlap at most 2 GEMM tiles.
                    gemm_bid_m0 = global_row_start // BM
                    gemm_bid_m1 = max(global_row_end - 1, global_row_start) // BM

                    # Poll local L2 with relaxed GPU-scope loads. The following
                    # multimem reduction reads every rank's L2 directly, so no
                    # acquire fence or L1 invalidation is needed.
                    if tid_ == 0:
                        # Poll the 1st GEMM tile.
                        flag_ptr = flags_uc_ptr + bid_n * grid_m + gemm_bid_m0
                        arrivals = cute.arch.load(
                            flag_ptr, Int32, sem="relaxed", scope="gpu"
                        )
                        while arrivals < num_ranks:
                            nanosleep(64)
                            arrivals = cute.arch.load(
                                flag_ptr, Int32, sem="relaxed", scope="gpu"
                            )
                    elif tid_ == 32 and gemm_bid_m1 != gemm_bid_m0:
                        # Poll the 2nd GEMM tile with another warp if needed.
                        flag_ptr = flags_uc_ptr + bid_n * grid_m + gemm_bid_m1
                        arrivals = cute.arch.load(
                            flag_ptr, Int32, sem="relaxed", scope="gpu"
                        )
                        while arrivals < num_ranks:
                            nanosleep(64)
                            arrivals = cute.arch.load(
                                flag_ptr, Int32, sem="relaxed", scope="gpu"
                            )
                    cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                    # Issue multimem.ld_reduce before storing any result.
                    reduced_vecs = []
                    for vec_iter in cutlass.range_constexpr(vecs_per_tile // 128):
                        vec_idx = tid_ + vec_iter * 128

                        local_row = local_row_start + vec_idx // vec_cols
                        global_row = rank_start + local_row
                        col = bid_n * vec_cols + vec_idx % vec_cols

                        reduced_vec = cute.make_rmem_tensor(vec_width, BFloat16)
                        if local_row < local_row_end and global_row < M:
                            tmp = multimem_ld_reduce_16B(
                                partial_vecs[None, (global_row, col)]
                            )
                            reduced_vec.store(tmp.load())
                        reduced_vecs.append(reduced_vec)

                    for vec_iter in cutlass.range_constexpr(vecs_per_tile // 128):
                        vec_idx = tid_ + vec_iter * 128

                        local_row = local_row_start + vec_idx // vec_cols
                        global_row = rank_start + local_row
                        col = bid_n * vec_cols + vec_idx % vec_cols

                        if local_row < local_row_end and global_row < M:
                            if cutlass.const_expr(self.all_reduce):
                                # Broadcast the result to all ranks.
                                multimem_st_16B(
                                    partial_vecs[None, (global_row, col)],
                                    reduced_vecs[vec_iter],
                                )
                            else:
                                # Store the result to local L2.
                                cute.copy(
                                    st_atom,
                                    reduced_vecs[vec_iter],
                                    output_vecs[None, (local_row, col)],
                                )
                    cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                    # Release each GEMM tile consumed by this RS tile. The last
                    # consumer resets its producer flag for the next launch.
                    def release_gemm_tile(
                        gemm_bid_m,
                        tile_M,
                        logical_M,
                        rank_start,
                        rank_end,
                        local_M,
                        grid_m,
                        tile_N,
                        flags,
                        num_ranks,
                    ):
                        gemm_start = gemm_bid_m * tile_M
                        gemm_end = min(gemm_start + tile_M, logical_M)
                        local_start = max(gemm_start, rank_start) - rank_start
                        local_end = min(gemm_end, rank_end) - rank_start

                        # Compute the number of consumers to identify the last
                        # arrival, which resets the producer flag.
                        first_rs_tile = (
                            cute.ceil_div((local_start + 1) * grid_m, local_M) - 1
                        )
                        last_rs_tile = cute.ceil_div(local_end * grid_m, local_M) - 1
                        num_consumers = last_rs_tile - first_rs_tile + 1

                        flag_ptr = flags + tile_N * grid_m + gemm_bid_m
                        old_count = cute.arch.atomic_add(
                            flag_ptr, Int32(1), sem="relaxed", scope="gpu"
                        )
                        if old_count == num_ranks + num_consumers - 1:
                            cute.arch.store(
                                flag_ptr, Int32(0), sem="relaxed", scope="gpu"
                            )

                    if tid_ == 0:
                        # Arrive at the 1st GEMM tile.
                        release_gemm_tile(
                            gemm_bid_m0,
                            BM,
                            M,
                            rank_start,
                            rank_end,
                            local_M,
                            grid_m,
                            bid_n,
                            flags_uc_ptr,
                            num_ranks,
                        )
                    elif tid_ == 32 and gemm_bid_m1 != gemm_bid_m0:
                        # Arrive at the 2nd GEMM tile if necessary.
                        release_gemm_tile(
                            gemm_bid_m1,
                            BM,
                            M,
                            rank_start,
                            rank_end,
                            local_M,
                            grid_m,
                            bid_n,
                            flags_uc_ptr,
                            num_ranks,
                        )

            # Exit barrier. GPU scope is sufficient for RS because the kernel
            # writes local global memory and only needs to flush it to local L2.
            # AR uses system scope to flush the multimem stores to remote L2.
            cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)
            if tid_ == 0:
                exit_flag = total_tiles + raw_bid
                scope = "sys" if self.all_reduce else "gpu"
                utils.distributed.multimem_red_add1(
                    flags_mc_ptr + exit_flag, order="release", scope=scope
                )
                utils.distributed.spin_lock_atom_cas_acquire_wait(
                    flags_uc_ptr + exit_flag,
                    expected_val=num_ranks,
                    reset_val=0,
                    scope=scope,
                )

        else:
            # Epilogue warps
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            peer_flag_bases = cute.make_rmem_tensor(num_ranks, Int64)
            if tid_ == 0:
                for rank in cutlass.range_constexpr(num_ranks):
                    peer_flag_bases[rank] = cute.arch.load(
                        (peer_flags.iterator + rank).llvm_ptr,
                        Int64,
                    )

            if warp_id_ == 0:
                _tcgen05.alloc(taddr, cta_group)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            WIDTH = cutlass.const_expr(16)
            partial_vecs = cute.zipped_divide(partial_uc, (1, WIDTH))

            bf16x16_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                BFloat16,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            tmem_stage = 0
            parity = 0

            if cutlass.const_expr(is_2cta):
                tmem_empty_mbar_ = to_cta0_smem(tmem_empty_mbar)
            else:
                tmem_empty_mbar_ = tmem_empty_mbar

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m = bid % grid_m
                bid_n = bid // grid_m

                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(BN // WIDTH):
                    tcol = tmem_stage * BN + i * WIDTH
                    regs = _tcgen05.ld(warp_id_ * 32, tcol, "32x32b", WIDTH)
                    _tcgen05.wait_ld()

                    if cutlass.const_expr(i == BN // WIDTH - 1):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, "cluster")

                    tmp = cute.make_rmem_tensor(WIDTH, BFloat16)
                    tmp.store(regs.to(BFloat16))

                    global_row = bid_m * BM + tid_
                    if global_row < M:
                        coord = (global_row, bid_n * (BN // WIDTH) + i)
                        cute.copy(bf16x16_atom, tmp, partial_vecs[None, coord])

                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)

                # Signal GEMM completion. GPU scope is sufficient because the
                # partial output only needs to be flushed to local L2.
                if tid_ == 0 and bid_m * BM < M:
                    gemm_start = bid_m * BM
                    gemm_end = min(gemm_start + BM, M)
                    first_owner = gemm_start // local_M
                    last_owner = (gemm_end - 1) // local_M

                    # signal to all consuming ranks (can be more than 1)
                    for rank in cutlass.range_constexpr(num_ranks):
                        if first_owner <= rank and rank <= last_owner:
                            ptr = cute.make_ptr(
                                Int32,
                                peer_flag_bases[rank],
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            ptr += bid_m + bid_n * grid_m
                            utils.distributed.red_add1(
                                ptr, order="release", scope="gpu"
                            )

                tmem_stage = (tmem_stage + 1) % num_tmem_stages
                if tmem_stage == 0:
                    parity ^= 1

            if cutlass.const_expr(is_2cta):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
            if warp_id_ == 0:
                _tcgen05.dealloc(cta_group)

    @cache
    @staticmethod
    def compile(
        rank: int,
        num_ranks: int,
        BN: int,
        cta_group: int,
        all_reduce: bool = False,
    ):
        M = cute.sym_int()
        padded_M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        local_M = cute.sym_int()
        num_flags = cute.sym_int()

        A = make_fake_tensor(
            BFloat16, (M, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        B = make_fake_tensor(
            BFloat16, (N, K), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        partial = make_fake_tensor(
            BFloat16,
            (padded_M, N),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=32,
        )
        partial_mc_ptr = nullptr(BFloat16, cute.AddressSpace.gmem, assumed_align=32)
        output = (
            None
            if all_reduce
            else make_fake_tensor(
                BFloat16,
                (local_M, N),
                (cute.sym_int64(divisibility=16), 1),
                assumed_align=32,
            )
        )
        flags = make_fake_tensor(Int32, (num_flags,), (1,), assumed_align=16)
        flags_mc_ptr = nullptr(Int32, cute.AddressSpace.gmem, assumed_align=16)
        peer_flag_ptr = nullptr(Int64, cute.AddressSpace.gmem, assumed_align=8)

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100GemmRsArBF16(
            rank, num_ranks, BN, cta_group, all_reduce=all_reduce
        )
        return cute.compile(
            kernel,
            A,
            B,
            partial,
            partial_mc_ptr,
            output,
            flags,
            flags_mc_ptr,
            peer_flag_ptr,
            128,
            stream,
            options="--enable-tvm-ffi",
        )


class GemmRsAr:
    """Own the symmetric workspace for Kimi-K3 GEMM-RS/AR launches.

    All TP ranks must belong to one NVLink domain for multimem instructions.

    Each instance is bound to either RS or AR. A vLLM worker has one static
    sequence-parallel topology, so the process-wide singleton only needs one
    mode. Two independent mode-specific singletons would lift that restriction
    but duplicate the large symmetric workspace. A future mixed-mode design
    should instead use lightweight RS/AR frontends over one shared multicast
    workspace; that is outside this integration's current scope.
    """

    def __init__(self, *, max_M: int, N: int, all_reduce: bool = False) -> None:
        tp_group = get_tp_group()
        group = tp_group.device_group
        rank = tp_group.rank_in_group
        world_size = tp_group.world_size
        device = torch.device("cuda", torch.accelerator.current_device_index())

        assert 1 < world_size <= 16
        assert 128 % world_size == 0
        assert max_M >= 128 and N % 128 == 0

        max_M = (max_M + world_size - 1) // world_size * world_size
        self.rank = rank
        self.world_size = world_size
        self.max_M = max_M
        self.N = N
        self.device = device
        self.all_reduce = all_reduce

        self.partial = symm_mem.empty((max_M, N), dtype=torch.bfloat16, device=device)
        self.partial_handle = symm_mem.rendezvous(self.partial, group)
        if self.partial_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-RS/AR requires NVLink multicast memory")
        self.partial_mc_ptr = make_ptr(
            BFloat16,
            self.partial_handle.multicast_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        )

        grid_m = (max_M + 127) // 128
        cta_group = 2 if max_M >= 1024 or grid_m % 2 == 0 else 1
        grid_m = (grid_m + cta_group - 1) // cta_group * cta_group
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_flags = grid_m * (N // 128) + self.num_sms
        self.flags = symm_mem.empty(max_flags, dtype=torch.int32, device=device)
        self.flags_handle = symm_mem.rendezvous(self.flags, group)
        if self.flags_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-RS/AR requires NVLink multicast memory")
        self.flags.zero_()
        self.flags_mc_ptr = make_ptr(
            Int32,
            self.flags_handle.multicast_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        self.peer_flag_ptr = make_ptr(
            Int64,
            self.flags_handle.buffer_ptrs_dev,
            cute.AddressSpace.gmem,
            assumed_align=8,
        )

        torch.accelerator.synchronize(device)
        tp_group.barrier()

    def can_run(self, linear: LinearBase) -> bool:
        # Validate projection-invariant requirements once during model init.
        # only supports BF16 for now
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            return False
        w = linear.weight
        if w.ndim != 2:
            return False
        K = w.shape[1]
        return (
            w.shape == (self.N, K)
            and K % 64 == 0
            and w.dtype == torch.bfloat16
            and w.device == self.device
            and w.is_contiguous()
        )

    def warn_incompatible_projection(self) -> None:
        logger.warning_once(
            "Some projections are incompatible with GEMM-RS/AR; using the "
            "unfused path instead.",
            scope="global",
        )

    def should_run(self, x: torch.Tensor) -> bool:
        # Use the same threshold for RS and AR for now. Small-M shapes are
        # supported but faster on the existing LL path.
        return x.shape[0] >= 128

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        M, K = x.shape
        assert 0 < M <= self.max_M
        assert w.shape == (self.N, K) and K % 64 == 0
        assert w.dtype == torch.bfloat16
        assert w.device == self.device
        assert w.is_contiguous()
        assert x.dtype == torch.bfloat16
        assert x.device == self.device
        assert x.is_contiguous()
        N = w.shape[0]
        padded_M = (M + self.world_size - 1) // self.world_size
        padded_M *= self.world_size
        local_M = padded_M // self.world_size

        grid_m = (M + 127) // 128
        # Avoid padding small odd grids; 2-CTA wins consistently for M >= 1024.
        cta_group = 2 if M >= 1024 or grid_m % 2 == 0 else 1
        grid_m = (grid_m + cta_group - 1) // cta_group * cta_group
        BN = 256 if M * K >= 24 * 1024 * 1024 else 128
        assert N % BN == 0

        num_tiles = grid_m * (N // BN)
        num_ctas = min(num_tiles, self.num_sms)
        num_ctas = num_ctas // cta_group * cta_group
        assert self.flags.numel() >= num_tiles + num_ctas

        output = None
        if not self.all_reduce:
            output = torch.empty((local_M, N), dtype=torch.bfloat16, device=self.device)
        compiled = Sm100GemmRsArBF16.compile(
            self.rank,
            self.world_size,
            BN,
            cta_group,
            self.all_reduce,
        )
        compiled(
            x,
            w,
            self.partial[:padded_M],
            self.partial_mc_ptr,
            output,
            self.flags,
            self.flags_mc_ptr,
            self.peer_flag_ptr,
            num_ctas,
        )
        if self.all_reduce:
            # AttnRes may retain output past the next workspace reuse.
            # A future kernel could overlap this copy using an extra warp or
            # the communication warp.
            return self.partial[:M].clone()
        assert output is not None
        return output


_gemm_rs_ar: GemmRsAr | None = None


def init_gemm_rs_ar(max_M: int, N: int, *, all_reduce: bool = False) -> None:
    """Collectively initialize the process-wide, mode-bound GEMM-RS/AR state."""
    global _gemm_rs_ar
    if _gemm_rs_ar is not None:
        if _gemm_rs_ar.all_reduce != all_reduce:
            current = "AR" if _gemm_rs_ar.all_reduce else "RS"
            requested = "AR" if all_reduce else "RS"
            raise RuntimeError(
                f"GEMM-RS/AR is already initialized for {current}; "
                f"a worker cannot reinitialize it for {requested}"
            )
        assert _gemm_rs_ar.max_M >= max_M and _gemm_rs_ar.N == N
        return
    _gemm_rs_ar = GemmRsAr(max_M=max_M, N=N, all_reduce=all_reduce)


def warmup_gemm_rs_ar() -> int:
    """Compile every reachable dispatch for the initialized GEMM-RS/AR mode."""
    # Initialization can be disabled or fail when multicast is unavailable.
    if _gemm_rs_ar is None:
        return 0
    # Keep these profiles in sync with the dispatch in GemmRsAr.__call__.
    profiles = ((128, 1), (128, 2), (256, 2))
    for BN, cta_group in profiles:
        Sm100GemmRsArBF16.compile(
            _gemm_rs_ar.rank,
            _gemm_rs_ar.world_size,
            BN,
            cta_group,
            _gemm_rs_ar.all_reduce,
        )
    return len(profiles)


def get_gemm_rs_ar() -> GemmRsAr:
    assert _gemm_rs_ar is not None, "GEMM-RS/AR is not initialized"
    return _gemm_rs_ar
