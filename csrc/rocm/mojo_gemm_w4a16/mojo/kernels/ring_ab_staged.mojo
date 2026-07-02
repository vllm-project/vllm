from std.gpu import barrier, block_idx, lane_id, thread_idx, warp_id
from std.gpu.memory import AddressSpace
from std.memory import bitcast, stack_allocation
from std.utils import IndexList

from linalg.structuring import SMemArray
from layout import TensorLayout, TileTensor

from mojo.common import (
    ASSUME_EVEN_K,
    ASSUME_EVEN_MN,
    BK,
    BM,
    BN,
    CD,
    COMPUTE_WARPS,
    GROUP_SIZE,
    LOAD_B_BY_QPACK,
    MMA_K,
    MMA_M,
    MMA_N,
    NUM_STAGES,
    PRODUCTION_TOTAL_THREADS,
    PRODUCTION_TOTAL_WARPS,
    RING_PRODUCER_WARPS,
    RING_STARTUP_ALL_WARPS,
    SCALE_AFTER_GROUP,
    SMEM_PAD,
    USE_FP16,
    USE_QZEROS,
    WARPS_M,
    WARPS_N,
    ZERO_OFFSET,
    ZP_BIAS,
    dtype_acc,
    dtype_in,
    dtype_out,
    dtype_q,
)
from mojo.kernel_common import (
    W4_KPACK_COLS,
    W4_PACK,
    accumulate_scaled_group,
    block_swizzle,
    compute_stage,
    compute_stage_direct_a,
    fdot2_bf16,
    load_a_smem_tile,
    load_b_dequant_smem_tile,
    load_b_dequant_smem_tile_qpack,
    store_accum,
    store_accum_partial,
    wmma,
)
from mojo.ring_buffer import (
    increment_counter_if_first_lane,
    stage_ptr,
    wait_for_counter,
    wait_for_ring_stage_stores,
)


def gemm_w4a16_ring_ab_staged_kernel[
    al: TensorLayout,
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    cl: TensorLayout,
    block_m: Int = BM,
    block_n: Int = BN,
    block_k: Int = BK,
    num_stages: Int = NUM_STAGES,
](
    c: TileTensor[mut=True, dtype_out, cl, MutAnyOrigin],
    a: TileTensor[dtype_in, al, ImmutAnyOrigin],
    qweight: TileTensor[dtype_q, ql, ImmutAnyOrigin],
    qzeros: TileTensor[dtype_q, zl, ImmutAnyOrigin],
    scales: TileTensor[dtype_in, sl, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert qzeros.flat_rank == 2
    comptime assert block_m % (WARPS_M * MMA_M) == 0
    comptime assert block_n % (WARPS_N * MMA_N) == 0
    comptime assert block_k % MMA_K == 0
    comptime assert block_k <= GROUP_SIZE
    comptime assert GROUP_SIZE % block_k == 0
    comptime assert num_stages > 0
    comptime SS = block_k + SMEM_PAD
    comptime A_STAGE_STRIDE = block_m * SS
    comptime B_STAGE_STRIDE = block_n * SS
    comptime warp_tile_m = block_m // WARPS_M // MMA_M
    comptime warp_tile_n = block_n // WARPS_N // MMA_N
    comptime num_c_tiles = warp_tile_m * warp_tile_n

    var grid_dim = IndexList[2](
        (n + block_n - 1) // block_n, (m + block_m - 1) // block_m
    )
    var block_xy = block_swizzle(
        IndexList[2](elems=(Int(block_idx.x), Int(block_idx.y))), grid_dim
    )
    var block_n_offset = Int(block_xy[0]) * block_n
    var block_m_offset = Int(block_xy[1]) * block_m
    var wid = Int(warp_id())
    var lid = Int(lane_id())
    var compute_wid = wid - RING_PRODUCER_WARPS

    var a_smem = stack_allocation[
        A_STAGE_STRIDE * num_stages,
        dtype_in,
        address_space=AddressSpace.SHARED,
    ]()
    var b_smem = stack_allocation[
        B_STAGE_STRIDE * num_stages,
        dtype_in,
        address_space=AddressSpace.SHARED,
    ]()
    var counters = SMemArray[Int32, num_stages].stack_allocation[alignment=32]()
    var producer_arrivals = SMemArray[Int32, num_stages].stack_allocation[
        alignment=32
    ]()

    if Int(thread_idx.x) == 0:
        comptime for stage in range(num_stages):
            counters[stage][] = 0
            producer_arrivals[stage][] = 0
    barrier()

    var num_k_tiles = (k + block_k - 1) // block_k
    if num_k_tiles == 0:
        return

    comptime if RING_STARTUP_ALL_WARPS:
        load_a_smem_tile[block_m, block_k, PRODUCTION_TOTAL_WARPS, al](
            stage_ptr[A_STAGE_STRIDE](a_smem, 0),
            a,
            block_m_offset,
            0,
            m,
            k,
            Int(thread_idx.x),
        )
        comptime if LOAD_B_BY_QPACK:
            load_b_dequant_smem_tile_qpack[
                block_n,
                block_k,
                PRODUCTION_TOTAL_WARPS,
                ql,
                zl,
                sl,
                ImmutAnyOrigin,
                ImmutAnyOrigin,
                ImmutAnyOrigin,
            ](
                stage_ptr[B_STAGE_STRIDE](b_smem, 0),
                qweight,
                qzeros,
                scales,
                block_n_offset,
                0,
                n,
                k,
                Int(thread_idx.x),
            )
        else:
            load_b_dequant_smem_tile[
                block_n,
                block_k,
                PRODUCTION_TOTAL_WARPS,
                ql,
                zl,
                sl,
                ImmutAnyOrigin,
                ImmutAnyOrigin,
                ImmutAnyOrigin,
            ](
                stage_ptr[B_STAGE_STRIDE](b_smem, 0),
                qweight,
                qzeros,
                scales,
                block_n_offset,
                0,
                n,
                k,
                Int(thread_idx.x),
            )
        wait_for_ring_stage_stores()
        barrier()
        if Int(thread_idx.x) == 0:
            counters[0][] = 1
        barrier()

    if wid < RING_PRODUCER_WARPS:
        var consume_phases = InlineArray[Int32, num_stages](fill=0)
        var publish_phases = InlineArray[Int32, num_stages](fill=0)
        consume_phases[0] = Int32(1 + COMPUTE_WARPS)

        comptime if not RING_STARTUP_ALL_WARPS:
            var a_stage = stage_ptr[A_STAGE_STRIDE](a_smem, 0)
            var b_stage = stage_ptr[B_STAGE_STRIDE](b_smem, 0)
            load_a_smem_tile[block_m, block_k, RING_PRODUCER_WARPS, al](
                a_stage, a, block_m_offset, 0, m, k, Int(thread_idx.x)
            )
            comptime if LOAD_B_BY_QPACK:
                load_b_dequant_smem_tile_qpack[
                    block_n,
                    block_k,
                    RING_PRODUCER_WARPS,
                    ql,
                    zl,
                    sl,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                ](
                    b_stage,
                    qweight,
                    qzeros,
                    scales,
                    block_n_offset,
                    0,
                    n,
                    k,
                    Int(thread_idx.x),
                )
            else:
                load_b_dequant_smem_tile[
                    block_n,
                    block_k,
                    RING_PRODUCER_WARPS,
                    ql,
                    zl,
                    sl,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                ](
                    b_stage,
                    qweight,
                    qzeros,
                    scales,
                    block_n_offset,
                    0,
                    n,
                    k,
                    Int(thread_idx.x),
                )
            wait_for_ring_stage_stores()
            increment_counter_if_first_lane(producer_arrivals[0])
            if wid == 0:
                wait_for_counter(
                    producer_arrivals[0], Int32(RING_PRODUCER_WARPS)
                )
                publish_phases[0] = Int32(RING_PRODUCER_WARPS)
                increment_counter_if_first_lane(counters[0])

        for local_tile in range(1, num_k_tiles):
            var stage = local_tile % num_stages
            wait_for_counter(counters[stage], consume_phases[stage])
            consume_phases[stage] += Int32(1 + COMPUTE_WARPS)

            var k_offset = local_tile * block_k
            var a_stage = stage_ptr[A_STAGE_STRIDE](a_smem, stage)
            var b_stage = stage_ptr[B_STAGE_STRIDE](b_smem, stage)
            load_a_smem_tile[block_m, block_k, RING_PRODUCER_WARPS, al](
                a_stage,
                a,
                block_m_offset,
                k_offset,
                m,
                k,
                Int(thread_idx.x),
            )
            comptime if LOAD_B_BY_QPACK:
                load_b_dequant_smem_tile_qpack[
                    block_n,
                    block_k,
                    RING_PRODUCER_WARPS,
                    ql,
                    zl,
                    sl,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                ](
                    b_stage,
                    qweight,
                    qzeros,
                    scales,
                    block_n_offset,
                    k_offset,
                    n,
                    k,
                    Int(thread_idx.x),
                )
            else:
                load_b_dequant_smem_tile[
                    block_n,
                    block_k,
                    RING_PRODUCER_WARPS,
                    ql,
                    zl,
                    sl,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                    ImmutAnyOrigin,
                ](
                    b_stage,
                    qweight,
                    qzeros,
                    scales,
                    block_n_offset,
                    k_offset,
                    n,
                    k,
                    Int(thread_idx.x),
                )
            wait_for_ring_stage_stores()
            increment_counter_if_first_lane(producer_arrivals[stage])
            if wid == 0:
                wait_for_counter(
                    producer_arrivals[stage],
                    publish_phases[stage] + Int32(RING_PRODUCER_WARPS),
                )
                publish_phases[stage] += Int32(RING_PRODUCER_WARPS)
                increment_counter_if_first_lane(counters[stage])
        return

    if compute_wid < 0 or compute_wid >= COMPUTE_WARPS:
        return

    var warp_m_idx, warp_n_idx = divmod(compute_wid, WARPS_N)
    var c_acc = InlineArray[SIMD[dtype_acc, CD], num_c_tiles](
        fill=SIMD[dtype_acc, CD](0)
    )
    var phases = InlineArray[Int32, num_stages](fill=1)
    var lane_row_offset, lane_col = divmod(lid, 16)

    comptime if SCALE_AFTER_GROUP:
        var group_acc = InlineArray[SIMD[dtype_acc, CD], num_c_tiles](
            fill=SIMD[dtype_acc, CD](0)
        )
        for local_tile in range(num_k_tiles):
            var stage = local_tile % num_stages
            wait_for_counter(counters[stage], phases[stage])
            phases[stage] += Int32(1 + COMPUTE_WARPS)

            var a_stage = stage_ptr[A_STAGE_STRIDE](a_smem, stage)
            var b_stage = stage_ptr[B_STAGE_STRIDE](b_smem, stage)
            compute_stage[block_k, SS, warp_tile_m, warp_tile_n, num_c_tiles](
                group_acc, a_stage, b_stage, warp_m_idx, warp_n_idx, lid
            )
            increment_counter_if_first_lane(counters[stage])

            if (
                (local_tile + 1) * block_k
            ) % GROUP_SIZE == 0 or local_tile + 1 == num_k_tiles:
                var group_id = (local_tile * block_k) // GROUP_SIZE
                accumulate_scaled_group[
                    sl, warp_tile_m, warp_tile_n, num_c_tiles, ImmutAnyOrigin
                ](
                    c_acc,
                    group_acc,
                    scales,
                    block_n_offset,
                    group_id,
                    warp_n_idx,
                    lane_col,
                    n,
                )
    else:
        for local_tile in range(num_k_tiles):
            var stage = local_tile % num_stages
            wait_for_counter(counters[stage], phases[stage])
            phases[stage] += Int32(1 + COMPUTE_WARPS)

            var a_stage = stage_ptr[A_STAGE_STRIDE](a_smem, stage)
            var b_stage = stage_ptr[B_STAGE_STRIDE](b_smem, stage)
            compute_stage[block_k, SS, warp_tile_m, warp_tile_n, num_c_tiles](
                c_acc, a_stage, b_stage, warp_m_idx, warp_n_idx, lid
            )
            increment_counter_if_first_lane(counters[stage])

    store_accum[cl, warp_tile_m, warp_tile_n, num_c_tiles](
        c,
        c_acc,
        block_m_offset,
        block_n_offset,
        warp_m_idx,
        warp_n_idx,
        lane_row_offset,
        lane_col,
        m,
        n,
    )
