from std.gpu import WARP_SIZE, barrier, block_idx, lane_id, thread_idx, warp_id
from std.gpu.memory import AddressSpace
from std.memory import bitcast, stack_allocation
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of
from std.utils import IndexList

from linalg.structuring import SMemArray
from layout import TensorLayout, TileTensor

from mojo.common import (
    AB,
    ASSUME_EVEN_N,
    ASSUME_EVEN_K,
    ASSUME_EVEN_MN,
    BK,
    BM,
    BN,
    BLOCK_SWIZZLE_SCALE,
    CD,
    COMPUTE_WARPS,
    DEQUANT_B_IN_BF16,
    GROUP_SIZE,
    GROUP_SIZE_M,
    LOAD_B_BY_QPACK,
    MMA_K,
    MMA_M,
    MMA_N,
    NUM_STAGES,
    PRODUCTION_TOTAL_WARPS,
    QPACK_K_VECTOR_WIDTH,
    RING_PRODUCER_WARPS,
    RING_STARTUP_ALL_WARPS,
    SCALE_AFTER_GROUP,
    SMEM_PAD,
    USE_FP16,
    USE_QZEROS,
    USE_LDS_SWIZZLE,
    WARPS_M,
    WARPS_N,
    ZERO_OFFSET,
    ZP_BIAS,
    dtype_acc,
    dtype_in,
    dtype_out,
    dtype_q,
)
from mojo.ring_buffer import (
    increment_counter_if_first_lane,
    stage_ptr,
    wait_for_counter,
    wait_for_ring_stage_stores,
)


comptime W4_PACK = 8
comptime W4_KPACK_COLS = 4
comptime W4_SIMD = min(W4_PACK, simd_width_of[dtype_in]())
comptime SMEM_VECTOR_WIDTH = simd_width_of[dtype_in]()


@always_inline
def wmma(
    a: SIMD[dtype_in, MMA_K],
    b: SIMD[dtype_in, MMA_K],
    c: SIMD[dtype_acc, CD],
) -> SIMD[dtype_acc, CD]:
    comptime if USE_FP16:
        return llvm_intrinsic[
            "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
            SIMD[dtype_acc, CD],
        ](a, b, c)
    var ai = bitcast[DType.int16](a)
    var bi = bitcast[DType.int16](b)
    return llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
        SIMD[dtype_acc, CD],
    ](ai, bi, c)


@always_inline
def fdot2_bf16(
    a: SIMD[dtype_in, 2],
    b: SIMD[dtype_in, 2],
    c: Float32,
) -> Float32:
    comptime if USE_FP16:
        return (
            c
            + a[0].cast[dtype_acc]() * b[0].cast[dtype_acc]()
            + a[1].cast[dtype_acc]() * b[1].cast[dtype_acc]()
        )
    return llvm_intrinsic[
        "llvm.amdgcn.fdot2.f32.bf16",
        Float32,
    ](a, b, c, False)


@always_inline
def block_swizzle(
    block_idx: IndexList[2, ...], grid_dim: type_of(block_idx)
) -> type_of(block_idx):
    comptime if GROUP_SIZE_M > 0:
        return _grouped_m_swizzle[GROUP_SIZE_M](block_idx, grid_dim)
    comptime if BLOCK_SWIZZLE_SCALE > 0:
        return _block_swizzle_by_scale[BLOCK_SWIZZLE_SCALE](block_idx, grid_dim)
    return block_idx


@always_inline
def _grouped_m_swizzle[
    group_size_m: Int
](block_idx: IndexList[2, ...], grid_dim: type_of(block_idx)) -> type_of(
    block_idx
):
    var num_pid_m = Int(grid_dim.data[1])
    var num_pid_n = Int(grid_dim.data[0])
    var pid = Int(block_idx.data[1]) * num_pid_n + Int(block_idx.data[0])
    var pids_per_group = group_size_m * num_pid_n
    var group_id = pid // pids_per_group
    var first_pid_m = group_id * group_size_m
    var rem_m = num_pid_m - first_pid_m
    var actual_group_size_m = rem_m if rem_m < group_size_m else group_size_m
    var pid_in_group = pid % pids_per_group
    var pid_m = first_pid_m + (pid_in_group % actual_group_size_m)
    var pid_n = pid_in_group // actual_group_size_m
    return {pid_n, pid_m}


@always_inline
def _block_swizzle_by_scale[
    scale0: Int
](block_idx: IndexList[2, ...], grid_dim: type_of(block_idx)) -> type_of(
    block_idx
):
    var scale = Scalar[block_idx.element_type](scale0)
    var num_partitions = 1 << Int(scale)
    while (
        grid_dim.data[0] & Scalar[block_idx.element_type](num_partitions - 1)
    ) and scale > 0:
        scale -= 1
        num_partitions = 1 << Int(scale)

    var bx = block_idx.data[0] >> scale
    var by = (block_idx.data[1] << scale) + (
        block_idx.data[0]
        & Scalar[block_idx.element_type]((1 << Int(scale)) - 1)
    )
    bx = bx + by // grid_dim.data[1] * (grid_dim.data[0] >> scale)
    by = by % grid_dim.data[1]
    return {Int(bx), Int(by)}


@always_inline
def a_smem_col(row: Int, col: Int) -> Int:
    comptime if USE_LDS_SWIZZLE:
        var chunk = col // AB
        var lane = col % AB
        return (chunk ^ (row & 1)) * AB + lane
    return col


@always_inline
def b_smem_col(row: Int, col: Int) -> Int:
    comptime if USE_LDS_SWIZZLE:
        var chunk = col // AB
        var lane = col % AB
        return (chunk ^ (row & 1)) * AB + lane
    return col


@always_inline
def load_a_smem_tile[
    BR: Int,
    block_k: Int,
    loader_warps: Int,
    al: TensorLayout,
](
    smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    a: TileTensor[dtype_in, al, ImmutAnyOrigin],
    block_row_offset: Int,
    k_offset: Int,
    max_rows: Int,
    max_k: Int,
    tid: Int,
):
    comptime assert a.flat_rank == 2
    comptime VW = (
        min(block_k, AB) if USE_LDS_SWIZZLE else min(block_k, SMEM_VECTOR_WIDTH)
    )
    comptime TV = (BR * block_k) // VW
    comptime VPT = (TV + loader_warps * WARP_SIZE - 1) // (
        loader_warps * WARP_SIZE
    )
    comptime SS = block_k + SMEM_PAD

    comptime for i in range(VPT):
        var vec_idx = i * (loader_warps * WARP_SIZE) + tid
        if vec_idx < TV:
            var elem_idx = vec_idx * VW
            var row = elem_idx // block_k
            var col = elem_idx % block_k
            var global_row = block_row_offset + row
            var vec = SIMD[dtype_in, VW](0)
            if global_row < max_rows:
                comptime if ASSUME_EVEN_K:
                    vec = a.load_linear[width=VW](
                        IndexList[2](elems=(global_row, k_offset + col))
                    )
                else:
                    var in_bounds = k_offset + col + VW <= max_k
                    if in_bounds:
                        vec = a.load_linear[width=VW](
                            IndexList[2](elems=(global_row, k_offset + col))
                        )
                    else:
                        comptime for v in range(VW):
                            var global_k = k_offset + col + v
                            if global_k < max_k:
                                vec[v] = a[global_row, global_k]
            smem.store(row * SS + a_smem_col(row, col), vec)


@always_inline
def _dequant_w4[
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    q_origin: Origin,
    z_origin: Origin,
    s_origin: Origin,
](
    qweight: TileTensor[dtype_q, ql, q_origin],
    qzeros: TileTensor[dtype_q, zl, z_origin],
    scales: TileTensor[dtype_in, sl, s_origin],
    global_k: Int,
    global_n: Int,
) -> Scalar[dtype_in]:
    comptime assert qweight.flat_rank == 2
    comptime assert qzeros.flat_rank == 2
    comptime assert scales.flat_rank == 2
    var packed = qweight[global_k, global_n // 8]
    var shift = Int32((global_n & 7) * 4)
    var nibble = (packed >> shift) & Scalar[dtype_q](0xF)
    var zero = Float32(ZP_BIAS)
    comptime if USE_QZEROS:
        var zpacked = qzeros[global_k // GROUP_SIZE, global_n // 8]
        var znibble = (zpacked >> shift) & Scalar[dtype_q](0xF)
        zero = znibble.cast[dtype_acc]() + Float32(ZERO_OFFSET)
    var signed_w = nibble.cast[dtype_acc]() - zero
    var scale = scales[global_k // GROUP_SIZE, global_n].cast[dtype_acc]()
    return Scalar[dtype_in](signed_w * scale)


@always_inline
def _fill_qpack_values[
    KQ: Int,
    QV: Int,
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    q_origin: Origin,
    z_origin: Origin,
    s_origin: Origin,
](
    mut values_by_k: InlineArray[SIMD[dtype_in, QV], KQ],
    qweight: TileTensor[dtype_q, ql, q_origin],
    qzeros: TileTensor[dtype_q, zl, z_origin],
    scales: TileTensor[dtype_in, sl, s_origin],
    global_k: Int,
    global_n_base: Int,
    group: Int,
    shifts: SIMD[dtype_q, QV],
    scale_vec: SIMD[dtype_in, QV],
    kv: Int,
):
    comptime assert qweight.flat_rank == 2
    comptime assert qzeros.flat_rank == 2
    comptime assert scales.flat_rank == 2
    var packed = rebind[Scalar[dtype_q]](
        qweight[global_k, global_n_base // W4_PACK]
    )
    var packed_vec = SIMD[dtype_q, QV](packed)
    var nibbles = (packed_vec >> shifts) & SIMD[dtype_q, QV](0xF)
    var zero_acc = SIMD[dtype_acc, QV](Float32(ZP_BIAS))
    comptime if USE_QZEROS:
        var zpacked = rebind[Scalar[dtype_q]](
            qzeros[group, global_n_base // W4_PACK]
        )
        var znibbles = (SIMD[dtype_q, QV](zpacked) >> shifts) & SIMD[
            dtype_q, QV
        ](0xF)
        zero_acc = znibbles.cast[dtype_acc]() + SIMD[dtype_acc, QV](
            Float32(ZERO_OFFSET)
        )
    comptime if SCALE_AFTER_GROUP:
        values_by_k[kv] = (nibbles.cast[dtype_acc]() - zero_acc).cast[
            dtype_in
        ]()
    elif DEQUANT_B_IN_BF16:
        var signed_w = (nibbles.cast[dtype_acc]() - zero_acc).cast[dtype_in]()
        values_by_k[kv] = signed_w * scale_vec
    else:
        var signed_w = nibbles.cast[dtype_acc]() - zero_acc
        values_by_k[kv] = (signed_w * scale_vec.cast[dtype_acc]()).cast[
            dtype_in
        ]()


@always_inline
def _fill_qpack_tail_values[
    KQ: Int,
    QV: Int,
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    q_origin: Origin,
    z_origin: Origin,
    s_origin: Origin,
](
    mut values_by_k: InlineArray[SIMD[dtype_in, QV], KQ],
    qweight: TileTensor[dtype_q, ql, q_origin],
    qzeros: TileTensor[dtype_q, zl, z_origin],
    scales: TileTensor[dtype_in, sl, s_origin],
    global_k: Int,
    global_n_base: Int,
    max_n: Int,
    group: Int,
    kv: Int,
):
    comptime assert qweight.flat_rank == 2
    comptime assert qzeros.flat_rank == 2
    comptime assert scales.flat_rank == 2
    var packed = rebind[Scalar[dtype_q]](
        qweight[global_k, global_n_base // W4_PACK]
    )
    comptime for v in range(QV):
        var global_n = global_n_base + v
        if global_n < max_n:
            var shift = Int32((global_n & 7) * 4)
            var nibble = (packed >> shift) & Scalar[dtype_q](0xF)
            var zero = Float32(ZP_BIAS)
            comptime if USE_QZEROS:
                var zpacked = rebind[Scalar[dtype_q]](
                    qzeros[group, global_n // W4_PACK]
                )
                var znibble = (zpacked >> shift) & Scalar[dtype_q](0xF)
                zero = znibble.cast[dtype_acc]() + Float32(ZERO_OFFSET)
            comptime if SCALE_AFTER_GROUP:
                values_by_k[kv][v] = Scalar[dtype_in](
                    nibble.cast[dtype_acc]() - zero
                )
            elif DEQUANT_B_IN_BF16:
                var signed_w = Scalar[dtype_in](nibble.cast[dtype_acc]() - zero)
                var scale = scales[group, global_n]
                values_by_k[kv][v] = signed_w * scale
            else:
                var signed_w = nibble.cast[dtype_acc]() - zero
                var scale = scales[group, global_n].cast[dtype_acc]()
                values_by_k[kv][v] = Scalar[dtype_in](signed_w * scale)


@always_inline
def load_b_dequant_smem_tile[
    BR: Int,
    block_k: Int,
    loader_warps: Int,
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    q_origin: Origin,
    z_origin: Origin,
    s_origin: Origin,
](
    smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    qweight: TileTensor[dtype_q, ql, q_origin],
    qzeros: TileTensor[dtype_q, zl, z_origin],
    scales: TileTensor[dtype_in, sl, s_origin],
    block_n_offset: Int,
    k_offset: Int,
    max_n: Int,
    max_k: Int,
    tid: Int,
):
    comptime VW = min(block_k, 8)
    comptime TV = (BR * block_k) // VW
    comptime VPT = (TV + loader_warps * WARP_SIZE - 1) // (
        loader_warps * WARP_SIZE
    )
    comptime SS = block_k + SMEM_PAD

    comptime for i in range(VPT):
        var vec_idx = i * (loader_warps * WARP_SIZE) + tid
        if vec_idx < TV:
            var elem_idx = vec_idx * VW
            var row = elem_idx // block_k
            var col = elem_idx % block_k
            var global_n = block_n_offset + row
            var vec = SIMD[dtype_in, VW](0)
            comptime if ASSUME_EVEN_N:
                comptime for v in range(VW):
                    var global_k = k_offset + col + v
                    comptime if ASSUME_EVEN_K:
                        vec[v] = _dequant_w4[
                            ql, zl, sl, q_origin, z_origin, s_origin
                        ](qweight, qzeros, scales, global_k, global_n)
                    else:
                        if global_k < max_k:
                            vec[v] = _dequant_w4[
                                ql, zl, sl, q_origin, z_origin, s_origin
                            ](qweight, qzeros, scales, global_k, global_n)
            else:
                if global_n < max_n:
                    comptime for v in range(VW):
                        var global_k = k_offset + col + v
                        comptime if ASSUME_EVEN_K:
                            vec[v] = _dequant_w4[
                                ql, zl, sl, q_origin, z_origin, s_origin
                            ](qweight, qzeros, scales, global_k, global_n)
                        else:
                            if global_k < max_k:
                                vec[v] = _dequant_w4[
                                    ql, zl, sl, q_origin, z_origin, s_origin
                                ](qweight, qzeros, scales, global_k, global_n)
            smem.store(row * SS + b_smem_col(row, col), vec)


@always_inline
def load_b_dequant_smem_tile_qpack[
    BR: Int,
    block_k: Int,
    loader_warps: Int,
    ql: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    q_origin: Origin,
    z_origin: Origin,
    s_origin: Origin,
](
    smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    qweight: TileTensor[dtype_q, ql, q_origin],
    qzeros: TileTensor[dtype_q, zl, z_origin],
    scales: TileTensor[dtype_in, sl, s_origin],
    block_n_offset: Int,
    k_offset: Int,
    max_n: Int,
    max_k: Int,
    tid: Int,
):
    comptime assert qweight.flat_rank == 2
    comptime assert qzeros.flat_rank == 2
    comptime assert scales.flat_rank == 2
    comptime assert BR % W4_PACK == 0
    comptime QV = W4_SIMD
    comptime assert W4_PACK % QV == 0
    comptime assert QPACK_K_VECTOR_WIDTH > 0
    comptime MAX_QPACK_K_VECTOR_WIDTH = AB if USE_LDS_SWIZZLE else MMA_K
    comptime assert QPACK_K_VECTOR_WIDTH <= MAX_QPACK_K_VECTOR_WIDTH
    comptime KQ = min(block_k, QPACK_K_VECTOR_WIDTH)
    comptime assert block_k % KQ == 0
    comptime TV = (BR // QV) * (block_k // KQ)
    comptime VPT = (TV + loader_warps * WARP_SIZE - 1) // (
        loader_warps * WARP_SIZE
    )
    comptime SS = block_k + SMEM_PAD

    comptime for i in range(VPT):
        var elem_idx = i * (loader_warps * WARP_SIZE) + tid
        if elem_idx < TV:
            var k_chunks = block_k // KQ
            var pack_row = elem_idx // k_chunks
            var col = (elem_idx % k_chunks) * KQ
            var global_n_base = block_n_offset + pack_row * QV
            var values_by_k = InlineArray[SIMD[dtype_in, QV], KQ](
                fill=SIMD[dtype_in, QV](0)
            )
            comptime if ASSUME_EVEN_N:
                var shifts = SIMD[dtype_q, QV](0)
                comptime for v in range(QV):
                    shifts[v] = Scalar[dtype_q](((global_n_base + v) & 7) * 4)

                # block_k is constrained to stay inside one quantization group.
                var group = k_offset // GROUP_SIZE
                var scale_vec = SIMD[dtype_in, QV](0)
                comptime if not SCALE_AFTER_GROUP:
                    scale_vec = scales.load_linear[width=QV](
                        IndexList[2](elems=(group, global_n_base))
                    )
                comptime for kv in range(KQ):
                    var global_k = k_offset + col + kv
                    comptime if ASSUME_EVEN_K:
                        _fill_qpack_values[
                            KQ, QV, ql, zl, sl, q_origin, z_origin, s_origin
                        ](
                            values_by_k,
                            qweight,
                            qzeros,
                            scales,
                            global_k,
                            global_n_base,
                            group,
                            shifts,
                            scale_vec,
                            kv,
                        )
                    else:
                        if global_k < max_k:
                            _fill_qpack_values[
                                KQ, QV, ql, zl, sl, q_origin, z_origin, s_origin
                            ](
                                values_by_k,
                                qweight,
                                qzeros,
                                scales,
                                global_k,
                                global_n_base,
                                group,
                                shifts,
                                scale_vec,
                                kv,
                            )
            else:
                if global_n_base < max_n:
                    var shifts = SIMD[dtype_q, QV](0)
                    comptime for v in range(QV):
                        shifts[v] = Scalar[dtype_q](
                            ((global_n_base + v) & 7) * 4
                        )

                    # block_k is constrained to stay inside one quantization group.
                    var group = k_offset // GROUP_SIZE
                    if global_n_base + QV <= max_n:
                        var scale_vec = SIMD[dtype_in, QV](0)
                        comptime if not SCALE_AFTER_GROUP:
                            scale_vec = scales.load_linear[width=QV](
                                IndexList[2](elems=(group, global_n_base))
                            )
                        comptime for kv in range(KQ):
                            var global_k = k_offset + col + kv
                            comptime if ASSUME_EVEN_K:
                                _fill_qpack_values[
                                    KQ,
                                    QV,
                                    ql,
                                    zl,
                                    sl,
                                    q_origin,
                                    z_origin,
                                    s_origin,
                                ](
                                    values_by_k,
                                    qweight,
                                    qzeros,
                                    scales,
                                    global_k,
                                    global_n_base,
                                    group,
                                    shifts,
                                    scale_vec,
                                    kv,
                                )
                            else:
                                if global_k < max_k:
                                    _fill_qpack_values[
                                        KQ,
                                        QV,
                                        ql,
                                        zl,
                                        sl,
                                        q_origin,
                                        z_origin,
                                        s_origin,
                                    ](
                                        values_by_k,
                                        qweight,
                                        qzeros,
                                        scales,
                                        global_k,
                                        global_n_base,
                                        group,
                                        shifts,
                                        scale_vec,
                                        kv,
                                    )
                    else:
                        comptime for kv in range(KQ):
                            var global_k = k_offset + col + kv
                            comptime if ASSUME_EVEN_K:
                                _fill_qpack_tail_values[
                                    KQ,
                                    QV,
                                    ql,
                                    zl,
                                    sl,
                                    q_origin,
                                    z_origin,
                                    s_origin,
                                ](
                                    values_by_k,
                                    qweight,
                                    qzeros,
                                    scales,
                                    global_k,
                                    global_n_base,
                                    max_n,
                                    group,
                                    kv,
                                )
                            else:
                                if global_k < max_k:
                                    _fill_qpack_tail_values[
                                        KQ,
                                        QV,
                                        ql,
                                        zl,
                                        sl,
                                        q_origin,
                                        z_origin,
                                        s_origin,
                                    ](
                                        values_by_k,
                                        qweight,
                                        qzeros,
                                        scales,
                                        global_k,
                                        global_n_base,
                                        max_n,
                                        group,
                                        kv,
                                    )
            comptime for v in range(QV):
                var values = SIMD[dtype_in, KQ](0)
                comptime for kv in range(KQ):
                    values[kv] = values_by_k[kv][v]
                smem.store(
                    (pack_row * QV + v) * SS
                    + b_smem_col(pack_row * QV + v, col),
                    values,
                )


@always_inline
def compute_stage[
    block_k: Int,
    SS: Int,
    warp_tile_m: Int,
    warp_tile_n: Int,
    num_c_tiles: Int,
](
    mut c_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    a_smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    b_smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    warp_m_idx: Int,
    warp_n_idx: Int,
    lid: Int,
):
    var elem_lane = lid % 16

    comptime for k_inner in range(block_k // MMA_K):
        comptime k_base = k_inner * MMA_K

        var a_frag = InlineArray[SIMD[dtype_in, MMA_K], warp_tile_m](
            fill=SIMD[dtype_in, MMA_K](0)
        )
        comptime for wm in range(warp_tile_m):
            var a_src_row = warp_m_idx * warp_tile_m * 16 + wm * 16 + elem_lane
            var a_row = a_src_row * SS
            comptime if USE_LDS_SWIZZLE:
                var a_lo = bitcast[dtype_in](
                    a_smem.load[width=AB](a_row + a_smem_col(a_src_row, k_base))
                )
                var a_hi = bitcast[dtype_in](
                    a_smem.load[width=AB](
                        a_row + a_smem_col(a_src_row, k_base + AB)
                    )
                )
                comptime for v in range(AB):
                    a_frag[wm][v] = a_lo[v]
                    a_frag[wm][v + AB] = a_hi[v]
            else:
                a_frag[wm] = bitcast[dtype_in](
                    a_smem.load[width=MMA_K](a_row + k_base)
                )

        var b_frag = InlineArray[SIMD[dtype_in, MMA_K], warp_tile_n](
            fill=SIMD[dtype_in, MMA_K](0)
        )
        comptime for wn in range(warp_tile_n):
            var b_src_row = warp_n_idx * warp_tile_n * 16 + wn * 16 + elem_lane
            var b_row = b_src_row * SS
            comptime if USE_LDS_SWIZZLE:
                var b_lo = bitcast[dtype_in](
                    b_smem.load[width=AB](b_row + b_smem_col(b_src_row, k_base))
                )
                var b_hi = bitcast[dtype_in](
                    b_smem.load[width=AB](
                        b_row + b_smem_col(b_src_row, k_base + AB)
                    )
                )
                comptime for v in range(AB):
                    b_frag[wn][v] = b_lo[v]
                    b_frag[wn][v + AB] = b_hi[v]
            else:
                b_frag[wn] = bitcast[dtype_in](
                    b_smem.load[width=MMA_K](b_row + k_base)
                )

        comptime for wm in range(warp_tile_m):
            comptime for wn in range(warp_tile_n):
                var c_idx = wm * warp_tile_n + wn
                c_acc[c_idx] = wmma(a_frag[wm], b_frag[wn], c_acc[c_idx])


@always_inline
def compute_stage_direct_a[
    block_k: Int,
    SS: Int,
    warp_tile_m: Int,
    warp_tile_n: Int,
    num_c_tiles: Int,
    al: TensorLayout,
    a_origin: Origin,
](
    mut c_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    a: TileTensor[dtype_in, al, a_origin],
    b_smem: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    block_m_offset: Int,
    k_offset: Int,
    max_m: Int,
    max_k: Int,
    warp_m_idx: Int,
    warp_n_idx: Int,
    lid: Int,
):
    comptime assert a.flat_rank == 2
    var elem_lane = lid % 16

    comptime for k_inner in range(block_k // MMA_K):
        comptime k_base = k_inner * MMA_K

        var a_frag = InlineArray[SIMD[dtype_in, MMA_K], warp_tile_m](
            fill=SIMD[dtype_in, MMA_K](0)
        )
        comptime for wm in range(warp_tile_m):
            var global_m = (
                block_m_offset
                + warp_m_idx * warp_tile_m * 16
                + wm * 16
                + elem_lane
            )
            if global_m < max_m:
                comptime if ASSUME_EVEN_K:
                    a_frag[wm] = bitcast[dtype_in](
                        a.load_linear[width=MMA_K](
                            IndexList[2](elems=(global_m, k_offset + k_base))
                        )
                    )
                else:
                    if k_offset + k_base + MMA_K <= max_k:
                        a_frag[wm] = bitcast[dtype_in](
                            a.load_linear[width=MMA_K](
                                IndexList[2](
                                    elems=(global_m, k_offset + k_base)
                                )
                            )
                        )
                    else:
                        var a_vec = SIMD[dtype_in, MMA_K](0)
                        comptime for v in range(MMA_K):
                            var global_k = k_offset + k_base + v
                            if global_k < max_k:
                                a_vec[v] = a[global_m, global_k]
                        a_frag[wm] = bitcast[dtype_in](a_vec)

        var b_frag = InlineArray[SIMD[dtype_in, MMA_K], warp_tile_n](
            fill=SIMD[dtype_in, MMA_K](0)
        )
        comptime for wn in range(warp_tile_n):
            var b_src_row = warp_n_idx * warp_tile_n * 16 + wn * 16 + elem_lane
            var b_row = b_src_row * SS
            comptime if USE_LDS_SWIZZLE:
                var b_lo = bitcast[dtype_in](
                    b_smem.load[width=AB](b_row + b_smem_col(b_src_row, k_base))
                )
                var b_hi = bitcast[dtype_in](
                    b_smem.load[width=AB](
                        b_row + b_smem_col(b_src_row, k_base + AB)
                    )
                )
                comptime for v in range(AB):
                    b_frag[wn][v] = b_lo[v]
                    b_frag[wn][v + AB] = b_hi[v]
            else:
                b_frag[wn] = bitcast[dtype_in](
                    b_smem.load[width=MMA_K](b_row + k_base)
                )

        comptime for wm in range(warp_tile_m):
            comptime for wn in range(warp_tile_n):
                var c_idx = wm * warp_tile_n + wn
                c_acc[c_idx] = wmma(a_frag[wm], b_frag[wn], c_acc[c_idx])


@always_inline
def accumulate_scaled_group[
    sl: TensorLayout,
    warp_tile_m: Int,
    warp_tile_n: Int,
    num_c_tiles: Int,
    s_origin: Origin,
](
    mut c_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    mut group_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    scales: TileTensor[dtype_in, sl, s_origin],
    block_n_offset: Int,
    group_id: Int,
    warp_n_idx: Int,
    lane_col: Int,
    n: Int,
):
    comptime assert scales.flat_rank == 2
    comptime for wn in range(warp_tile_n):
        var global_col = (
            block_n_offset + warp_n_idx * warp_tile_n * 16 + wn * 16 + lane_col
        )
        var scale = Scalar[dtype_acc](0)
        if global_col < n:
            scale = scales[group_id, global_col].cast[dtype_acc]()
        comptime for wm in range(warp_tile_m):
            var c_idx = wm * warp_tile_n + wn
            c_acc[c_idx] += group_acc[c_idx] * scale
            group_acc[c_idx] = SIMD[dtype_acc, CD](0)


@always_inline
def store_accum[
    cl: TensorLayout,
    warp_tile_m: Int,
    warp_tile_n: Int,
    num_c_tiles: Int,
](
    c: TileTensor[mut=True, dtype_out, cl, MutAnyOrigin],
    c_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    block_m_offset: Int,
    block_n_offset: Int,
    warp_m_idx: Int,
    warp_n_idx: Int,
    lane_row_offset: Int,
    lane_col: Int,
    m: Int,
    n: Int,
):
    comptime assert c.flat_rank == 2
    comptime for wm in range(warp_tile_m):
        comptime for wn in range(warp_tile_n):
            var c_idx = wm * warp_tile_n + wn
            comptime for v in range(CD):
                var global_row = (
                    block_m_offset
                    + warp_m_idx * warp_tile_m * 16
                    + wm * 16
                    + v * 2
                    + lane_row_offset
                )
                var global_col = (
                    block_n_offset
                    + warp_n_idx * warp_tile_n * 16
                    + wn * 16
                    + lane_col
                )
                comptime if ASSUME_EVEN_MN:
                    c[global_row, global_col] = c_acc[c_idx][v].cast[
                        dtype_out
                    ]()
                elif ASSUME_EVEN_N:
                    if global_row < m:
                        c[global_row, global_col] = c_acc[c_idx][v].cast[
                            dtype_out
                        ]()
                else:
                    if global_row < m and global_col < n:
                        c[global_row, global_col] = c_acc[c_idx][v].cast[
                            dtype_out
                        ]()


@always_inline
def store_accum_partial[
    pl: TensorLayout,
    warp_tile_m: Int,
    warp_tile_n: Int,
    num_c_tiles: Int,
](
    partial: TileTensor[mut=True, dtype_acc, pl, MutAnyOrigin],
    c_acc: InlineArray[SIMD[dtype_acc, CD], num_c_tiles],
    split: Int,
    block_m_offset: Int,
    block_n_offset: Int,
    warp_m_idx: Int,
    warp_n_idx: Int,
    lane_row_offset: Int,
    lane_col: Int,
    m: Int,
    n: Int,
):
    comptime assert partial.flat_rank == 3
    comptime for wm in range(warp_tile_m):
        comptime for wn in range(warp_tile_n):
            var c_idx = wm * warp_tile_n + wn
            comptime for v in range(CD):
                var global_row = (
                    block_m_offset
                    + warp_m_idx * warp_tile_m * 16
                    + wm * 16
                    + v * 2
                    + lane_row_offset
                )
                var global_col = (
                    block_n_offset
                    + warp_n_idx * warp_tile_n * 16
                    + wn * 16
                    + lane_col
                )
                comptime if ASSUME_EVEN_MN:
                    partial[split, global_row, global_col] = c_acc[c_idx][v]
                elif ASSUME_EVEN_N:
                    if global_row < m:
                        partial[split, global_row, global_col] = c_acc[c_idx][v]
                else:
                    if global_row < m and global_col < n:
                        partial[split, global_row, global_col] = c_acc[c_idx][v]
