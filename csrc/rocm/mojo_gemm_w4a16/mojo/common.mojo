from std.gpu import WARP_SIZE
from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.sys import get_defined_bool, get_defined_int, is_defined
from std.testing import assert_almost_equal

from layout.tile_layout import row_major

from mojo.config_defaults import (
    DEFAULT_ASSUME_EVEN_K,
    DEFAULT_ASSUME_EVEN_MN,
    DEFAULT_ASSUME_EVEN_N,
    DEFAULT_BENCH_ITERS,
    DEFAULT_BK,
    DEFAULT_BLOCK_SWIZZLE_SCALE,
    DEFAULT_BM,
    DEFAULT_BN,
    DEFAULT_SPLITK_BLOCK_K,
    DEFAULT_SPLITK_ROWS_PER_CTA,
    DEFAULT_SPLITK_THREADS,
    DEFAULT_DEQUANT_B_IN_BF16,
    DEFAULT_GROUP_SIZE,
    DEFAULT_GROUP_SIZE_M,
    DEFAULT_KERNEL_VARIANT,
    DEFAULT_LOAD_B_BY_QPACK,
    DEFAULT_MAX_K,
    DEFAULT_MAX_M,
    DEFAULT_MAX_N,
    DEFAULT_NUM_STAGES,
    DEFAULT_QPACK_K_VECTOR_WIDTH,
    DEFAULT_RING_STARTUP_ALL_WARPS,
    DEFAULT_RING_PRODUCER_WARPS,
    DEFAULT_SCALE_AFTER_GROUP,
    DEFAULT_SMEM_PAD,
    DEFAULT_USE_LDS_SWIZZLE,
    DEFAULT_USE_FP16,
    DEFAULT_WARMUP_ITERS,
    DEFAULT_WARPS_M,
    DEFAULT_WARPS_N,
    DEFAULT_USE_QZEROS,
    DEFAULT_ZERO_OFFSET,
    DEFAULT_ZP_BIAS,
)


comptime MAX_M = (
    get_defined_int["MAX_M"]() if is_defined["MAX_M"]() else DEFAULT_MAX_M
)
comptime MAX_N = (
    get_defined_int["MAX_N"]() if is_defined["MAX_N"]() else DEFAULT_MAX_N
)
comptime MAX_K = (
    get_defined_int["MAX_K"]() if is_defined["MAX_K"]() else DEFAULT_MAX_K
)
comptime USE_FP16 = (
    get_defined_bool["USE_FP16", False]() if is_defined[
        "USE_FP16"
    ]() else DEFAULT_USE_FP16
)

comptime BM = get_defined_int["BM"]() if is_defined["BM"]() else DEFAULT_BM
comptime BN = get_defined_int["BN"]() if is_defined["BN"]() else DEFAULT_BN
comptime BK = get_defined_int["BK"]() if is_defined["BK"]() else DEFAULT_BK

comptime GROUP_SIZE = (
    get_defined_int["GROUP_SIZE"]() if is_defined[
        "GROUP_SIZE"
    ]() else DEFAULT_GROUP_SIZE
)
comptime ZP_BIAS = (
    get_defined_int["ZP_BIAS"]() if is_defined["ZP_BIAS"]() else DEFAULT_ZP_BIAS
)
comptime USE_QZEROS = (
    get_defined_bool["USE_QZEROS", False]() if is_defined[
        "USE_QZEROS"
    ]() else DEFAULT_USE_QZEROS
)
comptime ZERO_OFFSET = (
    get_defined_int["ZERO_OFFSET"]() if is_defined[
        "ZERO_OFFSET"
    ]() else DEFAULT_ZERO_OFFSET
)

comptime MMA_M = 16
comptime MMA_N = 16
comptime MMA_K = 16
comptime AB = 8
comptime CD = 8
comptime SMEM_PAD = (
    get_defined_int["SMEM_PAD"]() if is_defined[
        "SMEM_PAD"
    ]() else DEFAULT_SMEM_PAD
)

comptime RING_PRODUCER_WARPS = (
    get_defined_int["RING_PRODUCER_WARPS"]() if is_defined[
        "RING_PRODUCER_WARPS"
    ]() else DEFAULT_RING_PRODUCER_WARPS
)
comptime NUM_STAGES = (
    get_defined_int["NUM_STAGES"]() if is_defined["NUM_STAGES"]() else (
        get_defined_int["RING_STAGES"]() if is_defined[
            "RING_STAGES"
        ]() else DEFAULT_NUM_STAGES
    )
)
comptime BLOCK_SWIZZLE_SCALE = (
    get_defined_int["BLOCK_SWIZZLE_SCALE"]() if is_defined[
        "BLOCK_SWIZZLE_SCALE"
    ]() else DEFAULT_BLOCK_SWIZZLE_SCALE
)
comptime GROUP_SIZE_M = (
    get_defined_int["GROUP_SIZE_M"]() if is_defined[
        "GROUP_SIZE_M"
    ]() else DEFAULT_GROUP_SIZE_M
)
comptime USE_LDS_SWIZZLE = (
    get_defined_bool["USE_LDS_SWIZZLE", False]() if is_defined[
        "USE_LDS_SWIZZLE"
    ]() else DEFAULT_USE_LDS_SWIZZLE
)
comptime RING_STARTUP_ALL_WARPS = (
    get_defined_bool["RING_STARTUP_ALL_WARPS", True]() if is_defined[
        "RING_STARTUP_ALL_WARPS"
    ]() else DEFAULT_RING_STARTUP_ALL_WARPS
)
comptime LOAD_B_BY_QPACK = (
    get_defined_bool["LOAD_B_BY_QPACK", True]() if is_defined[
        "LOAD_B_BY_QPACK"
    ]() else DEFAULT_LOAD_B_BY_QPACK
)
comptime QPACK_K_VECTOR_WIDTH = (
    get_defined_int["QPACK_K_VECTOR_WIDTH"]() if is_defined[
        "QPACK_K_VECTOR_WIDTH"
    ]() else DEFAULT_QPACK_K_VECTOR_WIDTH
)
comptime DEQUANT_B_IN_BF16 = (
    get_defined_bool["DEQUANT_B_IN_BF16", False]() if is_defined[
        "DEQUANT_B_IN_BF16"
    ]() else DEFAULT_DEQUANT_B_IN_BF16
)
comptime SCALE_AFTER_GROUP = (
    get_defined_bool["SCALE_AFTER_GROUP", True]() if is_defined[
        "SCALE_AFTER_GROUP"
    ]() else DEFAULT_SCALE_AFTER_GROUP
)
comptime ASSUME_EVEN_K = (
    get_defined_bool["ASSUME_EVEN_K", False]() if is_defined[
        "ASSUME_EVEN_K"
    ]() else DEFAULT_ASSUME_EVEN_K
)
comptime ASSUME_EVEN_MN = (
    get_defined_bool["ASSUME_EVEN_MN", False]() if is_defined[
        "ASSUME_EVEN_MN"
    ]() else DEFAULT_ASSUME_EVEN_MN
)
comptime ASSUME_EVEN_N = (
    get_defined_bool["ASSUME_EVEN_N", False]() if is_defined[
        "ASSUME_EVEN_N"
    ]() else DEFAULT_ASSUME_EVEN_N
)
comptime KERNEL_VARIANT = DEFAULT_KERNEL_VARIANT
comptime KERNEL_USES_FDOT2 = (
    get_defined_bool["USE_KPACKED_DOT2", False]() if is_defined[
        "USE_KPACKED_DOT2"
    ]() else KERNEL_VARIANT == "kpacked_dot2"
)
comptime USE_BENCH_PARTIAL = (
    get_defined_bool["USE_BENCH_PARTIAL", False]() if is_defined[
        "USE_BENCH_PARTIAL"
    ]() else False
)
comptime KERNEL_NEEDS_PARTIAL = (
    USE_BENCH_PARTIAL or KERNEL_VARIANT == "kpacked_dot2" or KERNEL_VARIANT == "wmma16"
)
comptime SPLITK_THREADS = (
    get_defined_int["SPLITK_THREADS"]() if is_defined[
        "SPLITK_THREADS"
    ]() else DEFAULT_SPLITK_THREADS
)
comptime SPLITK_BLOCK_K = (
    get_defined_int["SPLITK_BLOCK_K"]() if is_defined[
        "SPLITK_BLOCK_K"
    ]() else DEFAULT_SPLITK_BLOCK_K
)
comptime SPLITK_ROWS_PER_CTA = (
    get_defined_int["SPLITK_ROWS_PER_CTA"]() if is_defined[
        "SPLITK_ROWS_PER_CTA"
    ]() else DEFAULT_SPLITK_ROWS_PER_CTA
)
comptime WARPS_M = (
    get_defined_int["WARPS_M"]() if is_defined["WARPS_M"]() else DEFAULT_WARPS_M
)
comptime WARPS_N = (
    get_defined_int["WARPS_N"]() if is_defined["WARPS_N"]() else DEFAULT_WARPS_N
)
comptime COMPUTE_WARPS = WARPS_M * WARPS_N
comptime PRODUCTION_TOTAL_WARPS = RING_PRODUCER_WARPS + COMPUTE_WARPS
comptime PRODUCTION_TOTAL_THREADS = PRODUCTION_TOTAL_WARPS * WARP_SIZE

comptime dtype_in = DType.float16 if USE_FP16 else DType.bfloat16
comptime dtype_acc = DType.float32
comptime dtype_out = DType.float16 if USE_FP16 else DType.bfloat16
comptime dtype_q = DType.int32

comptime SPLITK_PARTITIONS = (MAX_K + SPLITK_BLOCK_K - 1) // SPLITK_BLOCK_K
comptime a_layout = row_major[MAX_M, MAX_K]()
comptime qweight_layout = row_major[MAX_K, MAX_N // 8]()
comptime qweight_kpacked_layout = row_major[MAX_K // 8, MAX_N]()
comptime qzeros_layout = row_major[MAX_K // GROUP_SIZE, MAX_N // 8]()
comptime scales_layout = row_major[MAX_K // GROUP_SIZE, MAX_N]()
comptime partial_layout = row_major[SPLITK_PARTITIONS, MAX_M, MAX_N]()
comptime c_layout = row_major[MAX_M, MAX_N]()
comptime ALayout = type_of(a_layout)
comptime QWeightLayout = type_of(qweight_layout)
comptime QWeightKPackedLayout = type_of(qweight_kpacked_layout)
comptime QZerosLayout = type_of(qzeros_layout)
comptime ScalesLayout = type_of(scales_layout)
comptime PartialLayout = type_of(partial_layout)
comptime CLayout = type_of(c_layout)

comptime WARMUP_ITERS = (
    get_defined_int["WARMUP_ITERS"]() if is_defined[
        "WARMUP_ITERS"
    ]() else DEFAULT_WARMUP_ITERS
)
comptime BENCH_ITERS = (
    get_defined_int["BENCH_ITERS"]() if is_defined[
        "BENCH_ITERS"
    ]() else DEFAULT_BENCH_ITERS
)


def a_value(row: Int, col: Int) -> Float32:
    return Float32(((row * 7 + col * 3) % 31) - 15) / 16.0


def qvalue(k: Int, n: Int) -> Int:
    # Keep the high packed nibble below bit 31 so int32 right shifts are benign.
    return (k * 13 + n * 5 + 3) & 7


def scale_value(group: Int, n: Int) -> Float32:
    return Float32(((group * 11 + n * 7) % 17) + 1) / 64.0


def qzero_value(group: Int, n: Int) -> Int:
    comptime if USE_QZEROS:
        return (group * 5 + n * 3 + 1) & 7
    return ZP_BIAS - ZERO_OFFSET


def dequant_weight(k: Int, n: Int) -> Float32:
    var group = k // GROUP_SIZE
    return Float32(
        qvalue(k, n) - (qzero_value(group, n) + ZERO_OFFSET)
    ) * scale_value(group, n)


def reference_value(row: Int, col: Int, k: Int) -> Float32:
    var acc: Float32 = 0.0
    for kk in range(k):
        acc += a_value(row, kk) * dequant_weight(kk, col)
    return acc


@fieldwise_init
struct Prob(Movable):
    var a: DeviceBuffer[dtype_in]
    var qweight: DeviceBuffer[dtype_q]
    var qweight_kpacked: DeviceBuffer[dtype_q]
    var qzeros: DeviceBuffer[dtype_q]
    var scales: DeviceBuffer[dtype_in]
    var partial: DeviceBuffer[dtype_acc]
    var c: DeviceBuffer[dtype_out]


def alloc_prob(ctx: DeviceContext) raises -> Prob:
    comptime assert MAX_N % 8 == 0
    comptime assert MAX_K % GROUP_SIZE == 0
    comptime assert BK <= GROUP_SIZE
    comptime assert GROUP_SIZE % BK == 0
    comptime assert SPLITK_BLOCK_K > 0
    comptime assert SPLITK_BLOCK_K % 2 == 0
    comptime assert SPLITK_ROWS_PER_CTA > 0
    comptime assert MAX_K % 8 == 0

    var a = ctx.enqueue_create_buffer[dtype_in](MAX_M * MAX_K)
    var qweight = ctx.enqueue_create_buffer[dtype_q](MAX_K * (MAX_N // 8))
    var qweight_kpacked = ctx.enqueue_create_buffer[dtype_q](
        (MAX_K // 8) * MAX_N
    )
    var qzeros = ctx.enqueue_create_buffer[dtype_q](
        (MAX_K // GROUP_SIZE) * (MAX_N // 8)
    )
    var scales = ctx.enqueue_create_buffer[dtype_in](
        (MAX_K // GROUP_SIZE) * MAX_N
    )
    var partial_count = 1
    comptime if KERNEL_NEEDS_PARTIAL:
        partial_count = SPLITK_PARTITIONS * MAX_M * MAX_N
    var partial = ctx.enqueue_create_buffer[dtype_acc](partial_count)
    var c = ctx.enqueue_create_buffer[dtype_out](MAX_M * MAX_N)

    with a.map_to_host() as h:
        for row in range(MAX_M):
            for col in range(MAX_K):
                h[row * MAX_K + col] = Scalar[dtype_in](a_value(row, col))

    with qweight.map_to_host() as h:
        for kk in range(MAX_K):
            for packed_n in range(MAX_N // 8):
                var packed = 0
                for i in range(8):
                    packed |= qvalue(kk, packed_n * 8 + i) << (i * 4)
                h[kk * (MAX_N // 8) + packed_n] = Scalar[dtype_q](packed)

    with qweight_kpacked.map_to_host() as h:
        for qk in range(MAX_K // 8):
            for col in range(MAX_N):
                var packed = 0
                for i in range(8):
                    packed |= qvalue(qk * 8 + i, col) << (i * 4)
                h[qk * MAX_N + col] = Scalar[dtype_q](packed)

    with qzeros.map_to_host() as h:
        for group in range(MAX_K // GROUP_SIZE):
            for packed_n in range(MAX_N // 8):
                var packed = 0
                for i in range(8):
                    packed |= qzero_value(group, packed_n * 8 + i) << (i * 4)
                h[group * (MAX_N // 8) + packed_n] = Scalar[dtype_q](packed)

    with scales.map_to_host() as h:
        for group in range(MAX_K // GROUP_SIZE):
            for col in range(MAX_N):
                h[group * MAX_N + col] = Scalar[dtype_in](
                    scale_value(group, col)
                )

    partial.enqueue_fill(0)
    c.enqueue_fill(0)
    return Prob(a, qweight, qweight_kpacked, qzeros, scales, partial, c)


def validate_output[
    out_dtype: DType,
](label: String, output: HostBuffer[out_dtype], m: Int, n: Int, k: Int) raises:
    for row in range(m):
        for col in range(n):
            var got = Float32(output[row * MAX_N + col].cast[dtype_acc]())
            var exp = reference_value(row, col, k)
            assert_almost_equal(got, exp, atol=0.75)

    print(label, "accuracy OK")
