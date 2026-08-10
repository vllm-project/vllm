// SPDX-License-Identifier: MIT
//
// Amalgamated and locally modified from DeepSeek FlashMLA
// commit 9241ae3ef9bac614dd25e45e507e089f888280e0.
// See LICENSE.deepseek-flashmla and the repository README.md.
#pragma once

// ---- FlashMLA kerutils common ----

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;


// ---- FlashMLA kerutils device common ----
/*
Common data types and macros that are used across the kerutils library.
*/

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/config.hpp>  // For CUTE_DEVICE

namespace kerutils {

// Cache hints
enum class CacheHint {
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST,
    EVICT_UNCHANGED,
    NO_ALLOCATE
};

// Prefetch size
enum class PrefetchSize {
    B64,
    B128,
    B256
};

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;
using nve4m3 = __nv_fp8_e4m3;
using nve4m3x2 = __nv_fp8x2_e4m3;
using nve4m3x4 = __nv_fp8x4_e4m3;

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
#define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#define KERUTILS_ENABLE_SM80
#define KERUTILS_ENABLE_SM90
#define KERUTILS_ENABLE_SM90A
#define KERUTILS_ENABLE_SM100
#define KERUTILS_ENABLE_SM100A
#endif

// ---- FlashMLA kerutils host support ----

#include <exception>
#include <string>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cutlass/cuda_host_adapter.hpp>


namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;

        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

#define KU_CUDA_CHECK(call)                                                                                  \
do {                                                                                                  \
    cudaError_t status_ = call;                                                                       \
    if (status_ != cudaSuccess) {                                                                     \
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        THROW_KU_EXCEPTION("CUDA", "CUDA error: ", cudaGetErrorString(status_));                       \
    }                                                                                                 \
} while(0)

#define KU_CUTLASS_CHECK(call) \
do {                                                                                                  \
    cutlass::Status status_ = call;                                                                   \
    if (status_ != cutlass::Status::kSuccess) {                                                      \
        fprintf(stderr, "CUTLASS error (%s:%d): %d\n", __FILE__, __LINE__, static_cast<int>(status_)); \
        THROW_KU_EXCEPTION("CUTLASS", "CUTLASS error: ", static_cast<int>(status_));                 \
    }                                                                                                 \
} while(0)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                                      \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion `%s` failed (%s:%d): ", #cond, __FILE__, __LINE__);          \
            if constexpr (sizeof(#__VA_ARGS__) > 1) {                                                \
                fprintf(stderr, ", " __VA_ARGS__);                                                        \
            }                                                                                             \
            fprintf(stderr, "\n");                                                                       \
            THROW_KU_EXCEPTION("Assertion", "Assertion `", #cond, "` failed.");                          \
        }                                                                                                 \
    } while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T>
inline __host__ __device__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __host__ __device__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

// A wrapper for make_tensor_map
static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t> &size,
    const std::vector<uint64_t> &strides,   // PAY ATTENTION: In BYTES
    const std::vector<uint32_t> &box_size,
    void* global_ptr,
    CUtensorMapDataType data_type,
    CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t> &element_strides_ = {}
) {
    int dim = size.size();
    KU_ASSERT(dim >= 1);

    std::vector<uint32_t> element_strides;
    if (element_strides_.empty()) {
        for (int i = 0; i < dim; ++i)
            element_strides.push_back(1);
    } else {
        element_strides = element_strides_;
    }
    KU_ASSERT(strides.size() == (uint32_t)dim-1 && box_size.size() == (uint32_t)dim && element_strides.size() == (uint32_t)dim);

    CUtensorMap result;
    CUresult ret_code = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &result,
        data_type,
        dim,
        global_ptr,
        size.data(),
        strides.data(),
        box_size.data(),
        element_strides.data(),
        interleave_mode,
        swizzle_mode,
        l2_promotion,
        oob_fill
    );
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto print_vector = [&](auto t, const char* fmt, const char end='\n') {
            for (auto elem : t) {
                printf(fmt, elem);
            }
            printf("%c", end);
        };
        fprintf(stderr, "Failed to create tensormap\n");
        fprintf(stderr, "Dim: %d\n", dim);
        printf("size: "); print_vector(size, "%lu ");
        printf("strides: "); print_vector(strides, "%lu ");
        printf("box_size: "); print_vector(box_size, "%u ");
        printf("element_strides: "); print_vector(element_strides, "%u ");
        printf("global ptr: 0x%lx\n", (int64_t)global_ptr);
        printf("data_type: %d\n", (int)data_type);
        printf("swizzle_mode: %d\n", (int)swizzle_mode);
        printf("l2_promotion: %d\n", (int)l2_promotion);
        printf("interleave_mode: %d\n", (int)interleave_mode);
        printf("oob_fill: %d\n", (int)oob_fill);
        KU_ASSERT(false);
    }
    return result;
}

// Given strides (in number of elements), this function converts their datatype in uint64_t and then multiplies by elem_size
template<typename T>
static inline std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides_in_elems, size_t elem_size) {
    std::vector<uint64_t> res;
    for (auto stride : strides_in_elems) {
        res.push_back(((uint64_t)stride) * elem_size);
    }
    return res;
}

}

// ---- FlashMLA TMA copy helper ----

#include <cute/tensor.hpp>


namespace kerutils {

template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    transac_bar_t &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar), 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

// In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
// You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

// In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in some rows. This function converts the local_elem_idx to the actual col_idx
// You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
    int col_idx = 8*(local_elem_idx/4) + (idx_in_warpgroup%4)*2 + (local_elem_idx&1);
    return col_idx;
}

template <bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC, bool zero_init) {
    using namespace cute;
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    tiled_mma.accumulate_ = zero_init ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_ss(bool clear_accum, TiledMma tiled_mma, Tensor0 const &sA, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sA_frag = thr_mma.partition_fragment_A(sA);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTE_DEVICE
void wgmma_rs(bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(rA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(rA_frag); ++k) {
        cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
}

}

// ---- FlashMLA SM100 intrinsics ----


namespace kerutils {

// tma gather4 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

// tma gather4 prefetch (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :
        : "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "l"(cache_hint)
    );
}

// tma gather4 with cta_group::2, allowing for synchronization across CTAs within a pair of CTAs (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
template<bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4_cta_group_2(const void* desc_ptr, transac_bar_t &mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= cute::Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory"
    );
}

// Vectorized addition for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
CUTE_DEVICE
float2 float2_add(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "add.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
    return c;
}

// Vectorized multiplication for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
CUTE_DEVICE
float2 float2_mul(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// Vectorized fused addition-multiplication for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
CUTE_DEVICE
float2 float2_fma(const float2 &a, const float2 &b, const float2 &c) {
    // return a*b+c
    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

// Vectorized negation for foat32
CUTE_DEVICE
float2 float2_neg(const float2 &a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

// st.bulk (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
CUTE_DEVICE
void st_bulk(void* dst_ptr, int64_t size) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile (
        "st.bulk.weak.shared::cta [%0], %1, 0;\n"
        :
        : "r"(dst_addr), "l"(size)
        : "memory"
    );
}

struct CUTE_ALIGNAS(16) CLCResponseObj {
    // An opaque 16B value
    char opaque[16];
};

struct CLCResult {
    int is_valid;
    int x, y, z;
};

// Issue a CLC try_cancel query (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
CUTE_DEVICE
void issue_clc_query(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

// Issue a CLC try_cancel query with .multicast::cluster::all (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
CUTE_DEVICE
void issue_clc_query_multicast_cluster_all(transac_bar_t &bar, CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr)
    );
}

// Get the result of a CLC query (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel)
// In this function, we separate get_first_ctaid::x/y/z and hope PTXAS's dead code elimination can remove unnecessary instructions
template<bool USE_LD_ACQUIRE>
CUTE_DEVICE
CLCResult get_clc_query_response(CLCResponseObj &response_obj) {
    uint32_t response_addr = cute::cast_smem_ptr_to_uint(&response_obj);
    CLCResult result;
    #define EMIT_ASM(LD_MODIFIER)                                                                   \
        asm volatile(                                                                               \
            "{\n"                                                                                   \
            ".reg .pred p1;\n\t"                                                                    \
            ".reg .b128 clc_result;\n\t"                                                            \
            "ld" LD_MODIFIER ".shared.b128 clc_result, [%4];\n\t"                                   \
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"           \
            "selp.u32 %3, 1, 0, p1;\n\t"                                                            \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_result;\n\t" \
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_result;\n\t" \
            "}\n"                                                                                   \
            : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.is_valid)                 \
            : "r"(response_addr)                                                                    \
            : "memory"                                                                              \
        );
    if constexpr (USE_LD_ACQUIRE) {
        EMIT_ASM(".acquire.cta");
    } else {
        EMIT_ASM("");
    }
    return result;
}

// LDG.256 or LDG.256 with non-coherent cache (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive dispatch based on template parameters if using function
// NC_STR should be either "" or ".nc"
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
#define KU_LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>, "`result` must be a pointer"); \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result); \
        asm volatile( \
            "ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR ".v4.u64 {%0, %1, %2, %3}, [%4];\n" \
            : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]), \
            "=l"(result_as_uint64_ptr[2]), "=l"(result_as_uint64_ptr[3]) \
            : "l"(global_addr) \
        ); \
    }

// STG.256 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
#define KU_STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR) \
    { \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>, "`global_addr` must be a pointer"); \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer"); \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src); \
        asm volatile( \
            "st.global.L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".v4.u64 [%0], {%1, %2, %3, %4};\n" \
            : \
            : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]), \
            "l"(src_as_uint64_ptr[2]), "l"(src_as_uint64_ptr[3]) \
        ); \
    }

}

namespace kerutils {

// tcgen05.commit.cta_group::1 (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

// tcgen05.commit.cta_group::1, with multicast (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_multicast_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

// tcgen05.commit.cta_group::2 (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_2x1SM_noelect(transac_bar_t &bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        :"r"(bar_intptr)
    );
}

// tcgen05.commit.cta_group::2, with multicast (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
CUTE_DEVICE
void umma_arrive_multicast_2x1SM_noelect(transac_bar_t &bar, uint16_t cta_mask) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        :"r"(bar_intptr), "h"(cta_mask)
    );
}

// tcgen05.fence::before_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}


// Load from tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumElements>
__device__ __forceinline__
void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
    // NOTE The following code crashes VSCode intellisense engine, so we disable it
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

// Load from tensor memory, 16 data path lanes, 128-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp128bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32 || kNumReplications == 64, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp128b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp128b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp128b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp128b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp128b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp128b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 64) {
            cute::SM100_TMEM_LOAD_16dp128b64x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*2>{});
#endif
}

// Load from tensor memory, 16 data path lanes, 256-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumReplications>
__device__ __forceinline__
void tmem_ld_16dp256bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumReplications == 1 || kNumReplications == 2 || kNumReplications == 4 || kNumReplications == 8 || kNumReplications == 16 || kNumReplications == 32, "Invalid kNumReplications");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumReplications == 1) {
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 2) {
            cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 4) {
            cute::SM100_TMEM_LOAD_16dp256b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 8) {
            cute::SM100_TMEM_LOAD_16dp256b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 16) {
            cute::SM100_TMEM_LOAD_16dp256b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumReplications == 32) {
            cute::SM100_TMEM_LOAD_16dp256b32x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumReplications*4>{});
#endif
}

// Store into tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
template <int kNumElements>
__device__ __forceinline__
void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

}

// ---- FlashMLA SM100 MMA helpers ----

#include <cute/tensor.hpp>


namespace kerutils {

// Perform SS UTCMMA
// sA and sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Perform TS UTCMMA
// sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tA_frag and tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16,
            UMMA::Layout_K_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_K_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_K_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_K_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_mn_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16,
            UMMA::Layout_MN_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_MN_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_MN_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_MN_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{});
}

template<cute::UMMA::Major MAJOR, int MN, int K, int SWIZZLE, typename T = bf16>
auto make_umma_canonical_layout() {
    if constexpr (MAJOR == cute::UMMA::Major::K) {
        return make_umma_canonical_k_major_layout<MN, K, SWIZZLE, T>();
    } else {
        return make_umma_canonical_mn_major_layout<MN, K, SWIZZLE, T>();
    }
}

}

// ---- FlashMLA SM100 two-CTA TMA traits ----

#include <cute/tensor.hpp>


namespace cute {

// Extensions to CuTe
// CuTe's built-in SM100_TMA_2SM_LOAD_1D series requires the number of participating threads to be 2 (using ThrID = Layout<_2>;) and also splits the data, which is really annoying to use, so we modified our own version. Additionally, to keep it consistent with other parts that use SM90 TMA, we made it accept TMA::CacheHintSm90 instead of TMA::CacheHintSm100.

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
struct SM100_TMA_2SM_LOAD_1D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_2D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_3D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy([[maybe_unused]] void const* desc_ptr, [[maybe_unused]] uint64_t* mbar_ptr, [[maybe_unused]] uint64_t cache_hint,
       [[maybe_unused]] void      * smem_ptr,
       [[maybe_unused]] int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_4D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_NOSPLIT
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    return SM100_TMA_2SM_LOAD_1D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM100_TMA_2SM_LOAD_2D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM100_TMA_2SM_LOAD_3D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM100_TMA_2SM_LOAD_4D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
  using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};
struct SM100_TMA_2SM_LOAD_NOSPLIT_OP : SM100_TMA_2SM_LOAD_NOSPLIT {};
// The non-executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
  // Don't try to execute a copy with SM100_TMA_2SM_LOAD_NOSPLIT before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};
// The executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;
  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
    : opargs_(desc, mbar, cache) {}
};

}

// ---- FlashMLA common types ----

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

// ---- FlashMLA sparse-attention parameters ----

#include "cutlass/bfloat16.h"

enum class ModelType {
    V32,
    MODEL1
};

struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Both inclusive
    int begin_block_idx, end_block_idx; // Inclusive, exclusive
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams { // TODO Change name to DenseAttnDecodeParams
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // The number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
    bool is_causal;
    float scale_softmax, scale_softmax_log2;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;
    float *__restrict__ oaccum_ptr;

    cudaStream_t stream;
};

struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    ModelType model_type;

    cutlass::bfloat16_t* __restrict__ q;   // [b, s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;  // [num_blocks, page_block_size, d_qk]
    int* __restrict__ indices;   // [b, s_q, topk]
    int* __restrict__ topk_length;  // [b], may be nullptr
    float* __restrict__ attn_sink;  // [h_q], may be nullptr

    float* __restrict__ lse;    // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;   // [b, s_q, h_q, d_v]

    int extra_num_blocks, extra_page_block_size, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;  // [extra_num_blocks, extra_page_block_size, d_qk]
    int* __restrict__ extra_indices;   // [b, s_q, extra_topk]
    int* __restrict__ extra_topk_length;  // [b], may be nullptr

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    cudaStream_t stream;

    // SplitKV-related parameters
    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;
};

struct CombineParams {
    int b, s_q, h_q, d_v;

    float* __restrict__ lse;    // [b, s_q, h_q]
    void* __restrict__ out;   // [b, s_q, h_q, d_v]
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;

    float* attn_sink;  // [h_q], may be nullptr

    cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
    int b;  // batch size
    int s_q;
    int block_size_n;
    int fixed_overhead_num_blocks;

    int topk, extra_topk;   // -1 if sparse attention (or extra topk) is disabled
    int *__restrict__ topk_length, *__restrict__ extra_topk_length;

    int *__restrict__ seqlens_k_ptr;    // Only necessary for dense attention

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int num_sm_parts;

    cudaStream_t stream;
};

struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]
    float* __restrict__ attn_sink;   // [h_q], may be nullptr
    int* __restrict__ topk_length;    // [s_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits; // [s_q, h_q]
    float* __restrict__ lse; // [s_q, h_q]

    int num_sm;
    cudaStream_t stream;

    // Masked group-packing extension (LiteTopK): when membership != nullptr,
    // head-row r attends token j iff membership[s_q_idx*topk + j] has bit
    // (r / h_per_q) set. Membership bits are only set for valid entries, so
    // this REPLACES the per-token validity mask in the masked path.
    const uint16_t* membership = nullptr;   // [s_q, topk]
    int h_per_q = 0;

    // fp8 variant only: epilogue output multiplier (V dequant scale, i.e.
    // k_scale). The bf16 kernel ignores it.
    float out_scale = 1.0f;

    // Per-QUERY causal prefix bounds on the ascending union list [s_q, G]
    // (G = h_q / h_per_q). Row r may attend list entries < bound of query
    // r / h_per_q. Replaces membership for causal-superset semantics.
    const int* topk_length_per_q = nullptr;

    // KV-split: q_group_div consecutive s_q rows (list slices) share ONE Q
    // row; outputs stay per-row and are LSE-merged by the caller.
    int q_group_div = 1;

    // Query-major membership bits [s_q, G, topk/32]: bit r of row (s, q)
    // says union entry r belongs to query q's own top-k. Exact semantics
    // like `membership`, but consumed as two direct word loads per 64-token
    // half-block (no warp13 bit transpose). Mutually exclusive with
    // `membership`.
    const uint32_t* membership_qm = nullptr;
};

// We have some kernels that implement both prefill and decode modes in a single kernel (with different template instantiations). The following enum helps to distinguish the modes.
enum class SparseAttnFwdMode {
    Prefill,            // Normal prefill mode
    DecodeWithSplitKV,  // To trigger decoding mode for kernels that support both prefill and decode
};

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = std::bool_constant<FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV>::value;

template<SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;

// ---- FlashMLA assertions and utilities ----

#include <cstdint>

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif


struct RingBufferState {
    uint32_t cur_block_idx = 0u;

    __device__ __forceinline__
    void update() {
        cur_block_idx += 1;
    }

    template<uint32_t NUM_STAGES>
    __device__ __forceinline__
    std::pair<uint32_t, bool> get() const {
        uint32_t stage_idx = cur_block_idx % NUM_STAGES;
        bool phase = (cur_block_idx / NUM_STAGES) & 1;
        return {stage_idx, phase};
    }

    __device__ __forceinline__
    RingBufferState offset_by(const int offset) const {
        // Must guarantee no underflow
        uint32_t new_block_idx = static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
        RingBufferState new_state;
        new_state.cur_block_idx = new_block_idx;
        return new_state;
    }
};

// ---- FlashMLA FP8 conversion helpers ----

#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>


namespace sm100 {

using namespace cute;

CUTE_DEVICE
int int4_max(int4 t) {
    return max(max(t.x, t.y), max(t.z, t.w));
}

CUTE_DEVICE
int int4_min(int4 t) {
    return min(min(t.x, t.y), min(t.z, t.w));
}

// Convert 2x fp8_e4m3 to 2x bf16 with scaling
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data, nv_bfloat16 scale) {
    // TODO Use native conversion for CUDA >= 13.1
    float2 data_float2 = (float2)data;
    nv_bfloat162 data_bf16x2 = __float22bfloat162_rn(data_float2);
    return nv_bfloat162 {
        data_bf16x2.x * scale,
        data_bf16x2.y * scale
    };
}

}

// ---- FlashMLA FP8 UMMA traits ----
// SM100_MMA_F8F6F4_2x1SM_SS without elect_one_sync(), mirroring the
// F16BF16 NOELECT atoms in the original FlashMLA SM100 GEMM header. The caller
// (single MMA warp, one issuing thread) already guarantees one issuer.
#include <cute/tensor.hpp>

namespace cute {

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F8F6F4_2x1SM_SS_NOELECT
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_2x1SM_SS_NOELECT without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT supports <= 8bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instruction's K extent is always 256bits -> 32 elements for 8bit
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

}  // namespace cute

// ---- LiteDSA SM100 FP8 kernel configuration ----

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>


namespace sm100::fwd::head128_fp8 {

using namespace cute;
using fp8_e4m3 = cutlass::float_e4m3_t;

// fp8 variant of the masked head128 sparse prefill kernel.
// Differences from the bf16 kernel (../head128):
//  - q/kv/P are e4m3; both GEMMs run tcgen05 kind::f8f6f4 (2x accum steps of
//    32 elems vs bf16's 16).
//  - Q shrinks to 36.9KB and lives ENTIRELY in smem -> the TMEM-Q + UTCCP
//    machinery is deleted (all-SS MMAs). TMEM only holds O and P accums.
//  - Q/K tiles keep the 64-element column geometry but use 64B swizzle
//    (SW64); V uses 128-element boxes with SW128 (D_V/2 = 256 = 2x128).
//    KV therefore needs TWO tensor maps (same data, different box/swizzle).
//  - P is stored to smem as e4m3 scaled by 448 (bake log2(448) into the
//    softmax exponent); the 448 cancels between O and li, lse subtracts
//    ln(448), and params.out_scale carries the V dequant scale (k_scale).

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;      // K gathers: box {64,1},  SWIZZLE_64B
    CUtensorMap tensor_map_kv_v;    // V gathers: box {128,1}, SWIZZLE_128B
};

template<int D_QK>
struct KernelTemplate {

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;

static constexpr int B_H = 128;    // For 2 CTAs
static constexpr int B_TOPK = 128; // For 2 CTAs
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_THREADS = 256 + 128 + 128;

// K pipeline split: part0 = 128 dims (one 128B gather box, early QK start),
// part1 = 448 (3x128B + 1x64B boxes). 128-elem SW128 boxes for the first
// 512 dims cut K's TMA issues from 9 to 5 per token-quad and double the
// DRAM request size (the 64B boxes were 10% of stall samples via
// mio_throttle).
static constexpr int D_PART0 = 128;
static constexpr int D_PART1 = D_QK - D_PART0;   // 448 = 384(SW128) + 64(SW64)
static constexpr int D_SW128 = 512;              // 128-elem box region
static constexpr int NUM_P0_TILES = D_PART0 / 64, NUM_P1_TILES = D_PART1 / 64;

// P is stored as e4m3 scaled by 7 (= 448 / 2^6): the online-softmax
// hysteresis lets logits exceed mi by up to 6, i.e. exp2 values up to 64,
// so the pre-scale must leave 2^6 headroom below e4m3's 448 max.
static constexpr float P_SCALE_LOG2 = 2.8073549220576042f;   // log2(7)
static constexpr float P_SCALE_LN = 1.9459101090932196f;     // ln(7)

// Tensor memory columns (fp32 accums only; no Q in TMEM)
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 320: P buffer 0
    // 320 ~ 384: P buffer 1 (double-buffered: QK(k+1) overlaps softmax(k))
    static constexpr int o = 0;
    static constexpr int p = 256;
    static constexpr int p1 = 320;
};

// Q/K: 64-element column tiles, 64B swizzle (same element geometry as the
// bf16 kernel's SW128 tiles, so all element-offset math carries over).
template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// K region layouts: 128-elem SW128 tiles for dims [0, 512), one SW64 tile
// for the rope tail [512, 576).
template<int NUM_TILES128>
using SmemLayoutK128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<fp8_e4m3>{},
    Shape<Int<B_TOPK/2>, Int<128*NUM_TILES128>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));
using SmemLayoutKTail = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<fp8_e4m3>{},
    Shape<Int<B_TOPK/2>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutV = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<fp8_e4m3>{},
    Shape<Int<256>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

// P/S: e4m3 [B_H/2, B_TOPK] per CTA, INTER (unswizzled) K-major atoms.
// The (8,16)-element fp8 atom is byte-identical to the bf16 (8,8) atom, so
// the u128 store pattern in the softmax warps mirrors the bf16 kernel's.
using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<fp8_e4m3>{},
    Shape<Int<B_H/2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<fp8_e4m3, cosize_v<SmemLayoutQTiles<D_Q/64>>> q_full;
            array_aligned<fp8_e4m3, cosize_v<SmemLayoutKTiles<D_K/64>>> k[2];
            array_aligned<fp8_e4m3, cosize_v<SmemLayoutV>> v[2];
        } s;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;   // epilogue only
    } u;
    array_aligned<fp8_e4m3, cosize_v<SmemLayoutS>> s[2];   // double-buffered
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    char is_kq_valid[NUM_BUFS][16][B_TOPK/8];
    transac_bar_t bar_prologue_q;
    transac_bar_t bar_qk_part_done[NUM_BUFS], bar_qk_done[NUM_BUFS];
    transac_bar_t bar_sv_part_done[NUM_BUFS], bar_sv_done[NUM_BUFS];
    transac_bar_t bar_k_part0_ready[NUM_BUFS], bar_k_part1_ready[NUM_BUFS];
    transac_bar_t bar_v_part0_ready[NUM_BUFS], bar_v_part1_ready[NUM_BUFS];
    transac_bar_t bar_p_free[NUM_BUFS];
    transac_bar_t bar_so_ready[NUM_BUFS];
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[2][128], rowwise_li_buf[128];  // max-buf parity-double-buffered: one barrier/block
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<fp8_e4m3, fp8_e4m3, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<fp8_e4m3, fp8_e4m3, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _32>{}  // CTA0 takes V[:, 0:256], CTA1 takes V[:, 256:512]; K-mode = 32 (8bit)
));

template<typename TmaParams>
static __device__ void
sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParams &tma_params);

};

}

// ---- LiteDSA SM100 FP8 sparse-attention kernel ----
#include <math_constants.h>
#include <cuda_fp8.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>



namespace sm100::fwd::head128_fp8 {

using namespace cute;

CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    const int4 lo = __ldg((const int4*)src_ptr);
    const int4 hi = __ldg((const int4*)src_ptr + 1);
    val.a0 = lo.x; val.a1 = lo.y; val.a2 = lo.z; val.a3 = lo.w;
    val.a4 = hi.x; val.a5 = hi.y; val.a6 = hi.z; val.a7 = hi.w;
    return val;
}

// Pipeline identical to the bf16 kernel (see ../head128/phase1.cuh); Q now
// lives entirely in smem so the S->T (UTCCP) stage is gone.

template<int D_QK>
template<typename TmaParams>
__device__ void
KernelTemplate<D_QK>::sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    if (params.topk_length_per_q != nullptr) {
        // group's block loop runs to the LAST query's bound (= union total)
        const int G_ = params.h_q / params.h_per_q;
        topk_length = __ldg(params.topk_length_per_q + (long)s_q_idx*G_ + (G_-1));
    }
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_v));
    }

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ_full = make_tensor(make_smem_ptr(plan.u.s.q_full.data()), SmemLayoutQTiles<D_Q/64>{});

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H/2>, Int<D_V>>{});
    tP.data().get() = tmem_cols::p;
    tO.data().get() = tmem_cols::o;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_prologue_q.init(1);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_part_done[i].init(1);
                plan.bar_qk_done[i].init(1);
                plan.bar_sv_part_done[i].init(1);
                plan.bar_sv_done[i].init(1);

                plan.bar_v_part0_ready[i].init(1);
                plan.bar_v_part1_ready[i].init(1);
                plan.bar_k_part0_ready[i].init(1);
                plan.bar_k_part1_ready[i].init(1);
                plan.bar_p_free[i].init(128*2);
                plan.bar_so_ready[i].init(128*2);
                plan.bar_k_valid_ready[i].init(16);
                plan.bar_k_valid_free[i].init(128);
            }
            fence_barrier_init();
        }
    }

    cute::cluster_sync();

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx / params.q_group_div),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ_full, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);
        }

        cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();    // Wait for TMEM allocation

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        // Scale & Exp warps

        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        const float2 scale = float2 {params.sm_scale_div_log2, params.sm_scale_div_log2};
        // e4m3 P store: thread t (and t+64) writes row t%64; its 64 values
        // form 4 u128 (16 e4m3 each) at column-block stride 64 u128s.
        // Per-query causal prefix bound (qlen mode): loaded once per row.
        int row_bound = INT_MAX;
        if (params.topk_length_per_q != nullptr) {
            const int row = cta_idx*(B_H/2) + idx_in_warpgroup%64;
            const int G_ = params.h_q / params.h_per_q;
            row_bound = __ldg(params.topk_length_per_q
                              + (long)s_q_idx*G_ + row / params.h_per_q);
        }
        // Query-major membership (exact tier, no warp13 transpose): this
        // row's two mask words per block, software-pipelined one block
        // ahead like the producers' index loads.
        const uint32_t* qm_row = nullptr;
        uint2 qm_cur, qm_nx;
        if (params.membership_qm != nullptr) {
            const int row = cta_idx*(B_H/2) + idx_in_warpgroup%64;
            const int G_ = params.h_q / params.h_per_q;
            const int capw = params.topk / 32;
            qm_row = params.membership_qm
                     + ((long)s_q_idx*G_ + row / params.h_per_q)*capw
                     + (idx_in_warpgroup >= 64 ? 2 : 0);
            qm_cur = __ldg((const uint2*)qm_row);
        }
        uint128_t* sS_base0 = (uint128_t*)plan.s[0].data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*4);
        uint128_t* sS_base1 = (uint128_t*)plan.s[1].data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*4);

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            if (qm_row != nullptr && k + 1 < num_k_blocks)
                qm_nx = __ldg((const uint2*)(qm_row + (size_t)(k+1)*(B_TOPK/32)));
            plan.bar_qk_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            ku::tcgen05_after_thread_sync();

            float2 p[(B_TOPK/2)/2];
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(
                (k % NUM_BUFS) ? tmem_cols::p1 : tmem_cols::p, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            plan.bar_p_free[k%NUM_BUFS].arrive(0u);

            // Mask
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            const char* kv_valid_base = plan.is_k_valid[k%NUM_BUFS];
            if (params.membership != nullptr) {
                const int row = cta_idx*(B_H/2) + idx_in_warpgroup%64;
                kv_valid_base = plan.is_kq_valid[k%NUM_BUFS][row / params.h_per_q];
            }
            uint32_t is_k_valid_lo = *(uint32_t*)(kv_valid_base + (idx_in_warpgroup>=64?B_TOPK/8/2:0));
            uint32_t is_k_valid_hi = *(uint32_t*)(kv_valid_base + (idx_in_warpgroup>=64?B_TOPK/8/2:0) + 4);
            if (params.topk_length_per_q != nullptr) {
                // causal prefix folded into the validity bits: two ops
                // instead of a 64-iteration compare loop
                const int abs0 = k*B_TOPK + (idx_in_warpgroup>=64 ? B_TOPK/2 : 0);
                const int n = min(max(row_bound - abs0, 0), (int)(B_TOPK/2));
                const uint64_t pm = n >= 64 ? ~0ull : ((1ull << n) - 1ull);
                is_k_valid_lo &= (uint32_t)pm;
                is_k_valid_hi &= (uint32_t)(pm >> 32);
            }
            if (qm_row != nullptr) {
                is_k_valid_lo &= qm_cur.x;
                is_k_valid_hi &= qm_cur.y;
                qm_cur = qm_nx;
            }
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                if (!(is_k_valid_lo >> i & 1))
                    p_float[i] = -CUDART_INF_F;
            }
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                if (!(is_k_valid_hi >> i & 1))
                    p_float[i+(B_TOPK/2)/2] = -CUDART_INF_F;
            }

            float cur_pi_max = -CUDART_INF_F;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2); i += 1) {
                cur_pi_max = max(cur_pi_max, p_float[i]);
            }
            cur_pi_max *= params.sm_scale_div_log2;

            plan.bar_k_valid_free[k%NUM_BUFS].arrive();

            // parity-buffered exchange: one barrier per block instead of two
            {
                float* mbuf = plan.rowwise_max_buf[k & 1];
                mbuf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, 0);
                cur_pi_max = max(cur_pi_max, mbuf[idx_in_warpgroup^64]);
            }
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

            float new_max, scale_for_old;
            if (!should_scale_o) {
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;
            li *= scale_for_old;

            // Calculate S = exp2(p*scale - new_max + log2(7)) as e4m3.
            // The pre-scale cancels between O and li at the epilogue; 7
            // leaves 2^6 headroom for the hysteresis overshoot (values can
            // reach exp2(6)*7 = 448 = e4m3 max, never clipped).
            // ex2 on f16x2 halves the SFU work; e4m3's 2^-3 rounding
            // dominates half's 2^-11, so P precision is unchanged.
            __nv_fp8x2_storage_t s8[(B_TOPK/2)/2];
            float2 neg_new_max = float2 {-new_max + P_SCALE_LOG2, -new_max + P_SCALE_LOG2};
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                float2 d = ku::float2_fma(p[i], scale, neg_new_max);
                d.x = exp2f(d.x);
                d.y = exp2f(d.y);
                li += d.x + d.y;
                s8[i] = __nv_cvt_float2_to_fp8x2(d, __NV_SATFINITE, __NV_E4M3);
            }

            // S double-buffered: wait only for the SV gemm that last
            // consumed THIS buffer (block k-2).
            if (k > 1) {
                plan.bar_sv_done[(k-2)%NUM_BUFS].wait(((k-2)/NUM_BUFS)&1);
            }
            uint128_t* sS_base = (k & 1) ? sS_base1 : sS_base0;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/16; i += 1) {
                sS_base[64*i] = *(uint128_t*)(s8 + i*8);
            }

            // Scale O (needs SV k-1 complete; only on rescale blocks)
            if (k > 0 && should_scale_o) {
                float2 scale_for_old_float2 = float2 {scale_for_old, scale_for_old};
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                ku::tcgen05_after_thread_sync();

                static constexpr int CHUNK_SIZE = 32;
                float2 o[CHUNK_SIZE/2];
                CUTE_UNROLL
                for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
                    ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::o + chunk_idx*CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                        o[i] = ku::float2_mul(o[i], scale_for_old_float2);
                    }
                    ku::tmem_st_32dp32bNx<CHUNK_SIZE>(tmem_cols::o + chunk_idx*CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_store();
                }
                ku::tcgen05_before_thread_sync();
            }

            fence_view_async_shared();
            plan.bar_so_ready[k%NUM_BUFS].arrive(0u);
        }

        // Epilogue

        if (real_mi == -CUDART_INF_F) {
            li = 0.0f;
            mi = -CUDART_INF_F;
        }

        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, 0);
        li += plan.rowwise_li_buf[idx_in_warpgroup^64];

        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + cta_idx*(B_H/2) + idx_in_warpgroup;
            // li carries the P pre-scale; remove it from the lse
            float cur_lse = logf(li) - P_SCALE_LN + mi*CUDART_LN2_F;
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // Store O. out_scale carries the V dequant factor (k_scale); the P
        // pre-scale cancels between the O accumulator and li.
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx*B_H/2 + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(params.out_scale, li + exp2f(attn_sink - mi + P_SCALE_LOG2));
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H/2>, Int<B_EPI>>{}
        )(_, _, cta_idx, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H/2>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float2 o[B_EPI/2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);
        if (!have_valid_indices) {
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        float2 output_scale_float2 = make_float2(output_scale, output_scale);

        CUTE_UNROLL
        for (int k = 0; k < (D_V/2)/B_EPI; ++k) {
            if (have_valid_indices) {
                ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::o + k*B_EPI, o);
                cutlass::arch::fence_view_async_tmem_load();
            }

            CUTE_UNROLL
            for (int i = 0; i < B_EPI/8; ++i) {
                __nv_bfloat162 o_bf16[4];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j) {
                    float2 d = ku::float2_mul(o[i*4+j], output_scale_float2);
                    o_bf16[j] = __float22bfloat162_rn(d);
                }
                int smem_row = idx_in_warpgroup % 64;
                int smem_col = (idx_in_warpgroup/64)*(D_V/2) + k*B_EPI + i*8;
                *(uint128_t*)(&sO(smem_row, smem_col)) = *(uint128_t*)(o_bf16);
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_divided(_, _, k)),
                    thr_tma.partition_D(tma_gO(_, _, k))
                );
            }
            if (warp_idx == 1 && elect_one_sync()) {
                int k2 = k + (D_V/B_EPI/2);
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_divided(_, _, k2)),
                    thr_tma.partition_D(tma_gO(_, _, k2))
                );
            }
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warps for K (fp8: 64-elem cols via the SW64 map)
        cutlass::arch::warpgroup_reg_dealloc<96>();
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/2)/4/NUM_WARPS;
        if (elect_one_sync()) {
            // index loads software-pipelined one block ahead (8.7%+4.3% of
            // stall samples were the raw __ldg latency here)
            int4 indices[NUM_LOCAL_ROWS_PER_WARP], indices_nx[NUM_LOCAL_ROWS_PER_WARP];
            CUTE_UNROLL
            for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
                indices[r] = __ldg((int4*)(gIndices + cta_idx*(B_TOPK/2)) + r*NUM_WARPS + warp_idx);
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                fp8_e4m3* sK_base = plan.u.s.k[k & 1].data() + warp_idx*4*64;
                if (k + 1 < num_k_blocks) {
                    CUTE_UNROLL
                    for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
                        indices_nx[r] = __ldg((int4*)(gIndices + (k+1)*B_TOPK + cta_idx*(B_TOPK/2)) + r*NUM_WARPS + warp_idx);
                }
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                // 128-elem SW128 boxes for dims [0,512): smem element base of
                // (token t, 128-chunk c) = c*(B_TOPK/2)*128 + t*128; the SW64
                // rope tail [512,576) lives after the SW128 region.
                fp8_e4m3* sK_tail = plan.u.s.k[k & 1].data() + (B_TOPK/2)*D_SW128 + warp_idx*4*64;
                auto load_k128 = [&](transac_bar_t &bar, int c0, int c1) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int c = c0; c < c1; ++c)
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv_v),
                                bar,
                                plan.u.s.k[k & 1].data() + c*((B_TOPK/2)*128) + (warp_idx*4 + local_row*(4*NUM_WARPS))*128,
                                c*128,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                    }
                };
                auto load_ktail = [&](transac_bar_t &bar) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row)
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            bar,
                            sK_tail + local_row*(4*NUM_WARPS)*64,
                            D_SW128,
                            indices[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                };

                int cur_buf = k%NUM_BUFS;
                // K double-buffered: only wait for QK of block k-2 (this
                // buffer's previous consumer) -> gathers run 2 blocks ahead.
                if (k > 1) {
                    plan.bar_qk_done[(k-2)%NUM_BUFS].wait(((k-2)/NUM_BUFS)&1);
                }
                if (!should_skip_tma) {
                    load_k128(plan.bar_k_part0_ready[cur_buf], 0, D_PART0/128);
                } else {
                    plan.bar_k_part0_ready[cur_buf].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_PART0*sizeof(fp8_e4m3), 1u);
                }

                if (!should_skip_tma) {
                    load_k128(plan.bar_k_part1_ready[cur_buf], D_PART0/128, D_SW128/128);
                    load_ktail(plan.bar_k_part1_ready[cur_buf]);
                } else {
                    plan.bar_k_part1_ready[cur_buf].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_PART1*sizeof(fp8_e4m3), 1u);
                }
                CUTE_UNROLL
                for (int r = 0; r < NUM_LOCAL_ROWS_PER_WARP; ++r)
                    indices[r] = indices_nx[r];
            }
        }
    } else if (warpgroup_idx == 2) {
        // Producer warps for V (fp8: 128-elem cols via the SW128 map; no
        // UTCCP wait — V no longer aliases Q)
        cutlass::arch::warpgroup_reg_dealloc<96>();
        int warp_idx = cutlass::canonical_warp_idx_sync() - 8;
        constexpr int NUM_WARPS = 4;

        if (elect_one_sync()) {
            constexpr int NROWS_V = B_TOPK/4/NUM_WARPS;
            int4 vrows[NROWS_V], vrows_nx[NROWS_V];
            CUTE_UNROLL
            for (int r = 0; r < NROWS_V; ++r)
                vrows[r] = __ldg((int4*)(gIndices) + r*NUM_WARPS + warp_idx);
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                fp8_e4m3* sV_base = plan.u.s.v[k & 1].data() + warp_idx*4*128;
                if (k + 1 < num_k_blocks) {
                    CUTE_UNROLL
                    for (int r = 0; r < NROWS_V; ++r)
                        vrows_nx[r] = __ldg((int4*)(gIndices + (k+1)*B_TOPK) + r*NUM_WARPS + warp_idx);
                }
                auto load_part_vi = [&](transac_bar_t &bar, int local_row_start, int local_row_end) {
                    CUTE_UNROLL
                    for (int local_row = local_row_start; local_row < local_row_end; ++local_row) {
                        int4 token_idxs = vrows[local_row];
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < (D_V/2)/128; ++local_col)
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv_v),
                                bar,
                                sV_base + local_row*(4*NUM_WARPS)*128 + local_col*(B_TOPK*128),
                                local_col*128 + (cta_idx?256:0),
                                token_idxs,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                    }
                };

                int cur_buf = k%NUM_BUFS;
                // V double-buffered: wait only for SV of block k-2.
                if (k > 1) {
                    plan.bar_sv_done[(k-2)%NUM_BUFS].wait(((k-2)/NUM_BUFS)&1);
                }
                load_part_vi(plan.bar_v_part0_ready[cur_buf], 0, (B_TOPK/2)/4/NUM_WARPS);
                load_part_vi(plan.bar_v_part1_ready[cur_buf], (B_TOPK/2)/4/NUM_WARPS, B_TOPK/4/NUM_WARPS);
                CUTE_UNROLL
                for (int r = 0; r < NROWS_V; ++r)
                    vrows[r] = vrows_nx[r];
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<168>();

        // MMA warp
        if (cta_idx == 0 && warp_idx == 12 && elect_one_sync()) {
            // Wait for Q (all-smem; no S->T copy)
            plan.bar_prologue_q.arrive_and_expect_tx(B_H*D_K*sizeof(fp8_e4m3));
            plan.bar_prologue_q.wait(0);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k%NUM_BUFS;
                    // Wire the declared-but-unused P double buffer: QK(k)
                    // accumulates into the buffer softmax(k-2) drained, so
                    // QK(k) overlaps softmax(k-1) instead of serializing
                    // on its TMEM drain.
                    tP.data().get() = cur_buf ? tmem_cols::p1
                                              : tmem_cols::p;
                    Tensor sQ0 = make_tensor(make_smem_ptr(plan.u.s.q_full.data()), SmemLayoutQTiles<NUM_P0_TILES>{});
                    Tensor sQ1 = make_tensor(make_smem_ptr(plan.u.s.q_full.data() + (B_H/2)*D_PART0), SmemLayoutQTiles<(D_SW128-D_PART0)/64>{});
                    Tensor sQ2 = make_tensor(make_smem_ptr(plan.u.s.q_full.data() + (B_H/2)*D_SW128), SmemLayoutQTiles<1>{});
                    Tensor sK0 = make_tensor(make_smem_ptr(plan.u.s.k[k & 1].data()), SmemLayoutK128<D_PART0/128>{});
                    Tensor sK1 = make_tensor(make_smem_ptr(plan.u.s.k[k & 1].data() + (B_TOPK/2)*D_PART0), SmemLayoutK128<(D_SW128-D_PART0)/128>{});
                    Tensor sK2 = make_tensor(make_smem_ptr(plan.u.s.k[k & 1].data() + (B_TOPK/2)*D_SW128), SmemLayoutKTail{});

                    plan.bar_k_part0_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_PART0*sizeof(fp8_e4m3));
                    plan.bar_k_part0_ready[cur_buf].wait((k/NUM_BUFS)&1);
                    if (k >= NUM_BUFS) {
                        plan.bar_p_free[cur_buf].wait(
                            ((k-NUM_BUFS)/NUM_BUFS)&1);
                    }
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_P, sQ0, sK0, tP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_part_done[cur_buf], 1|2);

                    plan.bar_k_part1_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_PART1*sizeof(fp8_e4m3));
                    plan.bar_k_part1_ready[cur_buf].wait((k/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_P, sQ1, sK1, tP, false);
                    ku::utcmma_ss(tiled_mma_P, sQ2, sK2, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[cur_buf], 1|2);
                }
                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s[(k-1) & 1].data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.s.v[(k-1) & 1].data()), SmemLayoutV{});
                    Tensor sS_divided = flat_divide(sS, Tile<Int<B_H/2>, _64>{})(_, _, _0{}, _);    // (B_H/2, 64, 2)
                    Tensor sV_divided = flat_divide(sV, Tile<Int<D_V/2>, _64>{})(_, _, _0{}, _);  // (D_V/2, 64, 2)

                    plan.bar_so_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);

                    plan.bar_v_part0_ready[cur_buf].arrive_and_expect_tx((B_TOPK/2)*D_V*sizeof(fp8_e4m3));
                    plan.bar_v_part0_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _0{}), sV_divided(_, _, _0{}), tO, k == 1);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_sv_part_done[cur_buf], 1|2);

                    plan.bar_v_part1_ready[cur_buf].arrive_and_expect_tx((B_TOPK/2)*D_V*sizeof(fp8_e4m3));
                    plan.bar_v_part1_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _1{}), sV_divided(_, _, _1{}), tO, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_sv_done[cur_buf], 1|2);
                }
            }
        } else if (warp_idx == 13) {
            // KV valid loading warp (+ per-query membership masks)
            static_assert(B_TOPK == 128);
            if (lane_idx < 16) {
                int32x8_t vind = ldg_256_indices(gIndices + lane_idx*8);
                int32x8_t vind_nx;
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int cur_buf = k%NUM_BUFS;
                    if (k + 1 < num_k_blocks)
                        vind_nx = ldg_256_indices(gIndices + (k+1)*B_TOPK + lane_idx*8);
                    const int32x8_t indices = vind;
                    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
                        int abs_pos = k*B_TOPK + lane_idx*8 + rel_pos_in_lane;
                        return index >= 0 && index < params.s_kv && abs_pos < topk_length;
                    };
                    char is_ks_valid_mask = \
                        is_valid(7, indices.a7) << 7 |
                        is_valid(6, indices.a6) << 6 |
                        is_valid(5, indices.a5) << 5 |
                        is_valid(4, indices.a4) << 4 |
                        is_valid(3, indices.a3) << 3 |
                        is_valid(2, indices.a2) << 2 |
                        is_valid(1, indices.a1) << 1 |
                        is_valid(0, indices.a0) << 0;

                    plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);
                    plan.is_k_valid[cur_buf][lane_idx] = is_ks_valid_mask;
                    if (params.membership != nullptr) {
                        const uint16_t* mrow = params.membership
                            + (long)s_q_idx * params.topk + k*B_TOPK + lane_idx*8;
                        uint4 mv = *(const uint4*)mrow;
                        const uint16_t* mw = (const uint16_t*)&mv;
                        const int G = params.h_q / params.h_per_q;
                        CUTE_UNROLL
                        for (int q = 0; q < 16; ++q) {
                            if (q >= G) break;
                            char mq = 0;
                            CUTE_UNROLL
                            for (int j = 0; j < 8; ++j)
                                mq |= ((mw[j] >> q) & 1) << j;
                            plan.is_kq_valid[cur_buf][q][lane_idx] =
                                mq & is_ks_valid_mask;
                        }
                    }
                    plan.bar_k_valid_ready[cur_buf].arrive();
                    vind = vind_nx;
                }
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    static_assert(D_QK == 576);
    using Kernel = KernelTemplate<D_QK>;
    using fp8_e4m3 = cutlass::float_e4m3_t;

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);
    KU_ASSERT(params.h_q == Kernel::B_H);
    KU_ASSERT(params.d_qk == D_QK);

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q / params.q_group_div);
    auto tma_Q = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},
        make_tensor(
            make_gmem_ptr((fp8_e4m3*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        (typename Kernel::template SmemLayoutQTiles<D_QK/64>){}
    );

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        (typename Kernel::template SmemLayoutOTiles<1>){}
    );

    // K gathers: 64-element boxes, 64B swizzle
    // The two tensor maps depend only on (kv pointer, s_kv, stride); cache
    // them per key to avoid two cuTensorMapEncodeTiled driver calls on every
    // invocation (78 layers x every chunk in production).
    struct TmapCacheEntry {
        const void* kv; uint64_t skv; uint64_t stride;
        CUtensorMap k_map, v_map;
    };
    static thread_local TmapCacheEntry tmap_cache[128];
    static thread_local bool tmap_valid[128] = {};
    const uint64_t tmap_key =
        (reinterpret_cast<uintptr_t>(params.kv) >> 8) * 1315423911u ^
        static_cast<uint64_t>(params.s_kv);
    const uint32_t tmap_slot = static_cast<uint32_t>(tmap_key & 127u);
    TmapCacheEntry& tce = tmap_cache[tmap_slot];
    const bool tmap_hit = tmap_valid[tmap_slot] &&
        tce.kv == params.kv &&
        tce.skv == static_cast<uint64_t>(params.s_kv) &&
        tce.stride == static_cast<uint64_t>(params.stride_kv_s_kv);

    CUtensorMap tensor_map_kv;
    if (tmap_hit) tensor_map_kv = tce.k_map; else {
        uint64_t size[2] = {D_QK, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(fp8_e4m3)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // V gathers: 128-element boxes, 128B swizzle
    CUtensorMap tensor_map_kv_v;
    if (tmap_hit) tensor_map_kv_v = tce.v_map; else {
        uint64_t size[2] = {D_QK, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(fp8_e4m3)};
        uint32_t box_size[2] = {128, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_v,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }
    if (!tmap_hit) {
        tce.kv = params.kv;
        tce.skv = static_cast<uint64_t>(params.s_kv);
        tce.stride = static_cast<uint64_t>(params.stride_kv_s_kv);
        tce.k_map = tensor_map_kv;
        tce.v_map = tensor_map_kv_v;
        tmap_valid[tmap_slot] = true;
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q, tma_Q,
        shape_O, tma_O,
        tensor_map_kv,
        tensor_map_kv_v
    };
    auto kernel = &sparse_attn_fwd_kernel<Kernel, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    static thread_local bool smem_attr_set = false;
    if (!smem_attr_set) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        smem_attr_set = true;
    }

    cutlass::ClusterLaunchParams launch_params = {
        dim3(2*params.s_q, 1, 1),
        dim3(Kernel::NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

}
