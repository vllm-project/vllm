#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include "kerutils/device/cuda/common.h"

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

// Load from tensor memory with column-wise reduction, 32 data path lanes, 32-bit pattern, repeated N times.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
// USE_MAX: true = .max reduction, false = .min reduction
// USE_ABS: true = .abs qualifier (use absolute values for reduction)
// USE_NAN: true = .NaN qualifier (propagate NaN in reduction)
template <int kNumElements, bool USE_MAX, bool USE_ABS, bool USE_NAN>
__device__ __forceinline__
void tmem_ld_red_32dp32bNx(uint32_t tmem_start, void* data_, float& redval) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                  kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                  kNumElements == 128,
                  "Invalid kNumElements for tcgen05.ld.red (must be power of 2, at least 2)");
#ifndef __VSCODE_IDE__
    static constexpr char s_max[] = ".max";
    static constexpr char s_min[] = ".min";
    static constexpr char s_abs[] = ".abs";
    static constexpr char s_nan[] = ".NaN";
    static constexpr char s_empty[] = "";
    // Operand numbering for each branch:
    //   outputs: %0..%{N-1} = data[0..N-1] ("=r"), %N = redval ("=f")
    //   inputs:  %{N+1} = tmem_start ("r"), %{N+2} = redop ("C"), %{N+3} = abs ("C"), %{N+4} = nan ("C")
    if constexpr (kNumElements == 2) {
        // outputs: %0..%1 = data, %2 = redval; inputs: %3 = taddr, %4 = redop, %5 = abs, %6 = nan
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x2%4%5%6.f32"
            " {%0, %1}, %2, [%3];\n"
            : "=r"(data[0]), "=r"(data[1]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 4) {
        // outputs: %0..%3 = data, %4 = redval; inputs: %5 = taddr, %6..%8 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x4%6%7%8.f32"
            " {%0, %1, %2, %3}, %4, [%5];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 8) {
        // outputs: %0..%7 = data, %8 = redval; inputs: %9 = taddr, %10..%12 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x8%10%11%12.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7}, %8, [%9];\n"
            : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
              "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 16) {
        // outputs: %0..%15 = data, %16 = redval; inputs: %17 = taddr, %18..%20 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x16%18%19%20.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15}, %16, [%17];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 32) {
        // outputs: %0..%31 = data, %32 = redval; inputs: %33 = taddr, %34..%36 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x32%34%35%36.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31}, %32, [%33];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 64) {
        // outputs: %0..%63 = data, %64 = redval; inputs: %65 = taddr, %66..%68 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x64%66%67%68.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63}, %64, [%65];\n"
            : "=r"(data[0]),  "=r"(data[1]),  "=r"(data[2]),  "=r"(data[3]),
              "=r"(data[4]),  "=r"(data[5]),  "=r"(data[6]),  "=r"(data[7]),
              "=r"(data[8]),  "=r"(data[9]),  "=r"(data[10]), "=r"(data[11]),
              "=r"(data[12]), "=r"(data[13]), "=r"(data[14]), "=r"(data[15]),
              "=r"(data[16]), "=r"(data[17]), "=r"(data[18]), "=r"(data[19]),
              "=r"(data[20]), "=r"(data[21]), "=r"(data[22]), "=r"(data[23]),
              "=r"(data[24]), "=r"(data[25]), "=r"(data[26]), "=r"(data[27]),
              "=r"(data[28]), "=r"(data[29]), "=r"(data[30]), "=r"(data[31]),
              "=r"(data[32]), "=r"(data[33]), "=r"(data[34]), "=r"(data[35]),
              "=r"(data[36]), "=r"(data[37]), "=r"(data[38]), "=r"(data[39]),
              "=r"(data[40]), "=r"(data[41]), "=r"(data[42]), "=r"(data[43]),
              "=r"(data[44]), "=r"(data[45]), "=r"(data[46]), "=r"(data[47]),
              "=r"(data[48]), "=r"(data[49]), "=r"(data[50]), "=r"(data[51]),
              "=r"(data[52]), "=r"(data[53]), "=r"(data[54]), "=r"(data[55]),
              "=r"(data[56]), "=r"(data[57]), "=r"(data[58]), "=r"(data[59]),
              "=r"(data[60]), "=r"(data[61]), "=r"(data[62]), "=r"(data[63]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    } else if constexpr (kNumElements == 128) {
        // outputs: %0..%127 = data, %128 = redval; inputs: %129 = taddr, %130..%132 = C
        asm volatile(
            "tcgen05.ld.red.sync.aligned.32x32b.x128%130%131%132.f32"
            " {%0, %1, %2, %3,"
            " %4, %5, %6, %7,"
            " %8, %9, %10, %11,"
            " %12, %13, %14, %15,"
            " %16, %17, %18, %19,"
            " %20, %21, %22, %23,"
            " %24, %25, %26, %27,"
            " %28, %29, %30, %31,"
            " %32, %33, %34, %35,"
            " %36, %37, %38, %39,"
            " %40, %41, %42, %43,"
            " %44, %45, %46, %47,"
            " %48, %49, %50, %51,"
            " %52, %53, %54, %55,"
            " %56, %57, %58, %59,"
            " %60, %61, %62, %63,"
            " %64, %65, %66, %67,"
            " %68, %69, %70, %71,"
            " %72, %73, %74, %75,"
            " %76, %77, %78, %79,"
            " %80, %81, %82, %83,"
            " %84, %85, %86, %87,"
            " %88, %89, %90, %91,"
            " %92, %93, %94, %95,"
            " %96, %97, %98, %99,"
            " %100, %101, %102, %103,"
            " %104, %105, %106, %107,"
            " %108, %109, %110, %111,"
            " %112, %113, %114, %115,"
            " %116, %117, %118, %119,"
            " %120, %121, %122, %123,"
            " %124, %125, %126, %127}, %128, [%129];\n"
            : "=r"(data[0]),   "=r"(data[1]),   "=r"(data[2]),   "=r"(data[3]),
              "=r"(data[4]),   "=r"(data[5]),   "=r"(data[6]),   "=r"(data[7]),
              "=r"(data[8]),   "=r"(data[9]),   "=r"(data[10]),  "=r"(data[11]),
              "=r"(data[12]),  "=r"(data[13]),  "=r"(data[14]),  "=r"(data[15]),
              "=r"(data[16]),  "=r"(data[17]),  "=r"(data[18]),  "=r"(data[19]),
              "=r"(data[20]),  "=r"(data[21]),  "=r"(data[22]),  "=r"(data[23]),
              "=r"(data[24]),  "=r"(data[25]),  "=r"(data[26]),  "=r"(data[27]),
              "=r"(data[28]),  "=r"(data[29]),  "=r"(data[30]),  "=r"(data[31]),
              "=r"(data[32]),  "=r"(data[33]),  "=r"(data[34]),  "=r"(data[35]),
              "=r"(data[36]),  "=r"(data[37]),  "=r"(data[38]),  "=r"(data[39]),
              "=r"(data[40]),  "=r"(data[41]),  "=r"(data[42]),  "=r"(data[43]),
              "=r"(data[44]),  "=r"(data[45]),  "=r"(data[46]),  "=r"(data[47]),
              "=r"(data[48]),  "=r"(data[49]),  "=r"(data[50]),  "=r"(data[51]),
              "=r"(data[52]),  "=r"(data[53]),  "=r"(data[54]),  "=r"(data[55]),
              "=r"(data[56]),  "=r"(data[57]),  "=r"(data[58]),  "=r"(data[59]),
              "=r"(data[60]),  "=r"(data[61]),  "=r"(data[62]),  "=r"(data[63]),
              "=r"(data[64]),  "=r"(data[65]),  "=r"(data[66]),  "=r"(data[67]),
              "=r"(data[68]),  "=r"(data[69]),  "=r"(data[70]),  "=r"(data[71]),
              "=r"(data[72]),  "=r"(data[73]),  "=r"(data[74]),  "=r"(data[75]),
              "=r"(data[76]),  "=r"(data[77]),  "=r"(data[78]),  "=r"(data[79]),
              "=r"(data[80]),  "=r"(data[81]),  "=r"(data[82]),  "=r"(data[83]),
              "=r"(data[84]),  "=r"(data[85]),  "=r"(data[86]),  "=r"(data[87]),
              "=r"(data[88]),  "=r"(data[89]),  "=r"(data[90]),  "=r"(data[91]),
              "=r"(data[92]),  "=r"(data[93]),  "=r"(data[94]),  "=r"(data[95]),
              "=r"(data[96]),  "=r"(data[97]),  "=r"(data[98]),  "=r"(data[99]),
              "=r"(data[100]), "=r"(data[101]), "=r"(data[102]), "=r"(data[103]),
              "=r"(data[104]), "=r"(data[105]), "=r"(data[106]), "=r"(data[107]),
              "=r"(data[108]), "=r"(data[109]), "=r"(data[110]), "=r"(data[111]),
              "=r"(data[112]), "=r"(data[113]), "=r"(data[114]), "=r"(data[115]),
              "=r"(data[116]), "=r"(data[117]), "=r"(data[118]), "=r"(data[119]),
              "=r"(data[120]), "=r"(data[121]), "=r"(data[122]), "=r"(data[123]),
              "=r"(data[124]), "=r"(data[125]), "=r"(data[126]), "=r"(data[127]),
              "=f"(redval)
            : "r"(tmem_start),
              "C"(USE_MAX ? s_max : s_min),
              "C"(USE_ABS ? s_abs : s_empty),
              "C"(USE_NAN ? s_nan : s_empty)
        );
    }
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

// Decompose a non-power-of-2 replication count into a sequence of power-of-2 calls.
//
// This helper splits kNumReplications into its constituent powers of 2 (from highest to lowest)
// and issues one call to tmem_ld_st_fn.template operator()<power>() for each, automatically
// advancing the tmem address by kNumElemsPerUnit columns and the data pointer by kNumElemsPerUnit
// uint32_t elements per unit.
//
// Template parameters:
//   kNumReplications — total number of replications (may be non-power-of-2)
//   kNumElemsPerUnit — number of tmem columns (and uint32_t data elements) consumed per unit
//                      (e.g. 1 for 32dp32b, 2 for 16dp128b, 4 for 16dp256b)
//   F                — a callable with a template operator(): f.template operator()<N>(uint32_t, void*)
//
// Usage example (with a template lambda):
//   tmem_ld_st_decomposed<6, 1>(tmem_start, data_ptr,
//       []<int N>(uint32_t ts, void* d) { tmem_ld_32dp32bNx<N>(ts, d); });
//   // Equivalent to:
//   //   tmem_ld_32dp32bNx<4>(tmem_start, data_ptr);
//   //   tmem_ld_32dp32bNx<2>(tmem_start + 4, (void*)((uint32_t*)data_ptr + 4));
template<int kNumReplications, int kNumElemsPerUnit, typename F>
__device__ __forceinline__
void tmem_ld_st_decomposed(uint32_t tmem_start, void* data, F&& tmem_ld_st_fn) {
    if constexpr (kNumReplications == 0) {
        // Base case: nothing to do
        return;
    } else {
        // Largest power-of-2 <= kNumReplications
        constexpr int kHighBit = 1 << (31 - __builtin_clz(kNumReplications));
        tmem_ld_st_fn.template operator()<kHighBit>(tmem_start, data);
        // Recurse on the remainder
        constexpr int kRemaining = kNumReplications - kHighBit;
        if constexpr (kRemaining > 0) {
            tmem_ld_st_decomposed<kRemaining, kNumElemsPerUnit>(
                tmem_start + kHighBit * kNumElemsPerUnit,
                (void*)((uint32_t*)data + kHighBit * kNumElemsPerUnit),
                tmem_ld_st_fn
            );
        }
    }
}

}
