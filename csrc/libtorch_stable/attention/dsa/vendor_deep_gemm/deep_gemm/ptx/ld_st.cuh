#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

namespace deep_gemm::ptx {

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
CUTLASS_HOST_DEVICE longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

/// LD/ST matrix
// TODO: remove `struct`
struct SM90_U32x2_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

struct SM90_U32x4_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_T {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)),
                        "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

template <typename dtype_t>
struct SM100_U8x4_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src = *reinterpret_cast<uint32_t*>(&src_0);
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src));
    }
};

template <typename dtype_t>
struct SM100_U8x8_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

/// Shared memory
CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

CUTLASS_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

CUTLASS_DEVICE void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

CUTLASS_DEVICE void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

CUTLASS_DEVICE void st_shared_bulk(void* smem_ptr, const uint32_t& num_bytes) {
    // `size` must be 64-bit before PTX ISA 9.0
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;" ::
                 "l"(__cvta_generic_to_shared(smem_ptr)), "l"(static_cast<uint64_t>(num_bytes)));
}

/// Global memory
CUTLASS_DEVICE uint64_t ld_volatile(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE void st_relaxed_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("st.L1::no_allocate.relaxed.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

/// Atomics
CUTLASS_DEVICE uint64_t atomic_add(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

CUTLASS_DEVICE void red_add(const int* ptr, const int& value) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_or_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.sys.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_or_rel_gpu(uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_gpu(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

/// Predicated loads
CUTLASS_DEVICE longlong4_t ld_gez_pred(const longlong4_t* ptr, const int& pred) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, 0;\n\t"
        "  @p ld.global.L2::256B.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(pred)
        : "memory");
    return ret;
}

/// Prefetch
CUTLASS_DEVICE void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

} // namespace deep_gemm::ptx
