#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace topk_select_common {

__device__ __forceinline__
uint32_t bf16x2_to_u32(const nv_bfloat162 &v) {
    static_assert(sizeof(nv_bfloat162) == sizeof(uint32_t));
    return *reinterpret_cast<const uint32_t *>(&v);
}

__device__ __forceinline__
nv_bfloat162 u32_to_bf16x2(uint32_t u) {
    static_assert(sizeof(nv_bfloat162) == sizeof(uint32_t));
    nv_bfloat162 v;
    *reinterpret_cast<uint32_t *>(&v) = u;
    return v;
}

// Distort and un-distort: Map IEEE 754 floating-point total order to unsigned integer order.
// IEEE 754 bit representation (sign-magnitude with biased exponent and mantissa) does not
// preserve value order when interpreted as unsigned integer, and distort() fixes this:
//   - For positive values (sign bit = 0): flip only the sign bit, mapping all positives to [0x80..., 0xFF...]
//   - For negative values (sign bit = 1): flip all bits (~), inverting the magnitude order so that
//     more negative (smaller value) => smaller uint
// After distortion, all negatives (0x00... ~ 0x7F...) < all positives (0x80... ~ 0xFF...) in
// unsigned integer comparison, which matches the floating-point total order.
//
// un_distort() is the exact inverse: flips the sign bit back for positives, flips all bits back for negatives.
template<typename UIntValueT>
__device__ __forceinline__
UIntValueT distort(const UIntValueT &x) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        UIntValueT mask = (x&0x8000) ? 0xFFFF : 0x8000;
        return x ^ mask;
    } else {
        // mask = (x >> 31) | 0x80000000 is 0x80000000 for positives and 0xFFFFFFFF for negatives
        uint32_t t, d;
        asm ("shr.s32 %0, %2, 31;\n"
             "lop3.b32 %1, %2, %0, 0x80000000, 0x1e;\n"
             : "=r"(t), "=r"(d)
             : "r"((uint32_t)x));
        return (UIntValueT)d;
    }
}

// A SIMD-like version of `distort`, which uses more efficient instructions for calculation
template<typename UIntValueT>
__device__ __forceinline__
void distort_x2(UIntValueT result[2], const UIntValueT input[2]) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        asm volatile (
            "{"
            ".reg .u32 mask;"
            "prmt.b32 mask, %1, 0, 0xbb99;" // mask[0:16] will be 0xFF if cur_value_packed[0]'s MSB is 1, 0x00 otherwise
            "or.b32 mask, mask, 0x80008000;"
            "xor.b32 %0, mask, %1;"
            "}"
            : "=r"(*(uint32_t*)(result))
            : "r"(*(uint32_t*)(input))
        );
    } else {
        result[0] = distort(input[0]);
        result[1] = distort(input[1]);
    }
}

template<typename UIntValueT>
__device__ __forceinline__
UIntValueT un_distort(const UIntValueT &x) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        UIntValueT mask = (x&0x8000) ? 0x8000 : 0xFFFF;
        return x ^ mask;
    } else {
        // mask = ~(x >> 31) | 0x80000000 is 0xFFFFFFFF for positives and 0x80000000 for negatives
        uint32_t t, d;
        asm ("shr.s32 %0, %2, 31;\n"
             "lop3.b32 %1, %2, %0, 0x80000000, 0x4b;\n"
             : "=r"(t), "=r"(d)
             : "r"((uint32_t)x));
        return (UIntValueT)d;
    }
}

} // namespace topk_select_common
