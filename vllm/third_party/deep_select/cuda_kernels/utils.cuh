#pragma once

#include <cstdint>

template<typename T>
__device__ __forceinline__ T warp_level_inclusive_prefix_sum(T x, uint32_t lane_idx) {
    static_assert(sizeof(T) == 4);
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        asm volatile (
            "{\n"
            ".reg .pred p;\n"
            ".reg .b32 t;\n"
            "shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff;\n"
            "@p add.u32 %0, %0, t;\n"
            "}\n"
            : "+r"(x)
            : "r"(i)
        );
    }
    return x;
}

template<typename T>
__device__ __forceinline__ T warp_level_exclusive_prefix_sum(T x, uint32_t lane_idx) {
    T inclusive_prefix_sum = warp_level_inclusive_prefix_sum(x, lane_idx);
    return inclusive_prefix_sum - x;
}

template<typename T>
__device__ __forceinline__ T warp_level_inclusive_suffix_sum(T x, uint32_t lane_idx) {
    static_assert(sizeof(T) == 4);
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        asm volatile (
            "{\n"
            ".reg .pred p;\n"
            ".reg .b32 t;\n"
            "shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff;\n"
            "@p add.u32 %0, %0, t;\n"
            "}\n"
            : "+r"(x)
            : "r"(i)
        );
    }
    return x;
}


template<typename T>
__device__ __forceinline__ T warp_level_exclusive_suffix_sum(T x, uint32_t lane_idx) {
    T inclusive_suffix_sum = warp_level_inclusive_suffix_sum(x, lane_idx);
    return inclusive_suffix_sum - x;
}
