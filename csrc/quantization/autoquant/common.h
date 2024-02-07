/*
 * Adapted from https://github.com/InternLM/lmdeploy
 * Copyright (c) OpenMMLab. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace vllm {
namespace autoquant {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#define VLLM_ARCH_SM75 1
#else
#define VLLM_ARCH_SM75 0
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define VLLM_ARCH_SM80 1
#else
#define VLLM_ARCH_SM80 0
#endif

constexpr int WARP_SIZE = 32;

#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define PRAGMA_UNROLL _Pragma("unroll")
#define PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define PRAGMA_UNROLL #pragma unroll
#define PRAGMA_NO_UNROLL #pragma unroll 1
#endif
#else
#define PRAGMA_UNROLL
#define PRAGMA_NO_UNROLL
#endif

static const float HALF_FLT_MAX = 65504.F;

// Modified from NVIDIA FasterTransformer:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
// Modified from llm-awq https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/dequantize.cuh
__inline__ __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions. In
    // addition, I exploit the fact that sub and fma have the same throughput in
    // order to convert elt_23 and elt_67 to fp16 without having to shift them to
    // the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2])
        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
}

__inline__ __device__ uint4 dequantize_s4_to_fp16x2_v2(uint32_t const& source)
{
    uint4 result;

    uint32_t*       h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK    = 0x000f000f;
    static constexpr uint32_t TOP_MASK    = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
    static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;

    if (0) {  // 1024 & 64
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(MAGIC_NUM_0));
        asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(MAGIC_NUM_1));
        asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(MAGIC_NUM_0));
        asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[3]) : "r"(h[3]), "r"(MAGIC_NUM_1));
    }
    else {  //  64 only, trade 4 hfma2 with 2 shifts
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[0] <<= 4;
        h[2] <<= 4;
        // we don't need to subtract the magic nums because zeros will go through the same dequant function
        // and carry the same magic constant, the magic num will be canceled out after subtracting zeros
    }

    return result;
}

__inline__ __device__ uint4 dequantize_s4_to_bf16x2_v2(uint32_t const& source)
{
    uint4 result;

    uint32_t*                h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const& source_i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = 0x000f000f;
    static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

    // We don't have enough mantissa to remove as much shift overhead as FP16, so we must loop.
    // No shift needed for first item.
    uint32_t i4s = source_i4s;
    asm ("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
    PRAGMA_UNROLL
    for (int ii = 1; ii < 4; ++ii)
    {
        i4s >>= 4;
        // (i4s & 0x000f000f) | 0x43004300
        asm("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(h[ii])
            : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
    }

    return result;
}

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr)
{
    uint32_t smem_int_ptr;

    asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_int_ptr)
        : "l"(ptr));

    return smem_int_ptr;
}

__inline__ __device__ void ldmatrix_m8n8_x4_b16(uint& d0, uint& d1, uint& d2, uint& d3, uint32_t smem_int_ptr)
{
#if VLLM_ARCH_SM75
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_int_ptr));
#else
    assert(VLLM_ARCH_SM75);
#endif
}

__inline__ __device__ void ldmatrix_m8n8_x2_b16(uint& d0, uint& d1, uint32_t smem_int_ptr)
{
#if VLLM_ARCH_SM75
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(d0), "=r"(d1) : "r"(smem_int_ptr));
#else
    assert(VLLM_ARCH_SM75);
#endif
}

__inline__ __device__ void wait_flag(int* lock, int status, int thread_id)
{
    int state = 0;
    while (__syncthreads_and(state != status)) {
        if (thread_id == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#else
            asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#endif
        }
    }

    __syncthreads();  // memory fence
}

__inline__ __device__ void release_flag(int* lock, int status, int thread_id)
{
    __syncthreads();  // memory fence

    if (thread_id == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#else
        asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
}

template <typename T>
__inline__ __device__ T apply_Q(const T& x, const T& q);

template <>
__inline__ __device__ half2 apply_Q(const half2& x, const half2& q)
{
    uint s, z;
    (half2&)z = __halves2half2(q.x, q.x);
    (half2&)s = __halves2half2(q.y, q.y);

    auto& t = (const uint&)x;
    uint  u, v;

    asm("sub.ftz.f16x2 %0, %1, %2;\n" : "=r"(u) : "r"(t), "r"(z));
    asm("mul.ftz.f16x2 %0, %1, %2;\n" : "=r"(v) : "r"(u), "r"(s));

    return (half2&)v;
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 a, const __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hsub2(a, b);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hmul2(a, b);
#endif
}

inline __device__ __nv_bfloat162 halves2bfloat162(const __nv_bfloat16 a, const __nv_bfloat16 b){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __halves2bfloat162(a, b);
#endif
}

inline __device__ float2 bfloat1622float2(const __nv_bfloat162 a){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat1622float2(a);
#endif
}

inline __device__ __nv_bfloat162 float22bfloat162_rn(const float2 a){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __float22bfloat162_rn(a);
#endif
}

template <>
__inline__ __device__ __nv_bfloat162 apply_Q(const __nv_bfloat162& x, const __nv_bfloat162& q)
{
    __nv_bfloat162 s, z;
    (__nv_bfloat162&)z = halves2bfloat162(q.x, q.x);
    (__nv_bfloat162&)s = halves2bfloat162(q.y, q.y);

    __nv_bfloat162 u, v;
    u = bf16hsub2(x, z);
    v = bf16hmul2(u, s);
    return v;

}

__device__ __forceinline__ float clamp_inf_for_half(const float input)
{
    // clamp inf values to enable fp16 training
    return input > 0.0f ? min(input, (HALF_FLT_MAX - 1000) / 2.0) : max(input, (-HALF_FLT_MAX + 1000) / 2.0);
}

template<typename T, int N>
struct Array {
    T a[N];

    __device__ __host__ constexpr T& operator[](int i) noexcept
    {
        return a[i];
    }
    __device__ __host__ constexpr const T& operator[](int i) const noexcept
    {
        return a[i];
    }
};

template<int... Ns>
struct Shape {
    static constexpr Array<int, sizeof...(Ns)> data_{Ns...};

    constexpr Shape() = default;

    Shape(std::integral_constant<int, Ns>...){};

    template<int index>
    constexpr auto get() const noexcept
    {
        return std::integral_constant<int, data_[index]>{};
    }

    constexpr auto m() const noexcept
    {
        return get<0>();
    }

    constexpr auto n() const noexcept
    {
        return get<1>();
    }

    constexpr auto k() const noexcept
    {
        return get<2>();
    }

    constexpr int c() const noexcept
    {
        return get<0>();
    }

    constexpr int s() const noexcept
    {
        return get<1>();
    }

    constexpr int count() const noexcept
    {
        return (Ns * ...);
    }
};

}  // namespace autoquant
}  // namespace vllm
