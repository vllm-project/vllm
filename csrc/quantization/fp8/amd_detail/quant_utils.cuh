#pragma once
#include "hip_float8.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

#include "../../../attention/dtype_float32.cuh"
#include "../../../attention/dtype_bfloat16.cuh"

namespace vllm
{

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(const uint8_t& a)
{
    hip_fp8 f8{a, hip_fp8::from_bits()};
    __half_raw res;
    res.data = static_cast<float>(f8);
    return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(const uint16_t& a)
{
#if defined(__HIP__MI300__) && defined(__HIP_FP8_EXPERIMENTAL_BULK_CONVERT__)
    const auto& f2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, 0);
    union {
        __half2_raw h2r;
        uint32_t ui32;
    } tmp;
    tmp.h2r.x.data = f2[0];
    tmp.h2r.y.data = f2[1];
    return tmp.ui32;
#else
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;

    tmp.u16[0] = vec_conversion<uint16_t, uint8_t>(static_cast<uint8_t>(a));
    tmp.u16[1] = vec_conversion<uint16_t, uint8_t>(static_cast<uint8_t>(a >> 8U));
    return tmp.u32;
#endif
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a)
{
    union {
        uint2 u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
    return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a)
{
    union {
        uint4 u64x2;
        uint2 u64[2];
    } tmp;
    tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x);
    tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y);
    return tmp.u64x2;
}

using __nv_bfloat16 = __hip_bfloat16;

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a)
{
    hip_fp8 f8{a, hip_fp8::from_bits()};
    float f{f8};
    return __float2bfloat16(f);
}

using __nv_bfloat162 = __hip_bfloat162;

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a)
{
    __nv_bfloat162 res;
    res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a);
    res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U));
    return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a)
{
    bf16_4_t res;
    res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a);
    res.y = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(const uint2& a)
{
    bf16_4_t tmp1, tmp2;
    tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x);
    tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y);
    bf16_8_t res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// fp8 -> float
template <>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a)
{
    hip_fp8 fp8{a, hip_fp8::from_bits()};
    return static_cast<float>(fp8);
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 vec_conversion<float2, uint16_t>(const uint16_t& a)
{
#if defined(__HIP__MI300__) && defined(__HIP_FP8_EXPERIMENTAL_BULK_CONVERT__)
    float2 res;
    const auto& f2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, 0);
    //res.x = vec_conversion<float, uint8_t>(static_cast<uint8_t>(a));
    //res.y = vec_conversion<float, uint8_t>(static_cast<uint8_t>(a >> 8U));
    res.x = f2[0];
    res.y = f2[1];
    return res;
#else
    float2 res;
    res.x = vec_conversion<float, uint8_t>(static_cast<uint8_t>(a));
    res.y = vec_conversion<float, uint8_t>(static_cast<uint8_t>(a >> 8U));
    return res;
#endif
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(const uint32_t& a)
{
    Float4_ res;
    res.x = vec_conversion<float2, uint16_t>((uint16_t)a);
    res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a)
{
    Float4_ tmp1, tmp2;
    tmp1 = vec_conversion<Float4_, uint32_t>(a.x);
    tmp2 = vec_conversion<Float4_, uint32_t>(a.y);
    Float8_ res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(const uint16_t& a)
{
    __half_raw tmp;
    tmp.x = a;

    hip_fp8 f8{static_cast<float>(tmp.data)};
    return f8.data;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a)
{
    hip_fp8 res{__bfloat162float(a)};
    return res.data;
}

// float -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(const float& a)
{
    hip_fp8 f8(a);
    return f8.data;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(const uint32_t& a)
{
    Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
    float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
    return res;
}

template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2 float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(a);
    return uint32;
}

template <>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2 b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y = vec_conversion<uint32_t, float2>(val);
    return b;
}

template <>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)
{
    float4 b;
    b.x = a.x.x;
    b.y = a.x.y;
    b.z = a.y.x;
    b.w = a.y.y;
    return b;
}

template <>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)
{
    uint4 b;
    b.x = vec_conversion<uint32_t, float2>(a.x);
    b.y = vec_conversion<uint32_t, float2>(a.y);
    b.z = vec_conversion<uint32_t, float2>(a.z);
    b.w = vec_conversion<uint32_t, float2>(a.w);
    return b;
}

template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(const float2& a)
{
    __nv_bfloat162 b = __float22bfloat162_rn(a);
    return b;
}

template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_& a)
{
    bf16_4_t b;
    b.x = __float22bfloat162_rn(a.x);
    b.y = __float22bfloat162_rn(a.y);
    return b;
}

template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_& a)
{
    bf16_8_t b;
    b.x = __float22bfloat162_rn(a.x);
    b.y = __float22bfloat162_rn(a.y);
    b.z = __float22bfloat162_rn(a.z);
    b.w = __float22bfloat162_rn(a.w);
    return b;
}
} // namespace vllm
