#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"
#include "../../attention/dtype_float32.cuh"
#include "../../attention/dtype_float16.cuh"
#include "../../attention/dtype_bfloat16.cuh"


namespace vllm {
#ifdef ENABLE_FP8_E5M2
namespace fp8_e5m2_unscaled {

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

// fp8 -> half
template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(const uint8_t& a)
{
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
    return res.x;
}

// fp8x2 -> half2
template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(const uint16_t& a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E5M2);
    tmp.u16[0] = res.x;
    tmp.u16[1] = res.y;
    return tmp.u32;
}

// fp8x4 -> half2x2
template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a)
{
    union {
        uint2    u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
    return tmp.u32x2;
}

// fp8x8 -> half2x4
template<>
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

// fp8 -> __nv_bfloat16
template<>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a)
{
    // Note there is no direct convert function from fp8 to bf16.
    // fp8 -> half
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
    // half -> float -> bf16
    float tmp = half_to_float(res.x);
    return __float2bfloat16(tmp);
}

// fp8x2 -> __nv_bfloat162
template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a)
{
    __nv_bfloat162 res;
    res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a);
    res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U));
    return res;
}

// fp8x4 -> bf16_4_t
template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a)
{
    bf16_4_t res;
    res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a);
    res.y = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> bf16_8_t
template<>
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
template<>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a)
{
    // fp8 -> half
    uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a);
    // half -> float
    return half_to_float(tmp);
}

// fp8x2 -> float2
template<>
__inline__ __device__ float2 vec_conversion<float2, uint16_t>(const uint16_t& a)
{
    // fp8x2 -> half2
    uint32_t tmp = vec_conversion<uint32_t, uint16_t>(a);
    // half2 -> float2
    return half2_to_float2(tmp);
}

// fp8x4 -> float4
template<>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(const uint32_t& a)
{
    Float4_ res;
    res.x = vec_conversion<float2, uint16_t>((uint16_t)a);
    res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> float8
template<>
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
template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(const uint16_t& a)
{
    __half_raw tmp;
    tmp.x = a;
    __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
}

// bf16 -> fp8
template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw(a), __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
#endif
}

// float -> fp8
template<>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(const float& a)
{
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E5M2);
    return (uint8_t)res;
}

// fp8x4 -> float4
template<>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(const uint32_t& a)
{
    Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
    float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
    return res;
}


template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2    float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(a);
    return uint32;
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2  b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x   = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y   = vec_conversion<uint32_t, float2>(val);

    return b;
}

template<>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)
{
    float4 b;
    b.x = a.x.x;
    b.y = a.x.y;
    b.z = a.y.x;
    b.w = a.y.y;
    return b;
}

template<>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)
{
    uint4 b;
    b.x = vec_conversion<uint32_t, float2>(a.x);
    b.y = vec_conversion<uint32_t, float2>(a.y);
    b.z = vec_conversion<uint32_t, float2>(a.z);
    b.w = vec_conversion<uint32_t, float2>(a.w);
    return b;
}

template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(const float2 &a) {
    __nv_bfloat162 b;
    from_float(b, a);
    return b;
}

template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_ &a) {
    bf16_4_t b;
    from_float(b, a);
    return b;
}

template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_ &a) {
    bf16_8_t b;
    from_float(b, a);
    return b;
}

} // namespace fp8_e5m2_unscaled
#endif // ENABLE_FP8_E5M2
} // namespace vllm
