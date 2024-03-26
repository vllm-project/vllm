// Adated from FasterTransformer, https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"

namespace vllm {
namespace int8 {
// float32 to int8
inline __device__ int8_t quant(float a, const float scale, const float zp)
{
    int8_t int8;
    int8 = round(max(-128.f, min(127.f, (a - zp) / scale)));
    return int8;
}

// float32x2 to int8x2
inline __device__ short quant(float2 a, const float scale, const float zp)
{
    union {
        int8_t int8[2];
        short  int16;
    };

    int8[0] = quant(a.x, scale, zp);
    int8[1] = quant(a.y, scale, zp);
    return int16;
}

// float32x4 to int8x4
inline __device__ int32_t quant(float4 a, const float scale, const float zp)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };

    int8[0] = quant(a.x, scale, zp);
    int8[1] = quant(a.y, scale, zp);
    int8[2] = quant(a.z, scale, zp);
    int8[3] = quant(a.w, scale, zp);
    return int32;
}

// float16 to int8
inline __device__ int8_t quant(uint16_t a, const float scale, const float zp)
{
    int8_t int8;
    float  b = half_to_float(a);
    int8     = quant(b, scale, zp);
    return int8;
}

// float16x2 to int8x2
inline __device__ int16_t quant(uint32_t a, const float scale, const float zp)
{
    union {
        int8_t int8[2];
        short  int16;
    };
    float2 b = half2_to_float2(a);

    int8[0] = quant(b.x, scale, zp);
    int8[1] = quant(b.y, scale, zp);
    return int16;
}

// float16x4 to int8x4
inline __device__ int32_t quant(uint2 a, const float scale, const float zp)
{
    union {
        int16_t int16[2];
        int32_t int32;
    };

    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    return int32;
}

// float16x8 to int8x8
inline __device__ int64_t quant(uint4 a, const float scale, const float zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };

    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    int16[2] = quant(a.z, scale, zp);
    int16[3] = quant(a.w, scale, zp);
    return int64;
}

// bf16 to int8
inline __device__ int8_t quant(__nv_bfloat16 a, const float scale, const float zp)
{
    int8_t int8;
    float  b = to_float(a);
    int8     = quant(b, scale, zp);
    return int8;
}

//bf16x2 to int8x2
inline __device__ int16_t quant(__nv_bfloat162 a, const float scale, const float zp)
{
    union {
        int8_t int8[2];
        short  int16;
    };
    float2 b = bf1622float2(a);

    int8[0] = quant(b.x, scale, zp);
    int8[1] = quant(b.y, scale, zp);
    return int16;
}

// bf16x4 to int8x4
inline __device__ int32_t quant(bf16_4_t a, const float scale, const float zp)
{
    union {
        int16_t int16[2];
        int32_t int32;
    };
    
    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    return int32;
}

// bf16x8 to int8x8
inline __device__ int64_t quant(bf16_8_t a, const float scale, const float zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };

    int16[0] = quant(a.x, scale, zp);
    int16[1] = quant(a.y, scale, zp);
    int16[2] = quant(a.z, scale, zp);
    int16[3] = quant(a.w, scale, zp);
    return int64;
}

// int8 to float32, then `vec_conversion` to target format
inline __device__ float dequant(int8_t a, const float scale, const float zp)
{
    float b = a * scale + zp;
    return b;
}

// int8x2 to float32x2
inline __device__ float2 dequant(int16_t a, const float scale, const float zp)
{
    union {
        int8_t  int8[2];
        int16_t int16;
    };
    int16 = a;

    float2 b;
    b.x = int8[0] * scale + zp;
    b.y = int8[1] * scale + zp;
    return b;
}

// int8x4 to float32x4
inline __device__ Float4_ dequant(int32_t a, const float scale, const float zp)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };
    int32 = a;

    Float4_ b;
    b.x.x = (int8[0] * scale) + zp;
    b.x.y = (int8[1] * scale) + zp;
    b.y.x = (int8[2] * scale) + zp;
    b.y.y = (int8[3] * scale) + zp;
    return b;
}

// int8x8 to float32x8
inline __device__ Float8_ dequant(int64_t a, const float scale, const float zp)
{
    union {
        int16_t int16[4];
        int64_t int64;
    };
    int64 = a;

    Float8_ b;
    b.x = dequant(int16[0], scale, zp);
    b.y = dequant(int16[1], scale, zp);
    b.z = dequant(int16[2], scale, zp);
    b.w = dequant(int16[3], scale, zp);
    return b;
}

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
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
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, float>(const float &a) {
   __nv_bfloat16 b;
   from_float(b, a);
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

} // namespace int8
} // namespace vllm
