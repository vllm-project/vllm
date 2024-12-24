// Adated from FasterTransformer, https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"
#include <stdio.h>

namespace vllm {
namespace int8 {

// KV-CACHE int8
static inline __device__ float int8_to_float(uint8_t x, const float scale, const float zero_point) {
  int8_t a = x - 128;
  float res = a * scale + zero_point;
  // printf("\n dequant scale= %f, zero_point= %f \n", scale, zero_point);
  // if(abs(res+1.268555)<=0.01)
  //   printf("\nI am here int8_to_float, x = %d, a= %d, res=%f, scale=%f, zero_point=%f \n",
  //           x, a, res, scale, zero_point);
  return res;
}

static inline __device__ uint8_t float_to_int8(float x, const float scale, const float zero_point) {
  int8_t fx = roundf(max(-128.f, min(127.f, (x-zero_point) / scale)));
  uint8_t res = fx + 128;
  // printf("\n quant scale= %f \n", scale);
  // if(abs(x+1.268555)<=0.00001)
  //   printf("\nI am here float_to_int8, x = %f, fx= %d, res=%d, scale=%f, zero_point=%f, (x-zero_point) / scale)=%f \n",
  //           x, fx, res, scale, zero_point, (x-zero_point) / scale);
  return res;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion_int8(const Tin& x,
                                                      const float scale, const float zero_point) {
  return x;
}

// int8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion_int8<uint16_t, uint8_t>(
    const uint8_t& a, const float scale, const float zero_point) {
  float res = int8_to_float(a, scale, zero_point);
  return float_to_half(res);
}

// int8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion_int8<uint32_t, uint16_t>(
    const uint16_t& a, const float scale, const float zero_point) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } res;
  res.u16[0] = scaled_vec_conversion_int8<uint16_t, uint8_t>((uint8_t)a, scale, zero_point);
  res.u16[1] =
      scaled_vec_conversion_int8<uint16_t, uint8_t>((uint8_t)(a >> 8U), scale, zero_point);

  // union {
  //     uint8_t  int8[2];
  //     uint16_t int16;
  // } tmp;
  // tmp.int16 = a;
  // res.u16[0] = float_to_half(int8_to_float(tmp.int8[0], scale, zero_point));
  // res.u16[1] = float_to_half(int8_to_float(tmp.int8[0], scale, zero_point));
  return res.u32;
}

// int8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion_int8<uint2, uint32_t>(
    const uint32_t& a, const float scale, const float zero_point) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion_int8<uint32_t, uint16_t>((uint16_t)a, scale, zero_point);
  tmp.u32[1] = scaled_vec_conversion_int8<uint32_t, uint16_t>(
      (uint16_t)(a >> 16U), scale, zero_point);
  return tmp.u32x2;
}

// int8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion_int8<uint4, uint2>(const uint2& a, const float scale, const float zero_point) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion_int8<uint2, uint32_t>(a.x, scale, zero_point);
  tmp.u64[1] = scaled_vec_conversion_int8<uint2, uint32_t>(a.y, scale, zero_point);
  return tmp.u64x2;
}

// int8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>(const uint8_t& a,
                                                   const float scale, const float zero_point) {
  // Note there is no direct convert function from int8 to bf16.
  float res = int8_to_float(a, scale, zero_point);
  return __float2bfloat16(res);
}

// int8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>(const uint16_t& a,
                                                     const float scale, const float zero_point) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>((uint8_t)a, scale, zero_point);
  res.y = scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                             scale, zero_point);
  return res;
}

// int8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion_int8<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale, const float zero_point) {
  bf16_4_t res;
  res.x =
      scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>((uint16_t)a, scale, zero_point);
  res.y = scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>(
      (uint16_t)(a >> 16U), scale, zero_point);
  return res;
}

// int8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t
scaled_vec_conversion_int8<bf16_8_t, uint2>(const uint2& a, const float scale, const float zero_point) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion_int8<bf16_4_t, uint32_t>(a.x, scale, zero_point);
  tmp2 = scaled_vec_conversion_int8<bf16_4_t, uint32_t>(a.y, scale, zero_point);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// int8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion_int8<float, uint8_t>(
    const uint8_t& a, const float scale, const float zero_point) {
  float res = int8_to_float(a, scale, zero_point);
  return res;
}

// int8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion_int8<float2, uint16_t>(
    const uint16_t& a, const float scale, const float zero_point) {
  // int8x2 -> half2
  uint32_t tmp = scaled_vec_conversion_int8<uint32_t, uint16_t>(a, scale, zero_point);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// int8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion_int8<Float4_, uint32_t>(
    const uint32_t& a, const float scale, const float zero_point) {
  Float4_ res;
  res.x = scaled_vec_conversion_int8<float2, uint16_t>((uint16_t)a, scale, zero_point);
  res.y =
      scaled_vec_conversion_int8<float2, uint16_t>((uint16_t)(a >> 16U), scale, zero_point);
  return res;
}

// int8x8 -> float8
template <>
__inline__ __device__ Float8_
scaled_vec_conversion_int8<Float8_, uint2>(const uint2& a, const float scale, const float zero_point) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion_int8<Float4_, uint32_t>(a.x, scale, zero_point);
  tmp2 = scaled_vec_conversion_int8<Float4_, uint32_t>(a.y, scale, zero_point);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> int8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion_int8<uint8_t, uint16_t>(
    const uint16_t& a, const float scale, const float zero_point) {
  uint8_t res = float_to_int8(half_to_float(a), scale, zero_point);
  // int8_t u8data = static_cast<uint8_t>(round(half_to_float(a)*255));
  // if(a==48403)
  //   printf("\nI am here scaled_vec_conversion half fp8, a = %d, half_to_float(a) = %f,  res= %d, a'=%f, a-a' = %f \n",
  //           a, half_to_float(a), (uint8_t)res, scaled_vec_conversion_int8<float, uint8_t>(res, scale, zero_point), (half_to_float(a)-scaled_vec_conversion_int8<float, uint8_t>(res, scale, zero_point)));
  return (uint8_t)res;
}

// bf16 -> int8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion_int8<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a,
                                                   const float scale, const float zero_point) {
  uint8_t res = float_to_int8(__bfloat162float(a), scale, zero_point);
  return (uint8_t)res;
}

// float -> int8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion_int8<uint8_t, float>(const float& a, const float scale, const float zero_point) {
  uint8_t res = float_to_int8(a, scale, zero_point);
  return (uint8_t)res;
}

// int8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion_int8<float4, uint32_t>(
    const uint32_t& a, const float scale, const float zero_point) {
  Float4_ tmp = scaled_vec_conversion_int8<Float4_, uint32_t>(a, scale, zero_point);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

} // namespace int8
} // namespace vllm
