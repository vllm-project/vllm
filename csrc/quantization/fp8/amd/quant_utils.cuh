#pragma once
#include "hip_float8.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

#include "../../../attention/attention_dtypes.h"

namespace vllm {
#ifdef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  return x;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale) {
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t
vec_conversion<uint16_t, uint8_t>(const uint8_t& a) {
  hip_fp8 f8{a, hip_fp8::from_bits()};
  __half_raw res;
  res.data = static_cast<float>(f8);
  return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t
vec_conversion<uint32_t, uint16_t>(const uint16_t& a) {
    #if defined(__HIP__MI300__) && \
        defined(__HIP_FP8_EXPERIMENTAL_BULK_CONVERT__)
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
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a) {
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
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a) {
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
__inline__ __device__ __nv_bfloat16
vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a) {
  hip_fp8 f8{a, hip_fp8::from_bits()};
  float f{f8};
  return __float2bfloat16(f);
}

using __nv_bfloat162 = __hip_bfloat162;

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a) {
  __nv_bfloat162 res;
  res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a);
  res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U));
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t
vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a) {
  bf16_4_t res;
  res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a);
  res.y = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U));
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(const uint2& a) {
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
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a) {
  hip_fp8 fp8{a, hip_fp8::from_bits()};
  return static_cast<float>(fp8);
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2
vec_conversion<float2, uint16_t>(const uint16_t& a) {
    #if defined(__HIP__MI300__) && \
        defined(__HIP_FP8_EXPERIMENTAL_BULK_CONVERT__)
  float2 res;
  const auto& f2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, 0);
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
__inline__ __device__ Float4_
vec_conversion<Float4_, uint32_t>(const uint32_t& a) {
  Float4_ res;
  res.x = vec_conversion<float2, uint16_t>((uint16_t)a);
  res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U));
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a) {
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
__inline__ __device__ uint8_t
vec_conversion<uint8_t, uint16_t>(const uint16_t& a) {
  __half_raw tmp;
  tmp.x = a;

  hip_fp8 f8{static_cast<float>(tmp.data)};
  return f8.data;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t
vec_conversion<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a) {
  hip_fp8 res{__bfloat162float(a)};
  return res.data;
}

// float -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(const float& a) {
  hip_fp8 f8(a);
  return f8.data;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4
vec_conversion<float4, uint32_t>(const uint32_t& a) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

// float2 -> half2
template <>
__inline__ __device__ uint32_t
vec_conversion<uint32_t, float2>(const float2& a) {
  union {
    half2 float16;
    uint32_t uint32;
  };

  float16 = __float22half2_rn(a);
  return uint32;
}

// Float4 -> half2x2
template <>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a) {
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

// Float4 -> float4
template <>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a) {
  float4 b;
  b.x = a.x.x;
  b.y = a.x.y;
  b.z = a.y.x;
  b.w = a.y.y;
  return b;
}

// Float8 -> half2x4
template <>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a) {
  uint4 b;
  b.x = vec_conversion<uint32_t, float2>(a.x);
  b.y = vec_conversion<uint32_t, float2>(a.y);
  b.z = vec_conversion<uint32_t, float2>(a.z);
  b.w = vec_conversion<uint32_t, float2>(a.w);
  return b;
}

// float2 -> bfloat162
template <>
__inline__ __device__ __nv_bfloat162
vec_conversion<__nv_bfloat162, float2>(const float2& a) {
  __nv_bfloat162 b = __float22bfloat162_rn(a);
  return b;
}

// Float4 -> bfloat162x2
template <>
__inline__ __device__ bf16_4_t
vec_conversion<bf16_4_t, Float4_>(const Float4_& a) {
  bf16_4_t b;
  b.x = __float22bfloat162_rn(a.x);
  b.y = __float22bfloat162_rn(a.y);
  return b;
}

// Float8 -> bfloat162x4
template <>
__inline__ __device__ bf16_8_t
vec_conversion<bf16_8_t, Float8_>(const Float8_& a) {
  bf16_8_t b;
  b.x = __float22bfloat162_rn(a.x);
  b.y = __float22bfloat162_rn(a.y);
  b.z = __float22bfloat162_rn(a.z);
  b.w = __float22bfloat162_rn(a.w);
  return b;
}

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains

   Convention of the scale in API, e.g: FP8_data = Quantization(
   High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8 Dequant(FP8) *
   scale =>  HP

 */

// fp8 -> half
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint8_t>(const uint8_t& a, float scale) {
  hip_fp8 f8{a, hip_fp8::from_bits()};
  __half_raw res;
  res.data = static_cast<float>(f8) * scale;
  return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t& a, float scale) {
    #if defined(__HIP__MI300__)
  const auto& f2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, 0);
  union {
    __half2_raw h2r;
    uint32_t ui32;
  } tmp;
  tmp.h2r.x.data = f2[0] * scale;
  tmp.h2r.y.data = f2[1] * scale;
  return tmp.ui32;
    #else
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;

  tmp.u16[0] =
      scaled_vec_conversion<uint16_t, uint8_t>(static_cast<uint8_t>(a), scale);
  tmp.u16[1] = scaled_vec_conversion<uint16_t, uint8_t>(
      static_cast<uint8_t>(a >> 8U), scale);
  return tmp.u32;
    #endif
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, uint32_t>(const uint32_t& a, float scale) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale);
  tmp.u32[1] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), scale);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2& a,
                                                                float scale) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale);
  return tmp.u64x2;
}

using __nv_bfloat16 = __hip_bfloat16;

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a, float scale) {
  hip_fp8 f8{a, hip_fp8::from_bits()};
  float f{f8};
  return __float2bfloat16(f * scale);
}

using __nv_bfloat162 = __hip_bfloat162;

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a,
                                                float scale) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale);
  res.y =
      scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), scale);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t
scaled_vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a, float scale) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t
scaled_vec_conversion<bf16_8_t, uint2>(const uint2& a, float scale) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, float scale) {
  hip_fp8 fp8{a, hip_fp8::from_bits()};
  return static_cast<float>(fp8) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2
scaled_vec_conversion<float2, uint16_t>(const uint16_t& a, float scale) {
    #if defined(__HIP__MI300__)
  float2 res;
  const auto& f2 = __builtin_amdgcn_cvt_pk_f32_fp8(a, 0);
  res.x = f2[0] * scale;
  res.y = f2[1] * scale;
  return res;
    #else
  float2 res;
  res.x = scaled_vec_conversion<float, uint8_t>(static_cast<uint8_t>(a), scale);
  res.y = scaled_vec_conversion<float, uint8_t>(static_cast<uint8_t>(a >> 8U),
                                                scale);
  return res;
    #endif
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_
scaled_vec_conversion<Float4_, uint32_t>(const uint32_t& a, const float scale) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale);
  return res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4
scaled_vec_conversion<float4, uint32_t>(const uint32_t& a, float scale) {
  Float4_ res = scaled_vec_conversion<Float4_, uint32_t>(a, scale);
  return {res.x.x, res.x.y, res.y.x, res.y.y};
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_
scaled_vec_conversion<Float8_, uint2>(const uint2& a, float scale) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, uint16_t>(const uint16_t& a, float scale) {
  __half_raw tmp;
  tmp.x = a;

  hip_fp8 f8{static_cast<float>(tmp.data) / scale};
  return f8.data;
}

// halfx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint32_t>(const uint32_t& a, float scale) {
    #ifdef __HIP__MI300__
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;

  union {
    uint32_t ui32;
    float f;
  } f1, f2;
  f1.f = tmp.h2r.x.data / scale;
  f2.f = tmp.h2r.y.data / scale;
  if ((f1.ui32 & 0x7F800000) != 0x7F800000) {
    f1.f = __builtin_amdgcn_fmed3f(f1.f, 224.0, -224.0);
  }
  if ((f2.ui32 & 0x7F800000) != 0x7F800000) {
    f2.f = __builtin_amdgcn_fmed3f(f2.f, 224.0, -224.0);
  }
  return __builtin_amdgcn_cvt_pk_fp8_f32(f1.f, f2.f, 0, 0);
    #else
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;

  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } res;
  res.ui8[0] = scaled_vec_conversion<uint8_t, uint16_t>(tmp.h2r.x.x, scale);
  res.ui8[1] = scaled_vec_conversion<uint8_t, uint16_t>(tmp.h2r.y.x, scale);
  return res.ui16;
    #endif
}

// half2x2 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint2>(const uint2& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, uint32_t>(a.x, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, uint32_t>(a.y, scale);
  return tmp.ui32;
}

// half2x4 -> fp8x8
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint4>(const uint4& a,
                                                                float scale) {
  union {
    uint2 ui2[2];
    uint4 ui4;
  } tmp;
  tmp.ui4 = a;
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[0], scale);
  res.y = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[1], scale);
  return res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, float scale) {
  hip_fp8 res{__bfloat162float(a) / scale};
  return res.data;
}

// bf16x2 -> fp8x2
// TODO(HaiShaw): Add packed convert (performance A.I.)
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, __nv_bfloat162>(
    const __nv_bfloat162& a, float scale) {
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.x, scale);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.y, scale);
  return tmp.ui16;
}

// bf16x4 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, bf16_4_t>(const bf16_4_t& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.x, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.y, scale);
  return tmp.ui32;
}

// bf16x8 -> fp8x8
template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, bf16_8_t>(const bf16_8_t& a, float scale) {
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, bf16_4_t>({a.x, a.y}, scale);
  res.y = scaled_vec_conversion<uint32_t, bf16_4_t>({a.z, a.w}, scale);
  return res;
}

// float -> fp8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, float>(const float& a, float scale) {
  hip_fp8 f8(a / scale);
  return f8.data;
}

// floatx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, float2>(const float2& a, float scale) {
    #ifdef __HIP__MI300__
  union {
    uint32_t ui32;
    float f;
  } f1, f2;
  f1.f = a.x / scale;
  f2.f = a.y / scale;
  if ((f1.ui32 & 0x7F800000) != 0x7F800000) {
    f1.f = __builtin_amdgcn_fmed3f(f1.f, 224.0, -224.0);
  }
  if ((f2.ui32 & 0x7F800000) != 0x7F800000) {
    f2.f = __builtin_amdgcn_fmed3f(f2.f, 224.0, -224.0);
  }
  return __builtin_amdgcn_cvt_pk_fp8_f32(f1.f, f2.f, 0, 0);
    #else
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, float>(a.x, scale);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, float>(a.y, scale);
  return tmp.ui16;
    #endif
}

// floatx4 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, float4>(const float4& a, float scale) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, float2>({a.x, a.y}, scale);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, float2>({a.z, a.w}, scale);
  return tmp.ui32;
}
  #endif  // ENABLE_FP8

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return vec_conversion<Tout, Tin>(x);
  }
  #endif
  assert(false);
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale);
  }
  #endif
  assert(false);
}

  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
      }                                                                        \
    }

}  // namespace fp8
#endif  // USE_ROCM
}  // namespace vllm
