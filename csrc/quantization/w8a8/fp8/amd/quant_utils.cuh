#pragma once
#include <hip/hip_fp8.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

#include "../../../../attention/attention_dtypes.h"
#include <torch/headeronly/core/ScalarType.h>

namespace vllm {
#ifdef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8

// Use hardware cvt instruction for fp8 on rocm
template <typename fp8_type>
__device__ __forceinline__ fp8_type cvt_c10(float const r) {
  return {};
}

// __hip_fp8_e4m3 only exists starting in ROCm 6.3. The macro
// HIP_FP8_TYPE_OCP comes from the hip_fp8.h header and also makes
// its first appearance in ROCm 6.3. Since VLLM_DISPATCH_FP8_TYPES
// on ROCm instantiates both OCP and FNUZ kernels, we need to replace
// the new HW cvt with something reasonable that doesn't rely on the
// ROCm 6.3 feature. This allows compiling on ROCm 6.2 or newer.
template <>
__device__ __forceinline__ c10::Float8_e4m3fn cvt_c10(float const r) {
    #if HIP_FP8_TYPE_OCP
  return c10::Float8_e4m3fn(
      __hip_cvt_float_to_fp8(r, __hip_fp8_e4m3::__default_saturation,
                             __hip_fp8_e4m3::__default_interpret),
      c10::Float8_e4m3fn::from_bits());
    #else
  // Cast implemented by pytorch. Uses bit manipulation instead of HW cvt.
  // HW cvt above is faster when it is available (ROCm 6.3 or newer).
  return static_cast<c10::Float8_e4m3fn>(r);
    #endif
}

template <>
__device__ __forceinline__ c10::Float8_e4m3fnuz cvt_c10(float const r) {
  return c10::Float8_e4m3fnuz(
      __hip_cvt_float_to_fp8(r, __hip_fp8_e4m3_fnuz::__default_saturation,
                             __hip_fp8_e4m3_fnuz::__default_interpret),
      c10::Float8_e4m3fnuz::from_bits());
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  return x;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale) {
  return x;
}

    #if HIP_FP8_TYPE_OCP
using fp8_type = __hip_fp8_e4m3;
using fp8x2_type = __hip_fp8x2_e4m3;
    #else
using fp8_type = __hip_fp8_e4m3_fnuz;
using fp8x2_type = __hip_fp8x2_e4m3_fnuz;
    #endif

// fp8x2 -> float2
template <>
__inline__ __device__ float2
vec_conversion<float2, uint16_t>(const uint16_t& a) {
  fp8x2_type f8x2;
  f8x2.__x = a;
  return static_cast<float2>(f8x2);
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

// fp8x4 -> float4
template <>
__inline__ __device__ float4
vec_conversion<float4, uint32_t>(const uint32_t& a) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains

   Convention of the scale in API, e.g: FP8_data = Quantization(
   High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8 Dequant(FP8) *
   scale =>  HP

 */

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a, float scale) {
  fp8_type f8;
  f8.__x = a;
  return __float2bfloat16(static_cast<float>(f8) * scale);
}

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
  fp8_type f8;
  f8.__x = a;
  return static_cast<float>(f8) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2
scaled_vec_conversion<float2, uint16_t>(const uint16_t& a, float scale) {
  fp8x2_type f8x2;
  f8x2.__x = a;
  return static_cast<float2>(f8x2) * scale;
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

// fp8 -> half
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint8_t>(const uint8_t& a, float scale) {
  __half_raw res;
  res.data = scaled_vec_conversion<float, uint8_t>(a, scale);
  return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t& a, float scale) {
  union {
    __half2_raw h2r;
    uint32_t ui32;
  } tmp;
  tmp.h2r = __hip_cvt_fp8x2_to_halfraw2(a, fp8_type::__default_interpret);
  tmp.h2r.x.data *= scale;
  tmp.h2r.y.data *= scale;
  return tmp.ui32;
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

// half -> fp8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, uint16_t>(const uint16_t& a, float scale) {
  __half_raw tmp;
  tmp.x = a;
  tmp.data /= scale;
  return __hip_cvt_halfraw_to_fp8(tmp, fp8_type::__default_saturation,
                                  fp8_type::__default_interpret);
}

// halfx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint32_t>(const uint32_t& a, float scale) {
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;
  tmp.h2r.x.data /= scale;
  tmp.h2r.y.data /= scale;
  return __hip_cvt_halfraw2_to_fp8x2(tmp.h2r, fp8_type::__default_saturation,
                                     fp8_type::__default_interpret);
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
  return __hip_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}

// bf16x2 -> fp8x2
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
  return __hip_cvt_float_to_fp8(a / scale, fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}

// floatx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, float2>(const float2& a, float scale) {
  return __hip_cvt_float2_to_fp8x2(a / scale, fp8_type::__default_saturation,
                                   fp8_type::__default_interpret);
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
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale);
  }
  #endif
  assert(false);
  return {};  // Squash missing return statement warning
}

  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    vllm::Fp8KVCacheDataType KV_CACHE_DTYPE =                                  \
        vllm::get_fp8_kv_cache_data_type(KV_DTYPE);                            \
    if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {                   \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);             \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);        \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else {                                                                   \
      STD_TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE); \
    }

}  // namespace fp8
#endif  // USE_ROCM
}  // namespace vllm
