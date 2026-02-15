#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

// Conditional compilation for FP4 element packing size
#if (defined(NVFP4_ENABLE_ELTS16) && (CUDART_VERSION >= 12090) && \
     defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
  #define ELTS_PER_THREAD 16
constexpr int CVT_FP4_ELTS_PER_THREAD = 16;
constexpr bool CVT_FP4_PACK16 = true;
#else
  #define ELTS_PER_THREAD 8
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr bool CVT_FP4_PACK16 = false;
#endif

constexpr int CVT_FP4_SF_VEC_SIZE = 16;

namespace vllm {

// Convert PyTorch cpp type to CUDA type
template <typename T>
struct CUDATypeConverter {
  using Type = T;
};

template <>
struct CUDATypeConverter<at::Half> {
  using Type = half;
};

template <>
struct CUDATypeConverter<at::BFloat16> {
  using Type = __nv_bfloat16;
};

// Get type2 from type or vice versa (half <-> half2, bfloat16 <-> bfloat162)
template <typename T>
struct TypeConverter {
  using Type = half2;
};

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

#if (defined(NVFP4_ENABLE_ELTS16) && (CUDART_VERSION >= 12090) && \
     defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100)
// Define a 32 bytes packed data type.
template <class Type>
struct alignas(32) PackedVec {
  typename TypeConverter<Type>::Type elts[8];
};
#else
// Define a 16 bytes packed data type.
template <class Type>
struct alignas(16) PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};
#endif

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

}  // namespace vllm
