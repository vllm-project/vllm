#pragma once

// NVFP4 quantization utilities extracted from TRT-LLM quantization_utils.cuh
// Ported from tlrmchlsmth/flashinfer:nvfp4-silu-mul-quant-opt
//
// Self-contained: no TRT-LLM common/ dependencies.
// Requires SM 100 (__CUDA_ARCH__ >= 1000) for PTX e2m1 conversion.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace nvfp4 {

// Vectorized memory access helpers
struct alignas(32) PackedU32x8 {
  uint32_t d[8];
};

struct alignas(16) PackedU32x4 {
  uint32_t d[4];
};

// Packed vector type for BF16/FP16 — pairs of elements for CUDA intrinsics
template <class Type, int NUM_ELTS = 8>
struct PackedVec;

template <>
struct PackedVec<__nv_bfloat16, 8> {
  __nv_bfloat162 elts[4];
};

template <>
struct PackedVec<__nv_bfloat16, 16> {
  __nv_bfloat162 elts[8];
};

template <>
struct PackedVec<half, 8> {
  half2 elts[4];
};

template <>
struct PackedVec<half, 16> {
  half2 elts[8];
};

template <typename VecT>
__device__ __forceinline__ void loadPackedVec(VecT& val, VecT const* ptr) {
  static_assert(sizeof(VecT) == 16 || sizeof(VecT) == 32,
                "Packed vector loads expect 16-byte or 32-byte vectors.");
  using VecT_ =
      std::conditional_t<sizeof(VecT) == 16, PackedU32x4, PackedU32x8>;
  VecT_& val_ = reinterpret_cast<VecT_&>(val);
  val_ = *reinterpret_cast<VecT_ const*>(ptr);
}

// Fast reciprocal via PTX
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// SiLU activation
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

// Convert 4 float2 values (8 floats) into 8 e2m1 values (packed uint32_t)
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Convert 8 float2 values (16 floats) into 16 e2m1 values (packed uint64_t)
inline __device__ uint64_t fp32_vec_to_e2m1(float2 (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint64_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      ".reg .b8 byte4;\n"
      ".reg .b8 byte5;\n"
      ".reg .b8 byte6;\n"
      ".reg .b8 byte7;\n"
      ".reg .b32 val0;\n"
      ".reg .b32 val1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
      "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
      "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
      "mov.b64 %0, {val0, val1};\n"
      "}"
      : "=l"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return val;
#else
  return 0;
#endif
}

// Scale factor offset in 128x4 swizzled layout
__device__ __forceinline__ uint8_t* get_sf_out_offset(int rowIdx, int colIdx,
                                                      int numCols,
                                                      uint32_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int SF_VEC_SIZE = 16;
  constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      SF_VEC_SIZE / 16;  // 1 for 16 elts/thread
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF != 0) return nullptr;

  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  int32_t mTileIdx = mIdx / (32 * 4);
  int factor = SF_VEC_SIZE * 4;
  int32_t numKTiles = (numCols + factor - 1) / factor;
  int64_t mTileStride = numKTiles * 32 * 4 * 4;

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * 4 * 4;

  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * 4;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4;

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride +
                     outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
#endif
  return nullptr;
}

// Fused silu+mul+quantize to FP4: BF16/FP16 gate+up → e2m1 output
// Two-pass to avoid register spilling:
//   Pass 1: compute silu(gate)*up and find max
//   Pass 2: recompute, scale, and convert to e2m1
template <class Type, int SF_VEC_SIZE, int CVT_ELTS_PER_THREAD, bool UE8M0_SF>
__device__ std::conditional_t<CVT_ELTS_PER_THREAD == 16, uint64_t, uint32_t>
cvt_silu_mul_fp16_to_fp4(PackedVec<Type, CVT_ELTS_PER_THREAD>& gate_vec,
                         PackedVec<Type, CVT_ELTS_PER_THREAD> const& up_vec,
                         float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_ELTS_PER_THREAD == 8 || CVT_ELTS_PER_THREAD == 16);
  using ReturnType =
      std::conditional_t<CVT_ELTS_PER_THREAD == 16, uint64_t, uint32_t>;

  // Pass 1: find max
  float localMax = 0.0f;
  #pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    float2 g, u;
    if constexpr (std::is_same_v<Type, half>) {
      g = __half22float2(gate_vec.elts[i]);
      u = __half22float2(up_vec.elts[i]);
    } else {
      g = __bfloat1622float2(gate_vec.elts[i]);
      u = __bfloat1622float2(up_vec.elts[i]);
    }
    localMax =
        fmaxf(localMax, fmaxf(fabsf(silu(g.x) * u.x), fabsf(silu(g.y) * u.y)));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
  if constexpr (CVT_NUM_THREADS_PER_SF >= 2) {
    localMax = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  float vecMax = localMax;

  // Compute scale factor
  uint8_t fp8SFVal;
  float outputScale;
  auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  __nv_fp8_e4m3 tmp;
  tmp.__x = __nv_cvt_float_to_fp8(SFValue >= 0 ? SFValue : 0.0f, __NV_SATFINITE,
                                  __NV_E4M3);
  fp8SFVal = tmp.__x;
  float SFValueF = __half2float(__nv_cvt_fp8_to_halfraw(tmp.__x, __NV_E4M3));
  outputScale = vecMax != 0
                    ? reciprocal_approximate_ftz(
                          SFValueF * reciprocal_approximate_ftz(SFScaleVal))
                    : 0.0f;

  if (SFout) {
    *SFout = fp8SFVal;
  }

  // Pass 2: recompute silu(gate)*up, scale, convert to e2m1
  constexpr int NUM_CHUNKS = CVT_ELTS_PER_THREAD / 8;
  ReturnType e2m1Vec = 0;
  #pragma unroll
  for (int c = 0; c < NUM_CHUNKS; c++) {
    float2 chunk[4];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 g, u;
      if constexpr (std::is_same_v<Type, half>) {
        g = __half22float2(gate_vec.elts[c * 4 + i]);
        u = __half22float2(up_vec.elts[c * 4 + i]);
      } else {
        g = __bfloat1622float2(gate_vec.elts[c * 4 + i]);
        u = __bfloat1622float2(up_vec.elts[c * 4 + i]);
      }
      chunk[i].x = silu(g.x) * u.x * outputScale;
      chunk[i].y = silu(g.y) * u.y * outputScale;
    }
    uint32_t bits = fp32_vec_to_e2m1(chunk);
    if constexpr (CVT_ELTS_PER_THREAD == 16) {
      e2m1Vec |= static_cast<uint64_t>(bits) << (c * 32);
    } else {
      e2m1Vec = bits;
    }
  }
  return e2m1Vec;
#else
  return 0;
#endif
}

}  // namespace nvfp4
