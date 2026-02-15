/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <cuda_fp8.h>

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

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

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

template <typename Int>
__host__ __device__ inline Int round_up(Int x, Int y) {
  static_assert(std::is_integral_v<Int>,
                "round_up argument must be integral type");
  return ((x + y - 1) / y) * y;
}

template <typename Int>
__host__ __device__ __forceinline__ Int div_round_up(Int x, Int y) {
  return (x + y - 1) / y;
}

// Compute effective rows for grid configuration with swizzled SF layouts.
inline int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
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
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
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
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

struct u32x2 {
  uint32_t lo, hi;
};

using fp4_packed_t = std::conditional_t<CVT_FP4_PACK16, u32x2, uint32_t>;

__device__ __forceinline__ u32x2 fp32_vec16_to_e2m1(float2 (&array)[8]) {
  u32x2 out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      ".reg .b8 b4;\n"
      ".reg .b8 b5;\n"
      ".reg .b8 b6;\n"
      ".reg .b8 b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b0,  %3,  %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b1,  %5,  %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b2,  %7,  %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b3,  %9,  %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32   b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}\n"
      : "=r"(out.lo), "=r"(out.hi)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y),
        "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
        "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
  return out;
}

__device__ __forceinline__ uint32_t pack_fp4(float2 (&v)[4]) {
  return fp32_vec8_to_e2m1(v);
}

__device__ __forceinline__ u32x2 pack_fp4(float2 (&v)[8]) {
  return fp32_vec16_to_e2m1(v);
}

// Fast reciprocal.
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

template <class Type>
__device__ __forceinline__ void ld128_or_zero_cg_u32(PackedVec<Type>& out,
                                                     const void* ptr,
                                                     bool pred) {
  uint32_t r0, r1, r2, r3;

  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %4, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  @pr ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"((int)pred), "l"(ptr));

  *reinterpret_cast<uint4*>(&out) = uint4{r0, r1, r2, r3};
}

template <class Type>
__device__ __forceinline__ void ld256_or_zero_cg_u32(PackedVec<Type>& out,
                                                     const void* ptr,
                                                     bool pred) {
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7;

  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %8, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  mov.u32 %4, 0;\n"
      "  mov.u32 %5, 0;\n"
      "  mov.u32 %6, 0;\n"
      "  mov.u32 %7, 0;\n"
      "  @pr ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6),
        "=r"(r7)
      : "r"((int)pred), "l"(ptr));

  reinterpret_cast<uint4*>(&out)[0] = uint4{r0, r1, r2, r3};
  reinterpret_cast<uint4*>(&out)[1] = uint4{r4, r5, r6, r7};
}

// Compute SF output offset for swizzled tensor core layout.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
// Caller must precompute: numKTiles = (numCols + 63) / 64
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int32_t numKTiles, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF != 0) {
    return nullptr;
  }

  // SF vector index (16 elements share one SF in the K dimension).
  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  // Decompose indices using bitwise ops (all divisors are powers of 2).
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  int32_t mTileIdx = mIdx >> 7;         // mIdx / 128
  int32_t outerMIdx = mIdx & 31;        // mIdx % 32
  int32_t innerMIdx = (mIdx >> 5) & 3;  // (mIdx / 32) % 4
  int32_t kTileIdx = kIdx >> 2;         // kIdx / 4
  int32_t innerKIdx = kIdx & 3;         // kIdx % 4

  // Compute global SF offset: mTileIdx * (numKTiles * 512) + kTileIdx * 512 +
  //                           outerMIdx * 16 + innerMIdx * 4 + innerKIdx
  // Use bitwise OR for non-overlapping lower bits.
  int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                         << 9 |
                     (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}

template <class SFType>
__device__ __forceinline__ uint8_t* sf_out_rowmajor_u8(int row, int pack,
                                                       int packs_per_row_sf,
                                                       SFType* SFout) {
  constexpr int PACK = CVT_FP4_ELTS_PER_THREAD;
  constexpr int THREADS_PER_SF =
      CVT_FP4_SF_VEC_SIZE / PACK;  // 1 if PACK=16, 2 else PACK=8

  if (threadIdx.x % THREADS_PER_SF != 0) return nullptr;

  int sf_col =
      pack / THREADS_PER_SF;  // PACK=16 => sf_col=pack; PACK=8 => sf_col=pack/2
  int64_t off = (int64_t)row * packs_per_row_sf + sf_col;

  return (uint8_t*)SFout + off;
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, int CVT_FP4_NUM_THREADS_PER_SF, bool UE8M0_SF = false>
__device__ __forceinline__ fp4_packed_t
cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

  // Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    // Extract the 8 exponent bits from float32.
    // float 32bits = 1 sign bit + 8 exponent bits + 23 mantissa bits.
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // Convert back to fp32.
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // Convert back to fp32.
    SFValue = float(tmp);
  }

  // Write the SF to global memory (STG.8).
  if (SFout) *SFout = fp8SFVal;

  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  float outputScale =
      SFValue != 0.0f ? reciprocal_approximate_ftz(
                            SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  return pack_fp4(fp2Vals);
}

// silu in float32
__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + __expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

template <class Type>
__inline__ __device__ PackedVec<Type> compute_silu_mul(
    const PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  PackedVec<Type> result;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    // silu_mul in float32
    if constexpr (std::is_same_v<Type, half>) {
      float2 silu_vec = silu2(__half22float2(x_vec.elts[i]));
      result.elts[i] = __float22half2_rn(
          __fmul2_rn(silu_vec, __half22float2(y_vec.elts[i])));
    } else {
      float2 silu_vec = silu2(__bfloat1622float2(x_vec.elts[i]));
      result.elts[i] = __float22bfloat162_rn(
          __fmul2_rn(silu_vec, __bfloat1622float2(y_vec.elts[i])));
    }
  }
  return result;
}

}  // namespace vllm
