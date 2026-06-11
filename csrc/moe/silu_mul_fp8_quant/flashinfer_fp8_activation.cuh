#pragma once

// Flashinfer FP8 activation kernels for DeepSeek MoE
// Ported from tlrmchlsmth/flashinfer:fp8-dynamic-stride-cudagraph
//
// FP8 in -> dequant -> SiLU+Mul -> requant -> FP8 out
// with per-128-element block scale factors

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace flashinfer {

constexpr int FI_ELTS_PER_SCALE_BLOCK = 128;
constexpr int FI_ELTS_PER_THREAD = 4;

using fp8_t = __nv_fp8_e4m3;

__device__ __forceinline__ float fp8_to_float(fp8_t v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3));
}

__device__ __forceinline__ float float_to_fp8_val(float v) {
  fp8_t r;
  r.__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return r.__x;
}

__device__ __forceinline__ fp8_t float_to_fp8(float v) {
  fp8_t r;
  r.__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return r;
}

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_tanh(float x) {
  return x * (0.5f + 0.5f * __tanhf(x * 0.5f));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

struct ActivationDeepSeekParams {
  fp8_t const* inPtr;
  fp8_t* outPtr;
  float* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t innerDim;
  int32_t const* totalNumPaddedTokens;
};

template <int WarpsPerCta, bool UseTanhSilu = false>
__global__ void __launch_bounds__(WarpsPerCta * 32)
    activationDeepSeekKernelVec(ActivationDeepSeekParams params) {
  float constexpr E4m3MaxVal{448.f};
  int const totalPadded = params.totalNumPaddedTokens[0];
  int const sfStride = totalPadded;
  int const halfDim = params.innerDim / 2;
  int const numOutputScaleBlocks = halfDim / FI_ELTS_PER_SCALE_BLOCK;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const scaleBlock = blockIdx.x * WarpsPerCta + warpId;

  if (scaleBlock >= numOutputScaleBlocks) return;

  int const elemBase =
      scaleBlock * FI_ELTS_PER_SCALE_BLOCK + laneId * FI_ELTS_PER_THREAD;

  int64_t const scale1Base = (int64_t)sfStride * scaleBlock;
  int64_t const scale2Base =
      (int64_t)sfStride * (scaleBlock + numOutputScaleBlocks);

  for (int permutedRow = blockIdx.y; permutedRow < totalPadded;
       permutedRow += gridDim.y) {
    float scale1, scale2;
    if (laneId == 0) {
      scale1 = params.inDqSfsPtr[permutedRow + scale1Base];
      scale2 = params.inDqSfsPtr[permutedRow + scale2Base];
    }
    scale1 = __shfl_sync(0xffffffff, scale1, 0);
    scale2 = __shfl_sync(0xffffffff, scale2, 0);

    int64_t const x1Offset = (int64_t)permutedRow * params.innerDim + elemBase;

    uint32_t packed_x1 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x1Offset]);
    uint32_t packed_x2 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x1Offset + halfDim]);

    fp8_t x1_vals[4], x2_vals[4];
    memcpy(x1_vals, &packed_x1, 4);
    memcpy(x2_vals, &packed_x2, 4);

    float localMax = 0.0f;
    float results[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float f1 = scale1 * fp8_to_float(x1_vals[i]);
      float f2 = scale2 * fp8_to_float(x2_vals[i]);
      if constexpr (UseTanhSilu) {
        results[i] = silu_tanh(f2) * f1;
      } else {
        results[i] = silu_f(f2) * f1;
      }
      localMax = fmaxf(localMax, fabsf(results[i]));
    }

    float aMax = warp_reduce_max(localMax);

    float scaleOut;
    if (laneId == 0) {
      scaleOut = fmaxf(aMax / E4m3MaxVal, FLT_MIN);
      params.outDqSfsPtr[permutedRow + scale1Base] = scaleOut;
    }
    scaleOut = __shfl_sync(0xffffffff, scaleOut, 0);

    float invScale = 1.0f / scaleOut;
    fp8_t out_vals[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      out_vals[i] = float_to_fp8(results[i] * invScale);
    }
    uint32_t packed_out;
    memcpy(&packed_out, out_vals, 4);

    int64_t const outOffset = (int64_t)permutedRow * halfDim + elemBase;
    *reinterpret_cast<uint32_t*>(&params.outPtr[outOffset]) = packed_out;
  }
}

inline void launch_fp8_silu_mul_baseline(
    void const* input, float* input_scales, void* output, float* output_scales,
    int32_t const* total_padded_tokens, int32_t inner_dim,
    int32_t max_padded_count, bool use_tanh_silu, cudaStream_t stream) {
  ActivationDeepSeekParams params;
  params.inPtr = static_cast<fp8_t const*>(input);
  params.outPtr = static_cast<fp8_t*>(output);
  params.inDqSfsPtr = input_scales;
  params.outDqSfsPtr = output_scales;
  params.innerDim = inner_dim;
  params.totalNumPaddedTokens = total_padded_tokens;

  constexpr int WarpsPerCta = 4;
  int const halfDim = inner_dim / 2;
  int const numScaleBlocks = halfDim / FI_ELTS_PER_SCALE_BLOCK;
  int const gridSizeX = (numScaleBlocks + WarpsPerCta - 1) / WarpsPerCta;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);
  int gridSizeY = min(numSms, max(1, max_padded_count));

  dim3 grid(gridSizeX, gridSizeY, 1);
  if (use_tanh_silu) {
    activationDeepSeekKernelVec<WarpsPerCta, true>
        <<<grid, WarpsPerCta * 32, 0, stream>>>(params);
  } else {
    activationDeepSeekKernelVec<WarpsPerCta, false>
        <<<grid, WarpsPerCta * 32, 0, stream>>>(params);
  }
}

}  // namespace flashinfer
