// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace beam {
using T = __nv_bfloat16;
constexpr int Packs = BEAM_MOE_PACKS_PER_THREAD;
union Vec {
  float4 packed;
  T elements[8];
};

__device__ float blockSum(float value) {
  __shared__ float partial[32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, offset));
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < blockDim.x / 32 ? partial[lane] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
      value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, offset));
    if (lane == 0) partial[0] = value;
  }
  __syncthreads();
  return partial[0];
}

__device__ Vec finalizePack(T const* experts, float const* weights, int token,
                            int column, int hidden, int topK,
                            float routedScale) {
  int64_t inputOffset = int64_t(token) * topK * hidden + column;
  Vec value;
  value.packed = *reinterpret_cast<float4 const*>(experts + inputOffset);
  float weight = weights[int64_t(token) * topK];
  float sum[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    sum[i] = __fmul_rn(__bfloat162float(value.elements[i]), weight);
  for (int k = 1; k < topK; ++k) {
    value.packed =
        *reinterpret_cast<float4 const*>(experts + inputOffset + k * hidden);
    weight = weights[int64_t(token) * topK + k];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum[i] = __fmaf_rn(__bfloat162float(value.elements[i]), weight, sum[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    T reduced = __float2bfloat16_rn(sum[i]);
    value.elements[i] =
        __float2bfloat16_rn(__fmul_rn(__bfloat162float(reduced), routedScale));
  }
  return value;
}

__device__ float invRms(Vec const (&values)[Packs], int hidden, float eps) {
  float squareSum = 0.f;
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float x = __bfloat162float(values[pack].elements[i]);
      squareSum = __fadd_rn(squareSum, __fmul_rn(x, x));
    }
  return rsqrtf(blockSum(squareSum) / hidden + eps);
}

__device__ Vec normalize(Vec value, T const* weight, int column, float rstd) {
  Vec affine;
  if (weight) affine.packed = *reinterpret_cast<float4 const*>(weight + column);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float x = __fmul_rn(__bfloat162float(value.elements[i]), rstd);
    if (weight) x = __fmul_rn(x, __bfloat162float(affine.elements[i]));
    value.elements[i] = __float2bfloat16_rn(x);
  }
  return value;
}

template <bool PostNorm>
__global__ void finalize(T const* experts, float const* weights, T* output,
                         T const* postWeight, int hidden, int topK,
                         float routedScale, float eps) {
  int token = blockIdx.x;
  Vec values[Packs]{};
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    int column = (threadIdx.x + pack * blockDim.x) * 8;
    if (column < hidden)
      values[pack] = finalizePack(experts, weights, token, column, hidden, topK,
                                  routedScale);
  }
  float rstd = 1.f;
  if constexpr (PostNorm) rstd = invRms(values, hidden, eps);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    int column = (threadIdx.x + pack * blockDim.x) * 8;
    if (column >= hidden) continue;
    Vec value = values[pack];
    if constexpr (PostNorm) value = normalize(value, postWeight, column, rstd);
    *reinterpret_cast<float4*>(output + int64_t(token) * hidden + column) =
        value.packed;
  }
}

struct BoundaryParams {
  T const* experts;
  float const* weights;
  T const* residual;
  T const* postWeight;
  T const* nextWeight;
  T* output;
  T* normalized;
  int hidden;
  int topK;
  float routedScale;
  float postEps;
  float nextEps;
  float xScale;
  float rScale;
  bool nextNorm;
  bool scaled;
};

template <bool PostNorm>
__global__ void __launch_bounds__(1024, Packs == 1 ? 2 : 1)
    finalizeBoundary(BoundaryParams p) {
  int token = blockIdx.x;
  Vec values[Packs]{};
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    int column = (threadIdx.x + pack * blockDim.x) * 8;
    if (column < p.hidden)
      values[pack] = finalizePack(p.experts, p.weights, token, column, p.hidden,
                                  p.topK, p.routedScale);
  }
  float rstd = 1.f;
  if constexpr (PostNorm) rstd = invRms(values, p.hidden, p.postEps);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    int column = (threadIdx.x + pack * blockDim.x) * 8;
    if (column >= p.hidden) continue;
    int64_t offset = int64_t(token) * p.hidden + column;
    Vec value = values[pack];
    if constexpr (PostNorm)
      value = normalize(value, p.postWeight, column, rstd);
    Vec residual;
    residual.packed = *reinterpret_cast<float4 const*>(p.residual + offset);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      T x = value.elements[i], r = residual.elements[i];
      if (p.scaled) {
        x = __float2bfloat16_rn(__fmul_rn(__bfloat162float(x), p.xScale));
        r = __float2bfloat16_rn(__fmul_rn(__bfloat162float(r), p.rScale));
      }
      value.elements[i] = __float2bfloat16_rn(
          __fadd_rn(__bfloat162float(x), __bfloat162float(r)));
    }
    values[pack] = value;
    *reinterpret_cast<float4*>(p.output + offset) = value.packed;
  }
  if (p.nextNorm) {
    // The second reduction reuses blockSum's shared storage.
    if constexpr (PostNorm) __syncthreads();
    rstd = invRms(values, p.hidden, p.nextEps);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      int column = (threadIdx.x + pack * blockDim.x) * 8;
      if (column >= p.hidden) continue;
      Vec value = normalize(values[pack], p.nextWeight, column, rstd);
      *reinterpret_cast<float4*>(p.normalized + int64_t(token) * p.hidden +
                                 column) = value.packed;
    }
  }
}

void launch(TensorView experts, TensorView weights, TensorView output,
            tvm::ffi::Optional<TensorView> postWeight, double routedScale,
            double eps, bool postNorm) {
  ffi::CUDADeviceGuard guard(output.device().device_id);
  int hidden = output.size(1);
  int topK = experts.size(1);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(experts.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(weights.dtype()), float32_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code);
  TVM_FFI_ICHECK(hidden >= 256 && hidden <= 8192 && hidden % 256 == 0);
  TVM_FFI_ICHECK(topK >= 1 && topK <= 32);
  if (output.size(0) == 0) return;
  auto kernel = postNorm ? finalize<true> : finalize<false>;
  int threads = ((hidden + 8 * Packs * 32 - 1) / (8 * Packs * 32)) * 32;
  kernel<<<output.size(0), threads, 0, get_stream(output.device())>>>(
      static_cast<T const*>(experts.data_ptr()),
      static_cast<float const*>(weights.data_ptr()),
      static_cast<T*>(output.data_ptr()),
      postWeight.has_value()
          ? static_cast<T const*>(postWeight.value().data_ptr())
          : nullptr,
      hidden, topK, routedScale, eps);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

void launchBoundary(TensorView experts, TensorView weights, TensorView residual,
                    tvm::ffi::Optional<TensorView> postWeight,
                    tvm::ffi::Optional<TensorView> nextWeight,
                    TensorView output,
                    tvm::ffi::Optional<TensorView> normalized,
                    double routedScale, double postEps, double nextEps,
                    double xScale, double rScale, bool postNorm, bool nextNorm,
                    bool scaled) {
  ffi::CUDADeviceGuard guard(output.device().device_id);
  int hidden = output.size(1);
  int topK = experts.size(1);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(experts.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(weights.dtype()), float32_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(residual.dtype()), bfloat16_code);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code);
  TVM_FFI_ICHECK(hidden >= 256 && hidden <= 8192 && hidden % 256 == 0);
  TVM_FFI_ICHECK(topK >= 1 && topK <= 32);
  if (output.size(0) == 0) return;
  auto optionalData = [](tvm::ffi::Optional<TensorView> tensor) -> T* {
    return tensor.has_value() ? static_cast<T*>(tensor.value().data_ptr())
                              : nullptr;
  };
  BoundaryParams p{static_cast<T const*>(experts.data_ptr()),
                   static_cast<float const*>(weights.data_ptr()),
                   static_cast<T const*>(residual.data_ptr()),
                   optionalData(postWeight),
                   optionalData(nextWeight),
                   static_cast<T*>(output.data_ptr()),
                   optionalData(normalized),
                   hidden,
                   topK,
                   static_cast<float>(routedScale),
                   static_cast<float>(postEps),
                   static_cast<float>(nextEps),
                   static_cast<float>(xScale),
                   static_cast<float>(rScale),
                   nextNorm,
                   scaled};
  int threads = ((hidden + 8 * Packs * 32 - 1) / (8 * Packs * 32)) * 32;
  auto kernel = postNorm ? finalizeBoundary<true> : finalizeBoundary<false>;
  kernel<<<output.size(0), threads, 0, get_stream(output.device())>>>(p);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}
}  // namespace beam

TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_moe_finalize, beam::launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_moe_finalize_norm_residual_norm,
                              beam::launchBoundary);
