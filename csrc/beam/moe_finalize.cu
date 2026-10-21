// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Finalizes MegaMoE's token-major expert rows [tokens, top_k, hidden] with
// Beam's post-MoE norms and residual.

#include "../torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <torch/headeronly/core/ScalarType.h>

namespace vllm {
namespace beam {
namespace {

using T = __nv_bfloat16;
constexpr int Packs = 1;
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

void checkInputs(torch::stable::Tensor const& experts,
                 torch::stable::Tensor const& weights,
                 torch::stable::Tensor const& output) {
  using torch::headeronly::ScalarType;
  int hidden = output.size(1);
  int topK = experts.size(1);
  STD_TORCH_CHECK(experts.scalar_type() == ScalarType::BFloat16 &&
                  weights.scalar_type() == ScalarType::Float &&
                  output.scalar_type() == ScalarType::BFloat16);
  STD_TORCH_CHECK(hidden >= 256 && hidden <= 8192 && hidden % 256 == 0);
  STD_TORCH_CHECK(topK >= 1 && topK <= 32);
}

int blockThreads(int hidden) {
  return ((hidden + 8 * Packs * 32 - 1) / (8 * Packs * 32)) * 32;
}

}  // namespace
}  // namespace beam
}  // namespace vllm

void beam_moe_finalize(torch::stable::Tensor const& experts,
                       torch::stable::Tensor const& weights,
                       torch::stable::Tensor& output,
                       std::optional<torch::stable::Tensor> post_weight,
                       double routed_scale, double eps, bool post_norm) {
  using namespace vllm::beam;
  int const device = output.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  checkInputs(experts, weights, output);
  if (output.size(0) == 0) return;
  int hidden = output.size(1);
  auto kernel = post_norm ? finalize<true> : finalize<false>;
  kernel<<<output.size(0), blockThreads(hidden), 0,
           get_current_cuda_stream(device)>>>(
      static_cast<T const*>(experts.data_ptr()),
      static_cast<float const*>(weights.data_ptr()),
      static_cast<T*>(output.data_ptr()),
      post_weight.has_value() ? static_cast<T const*>(post_weight->data_ptr())
                              : nullptr,
      hidden, experts.size(1), routed_scale, eps);
  auto status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void beam_moe_finalize_norm(torch::stable::Tensor const& experts,
                            torch::stable::Tensor const& weights,
                            torch::stable::Tensor const& residual,
                            std::optional<torch::stable::Tensor> post_weight,
                            std::optional<torch::stable::Tensor> next_weight,
                            torch::stable::Tensor& output,
                            std::optional<torch::stable::Tensor> normalized,
                            double routed_scale, double post_eps,
                            double next_eps, double x_scale, double r_scale,
                            bool post_norm, bool next_norm, bool scaled) {
  using namespace vllm::beam;
  int const device = output.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  checkInputs(experts, weights, output);
  STD_TORCH_CHECK(residual.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  if (output.size(0) == 0) return;
  auto optionalData = [](std::optional<torch::stable::Tensor> const& tensor) {
    return tensor.has_value() ? static_cast<T*>(tensor->data_ptr()) : nullptr;
  };
  int hidden = output.size(1);
  BoundaryParams p{static_cast<T const*>(experts.data_ptr()),
                   static_cast<float const*>(weights.data_ptr()),
                   static_cast<T const*>(residual.data_ptr()),
                   optionalData(post_weight),
                   optionalData(next_weight),
                   static_cast<T*>(output.data_ptr()),
                   optionalData(normalized),
                   hidden,
                   static_cast<int>(experts.size(1)),
                   static_cast<float>(routed_scale),
                   static_cast<float>(post_eps),
                   static_cast<float>(next_eps),
                   static_cast<float>(x_scale),
                   static_cast<float>(r_scale),
                   next_norm,
                   scaled};
  auto kernel = post_norm ? finalizeBoundary<true> : finalizeBoundary<false>;
  kernel<<<output.size(0), blockThreads(hidden), 0,
           get_current_cuda_stream(device)>>>(p);
  auto status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
