// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang's Muse Spark CUDA routing implementation.

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <limits>
#include <optional>

#include "torch_utils.h"

namespace vllm {
namespace moe {

__device__ __forceinline__ float warp_max(float value) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH__ >= 1000 && \
    __CUDA_ARCH__ < 1100
  float result;
  asm("redux.sync.max.f32 %0, %1, %2;"
      : "=f"(result)
      : "f"(value), "r"(0xffffffffu));
  return result;
#else
  #pragma unroll
  for (int offset = 16; offset; offset /= 2)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  return value;
#endif
}

inline constexpr float kNegInf = -__builtin_huge_valf();
// fp32(log2(e)) == 0x3FB8AA3B, the constant Triton folds into tl.exp.
inline constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Triton lowers `a / b` on fp32 to div.full.f32, not the div.rn.f32 nvcc emits.
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// triton/language/standard.py:50 -- `1 / (1 + exp(-x))`, lowered as
// sub.f32 / mul.f32 / ex2.approx.f32 / add.f32 / div.full.f32.
__device__ __forceinline__ float triton_sigmoid(float x) {
  return div_full(1.0f, ex2_approx((0.0f - x) * kLog2e) + 1.0f);
}

// kExpertsPerLane must be a multiple of 4 (128-bit row loads) and covers
// kExpertsPerLane * 32 experts. Lane `l` holds expert v*128 + l*4 + j at
// register slot v*4 + j, monotone in the slot -- both the stable sort and the
// group-ballot tie-break rely on that to reproduce tl.argmax.
template <int kWarpsPerBlock_>
struct RoutingTrait {
  static constexpr int kTopK = 8;
  static constexpr int kExpertsPerLane = 8;
  static constexpr int kVecsPerLane = 2;
  static constexpr int kWarpsPerBlock = kWarpsPerBlock_;
  static constexpr int kMaxExperts = 256;
};

struct RoutingParams {
  const float* __restrict__ gating_output;
  const float* __restrict__ bias;
  float* __restrict__ topk_weights;
  int32_t* __restrict__ topk_ids;
  const bool* __restrict__ is_padding;
  int64_t padding_stride;
  int num_tokens;
  int num_experts;
  int64_t stride_gm;
  int64_t stride_wm;
  int64_t stride_im;
  float router_output_multiplier;
  float route_scale;
  float division_eps;
};

template <typename Trait, bool kUsePDL>
__global__ void routing_kernel(const RoutingParams __grid_constant__ params) {
  constexpr int kEPL = Trait::kExpertsPerLane;
  constexpr int kTopK = Trait::kTopK;
  constexpr int kGroupExperts = 4 * 32;

  const int lane = threadIdx.x % 32;
  const int token =
      blockIdx.x * Trait::kWarpsPerBlock + static_cast<int>(threadIdx.x / 32);
  if (token >= params.num_tokens) return;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) cudaGridDependencySynchronize();
#endif
  if (params.is_padding != nullptr &&
      params.is_padding[token * params.padding_stride]) {
    if (lane < kTopK) {
      params.topk_ids[token * params.stride_im + lane] = -1;
      params.topk_weights[token * params.stride_wm + lane] = 0.0f;
    }
    return;
  }

  const float* __restrict__ row =
      params.gating_output + static_cast<int64_t>(token) * params.stride_gm;
  float score[kEPL];
  int expert[kEPL];

#pragma unroll
  for (int v = 0; v < Trait::kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 g = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
    if (base < params.num_experts)
      g = *reinterpret_cast<const float4*>(row + base);
    score[v * 4 + 0] = g.x;
    score[v * 4 + 1] = g.y;
    score[v * 4 + 2] = g.z;
    score[v * 4 + 3] = g.w;
#pragma unroll
    for (int j = 0; j < 4; ++j) expert[v * 4 + j] = base + j;
  }

#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] *= params.router_output_multiplier;

#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] = triton_sigmoid(score[s]);

  float biased[kEPL];
#pragma unroll
  for (int v = 0; v < Trait::kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < params.num_experts)
      b = *reinterpret_cast<const float4*>(params.bias + base);
    biased[v * 4 + 0] = b.x;
    biased[v * 4 + 1] = b.y;
    biased[v * 4 + 2] = b.z;
    biased[v * 4 + 3] = b.w;
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
#endif

  float key[kEPL];
#pragma unroll
  for (int s = 0; s < kEPL; ++s) {
    if (expert[s] < params.num_experts) {
      key[s] = biased[s] + score[s];
    } else {
      score[s] = 0.0f;
      key[s] = kNegInf;
    }
  }

  // Must be a stable sort: equal keys keep their slot order and slots are
  // monotone in expert id, which is tl.argmax's lowest-index-wins tie-break.
#pragma unroll
  for (int r = 0; r < kEPL; ++r) {
#pragma unroll
    for (int i = r % 2; i + 1 < kEPL; i += 2) {
      const bool swap = key[i + 1] > key[i];
      const float k0 = swap ? key[i + 1] : key[i];
      const float k1 = swap ? key[i] : key[i + 1];
      const int e0 = swap ? expert[i + 1] : expert[i];
      const int e1 = swap ? expert[i] : expert[i + 1];
      const float s0 = swap ? score[i + 1] : score[i];
      const float s1 = swap ? score[i] : score[i + 1];
      key[i] = k0;
      key[i + 1] = k1;
      expert[i] = e0;
      expert[i + 1] = e1;
      score[i] = s0;
      score[i + 1] = s1;
    }
  }

  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp_max(key[0]);
    const bool is_head = key[0] == round_max;

    // Lowest expert id among the tied heads: within one kGroupExperts-wide
    // group the id is monotone in the lane, so the lowest set bit of the lowest
    // non-empty group's ballot owns the round.
    unsigned owners = 0;
#pragma unroll
    for (int g = Trait::kVecsPerLane - 1; g >= 0; --g) {
      const unsigned m =
          __ballot_sync(0xffffffffu, is_head && expert[0] / kGroupExperts == g);
      owners = (m != 0u) ? m : owners;
    }
    const int owner = __ffs(owners) - 1;
    const int winner = __shfl_sync(0xffffffffu, expert[0], owner);
    const float selected = __shfl_sync(0xffffffffu, score[0], owner);

    // Tapered pop: after round k a lane can be popped at most kTopK-1-k more
    // times, so slots past that can never surface and need not be maintained.
    const bool pop = expert[0] == winner;
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
        score[s] = pop ? score[s + 1] : score[s];
      }
    // Only reachable when kTopK > kEPL: a lane popped kEPL times is exhausted
    // and must stop winning rounds.
    if (kEPL - 1 + k <= kTopK - 2)
      key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];

    score_sum += selected;
    if (lane == k) {
      out_id = winner;
      out_score = selected;
    }
  }

  score_sum += params.division_eps;
  const float inv_sum_scaled = div_full(params.route_scale, score_sum);
  if (lane < kTopK) {
    const int32_t id = out_id;
    const float weight = out_score * inv_sum_scaled;
    params.topk_ids[token * params.stride_im + lane] = id;
    params.topk_weights[token * params.stride_wm + lane] = weight;
  }
}

}  // namespace moe
}  // namespace vllm

namespace {
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

template <int Warps, bool PDL>
void launch_routing(const vllm::moe::RoutingParams& params,
                    cudaStream_t stream) {
  using Trait = vllm::moe::RoutingTrait<Warps>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((params.num_tokens + Warps - 1) / Warps);
  config.blockDim = dim3(32 * Warps);
  config.stream = stream;
  cudaLaunchAttribute attr{};
  if constexpr (PDL) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
  }
  auto err = cudaLaunchKernelEx(&config, vllm::moe::routing_kernel<Trait, PDL>,
                                params);
  STD_TORCH_CHECK(err == cudaSuccess, "Muse Spark routing launch failed: ",
                  cudaGetErrorString(err));
}
}  // namespace

void muse_spark_routing(Tensor& topk_weights, Tensor& topk_ids,
                        const Tensor& logits, const Tensor& bias,
                        double multiplier, double route_scale,
                        double division_eps,
                        const std::optional<Tensor>& is_padding) {
  STD_TORCH_CHECK(logits.device().is_cuda() && logits.dim() == 2 &&
                      logits.scalar_type() == ScalarType::Float &&
                      logits.size(1) == 256 && logits.stride(1) == 1 &&
                      logits.stride(0) >= 256 && logits.stride(0) % 4 == 0 &&
                      reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                  "logits must be aligned FP32 CUDA rows of 256 experts");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<int>::max(),
                  "Too many routing tokens");
  const int tokens = logits.size(0);
  STD_TORCH_CHECK(bias.device() == logits.device() && bias.dim() == 1 &&
                      bias.size(0) == 256 &&
                      bias.scalar_type() == ScalarType::Float &&
                      bias.stride(0) == 1 &&
                      reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
                  "bias must be an aligned FP32 tensor of 256 experts on the "
                  "logits device");
  for (const auto* output : {&topk_weights, &topk_ids}) {
    STD_TORCH_CHECK(
        output->device() == logits.device() && output->dim() == 2 &&
            output->size(0) == tokens && output->size(1) == 8 &&
            output->stride(1) == 1 && output->stride(0) >= 8,
        "routing outputs must have shape [tokens, 8] with contiguous rows");
  }
  STD_TORCH_CHECK(topk_weights.scalar_type() == ScalarType::Float &&
                      topk_ids.scalar_type() == ScalarType::Int,
                  "routing outputs must be FP32 weights and int32 IDs");
  if (is_padding) {
    STD_TORCH_CHECK(is_padding->device() == logits.device() &&
                        is_padding->scalar_type() == ScalarType::Bool &&
                        is_padding->dim() == 1 && is_padding->size(0) == tokens,
                    "is_padding must be a bool tensor of shape [tokens] on the "
                    "logits device");
  }
  if (tokens == 0) return;
  torch::stable::accelerator::DeviceGuard guard(logits.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(logits.get_device_index());
  vllm::moe::RoutingParams params{
      .gating_output = static_cast<const float*>(logits.data_ptr()),
      .bias = static_cast<const float*>(bias.data_ptr()),
      .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
      .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
      .is_padding = is_padding
                        ? static_cast<const bool*>(is_padding->data_ptr())
                        : nullptr,
      .padding_stride = is_padding ? is_padding->stride(0) : 0,
      .num_tokens = tokens,
      .num_experts = 256,
      .stride_gm = logits.stride(0),
      .stride_wm = topk_weights.stride(0),
      .stride_im = topk_ids.stride(0),
      .router_output_multiplier = static_cast<float>(multiplier),
      .route_scale = static_cast<float>(route_scale),
      .division_eps = static_cast<float>(division_eps),
  };
  const bool pdl = get_device_prop()->major >= 9;
  if (tokens <= 2048) {
    if (pdl)
      launch_routing<1, true>(params, stream);
    else
      launch_routing<1, false>(params, stream);
  } else {
    if (pdl)
      launch_routing<4, true>(params, stream);
    else
      launch_routing<4, false>(params, stream);
  }
}
