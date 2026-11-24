// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <climits>
#include <optional>

namespace vllm::batch_invariant {
namespace {

template <typename Index, typename Slot>
__global__ void gather_slots_kernel(const c10::BFloat16* values,
                                    const Index* routes, const int32_t* inverse,
                                    const int32_t* expert_map, Slot* slots,
                                    int64_t elements, int64_t width,
                                    int64_t topk, int64_t experts, int64_t rows,
                                    int64_t vs0, int64_t vs1, int64_t rs0,
                                    int64_t rs1, int64_t is0, int64_t is1) {
  const int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  const int64_t column = offset % width;
  const int64_t slot = offset / width % topk;
  const int64_t token = offset / (width * topk);
  const int64_t expert = routes[token * rs0 + slot * rs1];
  c10::BFloat16 value(0.0f);
  if (expert >= 0 && expert < experts && expert_map[expert] >= 0) {
    const int64_t row = inverse[token * is0 + slot * is1];
    if (row >= 0 && row < rows) value = values[row * vs0 + column * vs1];
  }
  slots[offset] = Slot(value);
}

// One CTA per (token, slot) row, 16-byte vectors; BF16 slots only.
template <typename Index>
__global__ void gather_slots_vec_kernel(
    const c10::BFloat16* values, const Index* routes, const int32_t* inverse,
    const int32_t* expert_map, c10::BFloat16* slots, int64_t width,
    int64_t topk, int64_t experts, int64_t rows, int64_t vs0, int64_t rs0,
    int64_t rs1, int64_t is0, int64_t is1) {
  const int64_t position = blockIdx.x;
  const int64_t slot = position % topk, token = position / topk;
  const int64_t expert = routes[token * rs0 + slot * rs1];
  int64_t row = -1;
  if (expert >= 0 && expert < experts && expert_map[expert] >= 0) {
    row = inverse[token * is0 + slot * is1];
    if (row >= rows) row = -1;
  }
  auto* dst = reinterpret_cast<uint4*>(slots + position * width);
  const auto* src = reinterpret_cast<const uint4*>(values + row * vs0);
  for (int64_t v = threadIdx.x; v < width / 8; v += blockDim.x)
    dst[v] = row >= 0 ? src[v] : make_uint4(0, 0, 0, 0);
}

template <typename Output, typename Index>
__global__ void combine_slots_kernel(const float* slots, const float* weights,
                                     const Index* routes, Output* output,
                                     int64_t elements, int64_t width,
                                     int64_t topk, int64_t experts, int64_t ss0,
                                     int64_t ss1, int64_t ss2, int64_t ws0,
                                     int64_t ws1, int64_t rs0, int64_t rs1) {
  const int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  const int64_t token = offset / width, column = offset % width;
  float total = 0;
  for (int64_t slot = 0; slot < topk; ++slot) {
    float weight = 0;
    float value = 0;
    if (!routes || (routes[token * rs0 + slot * rs1] >= 0 &&
                    routes[token * rs0 + slot * rs1] < experts)) {
      weight = weights[token * ws0 + slot * ws1];
      value = slots[token * ss0 + slot * ss1 + column * ss2];
    }
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(total)
        : "f"(value), "f"(weight), "f"(total));
  }
  output[offset] = Output(total);
}

int launch_blocks(int64_t elements) {
  const int64_t blocks = (elements + 255) / 256;
  TORCH_CHECK(blocks <= INT_MAX, "Ordered top-k launch is too large");
  return int(blocks);
}

void check_routes(const torch::Tensor& routes, const torch::Tensor& weights) {
  TORCH_CHECK(routes.dim() == 2 && routes.sizes() == weights.sizes() &&
                  routes.device() == weights.device() &&
                  (routes.scalar_type() == at::kInt ||
                   routes.scalar_type() == at::kLong),
              "Ordered routes must be signed INT32/INT64 [tokens, topk]");
}

}  // namespace

void gather_expert_slots(const torch::Tensor& values,
                         const torch::Tensor& routes,
                         const torch::Tensor& inverse,
                         const torch::Tensor& expert_map,
                         torch::Tensor& slots) {
  const c10::cuda::CUDAGuard guard(values.device());
  TORCH_CHECK(
      values.is_cuda() && values.scalar_type() == at::kBFloat16 &&
          values.dim() == 2 && slots.device() == values.device() &&
          (slots.scalar_type() == at::kFloat ||
           slots.scalar_type() == at::kBFloat16) &&
          slots.is_contiguous() && slots.dim() == 3 &&
          slots.size(2) == values.size(1) && slots.size(2) > 0,
      "Expert slots require BF16 rows and contiguous FP32/BF16 [tokens, "
      "topk, width]");
  TORCH_CHECK(inverse.device() == values.device() &&
                  inverse.scalar_type() == at::kInt &&
                  inverse.sizes() == routes.sizes() &&
                  expert_map.device() == values.device() &&
                  expert_map.scalar_type() == at::kInt &&
                  expert_map.dim() == 1 && expert_map.is_contiguous(),
              "Invalid expert inverse or ownership map");
  check_routes(routes, slots.select(2, 0));
  if (slots.numel() == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INTEGRAL_TYPES(routes.scalar_type(), "gather_expert_slots", [&] {
    auto launch = [&](auto* out) {
      gather_slots_kernel<scalar_t>
          <<<launch_blocks(slots.numel()), 256, 0, stream>>>(
              values.data_ptr<c10::BFloat16>(), routes.data_ptr<scalar_t>(),
              inverse.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(), out,
              slots.numel(), slots.size(2), slots.size(1), expert_map.numel(),
              values.size(0), values.stride(0), values.stride(1),
              routes.stride(0), routes.stride(1), inverse.stride(0),
              inverse.stride(1));
    };
    const bool vec = slots.scalar_type() == at::kBFloat16 &&
                     values.stride(1) == 1 && slots.size(2) % 8 == 0 &&
                     values.stride(0) % 8 == 0 &&
                     reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(slots.data_ptr()) % 16 == 0;
    if (vec)
      gather_slots_vec_kernel<scalar_t>
          <<<slots.size(0) * slots.size(1), 128, 0, stream>>>(
              values.data_ptr<c10::BFloat16>(), routes.data_ptr<scalar_t>(),
              inverse.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(),
              slots.data_ptr<c10::BFloat16>(), slots.size(2), slots.size(1),
              expert_map.numel(), values.size(0), values.stride(0),
              routes.stride(0), routes.stride(1), inverse.stride(0),
              inverse.stride(1));
    else if (slots.scalar_type() == at::kFloat)
      launch(slots.data_ptr<float>());
    else
      launch(slots.data_ptr<c10::BFloat16>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ordered_topk_combine(const torch::Tensor& slots,
                          const torch::Tensor& weights,
                          const std::optional<torch::Tensor>& routes,
                          int64_t experts, torch::Tensor& output) {
  const c10::cuda::CUDAGuard guard(slots.device());
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kFloat &&
                  slots.dim() == 3 && weights.device() == slots.device() &&
                  weights.scalar_type() == at::kFloat && weights.dim() == 2 &&
                  weights.size(0) == slots.size(0) &&
                  weights.size(1) == slots.size(1) &&
                  output.device() == slots.device() && output.dim() == 2 &&
                  output.size(0) == slots.size(0) &&
                  output.size(1) == slots.size(2) && output.is_contiguous(),
              "Ordered combine requires FP32 slots/weights and a matching "
              "contiguous output");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 ||
                  output.scalar_type() == at::kHalf ||
                  output.scalar_type() == at::kFloat,
              "Invalid ordered combine output dtype");
  if (routes) {
    check_routes(*routes, weights);
    TORCH_CHECK(experts >= 0, "Invalid expert count");
  }
  if (output.numel() == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, output.scalar_type(), "ordered_topk_combine",
      [&] {
        using Output = scalar_t;
        auto launch = [&]<typename Index>() {
          combine_slots_kernel<Output, Index>
              <<<launch_blocks(output.numel()), 256, 0, stream>>>(
                  slots.data_ptr<float>(), weights.data_ptr<float>(),
                  routes ? routes->data_ptr<Index>() : nullptr,
                  output.data_ptr<Output>(), output.numel(), slots.size(2),
                  slots.size(1), experts, slots.stride(0), slots.stride(1),
                  slots.stride(2), weights.stride(0), weights.stride(1),
                  routes ? routes->stride(0) : 0,
                  routes ? routes->stride(1) : 0);
        };
        if (routes && routes->scalar_type() == at::kLong)
          launch.template operator()<int64_t>();
        else
          launch.template operator()<int32_t>();
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
