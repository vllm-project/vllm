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

template <typename Index>
__device__ int local_expert(const Index* routes, const int32_t* map,
                            int64_t position, int topk, int64_t stride0,
                            int64_t stride1, int experts, int global_experts) {
  const int64_t expert =
      routes[position / topk * stride0 + position % topk * stride1];
  if (expert < 0 || expert >= global_experts) return -1;
  const int64_t local = map ? map[expert] : expert;
  return local >= 0 && local < experts ? int(local) : -1;
}

template <typename Index>
__global__ void expert_offsets_kernel(const Index* routes, const int32_t* map,
                                      int32_t* inverse, int32_t* counts,
                                      int64_t positions, int topk,
                                      int64_t stride0, int64_t stride1,
                                      int experts, int global_experts) {
  __shared__ int warp_counts[8];
  const int expert = blockIdx.x, lane = threadIdx.x % 32,
            warp = threadIdx.x / 32;
  int prior = 0;
  for (int64_t begin = 0; begin < positions; begin += blockDim.x) {
    const int64_t position = begin + threadIdx.x;
    const bool owned = position < positions &&
                       local_expert(routes, map, position, topk, stride0,
                                    stride1, experts, global_experts) == expert;
    int prefix = int(owned);
    for (int offset = 1; offset < 32; offset *= 2) {
      const int previous = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset) prefix += previous;
    }
    if (lane == 31) warp_counts[warp] = prefix;
    __syncthreads();
    int before_warp = 0, total = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) before_warp += warp_counts[w];
      total += warp_counts[w];
    }
    if (owned) inverse[position] = prior + before_warp + prefix - 1;
    prior += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[expert] = prior;
}

__global__ void expert_starts_kernel(const int32_t* counts, int32_t* starts,
                                     int experts, int alignment) {
  int64_t start = 0;
  for (int expert = 0; expert < experts; ++expert) {
    starts[expert] = int32_t(start);
    start += (int64_t(counts[expert]) + alignment - 1) / alignment * alignment;
  }
}

template <typename Index>
__global__ void expert_copy_kernel(const uint8_t* a, const float* scale,
                                   const Index* routes, const int32_t* map,
                                   int32_t* inverse, const int32_t* starts,
                                   uint8_t* output, float* output_scale,
                                   int32_t* layout, int topk, int width,
                                   int scale_columns, int experts,
                                   int global_experts, int capacity,
                                   int64_t as0, int64_t as1, int64_t ss0,
                                   int64_t ss1, int64_t rs0, int64_t rs1) {
  const int64_t position = blockIdx.x;
  const int expert = local_expert(routes, map, position, topk, rs0, rs1,
                                  experts, global_experts);
  if (expert < 0) return;
  // Local ranks remain unchanged until every channel block has finished
  // copying.
  const int64_t row = starts[expert] + inverse[position];
  CUDA_KERNEL_ASSERT(row >= 0 && row < capacity);
  const int64_t token = position / topk;
  const int column = blockIdx.y * blockDim.x + threadIdx.x;
  if (column < width)
    output[row * width + column] = a[token * as0 + column * as1];
  if (blockIdx.y == 0) {
    if (threadIdx.x == 0) layout[row] = expert;
    for (int s = threadIdx.x; s < scale_columns; s += blockDim.x) {
      output_scale[row * scale_columns + s] = scale[token * ss0 + s * ss1];
    }
  }
}

// Same copy as expert_copy_kernel with one CTA per routed position and
// 16-byte vectors for contiguous, aligned rows.
template <typename Index>
__global__ void expert_copy_vec_kernel(
    const uint8_t* a, const float* scale, const Index* routes,
    const int32_t* map, const int32_t* inverse, const int32_t* starts,
    uint8_t* output, float* output_scale, int32_t* layout, int topk, int width,
    int scale_columns, int experts, int global_experts, int capacity,
    int64_t as0, int64_t ss0, int64_t ss1, int64_t rs0, int64_t rs1) {
  const int64_t position = blockIdx.x;
  const int expert = local_expert(routes, map, position, topk, rs0, rs1,
                                  experts, global_experts);
  if (expert < 0) return;
  const int64_t row = starts[expert] + inverse[position];
  CUDA_KERNEL_ASSERT(row >= 0 && row < capacity);
  const int64_t token = position / topk;
  const auto* src = reinterpret_cast<const uint4*>(a + token * as0);
  auto* dst = reinterpret_cast<uint4*>(output + row * width);
  for (int v = threadIdx.x; v < width / 16; v += blockDim.x) dst[v] = src[v];
  if (threadIdx.x == 0) layout[row] = expert;
  for (int c = threadIdx.x; c < scale_columns; c += blockDim.x)
    output_scale[row * scale_columns + c] = scale[token * ss0 + c * ss1];
}

template <typename Index>
__global__ void inverse_finalize_kernel(const Index* routes, const int32_t* map,
                                        const int32_t* starts, int32_t* inverse,
                                        int64_t positions, int topk,
                                        int64_t rs0, int64_t rs1, int experts,
                                        int global_experts) {
  const int64_t position = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position >= positions) return;
  const int expert = local_expert(routes, map, position, topk, rs0, rs1,
                                  experts, global_experts);
  if (expert >= 0) inverse[position] += starts[expert];
}

}  // namespace

void expert_permute(const torch::Tensor& a, const torch::Tensor& scale,
                    const torch::Tensor& routes,
                    const std::optional<torch::Tensor>& expert_map,
                    torch::Tensor& output, torch::Tensor& output_scale,
                    torch::Tensor& layout, torch::Tensor& inverse,
                    torch::Tensor& counts, torch::Tensor& starts,
                    int64_t alignment) {
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 &&
                  a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  scale.dim() == 2 && scale.scalar_type() == at::kFloat &&
                  scale.device() == a.device() && scale.size(0) == a.size(0) &&
                  scale.size(1) > 0 && a.size(1) % scale.size(1) == 0,
              "Expert permutation requires E4M3 rows and FP32 group scales");
  TORCH_CHECK(routes.dim() == 2 && routes.size(0) == a.size(0) &&
                  routes.size(1) > 0 && routes.device() == a.device() &&
                  (routes.scalar_type() == at::kInt ||
                   routes.scalar_type() == at::kLong),
              "Expert routes must be signed INT32/INT64 [tokens, topk]");
  for (const auto& tensor :
       {output, output_scale, layout, inverse, counts, starts})
    TORCH_CHECK(
        tensor.device() == a.device() && tensor.is_contiguous(),
        "Expert permutation buffers must share contiguous CUDA storage");
  TORCH_CHECK(output.dim() == 2 && output.scalar_type() == a.scalar_type() &&
                  output.size(1) == a.size(1) && output_scale.dim() == 2 &&
                  output_scale.scalar_type() == at::kFloat &&
                  output_scale.size(0) == output.size(0) &&
                  output_scale.size(1) == scale.size(1) && layout.dim() == 1 &&
                  layout.size(0) == output.size(0) &&
                  inverse.sizes() == routes.sizes() && counts.dim() == 1 &&
                  starts.sizes() == counts.sizes() && counts.numel() > 0 &&
                  alignment > 0 && alignment <= INT_MAX,
              "Invalid expert permutation workspace geometry");
  for (const auto& tensor : {layout, inverse, counts, starts})
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                "Expert metadata must be INT32");
  if (expert_map)
    TORCH_CHECK(expert_map->device() == a.device() && expert_map->dim() == 1 &&
                    expert_map->is_contiguous() &&
                    expert_map->scalar_type() == at::kInt,
                "Expert ownership map must be contiguous CUDA INT32");
  const int64_t positions = routes.numel(), experts = counts.numel();
  TORCH_CHECK(positions <= INT_MAX && experts <= INT_MAX &&
                  output.size(0) <= INT_MAX && a.size(1) <= 65535 * 256 &&
                  scale.size(1) <= INT_MAX &&
                  (!expert_map || expert_map->numel() <= INT_MAX),
              "Expert permutation geometry exceeds launch limits");
  if (positions == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INTEGRAL_TYPES(routes.scalar_type(), "expert_permute", [&] {
    const auto* map = expert_map ? expert_map->data_ptr<int32_t>() : nullptr;
    const int global_experts = expert_map ? expert_map->numel() : experts;
    expert_offsets_kernel<scalar_t><<<experts, 256, 0, stream>>>(
        routes.data_ptr<scalar_t>(), map, inverse.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), positions, routes.size(1), routes.stride(0),
        routes.stride(1), experts, global_experts);
    expert_starts_kernel<<<1, 1, 0, stream>>>(counts.data_ptr<int32_t>(),
                                              starts.data_ptr<int32_t>(),
                                              experts, alignment);
    const bool vec = a.stride(1) == 1 && a.size(1) % 16 == 0 &&
                     a.stride(0) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0;
    if (vec) {
      expert_copy_vec_kernel<scalar_t><<<positions, 128, 0, stream>>>(
          reinterpret_cast<const uint8_t*>(a.data_ptr()),
          scale.data_ptr<float>(), routes.data_ptr<scalar_t>(), map,
          inverse.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          output_scale.data_ptr<float>(), layout.data_ptr<int32_t>(),
          routes.size(1), a.size(1), scale.size(1), experts, global_experts,
          output.size(0), a.stride(0), scale.stride(0), scale.stride(1),
          routes.stride(0), routes.stride(1));
    }
    const dim3 grid(vec ? 0 : positions, (a.size(1) + 255) / 256);
    if (!vec)
      expert_copy_kernel<scalar_t><<<grid, 256, 0, stream>>>(
          reinterpret_cast<const uint8_t*>(a.data_ptr()),
          scale.data_ptr<float>(), routes.data_ptr<scalar_t>(), map,
          inverse.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          output_scale.data_ptr<float>(), layout.data_ptr<int32_t>(),
          routes.size(1), a.size(1), scale.size(1), experts, global_experts,
          output.size(0), a.stride(0), a.stride(1), scale.stride(0),
          scale.stride(1), routes.stride(0), routes.stride(1));
    inverse_finalize_kernel<scalar_t>
        <<<(positions + 255) / 256, 256, 0, stream>>>(
            routes.data_ptr<scalar_t>(), map, starts.data_ptr<int32_t>(),
            inverse.data_ptr<int32_t>(), positions, routes.size(1),
            routes.stride(0), routes.stride(1), experts, global_experts);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
