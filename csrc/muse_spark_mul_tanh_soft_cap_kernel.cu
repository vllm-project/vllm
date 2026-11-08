// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>

#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"
#include "torch_utils.h"

namespace vllm {

// 16 B of output per thread; low-precision inputs are widened in registers.
static constexpr int kMuseSoftCapVecSize = 4;

// blockIdx.x walks rows, blockIdx.y splits a row. Rows are the sampled tokens
// (often one, at decode) while a row is the whole vocab, so splitting the row
// is what keeps the GPU busy; a row-per-block grid would leave a single-token
// decode on one SM.
template <typename scalar_t>
__global__ void muse_spark_mul_tanh_soft_cap_kernel(
    float* __restrict__ out, const scalar_t* __restrict__ input, int d,
    float multiplier, float soft_cap_temp) {
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * d;
  const scalar_t* __restrict__ in_row = input + row_offset;
  float* __restrict__ out_row = out + row_offset;

  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = gridDim.y * blockDim.x;

  vectorize_with_alignment<kMuseSoftCapVecSize>(
      in_row, out_row, d, tid, stride,
      [=] __device__(float& dst, const scalar_t& src) {
        // Multiply then divide, as MulTanhSoftCap.forward does, so fp32
        // rounding matches the eager path.
        dst = soft_cap_temp *
              tanhf(static_cast<float>(src) * multiplier / soft_cap_temp);
      });
}

}  // namespace vllm

// out = soft_cap_temp * tanh(input * multiplier / soft_cap_temp), fp32.
void muse_spark_mul_tanh_soft_cap(torch::stable::Tensor& out,
                                  torch::stable::Tensor& input,
                                  double multiplier, double soft_cap_temp) {
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Float ||
          input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "muse_spark_mul_tanh_soft_cap: input must be float32, float16 or "
      "bfloat16");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Float,
                  "muse_spark_mul_tanh_soft_cap: out must be float32");
  STD_TORCH_CHECK(input.is_contiguous(),
                  "muse_spark_mul_tanh_soft_cap: input must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(),
                  "muse_spark_mul_tanh_soft_cap: out must be contiguous");
  STD_TORCH_CHECK(out.numel() == input.numel(),
                  "muse_spark_mul_tanh_soft_cap: out and input must have the "
                  "same number of elements");
  STD_TORCH_CHECK(
      soft_cap_temp != 0.0,
      "muse_spark_mul_tanh_soft_cap: soft_cap_temp must be nonzero");

  if (input.numel() == 0) {
    return;
  }

  const int d = static_cast<int>(input.size(input.dim() - 1));
  const int64_t num_tokens = input.numel() / d;

  constexpr int kThreads = 128;
  const int64_t vecs_per_row =
      (d + vllm::kMuseSoftCapVecSize - 1) / vllm::kMuseSoftCapVecSize;
  const int64_t blocks_to_cover = (vecs_per_row + kThreads - 1) / kThreads;
  constexpr int64_t kTargetBlocks = 32768;
  const int64_t blocks_per_row = std::clamp<int64_t>(
      kTargetBlocks / num_tokens, 1, std::max<int64_t>(blocks_to_cover, 1));

  dim3 grid(static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(blocks_per_row));
  dim3 block(kThreads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "muse_spark_mul_tanh_soft_cap", [&] {
        vllm::muse_spark_mul_tanh_soft_cap_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<float>(),
                                         input.const_data_ptr<scalar_t>(), d,
                                         static_cast<float>(multiplier),
                                         static_cast<float>(soft_cap_temp));
      });
}
