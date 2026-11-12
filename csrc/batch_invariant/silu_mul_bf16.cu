// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <optional>

#include "activation_math.cuh"

namespace vllm::batch_invariant {
namespace {

__global__ void silu_mul_kernel(const c10::BFloat16* input,
                                c10::BFloat16* output, int64_t elements,
                                int64_t width, bool clamp, float limit) {
  const int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  const int64_t row = offset / width, column = offset % width;
  float gate = float(input[row * width * 2 + column]);
  float up = float(input[row * width * 2 + width + column]);
  if (clamp) {
    gate = float(c10::BFloat16(fminf(gate, limit)));
    up = float(c10::BFloat16(fminf(fmaxf(up, -limit), limit)));
  }
  output[offset] = c10::BFloat16(silu(gate) * up);
}

}  // namespace

void silu_mul_bf16(const torch::Tensor& input, torch::Tensor& output,
                   std::optional<double> clamp_limit) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
                  input.dim() == 2 && input.size(1) > 0 &&
                  input.size(1) % 2 == 0 && input.is_contiguous(),
              "silu_mul_bf16 requires contiguous CUDA BF16 input[M,2N]");
  TORCH_CHECK(output.device() == input.device() &&
                  output.scalar_type() == at::kBFloat16 && output.dim() == 2 &&
                  output.size(0) == input.size(0) &&
                  output.size(1) == input.size(1) / 2 && output.is_contiguous(),
              "silu_mul_bf16 requires contiguous BF16 output[M,N]");
  TORCH_CHECK(!clamp_limit || (std::isfinite(*clamp_limit) && *clamp_limit > 0),
              "SwiGLU clamp must be finite and positive");
  if (output.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t elements = output.numel();
  silu_mul_kernel<<<(elements + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(),
      elements, output.size(1), clamp_limit.has_value(),
      float(clamp_limit.value_or(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
