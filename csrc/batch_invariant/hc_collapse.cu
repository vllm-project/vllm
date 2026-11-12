// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace vllm::batch_invariant {
namespace {

__global__ void collapse_kernel(const c10::BFloat16* hidden, const float* pre,
                                c10::BFloat16* output, int64_t elements,
                                int64_t copies, int64_t width, int64_t hs0,
                                int64_t hs1, int64_t hs2, int64_t ps0,
                                int64_t ps1) {
  const int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  const int64_t token = offset / width, column = offset % width;
  float total = 0;
  for (int64_t copy = 0; copy < copies; ++copy) {
    const float value = float(hidden[token * hs0 + copy * hs1 + column * hs2]);
    total = __fadd_rn(total, __fmul_rn(value, pre[token * ps0 + copy * ps1]));
  }
  output[offset] = c10::BFloat16(total);
}

}  // namespace

void hc_collapse(const torch::Tensor& hidden, const torch::Tensor& pre,
                 torch::Tensor& output) {
  TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 3 &&
                  hidden.scalar_type() == at::kBFloat16,
              "hc_collapse requires CUDA BF16 hidden[T,C,D]");
  TORCH_CHECK(pre.device() == hidden.device() &&
                  pre.scalar_type() == at::kFloat && pre.dim() == 2 &&
                  pre.size(0) == hidden.size(0) &&
                  pre.size(1) == hidden.size(1),
              "hc_collapse requires FP32 pre[T,C] on the hidden device");
  TORCH_CHECK(hidden.size(1) > 0 && hidden.size(2) > 0,
              "hc_collapse requires positive copy and hidden dimensions");
  TORCH_CHECK(output.device() == hidden.device() &&
                  output.scalar_type() == at::kBFloat16 && output.dim() == 2 &&
                  output.size(0) == hidden.size(0) &&
                  output.size(1) == hidden.size(2) && output.is_contiguous(),
              "hc_collapse requires contiguous BF16 output[T,D]");
  if (output.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(hidden.device());
  const int64_t elements = output.numel();
  collapse_kernel<<<(elements + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      hidden.data_ptr<c10::BFloat16>(), pre.data_ptr<float>(),
      output.data_ptr<c10::BFloat16>(), elements, hidden.size(1),
      hidden.size(2), hidden.stride(0), hidden.stride(1), hidden.stride(2),
      pre.stride(0), pre.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
