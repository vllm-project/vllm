// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "../../torch_utils.h"
#include "silu_mul_fp8_sm90.cuh"

void silu_mul_fp8_sm90(torch::stable::Tensor& out,
                       const torch::stable::Tensor& input,
                       torch::stable::Tensor& scales, bool round_silu) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && scales.dim() == 2);
  STD_TORCH_CHECK(input.is_cuda() && out.is_cuda() && scales.is_cuda());
  STD_TORCH_CHECK(input.get_device_index() == out.get_device_index() &&
                  input.get_device_index() == scales.get_device_index());
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16 &&
                  out.scalar_type() == ScalarType::Float8_e4m3fn &&
                  scales.scalar_type() == ScalarType::Float);
  STD_TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  const int64_t m = input.size(0), k = out.size(1);
  STD_TORCH_CHECK(k > 0 && k % 128 == 0 && input.size(1) == 2 * k &&
                  out.size(0) == m);
  STD_TORCH_CHECK(scales.size(0) == m && scales.size(1) == k / 128 &&
                  scales.stride(0) == 1 &&
                  scales.stride(1) == ((m + 3) / 4) * 4);
  STD_TORCH_CHECK(m <= INT_MAX && k <= INT_MAX / 2 && m <= INT_MAX / (2 * k));
  const torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
  STD_TORCH_CHECK(get_device_prop()->major == 9,
                  "silu_mul_fp8_sm90 requires SM90");
  if (m == 0) return;
  const auto stream = get_current_cuda_stream(input.get_device_index());
  vllm::silu_mul_fp8_sm90_kernel<<<(m * (k / 128) + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(out.mutable_data_ptr()),
      scales.mutable_data_ptr<float>(), m, k, scales.stride(1), !round_silu);
}
