// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Fixed M<=16 Marlin input projection + BF16 convolution. The device template
// preserves its original Neural Magic / Elias Frantar / IST-DASLab notices.

#include "../../quantization/marlin/kernel.h"
#include "input_conv_params.h"

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

// Stock Marlin exports this selector. Keep the fixed Mamba launch policy here;
// the generic Marlin implementation and headers remain unchanged.
namespace marlin {
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
MarlinFuncPtr get_marlin_kernel(
    vllm::ScalarType a_type, vllm::ScalarType b_type, vllm::ScalarType c_type,
    vllm::ScalarType s_type, int thread_m_blocks, int thread_n_blocks,
    int thread_k_blocks, bool m_block_size_8, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages);
__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS);
}  // namespace marlin

namespace {
// clang-format off
using torch::stable::Tensor;
using torch::headeronly::ScalarType;
// clang-format on
using vllm_mamba_input_conv::ConvParams;

struct Span {
  uintptr_t start;
  uint64_t size;
};

Span span(const Tensor& tensor, uint64_t bytes) {
  const auto start = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  STD_TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - start,
                  "Mamba Marlin operand span overflow");
  return {start, bytes};
}

bool overlaps(Span a, Span b) {
  return a.size && b.size && a.start < b.start + b.size &&
         b.start < a.start + a.size;
}

void operand(const Tensor& tensor, const Tensor& input, ScalarType dtype) {
  STD_TORCH_CHECK(tensor.device().is_cuda() &&
                      tensor.get_device_index() == input.get_device_index() &&
                      tensor.scalar_type() == dtype,
                  "Mamba Marlin operand device/dtype mismatch");
}

void matrix(const Tensor& tensor, const Tensor& input, ScalarType dtype,
            int64_t rows, int64_t columns) {
  operand(tensor, input, dtype);
  STD_TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows &&
                      tensor.size(1) == columns && tensor.is_contiguous(),
                  "Mamba Marlin operand shape/contiguity mismatch");
}

}  // namespace

// No backend selection or hidden pointer ABI: callers must own actual packed
// Marlin weights. Unsupported geometry is rejected before any device launch.
void mamba_marlin_input_conv(Tensor& input, Tensor& projected, Tensor& weight,
                             Tensor& scales, Tensor& global_scale, Tensor& taps,
                             Tensor& bias, Tensor& history,
                             Tensor& cache_indices,
                             std::optional<Tensor> const& accepted_tokens,
                             int64_t pad_slot_id, int64_t null_block_id) {
  STD_TORCH_CHECK(input.device().is_cuda(), "Mamba Marlin input must be CUDA");
  torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
  const auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 12 && props->minor == 1 &&
                      props->multiProcessorCount == 48,
                  "Mamba Marlin fusion requires actual SM121/48SM hardware");
  STD_TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 16,
                  "Mamba Marlin fusion requires 1..16 actual rows");
  const int rows = static_cast<int>(input.size(0));
  STD_TORCH_CHECK(cache_indices.numel() >= 1 && cache_indices.numel() <= 4 &&
                      rows % cache_indices.numel() == 0,
                  "Mamba Marlin requires 1..4 uniform requests");
  const int batches = static_cast<int>(cache_indices.numel());
  const int tokens = rows / batches;
  STD_TORCH_CHECK(tokens >= 1 && tokens <= 8, "Mamba Marlin requires T1..8");
  matrix(input, input, ScalarType::BFloat16, rows, 4096);
  matrix(projected, input, ScalarType::BFloat16, rows, 18560);
  matrix(weight, input, ScalarType::Int, 256, 37120);
  matrix(scales, input, ScalarType::Float8_e4m3fn, 256, 18560);
  matrix(taps, input, ScalarType::BFloat16, 10240, 4);
  operand(global_scale, input, ScalarType::Float);
  operand(bias, input, ScalarType::BFloat16);
  operand(history, input, ScalarType::BFloat16);
  operand(cache_indices, input, ScalarType::Int);
  if (accepted_tokens.has_value())
    operand(*accepted_tokens, input, ScalarType::Int);
  STD_TORCH_CHECK(global_scale.numel() == 1 && global_scale.is_contiguous() &&
                      bias.numel() == 10240 && bias.is_contiguous() &&
                      cache_indices.dim() == 1 &&
                      cache_indices.size(0) == batches &&
                      cache_indices.stride(0) >= 1 &&
                      cache_indices.stride(0) <=
                          vllm_mamba_input_conv::kBatchIndexStrideMax &&
                      (!accepted_tokens.has_value() ||
                       (accepted_tokens->numel() == batches &&
                        accepted_tokens->is_contiguous())),
                  "Mamba Marlin scalar/metadata shape mismatch");
  STD_TORCH_CHECK(history.dim() == 3 && history.size(0) > 1 &&
                      history.size(0) <= std::numeric_limits<int32_t>::max() &&
                      history.size(1) >= tokens + 2 && history.size(1) <= 10 &&
                      history.size(2) == 10240 &&
                      history.stride(0) >= history.size(1) * 10240 &&
                      history.stride(0) % 8 == 0 &&
                      history.stride(1) == 10240 && history.stride(2) == 1,
                  "Mamba Marlin fusion requires an SD history view");
  const int history_len = static_cast<int>(history.size(1));
  STD_TORCH_CHECK(
      accepted_tokens.has_value() || (tokens == 1 && history_len == 3),
      "Only non-speculative T1/H3 can omit accepted tokens");
  STD_TORCH_CHECK(pad_slot_id == -1 && null_block_id == 0,
                  "Mamba Marlin padding/null convention mismatch");
  STD_TORCH_CHECK(
      static_cast<uint64_t>(history.size(0) - 1) <=
          (std::numeric_limits<uint64_t>::max() / 2 - history_len * 10240) /
              static_cast<uint64_t>(history.stride(0)),
      "Mamba Marlin history span overflow");
  const uint64_t history_elements =
      static_cast<uint64_t>(history.size(0) - 1) * history.stride(0) +
      static_cast<uint64_t>(history_len) * 10240;
  const Span output_span = span(projected, rows * 18560 * 2);
  const Span history_span = span(history, history_elements * 2);
  const std::array<Span, 8> reads = {
      span(input, rows * 4096 * 2),
      span(weight, 256 * 37120 * 4),
      span(scales, 256 * 18560),
      span(global_scale, 4),
      span(taps, 10240 * 4 * 2),
      span(bias, 10240 * 2),
      span(cache_indices,
           (static_cast<uint64_t>(batches - 1) * cache_indices.stride(0) + 1) *
               4),
      accepted_tokens.has_value() ? span(*accepted_tokens, batches * 4)
                                  : Span{0, 0}};
  STD_TORCH_CHECK(!overlaps(output_span, history_span),
                  "Projected output and history must be disjoint");
  for (Span read : reads) {
    STD_TORCH_CHECK(
        !overlaps(output_span, read) && !overlaps(history_span, read),
        "Mamba Marlin output/history overlaps an operand");
  }
  STD_TORCH_CHECK(
      (output_span.start | history_span.start |
       reinterpret_cast<uintptr_t>(input.const_data_ptr()) |
       reinterpret_cast<uintptr_t>(weight.const_data_ptr()) |
       reinterpret_cast<uintptr_t>(scales.const_data_ptr())) %
                  16 ==
              0 &&
          reinterpret_cast<uintptr_t>(taps.const_data_ptr()) % 8 == 0,
      "Mamba Marlin vector alignment mismatch");

  ConvParams conv{
      reinterpret_cast<const uint16_t*>(taps.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(bias.const_data_ptr()),
      reinterpret_cast<uint16_t*>(history.mutable_data_ptr()),
      reinterpret_cast<const int32_t*>(cache_indices.const_data_ptr()),
      accepted_tokens.has_value()
          ? reinterpret_cast<const int32_t*>(accepted_tokens->const_data_ptr())
          : nullptr,
      history.stride(0),
      history.stride(1),
      history.stride(2),
      static_cast<int32_t>(history.size(0)),
      static_cast<int32_t>(pad_slot_id),
      static_cast<int32_t>(null_block_id),
      static_cast<int32_t>((cache_indices.stride(0)
                            << vllm_mamba_input_conv::kBatchHistoryBits) |
                           history_len)};
  if (batches == 1) conv.reserved = tokens < 8 ? history_len : 0;
  const auto launch = batches == 1 ? vllm_mamba_input_conv::launch_single
                      : rows <= 8  ? vllm_mamba_input_conv::launch_batch_small
                                   : vllm_mamba_input_conv::launch_batch_wide;
  auto status = launch(
      tokens, rows, reinterpret_cast<const int4*>(input.const_data_ptr()),
      reinterpret_cast<const int4*>(weight.const_data_ptr()),
      reinterpret_cast<int4*>(projected.mutable_data_ptr()),
      reinterpret_cast<const int4*>(scales.const_data_ptr()),
      reinterpret_cast<const float*>(global_scale.const_data_ptr()), conv,
      get_current_cuda_stream(input.get_device_index()));
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

Tensor mamba_marlin_output(Tensor& input, Tensor& weight, Tensor& scales,
                           Tensor& global_scale, Tensor& workspace) {
  STD_TORCH_CHECK(input.device().is_cuda(), "Mamba Marlin input must be CUDA");
  torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
  const auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 12 && props->minor == 1 &&
                      props->multiProcessorCount == 48,
                  "Mamba Marlin output requires actual SM121/48SM hardware");
  matrix(input, input, ScalarType::BFloat16, 8, 8192);
  matrix(weight, input, ScalarType::Int, 512, 8192);
  matrix(scales, input, ScalarType::Float8_e4m3fn, 512, 4096);
  operand(global_scale, input, ScalarType::Float);
  operand(workspace, input, ScalarType::Int);
  STD_TORCH_CHECK(global_scale.numel() == 1 && global_scale.is_contiguous() &&
                      workspace.is_contiguous() && workspace.numel() >= 48 &&
                      workspace.numel() <= std::numeric_limits<int32_t>::max(),
                  "Mamba Marlin output scale/workspace shape mismatch");
  STD_TORCH_CHECK((reinterpret_cast<uintptr_t>(input.const_data_ptr()) |
                   reinterpret_cast<uintptr_t>(weight.const_data_ptr()) |
                   reinterpret_cast<uintptr_t>(scales.const_data_ptr())) %
                          16 ==
                      0,
                  "Mamba Marlin output vector alignment mismatch");
  auto kernel = marlin::get_marlin_kernel(vllm::kBFloat16, vllm::kFE2M1f,
                                          vllm::kBFloat16, vllm::kFE4M3fn, 1, 8,
                                          8, true, false, 1, 256, false, 4);
  STD_TORCH_CHECK(kernel != nullptr && kernel != marlin::MarlinDefault,
                  "Mamba Marlin output kernel is unavailable");
  constexpr int shared = 45056;
  STD_TORCH_CHECK(props->sharedMemPerBlockOptin >= shared,
                  "Mamba Marlin output requires 45056 bytes of shared memory");
  auto status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           props->sharedMemPerBlockOptin);
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  auto output = torch::stable::empty({8, 4096}, ScalarType::BFloat16,
                                     std::nullopt, input.device());
  // Preserve ordinary Marlin scratch allocation; full-K ownership does not
  // execute its partial-reduction or lock path.
  auto scratch = torch::stable::empty({48 * 16 * 256}, ScalarType::Float,
                                      std::nullopt, input.device());
  kernel<<<32, 256, shared,
           get_current_cuda_stream(input.get_device_index())>>>(
      reinterpret_cast<const int4*>(input.const_data_ptr()),
      reinterpret_cast<const int4*>(weight.const_data_ptr()),
      reinterpret_cast<int4*>(output.mutable_data_ptr()),
      reinterpret_cast<int4*>(scratch.mutable_data_ptr()), nullptr,
      reinterpret_cast<const int4*>(scales.const_data_ptr()),
      reinterpret_cast<const float*>(global_scale.const_data_ptr()), nullptr, 8,
      4096, 8192, 8192, reinterpret_cast<int*>(workspace.mutable_data_ptr()),
      false, false, true, shared);
  status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  return output;
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("mamba_marlin_input_conv", TORCH_BOX(&mamba_marlin_input_conv));
  m.impl("mamba_marlin_output", TORCH_BOX(&mamba_marlin_output));
}
