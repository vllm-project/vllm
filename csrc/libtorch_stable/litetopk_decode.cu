// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "torch_utils.h"
#include "litetopk_decode/select.cuh"

namespace {
using Tensor = torch::stable::Tensor;
using DType = torch::headeronly::ScalarType;

template <typename Kernel, typename Params>
void launch(Kernel kernel, const Params& params, int blocks, int threads,
            size_t smem, cudaStream_t stream) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  const auto err = cudaLaunchKernelEx(&config, kernel, params);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "LiteTopK launch failed: ", cudaGetErrorString(err));
}

void select_impl(const Tensor& scores, const Tensor& lengths, Tensor& histogram,
                 Tensor& output, Tensor& workspace, int64_t capacity) {
  const int64_t rows = scores.size(0);
  const int64_t max_rows = workspace.numel() / (16 + capacity * 8);
  auto* base = workspace.mutable_data_ptr<uint8_t>();
  litetopk::Params params{
      .scores = scores.const_data_ptr<float>(),
      .score_stride = scores.stride(0),
      .score_width = static_cast<uint32_t>(scores.size(1)),
      .lengths = lengths.const_data_ptr<int32_t>(),
      .histogram = histogram.mutable_data_ptr<int32_t>(),
      .out_stride = output.stride(0),
      .out = output.mutable_data_ptr<int32_t>(),
      .state = reinterpret_cast<litetopk::RowState*>(base),
      .candidates = reinterpret_cast<unsigned long long*>(base + max_rows * 16),
      .capacity = static_cast<uint32_t>(capacity),
      .rows = static_cast<uint32_t>(rows),
  };
  const int sms = get_device_prop()->multiProcessorCount;
  const auto stream = get_current_cuda_stream();
  constexpr size_t smem = std::max(sizeof(litetopk::Selector<512>::Shared),
                                   sizeof(litetopk::Selector<1024>::Shared));
  static_assert(smem <= 48 * 1024);
  if (rows < 16 || sms < 64) {
    launch(litetopk::select_kernel<1>, params, sms, 512, smem, stream);
  } else if (rows < 64) {
    launch(litetopk::select_kernel_wide, params, sms, 1024, smem, stream);
  } else if ((rows > sms && rows <= sms * 192 / 148) ||
             (rows > 2 * sms && rows <= 3 * sms)) {
    launch(litetopk::select_kernel<3>, params, 3 * sms, 512, smem, stream);
  } else {
    launch(litetopk::select_kernel<2>, params, 2 * sms, 512, smem, stream);
  }
}
}  // namespace

void litetopk_decode(const Tensor& scores, const Tensor& lengths,
                     Tensor& histogram, Tensor& output, Tensor& workspace,
                     int64_t capacity) {
  STD_TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  const torch::stable::accelerator::DeviceGuard guard(
      scores.get_device_index());
  STD_TORCH_CHECK(get_device_prop()->major == 10, "LiteTopK requires SM100");
  const Tensor* tensors[] = {&lengths, &histogram, &output, &workspace};
  for (const Tensor* t : tensors) {
    STD_TORCH_CHECK(
        t->is_cuda() && t->get_device_index() == scores.get_device_index(),
        "LiteTopK tensors must be on the same CUDA device");
  }
  STD_TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1 &&
                      scores.scalar_type() == DType::Float,
                  "scores must be a row-major FP32 matrix");
  STD_TORCH_CHECK(
      scores.size(1) < (1 << 24) && scores.stride(0) % 4 == 0 &&
          scores.stride(0) >= ((scores.size(1) + 3) / 4) * 4 &&
          reinterpret_cast<uintptr_t>(scores.const_data_ptr()) % 16 == 0,
      "scores require 16-byte aligned and padded rows, width < 2^24");
  const auto rows = scores.size(0);
  STD_TORCH_CHECK(lengths.scalar_type() == DType::Int &&
                      lengths.is_contiguous() && lengths.numel() == rows,
                  "lengths must be int32, one per row");
  STD_TORCH_CHECK(histogram.scalar_type() == DType::Int &&
                      histogram.is_contiguous() && histogram.dim() == 2 &&
                      histogram.size(0) == rows && histogram.size(1) == 1024,
                  "histogram must be int32 [rows, 1024]");
  STD_TORCH_CHECK(output.scalar_type() == DType::Int && output.dim() == 2 &&
                      output.size(0) == rows && output.size(1) == 2048 &&
                      output.stride(1) == 1 && output.stride(0) >= 2048,
                  "output must be int32 [rows, 2048]");
  STD_TORCH_CHECK(capacity > 0 && capacity <= (1 << 24),
                  "invalid candidate capacity");
  const int64_t row_bytes = 16 + capacity * 8;
  STD_TORCH_CHECK(
      workspace.scalar_type() == DType::Byte && workspace.dim() == 1 &&
          workspace.is_contiguous() && workspace.numel() % row_bytes == 0 &&
          workspace.numel() / row_bytes >= rows &&
          reinterpret_cast<uintptr_t>(workspace.const_data_ptr()) % 16 == 0,
      "workspace must hold aligned persistent states and candidates");
  if (rows != 0)
    select_impl(scores, lengths, histogram, output, workspace, capacity);
}
