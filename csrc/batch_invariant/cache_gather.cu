// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/extension.h>

#include <algorithm>
#include <optional>

namespace vllm::batch_invariant {
namespace {

__global__ void cast_indexer_logits_kernel(const c10::BFloat16* input,
                                           float* output, const int32_t* ends,
                                           int64_t width, int64_t input_stride,
                                           int64_t output_stride) {
  const int64_t row = blockIdx.x;
  const int32_t end = ends[row];
  CUDA_KERNEL_ASSERT(end >= 0 && end <= width);
  for (int64_t column = blockIdx.y * blockDim.x + threadIdx.x; column < end;
       column += gridDim.y * blockDim.x) {
    output[row * output_stride + column] =
        static_cast<float>(input[row * input_stride + column]);
  }
}

__global__ void dequant_gather_nvfp4_kernel(
    const uint8_t* cache, c10::BFloat16* output, const int32_t* seq_lens,
    const int32_t* gather_lens, const int32_t* table, int64_t table_width,
    int64_t page_stride, int64_t pages, int64_t block_size,
    int64_t output_stride0, int64_t output_stride1, int64_t output_rows,
    int64_t offset) {
  const int64_t request = blockIdx.x;
  const int32_t length = seq_lens[request];
  const int32_t gather = gather_lens ? gather_lens[request] : length;
  CUDA_KERNEL_ASSERT(gather >= 0 && gather <= length &&
                     offset + gather <= output_rows);
  constexpr float magnitudes[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int64_t i = blockIdx.y; i < gather; i += gridDim.y) {
    const int64_t position = length - gather + i;
    CUDA_KERNEL_ASSERT(position / block_size < table_width);
    const int32_t page_id =
        table[request * table_width + position / block_size];
    CUDA_KERNEL_ASSERT(page_id >= 0 && page_id < pages);
    const uint8_t* page = cache + int64_t(page_id) * page_stride;
    // Interleaved V4.1 NVFP4 record: 256 packed bytes, then 32 e4m3 scales.
    const uint8_t* record = page + (position % block_size) * 288;
    for (int dim = threadIdx.x; dim < 512; dim += blockDim.x) {
      const uint8_t packed = record[dim / 2];
      const uint8_t code = (packed >> ((dim & 1) * 4)) & 15;
      const uint8_t scale_bits = record[256 + dim / 16];
      const float scale = static_cast<float>(
          c10::Float8_e4m3fn(scale_bits, c10::Float8_e4m3fn::from_bits()));
      const float value =
          code & 8 ? -magnitudes[code & 7] : magnitudes[code & 7];
      output[request * output_stride0 + (offset + i) * output_stride1 + dim] =
          c10::BFloat16(value * scale);
    }
  }
}

}  // namespace

void cast_indexer_logits(const torch::Tensor& input,
                         const torch::Tensor& row_ends, torch::Tensor& output) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
                  input.dim() == 2 && input.stride(1) == 1,
              "cast_indexer_logits requires row-major CUDA BF16 input");
  TORCH_CHECK(row_ends.device() == input.device() && row_ends.dim() == 1 &&
                  row_ends.scalar_type() == at::kInt &&
                  row_ends.is_contiguous() && row_ends.numel() == input.size(0),
              "cast_indexer_logits requires contiguous int32 row ends");
  TORCH_CHECK(output.device() == input.device() &&
                  output.scalar_type() == at::kFloat &&
                  output.sizes() == input.sizes() && output.stride(1) == 1,
              "cast_indexer_logits requires matching CUDA FP32 output");
  if (input.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const dim3 grid(input.size(0),
                  std::min<int64_t>((input.size(1) + 255) / 256, 8));
  cast_indexer_logits_kernel<<<grid, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<c10::BFloat16>(), output.data_ptr<float>(),
      row_ends.data_ptr<int32_t>(), input.size(1), input.stride(0),
      output.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequant_gather_nvfp4(const torch::Tensor& cache,
                          const torch::Tensor& seq_lens,
                          const std::optional<torch::Tensor>& gather_lens,
                          const torch::Tensor& block_table, int64_t block_size,
                          int64_t offset, torch::Tensor& output) {
  TORCH_CHECK(
      cache.is_cuda() && cache.scalar_type() == at::kByte && cache.dim() == 3 &&
          cache.size(2) == 288 && cache.stride(2) == 1 && block_size > 0 &&
          cache.size(1) == block_size && cache.stride(0) >= block_size * 288,
      "dequant_gather_nvfp4 requires paged CUDA uint8 NVFP4 records");
  TORCH_CHECK(
      seq_lens.device() == cache.device() && seq_lens.dim() == 1 &&
          seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
      "dequant_gather_nvfp4 requires contiguous int32 sequence lengths");
  TORCH_CHECK(block_table.device() == cache.device() &&
                  block_table.dim() == 2 &&
                  block_table.scalar_type() == at::kInt &&
                  block_table.is_contiguous() &&
                  block_table.size(0) == seq_lens.numel(),
              "dequant_gather_nvfp4 requires contiguous int32 block table");
  TORCH_CHECK(output.device() == cache.device() &&
                  output.scalar_type() == at::kBFloat16 && output.dim() == 3 &&
                  output.size(0) == seq_lens.numel() && output.size(2) == 512 &&
                  output.stride(2) == 1 && offset >= 0 &&
                  offset <= output.size(1),
              "dequant_gather_nvfp4 requires BF16 output[requests, rows, 512]");
  if (gather_lens) {
    TORCH_CHECK(gather_lens->device() == cache.device() &&
                    gather_lens->scalar_type() == at::kInt &&
                    gather_lens->sizes() == seq_lens.sizes() &&
                    gather_lens->is_contiguous(),
                "dequant_gather_nvfp4 requires matching int32 gather lengths");
  }
  if (seq_lens.numel() == 0 || output.size(1) == offset) return;
  const c10::cuda::CUDAGuard guard(cache.device());
  const dim3 grid(seq_lens.numel(),
                  std::min<int64_t>(output.size(1) - offset, 2048));
  dequant_gather_nvfp4_kernel<<<grid, 256, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      cache.data_ptr<uint8_t>(), output.data_ptr<c10::BFloat16>(),
      seq_lens.data_ptr<int32_t>(),
      gather_lens ? gather_lens->data_ptr<int32_t>() : nullptr,
      block_table.data_ptr<int32_t>(), block_table.size(1), cache.stride(0),
      cache.size(0), block_size, output.stride(0), output.stride(1),
      output.size(1), offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
