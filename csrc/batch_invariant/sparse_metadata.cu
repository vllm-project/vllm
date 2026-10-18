// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include <climits>
#include <optional>

namespace vllm::batch_invariant {
namespace {

template <typename Index>
__global__ void compact_kernel(const bool* mask, const Index* main,
                               int32_t* output, int32_t* lengths, int64_t rows,
                               int64_t main_size, int64_t window, int64_t topk,
                               int64_t width, int64_t ms0, int64_t ms1,
                               int64_t ms2, int64_t is0, int64_t is1,
                               int64_t is2) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage storage;
  const int64_t row = blockIdx.x;
  const int64_t batch = row / rows, token = row % rows;
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    output[row * width + column] = -1;
  }
  __syncthreads();
  int count = 0;
  for (int part = 0; part < 2; ++part) {
    const int64_t limit = part == 0 ? topk : window;
    for (int64_t tile = 0; tile < limit; tile += blockDim.x) {
      const int64_t slot = tile + threadIdx.x;
      int64_t value = -1;
      bool valid = false;
      if (slot < limit) {
        if (part == 0) {
          value = main[batch * is0 + token * is1 + slot * is2];
          valid = value >= 0 && value < main_size;
        } else {
          value = main_size + slot;
          valid = mask[batch * ms0 + token * ms1 + slot * ms2];
        }
      }
      int prefix, total;
      Scan(storage).ExclusiveSum(int(valid), prefix, total);
      if (valid && count + prefix < width) {
        output[row * width + count + prefix] = int32_t(value);
      }
      count += total;
      __syncthreads();
    }
  }
  CUDA_KERNEL_ASSERT(count <= width);
  if (threadIdx.x == 0) lengths[row] = count;
}

}  // namespace

void sparse_metadata(const torch::Tensor& mask,
                     const std::optional<torch::Tensor>& main,
                     int64_t main_size, int64_t window, torch::Tensor& output,
                     torch::Tensor& lengths) {
  TORCH_CHECK(
      mask.is_cuda() && mask.scalar_type() == at::kBool && mask.dim() == 3,
      "sparse_metadata requires CUDA bool mask[B,S,N]");
  TORCH_CHECK(main_size >= 0 && window >= 0 && window <= mask.size(2) &&
                  main_size + window <= INT_MAX,
              "sparse_metadata has invalid main/window sizes");
  TORCH_CHECK(output.device() == mask.device() && output.dim() == 3 &&
                  output.scalar_type() == at::kInt && output.is_contiguous() &&
                  output.size(0) == mask.size(0) &&
                  output.size(1) == mask.size(1) && output.size(2) > 0,
              "sparse_metadata requires contiguous int32 output[B,S,W]");
  TORCH_CHECK(lengths.device() == mask.device() && lengths.dim() == 2 &&
                  lengths.scalar_type() == at::kInt &&
                  lengths.is_contiguous() && lengths.size(0) == mask.size(0) &&
                  lengths.size(1) == mask.size(1),
              "sparse_metadata requires contiguous int32 lengths[B,S]");
  if (main) {
    TORCH_CHECK(main->device() == mask.device() && main->dim() == 3 &&
                    main->size(0) == mask.size(0) &&
                    main->size(1) == mask.size(1) &&
                    (main->scalar_type() == at::kInt ||
                     main->scalar_type() == at::kLong),
                "sparse_metadata requires int32/int64 main[B,S,K]");
  }
  if (lengths.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(mask.device());
  const int64_t topk = main ? main->size(2) : 0;
  const int64_t is0 = main ? main->stride(0) : 0;
  const int64_t is1 = main ? main->stride(1) : 0;
  const int64_t is2 = main ? main->stride(2) : 0;
  const auto launch = [&]<typename Index>() {
    compact_kernel<<<lengths.numel(), 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        mask.data_ptr<bool>(), main ? main->data_ptr<Index>() : nullptr,
        output.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), mask.size(1),
        main_size, window, topk, output.size(2), mask.stride(0), mask.stride(1),
        mask.stride(2), is0, is1, is2);
  };
  if (main && main->scalar_type() == at::kLong) {
    launch.operator()<int64_t>();
  } else {
    launch.operator()<int32_t>();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
