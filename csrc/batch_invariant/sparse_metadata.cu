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

constexpr int kSwaFirstWarps = 8;
constexpr int kSwaFirstWorkers = 64;

// One warp per token row of request blockIdx.x. Same positions, lengths and
// ids as combine_topk_swa_indices, laid out in the mega decode kernel's slot
// space: [SWA padded to the window with -1; top-k; -1 up to the width].
__global__ void combine_swa_first_kernel(
    const int32_t* topk, int64_t topk_stride, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* gather_lens, int64_t window,
    int64_t compress_ratio, int64_t top_k, int64_t m, int64_t n,
    int64_t num_rows, int64_t width, int64_t output_stride, int32_t* output,
    int32_t* lengths) {
  const int64_t request = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t base = query_start_loc[0];
  const int64_t query_start = query_start_loc[request] - base;
  const int64_t query_end = query_start_loc[request + 1] - base;
  const int64_t seq_len = seq_lens[request];
  const int64_t gather_start = seq_len - gather_lens[request];
  const int64_t prefix = seq_len - (query_end - query_start);
  const int64_t request_offset = m * request;
  for (int64_t token =
           query_start + int64_t(blockIdx.y) * kSwaFirstWarps + warp;
       token < query_end; token += int64_t(gridDim.y) * kSwaFirstWarps) {
    CUDA_KERNEL_ASSERT(token >= 0 && token < num_rows);
    if (token < 0 || token >= num_rows) continue;
    const int64_t pos = prefix + token - query_start;
    int64_t topk_len = 0;
    if (compress_ratio > 0) {
      topk_len = (pos + 1) / compress_ratio;
      topk_len = topk_len < top_k ? topk_len : top_k;
      topk_len = topk_len > 0 ? topk_len : 0;
    }
    int64_t swa_start = pos - (window - 1);
    swa_start = swa_start > 0 ? swa_start : 0;
    swa_start = swa_start > gather_start ? swa_start : gather_start;
    const int64_t swa_len = pos - swa_start + 1;
    const int64_t swa_id = request_offset + n + swa_start - gather_start;
    const int32_t* row_topk = topk + token * topk_stride;
    int32_t* row = output + token * output_stride;
    for (int64_t slot = lane; slot < width; slot += 32) {
      int64_t value = -1;
      if (slot < window) {
        if (slot < swa_len) value = swa_id + slot;
      } else if (slot - window < topk_len) {
        const int64_t local = row_topk[slot - window];
        if (local >= 0 && local < n) value = local + request_offset;
      }
      row[slot] = static_cast<int32_t>(value);
    }
    if (lane == 0) lengths[token] = static_cast<int32_t>(window + topk_len);
  }
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

void combine_swa_first_indices(const torch::Tensor& topk,
                               const torch::Tensor& query_start_loc,
                               const torch::Tensor& seq_lens,
                               const torch::Tensor& gather_lens, int64_t window,
                               int64_t compress_ratio, int64_t top_k, int64_t m,
                               int64_t n, torch::Tensor& output,
                               torch::Tensor& lengths) {
  const auto device = topk.device();
  TORCH_CHECK(topk.is_cuda() && topk.dim() == 2 &&
                  topk.scalar_type() == at::kInt && topk.stride(1) == 1,
              "combine_swa_first_indices requires row-major int32 topk[T,K]");
  for (const auto* t : {&query_start_loc, &seq_lens, &gather_lens}) {
    TORCH_CHECK(t->device() == device && t->dim() == 1 &&
                    t->scalar_type() == at::kInt && t->is_contiguous(),
                "combine_swa_first_indices requires contiguous int32 metadata");
  }
  TORCH_CHECK(query_start_loc.numel() == seq_lens.numel() + 1 &&
                  gather_lens.numel() == seq_lens.numel(),
              "combine_swa_first_indices requires query_start_loc[R+1], "
              "seq_lens[R] and gather_lens[R]");
  TORCH_CHECK(window > 0 && compress_ratio >= 0 && top_k >= 0 &&
                  top_k <= topk.size(1) && m >= 0 && n >= 0,
              "combine_swa_first_indices has invalid window/ratio/top_k/m/n");
  const int64_t width = (top_k + window + 127) / 128 * 128;
  TORCH_CHECK(output.device() == device && output.dim() == 2 &&
                  output.scalar_type() == at::kInt && output.stride(1) == 1 &&
                  output.size(1) >= width && output.size(0) == topk.size(0),
              "combine_swa_first_indices requires row-major int32 "
              "output[T, >=round_up(top_k + window, 128)]");
  TORCH_CHECK(lengths.device() == device && lengths.dim() == 1 &&
                  lengths.scalar_type() == at::kInt &&
                  lengths.is_contiguous() && lengths.numel() == topk.size(0),
              "combine_swa_first_indices requires contiguous int32 lengths[T]");
  if (seq_lens.numel() == 0 || topk.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(device);
  const dim3 grid(seq_lens.numel(), kSwaFirstWorkers);
  combine_swa_first_kernel<<<grid, kSwaFirstWarps * 32, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      topk.data_ptr<int32_t>(), topk.stride(0),
      query_start_loc.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      gather_lens.data_ptr<int32_t>(), window, compress_ratio, top_k, m, n,
      topk.size(0), width, output.stride(0), output.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
