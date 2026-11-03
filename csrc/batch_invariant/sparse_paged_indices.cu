// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Per-token paged KV indices for the DS4.1 batch-invariant sparse path.
//
// Under batch invariance every causal row, decode or prefill, runs FlashMLA's
// unsplit sparse decode kernel straight on the quantized caches. This kernel
// gives each token its SWA slots and compressed top-k slots in that kernel's
// paged coordinates (page * block_size + offset, -1 for holes) together with
// lengths taken from the token's position alone, so the same (request,
// position) gets the same slot lists and lengths in any batch, chunk or step.
//
// The visible SWA range and the compressed length follow the eager prefill
// path: SWA covers max(pos - window + 1, 0, seq_len - gather_len) .. pos with
// gather_len = query_len + min(max(prefix - replay_start, 0), window - 1), and
// the compressed length is clamp((pos + 1) / compress_ratio, 0, top_k), with
// entries outside [0, seq_len / compress_ratio) as holes.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <optional>

namespace vllm::batch_invariant {
namespace {

constexpr int64_t kMaxThreads = 1024;

__device__ __forceinline__ int64_t imax(int64_t a, int64_t b) {
  return a > b ? a : b;
}
__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) {
  return a < b ? a : b;
}

__global__ void sparse_paged_indices_kernel(
    const int32_t* topk, int64_t topk_stride, const int32_t* token_to_req,
    const bool* is_valid, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* replay_start,
    const int32_t* compressed_table, int64_t compressed_table_stride,
    int64_t compressed_table_width, int64_t compressed_block_size,
    const int32_t* swa_table, int64_t swa_table_stride, int64_t swa_table_width,
    int64_t swa_block_size, int64_t token_offset, int64_t window,
    int64_t compress_ratio, int64_t top_k, int32_t* swa_indices,
    int64_t swa_stride, int32_t* swa_lens, int32_t* extra_indices,
    int64_t extra_stride, int32_t* extra_lens) {
  const int64_t row = blockIdx.x;
  const int64_t token = token_offset + row;

  int64_t request = 0, topk_len = 0, swa_start = 0, swa_len = 0;
  int64_t compressed_len = 0;
  if (is_valid[token]) {
    request = token_to_req[token];
    const int64_t query_start = query_start_loc[request];
    const int64_t query_len = query_start_loc[request + 1] - query_start;
    const int64_t seq_len = seq_lens[request];
    const int64_t prefix = seq_len - query_len;
    const int64_t pos = prefix + token - query_start;
    if (compress_ratio > 0) {
      topk_len = imax(imin((pos + 1) / compress_ratio, top_k), 0);
      compressed_len = seq_len / compress_ratio;
    }
    const int64_t visible_prefix = imax(prefix - replay_start[request], 0);
    const int64_t gather_len = query_len + imin(visible_prefix, window - 1);
    swa_start = imax(imax(pos - (window - 1), 0), seq_len - gather_len);
    swa_len = imax(pos - swa_start + 1, 0);
  }
  if (threadIdx.x == 0) {
    swa_lens[row] = static_cast<int32_t>(swa_len);
    if (extra_lens != nullptr) extra_lens[row] = static_cast<int32_t>(topk_len);
  }

  int32_t* swa_row = swa_indices + row * swa_stride;
  for (int64_t slot = threadIdx.x; slot < window; slot += blockDim.x) {
    int64_t value = -1;
    if (slot < swa_len) {
      const int64_t position = swa_start + slot;
      const int64_t block = position / swa_block_size;
      CUDA_KERNEL_ASSERT(block < swa_table_width);
      const int64_t page = swa_table[request * swa_table_stride + block];
      CUDA_KERNEL_ASSERT(page >= 0);
      value = page * swa_block_size + position % swa_block_size;
    }
    swa_row[slot] = static_cast<int32_t>(value);
  }
  if (extra_indices == nullptr) return;

  int32_t* extra_row = extra_indices + row * extra_stride;
  for (int64_t slot = threadIdx.x; slot < top_k; slot += blockDim.x) {
    int64_t value = -1;
    if (slot < topk_len) {
      const int64_t local = topk[token * topk_stride + slot];
      if (local >= 0 && local < compressed_len) {
        const int64_t block = local / compressed_block_size;
        CUDA_KERNEL_ASSERT(block < compressed_table_width);
        const int64_t page =
            compressed_table[request * compressed_table_stride + block];
        CUDA_KERNEL_ASSERT(page >= 0);
        value = page * compressed_block_size + local % compressed_block_size;
      }
    }
    extra_row[slot] = static_cast<int32_t>(value);
  }
}

void check_int32_rows(const torch::Tensor& t, const torch::Device& device,
                      int64_t rows, int64_t width, const char* name) {
  TORCH_CHECK(
      t.device() == device && t.scalar_type() == at::kInt && t.dim() == 2 &&
          t.stride(1) == 1 && t.size(0) >= rows && t.size(1) == width,
      name, " must be int32 [tokens, ", width, "] with unit column stride");
}

void check_int32_vector(const torch::Tensor& t, const torch::Device& device,
                        int64_t numel, const char* name) {
  TORCH_CHECK(t.device() == device && t.dim() == 1 &&
                  t.scalar_type() == at::kInt && t.is_contiguous() &&
                  t.numel() >= numel,
              name, " must be a contiguous int32 vector of at least ", numel);
}

}  // namespace

void sparse_paged_indices(
    const std::optional<torch::Tensor>& topk, const torch::Tensor& token_to_req,
    const torch::Tensor& is_valid, const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens, const torch::Tensor& replay_start,
    const std::optional<torch::Tensor>& compressed_table,
    int64_t compressed_block_size, const torch::Tensor& swa_table,
    int64_t swa_block_size, int64_t token_offset, int64_t num_tokens,
    int64_t window, int64_t compress_ratio, int64_t top_k,
    torch::Tensor& swa_indices, torch::Tensor& swa_lens,
    const std::optional<torch::Tensor>& extra_indices,
    const std::optional<torch::Tensor>& extra_lens) {
  const auto device = swa_indices.device();
  TORCH_CHECK(window > 0 && swa_block_size > 0 && token_offset >= 0 &&
                  num_tokens >= 0 && compress_ratio >= 0 && top_k >= 0,
              "invalid sparse paged index geometry");
  check_int32_rows(swa_indices, device, num_tokens, window, "SWA indices");
  check_int32_vector(swa_lens, device, num_tokens, "SWA lengths");
  TORCH_CHECK(swa_table.device() == device && swa_table.dim() == 2 &&
                  swa_table.scalar_type() == at::kInt &&
                  swa_table.stride(1) == 1,
              "SWA block table must be int32 with unit column stride");
  const bool has_compressed = compress_ratio > 0 && top_k > 0;
  TORCH_CHECK(has_compressed == extra_indices.has_value() &&
                  has_compressed == extra_lens.has_value() &&
                  has_compressed == compressed_table.has_value() &&
                  has_compressed == topk.has_value(),
              "compressed layers take top-k, a compressed table and extra "
              "outputs; SWA-only layers take none");
  if (has_compressed) {
    TORCH_CHECK(compressed_block_size > 0,
                "compressed block size must be positive");
    TORCH_CHECK(topk->device() == device && topk->dim() == 2 &&
                    topk->scalar_type() == at::kInt && topk->stride(1) == 1 &&
                    topk->size(0) >= token_offset + num_tokens &&
                    topk->size(1) >= top_k,
                "top-k indices must be int32 [tokens, >= top_k]");
    TORCH_CHECK(compressed_table->device() == device &&
                    compressed_table->dim() == 2 &&
                    compressed_table->scalar_type() == at::kInt &&
                    compressed_table->stride(1) == 1,
                "compressed block table must be int32 with unit column stride");
    check_int32_rows(*extra_indices, device, num_tokens, top_k,
                     "extra indices");
    check_int32_vector(*extra_lens, device, num_tokens, "extra lengths");
  }
  for (const torch::Tensor* t : {&query_start_loc, &seq_lens, &replay_start}) {
    check_int32_vector(*t, device, 1, "request metadata");
  }
  check_int32_vector(token_to_req, device, token_offset + num_tokens,
                     "token_to_req");
  TORCH_CHECK(is_valid.device() == device && is_valid.dim() == 1 &&
                  is_valid.scalar_type() == at::kBool &&
                  is_valid.is_contiguous() &&
                  is_valid.numel() >= token_offset + num_tokens,
              "is_valid must be a contiguous bool vector covering the tokens");
  TORCH_CHECK(num_tokens <= INT_MAX, "too many tokens for one launch");
  if (num_tokens == 0) return;

  const c10::cuda::CUDAGuard guard(device);
  // One thread per slot where possible: the row's slots are independent
  // chains of dependent loads, so width bounds the latency, not the count.
  int64_t widest = window > top_k ? window : top_k;
  widest = widest < kMaxThreads ? widest : kMaxThreads;
  const int64_t threads = (widest + 31) / 32 * 32;
  sparse_paged_indices_kernel<<<num_tokens, threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      has_compressed ? topk->data_ptr<int32_t>() : nullptr,
      has_compressed ? topk->stride(0) : 0, token_to_req.data_ptr<int32_t>(),
      is_valid.data_ptr<bool>(), query_start_loc.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), replay_start.data_ptr<int32_t>(),
      has_compressed ? compressed_table->data_ptr<int32_t>() : nullptr,
      has_compressed ? compressed_table->stride(0) : 0,
      has_compressed ? compressed_table->size(1) : 0, compressed_block_size,
      swa_table.data_ptr<int32_t>(), swa_table.stride(0), swa_table.size(1),
      swa_block_size, token_offset, window, compress_ratio,
      has_compressed ? top_k : 0, swa_indices.data_ptr<int32_t>(),
      swa_indices.stride(0), swa_lens.data_ptr<int32_t>(),
      has_compressed ? extra_indices->data_ptr<int32_t>() : nullptr,
      has_compressed ? extra_indices->stride(0) : 0,
      has_compressed ? extra_lens->data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
