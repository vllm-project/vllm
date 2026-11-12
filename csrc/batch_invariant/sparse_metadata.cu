// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include <climits>
#include <initializer_list>
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

// A causal decode row's compressed top-k length from its position, as
// combine_topk_swa_indices gives prefill rows; padding rows get 0.
__global__ void topk_lens_by_position_kernel(const int32_t* token_to_req,
                                             const bool* is_valid,
                                             const int32_t* query_start_loc,
                                             const int32_t* seq_lens,
                                             int64_t compress_ratio,
                                             int64_t top_k, int64_t num_tokens,
                                             int32_t* lengths) {
  const int64_t token = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= num_tokens) return;
  int64_t length = 0;
  if (is_valid[token]) {
    const int64_t request = token_to_req[token];
    const int64_t query_start = query_start_loc[request];
    const int64_t query_len = query_start_loc[request + 1] - query_start;
    const int64_t pos = seq_lens[request] - query_len + token - query_start;
    length = (pos + 1) / compress_ratio;
    length = length < top_k ? length : top_k;
    length = length > 0 ? length : 0;
  }
  lengths[token] = static_cast<int32_t>(length);
}

// One block per request: its kept rows are the last ones before its boundary
// and land at that boundary minus the rows trimmed before it; with
// prev_dropped they also get their index among the previous stage's rows.
__global__ void gather_replay_batch_kernel(
    const int32_t* query_start_loc, const int32_t* dropped_before,
    const int32_t* seq_lens, const int32_t* replay_start,
    const int64_t* positions, const int64_t* slot_mappings,
    int64_t slot_mappings_stride, int64_t num_groups, int64_t kept_rows,
    const int32_t* prev_dropped_before, int64_t* rows,
    int32_t* kept_query_start_loc, int64_t* kept_positions,
    int64_t* kept_slot_mappings, int64_t kept_slot_mappings_stride,
    int32_t* kept_kv_start, int64_t* select) {
  const int64_t request = blockIdx.x;
  const int64_t begin = query_start_loc[request];
  const int64_t end = query_start_loc[request + 1];
  const int64_t kept_begin = begin - dropped_before[request];
  const int64_t kept_end = end - dropped_before[request + 1];
  if (threadIdx.x == 0) {
    if (request == 0)
      kept_query_start_loc[0] = static_cast<int32_t>(kept_begin);
    kept_query_start_loc[request + 1] = static_cast<int32_t>(kept_end);
    // A trimmed request holds no replay-layer window KV below its kept rows.
    int64_t kv_start = replay_start[request];
    if (kept_end - kept_begin < end - begin) {
      const int64_t first_kv = seq_lens[request] - kept_rows;
      kv_start = kv_start > first_kv ? kv_start : first_kv;
    }
    kept_kv_start[request] = static_cast<int32_t>(kv_start);
  }
  const int64_t first_row = end - (kept_end - kept_begin);
  const int64_t prev =
      prev_dropped_before != nullptr ? prev_dropped_before[request + 1] : 0;
  for (int64_t token = kept_begin + threadIdx.x; token < kept_end;
       token += blockDim.x) {
    const int64_t row = first_row + (token - kept_begin);
    rows[token] = row;
    if (select != nullptr) select[token] = row - prev;
    kept_positions[token] = positions[row];
    for (int64_t group = 0; group < num_groups; ++group) {
      kept_slot_mappings[group * kept_slot_mappings_stride + token] =
          slot_mappings[group * slot_mappings_stride + row];
    }
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

void topk_lens_by_position(const torch::Tensor& token_to_req,
                           const torch::Tensor& is_valid,
                           const torch::Tensor& query_start_loc,
                           const torch::Tensor& seq_lens,
                           int64_t compress_ratio, int64_t top_k,
                           torch::Tensor& lengths) {
  const auto device = lengths.device();
  const int64_t num_tokens = lengths.numel();
  TORCH_CHECK(lengths.is_cuda() && lengths.dim() == 1 &&
                  lengths.scalar_type() == at::kInt && lengths.is_contiguous(),
              "topk_lens_by_position requires contiguous int32 lengths[T]");
  TORCH_CHECK(
      token_to_req.device() == device &&
          token_to_req.scalar_type() == at::kInt &&
          token_to_req.is_contiguous() && token_to_req.numel() >= num_tokens &&
          is_valid.device() == device && is_valid.scalar_type() == at::kBool &&
          is_valid.is_contiguous() && is_valid.numel() >= num_tokens,
      "topk_lens_by_position requires int32 token_to_req and bool "
      "is_valid covering the tokens");
  for (const auto* t : {&query_start_loc, &seq_lens}) {
    TORCH_CHECK(t->device() == device && t->dim() == 1 &&
                    t->scalar_type() == at::kInt && t->is_contiguous(),
                "topk_lens_by_position requires contiguous int32 metadata");
  }
  TORCH_CHECK(compress_ratio > 0 && top_k >= 0,
              "topk_lens_by_position needs compress_ratio > 0, top_k >= 0");
  if (num_tokens == 0) return;
  const c10::cuda::CUDAGuard guard(device);
  constexpr int kThreads = 256;
  const int64_t blocks = (num_tokens + kThreads - 1) / kThreads;
  topk_lens_by_position_kernel<<<blocks, kThreads, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      token_to_req.data_ptr<int32_t>(), is_valid.data_ptr<bool>(),
      query_start_loc.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      compress_ratio, top_k, num_tokens, lengths.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_replay_batch(
    const torch::Tensor& query_start_loc, const torch::Tensor& dropped_before,
    const torch::Tensor& seq_lens, const torch::Tensor& replay_start,
    const torch::Tensor& positions, const torch::Tensor& slot_mappings,
    int64_t num_reqs, int64_t kept_rows,
    const std::optional<torch::Tensor>& prev_dropped_before,
    torch::Tensor& rows, torch::Tensor& kept_query_start_loc,
    torch::Tensor& kept_positions, torch::Tensor& kept_slot_mappings,
    torch::Tensor& kept_kv_start, const std::optional<torch::Tensor>& select) {
  const auto device = positions.device();
  const auto int32_vector = [&](const torch::Tensor& t, int64_t n) {
    return t.is_cuda() && t.dim() == 1 && t.scalar_type() == at::kInt &&
           t.is_contiguous() && t.numel() >= n;
  };
  TORCH_CHECK(num_reqs >= 0 && int32_vector(query_start_loc, num_reqs + 1) &&
                  int32_vector(dropped_before, num_reqs + 1) &&
                  int32_vector(seq_lens, num_reqs) &&
                  int32_vector(replay_start, num_reqs) &&
                  int32_vector(kept_query_start_loc, num_reqs + 1) &&
                  int32_vector(kept_kv_start, num_reqs),
              "gather_replay_batch requires contiguous int32 per-request "
              "metadata covering num_reqs");
  TORCH_CHECK(
      !prev_dropped_before || int32_vector(*prev_dropped_before, num_reqs + 1),
      "gather_replay_batch requires int32 prev_dropped_before[R+1]");
  TORCH_CHECK(prev_dropped_before.has_value() == select.has_value(),
              "gather_replay_batch takes prev_dropped_before and select "
              "together");
  for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{
           &positions, &rows, &kept_positions}) {
    TORCH_CHECK(t->device() == device && t->dim() == 1 &&
                    t->scalar_type() == at::kLong && t->is_contiguous(),
                "gather_replay_batch requires contiguous int64 rows/positions");
  }
  if (select) {
    TORCH_CHECK(select->device() == device && select->dim() == 1 &&
                    select->scalar_type() == at::kLong &&
                    select->is_contiguous() && select->numel() >= rows.numel(),
                "gather_replay_batch requires contiguous int64 select");
  }
  TORCH_CHECK(slot_mappings.device() == device && slot_mappings.dim() == 2 &&
                  slot_mappings.scalar_type() == at::kLong &&
                  slot_mappings.stride(1) == 1 &&
                  kept_slot_mappings.device() == device &&
                  kept_slot_mappings.dim() == 2 &&
                  kept_slot_mappings.scalar_type() == at::kLong &&
                  kept_slot_mappings.stride(1) == 1 &&
                  kept_slot_mappings.size(0) == slot_mappings.size(0),
              "gather_replay_batch requires row-major int64 slot mappings "
              "[groups, tokens]");
  if (num_reqs == 0) return;
  const c10::cuda::CUDAGuard guard(device);
  gather_replay_batch_kernel<<<num_reqs, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      query_start_loc.data_ptr<int32_t>(), dropped_before.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), replay_start.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(), slot_mappings.data_ptr<int64_t>(),
      slot_mappings.stride(0), slot_mappings.size(0), kept_rows,
      prev_dropped_before ? prev_dropped_before->data_ptr<int32_t>() : nullptr,
      rows.data_ptr<int64_t>(), kept_query_start_loc.data_ptr<int32_t>(),
      kept_positions.data_ptr<int64_t>(),
      kept_slot_mappings.data_ptr<int64_t>(), kept_slot_mappings.stride(0),
      kept_kv_start.data_ptr<int32_t>(),
      select ? select->data_ptr<int64_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
