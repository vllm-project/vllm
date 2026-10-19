// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Per-token gather for the DS4.1 batch-invariant sparse decode path.
//
// The eager BI prefill path dequantizes whole requests into a chunk workspace
// and lets combine_topk_swa_indices point each query at its rows: top-k
// compressed rows first (holes stay -1), then its SWA window in position
// order, with combined_len = topk_len + swa_len. This kernel builds the same
// ordered row list for each token directly into a fixed [tokens, width, 512]
// workspace, so FlashMLA's sparse prefill kernel sees identical rows, order,
// holes and lengths while every shape depends only on the token count.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <optional>

namespace vllm::batch_invariant {
namespace {

constexpr int kHeadDim = 512;
constexpr int kMxfp8Record = 528;  // 512 e4m3, then 16 UE8M0 scales of 32.
constexpr int kNvfp4Record = 288;  // 256 packed e2m1, then 32 e4m3 of 16.
constexpr int kSlotsPerBlock = 8;  // One warp per gathered row.

__device__ __forceinline__ int64_t imax(int64_t a, int64_t b) {
  return a > b ? a : b;
}
__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) {
  return a < b ? a : b;
}

// Matches the CuteDSL FP8 gather: exact e4m3 -> bf16, then a bf16 multiply by
// the power-of-two scale built from its exponent bits.
__device__ __forceinline__ void dequant_mxfp8_lane(const uint8_t* record,
                                                   int lane,
                                                   __nv_bfloat16* out) {
  const uint16_t scale_bits = uint16_t(record[kHeadDim + lane / 2]) << 7;
  const __nv_bfloat16 scale = __ushort_as_bfloat16(scale_bits);
  const uint8_t* data = record + lane * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float value = static_cast<float>(
        c10::Float8_e4m3fn(data[i], c10::Float8_e4m3fn::from_bits()));
    out[i] = __hmul(__float2bfloat16_rn(value), scale);
  }
}

// Matches dequant_gather_nvfp4: FP32 product, then RNE to bf16.
__device__ __forceinline__ void dequant_nvfp4_lane(const uint8_t* record,
                                                   int lane,
                                                   __nv_bfloat16* out) {
  constexpr float magnitudes[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float scale = static_cast<float>(
      c10::Float8_e4m3fn(record[256 + lane], c10::Float8_e4m3fn::from_bits()));
  const uint8_t* packed = record + lane * 8;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint8_t code = (packed[i / 2] >> ((i & 1) * 4)) & 15;
    const float value = code & 8 ? -magnitudes[code & 7] : magnitudes[code & 7];
    out[i] = __float2bfloat16_rn(value * scale);
  }
}

__device__ __forceinline__ void store_row_lane(__nv_bfloat16* row, int lane,
                                               const __nv_bfloat16* values) {
  uint4* dst = reinterpret_cast<uint4*>(row + lane * 16);
  const uint4* src = reinterpret_cast<const uint4*>(values);
  dst[0] = src[0];
  dst[1] = src[1];
}

__global__ void sparse_decode_gather_kernel(
    const int32_t* topk, int64_t topk_stride, const int32_t* token_to_req,
    const bool* is_valid, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* replay_start,
    const uint8_t* compressed, int64_t compressed_pages,
    int64_t compressed_page_stride, int64_t compressed_record_stride,
    int64_t compressed_record, const int32_t* compressed_table,
    int64_t compressed_table_stride, int64_t compressed_table_width,
    int64_t compressed_block_size, const uint8_t* swa, int64_t swa_pages,
    int64_t swa_page_stride, int64_t swa_record_stride,
    const int32_t* swa_table, int64_t swa_table_stride, int64_t swa_table_width,
    int64_t swa_block_size, int64_t token_offset, int64_t window,
    int64_t compress_ratio, int64_t top_k, int64_t width, __nv_bfloat16* kv,
    int32_t* indices, int32_t* lens) {
  const int64_t token = blockIdx.x;
  const int64_t absolute = token_offset + token;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t slot = int64_t(blockIdx.y) * kSlotsPerBlock + warp;

  int64_t topk_len = 0, swa_start = 0, total = 0, request = 0;
  const bool valid = is_valid[absolute];
  if (valid) {
    request = token_to_req[absolute];
    const int64_t query_start = query_start_loc[request];
    const int64_t query_len = query_start_loc[request + 1] - query_start;
    const int64_t seq_len = seq_lens[request];
    const int64_t prefix = seq_len - query_len;
    const int64_t pos = prefix + absolute - query_start;
    if (compress_ratio > 0) {
      topk_len = imax(imin((pos + 1) / compress_ratio, top_k), 0);
    }
    const int64_t visible_prefix = imax(prefix - replay_start[request], 0);
    const int64_t gather_len = query_len + imin(visible_prefix, window - 1);
    swa_start = imax(imax(pos - (window - 1), 0), seq_len - gather_len);
    const int64_t swa_len = imax(pos - swa_start + 1, 0);
    total = topk_len + swa_len;
  }
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    lens[token] = static_cast<int32_t>(total);
  }
  if (slot >= width) return;

  int32_t index = -1;
  const uint8_t* record = nullptr;
  bool fp4 = false;
  if (slot < topk_len) {
    const int32_t local = topk[token * topk_stride + slot];
    const int64_t compressed_len = seq_lens[request] / compress_ratio;
    if (local >= 0 && local < compressed_len) {
      const int64_t block = local / compressed_block_size;
      CUDA_KERNEL_ASSERT(block < compressed_table_width);
      const int32_t page =
          compressed_table[request * compressed_table_stride + block];
      CUDA_KERNEL_ASSERT(page >= 0 && page < compressed_pages);
      record = compressed + int64_t(page) * compressed_page_stride +
               (local % compressed_block_size) * compressed_record_stride;
      fp4 = compressed_record == kNvfp4Record;
    }
  } else if (slot < total) {
    const int64_t position = swa_start + slot - topk_len;
    const int64_t block = position / swa_block_size;
    CUDA_KERNEL_ASSERT(block < swa_table_width);
    const int32_t page = swa_table[request * swa_table_stride + block];
    CUDA_KERNEL_ASSERT(page >= 0 && page < swa_pages);
    record = swa + int64_t(page) * swa_page_stride +
             (position % swa_block_size) * swa_record_stride;
  }
  if (record != nullptr) {
    index = static_cast<int32_t>(token * width + slot);
    __align__(16) __nv_bfloat16 values[16];
    if (fp4) {
      dequant_nvfp4_lane(record, lane, values);
    } else {
      dequant_mxfp8_lane(record, lane, values);
    }
    store_row_lane(kv + (token * width + slot) * kHeadDim, lane, values);
  }
  if (lane == 0) {
    indices[token * width + slot] = index;
  }
}

void check_paged_cache(const torch::Tensor& cache, const torch::Tensor& table,
                       int64_t block_size, const char* name) {
  TORCH_CHECK(
      cache.is_cuda() && cache.scalar_type() == at::kByte && cache.dim() == 3 &&
          cache.size(1) == block_size && cache.stride(2) == 1 &&
          (cache.size(2) == kMxfp8Record || cache.size(2) == kNvfp4Record) &&
          cache.stride(1) >= cache.size(2),
      name, " cache must be paged CUDA uint8 V4.1 records");
  TORCH_CHECK(table.device() == cache.device() && table.dim() == 2 &&
                  table.scalar_type() == at::kInt && table.stride(1) == 1,
              name, " block table must be int32 with unit column stride");
}

}  // namespace

void sparse_decode_gather(
    const torch::Tensor& topk, const torch::Tensor& token_to_req,
    const torch::Tensor& is_valid, const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens, const torch::Tensor& replay_start,
    const std::optional<torch::Tensor>& compressed_cache,
    const std::optional<torch::Tensor>& compressed_table,
    int64_t compressed_block_size, const torch::Tensor& swa_cache,
    const torch::Tensor& swa_table, int64_t swa_block_size,
    int64_t token_offset, int64_t num_tokens, int64_t window,
    int64_t compress_ratio, int64_t top_k, torch::Tensor& kv,
    torch::Tensor& indices, torch::Tensor& lens) {
  const auto device = swa_cache.device();
  check_paged_cache(swa_cache, swa_table, swa_block_size, "SWA");
  TORCH_CHECK(swa_cache.size(2) == kMxfp8Record,
              "SWA cache must hold MXFP8 records");
  TORCH_CHECK(window > 0 && token_offset >= 0 && num_tokens >= 0 &&
                  compress_ratio >= 0 && top_k >= 0,
              "invalid sparse decode gather geometry");
  const bool has_compressed = compress_ratio > 0 && top_k > 0;
  if (has_compressed) {
    TORCH_CHECK(compressed_cache && compressed_table,
                "compressed layers need a compressed cache and table");
    check_paged_cache(*compressed_cache, *compressed_table,
                      compressed_block_size, "compressed");
    TORCH_CHECK(topk.device() == device && topk.dim() == 2 &&
                    topk.scalar_type() == at::kInt && topk.stride(1) == 1 &&
                    topk.size(0) >= num_tokens && topk.size(1) >= top_k,
                "top-k indices must be int32 [tokens, >= top_k]");
  }
  for (const torch::Tensor* t :
       {&token_to_req, &query_start_loc, &seq_lens, &replay_start}) {
    TORCH_CHECK(t->device() == device && t->dim() == 1 &&
                    t->scalar_type() == at::kInt && t->is_contiguous(),
                "request metadata must be contiguous int32 vectors");
  }
  TORCH_CHECK(is_valid.device() == device && is_valid.dim() == 1 &&
                  is_valid.scalar_type() == at::kBool &&
                  is_valid.is_contiguous() &&
                  is_valid.numel() >= token_offset + num_tokens &&
                  token_to_req.numel() >= token_offset + num_tokens,
              "token metadata must cover the gathered tokens");
  TORCH_CHECK(kv.device() == device && kv.scalar_type() == at::kBFloat16 &&
                  kv.dim() == 3 && kv.size(2) == kHeadDim &&
                  kv.is_contiguous() && kv.size(0) >= num_tokens,
              "kv workspace must be contiguous BF16 [tokens, width, 512]");
  const int64_t width = kv.size(1);
  TORCH_CHECK(width >= top_k + window,
              "kv workspace width must hold top-k plus the window");
  TORCH_CHECK(indices.device() == device && indices.scalar_type() == at::kInt &&
                  indices.dim() == 2 && indices.is_contiguous() &&
                  indices.size(0) >= num_tokens && indices.size(1) == width,
              "indices must be contiguous int32 [tokens, width]");
  TORCH_CHECK(lens.device() == device && lens.scalar_type() == at::kInt &&
                  lens.dim() == 1 && lens.is_contiguous() &&
                  lens.numel() >= num_tokens,
              "lengths must be contiguous int32 [tokens]");
  if (num_tokens == 0) return;

  const c10::cuda::CUDAGuard guard(device);
  const dim3 grid(num_tokens, (width + kSlotsPerBlock - 1) / kSlotsPerBlock);
  const torch::Tensor* cc = has_compressed ? &*compressed_cache : nullptr;
  const torch::Tensor* ct = has_compressed ? &*compressed_table : nullptr;
  sparse_decode_gather_kernel<<<grid, kSlotsPerBlock * 32, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      has_compressed ? topk.data_ptr<int32_t>() : nullptr,
      has_compressed ? topk.stride(0) : 0, token_to_req.data_ptr<int32_t>(),
      is_valid.data_ptr<bool>(), query_start_loc.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), replay_start.data_ptr<int32_t>(),
      cc ? cc->data_ptr<uint8_t>() : nullptr, cc ? cc->size(0) : 0,
      cc ? cc->stride(0) : 0, cc ? cc->stride(1) : 0, cc ? cc->size(2) : 0,
      ct ? ct->data_ptr<int32_t>() : nullptr, ct ? ct->stride(0) : 0,
      ct ? ct->size(1) : 0, compressed_block_size,
      swa_cache.data_ptr<uint8_t>(), swa_cache.size(0), swa_cache.stride(0),
      swa_cache.stride(1), swa_table.data_ptr<int32_t>(), swa_table.stride(0),
      swa_table.size(1), swa_block_size, token_offset, window, compress_ratio,
      has_compressed ? top_k : 0, width,
      reinterpret_cast<__nv_bfloat16*>(kv.data_ptr<c10::BFloat16>()),
      indices.data_ptr<int32_t>(), lens.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
