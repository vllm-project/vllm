// SPDX-License-Identifier: Apache-2.0
//
// Fused rotary embedding (undo-then-redo) for Intel XPU.
//
// Algorithm
// ---------
//   For each token, "undo" the rotary embedding that was applied at
//   old_positions[token] and "redo" the rotary embedding at
//   new_positions[token].  Operates in-place on `key`.
//
// Intel XPU mapping
// -----------------
//   - 1-D nd_range, one work-group per token, work-group size =
//     min(num_kv_heads * embed_dim, MAX_WG_SIZE).
//   - sub_group_size locked to 16 (Intel native SIMD).
//   - The cos/sin cache rows for both old and new positions are loaded
//     once per token via cached global reads.  rot_dim is typically <=
//     256 floats so register/L1 pressure is fine.
//   - IS_NEOX is a template parameter so the IGC can specialise and drop
//     the run-time test inside the hot loop.
//
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <sycl/sycl.hpp>
#pragma GCC diagnostic pop

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include "cachegen_kernels_sycl.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int INTEL_SUB_GROUP_SIZE = 16;
constexpr int MAX_WG_SIZE = 512;
// Upper bound on rot_dim for SLM caching of (old_cos|old_sin|new_cos|new_sin).
// 4 * 256 * sizeof(fp32) = 4 KB SLM per WG -- well under PVC's 64 KB/sub-slice
// budget and lets multiple WGs co-reside.
constexpr int ROPE_MAX_ROT_DIM = 256;

template <typename scalar_t, bool IS_NEOX>
void rotary_embedding_kernel_sycl(sycl::queue& queue,
                                  const int64_t* old_positions,
                                  const int64_t* new_positions, scalar_t* key,
                                  const scalar_t* cos_sin_cache, int rot_dim,
                                  int64_t key_stride, int num_kv_heads,
                                  int head_size, int num_tokens) {
  // Total work items per token (one per (head, rot_pair)).
  const int embed_dim = rot_dim / 2;
  const int wg_size = std::min<int>(num_kv_heads * embed_dim, MAX_WG_SIZE);
  if (wg_size <= 0 || num_tokens <= 0) return;

  sycl::range<1> global_range(static_cast<size_t>(num_tokens) *
                              static_cast<size_t>(wg_size));
  sycl::range<1> local_range(static_cast<size_t>(wg_size));

  queue.submit([&](sycl::handler& cgh) {
    // Cache the per-token cos/sin rows in SLM so the inner loop hits SLM
    // (latency ~3-5 cycles on Xe) instead of HBM for every rot_offset.
    sycl::local_accessor<scalar_t, 1> cs_shared(
        sycl::range<1>(4 * ROPE_MAX_ROT_DIM), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(INTEL_SUB_GROUP_SIZE)]] {
              const int token_idx = static_cast<int>(item.get_group(0));
              const int tid = static_cast<int>(item.get_local_id(0));
              const int nthreads = static_cast<int>(item.get_local_range(0));

              const int64_t old_pos = old_positions[token_idx];
              const int64_t new_pos = new_positions[token_idx];

              const scalar_t* old_cache = cos_sin_cache + old_pos * rot_dim;
              const scalar_t* new_cache = cos_sin_cache + new_pos * rot_dim;

              // Stage [old_cos | old_sin | new_cos | new_sin], each of
              // length embed_dim, into SLM in one coalesced sweep.
              scalar_t* sl_old_cos = &cs_shared[0];
              scalar_t* sl_old_sin = &cs_shared[embed_dim];
              scalar_t* sl_new_cos = &cs_shared[2 * embed_dim];
              scalar_t* sl_new_sin = &cs_shared[3 * embed_dim];
              for (int j = tid; j < embed_dim; j += nthreads) {
                sl_old_cos[j] = old_cache[j];
                sl_old_sin[j] = old_cache[embed_dim + j];
                sl_new_cos[j] = new_cache[j];
                sl_new_sin[j] = new_cache[embed_dim + j];
              }
              item.barrier(sycl::access::fence_space::local_space);

              const int nk = num_kv_heads * embed_dim;
              for (int i = tid; i < nk; i += nthreads) {
                const int head_idx = i / embed_dim;
                const int rot_offset = i % embed_dim;
                const int64_t token_head =
                    static_cast<int64_t>(token_idx) * key_stride +
                    static_cast<int64_t>(head_idx) * head_size;
                scalar_t* arr = key + token_head;

                int x_index, y_index;
                if constexpr (IS_NEOX) {
                  x_index = rot_offset;
                  y_index = embed_dim + rot_offset;
                } else {
                  x_index = 2 * rot_offset;
                  y_index = 2 * rot_offset + 1;
                }
                const scalar_t oc = sl_old_cos[rot_offset];
                const scalar_t os_ = sl_old_sin[rot_offset];
                const scalar_t nc = sl_new_cos[rot_offset];
                const scalar_t ns = sl_new_sin[rot_offset];

                const scalar_t x = arr[x_index];
                const scalar_t y = arr[y_index];

                const scalar_t x_rev = x * oc + y * os_;
                const scalar_t y_rev = y * oc - x * os_;

                arr[x_index] = x_rev * nc - y_rev * ns;
                arr[y_index] = y_rev * nc + x_rev * ns;
              }
            });
  });
}

template <typename scalar_t>
void dispatch_neox(sycl::queue& queue, const int64_t* old_pos,
                   const int64_t* new_pos, scalar_t* key,
                   const scalar_t* cos_sin_cache, int rot_dim,
                   int64_t key_stride, int num_kv_heads, int head_size,
                   int num_tokens, bool is_neox) {
  if (is_neox) {
    rotary_embedding_kernel_sycl<scalar_t, true>(
        queue, old_pos, new_pos, key, cos_sin_cache, rot_dim, key_stride,
        num_kv_heads, head_size, num_tokens);
  } else {
    rotary_embedding_kernel_sycl<scalar_t, false>(
        queue, old_pos, new_pos, key, cos_sin_cache, rot_dim, key_stride,
        num_kv_heads, head_size, num_tokens);
  }
}

}  // namespace

void rotary_embedding_k_fused_xpu(const torch::Tensor& old_positions,
                                  const torch::Tensor& new_positions,
                                  torch::Tensor& key, int64_t head_size,
                                  const torch::Tensor& cos_sin_cache,
                                  bool is_neox) {
  if (!key.device().is_xpu()) {
    throw std::runtime_error(
        "rotary_embedding_k_fused_xpu: key must be an XPU tensor");
  }
  if (!old_positions.device().is_xpu() || !new_positions.device().is_xpu() ||
      !cos_sin_cache.device().is_xpu()) {
    throw std::runtime_error(
        "rotary_embedding_k_fused_xpu: all tensors must be on XPU");
  }
  if (old_positions.scalar_type() != at::kLong ||
      new_positions.scalar_type() != at::kLong) {
    throw std::runtime_error(
        "rotary_embedding_k_fused_xpu: positions must be int64");
  }
  if (cos_sin_cache.dim() != 2) {
    throw std::runtime_error(
        "rotary_embedding_k_fused_xpu: cos_sin_cache must be 2-D");
  }
  if (cos_sin_cache.size(1) > ROPE_MAX_ROT_DIM) {
    throw std::runtime_error(
        "rotary_embedding_k_fused_xpu: rot_dim exceeds ROPE_MAX_ROT_DIM (256)");
  }

  const int64_t num_tokens = key.numel() / (key.size(-1) * key.size(-2));
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  const int num_kv_heads = static_cast<int>(key.size(-2));
  const int64_t key_stride = static_cast<int64_t>(num_kv_heads) * head_size;

  const c10::DeviceGuard guard(key.device());
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(key.device().index()).queue();

  // Mirror LMC_DISPATCH_FLOATING_TYPES.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(),
      "rotary_embedding_k_fused_xpu", [&] {
        dispatch_neox<scalar_t>(
            queue, old_positions.data_ptr<int64_t>(),
            new_positions.data_ptr<int64_t>(), key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(), rot_dim, key_stride,
            num_kv_heads, static_cast<int>(head_size),
            static_cast<int>(num_tokens), is_neox);
      });
}
