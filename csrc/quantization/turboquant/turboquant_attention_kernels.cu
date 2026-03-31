// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused PagedAttention kernels for TurboQuant KV cache.
//
// These kernels extend vLLM's PagedAttention v2 to dequantize KV cache entries
// on-the-fly during the dot-product computation. The polar decode and optional
// QJL bias correction are performed in registers/shared memory — full-precision
// KV tensors are never materialized in HBM.
//
// Reference: Zandieh et al., "TurboQuant: Redefining AI Efficiency with
// Extreme Compression", ICLR 2026 (arXiv:2504.19874)

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "turboquant_utils.cuh"

using namespace vllm::turboquant;

// ============================================================================
// Fused Paged Attention with TurboQuant dequantization
// ============================================================================
//
// This kernel performs single-query attention (one query per sequence) with
// on-the-fly dequantization of the KV cache from TurboQuant format.
//
// Layout assumptions:
//   query:       [num_seqs, num_heads, head_size]
//   key_cache:   [num_blocks, block_bytes_k]  (packed TurboQuant)
//   value_cache: [num_blocks, block_bytes_v]  (packed TurboQuant)
//   block_tables: [num_seqs, max_num_blocks_per_seq]
//   context_lens: [num_seqs]
//
// Each warp handles one query head. We iterate over KV blocks, dequantize
// each KV token, compute QK dot products, and accumulate the softmax-weighted
// value output.

template <TQDataType DT>
__global__ void paged_attention_turboquant_kernel(
    float* __restrict__ output,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ query,       // [num_seqs, num_heads, head_size]
    const uint8_t* __restrict__ key_cache, // [num_blocks, block_bytes]
    const uint8_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    float scale,                           // attention scale (1/sqrt(d))
    int num_heads, int num_kv_heads, int head_size, int block_size,
    int max_blocks_per_seq, uint32_t layer_seed, int qjl_proj_dim) {
  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  // GQA: map query head to KV head
  int kv_head_idx = head_idx / (num_heads / num_kv_heads);

  int context_len = context_lens[seq_idx];
  if (context_len == 0) return;

  int num_blocks = (context_len + block_size - 1) / block_size;

  // Load query vector into registers
  float q[MAX_HEAD_SIZE];
  const float* q_ptr =
      query + (seq_idx * num_heads + head_idx) * head_size;
  for (int i = 0; i < head_size; i++) {
    q[i] = q_ptr[i];
  }

  // Block layout sizes (using padded alignment from turboquant_utils.cuh)
  constexpr int BITS = angle_bits(DT);
  int angle_bytes = padded_angle_bytes(head_size, BITS);
  int radius_bytes = 2;
  int qjl_bytes = has_qjl(DT) ? (qjl_proj_dim + 7) / 8 : 0;
  int residual_norm_bytes = has_qjl(DT) ? 2 : 0;
  int bpt_per_head = angle_bytes + radius_bytes + qjl_bytes +
                     residual_norm_bytes;
  int bytes_per_token = num_kv_heads * bpt_per_head;
  int block_total_bytes = block_size * bytes_per_token;

  // Per-head seeds
  uint32_t rotation_seed = derive_rotation_seed(layer_seed, kv_head_idx);
  uint32_t qjl_seed = derive_qjl_seed(layer_seed, kv_head_idx);

  // Softmax tracking
  float max_logit = -1e20f;
  float sum_exp = 0.0f;
  float acc[MAX_HEAD_SIZE];
  for (int i = 0; i < head_size; i++) {
    acc[i] = 0.0f;
  }

  // Iterate over blocks
  const int* block_table =
      block_tables + seq_idx * max_blocks_per_seq;

  for (int block_i = 0; block_i < num_blocks; block_i++) {
    int physical_block_idx = block_table[block_i];
    int tokens_in_block =
        (block_i == num_blocks - 1)
            ? (context_len - block_i * block_size)
            : block_size;

    for (int token_i = 0; token_i < tokens_in_block; token_i++) {
      // --- Decode key ---
      int token_head_offset =
          token_i * bytes_per_token +
          kv_head_idx * bpt_per_head;
      const uint8_t* k_block_ptr =
          key_cache + physical_block_idx * block_total_bytes;
      const uint8_t* k_angles_ptr = k_block_ptr + token_head_offset;
      half k_radius = *reinterpret_cast<const half*>(
          k_angles_ptr + angle_bytes);
      const uint8_t* k_qjl_ptr =
          has_qjl(DT)
              ? (k_angles_ptr + angle_bytes + radius_bytes)
              : nullptr;
      half k_residual_norm =
          has_qjl(DT)
              ? *reinterpret_cast<const half*>(
                    k_angles_ptr + angle_bytes + radius_bytes + qjl_bytes)
              : __float2half(0.0f);

      float k_vec[MAX_HEAD_SIZE];
      turboquant_decode_head<DT>(k_angles_ptr, k_radius, k_qjl_ptr,
                                 k_residual_norm, head_size, rotation_seed,
                                 qjl_seed, qjl_proj_dim, k_vec);

      // Compute QK dot product
      float logit = 0.0f;
      for (int i = 0; i < head_size; i++) {
        logit += q[i] * k_vec[i];
      }
      logit *= scale;

      // --- Online softmax update ---
      float old_max = max_logit;
      max_logit = fmaxf(max_logit, logit);
      float correction = expf(old_max - max_logit);
      sum_exp = sum_exp * correction + expf(logit - max_logit);

      // Rescale running accumulator
      for (int i = 0; i < head_size; i++) {
        acc[i] *= correction;
      }

      // --- Decode value and accumulate ---
      const uint8_t* v_block_ptr =
          value_cache + physical_block_idx * block_total_bytes;
      const uint8_t* v_angles_ptr = v_block_ptr + token_head_offset;
      half v_radius = *reinterpret_cast<const half*>(
          v_angles_ptr + angle_bytes);
      const uint8_t* v_qjl_ptr =
          has_qjl(DT)
              ? (v_angles_ptr + angle_bytes + radius_bytes)
              : nullptr;
      half v_residual_norm =
          has_qjl(DT)
              ? *reinterpret_cast<const half*>(
                    v_angles_ptr + angle_bytes + radius_bytes + qjl_bytes)
              : __float2half(0.0f);

      float v_vec[MAX_HEAD_SIZE];
      turboquant_decode_head<DT>(v_angles_ptr, v_radius, v_qjl_ptr,
                                 v_residual_norm, head_size, rotation_seed,
                                 qjl_seed, qjl_proj_dim, v_vec);

      float weight = expf(logit - max_logit);
      for (int i = 0; i < head_size; i++) {
        acc[i] += weight * v_vec[i];
      }
    }
  }

  // Normalize by softmax denominator
  float inv_sum = 1.0f / sum_exp;
  float* out_ptr =
      output + (seq_idx * num_heads + head_idx) * head_size;
  for (int i = 0; i < head_size; i++) {
    out_ptr[i] = acc[i] * inv_sum;
  }
}

// ============================================================================
// Host-callable wrapper
// ============================================================================

void paged_attention_turboquant(
    torch::Tensor output, torch::Tensor query, torch::Tensor key_cache,
    torch::Tensor value_cache, torch::Tensor block_tables,
    torch::Tensor context_lens, double scale, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_size, int64_t block_size,
    int64_t max_blocks_per_seq, const std::string& tq_type,
    int64_t layer_seed, int64_t qjl_proj_dim) {
  int num_seqs = query.size(0);
  dim3 grid(num_seqs, num_heads);
  dim3 block(1);  // Single thread per (seq, head) — to be optimized with
                  // warp-level parallelism in future PRs

  const auto stream = at::cuda::getCurrentCUDAStream();

  if (tq_type == "pq4") {
    paged_attention_turboquant_kernel<TQDataType::kPQ4>
        <<<grid, block, 0, stream>>>(
            output.data_ptr<float>(), query.data_ptr<float>(),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
            static_cast<float>(scale),
            num_heads, num_kv_heads, head_size, block_size,
            max_blocks_per_seq, layer_seed, qjl_proj_dim);
  } else if (tq_type == "tq3") {
    paged_attention_turboquant_kernel<TQDataType::kTQ3>
        <<<grid, block, 0, stream>>>(
            output.data_ptr<float>(), query.data_ptr<float>(),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
            static_cast<float>(scale),
            num_heads, num_kv_heads, head_size, block_size,
            max_blocks_per_seq, layer_seed, qjl_proj_dim);
  } else if (tq_type == "tq2") {
    paged_attention_turboquant_kernel<TQDataType::kTQ2>
        <<<grid, block, 0, stream>>>(
            output.data_ptr<float>(), query.data_ptr<float>(),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
            static_cast<float>(scale),
            num_heads, num_kv_heads, head_size, block_size,
            max_blocks_per_seq, layer_seed, qjl_proj_dim);
  }
}
