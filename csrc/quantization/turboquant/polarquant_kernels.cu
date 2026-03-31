// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PolarQuant + QJL encode/decode CUDA kernels for TurboQuant KV cache
// quantization.
//
// Reference: Zandieh et al., "TurboQuant: Redefining AI Efficiency with
// Extreme Compression", ICLR 2026 (arXiv:2504.19874)

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "turboquant_utils.cuh"

using namespace vllm::turboquant;

// ============================================================================
// Encode kernel: KV vector → packed TurboQuant representation
// ============================================================================
//
// Each thread block handles one (token, head) pair.
// Grid: (num_tokens, num_kv_heads)

template <TQDataType DT>
__global__ void turboquant_encode_kernel(
    const float* __restrict__ kv_data,  // [num_tokens, num_kv_heads, head_size]
    uint8_t* __restrict__ angles_out,   // packed angle storage
    half* __restrict__ radii_out,       // [num_tokens, num_kv_heads]
    uint8_t* __restrict__ qjl_out,      // packed QJL sign bits (nullptr if
                                        // PQ4)
    int num_kv_heads, int head_size, uint32_t layer_seed, int qjl_proj_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  // Pointer to this head's KV vector
  const float* vec =
      kv_data + (token_idx * num_kv_heads + head_idx) * head_size;

  // Output pointers
  constexpr int BITS = angle_bits(DT);
  int num_angles = head_size - 1;
  int angle_bytes_per_head = (num_angles * BITS + 7) / 8;
  int qjl_bytes_per_head = has_qjl(DT) ? (qjl_proj_dim + 7) / 8 : 0;

  int head_offset = token_idx * num_kv_heads + head_idx;
  uint8_t* angles_ptr = angles_out + head_offset * angle_bytes_per_head;
  half* radius_ptr = radii_out + head_offset;
  uint8_t* qjl_ptr =
      qjl_out ? (qjl_out + head_offset * qjl_bytes_per_head) : nullptr;

  // Derive per-head seeds
  uint32_t rotation_seed = derive_rotation_seed(layer_seed, head_idx);
  uint32_t qjl_seed = derive_qjl_seed(layer_seed, head_idx);

  // Single-thread encode (thread 0 of each block)
  // TODO: Parallelize across threads within a block for larger head sizes
  if (threadIdx.x == 0) {
    turboquant_encode_head<DT>(vec, head_size, rotation_seed, qjl_seed,
                               angles_ptr, radius_ptr, qjl_ptr, qjl_proj_dim);
  }
}

// ============================================================================
// Decode kernel: packed TurboQuant → reconstructed KV vector
// ============================================================================

template <TQDataType DT>
__global__ void turboquant_decode_kernel(
    const uint8_t* __restrict__ angles,
    const half* __restrict__ radii,
    const uint8_t* __restrict__ qjl_bits,
    float* __restrict__ kv_out,  // [num_tokens, num_kv_heads, head_size]
    int num_kv_heads, int head_size, uint32_t layer_seed, int qjl_proj_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  constexpr int BITS = angle_bits(DT);
  int num_angles = head_size - 1;
  int angle_bytes_per_head = (num_angles * BITS + 7) / 8;
  int qjl_bytes_per_head = has_qjl(DT) ? (qjl_proj_dim + 7) / 8 : 0;

  int head_offset = token_idx * num_kv_heads + head_idx;
  const uint8_t* angles_ptr = angles + head_offset * angle_bytes_per_head;
  half radius = radii[head_offset];
  const uint8_t* qjl_ptr =
      qjl_bits ? (qjl_bits + head_offset * qjl_bytes_per_head) : nullptr;

  float* out_ptr =
      kv_out + (token_idx * num_kv_heads + head_idx) * head_size;

  uint32_t rotation_seed = derive_rotation_seed(layer_seed, head_idx);
  uint32_t qjl_seed = derive_qjl_seed(layer_seed, head_idx);

  if (threadIdx.x == 0) {
    turboquant_decode_head<DT>(angles_ptr, radius, qjl_ptr, head_size,
                               rotation_seed, qjl_seed, qjl_proj_dim, out_ptr);
  }
}

// ============================================================================
// Reshape-and-cache kernel for TurboQuant
// ============================================================================
//
// This integrates with vLLM's PagedAttention block layout.
// Instead of storing raw KV values, we store the polar-encoded representation.
//
// Each block stores:
//   - Packed angles: [num_kv_heads, block_size, angle_bytes_per_head]
//   - Radii: [num_kv_heads, block_size] (fp16)
//   - QJL bits: [num_kv_heads, block_size, qjl_bytes_per_head] (if enabled)

template <TQDataType DT, typename scalar_t>
__global__ void reshape_and_cache_turboquant_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_kv_heads,
                                         // head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_kv_heads,
                                         // head_size]
    uint8_t* __restrict__ key_cache,     // [num_blocks, block_bytes_k]
    uint8_t* __restrict__ value_cache,   // [num_blocks, block_bytes_v]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    int num_kv_heads, int head_size, int block_size, uint32_t layer_seed,
    int qjl_proj_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  // 0 = key, 1 = value
  int kv_idx = blockIdx.z;

  if (threadIdx.x != 0) return;

  int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;  // Padding token

  int block_idx = slot / block_size;
  int block_offset = slot % block_size;

  // Select key or value input
  const scalar_t* kv_data =
      (kv_idx == 0) ? key : value;
  uint8_t* cache = (kv_idx == 0) ? key_cache : value_cache;

  const scalar_t* vec =
      kv_data + (token_idx * num_kv_heads + head_idx) * head_size;

  // Convert input to float
  float vec_f[256];
  for (int i = 0; i < head_size; i++) {
    vec_f[i] = static_cast<float>(vec[i]);
  }

  // Calculate storage layout within a block
  constexpr int BITS = angle_bits(DT);
  int num_angles = head_size - 1;
  int angle_bytes_per_head = (num_angles * BITS + 7) / 8;
  int qjl_bytes_per_head = has_qjl(DT) ? (qjl_proj_dim + 7) / 8 : 0;
  int radius_bytes = 2;  // fp16

  int bytes_per_token_per_head =
      angle_bytes_per_head + qjl_bytes_per_head + radius_bytes;
  int bytes_per_token = num_kv_heads * bytes_per_token_per_head;
  int block_total_bytes = block_size * bytes_per_token;

  // Offset within block for this (token_in_block, head)
  int token_head_offset =
      block_offset * bytes_per_token + head_idx * bytes_per_token_per_head;

  uint8_t* block_ptr = cache + block_idx * block_total_bytes;
  uint8_t* angles_ptr = block_ptr + token_head_offset;
  half* radius_ptr =
      reinterpret_cast<half*>(angles_ptr + angle_bytes_per_head);
  uint8_t* qjl_ptr =
      has_qjl(DT) ? (angles_ptr + angle_bytes_per_head + radius_bytes)
                   : nullptr;

  uint32_t rotation_seed = derive_rotation_seed(layer_seed, head_idx);
  uint32_t qjl_seed = derive_qjl_seed(layer_seed, head_idx);

  turboquant_encode_head<DT>(vec_f, head_size, rotation_seed, qjl_seed,
                             angles_ptr, radius_ptr, qjl_ptr, qjl_proj_dim);
}

// ============================================================================
// Host-callable wrappers
// ============================================================================

void turboquant_encode(torch::Tensor kv_data, torch::Tensor angles_out,
                       torch::Tensor radii_out, torch::Tensor qjl_out,
                       int64_t num_kv_heads, int64_t head_size,
                       const std::string& tq_type, int64_t layer_seed,
                       int64_t qjl_proj_dim) {
  int num_tokens = kv_data.size(0);
  dim3 grid(num_tokens, num_kv_heads);
  dim3 block(1);  // Single thread per (token, head) for now

  const auto stream = at::cuda::getCurrentCUDAStream();

  if (tq_type == "pq4") {
    turboquant_encode_kernel<TQDataType::kPQ4><<<grid, block, 0, stream>>>(
        kv_data.data_ptr<float>(), angles_out.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(radii_out.data_ptr<at::Half>()),
        nullptr, num_kv_heads, head_size, layer_seed, qjl_proj_dim);
  } else if (tq_type == "tq3") {
    turboquant_encode_kernel<TQDataType::kTQ3><<<grid, block, 0, stream>>>(
        kv_data.data_ptr<float>(), angles_out.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(radii_out.data_ptr<at::Half>()),
        qjl_out.data_ptr<uint8_t>(), num_kv_heads, head_size, layer_seed,
        qjl_proj_dim);
  } else if (tq_type == "tq2") {
    turboquant_encode_kernel<TQDataType::kTQ2><<<grid, block, 0, stream>>>(
        kv_data.data_ptr<float>(), angles_out.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(radii_out.data_ptr<at::Half>()),
        qjl_out.data_ptr<uint8_t>(), num_kv_heads, head_size, layer_seed,
        qjl_proj_dim);
  }
}

void turboquant_decode(torch::Tensor angles, torch::Tensor radii,
                       torch::Tensor qjl_bits, torch::Tensor kv_out,
                       int64_t num_kv_heads, int64_t head_size,
                       const std::string& tq_type, int64_t layer_seed,
                       int64_t qjl_proj_dim) {
  int num_tokens = kv_out.size(0);
  dim3 grid(num_tokens, num_kv_heads);
  dim3 block(1);

  const auto stream = at::cuda::getCurrentCUDAStream();

  if (tq_type == "pq4") {
    turboquant_decode_kernel<TQDataType::kPQ4><<<grid, block, 0, stream>>>(
        angles.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.data_ptr<at::Half>()),
        nullptr, kv_out.data_ptr<float>(), num_kv_heads, head_size,
        layer_seed, qjl_proj_dim);
  } else if (tq_type == "tq3") {
    turboquant_decode_kernel<TQDataType::kTQ3><<<grid, block, 0, stream>>>(
        angles.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.data_ptr<at::Half>()),
        qjl_bits.data_ptr<uint8_t>(), kv_out.data_ptr<float>(), num_kv_heads,
        head_size, layer_seed, qjl_proj_dim);
  } else if (tq_type == "tq2") {
    turboquant_decode_kernel<TQDataType::kTQ2><<<grid, block, 0, stream>>>(
        angles.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.data_ptr<at::Half>()),
        qjl_bits.data_ptr<uint8_t>(), kv_out.data_ptr<float>(), num_kv_heads,
        head_size, layer_seed, qjl_proj_dim);
  }
}

void reshape_and_cache_turboquant(
    torch::Tensor key, torch::Tensor value, torch::Tensor key_cache,
    torch::Tensor value_cache, torch::Tensor slot_mapping, int64_t num_kv_heads,
    int64_t head_size, int64_t block_size, const std::string& tq_type,
    int64_t layer_seed, int64_t qjl_proj_dim) {
  int num_tokens = key.size(0);
  // Grid: (num_tokens, num_kv_heads, 2) for key and value
  dim3 grid(num_tokens, num_kv_heads, 2);
  dim3 block(1);

  const auto stream = at::cuda::getCurrentCUDAStream();

  // Dispatch by input dtype and TQ type
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(),
      "reshape_and_cache_turboquant", [&] {
        if (tq_type == "pq4") {
          reshape_and_cache_turboquant_kernel<TQDataType::kPQ4, scalar_t>
              <<<grid, block, 0, stream>>>(
                  key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                  key_cache.data_ptr<uint8_t>(),
                  value_cache.data_ptr<uint8_t>(),
                  slot_mapping.data_ptr<int64_t>(), num_kv_heads, head_size,
                  block_size, layer_seed, qjl_proj_dim);
        } else if (tq_type == "tq3") {
          reshape_and_cache_turboquant_kernel<TQDataType::kTQ3, scalar_t>
              <<<grid, block, 0, stream>>>(
                  key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                  key_cache.data_ptr<uint8_t>(),
                  value_cache.data_ptr<uint8_t>(),
                  slot_mapping.data_ptr<int64_t>(), num_kv_heads, head_size,
                  block_size, layer_seed, qjl_proj_dim);
        } else if (tq_type == "tq2") {
          reshape_and_cache_turboquant_kernel<TQDataType::kTQ2, scalar_t>
              <<<grid, block, 0, stream>>>(
                  key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                  key_cache.data_ptr<uint8_t>(),
                  value_cache.data_ptr<uint8_t>(),
                  slot_mapping.data_ptr<int64_t>(), num_kv_heads, head_size,
                  block_size, layer_seed, qjl_proj_dim);
        }
      });
}
