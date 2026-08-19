// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Sparse-MLA PCP cache preparation.
//
// The dispatch kernel keeps the local Q result local, but quantizes the K-side
// tensors directly into their final cache representation and multicasts one
// packed row per token through an NVLS symmetric-memory window. Static E4M3
// stores a 576-byte FP8 MLA row; fp8_ds_mla stores a 656-byte row with tiled
// NoPE scales and a BF16 RoPE tail. Sparse layers append:
//
//   [ MLA row | indexer K:128 | indexer scale:4 | pad:12 ]
//
// The combine kernel reads the acquired rank-major rows and scatters them into
// the rank-local paged MLA and indexer caches.  Cache block mappings are local,
// so the sender deliberately does not write remote paged-cache addresses.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "../../dispatch_utils.h"
#include "../../type_convert.cuh"
#include "../dcp_utils/dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::multimem_store_16;
using vllm::direct_dcp::multimem_store_release_system;
using vllm::direct_dcp::wait_for_epoch;

constexpr int kThreads = 256;
constexpr int kKvLoraDim = 512;
constexpr int kRopeDim = 64;
constexpr int kIndexerDim = 128;
constexpr float kFp8Max = 448.0f;

enum class MlaCacheLayout { kStaticE4M3, kFp8DsMla };

constexpr int align_up_16(int value) { return (value + 15) & ~15; }

template <int q_lora_dim, int kv_lora_dim, int rope_dim, int indexer_dim,
          bool index_rope_interleave, MlaCacheLayout cache_layout>
struct FusedNormRopeConfig {
  static_assert(q_lora_dim % kThreads == 0);
  static_assert(kv_lora_dim % 128 == 0);
  static_assert(kv_lora_dim % 16 == 0);
  static_assert(rope_dim % 16 == 0);
  static_assert(indexer_dim % 16 == 0);

  static constexpr int kQLoraDim = q_lora_dim;
  static constexpr int kKvLoraDim = kv_lora_dim;
  static constexpr int kRopeDim = rope_dim;
  static constexpr int kIndexerDim = indexer_dim;
  static constexpr bool kIndexRopeInterleave = index_rope_interleave;
  static constexpr MlaCacheLayout kCacheLayout = cache_layout;
  static constexpr int kMlaNumTiles = kv_lora_dim / 128;
  static constexpr int kMlaScaleBytes = kMlaNumTiles * sizeof(float);
  static constexpr int kMlaRopeBytes =
      cache_layout == MlaCacheLayout::kFp8DsMla ? rope_dim * 2 : rope_dim;
  static constexpr int kMlaRopeOffset =
      cache_layout == MlaCacheLayout::kFp8DsMla ? kv_lora_dim + kMlaScaleBytes
                                                : kv_lora_dim;
  static constexpr int kMlaRowBytes = kMlaRopeOffset + kMlaRopeBytes;
  static constexpr int kIndexerOffset = kMlaRowBytes;
  static constexpr int kIndexerScaleOffset = kIndexerOffset + indexer_dim;
  static constexpr int kPackedRowBytes =
      align_up_16(kIndexerScaleOffset + sizeof(float));

  static_assert(kMlaRowBytes % 16 == 0);
  static_assert(kIndexerOffset % 16 == 0);
  static_assert(cache_layout != MlaCacheLayout::kFp8DsMla || kMlaNumTiles == 4);
};

template <int q_lora_dim, bool index_rope_interleave,
          MlaCacheLayout cache_layout>
using DSV32FusedNormRopeConfig =
    FusedNormRopeConfig<q_lora_dim, kKvLoraDim, kRopeDim, kIndexerDim,
                        index_rope_interleave, cache_layout>;

template <typename scalar_t>
__device__ __forceinline__ float load_half(const scalar_t* ptr) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using device_t = typename Converter::hip_type;
  return Converter::convert(*reinterpret_cast<const device_t*>(ptr));
}

template <typename scalar_t>
__device__ __forceinline__ void store_half(scalar_t* ptr, float value) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using device_t = typename Converter::hip_type;
  *reinterpret_cast<device_t*>(ptr) = Converter::convert(value);
}

template <typename scalar_t>
__device__ __forceinline__ float round_half(float value) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using device_t = typename Converter::hip_type;
  device_t rounded = Converter::convert(value);
  return Converter::convert(rounded);
}

template <typename scalar_t>
__device__ __forceinline__ float load_cos_sin(const void* ptr, int64_t offset,
                                              bool is_float) {
  if (is_float) {
    return static_cast<const float*>(ptr)[offset];
  }
  return load_half(static_cast<const scalar_t*>(ptr) + offset);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
  if (warp == 0) {
    value = warp_max(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ uint4 pack_fp8_16(const float* values,
                                             float inverse_scale) {
  uint4 output;
  auto* output_pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&output);
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    float2 pair = make_float2(values[2 * idx] * inverse_scale,
                              values[2 * idx + 1] * inverse_scale);
    pair.x = fminf(fmaxf(pair.x, -kFp8Max), kFp8Max);
    pair.y = fminf(fmaxf(pair.y, -kFp8Max), kFp8Max);
    output_pairs[idx] =
        __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
  }
  return output;
}

__device__ __forceinline__ uint4 pack_bfloat16_8(const float* values) {
  uint4 output;
  auto* output_values = reinterpret_cast<__nv_bfloat16*>(&output);
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    output_values[idx] = __float2bfloat16(values[idx]);
  }
  return output;
}

// One CTA becomes the invocation leader.  It advances the epoch and publishes
// it before any other CTA selects the double-buffer slot.  The last completed
// CTA resets phase to zero, so the next stream-ordered invocation can elect a
// new leader without a separate epoch-increment launch.
__device__ __forceinline__ uint32_t establish_epoch(int32_t* phase,
                                                    int64_t* epoch,
                                                    uint32_t* shared_epoch) {
  if (threadIdx.x == 0) {
    if (atomicCAS(phase, 0, 1) == 0) {
      auto* epoch_u64 = reinterpret_cast<unsigned long long*>(epoch);
      uint64_t next = atomicAdd(epoch_u64, 1ULL) + 1ULL;
      *shared_epoch = static_cast<uint32_t>(next);
      __threadfence();
      atomicExch(phase, 2);
    } else {
      while (atomicAdd(phase, 0) != 2) {
      }
      *shared_epoch = static_cast<uint32_t>(*epoch);
    }
  }
  __syncthreads();
  return *shared_epoch;
}

template <typename scalar_t, typename Config>
__global__ void fused_norm_rope_pcp_dispatch_kernel(
    const int64_t* __restrict__ positions, const scalar_t* __restrict__ q_c,
    int64_t q_stride, const scalar_t* __restrict__ q_weight, float q_eps,
    scalar_t* __restrict__ q_out, const scalar_t* __restrict__ kv_c,
    int64_t kv_stride, const scalar_t* __restrict__ kv_weight,
    scalar_t* __restrict__ kv_out, int64_t kv_out_stride, float kv_eps,
    const float* __restrict__ mla_k_scale, const scalar_t* __restrict__ k_pe,
    int64_t k_pe_stride, scalar_t* __restrict__ k_pe_out,
    int64_t k_pe_out_stride, const void* __restrict__ k_pe_cos_sin,
    int64_t k_pe_cos_sin_stride, bool k_pe_cos_sin_is_float,
    const scalar_t* __restrict__ index_k, int64_t index_k_stride,
    const float* __restrict__ index_weight,
    const float* __restrict__ index_bias, float index_eps,
    const void* __restrict__ index_cos_sin, int64_t index_cos_sin_stride,
    bool index_cos_sin_is_float, int32_t* __restrict__ topk_indices,
    uint4* __restrict__ payload_mc, uint32_t* __restrict__ signal_mc,
    const uint32_t* __restrict__ received_signal,
    uint32_t* __restrict__ completion, int64_t* __restrict__ epoch,
    int32_t* __restrict__ phase, int world_size, int rank, int max_local_tokens,
    int topk, int topk_stride, bool has_indexer) {
  __shared__ float reduce_scratch[8];
  __shared__ float kv_values[Config::kKvLoraDim];
  __shared__ float k_pe_values[Config::kRopeDim];
  __shared__ float index_values[Config::kIndexerDim];
  __shared__ float mla_tile_scales[Config::kMlaNumTiles];
  __shared__ uint32_t active_epoch;

  int token = blockIdx.x;
  int tid = threadIdx.x;
  uint32_t invocation = establish_epoch(phase, epoch, &active_epoch);
  int parity = invocation & 1u;

  // Q RMSNorm remains local.  Eight values per thread stay in registers across
  // the block reduction, avoiding an intermediate Q materialization.
  float q_values[Config::kQLoraDim / kThreads];
  float q_sum_sq = 0.0f;
#pragma unroll
  for (int idx = 0; idx < Config::kQLoraDim / kThreads; ++idx) {
    int dim = tid + idx * kThreads;
    float value = load_half(q_c + static_cast<int64_t>(token) * q_stride + dim);
    q_values[idx] = value;
    q_sum_sq += value * value;
  }
  float q_inv_rms = rsqrtf(block_sum(q_sum_sq, reduce_scratch) /
                               static_cast<float>(Config::kQLoraDim) +
                           q_eps);
#pragma unroll
  for (int idx = 0; idx < Config::kQLoraDim / kThreads; ++idx) {
    int dim = tid + idx * kThreads;
    float weight = load_half(q_weight + dim);
    store_half(q_out + static_cast<int64_t>(token) * Config::kQLoraDim + dim,
               q_values[idx] * q_inv_rms * weight);
  }

  // KV RMSNorm.  The normalized FP32 values remain in shared memory only until
  // the final per-tensor FP8 pack.
  float kv_sum_sq = 0.0f;
#pragma unroll
  for (int idx = 0; idx < Config::kKvLoraDim / kThreads; ++idx) {
    int dim = tid + idx * kThreads;
    float value =
        load_half(kv_c + static_cast<int64_t>(token) * kv_stride + dim);
    kv_values[dim] = value;
    kv_sum_sq += value * value;
  }
  float kv_inv_rms = rsqrtf(block_sum(kv_sum_sq, reduce_scratch) /
                                static_cast<float>(Config::kKvLoraDim) +
                            kv_eps);
#pragma unroll
  for (int idx = 0; idx < Config::kKvLoraDim / kThreads; ++idx) {
    int dim = tid + idx * kThreads;
    float weight = load_half(kv_weight + dim);
    kv_values[dim] = round_half<scalar_t>(kv_values[dim] * kv_inv_rms * weight);
    store_half(kv_out + static_cast<int64_t>(token) * kv_out_stride + dim,
               kv_values[dim]);
  }

  // This layout uses adjacent-pair (interleaved) RoPE for MLA K-pe.
  if (tid < Config::kRopeDim / 2) {
    int64_t position = positions[token];
    int64_t cos_sin_offset = position * k_pe_cos_sin_stride;
    float cosine = load_cos_sin<scalar_t>(k_pe_cos_sin, cos_sin_offset + tid,
                                          k_pe_cos_sin_is_float);
    float sine = load_cos_sin<scalar_t>(
        k_pe_cos_sin, cos_sin_offset + tid + Config::kRopeDim / 2,
        k_pe_cos_sin_is_float);
    const scalar_t* row = k_pe + static_cast<int64_t>(token) * k_pe_stride;
    float x = load_half(row + 2 * tid);
    float y = load_half(row + 2 * tid + 1);
    k_pe_values[2 * tid] = round_half<scalar_t>(x * cosine - y * sine);
    k_pe_values[2 * tid + 1] = round_half<scalar_t>(y * cosine + x * sine);
    scalar_t* output_row =
        k_pe_out + static_cast<int64_t>(token) * k_pe_out_stride;
    store_half(output_row + 2 * tid, k_pe_values[2 * tid]);
    store_half(output_row + 2 * tid + 1, k_pe_values[2 * tid + 1]);
  }

  // Shared layers skip all indexer reductions below.  Synchronize the KV and
  // K-pe producers explicitly before either branch packs their shared rows.
  __syncthreads();

  // Indexer K LayerNorm, interleaved RoPE, and UE8M0 FP8 quantization.
  float index_value = 0.0f;
  if (has_indexer) {
    if (tid < Config::kIndexerDim) {
      index_value = load_half(
          index_k + static_cast<int64_t>(token) * index_k_stride + tid);
    }
  }
  float index_scale = 1.0f;
  if (has_indexer) {
    float index_mean = block_sum(index_value, reduce_scratch) /
                       static_cast<float>(Config::kIndexerDim);
    float centered =
        tid < Config::kIndexerDim ? index_value - index_mean : 0.0f;
    float index_variance = block_sum(centered * centered, reduce_scratch) /
                           static_cast<float>(Config::kIndexerDim);
    if (tid < Config::kIndexerDim) {
      index_values[tid] =
          centered * rsqrtf(index_variance + index_eps) * index_weight[tid] +
          index_bias[tid];
    }
    __syncthreads();
    if (tid < Config::kRopeDim / 2) {
      int64_t position = positions[token];
      int64_t cos_sin_offset = position * index_cos_sin_stride;
      float cosine = load_cos_sin<scalar_t>(index_cos_sin, cos_sin_offset + tid,
                                            index_cos_sin_is_float);
      float sine = load_cos_sin<scalar_t>(
          index_cos_sin, cos_sin_offset + tid + Config::kRopeDim / 2,
          index_cos_sin_is_float);
      if constexpr (Config::kIndexRopeInterleave) {
        float x = index_values[2 * tid];
        float y = index_values[2 * tid + 1];
        index_values[2 * tid] = x * cosine - y * sine;
        index_values[2 * tid + 1] = y * cosine + x * sine;
      } else {
        float x = index_values[tid];
        float y = index_values[tid + Config::kRopeDim / 2];
        index_values[tid] = x * cosine - y * sine;
        index_values[tid + Config::kRopeDim / 2] = y * cosine + x * sine;
      }
    }
    __syncthreads();
    if (tid < Config::kIndexerDim) {
      index_values[tid] = round_half<scalar_t>(index_values[tid]);
    }
    __syncthreads();
    float local_max =
        tid < Config::kIndexerDim ? fabsf(index_values[tid]) : 0.0f;
    index_scale =
        fmaxf(block_max(local_max, reduce_scratch), 1.0e-4f) / kFp8Max;
    index_scale = exp2f(ceilf(log2f(index_scale)));
  }

  int64_t slot_stride = static_cast<int64_t>(world_size) * max_local_tokens *
                        Config::kPackedRowBytes;
  int64_t row_offset = static_cast<int64_t>(parity) * slot_stride +
                       (static_cast<int64_t>(rank) * max_local_tokens + token) *
                           Config::kPackedRowBytes;
  auto* row_mc = reinterpret_cast<uint8_t*>(payload_mc) + row_offset;

  if constexpr (Config::kCacheLayout == MlaCacheLayout::kStaticE4M3) {
    // Quantize the rounded input values with the cache's static scale.
    if (tid < Config::kKvLoraDim / 16) {
      float values[16];
#pragma unroll
      for (int idx = 0; idx < 16; ++idx) {
        values[idx] = kv_values[tid * 16 + idx];
      }
      float inverse_scale = 1.0f / mla_k_scale[0];
      multimem_store_16(reinterpret_cast<uint4*>(row_mc) + tid,
                        pack_fp8_16(values, inverse_scale));
    }
    if (tid < Config::kRopeDim / 16) {
      float values[16];
#pragma unroll
      for (int idx = 0; idx < 16; ++idx) {
        values[idx] = k_pe_values[tid * 16 + idx];
      }
      float inverse_scale = 1.0f / mla_k_scale[0];
      multimem_store_16(
          reinterpret_cast<uint4*>(row_mc + Config::kMlaRopeOffset) + tid,
          pack_fp8_16(values, inverse_scale));
    }
  } else {
    // fp8_ds_mla uses one dynamic scale per 128-value NoPE tile and keeps
    // the RoPE tail in BF16.
    if (tid < 32) {
#pragma unroll
      for (int tile = 0; tile < Config::kMlaNumTiles; ++tile) {
        float local_max = 0.0f;
#pragma unroll
        for (int idx = 0; idx < 4; ++idx) {
          local_max = fmaxf(
              local_max,
              fabsf(kv_values[tile * 128 + tid + static_cast<int>(idx) * 32]));
        }
        float tile_max = warp_max(local_max);
        if (tid == 0) {
          mla_tile_scales[tile] =
              fmaxf(tile_max * (1.0f / kFp8Max), 1.1754944e-38f);
        }
      }
    }
    __syncthreads();
    if (tid < Config::kKvLoraDim / 16) {
      float values[16];
#pragma unroll
      for (int idx = 0; idx < 16; ++idx) {
        values[idx] = kv_values[tid * 16 + idx];
      }
      int tile = tid / (128 / 16);
      multimem_store_16(reinterpret_cast<uint4*>(row_mc) + tid,
                        pack_fp8_16(values, 1.0f / mla_tile_scales[tile]));
    }
    if (tid == 0) {
      uint4 scales = {
          __float_as_uint(mla_tile_scales[0]),
          __float_as_uint(mla_tile_scales[1]),
          __float_as_uint(mla_tile_scales[2]),
          __float_as_uint(mla_tile_scales[3]),
      };
      multimem_store_16(reinterpret_cast<uint4*>(row_mc + Config::kKvLoraDim),
                        scales);
    }
    if (tid < Config::kRopeDim / 8) {
      float values[8];
#pragma unroll
      for (int idx = 0; idx < 8; ++idx) {
        values[idx] = k_pe_values[tid * 8 + idx];
      }
      multimem_store_16(
          reinterpret_cast<uint4*>(row_mc + Config::kMlaRopeOffset) + tid,
          pack_bfloat16_8(values));
    }
  }
  if (has_indexer) {
    if (tid < Config::kIndexerDim / 16) {
      float values[16];
#pragma unroll
      for (int idx = 0; idx < 16; ++idx) {
        values[idx] = index_values[tid * 16 + idx];
      }
      multimem_store_16(
          reinterpret_cast<uint4*>(row_mc + Config::kIndexerOffset) + tid,
          pack_fp8_16(values, 1.0f / index_scale));
    }
    if (tid == 0) {
      uint4 scale_and_padding = {0u, 0u, 0u, 0u};
      scale_and_padding.x = __float_as_uint(index_scale);
      multimem_store_16(
          reinterpret_cast<uint4*>(row_mc + Config::kIndexerScaleOffset),
          scale_and_padding);
    }
    for (int idx = tid; idx < topk; idx += blockDim.x) {
      topk_indices[static_cast<int64_t>(token) * topk_stride + idx] = -1;
    }
  }

  __threadfence_system();
  __syncthreads();
  if (tid != 0) {
    return;
  }
  uint32_t completed = atomicAdd(completion + parity, 1u);
  if (completed + 1u != gridDim.x) {
    return;
  }
  atomicExch(completion + parity, 0u);
  multimem_store_release_system(signal_mc + parity * world_size + rank,
                                invocation);
  for (int source = 0; source < world_size; ++source) {
    if (!wait_for_epoch(received_signal + parity * world_size + source,
                        invocation)) {
      printf("fused_norm_rope_pcp dispatch timeout source=%d epoch=%u\n",
             source, invocation);
      asm volatile("trap;");
    }
  }
  __threadfence();
  atomicExch(phase, 0);
}

template <typename Config>
__global__ void fused_norm_rope_pcp_combine_kernel(
    const uint8_t* __restrict__ received_payload,
    const int64_t* __restrict__ epoch,
    const int64_t* __restrict__ mla_slot_mapping,
    const int64_t* __restrict__ index_slot_mapping,
    uint8_t* __restrict__ mla_cache, int64_t mla_block_stride,
    int64_t mla_token_stride, uint8_t* __restrict__ index_cache,
    int64_t index_block_stride, int world_size, int local_tokens,
    int num_decode_tokens, int max_local_tokens, int cache_block_size,
    bool has_indexer) {
  int cache_token = blockIdx.x;
  int source;
  int source_token;
  int mapping_token;
  if (cache_token < num_decode_tokens) {
    source = 0;
    source_token = cache_token;
    mapping_token = cache_token;
  } else {
    int local_prefill_tokens = local_tokens - num_decode_tokens;
    int prefill_token = cache_token - num_decode_tokens;
    source = prefill_token / local_prefill_tokens;
    source_token =
        num_decode_tokens + prefill_token - source * local_prefill_tokens;
    mapping_token = source * local_tokens + source_token;
  }
  int parity = static_cast<uint32_t>(epoch[0]) & 1u;
  int64_t row_offset =
      (static_cast<int64_t>(parity) * world_size * max_local_tokens +
       static_cast<int64_t>(source) * max_local_tokens + source_token) *
      Config::kPackedRowBytes;
  const uint8_t* row = received_payload + row_offset;

  int64_t mla_slot = mla_slot_mapping[mapping_token];
  if (mla_slot >= 0) {
    int64_t block = mla_slot / cache_block_size;
    int64_t block_offset = mla_slot % cache_block_size;
    uint8_t* destination =
        mla_cache + block * mla_block_stride + block_offset * mla_token_stride;
    for (int chunk = threadIdx.x; chunk < Config::kMlaRowBytes / 16;
         chunk += blockDim.x) {
      reinterpret_cast<uint4*>(destination)[chunk] =
          reinterpret_cast<const uint4*>(row)[chunk];
    }
  }

  if (has_indexer) {
    int64_t index_slot = index_slot_mapping[mapping_token];
    if (index_slot >= 0) {
      int64_t block = index_slot / cache_block_size;
      int64_t block_offset = index_slot % cache_block_size;
      uint8_t* block_base = index_cache + block * index_block_stride;
      uint8_t* value_destination =
          block_base + block_offset * Config::kIndexerDim;
      for (int chunk = threadIdx.x; chunk < Config::kIndexerDim / 16;
           chunk += blockDim.x) {
        reinterpret_cast<uint4*>(value_destination)[chunk] =
            reinterpret_cast<const uint4*>(row + Config::kIndexerOffset)[chunk];
      }
      if (threadIdx.x == 0) {
        uint8_t* scale_destination = block_base +
                                     cache_block_size * Config::kIndexerDim +
                                     block_offset * 4;
        *reinterpret_cast<uint32_t*>(scale_destination) =
            *reinterpret_cast<const uint32_t*>(row +
                                               Config::kIndexerScaleOffset);
      }
    }
  }
}

void fused_norm_rope_pcp_dispatch(
    const torch::stable::Tensor& positions, const torch::stable::Tensor& q_c,
    const torch::stable::Tensor& q_weight, double q_eps,
    torch::stable::Tensor& q_out, const torch::stable::Tensor& kv_c,
    const torch::stable::Tensor& kv_weight, torch::stable::Tensor& kv_out,
    const torch::stable::Tensor& mla_k_scale, double kv_eps,
    const torch::stable::Tensor& k_pe, torch::stable::Tensor& k_pe_out,
    const torch::stable::Tensor& k_pe_cos_sin,
    const torch::stable::Tensor& index_k,
    const torch::stable::Tensor& index_weight,
    const torch::stable::Tensor& index_bias, double index_eps,
    const torch::stable::Tensor& index_cos_sin,
    torch::stable::Tensor& topk_indices,
    torch::stable::Tensor& received_payload,
    torch::stable::Tensor& received_signal, torch::stable::Tensor& completion,
    torch::stable::Tensor& epoch, torch::stable::Tensor& phase,
    int64_t world_size, int64_t rank, bool has_indexer, bool fp8_ds_mla,
    bool index_rope_interleave, int64_t payload_mc_ptr, int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;
  auto valid_rows = [](const torch::stable::Tensor& tensor, int64_t rows,
                       int64_t columns, ScalarType dtype) {
    return tensor.is_cuda() && tensor.dim() == 2 && tensor.size(0) == rows &&
           tensor.size(1) == columns && tensor.stride(1) == 1 &&
           tensor.scalar_type() == dtype;
  };
  STD_TORCH_CHECK(q_c.is_cuda() && q_c.dim() == 2 && q_c.stride(1) == 1,
                  "q_c must be a row-contiguous CUDA matrix");
  int64_t q_lora_dim = q_c.size(1);
  STD_TORCH_CHECK(q_lora_dim == 1536 || q_lora_dim == 2048,
                  "q_c width must be 1536 or 2048");
  ScalarType dtype = q_c.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "fused_norm_rope_pcp only supports FP16/BF16 inputs");
  int64_t num_tokens = q_c.size(0);
  STD_TORCH_CHECK(num_tokens > 0, "fused_norm_rope_pcp requires tokens");
  STD_TORCH_CHECK(valid_rows(kv_c, num_tokens, kKvLoraDim, dtype),
                  "kv_c must be [T,512] with contiguous rows and the Q dtype");
  STD_TORCH_CHECK(valid_rows(k_pe, num_tokens, kRopeDim, dtype),
                  "k_pe must be [T,64] with contiguous rows and the Q dtype");
  if (has_indexer) {
    STD_TORCH_CHECK(
        valid_rows(index_k, num_tokens, kIndexerDim, dtype),
        "index_k must be [T,128] with contiguous rows and the Q dtype");
  }
  STD_TORCH_CHECK(q_weight.is_cuda() && q_weight.is_contiguous() &&
                      q_weight.scalar_type() == dtype &&
                      q_weight.numel() == q_lora_dim,
                  "q_weight must match the Q width and dtype");
  STD_TORCH_CHECK(kv_weight.is_cuda() && kv_weight.is_contiguous() &&
                      kv_weight.scalar_type() == dtype &&
                      kv_weight.numel() == kKvLoraDim,
                  "kv_weight must be contiguous [512] with the Q dtype");
  STD_TORCH_CHECK(mla_k_scale.is_cuda() && mla_k_scale.is_contiguous() &&
                      mla_k_scale.scalar_type() == ScalarType::Float &&
                      mla_k_scale.numel() == 1,
                  "mla_k_scale must be one CUDA float32 value");
  if (has_indexer) {
    STD_TORCH_CHECK(index_weight.is_cuda() && index_weight.is_contiguous() &&
                        index_weight.scalar_type() == ScalarType::Float &&
                        index_weight.numel() == kIndexerDim &&
                        index_bias.is_cuda() && index_bias.is_contiguous() &&
                        index_bias.scalar_type() == ScalarType::Float &&
                        index_bias.numel() == kIndexerDim,
                    "indexer norm weight/bias must be CUDA float32 [128]");
  }
  STD_TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                      positions.scalar_type() == ScalarType::Long &&
                      positions.numel() == num_tokens,
                  "positions must be contiguous CUDA int64 [T]");
  auto valid_cos_sin = [&](const torch::stable::Tensor& tensor) {
    auto cache_dtype = tensor.scalar_type();
    return tensor.is_cuda() && tensor.dim() == 2 &&
           tensor.size(1) == kRopeDim && tensor.stride(1) == 1 &&
           (cache_dtype == ScalarType::Float || cache_dtype == dtype);
  };
  STD_TORCH_CHECK(valid_cos_sin(k_pe_cos_sin),
                  "K RoPE cache must be CUDA float32/input-dtype [P,64] "
                  "with contiguous rows");
  if (has_indexer) {
    STD_TORCH_CHECK(valid_cos_sin(index_cos_sin),
                    "index RoPE cache must be CUDA float32/input-dtype "
                    "[P,64] with contiguous rows");
  }
  STD_TORCH_CHECK(q_out.is_cuda() && q_out.is_contiguous() &&
                      q_out.scalar_type() == dtype && q_out.dim() == 2 &&
                      q_out.size(0) == num_tokens &&
                      q_out.size(1) == q_lora_dim,
                  "q_out must match the Q shape and dtype");
  STD_TORCH_CHECK(
      valid_rows(kv_out, num_tokens, kKvLoraDim, dtype),
      "kv_out must be [T,512] with contiguous rows and the Q dtype");
  STD_TORCH_CHECK(
      valid_rows(k_pe_out, num_tokens, kRopeDim, dtype),
      "k_pe_out must be [T,64] with contiguous rows and the Q dtype");
  if (has_indexer) {
    STD_TORCH_CHECK(topk_indices.is_cuda() &&
                        topk_indices.scalar_type() == ScalarType::Int &&
                        topk_indices.dim() == 2 &&
                        topk_indices.stride(1) == 1 &&
                        topk_indices.size(0) >= num_tokens,
                    "topk_indices must be row-contiguous CUDA int32 [>=T,K]");
  }
  STD_TORCH_CHECK(world_size > 1 && rank >= 0 && rank < world_size,
                  "invalid PCP world_size/rank");
  constexpr int kStaticPackedRowBytes =
      DSV32FusedNormRopeConfig<1536, true,
                               MlaCacheLayout::kStaticE4M3>::kPackedRowBytes;
  constexpr int kDsMlaPackedRowBytes =
      DSV32FusedNormRopeConfig<1536, true,
                               MlaCacheLayout::kFp8DsMla>::kPackedRowBytes;
  int expected_packed_row_bytes =
      fp8_ds_mla ? kDsMlaPackedRowBytes : kStaticPackedRowBytes;
  STD_TORCH_CHECK(
      received_payload.is_cuda() && received_payload.is_contiguous() &&
          received_payload.scalar_type() == ScalarType::Byte &&
          received_payload.dim() == 4 && received_payload.size(0) == 2 &&
          received_payload.size(1) == world_size &&
          received_payload.size(3) == expected_packed_row_bytes &&
          num_tokens <= received_payload.size(2),
      "received_payload has the wrong PCP cache-layout row size");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == 2 &&
          received_signal.size(1) == world_size,
      "received_signal must be int32 [2,W]");
  STD_TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                      completion.scalar_type() == ScalarType::Int &&
                      completion.numel() == 2,
                  "completion must be int32 [2]");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be int64 [1]");
  STD_TORCH_CHECK(phase.is_cuda() && phase.is_contiguous() &&
                      phase.scalar_type() == ScalarType::Int &&
                      phase.numel() == 1,
                  "phase must be int32 [1]");
  STD_TORCH_CHECK(payload_mc_ptr != 0 && signal_mc_ptr != 0,
                  "fused_norm_rope_pcp requires NVLS multicast pointers");
  int device = q_c.get_device_index();
  STD_TORCH_CHECK(
      kv_c.get_device_index() == device && k_pe.get_device_index() == device &&
          received_payload.get_device_index() == device &&
          received_signal.get_device_index() == device,
      "fused_norm_rope_pcp tensors must be on the same CUDA device");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  cudaStream_t stream = get_current_cuda_stream(device);
  int max_local_tokens = static_cast<int>(received_payload.size(2));
  int topk = static_cast<int>(topk_indices.size(1));
  int topk_stride = static_cast<int>(topk_indices.stride(0));
  auto launch = [&]<typename Config>() {
    VLLM_STABLE_DISPATCH_HALF_TYPES(dtype, "fused_norm_rope_pcp_dispatch", [&] {
      fused_norm_rope_pcp_dispatch_kernel<scalar_t, Config>
          <<<num_tokens, kThreads, 0, stream>>>(
              positions.const_data_ptr<int64_t>(),
              reinterpret_cast<const scalar_t*>(q_c.const_data_ptr()),
              q_c.stride(0),
              reinterpret_cast<const scalar_t*>(q_weight.const_data_ptr()),
              static_cast<float>(q_eps),
              reinterpret_cast<scalar_t*>(q_out.mutable_data_ptr()),
              reinterpret_cast<const scalar_t*>(kv_c.const_data_ptr()),
              kv_c.stride(0),
              reinterpret_cast<const scalar_t*>(kv_weight.const_data_ptr()),
              reinterpret_cast<scalar_t*>(kv_out.mutable_data_ptr()),
              kv_out.stride(0), static_cast<float>(kv_eps),
              mla_k_scale.const_data_ptr<float>(),
              reinterpret_cast<const scalar_t*>(k_pe.const_data_ptr()),
              k_pe.stride(0),
              reinterpret_cast<scalar_t*>(k_pe_out.mutable_data_ptr()),
              k_pe_out.stride(0), k_pe_cos_sin.const_data_ptr(),
              k_pe_cos_sin.stride(0),
              k_pe_cos_sin.scalar_type() == ScalarType::Float,
              reinterpret_cast<const scalar_t*>(index_k.const_data_ptr()),
              has_indexer ? index_k.stride(0) : 0,
              index_weight.const_data_ptr<float>(),
              index_bias.const_data_ptr<float>(), static_cast<float>(index_eps),
              index_cos_sin.const_data_ptr(),
              has_indexer ? index_cos_sin.stride(0) : 0,
              has_indexer && index_cos_sin.scalar_type() == ScalarType::Float,
              topk_indices.mutable_data_ptr<int32_t>(),
              reinterpret_cast<uint4*>(static_cast<uintptr_t>(payload_mc_ptr)),
              reinterpret_cast<uint32_t*>(
                  static_cast<uintptr_t>(signal_mc_ptr)),
              reinterpret_cast<const uint32_t*>(
                  received_signal.const_data_ptr<int32_t>()),
              reinterpret_cast<uint32_t*>(
                  completion.mutable_data_ptr<int32_t>()),
              epoch.mutable_data_ptr<int64_t>(),
              phase.mutable_data_ptr<int32_t>(), static_cast<int>(world_size),
              static_cast<int>(rank), max_local_tokens, topk, topk_stride,
              has_indexer);
    });
  };
  auto launch_q_dim = [&]<int q_dim>() {
    if (fp8_ds_mla) {
      if (index_rope_interleave) {
        using Config =
            DSV32FusedNormRopeConfig<q_dim, true, MlaCacheLayout::kFp8DsMla>;
        launch.template operator()<Config>();
      } else {
        using Config =
            DSV32FusedNormRopeConfig<q_dim, false, MlaCacheLayout::kFp8DsMla>;
        launch.template operator()<Config>();
      }
    } else if (index_rope_interleave) {
      using Config =
          DSV32FusedNormRopeConfig<q_dim, true, MlaCacheLayout::kStaticE4M3>;
      launch.template operator()<Config>();
    } else {
      using Config =
          DSV32FusedNormRopeConfig<q_dim, false, MlaCacheLayout::kStaticE4M3>;
      launch.template operator()<Config>();
    }
  };
  if (q_lora_dim == 1536) {
    launch_q_dim.template operator()<1536>();
  } else {
    launch_q_dim.template operator()<2048>();
  }
  check_cuda_launch("fused_norm_rope_pcp_dispatch");
}

void fused_norm_rope_pcp_combine(
    const torch::stable::Tensor& received_payload,
    const torch::stable::Tensor& epoch,
    const torch::stable::Tensor& mla_slot_mapping,
    const torch::stable::Tensor& index_slot_mapping,
    torch::stable::Tensor& mla_cache, torch::stable::Tensor& index_cache,
    int64_t local_tokens, int64_t num_decode_tokens, bool has_indexer,
    bool fp8_ds_mla) {
  using torch::headeronly::ScalarType;
  constexpr int kStaticPackedRowBytes =
      DSV32FusedNormRopeConfig<1536, true,
                               MlaCacheLayout::kStaticE4M3>::kPackedRowBytes;
  constexpr int kDsMlaPackedRowBytes =
      DSV32FusedNormRopeConfig<1536, true,
                               MlaCacheLayout::kFp8DsMla>::kPackedRowBytes;
  int expected_packed_row_bytes =
      fp8_ds_mla ? kDsMlaPackedRowBytes : kStaticPackedRowBytes;
  STD_TORCH_CHECK(
      received_payload.is_cuda() && received_payload.is_contiguous() &&
          received_payload.scalar_type() == ScalarType::Byte &&
          received_payload.dim() == 4 && received_payload.size(0) == 2 &&
          received_payload.size(3) == expected_packed_row_bytes,
      "received_payload has the wrong PCP cache-layout row size");
  int64_t world_size = received_payload.size(1);
  int64_t max_local_tokens = received_payload.size(2);
  STD_TORCH_CHECK(local_tokens > 0 && local_tokens <= max_local_tokens,
                  "invalid local token count");
  STD_TORCH_CHECK(num_decode_tokens >= 0 && num_decode_tokens <= local_tokens,
                  "invalid decode token count");
  int64_t required_mapping_tokens = num_decode_tokens == local_tokens
                                        ? num_decode_tokens
                                        : world_size * local_tokens;
  auto valid_mapping = [&](const torch::stable::Tensor& mapping) {
    return mapping.is_cuda() && mapping.is_contiguous() &&
           mapping.scalar_type() == ScalarType::Long &&
           mapping.numel() >= required_mapping_tokens;
  };
  STD_TORCH_CHECK(valid_mapping(mla_slot_mapping),
                  "MLA slot mapping is too short for the PCP token layout");
  if (has_indexer) {
    STD_TORCH_CHECK(valid_mapping(index_slot_mapping),
                    "index slot mapping is too short for the PCP token layout");
  }
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be int64 [1]");
  auto mla_dtype = mla_cache.scalar_type();
  int expected_mla_row_bytes = fp8_ds_mla ? 656 : 576;
  bool valid_mla_dtype = fp8_ds_mla
                             ? mla_dtype == ScalarType::Byte
                             : mla_dtype == ScalarType::Byte ||
                                   mla_dtype == ScalarType::Float8_e4m3fn;
  STD_TORCH_CHECK(mla_cache.is_cuda() && mla_cache.is_contiguous() &&
                      valid_mla_dtype && mla_cache.dim() == 3 &&
                      mla_cache.size(2) == expected_mla_row_bytes,
                  "MLA cache has the wrong dtype or cache-layout row size");
  if (has_indexer) {
    STD_TORCH_CHECK(index_cache.is_cuda() && index_cache.is_contiguous() &&
                        index_cache.scalar_type() == ScalarType::Byte &&
                        index_cache.dim() == 3 &&
                        index_cache.size(2) >= kIndexerDim + 4,
                    "index cache must be contiguous uint8 [B,block,>=132]");
    STD_TORCH_CHECK(mla_cache.size(1) == index_cache.size(1),
                    "MLA and index caches must use the same block size");
  }
  int device = received_payload.get_device_index();
  STD_TORCH_CHECK(
      epoch.get_device_index() == device &&
          mla_slot_mapping.get_device_index() == device &&
          mla_cache.get_device_index() == device &&
          (!has_indexer || (index_slot_mapping.get_device_index() == device &&
                            index_cache.get_device_index() == device)),
      "fused_norm_rope_pcp tensors must be on one device");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  cudaStream_t stream = get_current_cuda_stream(device);
  int blocks = static_cast<int>(
      num_decode_tokens + world_size * (local_tokens - num_decode_tokens));
  auto launch = [&]<typename Config>() {
    fused_norm_rope_pcp_combine_kernel<Config><<<blocks, 128, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(received_payload.const_data_ptr()),
        epoch.const_data_ptr<int64_t>(),
        mla_slot_mapping.const_data_ptr<int64_t>(),
        index_slot_mapping.const_data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(mla_cache.mutable_data_ptr()),
        mla_cache.stride(0), mla_cache.stride(1),
        reinterpret_cast<uint8_t*>(index_cache.mutable_data_ptr()),
        has_indexer ? index_cache.stride(0) : 0, static_cast<int>(world_size),
        static_cast<int>(local_tokens), static_cast<int>(num_decode_tokens),
        static_cast<int>(max_local_tokens), static_cast<int>(mla_cache.size(1)),
        has_indexer);
  };
  if (fp8_ds_mla) {
    using Config =
        DSV32FusedNormRopeConfig<1536, true, MlaCacheLayout::kFp8DsMla>;
    launch.template operator()<Config>();
  } else {
    using Config =
        DSV32FusedNormRopeConfig<1536, true, MlaCacheLayout::kStaticE4M3>;
    launch.template operator()<Config>();
  }
  check_cuda_launch("fused_norm_rope_pcp_combine");
}

void fused_norm_rope_pcp(
    const torch::stable::Tensor& positions, const torch::stable::Tensor& q_c,
    const torch::stable::Tensor& q_weight, double q_eps,
    torch::stable::Tensor& q_out, const torch::stable::Tensor& kv_c,
    const torch::stable::Tensor& kv_weight, torch::stable::Tensor& kv_out,
    const torch::stable::Tensor& mla_k_scale, double kv_eps,
    const torch::stable::Tensor& k_pe, torch::stable::Tensor& k_pe_out,
    const torch::stable::Tensor& k_pe_cos_sin,
    const std::optional<torch::stable::Tensor>& index_k,
    const std::optional<torch::stable::Tensor>& index_weight,
    const std::optional<torch::stable::Tensor>& index_bias, double index_eps,
    const std::optional<torch::stable::Tensor>& index_cos_sin,
    torch::stable::Tensor& topk_indices,
    torch::stable::Tensor& received_payload,
    torch::stable::Tensor& received_signal, torch::stable::Tensor& completion,
    torch::stable::Tensor& epoch, torch::stable::Tensor& phase,
    const torch::stable::Tensor& mla_slot_mapping,
    const std::optional<torch::stable::Tensor>& index_slot_mapping,
    torch::stable::Tensor& mla_cache,
    std::optional<torch::stable::Tensor> index_cache, int64_t num_decode_tokens,
    int64_t rank, bool fp8_ds_mla, bool index_rope_interleave,
    int64_t payload_mc_ptr, int64_t signal_mc_ptr) {
  bool has_indexer = index_k.has_value();
  STD_TORCH_CHECK(
      index_weight.has_value() == has_indexer &&
          index_bias.has_value() == has_indexer &&
          index_cos_sin.has_value() == has_indexer &&
          index_slot_mapping.has_value() == has_indexer &&
          index_cache.has_value() == has_indexer,
      "indexer tensors must either all be present or all be absent");
  STD_TORCH_CHECK(received_payload.dim() == 4,
                  "received_payload must be rank four");

  const auto& index_k_arg = has_indexer ? *index_k : q_c;
  const auto& index_weight_arg = has_indexer ? *index_weight : mla_k_scale;
  const auto& index_bias_arg = has_indexer ? *index_bias : mla_k_scale;
  const auto& index_cos_sin_arg = has_indexer ? *index_cos_sin : k_pe_cos_sin;
  const auto& index_slot_mapping_arg =
      has_indexer ? *index_slot_mapping : mla_slot_mapping;
  auto& index_cache_arg = has_indexer ? *index_cache : mla_cache;
  int64_t world_size = received_payload.size(1);

  fused_norm_rope_pcp_dispatch(
      positions, q_c, q_weight, q_eps, q_out, kv_c, kv_weight, kv_out,
      mla_k_scale, kv_eps, k_pe, k_pe_out, k_pe_cos_sin, index_k_arg,
      index_weight_arg, index_bias_arg, index_eps, index_cos_sin_arg,
      topk_indices, received_payload, received_signal, completion, epoch, phase,
      world_size, rank, has_indexer, fp8_ds_mla, index_rope_interleave,
      payload_mc_ptr, signal_mc_ptr);
  fused_norm_rope_pcp_combine(received_payload, epoch, mla_slot_mapping,
                              index_slot_mapping_arg, mla_cache,
                              index_cache_arg, q_c.size(0), num_decode_tokens,
                              has_indexer, fp8_ds_mla);
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, fused_norm_rope_pcp_ops) {
  fused_norm_rope_pcp_ops.def(
      "fused_norm_rope_pcp("
      "Tensor positions, Tensor q_c, Tensor q_weight, float q_eps, "
      "Tensor! q_out, Tensor kv_c, Tensor kv_weight, Tensor! kv_out, "
      "Tensor mla_k_scale, float kv_eps, Tensor k_pe, Tensor! k_pe_out, "
      "Tensor k_pe_cos_sin, Tensor? index_k, "
      "Tensor? index_weight, Tensor? index_bias, float index_eps, "
      "Tensor? index_cos_sin, Tensor! topk_indices, Tensor! received_payload, "
      "Tensor! received_signal, Tensor! completion, Tensor! epoch, "
      "Tensor! phase, Tensor mla_slot_mapping, Tensor? index_slot_mapping, "
      "Tensor! mla_cache, Tensor!? index_cache, int num_decode_tokens, int "
      "rank, bool fp8_ds_mla, bool index_rope_interleave, "
      "int payload_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, fused_norm_rope_pcp_ops) {
  fused_norm_rope_pcp_ops.impl("fused_norm_rope_pcp",
                               TORCH_BOX(&fused_norm_rope_pcp));
}
