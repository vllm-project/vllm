// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#ifndef CHECK_CUDA_CALL
  #define CHECK_CUDA_CALL(call)                                             \
    do {                                                                    \
      cudaError_t err = call;                                               \
      if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        throw std::runtime_error(                                           \
            std::string("CUDA error in file '") + __FILE__ + "' in line " + \
            std::to_string(__LINE__) + " : " + cudaGetErrorString(err));    \
      }                                                                     \
    } while (0)
#endif

namespace lmc {

// inline helper to check HND layout (callable from device and host)
__host__ __device__ __forceinline__ bool is_hnd(
    const EngineKVFormat engine_kv_format) {
  return engine_kv_format ==
             EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||  // flash attn HND
         engine_kv_format ==
             EngineKVFormat::NL_X_NB_TWO_NH_BS_HS;  // flash infer HND
}

// All paged (non-MLA) formats rely on block_size for offset computation.
// The blocked-scale indexer format is MLA-like for kv_size but its paged
// addressing is per-block, so it needs block_size too.
inline void check_block_size(const EngineKVFormat engine_kv_format,
                             const int block_size) {
  TORCH_CHECK((is_mla(engine_kv_format) &&
               engine_kv_format != EngineKVFormat::NL_X_NB_BSV_BSS) ||
                  block_size > 0,
              "block_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

// HND formats additionally need head_size to decompose scalar offsets.
inline void check_head_size(const EngineKVFormat engine_kv_format,
                            const int head_size) {
  TORCH_CHECK(!is_hnd(engine_kv_format) || head_size > 0,
              "head_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

template <typename scalar_t>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key_value,  // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,    // [num_blocks, block_size,
                                               // num_heads, head_size]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, block_size,
                                               // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_cache[src_key_value_idx];
    scalar_t tgt_value = value_cache[src_key_value_idx];

    key_value[tgt_key_idx] = tgt_key;
    key_value[tgt_value_idx] = tgt_value;
  }
}

template <typename scalar_t>
__global__ void reshape_and_cache_back_flash_kernel(
    const scalar_t* __restrict__ key_value,  // [num_tokens, num_heads,
                                             // head_size]
    scalar_t* __restrict__ key_cache,    // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ value_cache,  // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_value[tgt_key_idx];
    scalar_t tgt_value = key_value[tgt_value_idx];

    key_cache[src_key_value_idx] = tgt_key;
    value_cache[src_key_value_idx] = tgt_value;
  }
}

template <typename scalar_t, EngineKVFormat format>
__global__ void single_layer_kv_transfer_kernel(
    scalar_t* __restrict__ lmc_key_value_cache,
    scalar_t* __restrict__ vllm_key_value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  constexpr bool USE_MLA = (format == EngineKVFormat::NL_X_NB_BS_HS);
  constexpr bool HND_LAYOUT = (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
                               format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;

    int64_t vllm_key_idx;
    if constexpr (HND_LAYOUT) {
      // HND layout: [..., num_heads, block_size, head_size]
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     head_idx * block_size * head_size_in_64bit +
                     block_offset * head_size_in_64bit + head_offset;
    } else {
      // NHD layout: [..., block_size, num_heads, head_size]
      // (also correct for MLA where num_heads==1)
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     block_offset * num_heads * head_size_in_64bit +
                     head_idx * head_size_in_64bit + head_offset;
    }

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}

template <EngineKVFormat format>
__device__ __forceinline__ int64_t page_buffer_offset(
    const int k_or_v, const int token_idx, const int scalar_offset,
    const int scalars_per_token, const int page_buffer_size,
    const int block_size, const int head_size) {
  /*
  logical semantics of arguments (agnostic to physical format):
  k_or_v:            0 for key, 1 for value
  token_idx:         flat slot index from slot_mapping[] = block_id * block_size
  + offset_in_block scalar_offset:     thread-loop index in [0,
  scalars_per_token), flat offset within one token's data (NH*HS in xword units)
  scalars_per_token: NH * HS in xword units — total data elements per token slot
  page_buffer_size:  NB * BS — total token slots in the paged buffer
  block_size:        BS — number of token slots per block
  head_size:         HS in xword units — only used by HND formats to decompose
  scalar_offset into (head_idx, head_offset)

  The job of page_buffer_offset is to translate these logical arguments into a
  physical address based on the EngineKVFormat.

  NOTE(perf): For HND formats, threads within a warp access non-contiguous
  addresses when crossing head boundaries (stride BS*HS between heads),
  harming memory coalescing
  TODO: A dedicated HND kernel could launch with
  grid=(2, L, T*NH) thread=(HS,,) to keep warps within one head's contiguous
  HS run
  However, most models have head_size = 128 using bf16 or fp16 which is 256
  bytes when divided by xwords (8 bytes) is exactly 32 xwords which fits one
  warp Worst case if HS is smaller or quantization is smaller, we will have to
  make two vectorized loads per warp that are BS * HS (the head stride) apart
  */

  // vllm cross layer
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash attention (NHD)
  else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash infer (NHD)
  else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    return block_idx * 2 * block_size * scalars_per_token +
           k_or_v * block_size * scalars_per_token +
           block_offset * scalars_per_token + scalar_offset;
  }
  // MLA formats: vLLM (NL_X_NB_BS_HS) and SGLang (NL_X_NBBS_ONE_HS)
  else if constexpr (format == EngineKVFormat::NL_X_NB_BS_HS ||
                     format == EngineKVFormat::NL_X_NBBS_ONE_HS) {
    return token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash attention (HND) — physical: [2, NB, NH, BS, HS]
  else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    const int head_idx = scalar_offset / head_size;
    const int head_offset = scalar_offset % head_size;
    const int num_heads = scalars_per_token / head_size;
    return k_or_v * page_buffer_size * scalars_per_token +
           block_idx * num_heads * block_size * head_size +
           head_idx * block_size * head_size + block_offset * head_size +
           head_offset;
  }
  // vllm flash infer (HND) — physical: [NB, 2, NH, BS, HS]
  else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    const int head_idx = scalar_offset / head_size;
    const int head_offset = scalar_offset % head_size;
    const int num_heads = scalars_per_token / head_size;
    return block_idx * 2 * num_heads * block_size * head_size +
           k_or_v * num_heads * block_size * head_size +
           head_idx * block_size * head_size + block_offset * head_size +
           head_offset;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS ||
                       format == EngineKVFormat::NL_X_NB_NH_BS_CS) {
    const int hs2 = 2 * head_size;  // packed K+V width per head (xword units)
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    const int head_idx = scalar_offset / hs2;
    const int head_offset = scalar_offset % hs2;
    const int num_heads = scalars_per_token / hs2;
    return block_idx * num_heads * block_size * hs2 +
           head_idx * block_size * hs2 + block_offset * hs2 + head_offset;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_NH_TWO_HS ||
                       format == EngineKVFormat::NL_X_NB_BS_NH_CS) {
    return token_idx * scalars_per_token + scalar_offset;
  }
  // DSA indexer: page [BSxvals][BSxscales]; 4B units, so scale == 1 unit
  else if constexpr (format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    const int64_t block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    const int vals = scalars_per_token - 1;
    const int64_t block_base =
        block_idx * static_cast<int64_t>(block_size) * scalars_per_token;
    if (scalar_offset < vals) {
      return block_base + static_cast<int64_t>(block_offset) * vals +
             scalar_offset;
    }
    return block_base + static_cast<int64_t>(block_size) * vals + block_offset;
  }
}

__device__ __forceinline__ int64_t page_buffer_offset_unilateral(
    const int token_idx, const int scalar_offset, const int scalars_per_token) {
  return token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int64_t
key_value_offset(const int k_or_v, const int layer_idx, const int token_idx,
                 const int scalar_offset, const int scalars_per_token,
                 const int num_tokens, const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

template <typename scalar_t>
__global__ void single_layer_kv_transfer_sgl_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,  // [num_tokens, 2,
                                                 // num_heads*head_size]
                                                 // or
                                                 // [2, num_tokens,
                                                 // num_heads*head_size]
    scalar_t* __restrict__ sgl_key_cache,        // [num_blocks, block_size,
                                                 // num_heads, head_size]
    scalar_t* __restrict__ sgl_value_cache,      // [num_blocks, block_size,
                                                 // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,    // [num_tokens]
    const int block_stride_in_64bit, const int lmc_stride,
    const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;
    const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t sgl_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    if (direction == TransferDirection::D2H) {
      lmc_key_value_cache[lmc_key_idx] = sgl_key_cache[sgl_key_value_idx];
      lmc_key_value_cache[lmc_value_idx] = sgl_value_cache[sgl_key_value_idx];
    } else {  // direction == TransferDirection::H2D
      sgl_key_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_key_idx];
      sgl_value_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_value_idx];
    }
  }
}

// One chunk of a fused transfer. The whole pack travels by value in the
// kernel-arg buffer (a full pack fits the 4 KB limit), so a fused launch
// needs no device-side parameter upload.
template <typename scalar_t>
struct FusedTransferChunk {
  scalar_t* key_value;          // chunk's contiguous temp buffer base
  const int64_t* slot_mapping;  // device int64*, this chunk's slice
  int n_tok;                    // tokens to move (0 for unused pack tail)
};

template <typename scalar_t>
struct FusedTransferPack {
  FusedTransferChunk<scalar_t> chunks[MAX_FUSED_TRANSFER_CHUNKS];
};

/**
 * Fused multi-chunk variant of load_and_reshape_multi_layer_kernel below:
 * one launch moves up to MAX_FUSED_TRANSFER_CHUNKS same-geometry chunks,
 * replacing the per-chunk launch storm.
 * chunk = pack.chunks[block.z / k_or_v_size], k_or_v = block.z % k_or_v_size
 * slot_id = chunk.slot_mapping[block.x]
 * chunk.key_value[k_or_v, block.y, block.x, thread.x] <=>
 *     ptrs[block.y][k_or_v, slot_id, thread.x]
 * Chunks shorter than grid.x early-out on their n_tok.
 */
template <typename scalar_t, bool DIRECTION, EngineKVFormat format>
__global__ void load_and_reshape_multi_layer_fused_kernel(
    const FusedTransferPack<scalar_t> pack,
    scalar_t** __restrict__ paged_buffer_ptrs, const int k_or_v_size,
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size, const int block_size, const int head_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int chunk_id = blockIdx.z / k_or_v_size;
  const int k_or_v = blockIdx.z % k_or_v_size;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const FusedTransferChunk<scalar_t>& chunk = pack.chunks[chunk_id];
  if (token_id >= chunk.n_tok) {
    return;
  }
  const int64_t slot_idx = chunk.slot_mapping[token_id];
  scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];
  if (slot_idx < 0) {
    return;
  }

  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);
    const int64_t vllm_offset =
        page_buffer_offset<format>(k_or_v, slot_idx, i, scalars_per_token,
                                   page_buffer_size, block_size, head_size);
    if (DIRECTION)
      chunk.key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
    else
      paged_buffer_ptr[vllm_offset] = chunk.key_value[lmcache_offset];
  }
}

/**
 * Quickly load KV cache between vLLM paged memory and offloading buffer
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] <=> ptrs[block.y][block.z,
 * slot_id, thread.x]
 */
template <typename scalar_t, bool DIRECTION, EngineKVFormat format>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers] * [2,
                                                // PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
                                                // or
                                                // [num_layers] * [num_blocks,
                                                // 2, block_size,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size, const int block_size, const int head_size,
    const int skip_prefix_n_tokens) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int kv_token_id = token_id + skip_prefix_n_tokens;
  const int64_t slot_idx = slot_mapping[kv_token_id];
  scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, kv_token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t vllm_offset =
        page_buffer_offset<format>(k_or_v, slot_idx, i, scalars_per_token,
                                   page_buffer_size, block_size, head_size);

    if (DIRECTION)  // 1 is paged buffer to LMCache
      key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
    else  // 0 is LMCache to paged buffer
      paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
  }
}

/*
 * handle sglang MHA offload between CPU and GPU
 * DIRECTION = 1 (true) means paged buffer to LMCache (D2H)
 * DIRECTION = 0 (false) means LMCache to paged buffer (H2D)
 */
template <typename scalar_t, bool DIRECTION>
__global__ void load_and_reshape_multi_layer_kernel_unilateral(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers *2] *
                                                // [PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  scalar_t* key_ptr = paged_buffer_ptrs[layer_id];
  scalar_t* value_ptr = paged_buffer_ptrs[layer_id + num_layers];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t sgl_offset =
        page_buffer_offset_unilateral(slot_idx, i, scalars_per_token);

    if (k_or_v == 0) {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = key_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        key_ptr[sgl_offset] = key_value[lmcache_offset];
    } else {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = value_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        value_ptr[sgl_offset] = key_value[lmcache_offset];
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
#define LAUNCH_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)                      \
  lmc::load_and_reshape_multi_layer_kernel<T, DIRECTION, FORMAT>             \
      <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,          \
                                   slot_mapping_ptr, num_xwords, num_tokens, \
                                   num_layers, page_buffer_size, block_size, \
                                   head_size_xword, skip_prefix_n_tokens);   \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

template <typename T>
void multi_layer_kv_transfer_templated(
    torch::Tensor&
        key_value,  // key/value must be on gpu/pinned cpu.
                    // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn.
                    // [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA.
    const torch::Tensor& key_value_ptrs,  // [num_layers]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const EngineKVFormat engine_kv_format,
    const int block_size, const int head_size, const int skip_prefix_n_tokens) {
  T* key_value_ptr = get_kernel_ptr<T, torch::Tensor>(key_value);
  T** page_buffer_ptrs =
      get_kernel_ptr<T*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = key_value.size(2);
  int num_transfer_tokens = num_tokens - skip_prefix_n_tokens;
  int num_origin_elements = key_value.size(3);
  int elements_per_xword = sizeof(T) / key_value.element_size();
  int num_xwords = num_origin_elements / elements_per_xword;
  // head_size is in element units
  // convert to xword units
  // to match scalars_per_token (num_xwords) which is also in xword units.
  int head_size_xword = head_size > 0 ? head_size / elements_per_xword : 0;

  lmc::check_block_size(engine_kv_format, block_size);
  lmc::check_head_size(engine_kv_format, head_size_xword);
  TORCH_CHECK(
      engine_kv_format != EngineKVFormat::NL_X_NB_BSV_BSS || sizeof(T) == 4,
      "NL_X_NB_BSV_BSS requires 4-byte transfer units (row bytes "
      "must be divisible by 4 and not by 8)");

  // Fused packs K+V in the trailing dim (kv_size == 1, like MLA): single pass.
  int k_or_v_size =
      (::is_mla(engine_kv_format) || ::is_fused_packed(engine_kv_format)) ? 1
                                                                          : 2;

  dim3 grid(num_transfer_tokens, num_layers, k_or_v_size);
  dim3 block(std::min(num_xwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    switch (engine_kv_format) {
      case EngineKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NB_BS_HS);
        break;
      case EngineKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NBBS_ONE_HS);
        break;
      case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_TWO_NB_NH_BS_HS);
        break;
      case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);
        break;
      case EngineKVFormat::NL_X_NB_NH_BS_TWO_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_NB_NH_BS_TWO_HS);
        break;
      case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false,
                                  EngineKVFormat::NL_X_NB_BS_NH_TWO_HS);
        break;
      case EngineKVFormat::NL_X_NB_NH_BS_CS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NB_NH_BS_CS);
        break;
      case EngineKVFormat::NL_X_NB_BS_NH_CS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NB_BS_NH_CS);
        break;
      case EngineKVFormat::NL_X_NB_BSV_BSS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NB_BSV_BSS);
        break;
      default:
        throw std::runtime_error("Unsupported EngineKVFormat");
    }
  } else {
    switch (engine_kv_format) {
      case EngineKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case EngineKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NB_BS_HS);
        break;
      case EngineKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NBBS_ONE_HS);
        break;
      case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_TWO_NB_NH_BS_HS);
        break;
      case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);
        break;
      case EngineKVFormat::NL_X_NB_NH_BS_TWO_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_NB_NH_BS_TWO_HS);
        break;
      case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true,
                                  EngineKVFormat::NL_X_NB_BS_NH_TWO_HS);
        break;
      case EngineKVFormat::NL_X_NB_NH_BS_CS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NB_NH_BS_CS);
        break;
      case EngineKVFormat::NL_X_NB_BS_NH_CS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NB_BS_NH_CS);
        break;
      case EngineKVFormat::NL_X_NB_BSV_BSS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NB_BSV_BSS);
        break;
      default:
        throw std::runtime_error("Unsupported EngineKVFormat");
    }
  }
}

#undef LAUNCH_KERNEL_WITH_FORMAT

template <typename T>
void multi_layer_kv_transfer_fused_templated(
    const std::vector<uintptr_t>& key_values,
    const std::vector<uintptr_t>& slot_mappings, const std::vector<int>& n_toks,
    uintptr_t page_buffer_ptrs_raw, const int num_layers,
    const int layout_num_tokens, const int num_origin_elements,
    const int element_size, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size,
    const int head_size) {
  const int n_chunks = static_cast<int>(key_values.size());
  TORCH_CHECK(n_chunks >= 1 && n_chunks <= MAX_FUSED_TRANSFER_CHUNKS,
              "fused transfer chunk count out of range: ", n_chunks);
  TORCH_CHECK(slot_mappings.size() == key_values.size() &&
                  n_toks.size() == key_values.size(),
              "fused transfer parameter vectors must be the same length");

  lmc::FusedTransferPack<T> pack{};
  int max_tok = 0;
  for (int c = 0; c < n_chunks; ++c) {
    pack.chunks[c].key_value = reinterpret_cast<T*>(key_values[c]);
    pack.chunks[c].slot_mapping =
        reinterpret_cast<const int64_t*>(slot_mappings[c]);
    pack.chunks[c].n_tok = n_toks[c];
    max_tok = std::max(max_tok, n_toks[c]);
  }
  // Unused pack tail: n_tok stays 0, every block early-outs.

  int num_tokens = layout_num_tokens;
  int elements_per_xword = sizeof(T) / element_size;
  int num_xwords = num_origin_elements / elements_per_xword;
  int head_size_xword = head_size > 0 ? head_size / elements_per_xword : 0;

  lmc::check_block_size(engine_kv_format, block_size);
  lmc::check_head_size(engine_kv_format, head_size_xword);
  TORCH_CHECK(
      engine_kv_format != EngineKVFormat::NL_X_NB_BSV_BSS || sizeof(T) == 4,
      "NL_X_NB_BSV_BSS requires 4-byte transfer units (row bytes "
      "must be divisible by 4 and not by 8)");

  int k_or_v_size =
      (::is_mla(engine_kv_format) || ::is_fused_packed(engine_kv_format)) ? 1
                                                                          : 2;

  T** page_buffer_ptrs = reinterpret_cast<T**>(page_buffer_ptrs_raw);
  dim3 grid(max_tok, num_layers, k_or_v_size * n_chunks);
  dim3 block(std::min(num_xwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#ifndef LAUNCH_FUSED_WITH_FORMAT
  #define LAUNCH_FUSED_WITH_FORMAT(T_, DIR, FORMAT)                      \
    lmc::load_and_reshape_multi_layer_fused_kernel<T_, DIR, FORMAT>      \
        <<<grid, block, 0, stream>>>(                                    \
            pack, page_buffer_ptrs, k_or_v_size, num_xwords, num_tokens, \
            num_layers, page_buffer_size, block_size, head_size_xword);  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
#ifndef FUSED_FORMAT_SWITCH
  #define FUSED_FORMAT_SWITCH(T_, DIR)                                        \
    switch (engine_kv_format) {                                               \
      case EngineKVFormat::NB_NL_TWO_BS_NH_HS:                                \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NB_NL_TWO_BS_NH_HS) \
        break;                                                                \
      case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_BS_HS:                                     \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NL_X_NB_BS_HS)      \
        break;                                                                \
      case EngineKVFormat::NL_X_NBBS_ONE_HS:                                  \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NL_X_NBBS_ONE_HS)   \
        break;                                                                \
      case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_NH_BS_TWO_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:                              \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR,                                     \
                                 EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)        \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_NH_BS_CS:                                  \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NL_X_NB_NH_BS_CS)   \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_BS_NH_CS:                                  \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NL_X_NB_BS_NH_CS)   \
        break;                                                                \
      case EngineKVFormat::NL_X_NB_BSV_BSS:                                   \
        LAUNCH_FUSED_WITH_FORMAT(T_, DIR, EngineKVFormat::NL_X_NB_BSV_BSS)    \
        break;                                                                \
      default:                                                                \
        throw std::runtime_error("Unsupported EngineKVFormat");               \
    }
#endif
  if (direction == TransferDirection::H2D) {
    FUSED_FORMAT_SWITCH(T, false)
  } else {
    FUSED_FORMAT_SWITCH(T, true)
  }
#undef FUSED_FORMAT_SWITCH
#undef LAUNCH_FUSED_WITH_FORMAT
}

void multi_layer_kv_transfer_fused_ptr(
    const std::vector<uintptr_t>& key_values,
    const std::vector<uintptr_t>& slot_mappings, const std::vector<int>& n_toks,
    uintptr_t page_buffer_ptrs, const int num_layers,
    const int layout_num_tokens, const int num_origin_elements,
    const int element_size, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size,
    const int head_size) {
  int copy_size = num_origin_elements * element_size;
#ifndef LAUNCH_FUSED_TRANSFER
  #define LAUNCH_FUSED_TRANSFER(type)                                         \
    do {                                                                      \
      multi_layer_kv_transfer_fused_templated<type>(                          \
          key_values, slot_mappings, n_toks, page_buffer_ptrs, num_layers,    \
          layout_num_tokens, num_origin_elements, element_size,               \
          paged_memory_device, page_buffer_size, direction, engine_kv_format, \
          block_size, head_size);                                             \
    } while (0)
#endif
  if (copy_size % 8 == 0) {
    LAUNCH_FUSED_TRANSFER(int64_t);
  } else if (copy_size % 4 == 0) {
    LAUNCH_FUSED_TRANSFER(int32_t);
  } else if (copy_size % 2 == 0) {
    LAUNCH_FUSED_TRANSFER(int16_t);
  } else {
    LAUNCH_FUSED_TRANSFER(int8_t);
  }
#undef LAUNCH_FUSED_TRANSFER
}

/**
 * @see multi_layer_kv_transfer_templated
 */
void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size,
    const int head_size, const int skip_prefix_n_tokens) {
  int num_origin_elements = key_value.size(3);
  int copy_size = num_origin_elements * key_value.element_size();
#ifndef LAUNCH_MULTI_LAYER_KV_TRANSFER
  #define LAUNCH_MULTI_LAYER_KV_TRANSFER(type)                          \
    do {                                                                \
      multi_layer_kv_transfer_templated<type>(                          \
          key_value, key_value_ptrs, slot_mapping, paged_memory_device, \
          page_buffer_size, direction, engine_kv_format, block_size,    \
          head_size, skip_prefix_n_tokens);                             \
    } while (0)
#endif
  if (copy_size % 8 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int64_t);
  } else if (copy_size % 4 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int32_t);
  } else if (copy_size % 2 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int16_t);
  } else {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int8_t);
  }
#undef LAUNCH_MULTI_LAYER_KV_TRANSFER
}

/**
 * Quickly offload KV cache from SGLang paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in SGLang's K/V buffer has a shape of
 * [PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer_unilateral(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA key/value must be on gpu/pinned cpu

    const torch::Tensor& key_value_ptrs,  // [num_layers*2]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const EngineKVFormat engine_kv_format) {
  const bool use_mla = ::is_mla(engine_kv_format);
  // MLA case collapses back to multi_layer_kv_transfer
  // (vLLM and SGLang indexing are compatible)
  if (use_mla) {
    return multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                                   paged_memory_device, page_buffer_size,
                                   direction, engine_kv_format);
  }

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;

  dim3 grid(key_value.size(2), key_value.size(1), k_or_v_size);
  dim3 block(std::min(num_qwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, false>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, true>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void single_layer_kv_transfer(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]

    // torch::Tensor&
    //     vllm_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    // torch::Tensor&
    //     vllm_value_cache,  // [num_blocks, block_size, num_heads, head_size]
    //  key_cache/value_cache must be on gpu
    torch::Tensor&
        vllm_key_value_cache,  // NHD: [2, num_blocks, block_size, num_heads,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, block_size, num_heads,
                               //       head_size] for flash infer
                               // HND: [2, num_blocks, num_heads, block_size,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, num_heads, block_size,
                               //       head_size] for flash infer
                               // MLA: [num_blocks, block_size, head_size]

    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction, const EngineKVFormat engine_kv_format,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);
  // int64_t* vllm_value_cache_ptr =
  //     get_kernel_ptr<int64_t, torch::Tensor>(vllm_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = ::is_mla(engine_kv_format);
  const bool hnd_layout = lmc::is_hnd(engine_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else if (hnd_layout) {
    // HND format: [..., num_heads, block_size, head_size]
    num_heads = vllm_key_value_cache.size(2);
    block_size = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  } else {
    // NHD format: [..., block_size, num_heads, head_size]
    block_size = vllm_key_value_cache.size(2);
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  }

  lmc::check_block_size(engine_kv_format, block_size);
  lmc::check_head_size(engine_kv_format, head_size_in_64bit);

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (engine_kv_format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS ||
             engine_kv_format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  // int block_stride_in_64bit = vllm_key_cache.stride(0) / elements_per_entry;
  // TORCH_CHECK(vllm_key_cache.stride(0) == vllm_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on EngineKVFormat
#define LAUNCH_SINGLE_LAYER_KERNEL(FORMAT)                                     \
  lmc::single_layer_kv_transfer_kernel<int64_t, FORMAT>                        \
      <<<grid, block, 0, stream>>>(                                            \
          lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr, \
          vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,       \
          lmc_value_offset, num_heads, head_size_in_64bit, block_size,         \
          direction);                                                          \
  break;

  switch (engine_kv_format) {
    case EngineKVFormat::NL_X_NB_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_BS_HS)
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
    case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
    case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
    default:
      TORCH_CHECK(false,
                  "Unsupported EngineKVFormat for single_layer_kv_transfer: ",
                  static_cast<int>(engine_kv_format));
  }
#undef LAUNCH_SINGLE_LAYER_KERNEL
}

void load_and_reshape_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens],
    const int layer_idx) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::load_and_reshape_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void reshape_and_cache_back_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const int layer_idx) {
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::reshape_and_cache_back_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void single_layer_kv_transfer_sgl(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]

    torch::Tensor&
        sgl_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        sgl_value_cache,  // [num_blocks, block_size, num_heads, head_size]
                          // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* sgl_key_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_key_cache);
  int64_t* sgl_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / sgl_key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = sgl_key_cache.size(2);
  int head_size_in_64bit = sgl_key_cache.size(3) / elements_per_entry;

  int block_size = sgl_key_cache.size(1);

  int lmc_stride;
  int lmc_value_offset;
  if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int block_stride_in_64bit = sgl_key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(sgl_key_cache.stride(0) == sgl_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sgl_key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::single_layer_kv_transfer_sgl_kernel<int64_t><<<grid, block, 0, stream>>>(
      lmc_key_value_cache_ptr, sgl_key_cache_ptr, sgl_value_cache_ptr,
      slot_mapping_ptr, block_stride_in_64bit, lmc_stride, lmc_value_offset,
      num_heads, head_size_in_64bit, block_size, direction);
}

/**
 * Perform asynchronous memory copy between lmcache host buffer (memory obj)
 * and a device buffer.
 * The copy will be performed asynchronously on the current CUDA stream.
 * They copy will be split into multiple smaller copies based on the host buffer
 * offset and host buffer alignment requirements.
 *
 * @param dest Destination pointer (device or host)
 * @param src Source pointer (device or host)
 * @param nbytes Number of bytes to copy
 * @param direction H2D or D2H
 * @param host_buffer_offset the virtual offset in the lmcache memory allocator
 * @param host_buffer_alignments the alignment (i.e., cudaHostRegister
 * granularity) requirement of the host buffer. Must be power of two.
 */
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK((host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  cudaMemcpyKind kind = (direction == TransferDirection::H2D)
                            ? cudaMemcpyHostToDevice
                            : cudaMemcpyDeviceToHost;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  while (offset < nbytes) {
    size_t current_src = src + offset;
    size_t current_dest = dest + offset;

    size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    // Use std::min<size_t> so HIP's overload set cannot silently narrow
    // these values to int and produce a garbage length past the 2 GB mark.
    size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    size_t max_nbytes = real_end - offset - host_buffer_offset;

    CHECK_CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(current_dest),
                                    reinterpret_cast<const void*>(current_src),
                                    max_nbytes, kind, stream));

    offset += max_nbytes;
  }
}
