// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HiSparse decode hot-buffer kernels for sparse MLA (GLM-5 / DeepSeek-V3.2).
//
// The swap-in algorithm is a port of SGLang's hisparse
// load_cache_to_device_buffer kernel (sgl jit_kernel/csrc/hisparse.cuh),
// adapted to vLLM addressing:
//  - tokens are keyed by their global KV slot id (block_table-converted
//    indexer output) instead of in-request positions, so no per-request
//    host-location table is needed: host pool row i is global slot i.
//  - each batch row owns a fixed region of `region_stride` hot rows;
//    slots [0, hot_size) are LRU-managed, slot `hot_size` holds the row's
//    newest token (written directly by the KV-cache update).
//
// The kernels only move bytes; they are dtype agnostic.

#include "torch_utils.h"
#include "ops.h"
#include "../cuda_utils.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
// Sentinel in the shared top-k scratch: entry already resolved (hit /
// newest / invalid), no miss handling needed.
constexpr int32_t kTokenDone = -1;
constexpr int32_t kHashEmpty = -1;

__device__ __forceinline__ int32_t hash_slot(int32_t key, int32_t hash_size) {
  // Knuth multiplicative hash for the open-addressing table.
  return static_cast<int32_t>((static_cast<uint32_t>(key) * 2654435761u) %
                              static_cast<uint32_t>(hash_size));
}

// Copy one row of `row_bytes` bytes with a single warp. MLA rows take the
// 16-byte vectorized path; packed indexer rows use the scalar tail-safe path.
__device__ __forceinline__ void copy_row_warp(int lane_id, const char* src,
                                              char* dst, int64_t row_bytes) {
  const auto alignment = reinterpret_cast<uintptr_t>(src) |
                         reinterpret_cast<uintptr_t>(dst) |
                         static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      __stcg(dst4 + j, __ldcg(src4 + j));
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    const unsigned int* src_words = reinterpret_cast<const unsigned int*>(src);
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      __stcg(dst_words + j, __ldcg(src_words + j));
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    __stcg(dst + j, __ldcg(src + j));
  }
}

// Zero one row with a single warp (16B vectorized, L2-only stores), same
// addressing contract as copy_row_warp.
__device__ __forceinline__ void zero_row_warp(int lane_id, char* dst,
                                              int64_t row_bytes) {
  const auto alignment =
      reinterpret_cast<uintptr_t>(dst) | static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    uint64_t* dst8 = reinterpret_cast<uint64_t*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      uint64_t* d = dst8 + j * 2;
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(0ULL),
                   "l"(0ULL)
                   : "memory");
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      __stcg(dst_words + j, 0u);
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    __stcg(dst + j, static_cast<char>(0));
  }
}

__device__ __forceinline__ const char* cache_row_ptr(const char* cache,
                                                     int64_t row,
                                                     int32_t block_size,
                                                     int64_t block_stride,
                                                     int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

__device__ __forceinline__ char* cache_row_ptr(char* cache, int64_t row,
                                               int32_t block_size,
                                               int64_t block_stride,
                                               int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

__device__ __forceinline__ void copy_cache_row_warp(
    int lane_id, const char* src_cache, int64_t src_row, int32_t src_block_size,
    int64_t src_block_stride, char* dst_cache, int64_t dst_row,
    int32_t dst_block_size, int64_t dst_block_stride, int64_t row_bytes,
    int64_t value_bytes) {
  if (value_bytes <= 0) {
    copy_row_warp(lane_id,
                  cache_row_ptr(src_cache, src_row, src_block_size,
                                src_block_stride, row_bytes),
                  cache_row_ptr(dst_cache, dst_row, dst_block_size,
                                dst_block_stride, row_bytes),
                  row_bytes);
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t src_offset = src_row % src_block_size;
  const int64_t dst_offset = dst_row % dst_block_size;
  const char* src_page =
      src_cache + (src_row / src_block_size) * src_block_stride;
  char* dst_page = dst_cache + (dst_row / dst_block_size) * dst_block_stride;
  copy_row_warp(lane_id, src_page + src_offset * value_bytes,
                dst_page + dst_offset * value_bytes, value_bytes);
  copy_row_warp(
      lane_id,
      src_page + src_block_size * value_bytes + src_offset * scale_bytes,
      dst_page + dst_block_size * value_bytes + dst_offset * scale_bytes,
      scale_bytes);
}

__device__ __forceinline__ void zero_cache_row_warp(
    int lane_id, char* cache, int64_t row, int32_t block_size,
    int64_t block_stride, int64_t row_bytes, int64_t value_bytes) {
  if (value_bytes <= 0) {
    zero_row_warp(
        lane_id, cache_row_ptr(cache, row, block_size, block_stride, row_bytes),
        row_bytes);
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t offset = row % block_size;
  char* page = cache + (row / block_size) * block_stride;
  zero_row_warp(lane_id, page + offset * value_bytes, value_bytes);
  zero_row_warp(lane_id, page + block_size * value_bytes + offset * scale_bytes,
                scale_bytes);
}

// In-place inclusive scan over s_data[offset, count) performed by warp 0,
// carrying `accumulator` across calls. Returns the running total.
__device__ __forceinline__ int warp_inclusive_scan(int32_t* s_data, int lane_id,
                                                   int offset, int count,
                                                   int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;
#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  return __shfl_sync(0xffffffff, val, 31);
}

__device__ __forceinline__ int64_t
get_physical_hot_row(const int32_t* hot_block_table, int32_t row,
                     int64_t table_stride, int32_t block_size, int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int64_t>(block) * block_size + slot % block_size;
}

__device__ __forceinline__ void store_hot_index(
    int32_t* hot_indices, int32_t* attention_indices, int32_t index,
    int32_t physical_row, int32_t hot_block_size,
    int64_t attention_block_stride) {
  hot_indices[index] = physical_row;
  if (attention_indices != nullptr) {
    attention_indices[index] =
        physical_row < 0
            ? -1
            : (physical_row / hot_block_size) * attention_block_stride +
                  physical_row % hot_block_size;
  }
}

// One block per batch row.
//
// Shared memory layout (int32 region followed by int16 region):
//   s_topk[top_k]            top-k global ids; reused as miss scratch
//   s_chunk_off[nbc + 1]     prefix sums for hit (then miss) compaction
//   s_evict_off[nbc + 1]     prefix sums for evictable compaction
//   s_hash_keys[hash_size]   open addressing: global id -> top-k index
//   s_counters[3]            hits, phase-1 resolved, valid count
//   s_lru_out[hot_size]      int16, compacted slots: [hits fwd | evict bwd]
//   s_hash_vals[hash_size]   int16 hash values (top-k index)
// Valid global ids must be unique within each row.
__global__ void hisparse_swap_in_kernel(
    const char* __restrict__ host_cache,          // [host_rows, row_bytes]
    char* __restrict__ hot_cache,                 // packed HMA pages
    const int32_t* __restrict__ hot_block_table,  // [max_rows, hot_blocks]
    const int32_t* __restrict__ global_indices,   // global or request-relative
    const int32_t* __restrict__ request_ids,      // [num_rows] or nullptr
    const int32_t* __restrict__ source_block_table,  // [num_reqs, max_blocks]
    const int32_t* __restrict__ resident_block_table,
    int32_t* __restrict__ resolved_global_indices,  // [num_rows, top_k]
    int32_t* __restrict__ valid_counts,             // [num_rows] or nullptr
    int32_t* __restrict__ compact_miss_globals,     // [num_rows, top_k]
    int32_t* __restrict__ compact_miss_hots,        // [num_rows, top_k]
    int32_t* __restrict__ compact_miss_counts,      // [num_rows]
    int32_t* __restrict__ hot_indices,              // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,        // [num_rows, top_k]
    int32_t* __restrict__ miss_mask,  // [num_rows, top_k] or nullptr
    int32_t* __restrict__ device_global_indices,  // [max_rows, region_stride]
    int16_t* __restrict__ lru_slots,              // [max_rows, hot_size]
    const int32_t* __restrict__ request_state_indices,  // [num_rows] or nullptr
    const int64_t host_rows, const int32_t host_block_size,
    const int64_t host_block_stride, const int64_t row_bytes,
    const int64_t row_value_bytes, const int64_t hot_block_stride,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int32_t top_k, const int32_t hot_size, const int32_t hash_size,
    const int64_t region_stride, const int64_t attention_block_stride,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;
  const int num_token_chunks = (top_k + kWarpSize - 1) / kWarpSize;

  const int batch_row = blockIdx.x;
  const int state_row = request_state_indices != nullptr
                            ? request_state_indices[batch_row]
                            : batch_row;
  // V2 publishes -1 for CUDA-graph padding rows.
  if (state_row < 0) {
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
      const int64_t index = static_cast<int64_t>(batch_row) * top_k + i;
      hot_indices[index] = -1;
      if (attention_indices != nullptr) attention_indices[index] = -1;
      if (resolved_global_indices != nullptr) {
        resolved_global_indices[index] = -1;
      }
    }
    if (valid_counts != nullptr && threadIdx.x == 0) {
      valid_counts[batch_row] = 0;
    }
    if (compact_miss_counts != nullptr && threadIdx.x == 0) {
      compact_miss_counts[batch_row] = 0;
    }
    return;
  }
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  const int32_t* row_topk =
      global_indices + static_cast<int64_t>(batch_row) * top_k;
  int32_t* row_out = hot_indices + static_cast<int64_t>(batch_row) * top_k;
  int32_t* row_attention =
      attention_indices != nullptr
          ? attention_indices + static_cast<int64_t>(batch_row) * top_k
          : nullptr;
  int32_t* row_miss = (miss_mask != nullptr)
                          ? miss_mask + static_cast<int64_t>(batch_row) * top_k
                          : nullptr;
  int32_t* row_dgi =
      device_global_indices + static_cast<int64_t>(state_row) * region_stride;
  int16_t* row_lru = lru_slots + static_cast<int64_t>(state_row) * hot_size;

  extern __shared__ char smem_raw[];
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw);
  int32_t* s_chunk_off = s_topk + top_k;
  int32_t* s_evict_off = s_chunk_off + (num_buffer_chunks + 1);
  int32_t* s_hash_keys = s_evict_off + (num_buffer_chunks + 1);
  int32_t* s_counters = s_hash_keys + hash_size;
  int16_t* s_lru_out = reinterpret_cast<int16_t*>(s_counters + 3);
  int16_t* s_hash_vals = s_lru_out + hot_size;

  if (tid < 3) {
    s_counters[tid] = 0;
  }
  for (int i = tid; i < hash_size; i += blockDim.x) {
    s_hash_keys[i] = kHashEmpty;
  }
  for (int i = tid; i < num_buffer_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
    s_evict_off[i] = 0;
  }
  __syncthreads();

  // Phase 1: resolve invalid / newest entries, hash the rest.
  for (int i = tid; i < top_k; i += blockDim.x) {
    const int32_t token_index = row_topk[i];
    int32_t g = token_index;
    int32_t resident_row = -1;
    if (source_block_table != nullptr) {
      const int32_t request_id = request_ids[batch_row];
      const int32_t source_block =
          token_index >= 0 ? token_index / source_block_size : -1;
      if (request_id >= 0 && request_id < source_num_reqs &&
          source_block >= 0 && source_block < source_num_blocks) {
        const int32_t physical_block =
            source_block_table[static_cast<int64_t>(request_id) *
                                   source_bt_stride +
                               source_block];
        g = physical_block >= 0 ? physical_block * source_block_size +
                                      token_index % source_block_size
                                : -1;
      } else {
        g = -1;
      }
      if (resident_block_table != nullptr) {
        const int32_t resident_block =
            token_index >= 0 ? token_index / resident_block_size : -1;
        if (request_id >= 0 && request_id < resident_num_reqs &&
            resident_block >= 0 && resident_block < resident_num_blocks) {
          const int32_t physical_block =
              resident_block_table[static_cast<int64_t>(request_id) *
                                       resident_bt_stride +
                                   resident_block];
          if (physical_block != resident_null_block && physical_block >= 0) {
            resident_row = physical_block * resident_block_size +
                           token_index % resident_block_size;
          }
        }
      }
    }
    if (resolved_global_indices != nullptr) {
      resolved_global_indices[static_cast<int64_t>(batch_row) * top_k + i] = g;
    }
    if (g >= 0) atomicAdd(&s_counters[2], 1);
    if (row_miss != nullptr) row_miss[i] = 0;
    if (resident_row >= 0) {
      store_hot_index(row_out, row_attention, i, resident_row, hot_block_size,
                      attention_block_stride);
      s_topk[i] = kTokenDone;
      atomicAdd(&s_counters[1], 1);
    } else if (g < 0) {
      store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                      attention_block_stride);
      s_topk[i] = kTokenDone;
      atomicAdd(&s_counters[1], 1);
    } else {
      int slot = hash_slot(g, hash_size);
      while (true) {
        const int32_t old = atomicCAS(&s_hash_keys[slot], kHashEmpty, g);
        if (old == kHashEmpty || old == g) {
          s_hash_vals[slot] = static_cast<int16_t>(i);
          break;
        }
        slot = (slot + 1) % hash_size;
      }
      s_topk[i] = g;
    }
  }
  __syncthreads();
  if (valid_counts != nullptr && tid == 0) {
    valid_counts[batch_row] = s_counters[2];
  }
  // Fully resident rows need only request-relative page translation. Avoid
  // scanning or rewriting the hot LRU when no selected row can consult it.
  if (resident_block_table != nullptr && s_counters[1] == top_k) {
    if (tid == 0 && compact_miss_counts != nullptr) {
      compact_miss_counts[batch_row] = 0;
    }
    return;
  }

  // Phase 2: walk hot slots in LRU order, classify hit / evictable, and
  // compact them (hits forward, evictables backward) into s_lru_out.
  const int iters_buffer = (num_buffer_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < iters_buffer; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_buffer_chunks;

    const int pos = chunk_idx * kWarpSize + lane_id;
    const bool has_valid_pos = has_valid_chunk && (pos < hot_size);
    const int16_t slot = has_valid_pos ? row_lru[pos] : int16_t(-1);
    // Corruption tripwire: lru/dgi are long-lived device state; if some
    // external writer (e.g. stray RDMA into reused VRAM) corrupts a slot
    // out of [0, region_stride), treat it as no-hit so it degrades to a re-miss
    // instead of an unbounded dgi read (phases 3/5 bound the write side).
    const int32_t cached_g =
        (slot >= 0 && slot < region_stride) ? row_dgi[slot] : -1;

    int found_topk_idx = -1;
    if (cached_g >= 0) {
      int h = hash_slot(cached_g, hash_size);
      while (true) {
        const int32_t k = s_hash_keys[h];
        if (k == cached_g) {
          found_topk_idx = static_cast<int32_t>(s_hash_vals[h]);
          break;
        }
        if (k == kHashEmpty) break;
        h = (h + 1) % hash_size;
      }
    }
    const bool is_hit = found_topk_idx >= 0;
    const bool is_evictable = has_valid_pos && !is_hit;

    if (is_hit) {
      s_topk[found_topk_idx] = kTokenDone;
      store_hot_index(row_out, row_attention, found_topk_idx,
                      static_cast<int32_t>(get_physical_hot_row(
                          hot_block_table, batch_row, hot_table_stride,
                          hot_block_size, slot)),
                      hot_block_size, attention_block_stride);
    }

    int local_hit_off = 0;
    int local_evict_off = 0;
    if (has_valid_chunk) {
      const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, is_hit);
      const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_hit_off = __popc(hit_mask & lanes_before);
      local_evict_off = __popc(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(hit_mask);
        s_evict_off[chunk_idx + 1] = __popc(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_evict_count);
      if (tid == 0) {
        s_counters[0] = total_hit_count;
      }
    }
    __syncthreads();

    if (is_hit) {
      const int off = s_chunk_off[chunk_idx] + local_hit_off;
      s_lru_out[off] = slot;
    }
    if (is_evictable) {
      const int off = s_evict_off[chunk_idx] + local_evict_off;
      s_lru_out[hot_size - 1 - off] = slot;
    }
  }
  __syncthreads();

  // Reset prefix sums for the miss compaction (token chunks <= buffer
  // chunks because hot_size >= top_k).
  for (int i = tid; i < num_token_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
  }
  __syncthreads();

  // Phase 3: compact misses, assign them eviction slots (oldest first) and
  // record the new ownership in device_global_indices.
  const int iters_token = (num_token_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int miss_running_total = 0;
  for (int iter = 0; iter < iters_token; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_token_chunks;

    const int i = chunk_idx * kWarpSize + lane_id;
    const bool has_valid_token = has_valid_chunk && (i < top_k);

    int32_t g = 0;
    bool is_miss = false;
    if (has_valid_token) {
      is_miss = s_topk[i] != kTokenDone;
      if (is_miss) {
        g = s_topk[i];
      }
    }

    int local_miss_off = 0;
    if (has_valid_chunk) {
      const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
      local_miss_off = __popc(miss_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(miss_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      miss_running_total =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_token_chunks + 1, miss_running_total);
    }
    __syncthreads();

    if (is_miss) {
      const int m = s_chunk_off[chunk_idx] + local_miss_off;
      const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
      if (evict_slot < 0 || evict_slot >= region_stride) {
        // Corruption tripwire: an out-of-range slot (corrupted lru state,
        // see phase 2) must not become a hot-cache/dgi write. Resolve the
        // entry as invalid (-1, masked by attention, not in the miss set);
        // it re-misses on a later step. Phase 5 re-checks the same
        // s_lru_out value, so its copy is skipped consistently.
        store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                        attention_block_stride);
        if (compact_miss_globals != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          compact_miss_globals[compact_index] = g;
          compact_miss_hots[compact_index] = -1;
        }
      } else {
        // Reuse s_topk as compacted miss scratch: m < i always holds (done
        // entries are skipped), so writes never overrun pending reads.
        s_topk[m] = g;
        const int32_t physical_row = static_cast<int32_t>(
            get_physical_hot_row(hot_block_table, batch_row, hot_table_stride,
                                 hot_block_size, evict_slot));
        store_hot_index(row_out, row_attention, i, physical_row, hot_block_size,
                        attention_block_stride);
        if (compact_miss_globals != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          compact_miss_globals[compact_index] = g;
          compact_miss_hots[compact_index] = physical_row;
        }
        if (row_miss != nullptr) row_miss[i] = 1;
        row_dgi[evict_slot] = g;
      }
    }
  }
  __syncthreads();

  const int total_hits = s_counters[0];
  const int total_misses = top_k - total_hits - s_counters[1];
  if (compact_miss_counts != nullptr && tid == 0) {
    compact_miss_counts[batch_row] = total_misses;
  }
  // Phase 4: write back the LRU order: stale evictables at the front,
  // freshly loaded misses next, then hits at MRU.
  const int total_evictable = hot_size - total_hits;
  const int remaining_evictable = total_evictable - total_misses;
  for (int i = tid; i < hot_size; i += blockDim.x) {
    if (i < remaining_evictable) {
      row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    } else if (i < remaining_evictable + total_misses) {
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining_evictable)];
    } else {
      row_lru[i] = s_lru_out[i - remaining_evictable - total_misses];
    }
  }

  // Phase 5: copy missed rows from the host pool, one warp per miss.
  for (int m = warp_id; m < total_misses; m += NUM_WARPS) {
    const int32_t g = s_topk[m];
    const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
    if (evict_slot < 0 || evict_slot >= region_stride) {
      // Corruption tripwire (see phase 3): phase 3 skipped this entry, so
      // s_topk[m] is stale; skip the copy instead of writing out of range.
      continue;
    }
    const int64_t physical_row =
        get_physical_hot_row(hot_block_table, batch_row, hot_table_stride,
                             hot_block_size, evict_slot);
    if (g >= 0 && g < host_rows) {
      copy_cache_row_warp(lane_id, host_cache, g, host_block_size,
                          host_block_stride, hot_cache, physical_row,
                          hot_block_size, hot_block_stride, row_bytes,
                          row_value_bytes);
    } else {
      // No source row for g: never serve the evicted slot's stale bytes
      // as g. Zero the row (deterministic, visible in output quality) and
      // withdraw the phase-3 ownership claim so later steps re-miss instead
      // of hitting the unfilled slot.
      zero_cache_row_warp(lane_id, hot_cache, physical_row, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
      __syncwarp();
      if (lane_id == 0) {
        row_dgi[evict_slot] = -1;
      }
    }
  }
}

// Shared-layer plan replay. Given a plan (hot_indices + miss_mask) already
// computed by the group's "full" layer via hisparse_swap_in, gather THIS
// layer's own missed KV rows into the planned hot slots. No LRU resolution:
// index-sharing shared layers see identical global_indices/hot_indices, so the
// slot assignment is identical -- only the per-layer bytes differ. Fixed shape
// (num_rows x top_k), so it is CUDA-graph-capture safe.
__global__ void hisparse_gather_plan_kernel(
    const char* __restrict__ host_cache,         // [host_rows, row_bytes]
    char* __restrict__ hot_cache,                // [hot_rows, row_bytes]
    const int32_t* __restrict__ global_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ hot_indices,  // [num_rows, top_k] abs hot rows
    const int32_t* __restrict__ miss_mask,    // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ request_state_indices,  // [num_rows] or nullptr
    const int64_t host_rows, const int32_t host_block_size,
    const int64_t host_block_stride, const int64_t hot_rows,
    const int64_t row_bytes, const int64_t row_value_bytes,
    const int64_t hot_block_stride, const int32_t hot_block_size,
    const int32_t top_k, const int64_t attention_block_stride) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  // Columns are interleaved across gridDim.y blocks per row so few-row
  // launches (local-prefill staging's single-row layout, small decode
  // batches) still fill the device with outstanding host reads instead of
  // starving on one block per row.
  const int row = blockIdx.x;
  const bool is_padding =
      request_state_indices != nullptr && request_state_indices[row] < 0;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < top_k; col += col_stride) {
    const int32_t dst = hot_indices[base + col];
    if (lane_id == 0 && attention_indices != nullptr) {
      attention_indices[base + col] =
          is_padding
              ? -1
              : (dst < 0 ? -1
                         : (dst / hot_block_size) * attention_block_stride +
                               dst % hot_block_size);
    }
    if (is_padding) {
      continue;
    }
    if (miss_mask[base + col] == 0) {
      continue;
    }
    const int32_t g = global_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_cache_row_warp(lane_id, host_cache, g, host_block_size,
                          host_block_stride, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    } else {
      // No source row for g: zero the planned slot rather than serving
      // whatever bytes it held (see the swap-in kernel's phase 5).
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    }
  }
}

// Replay only the compact miss prefix produced by hisparse_swap_in_kernel.
// The leader's attention indices are shared directly by index-sharing layers,
// so hits require no work here.
__global__ void hisparse_gather_compact_kernel(
    const char* __restrict__ host_cache, char* __restrict__ hot_cache,
    const int32_t* __restrict__ miss_global_indices,
    const int32_t* __restrict__ miss_hot_indices,
    const int32_t* __restrict__ miss_counts, const int64_t host_rows,
    const int32_t host_block_size, const int64_t host_block_stride,
    const int64_t hot_rows, const int64_t row_bytes,
    const int64_t row_value_bytes, const int64_t hot_block_stride,
    const int32_t hot_block_size, const int32_t top_k) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int row = blockIdx.x;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int miss_count = min(max(miss_counts[row], 0), top_k);
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < miss_count; col += col_stride) {
    const int32_t g = miss_global_indices[base + col];
    const int32_t dst = miss_hot_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_cache_row_warp(lane_id, host_cache, g, host_block_size,
                          host_block_stride, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    } else {
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    }
  }
}

// One warp per item: gather `src_cache[src_indices[i]]` into
// `host_cache[dst_slots[i]]`. Used to scatter KV rows into the pinned host
// pool fully stream-ordered (no host synchronization).
__global__ void hisparse_backup_kernel(
    const char* __restrict__ src_cache, const int64_t* __restrict__ src_indices,
    char* __restrict__ host_cache, const int64_t* __restrict__ dst_slots,
    const int64_t row_bytes, const int64_t row_value_bytes,
    const int64_t src_block_stride, const int32_t src_block_size,
    const int64_t host_block_stride, const int32_t host_block_size,
    const int32_t num_items, const int64_t src_rows, const int64_t host_rows) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warp_id = blockIdx.x * NUM_WARPS + threadIdx.x / kWarpSize;
  const int total_warps = gridDim.x * NUM_WARPS;

  for (int i = warp_id; i < num_items; i += total_warps) {
    const int64_t s = src_indices[i];
    const int64_t d = dst_slots[i];
    if (s < 0 || s >= src_rows || d < 0 || d >= host_rows) {
      continue;
    }
    copy_cache_row_warp(lane_id, src_cache, s, src_block_size, src_block_stride,
                        host_cache, d, host_block_size, host_block_stride,
                        row_bytes, row_value_bytes);
  }
}

// One warp per (layer, item): scatter the current decode row from every hot
// cache layer into its pinned host source. Layer metadata is immutable and
// uploaded once at initialization, so the decode hot path needs one launch and
// no pointer-table construction.
__global__ void hisparse_backup_layers_kernel(
    const char* __restrict__ hot_backing,
    const int64_t* __restrict__ layer_offsets,
    const uint64_t* __restrict__ src_indices_ptrs,
    const uint64_t* __restrict__ host_cache_ptrs,
    const int64_t* __restrict__ dst_slots, const int64_t row_bytes,
    const int64_t row_value_bytes, const int64_t src_block_stride,
    const int32_t src_block_size, const int64_t host_block_stride,
    const int32_t host_block_size, const int32_t num_items,
    const int32_t num_layers, const int64_t src_rows, const int64_t host_rows) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * NUM_WARPS + threadIdx.x / kWarpSize;
  const int64_t total_warps = static_cast<int64_t>(gridDim.x) * NUM_WARPS;
  const int64_t total_items = static_cast<int64_t>(num_layers) * num_items;

  for (int64_t index = warp_id; index < total_items; index += total_warps) {
    const int32_t layer = static_cast<int32_t>(index / num_items);
    const int32_t item = static_cast<int32_t>(index % num_items);
    const auto* src_indices = reinterpret_cast<const int64_t*>(
        static_cast<uintptr_t>(src_indices_ptrs[layer]));
    auto* host_cache =
        reinterpret_cast<char*>(static_cast<uintptr_t>(host_cache_ptrs[layer]));
    const int64_t src_index = src_indices[item];
    const int64_t dst_slot = dst_slots[item];
    if (src_index < 0 || src_index >= src_rows || dst_slot < 0 ||
        dst_slot >= host_rows) {
      continue;
    }
    const char* src_cache = hot_backing + layer_offsets[layer];
    copy_cache_row_warp(lane_id, src_cache, src_index, src_block_size,
                        src_block_stride, host_cache, dst_slot, host_block_size,
                        host_block_stride, row_bytes, row_value_bytes);
  }
}

// Indexer pages store all quantized values first, followed by all per-token
// scales. The apparent [block_size, value_bytes + scale_bytes] tensor shape is
// only an allocation view; a logical token row is split across those regions.
__global__ void hisparse_backup_indexer_kernel(
    const char* __restrict__ src_cache, const int64_t* __restrict__ src_indices,
    char* __restrict__ host_cache, const int64_t* __restrict__ dst_slots,
    const int64_t value_bytes, const int64_t scale_bytes,
    const int64_t src_block_stride, const int64_t dst_block_stride,
    const int32_t block_size, const int32_t num_items, const int64_t src_blocks,
    const int64_t dst_blocks) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warp_id = blockIdx.x * NUM_WARPS + threadIdx.x / kWarpSize;
  const int total_warps = gridDim.x * NUM_WARPS;

  for (int item = warp_id; item < num_items; item += total_warps) {
    const int64_t s = src_indices[item];
    const int64_t d = dst_slots[item];
    if (s < 0 || s >= src_blocks * block_size || d < 0 ||
        d >= dst_blocks * block_size) {
      continue;
    }
    const int64_t src_block = s / block_size;
    const int64_t src_offset = s % block_size;
    const int64_t dst_block = d / block_size;
    const int64_t dst_offset = d % block_size;
    const char* src_page = src_cache + src_block * src_block_stride;
    char* dst_page = host_cache + dst_block * dst_block_stride;
    copy_row_warp(lane_id, src_page + src_offset * value_bytes,
                  dst_page + dst_offset * value_bytes, value_bytes);
    copy_row_warp(
        lane_id, src_page + block_size * value_bytes + src_offset * scale_bytes,
        dst_page + block_size * value_bytes + dst_offset * scale_bytes,
        scale_bytes);
  }
}

// Copy complete cache blocks from the pinned CPU prefix source into newly
// allocated GPU HMA blocks. One warp copies one row.
__global__ void hisparse_copy_blocks_kernel(
    const char* __restrict__ src_cache, char* __restrict__ dst_cache,
    const int32_t* __restrict__ src_block_ids,
    const int32_t* __restrict__ dst_block_ids, const int64_t row_bytes,
    const int64_t src_block_stride, const int64_t dst_block_stride,
    const int32_t block_size, const int32_t num_pairs,
    const int64_t num_src_blocks, const int64_t num_dst_blocks) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warp_id = blockIdx.x * NUM_WARPS + threadIdx.x / kWarpSize;
  const int total_warps = gridDim.x * NUM_WARPS;
  const int num_rows = num_pairs * block_size;

  for (int row = warp_id; row < num_rows; row += total_warps) {
    const int pair = row / block_size;
    const int block_offset = row % block_size;
    const int32_t src_block = src_block_ids[pair];
    const int32_t dst_block = dst_block_ids[pair];
    if (src_block < 0 || src_block >= num_src_blocks || dst_block < 0 ||
        dst_block >= num_dst_blocks) {
      continue;
    }
    copy_row_warp(
        lane_id,
        src_cache + static_cast<int64_t>(src_block) * src_block_stride +
            static_cast<int64_t>(block_offset) * row_bytes,
        dst_cache + static_cast<int64_t>(dst_block) * dst_block_stride +
            static_cast<int64_t>(block_offset) * row_bytes,
        row_bytes);
  }
}

struct CacheLayout {
  int64_t rows;
  int32_t block_size;
  int64_t block_stride;
};

CacheLayout check_cache_rows(const torch::stable::Tensor& t, const char* name,
                             int64_t row_bytes) {
  STD_TORCH_CHECK(t.dim() == 2 || t.dim() == 3, name,
                  " must be a 2D row cache or 3D paged cache");
  STD_TORCH_CHECK(t.size(-1) * t.element_size() == row_bytes, name,
                  " row width mismatch");
  if (t.dim() == 2) {
    STD_TORCH_CHECK(t.is_contiguous(), name, " must have contiguous rows");
    return {t.size(0), 1, row_bytes};
  }
  STD_TORCH_CHECK(t.stride(1) * t.element_size() == row_bytes, name,
                  " rows must be contiguous within each block");
  return {t.size(0) * t.size(1), static_cast<int32_t>(t.size(1)),
          static_cast<int64_t>(t.stride(0) * t.element_size())};
}

}  // namespace

void hisparse_swap_in(
    torch::stable::Tensor const& host_cache, torch::stable::Tensor& hot_cache,
    torch::stable::Tensor const& hot_block_table,
    torch::stable::Tensor const& global_indices,
    torch::stable::Tensor& hot_indices,
    torch::stable::Tensor& device_global_indices,
    torch::stable::Tensor& lru_slots,
    std::optional<torch::stable::Tensor> const& request_state_indices,
    int64_t region_stride,
    std::optional<torch::stable::Tensor> const& miss_mask,
    std::optional<torch::stable::Tensor> const& attention_indices,
    int64_t attention_block_stride,
    std::optional<torch::stable::Tensor> const& request_ids,
    std::optional<torch::stable::Tensor> const& source_block_table,
    int64_t source_block_size,
    std::optional<torch::stable::Tensor> const& resolved_global_indices,
    std::optional<torch::stable::Tensor> const& valid_counts,
    std::optional<torch::stable::Tensor> const& compact_miss_globals,
    std::optional<torch::stable::Tensor> const& compact_miss_hots,
    std::optional<torch::stable::Tensor> const& compact_miss_counts,
    std::optional<torch::stable::Tensor> const& resident_block_table,
    int64_t resident_block_size, int64_t resident_null_block,
    int64_t row_value_bytes) {
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
                  "host_cache must be CPU memory");
  STD_TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  STD_TORCH_CHECK(
      hot_block_table.is_cuda() &&
          hot_block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
          hot_block_table.dim() == 2,
      "hot_block_table must be a 2D int32 CUDA tensor");
  STD_TORCH_CHECK(global_indices.is_cuda() && hot_indices.is_cuda() &&
                      device_global_indices.is_cuda() && lru_slots.is_cuda(),
                  "index tensors must be on CUDA");
  STD_TORCH_CHECK(
      global_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          hot_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "global_indices/hot_indices must be int32");
  STD_TORCH_CHECK(
      device_global_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "device_global_indices must be int32");
  STD_TORCH_CHECK(
      lru_slots.scalar_type() == torch::headeronly::ScalarType::Short,
      "lru_slots must be int16");
  STD_TORCH_CHECK(global_indices.dim() == 2 && global_indices.is_contiguous(),
                  "global_indices must be contiguous 2D");
  STD_TORCH_CHECK(hot_indices.size(0) == global_indices.size(0) &&
                      hot_indices.size(1) == global_indices.size(1) &&
                      hot_indices.is_contiguous(),
                  "hot_indices must match global_indices");
  STD_TORCH_CHECK(
      device_global_indices.dim() == 2 && device_global_indices.is_contiguous(),
      "device_global_indices must be contiguous 2D");
  STD_TORCH_CHECK(
      lru_slots.dim() == 2 &&
          lru_slots.size(0) == device_global_indices.size(0) &&
          lru_slots.is_contiguous(),
      "lru_slots must be contiguous with one row per request state");
  STD_TORCH_CHECK(hot_cache.dim() == 3,
                  "hot_cache must be [num_blocks, block_size, row_width]");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  STD_TORCH_CHECK(row_value_bytes > 0 || row_bytes % 16 == 0,
                  "non-split KV rows must be 16-byte aligned");
  STD_TORCH_CHECK(hot_cache.stride(1) * hot_cache.element_size() == row_bytes,
                  "hot-cache rows must be contiguous");
  const int64_t hot_block_size = hot_cache.size(1);
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  const auto host_layout =
      check_cache_rows(host_cache, "host_cache", row_bytes);
  const int64_t host_rows = host_layout.rows;
  STD_TORCH_CHECK(row_value_bytes == 0 ||
                      (row_value_bytes > 0 && row_value_bytes < row_bytes &&
                       host_layout.block_size == hot_block_size),
                  "split rows require matching paged host/hot block sizes");

  const auto num_rows = static_cast<int32_t>(global_indices.size(0));
  const auto top_k = static_cast<int32_t>(global_indices.size(1));
  const auto hot_size = static_cast<int32_t>(lru_slots.size(1));
  STD_TORCH_CHECK(hot_size >= top_k, "hot buffer size must be >= top_k");
  STD_TORCH_CHECK(hot_size <= 32768, "hot buffer size must fit int16 slots");
  STD_TORCH_CHECK(region_stride == hot_size,
                  "region_stride must match the LRU size");
  STD_TORCH_CHECK(device_global_indices.size(1) == region_stride,
                  "device_global_indices must cover the full hot region");
  STD_TORCH_CHECK(device_global_indices.size(0) >= num_rows,
                  "device_global_indices has too few rows");
  STD_TORCH_CHECK(hot_block_table.size(0) >= num_rows &&
                      hot_block_table.size(1) >=
                          (region_stride + hot_block_size - 1) / hot_block_size,
                  "hot_block_table has too few entries");
  STD_TORCH_CHECK(hot_rows < INT32_MAX, "hot indices must fit int32");

  const int32_t* request_ids_ptr = nullptr;
  const int32_t* source_block_table_ptr = nullptr;
  int64_t source_bt_stride = 0;
  int32_t source_num_reqs = 0;
  int32_t source_num_blocks = 0;
  STD_TORCH_CHECK(
      request_ids.has_value() == source_block_table.has_value(),
      "request_ids and source_block_table must be provided together");
  if (source_block_table.has_value()) {
    auto const& req = request_ids.value();
    auto const& table = source_block_table.value();
    STD_TORCH_CHECK(
        req.is_cuda() &&
            req.scalar_type() == torch::headeronly::ScalarType::Int &&
            req.numel() >= num_rows && req.is_contiguous(),
        "request_ids must be contiguous int32 on CUDA with one entry per row");
    STD_TORCH_CHECK(
        table.is_cuda() &&
            table.scalar_type() == torch::headeronly::ScalarType::Int &&
            table.dim() == 2 && table.stride(1) == 1,
        "source_block_table must be a row-major 2D int32 CUDA tensor");
    STD_TORCH_CHECK(source_block_size > 0,
                    "source_block_size must be positive");
    request_ids_ptr = req.const_data_ptr<int32_t>();
    source_block_table_ptr = table.const_data_ptr<int32_t>();
    source_bt_stride = table.stride(0);
    source_num_reqs = static_cast<int32_t>(table.size(0));
    source_num_blocks = static_cast<int32_t>(table.size(1));
  }

  const int32_t* resident_block_table_ptr = nullptr;
  int64_t resident_bt_stride = 0;
  int32_t resident_num_reqs = 0;
  int32_t resident_num_blocks = 0;
  if (resident_block_table.has_value()) {
    auto const& table = resident_block_table.value();
    STD_TORCH_CHECK(
        source_block_table.has_value() && request_ids.has_value(),
        "resident lookup requires request_ids and source_block_table");
    STD_TORCH_CHECK(
        table.is_cuda() &&
            table.scalar_type() == torch::headeronly::ScalarType::Int &&
            table.dim() == 2 && table.stride(1) == 1,
        "resident_block_table must be a row-major 2D int32 CUDA tensor");
    STD_TORCH_CHECK(resident_block_size == hot_block_size,
                    "resident and hot block sizes must match");
    STD_TORCH_CHECK(resident_null_block >= 0,
                    "resident null block must be non-negative");
    resident_block_table_ptr = table.const_data_ptr<int32_t>();
    resident_bt_stride = table.stride(0);
    resident_num_reqs = static_cast<int32_t>(table.size(0));
    resident_num_blocks = static_cast<int32_t>(table.size(1));
  }

  int32_t* resolved_global_indices_ptr = nullptr;
  if (resolved_global_indices.has_value()) {
    auto const& resolved = resolved_global_indices.value();
    STD_TORCH_CHECK(
        resolved.is_cuda() &&
            resolved.scalar_type() == torch::headeronly::ScalarType::Int &&
            resolved.dim() == 2 && resolved.size(0) == global_indices.size(0) &&
            resolved.size(1) == global_indices.size(1) &&
            resolved.is_contiguous(),
        "resolved_global_indices must be contiguous int32 matching indices");
    resolved_global_indices_ptr = resolved.mutable_data_ptr<int32_t>();
  }

  int32_t* valid_counts_ptr = nullptr;
  if (valid_counts.has_value()) {
    auto const& counts = valid_counts.value();
    STD_TORCH_CHECK(
        counts.is_cuda() &&
            counts.scalar_type() == torch::headeronly::ScalarType::Int &&
            counts.numel() >= num_rows && counts.is_contiguous(),
        "valid_counts must be contiguous int32 on CUDA with one entry per row");
    valid_counts_ptr = counts.mutable_data_ptr<int32_t>();
  }

  int32_t* compact_miss_globals_ptr = nullptr;
  int32_t* compact_miss_hots_ptr = nullptr;
  int32_t* compact_miss_counts_ptr = nullptr;
  const bool has_compact_plan = compact_miss_globals.has_value();
  STD_TORCH_CHECK(has_compact_plan == compact_miss_hots.has_value() &&
                      has_compact_plan == compact_miss_counts.has_value(),
                  "compact miss plan tensors must be provided together");
  if (has_compact_plan) {
    auto const& globals = compact_miss_globals.value();
    auto const& hots = compact_miss_hots.value();
    auto const& counts = compact_miss_counts.value();
    STD_TORCH_CHECK(
        globals.is_cuda() && hots.is_cuda() && counts.is_cuda() &&
            globals.scalar_type() == torch::headeronly::ScalarType::Int &&
            hots.scalar_type() == torch::headeronly::ScalarType::Int &&
            counts.scalar_type() == torch::headeronly::ScalarType::Int &&
            globals.dim() == 2 && hots.dim() == 2 &&
            globals.size(0) == global_indices.size(0) &&
            globals.size(1) == global_indices.size(1) &&
            hots.size(0) == global_indices.size(0) &&
            hots.size(1) == global_indices.size(1) &&
            counts.numel() >= num_rows && globals.is_contiguous() &&
            hots.is_contiguous() && counts.is_contiguous(),
        "compact miss plan must be contiguous int32 matching indices");
    compact_miss_globals_ptr = globals.mutable_data_ptr<int32_t>();
    compact_miss_hots_ptr = hots.mutable_data_ptr<int32_t>();
    compact_miss_counts_ptr = counts.mutable_data_ptr<int32_t>();
  }

  const int32_t* request_state_ptr = nullptr;
  if (request_state_indices.has_value()) {
    auto const& state_indices = request_state_indices.value();
    STD_TORCH_CHECK(
        state_indices.is_cuda() &&
            state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
            state_indices.numel() >= num_rows,
        "request_state_indices must be int32 on CUDA with one entry per row");
    request_state_ptr = state_indices.const_data_ptr<int32_t>();
  }

  // Optional plan output: 1 at columns resolved as a miss (loaded from the
  // host pool this call), 0 elsewhere. Lets index-sharing "shared" layers
  // replay the same gather via hisparse_gather_plan without re-resolving LRU.
  int32_t* miss_mask_ptr = nullptr;
  if (miss_mask.has_value()) {
    auto const& mm = miss_mask.value();
    STD_TORCH_CHECK(
        mm.is_cuda() &&
            mm.scalar_type() == torch::headeronly::ScalarType::Int &&
            mm.dim() == 2 && mm.is_contiguous() &&
            mm.size(0) == global_indices.size(0) &&
            mm.size(1) == global_indices.size(1),
        "miss_mask must be a contiguous int32 CUDA tensor matching "
        "global_indices");
    miss_mask_ptr = mm.mutable_data_ptr<int32_t>();
  }

  int32_t* attention_indices_ptr = nullptr;
  if (attention_indices.has_value()) {
    auto const& indices = attention_indices.value();
    STD_TORCH_CHECK(
        indices.is_cuda() &&
            indices.scalar_type() == torch::headeronly::ScalarType::Int &&
            indices.dim() == 2 && indices.is_contiguous() &&
            indices.size(0) == global_indices.size(0) &&
            indices.size(1) == global_indices.size(1),
        "attention_indices must be a contiguous int32 CUDA tensor matching "
        "global_indices");
    STD_TORCH_CHECK(attention_block_stride >= hot_block_size,
                    "attention block stride must cover one hot block");
    attention_indices_ptr = indices.mutable_data_ptr<int32_t>();
  }

  if (num_rows == 0 || top_k == 0) {
    return;
  }

  constexpr int kBlockSize = 1024;
  const int hash_size = 2 * top_k;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;
  const size_t smem_bytes =
      sizeof(int32_t) * (top_k + 2 * (num_buffer_chunks + 1) + hash_size + 3) +
      sizeof(int16_t) * (hot_size + hash_size);

  const torch::stable::accelerator::DeviceGuard device_guard(
      hot_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  auto kernel = hisparse_swap_in_kernel;
  if (smem_bytes > 48 * 1024) {
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    STD_TORCH_CHECK(attribute_error == cudaSuccess,
                    "failed to configure HiSparse swap-in shared memory: ",
                    cudaGetErrorString(attribute_error));
  }
  kernel<<<num_rows, kBlockSize, smem_bytes, stream>>>(
      static_cast<const char*>(host_cache.const_data_ptr()),
      static_cast<char*>(hot_cache.mutable_data_ptr()),
      hot_block_table.const_data_ptr<int32_t>(),
      global_indices.const_data_ptr<int32_t>(), request_ids_ptr,
      source_block_table_ptr, resident_block_table_ptr,
      resolved_global_indices_ptr, valid_counts_ptr, compact_miss_globals_ptr,
      compact_miss_hots_ptr, compact_miss_counts_ptr,
      hot_indices.mutable_data_ptr<int32_t>(), attention_indices_ptr,
      miss_mask_ptr, device_global_indices.mutable_data_ptr<int32_t>(),
      lru_slots.mutable_data_ptr<int16_t>(), request_state_ptr, host_rows,
      host_layout.block_size, host_layout.block_stride, row_bytes,
      row_value_bytes, hot_block_stride, hot_block_table.stride(0),
      hot_block_size, top_k, hot_size, hash_size, region_stride,
      attention_block_stride, source_bt_stride, source_num_reqs,
      source_num_blocks, static_cast<int32_t>(source_block_size),
      resident_bt_stride, resident_num_reqs, resident_num_blocks,
      static_cast<int32_t>(resident_block_size),
      static_cast<int32_t>(resident_null_block));
  const cudaError_t launch_error = cudaGetLastError();
  STD_TORCH_CHECK(launch_error == cudaSuccess,
                  "HiSparse swap-in kernel launch failed: ",
                  cudaGetErrorString(launch_error));
}

void hisparse_gather_plan(
    torch::stable::Tensor const& host_cache, torch::stable::Tensor& hot_cache,
    torch::stable::Tensor const& global_indices,
    torch::stable::Tensor const& hot_indices,
    torch::stable::Tensor const& miss_mask,
    std::optional<torch::stable::Tensor> const& request_state_indices,
    std::optional<torch::stable::Tensor> const& attention_indices,
    int64_t attention_block_stride, int64_t row_value_bytes) {
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
                  "host_cache must be CPU memory");
  STD_TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  STD_TORCH_CHECK(
      global_indices.is_cuda() && hot_indices.is_cuda() && miss_mask.is_cuda(),
      "plan tensors must be on CUDA");
  STD_TORCH_CHECK(
      global_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          hot_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          miss_mask.scalar_type() == torch::headeronly::ScalarType::Int,
      "plan tensors must be int32");
  STD_TORCH_CHECK(global_indices.dim() == 2 && global_indices.is_contiguous(),
                  "global_indices must be contiguous 2D");
  STD_TORCH_CHECK(
      hot_indices.size(0) == global_indices.size(0) &&
          hot_indices.size(1) == global_indices.size(1) &&
          miss_mask.size(0) == global_indices.size(0) &&
          miss_mask.size(1) == global_indices.size(1) &&
          hot_indices.is_contiguous() && miss_mask.is_contiguous(),
      "hot_indices/miss_mask must match contiguous 2D global_indices");

  STD_TORCH_CHECK(hot_cache.dim() == 2 || hot_cache.dim() == 3,
                  "hot_cache must be a 2D staging buffer or paged 3D cache");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  STD_TORCH_CHECK(row_value_bytes > 0 || row_bytes % 16 == 0,
                  "non-split KV rows must be 16-byte aligned");
  STD_TORCH_CHECK(
      hot_cache.stride(hot_cache.dim() - 2) * hot_cache.element_size() ==
          row_bytes,
      "hot-cache rows must be contiguous");
  const auto host_layout =
      check_cache_rows(host_cache, "host_cache", row_bytes);
  const int64_t host_rows = host_layout.rows;

  const auto num_rows = static_cast<int32_t>(global_indices.size(0));
  const auto top_k = static_cast<int32_t>(global_indices.size(1));

  const int32_t* request_state_ptr = nullptr;
  if (request_state_indices.has_value()) {
    auto const& state_indices = request_state_indices.value();
    STD_TORCH_CHECK(
        state_indices.is_cuda() &&
            state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
            state_indices.numel() >= num_rows,
        "request_state_indices must be int32 on CUDA with one entry per row");
    request_state_ptr = state_indices.const_data_ptr<int32_t>();
  }

  int32_t* attention_indices_ptr = nullptr;
  if (attention_indices.has_value()) {
    auto const& indices = attention_indices.value();
    STD_TORCH_CHECK(
        indices.is_cuda() &&
            indices.scalar_type() == torch::headeronly::ScalarType::Int &&
            indices.dim() == 2 && indices.is_contiguous() &&
            indices.size(0) == global_indices.size(0) &&
            indices.size(1) == global_indices.size(1),
        "attention_indices must be a contiguous int32 CUDA tensor matching "
        "global_indices");
    attention_indices_ptr = indices.mutable_data_ptr<int32_t>();
  }

  if (num_rows == 0 || top_k == 0) {
    return;
  }

  // Match the swap-in kernel's block size: the gather serves 3 of every 4
  // layers' misses (index-sharing replay), so per-row copy parallelism is
  // the throughput limiter on cold rows.
  constexpr int kBlockSize = 1024;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  // Interleave columns over enough blocks per row to cover the device even
  // for few-row launches (local-prefill staging's single-row layout, small
  // decode batches), keeping >= 1 column per warp.
  constexpr int kTargetBlocks = 256;
  const int max_chunks = std::max(1, (top_k + kNumWarps - 1) / kNumWarps);
  const int num_chunks =
      std::min(max_chunks, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  const int32_t hot_block_size =
      hot_cache.dim() == 3 ? static_cast<int32_t>(hot_cache.size(1)) : 1;
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  STD_TORCH_CHECK(row_value_bytes == 0 ||
                      (row_value_bytes > 0 && row_value_bytes < row_bytes &&
                       host_layout.block_size == hot_block_size),
                  "split rows require matching paged host/hot block sizes");
  const torch::stable::accelerator::DeviceGuard device_guard(
      hot_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_gather_plan_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(host_cache.const_data_ptr()),
      static_cast<char*>(hot_cache.mutable_data_ptr()),
      global_indices.const_data_ptr<int32_t>(),
      hot_indices.const_data_ptr<int32_t>(),
      miss_mask.const_data_ptr<int32_t>(), attention_indices_ptr,
      request_state_ptr, host_rows, host_layout.block_size,
      host_layout.block_stride, hot_rows, row_bytes, row_value_bytes,
      hot_block_stride, hot_block_size, top_k, attention_block_stride);
}

void hisparse_gather_compact(torch::stable::Tensor const& host_cache,
                             torch::stable::Tensor& hot_cache,
                             torch::stable::Tensor const& miss_global_indices,
                             torch::stable::Tensor const& miss_hot_indices,
                             torch::stable::Tensor const& miss_counts,
                             int64_t row_value_bytes) {
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
                  "host_cache must be CPU memory");
  STD_TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  STD_TORCH_CHECK(
      miss_global_indices.is_cuda() && miss_hot_indices.is_cuda() &&
          miss_counts.is_cuda() &&
          miss_global_indices.scalar_type() ==
              torch::headeronly::ScalarType::Int &&
          miss_hot_indices.scalar_type() ==
              torch::headeronly::ScalarType::Int &&
          miss_counts.scalar_type() == torch::headeronly::ScalarType::Int &&
          miss_global_indices.dim() == 2 && miss_hot_indices.dim() == 2 &&
          miss_global_indices.size(0) == miss_hot_indices.size(0) &&
          miss_global_indices.size(1) == miss_hot_indices.size(1) &&
          miss_counts.numel() >= miss_global_indices.size(0) &&
          miss_global_indices.is_contiguous() &&
          miss_hot_indices.is_contiguous() && miss_counts.is_contiguous(),
      "compact miss plan must be matching contiguous int32 CUDA tensors");
  STD_TORCH_CHECK(hot_cache.dim() == 3,
                  "hot_cache must be [num_blocks, block_size, row_width]");
  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  STD_TORCH_CHECK(row_value_bytes > 0 || row_bytes % 16 == 0,
                  "non-split KV rows must be 16-byte aligned");
  STD_TORCH_CHECK(hot_cache.stride(1) * hot_cache.element_size() == row_bytes,
                  "hot-cache rows must be contiguous");
  const auto host_layout =
      check_cache_rows(host_cache, "host_cache", row_bytes);
  const int64_t host_rows = host_layout.rows;
  const auto num_rows = static_cast<int32_t>(miss_global_indices.size(0));
  const auto top_k = static_cast<int32_t>(miss_global_indices.size(1));
  if (num_rows == 0 || top_k == 0) {
    return;
  }

  constexpr int kBlockSize = 512;
  constexpr int kTargetBlocks = 64;
  const int num_chunks = std::min(8, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  const int32_t hot_block_size = static_cast<int32_t>(hot_cache.size(1));
  const int64_t hot_rows = hot_cache.size(0) * hot_block_size;
  const int64_t hot_block_stride =
      hot_cache.stride(0) * hot_cache.element_size();
  STD_TORCH_CHECK(row_value_bytes == 0 ||
                      (row_value_bytes > 0 && row_value_bytes < row_bytes &&
                       host_layout.block_size == hot_block_size),
                  "split rows require matching paged host/hot block sizes");
  const torch::stable::accelerator::DeviceGuard device_guard(
      hot_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_gather_compact_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(host_cache.const_data_ptr()),
      static_cast<char*>(hot_cache.mutable_data_ptr()),
      miss_global_indices.const_data_ptr<int32_t>(),
      miss_hot_indices.const_data_ptr<int32_t>(),
      miss_counts.const_data_ptr<int32_t>(), host_rows, host_layout.block_size,
      host_layout.block_stride, hot_rows, row_bytes, row_value_bytes,
      hot_block_stride, hot_block_size, top_k);
}

void hisparse_backup(torch::stable::Tensor const& src_cache,
                     torch::stable::Tensor const& src_indices,
                     torch::stable::Tensor& host_cache,
                     torch::stable::Tensor const& dst_slots,
                     int64_t row_value_bytes) {
  STD_TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on CUDA");
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
                  "host_cache must be CPU memory");
  STD_TORCH_CHECK(src_indices.is_cuda() && dst_slots.is_cuda(),
                  "src_indices/dst_slots must be on CUDA");
  STD_TORCH_CHECK(
      src_indices.scalar_type() == torch::headeronly::ScalarType::Long &&
          src_indices.is_contiguous(),
      "src_indices must be contiguous int64");
  STD_TORCH_CHECK(
      dst_slots.scalar_type() == torch::headeronly::ScalarType::Long &&
          dst_slots.is_contiguous(),
      "dst_slots must be contiguous int64");
  STD_TORCH_CHECK(src_indices.numel() == dst_slots.numel(),
                  "src_indices and dst_slots must have the same length");

  const int64_t row_bytes = src_cache.size(-1) * src_cache.element_size();
  STD_TORCH_CHECK(src_cache.dim() == 2 || src_cache.dim() == 3,
                  "src_cache must be 2D or paged 3D");
  const int32_t src_block_size =
      src_cache.dim() == 3 ? static_cast<int32_t>(src_cache.size(1)) : 1;
  const int64_t src_rows = src_cache.size(0) * src_block_size;
  const int64_t src_block_stride =
      src_cache.stride(0) * src_cache.element_size();
  STD_TORCH_CHECK(src_cache.stride(-2) * src_cache.element_size() == row_bytes,
                  "src-cache rows must be contiguous");
  const auto host_layout =
      check_cache_rows(host_cache, "host_cache", row_bytes);
  const int64_t host_rows = host_layout.rows;
  STD_TORCH_CHECK(row_value_bytes == 0 ||
                      (row_value_bytes > 0 && row_value_bytes < row_bytes &&
                       host_layout.block_size == src_block_size),
                  "split rows require matching paged source/host block sizes");

  const auto num_items = static_cast<int32_t>(src_indices.numel());
  if (num_items == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  const int grid = (num_items + kNumWarps - 1) / kNumWarps;

  const torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_backup_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(src_cache.const_data_ptr()),
      src_indices.const_data_ptr<int64_t>(),
      static_cast<char*>(host_cache.mutable_data_ptr()),
      dst_slots.const_data_ptr<int64_t>(), row_bytes, row_value_bytes,
      src_block_stride, src_block_size, host_layout.block_stride,
      host_layout.block_size, num_items, src_rows, host_rows);
}

void hisparse_backup_layers(torch::stable::Tensor const& hot_backing,
                            torch::stable::Tensor const& layer_offsets,
                            torch::stable::Tensor const& src_indices_ptrs,
                            torch::stable::Tensor& host_anchor,
                            torch::stable::Tensor const& host_cache_ptrs,
                            torch::stable::Tensor const& dst_slots,
                            int64_t num_items, int64_t src_block_stride,
                            int64_t src_block_size, int64_t src_rows,
                            int64_t row_value_bytes) {
  STD_TORCH_CHECK(hot_backing.is_cuda(), "hot_backing must be on CUDA");
  STD_TORCH_CHECK(hot_backing.is_contiguous(),
                  "hot_backing must be contiguous");
  STD_TORCH_CHECK(host_anchor.device().is_cpu(),
                  "host_anchor must be CPU memory");
  STD_TORCH_CHECK(layer_offsets.is_cuda() && src_indices_ptrs.is_cuda() &&
                      host_cache_ptrs.is_cuda() && dst_slots.is_cuda(),
                  "backup metadata must be on CUDA");
  STD_TORCH_CHECK(
      layer_offsets.scalar_type() == torch::headeronly::ScalarType::Long &&
          layer_offsets.is_contiguous(),
      "layer_offsets must be contiguous int64");
  STD_TORCH_CHECK(
      src_indices_ptrs.scalar_type() == torch::headeronly::ScalarType::UInt64 &&
          host_cache_ptrs.scalar_type() ==
              torch::headeronly::ScalarType::UInt64 &&
          src_indices_ptrs.is_contiguous() && host_cache_ptrs.is_contiguous(),
      "pointer tables must be contiguous uint64");
  STD_TORCH_CHECK(
      dst_slots.scalar_type() == torch::headeronly::ScalarType::Long &&
          dst_slots.is_contiguous(),
      "dst_slots must be contiguous int64");
  STD_TORCH_CHECK(layer_offsets.dim() == 1 && src_indices_ptrs.dim() == 1 &&
                      host_cache_ptrs.dim() == 1 &&
                      layer_offsets.numel() == src_indices_ptrs.numel() &&
                      layer_offsets.numel() == host_cache_ptrs.numel(),
                  "layer metadata must have matching 1D shapes");
  STD_TORCH_CHECK(num_items >= 0 && num_items <= dst_slots.numel(),
                  "num_items exceeds dst_slots");
  STD_TORCH_CHECK(src_block_size > 0 && src_block_size <= INT32_MAX &&
                      src_block_stride > 0 && src_rows > 0,
                  "invalid source-cache layout");

  const int64_t row_bytes = host_anchor.size(-1) * host_anchor.element_size();
  const auto host_layout =
      check_cache_rows(host_anchor, "host_anchor", row_bytes);
  const int64_t host_rows = host_layout.rows;
  STD_TORCH_CHECK(row_value_bytes == 0 ||
                      (row_value_bytes > 0 && row_value_bytes < row_bytes &&
                       host_layout.block_size == src_block_size),
                  "split rows require matching paged source/host block sizes");
  STD_TORCH_CHECK(src_block_stride >= src_block_size * row_bytes &&
                      src_rows % src_block_size == 0,
                  "source-cache layout is inconsistent");
  const auto num_layers = static_cast<int32_t>(layer_offsets.numel());
  if (num_items == 0 || num_layers == 0) {
    return;
  }
  STD_TORCH_CHECK(num_items <= INT32_MAX && num_layers <= INT32_MAX,
                  "backup dimensions must fit int32");

  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  const int64_t total_items = num_items * num_layers;
  const int64_t grid64 = (total_items + kNumWarps - 1) / kNumWarps;
  STD_TORCH_CHECK(grid64 <= INT32_MAX, "backup grid exceeds CUDA limits");

  const torch::stable::accelerator::DeviceGuard device_guard(
      hot_backing.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_backup_layers_kernel<<<static_cast<int>(grid64), kBlockSize, 0,
                                  stream>>>(
      static_cast<const char*>(hot_backing.const_data_ptr()),
      layer_offsets.const_data_ptr<int64_t>(),
      static_cast<const uint64_t*>(src_indices_ptrs.const_data_ptr()),
      static_cast<const uint64_t*>(host_cache_ptrs.const_data_ptr()),
      dst_slots.const_data_ptr<int64_t>(), row_bytes, row_value_bytes,
      src_block_stride, static_cast<int32_t>(src_block_size),
      host_layout.block_stride, host_layout.block_size,
      static_cast<int32_t>(num_items), num_layers, src_rows, host_rows);
}

void hisparse_backup_indexer(torch::stable::Tensor const& src_cache,
                             torch::stable::Tensor const& src_indices,
                             torch::stable::Tensor& host_cache,
                             torch::stable::Tensor const& dst_slots,
                             int64_t value_bytes) {
  STD_TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on CUDA");
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
                  "host_cache must be CPU memory");
  STD_TORCH_CHECK(src_cache.dim() == 3 && host_cache.dim() == 3,
                  "indexer caches must be paged 3D tensors");
  STD_TORCH_CHECK(src_indices.is_cuda() && dst_slots.is_cuda(),
                  "slot tensors must be on CUDA");
  STD_TORCH_CHECK(
      src_indices.scalar_type() == torch::headeronly::ScalarType::Long &&
          dst_slots.scalar_type() == torch::headeronly::ScalarType::Long &&
          src_indices.is_contiguous() && dst_slots.is_contiguous(),
      "slot tensors must be contiguous int64 tensors");
  STD_TORCH_CHECK(src_indices.numel() == dst_slots.numel(),
                  "src_indices and dst_slots must have the same length");
  STD_TORCH_CHECK(src_cache.size(1) == host_cache.size(1),
                  "indexer cache block sizes must match");

  const int64_t src_page_row_bytes =
      src_cache.size(2) * src_cache.element_size();
  const int64_t dst_page_row_bytes =
      host_cache.size(2) * host_cache.element_size();
  STD_TORCH_CHECK(src_page_row_bytes == dst_page_row_bytes,
                  "indexer cache page widths must match");
  STD_TORCH_CHECK(value_bytes > 0 && value_bytes < src_page_row_bytes,
                  "value_bytes must leave a non-empty scale region");
  STD_TORCH_CHECK(
      src_cache.stride(1) * src_cache.element_size() == src_page_row_bytes &&
          host_cache.stride(1) * host_cache.element_size() ==
              dst_page_row_bytes,
      "indexer cache pages must be contiguous within each block");

  const auto num_items = static_cast<int32_t>(src_indices.numel());
  if (num_items == 0) {
    return;
  }
  const auto block_size = static_cast<int32_t>(src_cache.size(1));
  const int64_t scale_bytes = src_page_row_bytes - value_bytes;
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  const int grid = (num_items + kNumWarps - 1) / kNumWarps;

  const torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_backup_indexer_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(src_cache.const_data_ptr()),
      src_indices.const_data_ptr<int64_t>(),
      static_cast<char*>(host_cache.mutable_data_ptr()),
      dst_slots.const_data_ptr<int64_t>(), value_bytes, scale_bytes,
      src_cache.stride(0) * src_cache.element_size(),
      host_cache.stride(0) * host_cache.element_size(), block_size, num_items,
      src_cache.size(0), host_cache.size(0));
}

void hisparse_copy_blocks(torch::stable::Tensor const& src_cache,
                          torch::stable::Tensor& dst_cache,
                          torch::stable::Tensor const& src_block_ids,
                          torch::stable::Tensor const& dst_block_ids) {
  STD_TORCH_CHECK(src_cache.device().is_cpu(), "src_cache must be CPU memory");
  STD_TORCH_CHECK(dst_cache.is_cuda(), "dst_cache must be on CUDA");
  STD_TORCH_CHECK(src_cache.dim() == 3 && dst_cache.dim() == 3,
                  "cache tensors must be paged 3D tensors");
  STD_TORCH_CHECK(src_block_ids.is_cuda() && dst_block_ids.is_cuda(),
                  "block IDs must be on CUDA");
  STD_TORCH_CHECK(
      src_block_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          dst_block_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          src_block_ids.is_contiguous() && dst_block_ids.is_contiguous(),
      "block IDs must be contiguous int32 tensors");
  STD_TORCH_CHECK(src_block_ids.numel() == dst_block_ids.numel(),
                  "source and destination block IDs must have equal length");
  STD_TORCH_CHECK(src_cache.size(1) == dst_cache.size(1),
                  "cache block sizes must match");

  const int64_t row_bytes = src_cache.size(2) * src_cache.element_size();
  STD_TORCH_CHECK(dst_cache.size(2) * dst_cache.element_size() == row_bytes,
                  "cache row widths must match");
  STD_TORCH_CHECK(
      src_cache.stride(1) * src_cache.element_size() == row_bytes &&
          dst_cache.stride(1) * dst_cache.element_size() == row_bytes,
      "cache rows must be contiguous");

  const auto num_pairs = static_cast<int32_t>(src_block_ids.numel());
  if (num_pairs == 0) {
    return;
  }
  const auto block_size = static_cast<int32_t>(src_cache.size(1));
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  const int64_t num_rows = static_cast<int64_t>(num_pairs) * block_size;
  const int grid = static_cast<int>((num_rows + kNumWarps - 1) / kNumWarps);

  const torch::stable::accelerator::DeviceGuard device_guard(
      dst_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_copy_blocks_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(src_cache.const_data_ptr()),
      static_cast<char*>(dst_cache.mutable_data_ptr()),
      src_block_ids.const_data_ptr<int32_t>(),
      dst_block_ids.const_data_ptr<int32_t>(), row_bytes,
      src_cache.stride(0) * src_cache.element_size(),
      dst_cache.stride(0) * dst_cache.element_size(), block_size, num_pairs,
      src_cache.size(0), dst_cache.size(0));
}
