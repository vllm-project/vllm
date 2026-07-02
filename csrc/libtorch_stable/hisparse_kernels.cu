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
//    host-location table is needed: host mirror row i caches global slot i.
//  - each batch row owns a fixed region of `region_stride` hot rows;
//    slots [0, hot_size) are LRU-managed, slot `hot_size` holds the row's
//    newest token (written directly by the KV-cache update).
//
// Both kernels only move bytes; they are dtype agnostic.

#include "torch_utils.h"
#include "ops.h"
#include "../cuda_utils.h"

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

// Copy one row of `row_bytes` bytes with a single warp (16B vectorized;
// callers guarantee 16B-aligned rows).
__device__ __forceinline__ void copy_row_warp(int lane_id, const char* src,
                                              char* dst, int64_t row_bytes) {
  const int64_t num_vec = row_bytes / 16;
  const uint4* src4 = reinterpret_cast<const uint4*>(src);
  uint4* dst4 = reinterpret_cast<uint4*>(dst);
  for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
    dst4[j] = src4[j];
  }
}

// In-place inclusive scan over s_data[offset, count) performed by warp 0,
// carrying `accumulator` across calls. Returns the running total.
__device__ __forceinline__ int warp_inclusive_scan(int32_t* s_data,
                                                   int lane_id, int offset,
                                                   int count,
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

// One block per batch row.
//
// Shared memory layout (int32 region followed by int16 region):
//   s_topk[top_k]            top-k global ids; reused as miss scratch
//   s_chunk_off[nbc + 1]     prefix sums for hit (then miss) compaction
//   s_evict_off[nbc + 1]     prefix sums for evictable compaction
//   s_hash_keys[hash_size]   open addressing: global id -> top-k index
//   s_counters[2]            [0] buffer hits, [1] resolved-in-phase-1 count
//   s_lru_out[hot_size]      int16, compacted slots: [hits fwd | evict bwd]
//   s_hash_vals[hash_size]   int16 hash values (top-k index)
template <int BLOCK_SIZE>
__global__ void hisparse_swap_in_kernel(
    const char* __restrict__ source_cache,        // [source_rows, row_bytes]
    const char* __restrict__ host_cache,          // [host_rows, row_bytes]
    const bool* __restrict__ host_cache_valid,    // [host_rows]
    char* __restrict__ hot_cache,                 // [n_rows*stride, row_bytes]
    const int32_t* __restrict__ global_indices,   // [num_rows, top_k]
    const int32_t* __restrict__ newest_global,    // [num_rows] or nullptr
    int32_t* __restrict__ hot_indices,            // [num_rows, top_k]
    int32_t* __restrict__ miss_mask,              // [num_rows, top_k] or nullptr
    int32_t* __restrict__ device_global_indices,  // [max_rows, hot_size]
    int16_t* __restrict__ lru_slots,              // [max_rows, hot_size]
    const int32_t* __restrict__ num_real_reqs,    // [1] or nullptr
    const int64_t source_rows, const int64_t host_rows,
    const int64_t row_bytes, const int32_t top_k, const int32_t hot_size,
    const int32_t hash_size, const int64_t region_stride) {
  constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;
  const int num_token_chunks = (top_k + kWarpSize - 1) / kWarpSize;

  const int row = blockIdx.x;
  // CUDA-graph padding: rows past the real batch carry stale top-k/state and
  // must not be processed. Read from device memory so graph replays see the
  // per-step value.
  if (num_real_reqs != nullptr && row >= num_real_reqs[0]) {
    return;
  }
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  const int64_t hot_base = static_cast<int64_t>(row) * region_stride;
  const int32_t newest_id =
      (newest_global != nullptr) ? newest_global[row] : -1;

  const int32_t* row_topk = global_indices + static_cast<int64_t>(row) * top_k;
  int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
  int32_t* row_miss =
      (miss_mask != nullptr) ? miss_mask + static_cast<int64_t>(row) * top_k
                             : nullptr;
  int32_t* row_dgi =
      device_global_indices + static_cast<int64_t>(row) * hot_size;
  int16_t* row_lru = lru_slots + static_cast<int64_t>(row) * hot_size;

  extern __shared__ char smem_raw[];
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw);
  int32_t* s_chunk_off = s_topk + top_k;
  int32_t* s_evict_off = s_chunk_off + (num_buffer_chunks + 1);
  int32_t* s_hash_keys = s_evict_off + (num_buffer_chunks + 1);
  int32_t* s_counters = s_hash_keys + hash_size;
  int16_t* s_lru_out = reinterpret_cast<int16_t*>(s_counters + 2);
  int16_t* s_hash_vals = s_lru_out + hot_size;

  if (tid < 2) {
    s_counters[tid] = 0;
  }
  for (int i = tid; i < hash_size; i += BLOCK_SIZE) {
    s_hash_keys[i] = kHashEmpty;
  }
  for (int i = tid; i < num_buffer_chunks + 1; i += BLOCK_SIZE) {
    s_chunk_off[i] = 0;
    s_evict_off[i] = 0;
  }
  __syncthreads();

  // Phase 1: resolve invalid / newest entries, hash the rest.
  for (int i = tid; i < top_k; i += BLOCK_SIZE) {
    const int32_t g = row_topk[i];
    if (row_miss != nullptr) row_miss[i] = 0;
    if (g < 0) {
      row_out[i] = -1;
      s_topk[i] = kTokenDone;
      atomicAdd(&s_counters[1], 1);
    } else if (g == newest_id) {
      // Newest token lives in the reserved slot past the LRU range.
      row_out[i] = static_cast<int32_t>(hot_base) + hot_size;
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
    const int32_t cached_g = (slot >= 0) ? row_dgi[slot] : -1;

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
      row_out[found_topk_idx] = static_cast<int32_t>(hot_base) + slot;
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
  for (int i = tid; i < num_token_chunks + 1; i += BLOCK_SIZE) {
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
      // Reuse s_topk as compacted miss scratch: m < i always holds (done
      // entries are skipped), so writes never overrun pending reads.
      s_topk[m] = g;
      row_out[i] = static_cast<int32_t>(hot_base) + evict_slot;
      if (row_miss != nullptr) row_miss[i] = 1;
      row_dgi[evict_slot] = g;
    }
  }
  __syncthreads();

  const int total_hits = s_counters[0];
  const int total_misses = top_k - total_hits - s_counters[1];

  // Phase 4: write back the LRU order: stale evictables at the front,
  // freshly loaded misses next, hits at the MRU back.
  const int total_evictable = hot_size - total_hits;
  for (int i = tid; i < hot_size; i += BLOCK_SIZE) {
    if (i < total_misses) {
      row_lru[total_evictable - total_misses + i] =
          s_lru_out[hot_size - 1 - i];
    } else if (i < total_evictable) {
      row_lru[i - total_misses] = s_lru_out[hot_size - 1 - i];
    } else {
      row_lru[i] = s_lru_out[i - total_evictable];
    }
  }

  // Phase 5: copy missed rows, one warp per miss. Prefer the host mirror;
  // fall back to the full GPU source cache when the row was never mirrored.
  for (int m = warp_id; m < total_misses; m += NUM_WARPS) {
    const int32_t g = s_topk[m];
    const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
    char* dst = hot_cache + (hot_base + evict_slot) * row_bytes;

    const char* src = nullptr;
    if (g < host_rows && host_cache_valid[g]) {
      src = host_cache + static_cast<int64_t>(g) * row_bytes;
    } else if (g < source_rows) {
      src = source_cache + static_cast<int64_t>(g) * row_bytes;
    }
    if (src != nullptr) {
      copy_row_warp(lane_id, src, dst, row_bytes);
    }
  }
}

// Shared-layer plan replay. Given a plan (hot_indices + miss_mask) already
// computed by the group's "full" layer via hisparse_swap_in, gather THIS
// layer's own missed KV rows into the planned hot slots. No LRU resolution:
// index-sharing shared layers see identical global_indices/hot_indices, so the
// slot assignment is identical -- only the per-layer bytes differ. Fixed shape
// (num_rows x top_k), so it is CUDA-graph-capture safe.
template <int BLOCK_SIZE>
__global__ void hisparse_gather_plan_kernel(
    const char* __restrict__ source_cache,        // [source_rows, row_bytes]
    const char* __restrict__ host_cache,          // [host_rows, row_bytes]
    const bool* __restrict__ host_cache_valid,    // [host_rows]
    char* __restrict__ hot_cache,                 // [hot_rows, row_bytes]
    const int32_t* __restrict__ global_indices,   // [num_rows, top_k]
    const int32_t* __restrict__ hot_indices,      // [num_rows, top_k] abs hot rows
    const int32_t* __restrict__ miss_mask,        // [num_rows, top_k]
    const int32_t* __restrict__ num_real_reqs,    // [1] or nullptr
    const int64_t source_rows, const int64_t host_rows,
    const int64_t row_bytes, const int32_t top_k) {
  constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;
  const int row = blockIdx.x;
  if (num_real_reqs != nullptr && row >= num_real_reqs[0]) {
    return;
  }
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = warp_id; col < top_k; col += NUM_WARPS) {
    if (miss_mask[base + col] == 0) {
      continue;
    }
    const int32_t g = global_indices[base + col];
    const int32_t dst = hot_indices[base + col];
    if (g < 0 || dst < 0) {
      continue;
    }
    const char* src = nullptr;
    if (g < host_rows && host_cache_valid[g]) {
      src = host_cache + static_cast<int64_t>(g) * row_bytes;
    } else if (g < source_rows) {
      src = source_cache + static_cast<int64_t>(g) * row_bytes;
    }
    if (src != nullptr) {
      copy_row_warp(lane_id, src,
                    hot_cache + static_cast<int64_t>(dst) * row_bytes, row_bytes);
    }
  }
}

// One warp per item: gather `src_cache[src_indices[i]]` into
// `host_cache[dst_slots[i]]` and mark the row valid. Used to mirror KV rows
// into the pinned host cache fully stream-ordered (no host synchronization).
template <int BLOCK_SIZE>
__global__ void hisparse_backup_kernel(
    const char* __restrict__ src_cache, const int64_t* __restrict__ src_indices,
    char* __restrict__ host_cache, bool* __restrict__ host_cache_valid,
    const int64_t* __restrict__ dst_slots, const int64_t row_bytes,
    const int32_t num_items, const int64_t src_rows, const int64_t host_rows) {
  constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warp_id = blockIdx.x * NUM_WARPS + threadIdx.x / kWarpSize;
  const int total_warps = gridDim.x * NUM_WARPS;

  for (int i = warp_id; i < num_items; i += total_warps) {
    const int64_t s = src_indices[i];
    const int64_t d = dst_slots[i];
    if (s < 0 || s >= src_rows || d < 0 || d >= host_rows) {
      continue;
    }
    copy_row_warp(lane_id, src_cache + s * row_bytes,
                  host_cache + d * row_bytes, row_bytes);
    __syncwarp();
    if (lane_id == 0) {
      host_cache_valid[d] = true;
    }
  }
}

int64_t check_2d_rows(const torch::stable::Tensor& t, const char* name,
                      int64_t row_bytes) {
  STD_TORCH_CHECK(t.dim() == 2, name, " must be 2D");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(t.size(1) * t.element_size() == row_bytes, name,
              " row width mismatch");
  return t.size(0);
}

}  // namespace

void hisparse_swap_in(torch::stable::Tensor const& source_cache,
                      torch::stable::Tensor const& host_cache,
                      torch::stable::Tensor const& host_cache_valid,
                      torch::stable::Tensor& hot_cache,
                      torch::stable::Tensor const& global_indices,
                      std::optional<torch::stable::Tensor> const& newest_global_indices,
                      torch::stable::Tensor& hot_indices,
                      torch::stable::Tensor& device_global_indices,
                      torch::stable::Tensor& lru_slots,
                      std::optional<torch::stable::Tensor> const& num_real_reqs,
                      int64_t region_stride,
                      std::optional<torch::stable::Tensor> const& miss_mask) {
  STD_TORCH_CHECK(source_cache.is_cuda(), "source_cache must be on CUDA");
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
              "host_cache must be CPU memory");
  STD_TORCH_CHECK(host_cache_valid.device().is_cpu(),
              "host_cache_valid must be CPU memory");
  STD_TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  STD_TORCH_CHECK(global_indices.is_cuda() && hot_indices.is_cuda() &&
                  device_global_indices.is_cuda() && lru_slots.is_cuda(),
              "index tensors must be on CUDA");
  STD_TORCH_CHECK(global_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
                  hot_indices.scalar_type() == torch::headeronly::ScalarType::Int,
              "global_indices/hot_indices must be int32");
  STD_TORCH_CHECK(device_global_indices.scalar_type() == torch::headeronly::ScalarType::Int,
              "device_global_indices must be int32");
  STD_TORCH_CHECK(lru_slots.scalar_type() == torch::headeronly::ScalarType::Short, "lru_slots must be int16");
  STD_TORCH_CHECK(host_cache_valid.scalar_type() == torch::headeronly::ScalarType::Bool,
              "host_cache_valid must be bool");
  STD_TORCH_CHECK(global_indices.dim() == 2 && global_indices.is_contiguous(),
              "global_indices must be contiguous 2D");
  STD_TORCH_CHECK(hot_indices.size(0) == global_indices.size(0) &&
                  hot_indices.size(1) == global_indices.size(1) &&
                  hot_indices.is_contiguous(),
              "hot_indices must match global_indices");
  STD_TORCH_CHECK(
      device_global_indices.dim() == 2 && device_global_indices.is_contiguous(),
      "device_global_indices must be contiguous 2D");
  STD_TORCH_CHECK(lru_slots.size(0) == device_global_indices.size(0) &&
                  lru_slots.size(1) == device_global_indices.size(1) &&
                  lru_slots.is_contiguous(),
              "lru_slots must match device_global_indices");

  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  STD_TORCH_CHECK(row_bytes % 16 == 0, "KV row must be 16-byte aligned");
  auto hot_cache_2d = torch::stable::reshape(hot_cache, {-1, hot_cache.size(-1)});
  const int64_t hot_rows = hot_cache_2d.size(0);
  auto source_2d = torch::stable::reshape(source_cache, {-1, source_cache.size(-1)});
  const int64_t source_rows = check_2d_rows(source_2d, "source_cache",
                                            row_bytes);
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  STD_TORCH_CHECK(host_cache_valid.numel() >= host_rows,
              "host_cache_valid has too few rows");

  const auto num_rows = static_cast<int32_t>(global_indices.size(0));
  const auto top_k = static_cast<int32_t>(global_indices.size(1));
  const auto hot_size = static_cast<int32_t>(device_global_indices.size(1));
  STD_TORCH_CHECK(hot_size >= top_k, "hot buffer size must be >= top_k");
  STD_TORCH_CHECK(hot_size < 32767, "hot buffer size must fit int16 slots");
  STD_TORCH_CHECK(region_stride > hot_size,
              "region_stride must reserve a newest slot past hot_size");
  STD_TORCH_CHECK(device_global_indices.size(0) >= num_rows,
              "device_global_indices has too few rows");
  STD_TORCH_CHECK(static_cast<int64_t>(num_rows) * region_stride <= hot_rows,
              "hot_cache has too few rows");
  STD_TORCH_CHECK(hot_rows < INT32_MAX, "hot indices must fit int32");

  const int32_t* newest_ptr = nullptr;
  if (newest_global_indices.has_value()) {
    auto const& newest = newest_global_indices.value();
    STD_TORCH_CHECK(newest.is_cuda() && newest.scalar_type() == torch::headeronly::ScalarType::Int &&
                    newest.numel() >= num_rows && newest.is_contiguous(),
                "newest_global_indices must be contiguous int32 on CUDA");
    newest_ptr = newest.const_data_ptr<int32_t>();
  }

  const int32_t* num_real_ptr = nullptr;
  if (num_real_reqs.has_value()) {
    auto const& num_real = num_real_reqs.value();
    STD_TORCH_CHECK(num_real.is_cuda() && num_real.scalar_type() == torch::headeronly::ScalarType::Int &&
                    num_real.numel() >= 1,
                "num_real_reqs must be int32 on CUDA");
    num_real_ptr = num_real.const_data_ptr<int32_t>();
  }

  // Optional plan output: 1 at columns resolved as a miss (loaded from
  // host/source this call), 0 elsewhere. Lets index-sharing "shared" layers
  // replay the same gather via hisparse_gather_plan without re-resolving LRU.
  int32_t* miss_mask_ptr = nullptr;
  if (miss_mask.has_value()) {
    auto const& mm = miss_mask.value();
    STD_TORCH_CHECK(mm.is_cuda() && mm.scalar_type() == torch::headeronly::ScalarType::Int &&
                    mm.dim() == 2 && mm.is_contiguous() &&
                    mm.size(0) == global_indices.size(0) &&
                    mm.size(1) == global_indices.size(1),
                "miss_mask must be a contiguous int32 CUDA tensor matching global_indices");
    miss_mask_ptr = mm.mutable_data_ptr<int32_t>();
  }

  if (num_rows == 0 || top_k == 0) {
    return;
  }

  constexpr int kBlockSize = 1024;
  const int hash_size = 2 * top_k;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;
  const size_t smem_bytes =
      sizeof(int32_t) * (top_k + 2 * (num_buffer_chunks + 1) + hash_size + 2) +
      sizeof(int16_t) * (hot_size + hash_size);

  const torch::stable::accelerator::DeviceGuard device_guard(hot_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  auto kernel = hisparse_swap_in_kernel<kBlockSize>;
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }
  kernel<<<num_rows, kBlockSize, smem_bytes, stream>>>(
      static_cast<const char*>(source_2d.const_data_ptr()),
      static_cast<const char*>(host_cache.const_data_ptr()),
      host_cache_valid.const_data_ptr<bool>(),
      static_cast<char*>(hot_cache_2d.mutable_data_ptr()),
      global_indices.const_data_ptr<int32_t>(), newest_ptr,
      hot_indices.mutable_data_ptr<int32_t>(), miss_mask_ptr,
      device_global_indices.mutable_data_ptr<int32_t>(),
      lru_slots.mutable_data_ptr<int16_t>(), num_real_ptr, source_rows, host_rows,
      row_bytes, top_k, hot_size, hash_size, region_stride);
}

void hisparse_gather_plan(torch::stable::Tensor const& source_cache,
                          torch::stable::Tensor const& host_cache,
                          torch::stable::Tensor const& host_cache_valid,
                          torch::stable::Tensor& hot_cache,
                          torch::stable::Tensor const& global_indices,
                          torch::stable::Tensor const& hot_indices,
                          torch::stable::Tensor const& miss_mask,
                          std::optional<torch::stable::Tensor> const& num_real_reqs) {
  STD_TORCH_CHECK(source_cache.is_cuda(), "source_cache must be on CUDA");
  STD_TORCH_CHECK(host_cache.device().is_cpu(), "host_cache must be CPU memory");
  STD_TORCH_CHECK(host_cache_valid.device().is_cpu(),
              "host_cache_valid must be CPU memory");
  STD_TORCH_CHECK(hot_cache.is_cuda(), "hot_cache must be on CUDA");
  STD_TORCH_CHECK(global_indices.is_cuda() && hot_indices.is_cuda() &&
                  miss_mask.is_cuda(),
              "plan tensors must be on CUDA");
  STD_TORCH_CHECK(global_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
                  hot_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
                  miss_mask.scalar_type() == torch::headeronly::ScalarType::Int,
              "plan tensors must be int32");
  STD_TORCH_CHECK(global_indices.dim() == 2 && global_indices.is_contiguous(),
              "global_indices must be contiguous 2D");
  STD_TORCH_CHECK(hot_indices.size(0) == global_indices.size(0) &&
                  hot_indices.size(1) == global_indices.size(1) &&
                  miss_mask.size(0) == global_indices.size(0) &&
                  miss_mask.size(1) == global_indices.size(1) &&
                  hot_indices.is_contiguous() && miss_mask.is_contiguous(),
              "hot_indices/miss_mask must match contiguous 2D global_indices");
  STD_TORCH_CHECK(host_cache_valid.scalar_type() == torch::headeronly::ScalarType::Bool,
              "host_cache_valid must be bool");

  const int64_t row_bytes = hot_cache.size(-1) * hot_cache.element_size();
  STD_TORCH_CHECK(row_bytes % 16 == 0, "KV row must be 16-byte aligned");
  auto hot_cache_2d = torch::stable::reshape(hot_cache, {-1, hot_cache.size(-1)});
  auto source_2d = torch::stable::reshape(source_cache, {-1, source_cache.size(-1)});
  const int64_t source_rows = check_2d_rows(source_2d, "source_cache", row_bytes);
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  STD_TORCH_CHECK(host_cache_valid.numel() >= host_rows,
              "host_cache_valid has too few rows");

  const auto num_rows = static_cast<int32_t>(global_indices.size(0));
  const auto top_k = static_cast<int32_t>(global_indices.size(1));

  const int32_t* num_real_ptr = nullptr;
  if (num_real_reqs.has_value()) {
    auto const& num_real = num_real_reqs.value();
    STD_TORCH_CHECK(num_real.is_cuda() && num_real.scalar_type() == torch::headeronly::ScalarType::Int &&
                    num_real.numel() >= 1,
                "num_real_reqs must be int32 on CUDA");
    num_real_ptr = num_real.const_data_ptr<int32_t>();
  }

  if (num_rows == 0 || top_k == 0) {
    return;
  }

  // Match the swap-in kernel's block size: the gather serves 3 of every 4
  // layers' misses (index-sharing replay), so per-row copy parallelism is
  // the throughput limiter on cold rows.
  constexpr int kBlockSize = 1024;
  const torch::stable::accelerator::DeviceGuard device_guard(hot_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_gather_plan_kernel<kBlockSize><<<num_rows, kBlockSize, 0, stream>>>(
      static_cast<const char*>(source_2d.const_data_ptr()),
      static_cast<const char*>(host_cache.const_data_ptr()),
      host_cache_valid.const_data_ptr<bool>(),
      static_cast<char*>(hot_cache_2d.mutable_data_ptr()),
      global_indices.const_data_ptr<int32_t>(),
      hot_indices.const_data_ptr<int32_t>(),
      miss_mask.const_data_ptr<int32_t>(), num_real_ptr, source_rows, host_rows,
      row_bytes, top_k);
}

void hisparse_backup(torch::stable::Tensor const& src_cache,
                     torch::stable::Tensor const& src_indices,
                     torch::stable::Tensor& host_cache,
                     torch::stable::Tensor& host_cache_valid,
                     torch::stable::Tensor const& dst_slots) {
  STD_TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on CUDA");
  STD_TORCH_CHECK(host_cache.device().is_cpu(),
              "host_cache must be CPU memory");
  STD_TORCH_CHECK(host_cache_valid.device().is_cpu(),
              "host_cache_valid must be CPU memory");
  STD_TORCH_CHECK(host_cache_valid.scalar_type() == torch::headeronly::ScalarType::Bool,
              "host_cache_valid must be bool");
  STD_TORCH_CHECK(src_indices.is_cuda() && dst_slots.is_cuda(),
              "src_indices/dst_slots must be on CUDA");
  STD_TORCH_CHECK(
      src_indices.scalar_type() == torch::headeronly::ScalarType::Long && src_indices.is_contiguous(),
      "src_indices must be contiguous int64");
  STD_TORCH_CHECK(dst_slots.scalar_type() == torch::headeronly::ScalarType::Long && dst_slots.is_contiguous(),
              "dst_slots must be contiguous int64");
  STD_TORCH_CHECK(src_indices.numel() == dst_slots.numel(),
              "src_indices and dst_slots must have the same length");

  const int64_t row_bytes = src_cache.size(-1) * src_cache.element_size();
  STD_TORCH_CHECK(row_bytes % 16 == 0, "KV row must be 16-byte aligned");
  auto src_2d = torch::stable::reshape(src_cache, {-1, src_cache.size(-1)});
  const int64_t src_rows = src_2d.size(0);
  const int64_t host_rows = check_2d_rows(host_cache, "host_cache", row_bytes);
  STD_TORCH_CHECK(host_cache_valid.numel() >= host_rows,
              "host_cache_valid has too few rows");

  const auto num_items = static_cast<int32_t>(src_indices.numel());
  if (num_items == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  const int grid = (num_items + kNumWarps - 1) / kNumWarps;

  const torch::stable::accelerator::DeviceGuard device_guard(src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  hisparse_backup_kernel<kBlockSize><<<grid, kBlockSize, 0, stream>>>(
      static_cast<const char*>(src_2d.const_data_ptr()),
      src_indices.const_data_ptr<int64_t>(),
      static_cast<char*>(host_cache.mutable_data_ptr()),
      host_cache_valid.mutable_data_ptr<bool>(), dst_slots.const_data_ptr<int64_t>(),
      row_bytes, num_items, src_rows, host_rows);
}
