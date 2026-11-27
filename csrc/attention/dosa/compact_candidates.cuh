#pragma once
#include "common.cuh"
#include "packed20.cuh"

namespace DOSA_NAMESPACE::candidate_packets {
struct Args {
  int *pool, *solo_n, *common_n, *error;
  unsigned long long* head;
  CandidateChunk *solo_tasks, *common_tasks;
  const uint4* words;
  int word_count, nq, nk, pages, solo_task_cap, common_task_cap;
};

__device__ __forceinline__ int reserve(int* counter, int count, int capacity) {
  int first = atomicAdd(counter, 0);
  while (first >= 0 && int64_t(first) + count <= capacity) {
    int previous = atomicCAS(counter, first, first + count);
    if (previous == first) return first;
    first = previous;
  }
  return -1;
}
// Page attempts may exceed capacity; consumers read only published tasks.
__device__ __forceinline__ int reserve_pool(unsigned long long* head, int count,
                                            int capacity) {
  const auto first = atomicAdd(head, static_cast<unsigned long long>(count));
  return first <= static_cast<unsigned long long>(capacity) &&
                 static_cast<unsigned long long>(count) <= capacity - first
             ? int(first)
             : -1;
}
__device__ __forceinline__ void fail(const Args& a, int row, bool common) {
  for (int j = 0; j < (common ? 4 : 1); ++j)
    atomicOr(a.error + row * (common ? 4 : 1) + j, 1);
}

struct WordMasks {
  uint4 own;
  unsigned common;
};
__device__ __forceinline__ WordMasks classify(const Args& a, int row, int word,
                                              int words, int logical, int seed,
                                              int lo, int hi) {
  uint4 p = word < words ? a.words[int64_t(row) * a.word_count + word]
                         : make_uint4(0, 0, 0, 0);
  const int key = seed + word * 32, remaining = logical - key;
  const unsigned tail = remaining >= 32  ? 0xFFFFFFFFu
                        : remaining <= 0 ? 0u
                                         : ((1u << remaining) - 1u);
  p.x &= tail;
  p.y &= tail;
  p.z &= tail;
  p.w &= tail;
  const int64_t left = max(int64_t(lo) - key, int64_t(0)),
                right = int64_t(hi) - key;
  const unsigned lower = left >= 32 ? 0u : (0xFFFFFFFFu << int(left));
  const unsigned upper = right >= 32  ? 0xFFFFFFFFu
                         : right <= 0 ? 0u
                                      : ((1u << int(right)) - 1u);
  return {p, ((p.x & p.y) | (p.x & p.z) | (p.x & p.w) | (p.y & p.z) |
              (p.y & p.w) | (p.z & p.w)) &
                 lower & upper};
}
__device__ __forceinline__ unsigned stream_mask(WordMasks word, int stream) {
  const unsigned own = stream == 0   ? word.own.x
                       : stream == 1 ? word.own.y
                       : stream == 2 ? word.own.z
                                     : word.own.w;
  return stream == 4 ? word.common : own & ~word.common;
}

constexpr int ChunkIds = 4096, ChunkPages = ChunkIds / 128;

__device__ __forceinline__ int reserve_tasks(int* counter, int chunks,
                                             int capacity, int pool_pages) {
  // Pool grants bound tasks; smaller queues retain bounded CAS.
  if (capacity >= pool_pages / ChunkPages) return atomicAdd(counter, chunks);
  return reserve(counter, chunks, capacity);
}

// Tight retry grants complete streams, leaving failed grants available to
// smaller streams. Unlike the normal path, page attempts do not advance head.
__device__ int reserve_complete_pages(unsigned long long* head, int count,
                                      int capacity) {
  auto first = atomicAdd(head, 0ULL);
  while (first <= static_cast<unsigned long long>(capacity) &&
         static_cast<unsigned long long>(count) <= capacity - first) {
    auto previous = atomicCAS(head, first, first + count);
    if (previous == first) return int(first);
    first = previous;
  }
  return -1;
}

// Called only by the Tight specialization; its shared reduction/count storage
// does not enter the ordinary specialization. Existing stream arrays become
// first-slot, written-count, first-task and skip in Tight mode.
__device__ void prepare_complete(const Args& a, int row, int words, int logical,
                                 int seed, int lo, int hi, int* first_slot,
                                 int* written, int* first_task, int* skipped) {
  using Reduce = cub::BlockReduce<int, 256>;
  __shared__ typename Reduce::TempStorage reduce;
  __shared__ int count[5];
  const int t = threadIdx.x;
  int local[5]{};
  for (int word = t; word < words; word += 256) {
    const WordMasks masks =
        classify(a, row, word, words, logical, seed, lo, hi);
#pragma unroll
    for (int stream = 0; stream < 5; ++stream)
      local[stream] += __popc(stream_mask(masks, stream));
  }
#pragma unroll
  for (int stream = 0; stream < 5; ++stream) {
    const int total = Reduce(reduce).Sum(local[stream]);
    if (t == 0) count[stream] = total;
    __syncthreads();
  }
  if (t == 0) {
    int pages[5], total_pages = 0;
#pragma unroll
    for (int stream = 0; stream < 5; ++stream) {
      pages[stream] =
          (count[stream] + packed20::PageIds - 1) / packed20::PageIds;
      total_pages += pages[stream];
      first_slot[stream] = first_task[stream] = -1;
      written[stream] = 0;
    }
    int group_start = reserve_complete_pages(a.head, total_pages, a.pages);
    if (group_start >= 0) {
#pragma unroll
      for (int stream = 0; stream < 5; ++stream) {
        first_slot[stream] = group_start * packed20::PageIds;
        group_start += pages[stream];
      }
    } else {
      // Prioritize common work. A failed common grant invalidates all four rows.
      for (int i = 0; i < 5; ++i) {
        const int stream = (i + 4) % 5;
        const int slot = reserve_complete_pages(a.head, pages[stream], a.pages);
        if (slot >= 0)
          first_slot[stream] = slot * packed20::PageIds;
        else {
          fail(a, stream == 4 ? row : row * 4 + stream, stream == 4);
          if (stream == 4) break;
        }
      }
    }
#pragma unroll
    for (int stream = 0; stream < 5; ++stream) {
      if (first_slot[stream] >= 0 && count[stream] > 0) {
        const int chunks = (count[stream] + ChunkIds - 1) / ChunkIds;
        first_task[stream] =
            reserve(stream == 4 ? a.common_n : a.solo_n, chunks,
                    stream == 4 ? a.common_task_cap : a.solo_task_cap);
        if (first_task[stream] < 0)
          fail(a, stream == 4 ? row : row * 4 + stream, stream == 4);
      }
      skipped[stream] = first_task[stream] < 0;
    }
  }
  __syncthreads();
#pragma unroll
  for (int stream = 0; stream < 5; ++stream) {
    if (skipped[stream]) continue;
    CandidateChunk* tasks = stream == 4 ? a.common_tasks : a.solo_tasks;
    const int chunks = (count[stream] + ChunkIds - 1) / ChunkIds;
    for (int chunk = t; chunk < chunks; chunk += 256)
      tasks[first_task[stream] + chunk] = {
          stream == 4 ? row : row * 4 + stream,
          first_slot[stream] + chunk * ChunkIds,
          min(ChunkIds, count[stream] - chunk * ChunkIds)};
  }
}

// Tight=false preserves the normal4096-ID tail reservation path.
template <bool Tight>
__global__ void compact(const __grid_constant__ Args a, const int* budget,
                        int seed, const int* starts, const int* ends,
                        const int* live_nk, bool defer_capacity_error,
                        cudaGraphConditionalHandle retry_handle) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ uint2 output_word[8][256];
  __shared__ int thread_end[256];
  __shared__ uint8_t warp_start_owner[2049];
  __shared__ int current_base[5], used[5], task_index[5], failed[5];
  __shared__ int aggregate, old_base, old_used, old_take;
  __shared__ int new_base, new_chunks, new_task, valid;
  const int t = threadIdx.x, row = blockIdx.y, logical = *live_nk;
  if (dosa_geometry::native_dense_route(budget[row], a.error + row * 4))
    return;
  if (logical < seed || logical > a.nk ||
      !dosa_geometry::valid_certificate_budget(budget[row]) || a.pages < 0 ||
      int64_t(a.pages) * 128 > INT_MAX) {
    if (t == 0) fail(a, row, true);
    return;
  }
  if (t < 5) {
    current_base[t] = task_index[t] = -1;
    used[t] = failed[t] = 0;
  }
  __syncthreads();
  const int words = (logical - seed + 31) / 32;
  const int lo = max(max(starts[row * 4], starts[row * 4 + 1]),
                     max(starts[row * 4 + 2], starts[row * 4 + 3]));
  const int hi = min(min(ends[row * 4], ends[row * 4 + 1]),
                     min(ends[row * 4 + 2], ends[row * 4 + 3]));
  if constexpr (Tight) {
    prepare_complete(a, row, words, logical, seed, lo, hi, current_base, used,
                     task_index, failed);
    bool active = false;
#pragma unroll
    for (int stream = 0; stream < 5; ++stream) active |= !failed[stream];
    if (!active) return;
  }
  for (int first_tile = 0; first_tile < words;) {
    const int next_tile =
        first_tile + 2048 - (seed / 32 + first_tile) % 2048;
    WordMasks packed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      packed[j] = classify(a, row, first_tile + t + j * 256,
                           min(words, next_tile),
                           logical, seed, lo, hi);
#pragma unroll
    for (int stream = 0; stream < 5; ++stream) {
      if (failed[stream]) continue;
      unsigned mask[8];
      int n = 0, offset, total;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mask[j] = stream_mask(packed[j], stream);
        n += __popc(mask[j]);
      }
      Scan(scan).ExclusiveSum(n, offset, total);
      if (n) {
        // Publish the source owner of each aligned 32-output interval start.
        for (int rank = (offset + 31) & ~31; rank < offset + n; rank += 32)
          warp_start_owner[rank / 32] = uint8_t(t);
        int word_end = offset;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          word_end += __popc(mask[j]);
          output_word[j][t] = make_uint2(mask[j], word_end);
        }
      }
      thread_end[t] = offset + n;
      if (t == 0) {
        aggregate = total;
        warp_start_owner[(total + 31) / 32] = 255;
        if constexpr (Tight) {
          old_base = current_base[stream];
          old_used = used[stream];
          old_take = total;
          new_chunks = 0;
          valid = 1;
          used[stream] += total;
        } else {
          old_base = current_base[stream];
          old_used = used[stream];
          old_take = old_base < 0 ? 0 : min(total, ChunkIds - old_used);
          const int remaining = total - old_take;
          new_chunks = (remaining + ChunkIds - 1) / ChunkIds;
          new_base = new_task = -1;
          valid = 1;
          bool retry_pending = false;
          if (new_chunks) {
            const int page =
                reserve_pool(a.head, new_chunks * ChunkPages, a.pages);
            retry_pending = page < 0 && defer_capacity_error;
            new_task =
                page < 0
                    ? -1
                    : reserve_tasks(
                          stream == 4 ? a.common_n : a.solo_n, new_chunks,
                          stream == 4 ? a.common_task_cap : a.solo_task_cap,
                          a.pages);
            const int capacity =
                stream == 4 ? a.common_task_cap : a.solo_task_cap;
            valid = new_task >= 0 && int64_t(new_task) + new_chunks <= capacity;
            if (valid) new_base = page * 128;
          }
          if (!valid) {
            failed[stream] = 1;
            if (retry_pending) {
              const int previous = atomicExch(a.solo_n + 1, 1);
              if (previous == 0 && retry_handle)
                cudaGraphSetConditional(retry_handle, 1);
            } else
              fail(a, stream == 4 ? row : row * 4 + stream, stream == 4);
          } else {
            CandidateChunk* tasks = stream == 4 ? a.common_tasks : a.solo_tasks;
            if (old_take) tasks[task_index[stream]].count = old_used + old_take;
            if (new_chunks) {
              const int last = new_chunks - 1;
              current_base[stream] = new_base + last * ChunkIds;
              used[stream] = remaining - last * ChunkIds;
              task_index[stream] = new_task + last;
            } else {
              used[stream] += old_take;
            }
          }
        }
      }
      __syncthreads();
      if (!aggregate) continue;
      if (!valid) {
        __syncthreads();
        continue;
      }
      CandidateChunk* tasks = stream == 4 ? a.common_tasks : a.solo_tasks;
      for (int chunk = t; chunk < new_chunks; chunk += 256)
        tasks[new_task + chunk] = {
            stream == 4 ? row : row * 4 + stream, new_base + chunk * ChunkIds,
            min(ChunkIds, aggregate - old_take - chunk * ChunkIds)};
      for (int rank = t; rank < aggregate; rank += 256) {
        int owner = warp_start_owner[rank / 32],
            hi = warp_start_owner[rank / 32 + 1];
        while (owner < hi) {
          int mid = (owner + hi) / 2;
          if (rank >= thread_end[mid])
            owner = mid + 1;
          else
            hi = mid;
        }
        int word = 0;
#pragma unroll
        for (int step = 4; step; step >>= 1)
          if (rank >= int(output_word[word + step - 1][owner].y)) word += step;
        const uint2 source = output_word[word][owner];
        const int bit_rank = rank - (int(source.y) - __popc(source.x));
        const int bit = __fns(source.x, 0, bit_rank + 1);
        const int64_t position = rank < old_take
                                     ? int64_t(old_base) + old_used + rank
                                     : int64_t(new_base) + rank - old_take;
        const int id = seed + (first_tile + owner + word * 256) * 32 + bit;
        packed20::store_low(a.pool, position, id);
      }
      const uint32_t nibble = uint32_t(seed + first_tile * 32) >> 16;
      if (old_take)
        packed20::fill_high(a.pool, a.pages, int64_t(old_base) + old_used,
                            old_take, nibble, t);
      const int remaining = aggregate - old_take;
      if (remaining)
        packed20::fill_high(a.pool, a.pages, new_base, remaining, nibble, t);
      __syncthreads();
    }
    first_tile = next_tile;
  }
}
}  // namespace DOSA_NAMESPACE::candidate_packets
