#include "common.cuh"

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

// Fill each stream's4096-ID tail before reserving more pages.
__global__ void compact(const __grid_constant__ Args a, const int* budget,
                        int seed, const int* starts, const int* ends,
                        const int* live_nk) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ uint2 output_word[8][256];
  __shared__ int thread_end[256];
  __shared__ int current_base[5], used[5], task_index[5], failed[5];
  __shared__ int aggregate, old_base, old_used, old_take;
  __shared__ int new_base, new_chunks, new_task, valid;
  const int t = threadIdx.x, row = blockIdx.y, logical = *live_nk;
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
  for (int first_tile = 0; first_tile < words; first_tile += 2048) {
    WordMasks packed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      packed[j] = classify(a, row, first_tile + t + j * 256, words, logical,
                           seed, lo, hi);
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
        old_base = current_base[stream];
        old_used = used[stream];
        old_take = old_base < 0 ? 0 : min(total, ChunkIds - old_used);
        const int remaining = total - old_take;
        new_chunks = (remaining + ChunkIds - 1) / ChunkIds;
        new_base = new_task = -1;
        valid = 1;
        if (new_chunks) {
          const int page =
              reserve_pool(a.head, new_chunks * ChunkPages, a.pages);
          new_task =
              page < 0 ? -1
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
        int owner = 0;
#pragma unroll
        for (int step = 128; step; step >>= 1)
          if (rank >= thread_end[owner + step - 1]) owner += step;
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
        a.pool[position] = seed + (first_tile + owner + word * 256) * 32 + bit;
      }
      __syncthreads();
    }
  }
}
}  // namespace DOSA_NAMESPACE::candidate_packets

namespace DOSA_NAMESPACE {
void compact_spans_live_nk(Tensor pool, Tensor head, Tensor solo_tasks,
                           Tensor common_tasks, Tensor solo_state,
                           Tensor common_state, Tensor error, Tensor words,
                           Tensor budget, int64_t seed, int64_t nk_capacity,
                           Tensor starts, Tensor ends, Tensor live_nk) {
  DeviceGuard guard(words.get_device_index());
  candidate_packets::Args a{
      pool.mutable_data_ptr<int>(),
      solo_state.mutable_data_ptr<int>(),
      common_state.mutable_data_ptr<int>(),
      error.mutable_data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(head.mutable_data_ptr<int64_t>()),
      reinterpret_cast<CandidateChunk*>(solo_tasks.mutable_data_ptr<int>()),
      reinterpret_cast<CandidateChunk*>(common_tasks.mutable_data_ptr<int>()),
      reinterpret_cast<const uint4*>(words.const_data_ptr<int>()),
      int(words.size(1)),
      int(words.size(0) * 4),
      int(nk_capacity),
      int(pool.size(0)),
      int(solo_tasks.size(0)),
      int(common_tasks.size(0))};
  launch<candidate_packets::compact, 256>(
      dim3(1, a.nq / 4), a, budget.const_data_ptr<int>(), int(seed),
      starts.const_data_ptr<int>(), ends.const_data_ptr<int>(),
      live_nk.const_data_ptr<int>());
}
}  // namespace DOSA_NAMESPACE

DOSA_REGISTER(
    "compact_spans_live_nk",
    "(Tensor! pool, Tensor! head, Tensor! solo_tasks, Tensor! common_tasks, "
    "Tensor! solo_state, Tensor! common_state, Tensor! error, Tensor words, "
    "Tensor budget, int seed, int nk_capacity, Tensor starts, Tensor ends, "
    "Tensor live_nk) -> ()",
    CUDA, compact_spans_live_nk)
