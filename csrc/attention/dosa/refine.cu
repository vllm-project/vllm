#include "common.cuh"
#include "score.cuh"
#include "record.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
// Logical extents bound every candidate before its scale or TMA access.
namespace live_record5_completion {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr int GroupIds = 4096, PacketIds = 128;
template <int BQ>
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[2][128 * 128], q[BQ * kHeads * 128];
  alignas(128) float weights[BQ * kHeads], scale[2][128];
  alignas(128) int ids[2][128], coordinates[2][128];
  Barrier gathered[2], done;
  uint32_t tmem;
  int job;
  int page[2], bad[2], skip;
  uint32_t emit_counts[BQ][4], emit_base[BQ];
  uint64_t emit_starts[BQ][4];
};
struct Args {
  const uint8_t* q;
  const float *w, *scale, *tau;
  const int *ids, *counts, *packs;
  const int2* tasks;
  void* scores;
  int *outids, *outcounts;
  int nk, stride, origin;
  int64_t ids_stride;
  int* cursor;
  const int *page_table, *starts, *ends;
  int* error;
  int qrows, stream_rows, table_width, pool_pages, output_cap;
  int64_t table_stride, pool_stride, score_stride, output_id_stride;
  const int* live_nk;
  int candidate_seed;
};
__device__ __forceinline__ int swizzle(int row, int vec) {
  return row * 8 + (vec ^ (row % 8));
}
template <int BQ>
__device__ __forceinline__ void flag_group(const Args& a, int row, int bit) {
  for (int qb = 0; qb < BQ; ++qb)
    if (row * BQ + qb < a.qrows) atomicOr(a.error + row * BQ + qb, bit);
}
struct Visibility {
  int begin, end;
};
template <int BQ>
__device__ __forceinline__ Visibility group_visibility(const Args& a, int row) {
  Visibility v{INT_MIN, INT_MAX};
#pragma unroll
  for (int qb = 0; qb < BQ; ++qb) {
    const int qr = row * BQ + qb;
    if (qr < a.qrows) {
      v.begin = max(v.begin, a.starts[qr]);
      v.end = min(v.end, a.ends[qr]);
    }
  }
  return v;
}
template <int BQ, bool CachedVisibility = false>
__device__ __forceinline__ void prepare_metadata(Smem<BQ>& s, const Args& a,
                                                 int row, int begin, int stop,
                                                 int slot, int t,
                                                 Visibility visibility = {}) {
  if (t == 0) {
    int packet = begin / PacketIds;
    int page = (begin >= 0 && packet < a.table_width)
                   ? a.page_table[int64_t(row) * a.table_stride + packet]
                   : -1;
    s.page[slot] = page;
    s.bad[slot] = (page < 0 || page >= a.pool_pages) ? 1 : 0;
  }
  __syncthreads();
  bool active = begin + t < stop;
  int page = s.page[slot];
  bool address = active && page >= 0 && page < a.pool_pages;
  int id = address ? a.ids[int64_t(page) * a.pool_stride + t] : 0;
  bool valid = address && id >= 0 && id < a.nk;
  if (valid) {
    int64_t visible_id = int64_t(id) + a.origin;
    if constexpr (CachedVisibility) {
      valid = visible_id >= visibility.begin && visible_id < visibility.end;
    } else {
      for (int qb = 0; qb < BQ; ++qb) {
        int qr = row * BQ + qb;
        if (qr < a.qrows &&
            (visible_id < a.starts[qr] || visible_id >= a.ends[qr]))
          valid = false;
      }
    }
  }
  if (active && !valid) atomicOr(&s.bad[slot], 1);
  // Invalid input never reaches a scale access or TMA coordinate. The original
  // MMA is still drained uniformly; sticky error suppresses partial output.
  s.ids[slot][t] = valid ? id : 0;
  s.coordinates[slot][t] = valid ? id : 0;
  s.scale[slot][t] = valid ? a.scale[id] : 0;
  __syncthreads();
  if (t == 0 && s.bad[slot]) flag_group<BQ>(a, row, 1);
  __syncthreads();
}
// Future metadata stays private until the current packet has consumed its
// shared ID/scale. Warp leaders read the immutable page entry independently;
// this removes the CTA rendezvous from the dependency chain before the loads.
struct FutureMetadata {
  int page, id;
  float scale;
  bool bad;
};
// Hold the immutable page/ID load while one independent H32 query is reduced.
// Scale and visibility are still checked before the original publication point.
struct FutureId {
  int page, id;
  uint32_t flags;
};
__device__ __forceinline__ FutureId load_future_id(const Args& a, int row,
                                                   int begin, int stop, int t) {
  int page = -1;
  if ((t & 31) == 0) {
    const int packet = begin / PacketIds;
    page = (begin >= 0 && packet < a.table_width)
               ? a.page_table[int64_t(row) * a.table_stride + packet]
               : -1;
  }
  page = __shfl_sync(0xffffffffu, page, 0);
  const bool active = begin + t < stop;
  const bool address = active && page >= 0 && page < a.pool_pages;
  const int id = address ? a.ids[int64_t(page) * a.pool_stride + t] : 0;
  return {page, id, uint32_t(active) | (uint32_t(address) << 1)};
}
__device__ __forceinline__ FutureMetadata
finish_future_scale(const Args& a, const FutureId& f, Visibility visibility) {
  // Keep every ID-dependent operation in this late stage. In the equivalent
  // C++ expression ptxas hoisted id>=0 immediately after the early ID load,
  // defeating its latency overlap even though the scale load stayed late.
  int id, bad;
  float scale;
  asm volatile(
      "{ .reg .u32 flags; .reg .s64 visible; .reg .u64 addr;\n"
      "  .reg .pred valid,active,badp,p;\n"
      "  mov.u32 %0,0; mov.f32 %1,0f00000000;\n"
      "  and.b32 flags,%5,2; setp.ne.u32 valid,flags,0;\n"
      "  setp.ge.s32 p,%4,0; and.pred valid,valid,p;\n"
      "  setp.lt.s32 p,%4,%6; and.pred valid,valid,p;\n"
      "  cvt.s64.s32 visible,%4; add.s64 visible,visible,%8;\n"
      "  setp.ge.s64 p,visible,%9; and.pred valid,valid,p;\n"
      "  setp.lt.s64 p,visible,%10; and.pred valid,valid,p;\n"
      "  @valid mad.wide.s32 addr,%4,4,%11;\n"
      "  @valid ld.global.f32 %1,[addr]; @valid mov.b32 %0,%4;\n"
      "  setp.lt.s32 badp,%3,0; setp.ge.s32 p,%3,%7; or.pred badp,badp,p;\n"
      "  and.b32 flags,%5,1; setp.ne.u32 active,flags,0;\n"
      "  not.pred p,valid; and.pred p,p,active; or.pred badp,badp,p;\n"
      "  selp.u32 %2,1,0,badp; }\n"
      : "=&r"(id), "=&f"(scale), "=&r"(bad)
      : "r"(f.page), "r"(f.id), "r"(f.flags), "r"(a.nk), "r"(a.pool_pages),
        "l"(int64_t(a.origin)), "l"(int64_t(visibility.begin)),
        "l"(int64_t(visibility.end)), "l"(a.scale)
      : "memory");
  return {f.page, id, scale, bad != 0};
}
template <int BQ, bool CachedVisibility = false>
__device__ __forceinline__ FutureMetadata
load_future(const Args& a, int row, int begin, int stop, int t,
            Visibility visibility = {}) {
  int page = -1;
  if ((t & 31) == 0) {
    const int packet = begin / PacketIds;
    page = (begin >= 0 && packet < a.table_width)
               ? a.page_table[int64_t(row) * a.table_stride + packet]
               : -1;
  }
  page = __shfl_sync(0xffffffffu, page, 0);
  const bool active = begin + t < stop;
  const bool address = active && page >= 0 && page < a.pool_pages;
  const int id = address ? a.ids[int64_t(page) * a.pool_stride + t] : 0;
  bool valid = address && id >= 0 && id < a.nk;
  if (valid) {
    const int64_t visible_id = int64_t(id) + a.origin;
    if constexpr (CachedVisibility) {
      valid = visible_id >= visibility.begin && visible_id < visibility.end;
    } else {
      for (int qb = 0; qb < BQ; ++qb) {
        const int qr = row * BQ + qb;
        if (qr < a.qrows &&
            (visible_id < a.starts[qr] || visible_id >= a.ends[qr]))
          valid = false;
      }
    }
  }
  return {page, valid ? id : 0, valid ? a.scale[id] : 0.0f,
          page < 0 || page >= a.pool_pages || (active && !valid)};
}
template <int BQ>
__device__ __forceinline__ void publish_future(Smem<BQ>& s, const Args& a,
                                               int row, int slot, int t,
                                               const FutureMetadata& future) {
  s.ids[slot][t] = future.id;
  s.coordinates[slot][t] = future.id;
  s.scale[slot][t] = future.scale;
  const int bad = __syncthreads_or(future.bad);
  if (t == 0) {
    s.page[slot] = future.page;
    s.bad[slot] = bad;
    if (bad) flag_group<BQ>(a, row, 1);
  }
  // The original end-of-packet CTA barrier follows, publishing the group error
  // and all metadata before the next iteration can issue the future gather.
}
template <int BQ>
__device__ __forceinline__ void gather(Smem<BQ>& s,
                                       const cute::TmaDescriptor* map, int slot,
                                       int t) {
  if ((t & 31) != 0) return;
  const int warp = t / 32;
#pragma unroll
  for (int packet_part = 0; packet_part < 8; ++packet_part) {
    const int r = warp * 4 + packet_part * 16;
    int4 coord = *reinterpret_cast<const int4*>(s.coordinates[slot] + r);
    deep_gemm::ptx::tma_gather4(
        map, s.gathered[slot], s.keys[slot] + r * 128, 0, coord,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
  }
}
// Dense, GPU-only filtering of the unchanged original 128-boundary tasks.
template <int BQ>
__global__ void filter_groups(const int2* input, const int* pack_count,
                              int2* groups, int* state, int target_groups,
                              int task_cap, const int* counts, int table_width,
                              int stream_rows, int qrows, int* error,
                              const int* live_nk, int nk_capacity, int seed) {
  const int logical = *live_nk;
  if (logical < seed || logical > nk_capacity) {
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < qrows;
         qr += gridDim.x * blockDim.x)
      atomicOr(error + qr, 2);
    return;
  }
  const int max_records = logical - seed;
  table_width = (max_records + PacketIds - 1) / PacketIds;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32, raw = *pack_count;
  int count = max(0, min(raw, task_cap));
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < stream_rows;
       r += gridDim.x * blockDim.x) {
    int n = counts[r];
    if (n < 0 || int64_t(n) > int64_t(max_records))
      for (int qb = 0; qb < BQ; ++qb)
        if (r * BQ + qb < qrows) atomicOr(error + r * BQ + qb, 2);
  }
  if (raw < 0 || raw > task_cap) {
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < qrows;
         qr += gridDim.x * blockDim.x)
      atomicOr(error + qr, 2);
  }
  int ppg = 1;
#pragma unroll
  for (int size = 2; size <= 32; size *= 2)
    if (count >= target_groups * size) ppg = size;
  int group_ids = PacketIds * ppg;
  if (blockIdx.x == 0 && threadIdx.x == 0) state[2] = ppg;
  for (int base = blockIdx.x * blockDim.x + warp * 32; base < count;
       base += gridDim.x * blockDim.x) {
    int i = base + lane;
    bool active = i < count;
    int2 task = active ? input[i] : make_int2(0, 1);
    bool row_ok = task.x >= 0 && task.x < stream_rows;
    int n = active && row_ok ? counts[task.x] : 0;
    bool valid = active && row_ok && task.y >= 0 && (task.y % PacketIds) == 0 &&
                 task.y < n && n >= 0 && int64_t(n) <= int64_t(max_records);
    if (active && !valid) {
      if (row_ok) {
        for (int qb = 0; qb < BQ; ++qb)
          if (task.x * BQ + qb < qrows) atomicOr(error + task.x * BQ + qb, 2);
      } else {
        for (int qr = 0; qr < qrows; ++qr) atomicOr(error + qr, 2);
      }
    }
    bool keep = valid && (task.y & (group_ids - 1)) == 0;
    uint32_t mask = __ballot_sync(0xffffffffu, keep), nkeep = __popc(mask),
             start = 0;
    if (lane == 0 && nkeep) start = atomicAdd(state, nkeep);
    start = __shfl_sync(0xffffffffu, start, 0);
    if (keep)
      groups[start + __popc(lane ? mask & ((1u << lane) - 1) : 0)] = task;
  }
}

}  // namespace live_record5_completion

// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace live_record5_completion {

// These are candidate-slot coordinates. They are deliberately distinct from
// the prefix-permuted K IDs stored in the pool and from Full's affine origin.
struct SparseWorkItem {
  int stream_row;
  int first_slot;
  int end_slot;
  uint32_t failed_queries;
};

// Solo and common streams share128-candidate packets, two K slots and one
// TMEM accumulator. This object owns only input loading and pipeline credits.
template <int BQ>
class PagedScoreAccess {
 public:
  static_assert(BQ == 1 || BQ == 4);
  static constexpr int kMmaQueries = BQ * kHeads > 128 ? 128 / kHeads : BQ;
  static constexpr int kConsumerThreads = 128;

  __device__ __forceinline__ PagedScoreAccess(
      Smem<BQ>& shared, const Args& args, const cute::TmaDescriptor* key_map,
      SparseWorkItem work, uint32_t tmem, uint32_t& raw_phases,
      uint32_t& mma_sequence, int thread)
      : s_(shared),
        a_(args),
        map_(key_map),
        work_(work),
        tm_(tmem),
        raw_phases_(raw_phases),
        mma_sequence_(mma_sequence),
        thread_(thread),
        packets_((work.end_slot - work.first_slot + PacketIds - 1) /
                 PacketIds) {}

  __device__ __forceinline__ void initialize_work() {
    const int row = work_.stream_row;
    if constexpr (BQ == 4) {
      visibility_ = group_visibility<BQ>(a_, row);
#pragma unroll
      for (int query = 0; query < BQ; ++query)
        if (row * BQ + query < a_.qrows)
          group_tau_[query] = a_.tau[row * BQ + query];
    } else if constexpr (BQ == 1) {
      // Same immutable-per-work tau contract as BQ4; errors remain fresh.
      if (row < a_.qrows) group_tau_[0] = a_.tau[row];
    }
    for (int i = thread_; i < BQ * kHeads * 8; i += kConsumerThreads)
      reinterpret_cast<uint4*>(s_.q)[swizzle(i / 8, i % 8)] =
          row * BQ + i / (kHeads * 8) < a_.qrows
              ? reinterpret_cast<const uint4*>(a_.q + uint64_t(row) * BQ *
                                                          kHeads * 128)[i]
              : make_uint4(0, 0, 0, 0);
    for (int head = thread_; head < BQ * kHeads; head += kConsumerThreads)
      s_.weights[head] = row * BQ + head / kHeads < a_.qrows
                             ? a_.w[row * BQ * kHeads + head]
                             : 0.0f;
    prepare_metadata<BQ, BQ == 4>(s_, a_, row, work_.first_slot, work_.end_slot,
                                  0, thread_, visibility_);
    if (packets_ > 1)
      prepare_metadata<BQ, BQ == 4>(s_, a_, row, work_.first_slot + PacketIds,
                                    work_.end_slot, 1, thread_, visibility_);
    if (thread_ == 0) s_.gathered[0].arrive_and_expect_tx(128 * 128);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    gather(s_, map_, 0, thread_);
    if constexpr (BQ == 4)
      if (packets_ > 2)
        next_ = load_future<BQ, true>(a_, row, work_.first_slot + 2 * PacketIds,
                                      work_.end_slot, thread_, visibility_);
  }

  __device__ __forceinline__ int tile_count() const { return packets_; }

  __device__ __forceinline__ void before_tile(int tile) {
    slot_ = tile & 1;
    active_keys_ =
        min(PacketIds, work_.end_slot - work_.first_slot - tile * PacketIds);
    if (thread_ == 0) {
      s_.gathered[slot_].wait((raw_phases_ >> slot_) & 1u);
      raw_phases_ ^= 1u << slot_;
      deep_gemm::ptx::tcgen05_after_thread_sync();
      issue_h32_mma<128, kMmaQueries * kHeads>(
          reinterpret_cast<cutlass::float_e4m3_t*>(s_.keys[slot_]),
          reinterpret_cast<cutlass::float_e4m3_t*>(s_.q), 0, tm_,
          cute::cast_smem_ptr_to_uint(&s_.done));
      if (tile + 1 < packets_)
        s_.gathered[slot_ ^ 1].arrive_and_expect_tx(128 * 128);
    }
    // All other gather issuers observe the complete alternate-slot credit.
    __syncthreads();
    if (tile + 1 < packets_) gather(s_, map_, slot_ ^ 1, thread_);
    future_ = {};
    if constexpr (BQ == 4) {
      future_ = next_;
      pending_scale_ = tile + 3 < packets_;
      if (pending_scale_)
        id_stage_ = load_future_id(a_, work_.stream_row,
                                   work_.first_slot + (tile + 3) * PacketIds,
                                   work_.end_slot, thread_);
    } else if (tile + 2 < packets_) {
      future_ = load_future<BQ>(a_, work_.stream_row,
                                work_.first_slot + (tile + 2) * PacketIds,
                                work_.end_slot, thread_);
    }
    s_.done.wait(mma_sequence_ & 1u);
    deep_gemm::ptx::tcgen05_after_thread_sync();
  }

  __device__ __forceinline__ void finish_future_scale(int query) {
    if constexpr (BQ == 4) {
      if (query == 0 && pending_scale_)
        next_ = live_record5_completion::finish_future_scale(a_, id_stage_,
                                                             visibility_);
    }
  }

  // Bound TMEM to128columns while retaining the same candidate tile and
  // exact per-head arithmetic. Drain each query half before accumulator reuse.
  __device__ __forceinline__ void advance_query_stage(int query) {
    if constexpr (kMmaQueries < BQ) {
      if (query + 1 < BQ && (query + 1) % kMmaQueries == 0) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        __syncthreads();
        ++mma_sequence_;
        if (thread_ == 0) {
          deep_gemm::ptx::tcgen05_after_thread_sync();
          issue_h32_mma<128, kMmaQueries * kHeads>(
              reinterpret_cast<cutlass::float_e4m3_t*>(s_.keys[slot_]),
              reinterpret_cast<cutlass::float_e4m3_t*>(s_.q + (query + 1) *
                                                                  kHeads * 128),
              0, tm_, cute::cast_smem_ptr_to_uint(&s_.done));
        }
        s_.done.wait(mma_sequence_ & 1u);
        deep_gemm::ptx::tcgen05_after_thread_sync();
      }
    }
  }

  __device__ __forceinline__ void after_emit(int tile) {
    if (tile + 2 < packets_)
      publish_future(s_, a_, work_.stream_row, slot_, thread_, future_);
    deep_gemm::ptx::tcgen05_before_thread_sync();
    __syncthreads();
    ++mma_sequence_;
  }

  // Read one query at a time. The common reducer preserves its four FP32
  // chains, RN adds, half factor and current K scale across every exact path.
  __device__ __forceinline__ void run() {
    for (int tile = 0; tile < packets_; ++tile) {
      before_tile(tile);
      float scores[BQ]{};
      uint32_t keep = 0;
#pragma unroll
      for (int query = 0; query < BQ; ++query) {
        const int row = work_.stream_row * BQ + query;
        if (row >= a_.qrows || (work_.failed_queries & (1u << query))) {
          finish_future_scale(query);
          advance_query_stage(query);
          continue;
        }
        float accum[kHeads];
#pragma unroll
        for (int part = 0; part < kHeads / 16; ++part)
          tmem_load(tm_ + (query % kMmaQueries) * kHeads + part * 16,
                    accum + part * 16);
        cutlass::arch::fence_view_async_tmem_load();
        const float value = native_bf16_reduce(
            accum, s_.weights + query * kHeads, s_.scale[slot_][thread_]);
        scores[query] = value;
        const bool active = thread_ < active_keys_;
        if (active && !isfinite(value)) atomicOr(a_.error + row, 8);
        if (active && a_.error[row] == 0 &&
            (!isfinite(group_tau_[query]) || value >= group_tau_[query]))
          keep |= 1u << query;
        finish_future_scale(query);
        advance_query_stage(query);
      }
      emit(scores, keep);
      after_emit(tile);
    }
  }

  // Solo streams reserve per warp. Common Q4 streams reserve one contiguous
  // segment per CTA, then give each warp its original exclusive subsegment.
  __device__ __forceinline__ void emit(const float* scores, uint32_t keep) {
    constexpr uint32_t Full = 0xffffffffu;
    const int lane = thread_ & 31, warp = thread_ / 32,
              row_base = work_.stream_row * BQ;
    uint32_t offsets[BQ];
    int base_solo = 0;
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      const uint32_t bits = __ballot_sync(Full, (keep >> query) & 1u),
                     total = __popc(bits);
      offsets[query] = __popc(lane ? bits & ((1u << lane) - 1u) : 0u);
      if constexpr (BQ == 4) {
        if (lane == 0) s_.emit_counts[query][warp] = total;
      } else if (lane == 0 && row_base < a_.qrows && total) {
        base_solo = atomicAdd(a_.outcounts + row_base, int(total));
      }
    }
    if constexpr (BQ == 4) {
      __syncthreads();
      if (thread_ < 4) {
        const uint32_t c0 = s_.emit_counts[thread_][0],
                       c1 = s_.emit_counts[thread_][1];
        const uint32_t c2 = s_.emit_counts[thread_][2],
                       c3 = s_.emit_counts[thread_][3];
        const uint32_t total = c0 + c1 + c2 + c3;
        const uint32_t base =
            total
                ? uint32_t(atomicAdd(a_.outcounts + row_base + thread_, total))
                : 0u;
        s_.emit_starts[thread_][0] = uint64_t(base);
        s_.emit_starts[thread_][1] = uint64_t(base) + c0;
        s_.emit_starts[thread_][2] = uint64_t(base) + c0 + c1;
        s_.emit_starts[thread_][3] = uint64_t(base) + c0 + c1 + c2;
      }
      __syncthreads();
    }
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      int64_t base;
      if constexpr (BQ == 4)
        base = int64_t(s_.emit_starts[query][warp]);
      else
        base = int64_t(uint32_t(__shfl_sync(Full, base_solo, 0)));
      if (!((keep >> query) & 1u)) continue;
      const float value = scores[query], tau = group_tau_[query];
      const int64_t key = uint32_t(int64_t(s_.ids[slot_][thread_]) + a_.origin);
      const uint32_t code = h32_record5::encode(value);
      const bool valid = isfinite(value) && isfinite(tau) && value >= tau &&
                         __float_as_uint(value) != 0x80000000u && key >= 0 &&
                         key < 0x100000 && code >= h32_record5::Min &&
                         code <= h32_record5::Max;
      const uint32_t stored = valid ? code : h32_record5::Invalid;
      const int64_t column = base + offsets[query];
      const int row = row_base + query;
      if (column >= 0 && column < a_.output_cap) {
        static_cast<uint8_t*>(
            a_.scores)[int64_t(row) * a_.score_stride + column] =
            uint8_t(stored);
        a_.outids[int64_t(row) * a_.output_id_stride + column] =
            int32_t(h32_record5::pack_word(stored, uint32_t(key)));
        if (!valid) atomicOr(a_.error + row, 32);
      } else
        atomicOr(a_.error + row, 16);
    }
  }

 private:
  Smem<BQ>& s_;
  const Args& a_;
  const cute::TmaDescriptor* map_;
  SparseWorkItem work_;
  uint32_t tm_;
  uint32_t& raw_phases_;
  uint32_t& mma_sequence_;
  int thread_, packets_;
  int slot_ = 0, active_keys_ = 0;
  Visibility visibility_{};
  float group_tau_[BQ]{};
  FutureMetadata next_{}, future_{};
  FutureId id_stage_{};
  bool pending_scale_ = false;
};

}  // namespace live_record5_completion

namespace live_record5_completion {
template <int BQ>
CUTLASS_GLOBAL void __launch_bounds__(128, 4)
    exact(Args a, const __grid_constant__ cute::TmaDescriptor map) {
  constexpr int TmemColumns = BQ * kHeads > 128 ? 128 : BQ * kHeads;
  // The descriptor keeps physical capacity; every candidate is checked
  // against this same epoch's logical extent before scale/TMA access.
  const int logical = *a.live_nk;
  if (logical < a.candidate_seed || logical > a.nk) {
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < a.qrows;
         qr += gridDim.x * blockDim.x)
      atomicOr(a.error + qr, 2);
    return;
  }
  a.nk = logical;
  a.table_width = (logical - a.candidate_seed + PacketIds - 1) / PacketIds;
  int task_count = *a.packs, group_ids = PacketIds * a.packs[2];
  if (blockIdx.x >= task_count) return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto& s = *reinterpret_cast<Smem<BQ>*>(buf);
  int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
    s.gathered[0].init(1);
    s.gathered[1].init(1);
    s.done.init(1);
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(&map);
  }
  if (warp == 0) {
    cute::TMEM::Allocator1Sm().allocate(TmemColumns, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();
  int tm = deep_gemm::ptx::ld_shared(&s.tmem);
  uint32_t phase_bits = 0, done_sequence = 0;
  for (;;) {
    if (t == 0) s.job = atomicAdd(a.cursor, 1);
    __syncthreads();
    int job = s.job;
    if (job >= task_count) break;
    int2 task = a.tasks[job];
    int row = task.x, begin = task.y,
        stop = min(begin + group_ids, a.counts[row]);
    if (begin >= stop) {
      __syncthreads();
      continue;
    }
    int failed_columns = 0;
    if constexpr (BQ == 1) {
      if (t == 0) {
        s.skip = 0;
        for (int qb = 0; qb < BQ; ++qb)
          if (row * BQ + qb < a.qrows && a.error[row * BQ + qb] != 0)
            s.skip = 1;
      }
      __syncthreads();
      if (s.skip) {
        if (t == 0) flag_group<BQ>(a, row, 4);
        __syncthreads();
        continue;
      }
    } else {
      if (t == 0) {
        s.skip = 0;
        for (int qb = 0; qb < BQ; ++qb)
          if (row * BQ + qb < a.qrows && a.error[row * BQ + qb] != 0)
            s.skip |= 1 << qb;
      }
      __syncthreads();
      failed_columns = s.skip;
      const int real_columns = min(BQ, a.qrows - row * BQ);
      if (failed_columns == ((1 << real_columns) - 1)) {
        if (t == 0) flag_group<BQ>(a, row, 4);
        __syncthreads();
        continue;
      }
    }

    PagedScoreAccess<BQ> access(s, a, &map,
                                {row, begin, stop, uint32_t(failed_columns)},
                                uint32_t(tm), phase_bits, done_sequence, t);
    access.initialize_work();
    access.run();
    // All issued gathers have been consumed. Per-slot phases remain global
    // across groups, including groups with an odd number of packets.
  }
  if (warp == 0) cute::TMEM::Allocator1Sm().free(tm, TmemColumns);
}
}  // namespace live_record5_completion

namespace live_record5_host {
template <int BQ>
void exact_paged_completion(Tensor q, Tensor k, Tensor scale, Tensor weights,
                            Tensor tau, Tensor ids, Tensor counts, Tensor tasks,
                            Tensor packs, Tensor scores, Tensor outids,
                            Tensor outcounts, int64_t origin, int64_t sms,
                            Tensor page_table, int64_t pool_pages,
                            Tensor starts, Tensor ends, Tensor error,
                            Tensor live_nk, int64_t seed, Tensor group_tasks,
                            Tensor state) {
  using namespace live_record5_completion;
  // Owned contiguous FP8 Q[Q,H,128]/K[Kcap,128], F32 weights/scales/tau,
  // I32 page128 pool and stream tasks, U8/I32[Q,114688] score cells/IDs.
  // Table views keep their physical row stride. origin=0, seed=12288 and
  // Kcap<=2^20; liveNK is immutable until this current-stream launch completes.
  DeviceGuard guard(q.get_device_index());
  const int64_t qr = q.size(0), streams = (qr + BQ - 1) / BQ;
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.data_ptr(), 0, 3 * sizeof(int),
                                  get_current_cuda_stream()));
  filter_groups<BQ><<<4 * sms, 128, 0, get_current_cuda_stream()>>>(
      reinterpret_cast<const int2*>(tasks.mutable_data_ptr<int>()),
      packs.mutable_data_ptr<int>(),
      reinterpret_cast<int2*>(group_tasks.mutable_data_ptr<int>()),
      state.mutable_data_ptr<int>(), 8 * sms, tasks.size(0),
      counts.mutable_data_ptr<int>(), page_table.size(1), streams, qr,
      error.mutable_data_ptr<int>(), live_nk.mutable_data_ptr<int>(),
      int(k.size(0)), int(seed));
  DOSA_CUDA_CHECK(cudaGetLastError());
  Args a{static_cast<const uint8_t*>(q.data_ptr()),
         weights.mutable_data_ptr<float>(),
         scale.mutable_data_ptr<float>(),
         tau.mutable_data_ptr<float>(),
         ids.mutable_data_ptr<int>(),
         counts.mutable_data_ptr<int>(),
         state.mutable_data_ptr<int>(),
         reinterpret_cast<const int2*>(group_tasks.mutable_data_ptr<int>()),
         scores.data_ptr(),
         outids.mutable_data_ptr<int>(),
         outcounts.mutable_data_ptr<int>(),
         int(k.size(0)),
         int(scores.size(1)),
         int(origin),
         ids.stride(0),
         state.mutable_data_ptr<int>() + 1,
         page_table.mutable_data_ptr<int>(),
         starts.mutable_data_ptr<int>(),
         ends.mutable_data_ptr<int>(),
         error.mutable_data_ptr<int>(),
         int(qr),
         int(streams),
         int(page_table.size(1)),
         int(pool_pages),
         int(std::min(scores.size(1), outids.size(1))),
         page_table.stride(0),
         ids.stride(0),
         scores.stride(0),
         outids.stride(0),
         live_nk.mutable_data_ptr<int>(),
         int(seed)};
  auto map = make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                      {128, (cuuint64_t)k.size(0)}, {128}, {128, 1},
                      CU_TENSOR_MAP_SWIZZLE_128B);
  auto fn = exact<BQ>;
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<BQ>)));
  fn<<<4 * sms, 128, sizeof(Smem<BQ>), get_current_cuda_stream()>>>(a, map);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace live_record5_host

namespace live_record5_seed {
constexpr int Top = 2048;
__global__ void validate5(const int* words, int64_t stride, int* counts,
                          const float* seed_tau, const float* safe_tau,
                          const int* starts, const int* ends, int* error,
                          const int* live_nk, int nk_capacity, int seed) {
  const int nk = *live_nk;
  if (nk < seed || nk > nk_capacity) {
    if (threadIdx.x == 0) {
      atomicOr(error + blockIdx.x, 1);
      counts[blockIdx.x] = 0;
    }
    return;
  }
  const int row = blockIdx.x, t = threadIdx.x;
  // The original initializer publishes the exact minimum selected seed score.
  // This witness proves the former per-record score >= safe_tau check without
  // retaining or rereading an intermediate selected F32 table.
  if (error[row] != 0 || counts[row] != Top || !isfinite(seed_tau[row]) ||
      !isfinite(safe_tau[row]) || safe_tau[row] > seed_tau[row] ||
      starts[row] < 0 || ends[row] < starts[row] || ends[row] > nk) {
    if (t == 0) {
      atomicOr(error + row, 1);
      counts[row] = 0;
    }
    return;
  }
  bool bad = false;
  for (int j = t; j < Top; j += blockDim.x) {
    const uint32_t word = uint32_t(words[int64_t(row) * stride + j]);
    const uint32_t high = word >> 20;
    const int id = int(word & 0xFFFFFu);
    // The finite20bit interval has complete low8 bins at both endpoints, so
    // checking its high12 bits is equivalent and does not read the byte plane.
    bad |= high < (h32_record5::Min >> 8) || high > (h32_record5::Max >> 8) ||
           id < starts[row] || id >= ends[row] || id >= nk;
  }
  if (__syncthreads_or(bad) && t == 0) {
    atomicOr(error + row, 1);
    counts[row] = 0;
  }
}
}  // namespace live_record5_seed
namespace live_record5_host {
void validate_seed5_live_nk(Tensor words, Tensor counts, Tensor seed_tau,
                            Tensor safe_tau, Tensor starts, Tensor ends,
                            Tensor error, int64_t nk, Tensor live_nk,
                            int64_t seed) {
  DeviceGuard guard(words.get_device_index());
  // Owner's I32[Q,114688] packed IDs, F32 tau witnesses, I32 intervals/status
  // and I32[1] liveNK are independent allocations. The kernel validates their
  // current values; the serial owner retains all views until stream completion.
  const int64_t rows = words.size(0);
  live_record5_seed::validate5<<<rows, 256, 0, get_current_cuda_stream()>>>(
      words.mutable_data_ptr<int>(), words.stride(0),
      counts.mutable_data_ptr<int>(), seed_tau.mutable_data_ptr<float>(),
      safe_tau.mutable_data_ptr<float>(), starts.mutable_data_ptr<int>(),
      ends.mutable_data_ptr<int>(), error.mutable_data_ptr<int>(),
      live_nk.mutable_data_ptr<int>(), int(nk), int(seed));
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace live_record5_host

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("validate_seed5_live_nk") "(Tensor words, Tensor! counts, Tensor seed_tau, Tensor safe_tau, Tensor starts, Tensor ends, "
      "Tensor! error, int nk, Tensor live_nk, int seed) -> ()");
  ops.def(DOSA_OP("exact_paged_solo_live_nk") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor tau, Tensor ids, Tensor counts, "
      "Tensor tasks, Tensor packs, Tensor! scores, Tensor! outids, Tensor! outcounts, int origin, int sms, "
      "Tensor table, int pages, Tensor starts, Tensor ends, Tensor! error, Tensor live_nk, int seed, "
      "Tensor! group_tasks, Tensor! state) -> ()");
  ops.def(DOSA_OP("exact_paged_common_live_nk") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor tau, Tensor ids, Tensor counts, "
      "Tensor tasks, Tensor packs, Tensor! scores, Tensor! outids, Tensor! outcounts, int origin, int sms, "
      "Tensor table, int pages, Tensor starts, Tensor ends, Tensor! error, Tensor live_nk, int seed, "
      "Tensor! group_tasks, Tensor! state) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(
      DOSA_OP("validate_seed5_live_nk"),
      TORCH_BOX(&DOSA_NAMESPACE::live_record5_host::validate_seed5_live_nk));
  ops.impl(
      DOSA_OP("exact_paged_solo_live_nk"),
      TORCH_BOX(&DOSA_NAMESPACE::live_record5_host::exact_paged_completion<1>));
  ops.impl(
      DOSA_OP("exact_paged_common_live_nk"),
      TORCH_BOX(&DOSA_NAMESPACE::live_record5_host::exact_paged_completion<4>));
}
