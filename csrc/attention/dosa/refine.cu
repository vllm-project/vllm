#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace live_record5_completion {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr int PacketIds = 128;
template <int BQ>
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[2][128 * 128], q[BQ * kHeads * 128];
  alignas(128) float weights[BQ * kHeads], scale[2][128];
  alignas(128) int ids[2][128], coordinates[2][128];
  Barrier gathered[2], done;
  uint32_t tmem;
  int job;
  int reserved[4], skip;
  uint32_t emit_counts[BQ][4], emit_base[BQ];
  uint64_t emit_starts[BQ][4];
};
struct Args {
  const uint8_t* q;
  const float *w, *scale, *tau;
  const int *ids, *packs;
  const CandidateChunk* tasks;
  void* scores;
  int *outids, *outcounts, *cursor;
  const int *starts, *ends, *live_nk;
  int* error;
  int nk, origin, qrows, output_cap, candidate_seed, task_cap;
  int64_t id_capacity, score_stride, output_id_stride;
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
struct FutureMetadata {
  int id;
  float scale;
  bool bad;
};
struct FutureId {
  int range_tag, id;
  uint32_t flags;
};
__device__ __forceinline__ FutureId load_future_id(const Args& a, int row,
                                                   int begin, int stop, int t) {
  const bool range = begin >= 0 && stop >= begin && stop <= a.id_capacity;
  const bool active = int64_t(begin) + t < stop;
  const bool address = active && range;
  const int id = address ? a.ids[int64_t(begin) + t] : 0;
  return {range ? 0 : -1, id, uint32_t(active) | (uint32_t(address) << 1)};
}
__device__ __forceinline__ FutureMetadata
finish_future_scale(const Args& a, const FutureId& f, Visibility visibility) {
  // Inline PTX keeps ID checks late; ptxas hoists equivalent C++ checks.
  int id, bad;
  float scale;
  asm volatile(
      "{ .reg .u32 flags; .reg .s64 visible; .reg .u64 addr;\n"
      "  .reg .pred valid,active,badp,p;\n"
      "  mov.u32 %0,0; mov.f32 %1,0f00000000;\n"
      "  and.b32 flags,%5,2; setp.ne.u32 valid,flags,0;\n"
      "  setp.ge.s32 p,%4,%12; and.pred valid,valid,p;\n"
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
      : "r"(f.range_tag), "r"(f.id), "r"(f.flags), "r"(a.nk), "r"(1),
        "l"(int64_t(a.origin)), "l"(int64_t(visibility.begin)),
        "l"(int64_t(visibility.end)), "l"(a.scale), "r"(a.candidate_seed)
      : "memory");
  return {id, scale, bad != 0};
}
template <int BQ, bool CachedVisibility = false>
__device__ __forceinline__ FutureMetadata
load_future(const Args& a, int row, int begin, int stop, int t,
            Visibility visibility = {}) {
  const bool range = begin >= 0 && stop >= begin && stop <= a.id_capacity;
  const bool active = int64_t(begin) + t < stop;
  const bool address = active && range;
  const int id = address ? a.ids[int64_t(begin) + t] : 0;
  bool valid = address && id >= a.candidate_seed && id < a.nk;
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
  return {valid ? id : 0, valid ? a.scale[id] : 0.0f,
          !range || (active && !valid)};
}
template <int BQ>
__device__ __forceinline__ void publish_future(Smem<BQ>& s, const Args& a,
                                               int row, int slot, int t,
                                               const FutureMetadata& future) {
  s.ids[slot][t] = future.id;
  s.coordinates[slot][t] = future.id;
  s.scale[slot][t] = future.scale;
  const int bad = __syncthreads_or(future.bad);
  if (t == 0 && bad) flag_group<BQ>(a, row, 1);
}
template <int BQ, bool CachedVisibility = false>
__device__ __forceinline__ void prepare_metadata(Smem<BQ>& s, const Args& a,
                                                 int row, int begin, int stop,
                                                 int slot, int t,
                                                 Visibility visibility = {}) {
  __syncthreads();
  const auto future =
      load_future<BQ, CachedVisibility>(a, row, begin, stop, t, visibility);
  publish_future(s, a, row, slot, t, future);
  __syncthreads();
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
}  // namespace live_record5_completion

// Copyright (c) 2025 DeepSeek
// MIT notice: see common.cuh.

namespace live_record5_completion {

struct SparseWorkItem {
  int stream_row;
  int first_slot;
  int end_slot;
  uint32_t failed_queries;
};

template <int BQ>
struct SpanScoreAccess {
  static_assert(BQ == 1 || BQ == 4);
  static constexpr int kMmaQueries = BQ * kHeads > 128 ? 128 / kHeads : BQ;
  static constexpr int kConsumerThreads = 128;

  __device__ __forceinline__ void initialize_work() {
    const int row = work_.stream_row;
    if constexpr (BQ == 4) {
      visibility_ = group_visibility<BQ>(a_, row);
#pragma unroll
      for (int query = 0; query < BQ; ++query)
        if (row * BQ + query < a_.qrows)
          group_tau_[query] = a_.tau[row * BQ + query];
    } else if constexpr (BQ == 1) {
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

  // Drain each query half before reusing the128-column TMEM accumulator.
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
          load_tmem<16>(tm_ + (query % kMmaQueries) * kHeads + part * 16,
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

  Smem<BQ>& s_;
  const Args& a_;
  const cute::TmaDescriptor* map_;
  SparseWorkItem work_;
  uint32_t tm_;
  uint32_t& raw_phases_;
  uint32_t& mma_sequence_;
  int thread_;
  int packets_ =
      (work_.end_slot - work_.first_slot + PacketIds - 1) / PacketIds;
  int slot_ = 0, active_keys_ = 0;
  Visibility visibility_{};
  float group_tau_[BQ]{};
  FutureMetadata next_{}, future_{};
  FutureId id_stage_{};
  bool pending_scale_ = false;
};

__device__ __forceinline__ bool valid_chunk(const Args& a, CandidateChunk task,
                                            int stream_rows) {
  return task.row >= 0 && task.row < stream_rows && task.begin >= 0 &&
         task.begin % 4096 == 0 && task.count > 0 && task.count <= 4096 &&
         task.count <= a.nk - a.candidate_seed &&
         int64_t(task.begin) + 4096 <= min(a.id_capacity, int64_t(INT_MAX));
}

template <int BQ>
CUTLASS_GLOBAL void __launch_bounds__(128, 4)
    exact(Args a, const __grid_constant__ cute::TmaDescriptor map) {
  constexpr int TmemColumns = BQ * kHeads > 128 ? 128 : BQ * kHeads;
  const int logical = *a.live_nk, task_count = *a.packs;
  if (logical < a.candidate_seed || logical > a.nk || task_count < 0 ||
      task_count > a.task_cap) {
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < a.qrows;
         qr += gridDim.x * blockDim.x)
      atomicOr(a.error + qr, 2);
    return;
  }
  a.nk = logical;
  if (blockIdx.x >= task_count) return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto& s = *reinterpret_cast<Smem<BQ>*>(buf);
  const int t = threadIdx.x, warp = t / 32;
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
  const int tm = deep_gemm::ptx::ld_shared(&s.tmem);
  uint32_t phase_bits = 0, done_sequence = 0;
  for (;;) {
    if (t == 0) s.job = atomicAdd(a.cursor, 1);
    __syncthreads();
    const int index = s.job;
    if (index >= task_count) break;
    const CandidateChunk task = a.tasks[index];
    const int row = task.row;
    if (!valid_chunk(a, task, (a.qrows + BQ - 1) / BQ)) {
      if (row >= 0 && row < (a.qrows + BQ - 1) / BQ) {
        if (t == 0) flag_group<BQ>(a, row, 2);
      } else {
        for (int qr = t; qr < a.qrows; qr += blockDim.x)
          atomicOr(a.error + qr, 2);
      }
      __syncthreads();
      continue;
    }
    if (t == 0) {
      s.skip = 0;
      for (int qb = 0; qb < BQ; ++qb)
        if (row * BQ + qb < a.qrows && a.error[row * BQ + qb] != 0)
          s.skip |= 1 << qb;
    }
    __syncthreads();
    const int failed = s.skip, real = min(BQ, a.qrows - row * BQ);
    if (failed == (1 << real) - 1) {
      __syncthreads();
      continue;
    }
    SpanScoreAccess<BQ> access{
        s,
        a,
        &map,
        {row, task.begin, task.begin + task.count, uint32_t(failed)},
        uint32_t(tm),
        phase_bits,
        done_sequence,
        t};
    access.initialize_work();
    access.run();
  }
  if (warp == 0) cute::TMEM::Allocator1Sm().free(tm, TmemColumns);
}
}  // namespace live_record5_completion

namespace live_record5_host {
template <int BQ>
void exact_span_completion(Tensor q, Tensor k, Tensor scale, Tensor weights,
                           Tensor tau, Tensor ids, Tensor tasks, Tensor state,
                           Tensor scores, Tensor outids, Tensor outcounts,
                           int64_t origin, int64_t sms, Tensor starts,
                           Tensor ends, Tensor error, Tensor live_nk,
                           int64_t seed) {
  using namespace live_record5_completion;
  // Read only published I32[capacity,3] tasks, each containing<=4096 IDs.
  DeviceGuard guard(q.get_device_index());
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.mutable_data_ptr<int>() + 1, 0,
                                  sizeof(int), get_current_cuda_stream()));
  Args a{static_cast<const uint8_t*>(q.data_ptr()),
         weights.const_data_ptr<float>(),
         scale.const_data_ptr<float>(),
         tau.const_data_ptr<float>(),
         ids.const_data_ptr<int>(),
         state.const_data_ptr<int>(),
         reinterpret_cast<const CandidateChunk*>(tasks.const_data_ptr<int>()),
         scores.data_ptr(),
         outids.mutable_data_ptr<int>(),
         outcounts.mutable_data_ptr<int>(),
         state.mutable_data_ptr<int>() + 1,
         starts.const_data_ptr<int>(),
         ends.const_data_ptr<int>(),
         live_nk.const_data_ptr<int>(),
         error.mutable_data_ptr<int>(),
         int(k.size(0)),
         int(origin),
         int(q.size(0)),
         int(std::min(scores.size(1), outids.size(1))),
         int(seed),
         int(tasks.size(0)),
         ids.numel(),
         scores.stride(0),
         outids.stride(0)};
  auto map = fp8_map(k, 1);
  launch<exact<BQ>, 128, sizeof(Smem<BQ>)>(4 * sms, a, map);
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
  // The minimum selected seed score witnesses every seed record>=safe_tau.
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
  launch<live_record5_seed::validate5, 256>(
      words.size(0), words.mutable_data_ptr<int>(), words.stride(0),
      counts.mutable_data_ptr<int>(), seed_tau.mutable_data_ptr<float>(),
      safe_tau.mutable_data_ptr<float>(), starts.mutable_data_ptr<int>(),
      ends.mutable_data_ptr<int>(), error.mutable_data_ptr<int>(),
      live_nk.mutable_data_ptr<int>(), int(nk), int(seed));
}

}  // namespace live_record5_host

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("validate_seed5_live_nk",
              "(Tensor words, Tensor! counts, Tensor seed_tau, Tensor "
              "safe_tau, Tensor starts, Tensor ends, "
              "Tensor! error, int nk, Tensor live_nk, int seed) -> ()",
              CUDA, live_record5_host::validate_seed5_live_nk)
DOSA_REGISTER(
    "exact_span_solo_live_nk",
    "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor tau, Tensor "
    "ids, "
    "Tensor tasks, Tensor! state, Tensor! scores, Tensor! outids, "
    "Tensor! outcounts, int origin, int sms, Tensor starts, Tensor ends, "
    "Tensor! error, Tensor live_nk, int seed) -> ()",
    CUDA, live_record5_host::exact_span_completion<1>)
DOSA_REGISTER(
    "exact_span_common_live_nk",
    "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor tau, Tensor "
    "ids, "
    "Tensor tasks, Tensor! state, Tensor! scores, Tensor! outids, "
    "Tensor! outcounts, int origin, int sms, Tensor starts, Tensor ends, "
    "Tensor! error, Tensor live_nk, int seed) -> ()",
    CUDA, live_record5_host::exact_span_completion<4>)
