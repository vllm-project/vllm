#pragma once
#include "common.cuh"
#include "native_fp32_reduce.cuh"
#include "packed20.cuh"

namespace DOSA_NAMESPACE::refine_tasks {
using namespace certicascade_streaming_exact;

inline constexpr int PacketIds = 128;
struct SparseWorkItem {
  int stream_row;
  int first_slot;
  int end_slot;
  uint32_t failed_queries;
};

template <int BQ, class Args>
__device__ __forceinline__ void flag_group(const Args& a, int row, int bit) {
  for (int qb = 0; qb < BQ; ++qb)
    if (row * BQ + qb < a.qrows) atomicOr(a.error + row * BQ + qb, bit);
}
struct KeyRange {
  uint32_t begin, span;
};
template <int BQ, class Args>
__device__ __forceinline__ KeyRange group_key_range(const Args& a, int row) {
  int64_t begin = a.candidate_seed, end = a.nk;
#pragma unroll
  for (int qb = 0; qb < BQ; ++qb) {
    const int qr = row * BQ + qb;
    if (qr < a.qrows) {
      begin = max(begin, int64_t(a.starts[qr]));
      end = min(end, int64_t(a.ends[qr]));
    }
  }
  return {uint32_t(begin), end > begin ? uint32_t(end - begin) : 0u};
}
template <int Alignment, class Args>
__device__ __forceinline__ bool valid_chunk(const Args& a, CandidateChunk task,
                                            int stream_rows) {
  const int64_t storage =
      ((int64_t(task.count) + Alignment - 1) / Alignment) * Alignment;
  return task.row >= 0 && task.row < stream_rows && task.begin >= 0 &&
         task.begin % Alignment == 0 && task.count > 0 && task.count <= 4096 &&
         task.count <= a.nk - a.candidate_seed &&
         int64_t(task.begin) + storage <= min(a.id_capacity, int64_t(INT_MAX));
}

// The two 128-key slots keep their epochs across work-item boundaries.
template <int BQ, bool Packed20, class Access>
__device__ __forceinline__ void produce(Access& ctx) {
  constexpr uint32_t Full = 0xffffffffu;
  constexpr bool ReadBeforeWait = Access::kMx && Packed20;
  const bool publisher = cute::elect_one_sync();
  auto& s = ctx.s_;
  auto const& a = ctx.a_;
  auto const& work = ctx.work_;
  const int t = ctx.thread_;
  const KeyRange range = group_key_range<BQ>(a, work.stream_row);
  for (int tile = 0; tile < ctx.packets_; ++tile) {
    const uint32_t epoch = ctx.first_packet_ + tile;
    const unsigned slot = epoch & 1, phase = (epoch >> 1) & 1;
    if constexpr (!ReadBeforeWait) s.released[slot].wait(phase ^ 1);
    const int at = work.first_slot + tile * PacketIds + t;
    const bool active = at < work.end_slot;
    int id = 0;
    if (active) {
      if constexpr (ReadBeforeWait) {
        static_assert(PacketIds == packed20::PageIds);
        // CandidateChunk and nonempty split starts are page-aligned.
        const int page = work.first_slot / PacketIds + tile;
        id = packed20::load(a.ids, int(a.id_capacity / PacketIds), page, t);
      } else if constexpr (Packed20) {
        id = packed20::load(a.ids, int(a.id_capacity / PacketIds),
                            at / PacketIds, at % PacketIds);
      } else {
        id = a.ids[at];
      }
    }
    const bool valid = active && uint32_t(id) - range.begin < range.span;
    const auto scale = ctx.key_scale(id, valid);
    if constexpr (ReadBeforeWait) s.released[slot].wait(phase ^ 1);
    s.ids[slot][t] = valid ? id : 0;
    if (__any_sync(Full, active && !valid) && (t & 31) == 0)
      flag_group<BQ>(a, work.stream_row, 1);
    // Each warp registers and publishes only its own ID slice.
    __syncwarp(Full);
    if (publisher) {
      const bool with_query = t < 32 && tile == 0;
      s.gathered[slot].expect_transaction(
          32 * 128 + (with_query ? BQ * kHeads * 128 : 0));
      if (with_query)
        deep_gemm::tma::copy<128, BQ * kHeads, 128>(
            ctx.query_map_, &s.gathered[slot], s.q, 0,
            work.stream_row * BQ * kHeads);
#pragma unroll
      for (int part = 0; part < 8; ++part) {
        const int row = 32 * (t / 32) + 4 * part;
        int4 coords = *reinterpret_cast<const int4*>(s.ids[slot] + row);
        deep_gemm::ptx::tma_gather4(
            ctx.map_, s.gathered[slot], s.keys[slot] + row * 128, 0, coords,
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
      }
    }
    s.scale[slot][t] = scale;
    __syncwarp(Full);
    if (publisher) s.gathered[slot].arrive();
  }
}

template <int BQ, class Access>
__device__ __forceinline__ void advance_query_stage(Access& ctx, int query) {
  constexpr int MmaQueries = Access::kMmaQueries;
  if constexpr (MmaQueries < BQ) {
    if (query + 1 < BQ && (query + 1) % MmaQueries == 0) {
      deep_gemm::ptx::tcgen05_before_thread_sync();
      named_sync(Access::kConsumerThreads, 3);
      ++ctx.mma_sequence_;
      if (ctx.thread_ == 0) {
        deep_gemm::ptx::tcgen05_after_thread_sync();
        ctx.issue(query + 1);
      }
      ctx.s_.done.wait(ctx.mma_sequence_ & 1u);
      deep_gemm::ptx::tcgen05_after_thread_sync();
    }
  }
}

template <class Access>
__device__ __forceinline__ void before_tile(Access& ctx, int tile) {
  const uint32_t epoch = ctx.first_packet_ + tile;
  ctx.slot_ = epoch & 1;
  ctx.active_keys_ = min(PacketIds, ctx.work_.end_slot - ctx.work_.first_slot -
                                      tile * PacketIds);
  ctx.s_.gathered[ctx.slot_].wait((epoch >> 1) & 1);
  asm volatile("" ::: "memory");
  if constexpr (Access::kMx) ctx.upload_scales();
  if (ctx.thread_ == 0) {
    deep_gemm::ptx::tcgen05_after_thread_sync();
    ctx.issue(0);
  }
  ctx.s_.done.wait(ctx.mma_sequence_ & 1u);
  deep_gemm::ptx::tcgen05_after_thread_sync();
}

template <int BQ, class Access>
__device__ __forceinline__ void consume(Access& ctx) {
  constexpr int MmaQueries = Access::kMmaQueries;
  bool bounded_weights = false;
  if constexpr (!Access::kMx) bounded_weights = ctx.s_.power_safe != 0;
  for (int tile = 0; tile < ctx.packets_; ++tile) {
    before_tile(ctx, tile);
    float scale = 0.f;
    bool rescale = false;
    if constexpr (!Access::kMx) {
      scale = ctx.s_.scale[ctx.slot_][ctx.thread_];
      rescale = bounded_weights &&
                __all_sync(0xffffffffu, ctx.thread_ >= ctx.active_keys_ ||
                                           bounded_power(scale));
    }
    float scores[BQ]{};
    uint32_t votes[BQ]{};
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      const int row = ctx.work_.stream_row * BQ + query;
      if (row >= ctx.a_.qrows || (ctx.work_.failed_queries & (1u << query))) {
        advance_query_stage<BQ>(ctx, query);
        continue;
      }
      float accum[kHeads];
#pragma unroll
      for (int part = 0; part < kHeads / 16; ++part)
        load_tmem<16>(ctx.tm_ + (query % MmaQueries) * kHeads + part * 16,
                      accum + part * 16);
      cutlass::arch::fence_view_async_tmem_load();
      float value;
      if constexpr (Access::kMx) {
        native_fp32::Reduction total;
        native_fp32::update(total, accum, ctx.s_.weights + query * kHeads,
                            kHeads);
        value = total.finish();
      } else {
        const auto* weights = ctx.s_.weights + query * kHeads / 2;
        value = native_bf16_reduce(accum, weights, scale, rescale);
      }
      scores[query] = value;
      const bool active = ctx.thread_ < ctx.active_keys_;
      if constexpr (Access::kScoreOnly)
        if (active)
          ctx.a_.scores[ctx.work_.first_slot + tile * PacketIds + ctx.thread_] =
              value;
      if (active && !isfinite(value)) atomicOr(ctx.a_.error + row, 8);
      if constexpr (!Access::kScoreOnly) {
        const float tau = ctx.group_tau_[query];
        const bool qualifies = Access::kStrict ? value > tau : value >= tau;
        votes[query] =
            __ballot_sync(0xffffffffu, active && (!isfinite(tau) || qualifies));
      }
      advance_query_stage<BQ>(ctx, query);
    }
    if constexpr (!Access::kScoreOnly) ctx.emit(scores, votes);
    deep_gemm::ptx::tcgen05_before_thread_sync();
    named_sync(Access::kConsumerThreads, 3);
    if (ctx.thread_ == 0) ctx.s_.released[ctx.slot_].arrive();
    ++ctx.mma_sequence_;
  }
}

// Keep each register assignment and its complete task loop in one
// warp-group branch. Named barrier 4 joins both roles at task boundaries.
template <int BQ, bool Producer, int Alignment, class Access, class Args,
          class Storage>
__device__ __forceinline__ void run(const Args& a, Storage& s,
                                          const cute::TmaDescriptor* map,
                                          const cute::TmaDescriptor* query_map,
                                          int tm, int t, unsigned first_split,
                                          unsigned jobs) {
  if constexpr (!Producer)
    cutlass::arch::warpgroup_reg_alloc<(BQ * kHeads > 128 ? 128 : 96)>();
  else
    cutlass::arch::warpgroup_reg_dealloc<32>();
  uint32_t first_packet = 0, done_sequence = 0;
  for (;;) {
    if (!Producer && t == 0) s.job = atomicAdd(a.cursor, 1);
    named_sync(256, 4);
    const unsigned ticket = uint32_t(s.job);
    if (ticket >= jobs) break;
    const bool split = ticket >= first_split;
    const unsigned local = ticket - first_split;
    const CandidateChunk task =
        a.tasks[split ? first_split + local / 2 : ticket];
    const int row = task.row, streams = (a.qrows + BQ - 1) / BQ;
    if (!valid_chunk<Alignment>(a, task, streams)) {
      if (row >= 0 && row < streams) {
        if (t == 0) flag_group<BQ>(a, row, 2);
      } else {
        for (int qr = t; qr < a.qrows; qr += blockDim.x)
          atomicOr(a.error + qr, 2);
      }
      named_sync(256, 4);
      continue;
    }
    if (t == 0) {
      s.skip = 0;
      if constexpr (requires { s.power_safe; }) s.power_safe = 1;
      for (int qb = 0; qb < BQ; ++qb)
        if (row * BQ + qb < a.qrows && a.error[row * BQ + qb] != 0)
          s.skip |= 1 << qb;
    }
    named_sync(256, 4);
    const int failed = s.skip, real = min(BQ, a.qrows - row * BQ);
    // Split the validated tail task on an existing 128-key packet boundary.
    const int cut =
        min(task.count, cute::ceil_div(task.count, 2 * PacketIds) * PacketIds);
    const int first = split && (local & 1) ? cut : 0;
    const int stop = split && !(local & 1) ? cut : task.count;
    if (first == stop || failed == (1 << real) - 1) {
      named_sync(256, 4);
      continue;
    }
    Access access{
        s,
        a,
        map,
        query_map,
        {row, task.begin + first, task.begin + stop, uint32_t(failed)},
        uint32_t(tm),
        first_packet,
        done_sequence,
        t % PacketIds};
    if constexpr (Producer) {
      access.produce();
    } else {
      access.initialize_work();
      named_sync(Access::kConsumerThreads, 3);
      access.run();
    }
    first_packet += access.packets_;
    named_sync(256, 4);
  }
}

template <int BQ, int Alignment, int Columns, class Access, class Args,
          class Storage>
__device__ __forceinline__ void run_cta(Args& a, Storage& s,
                                         const cute::TmaDescriptor& map,
                                         const cute::TmaDescriptor& query_map) {
  const int logical = *a.live_nk, task_count = *a.packs;
  if (logical < a.candidate_seed || logical > a.nk || task_count < 0 ||
      task_count > a.task_cap) {
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < a.qrows;
         qr += gridDim.x * blockDim.x)
      atomicOr(a.error + qr, 2);
    return;
  }
  a.nk = logical;
  const unsigned tail = min(unsigned(task_count), gridDim.x);
  const unsigned jobs = unsigned(task_count) + tail,
                 first_split = unsigned(task_count) - tail;
  if (blockIdx.x >= jobs) return;
  const int t = threadIdx.x, warp = t / 32;
  if (t == 0) {
    s.gathered[0].init(4);
    s.gathered[1].init(4);
    s.released[0].init(1);
    s.released[1].init(1);
    s.done.init(1);
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(&map);
    cute::prefetch_tma_descriptor(&query_map);
  }
  if (warp == 0) {
    cute::TMEM::Allocator1Sm().allocate(Columns, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();
  const int tm = deep_gemm::ptx::ld_shared(&s.tmem);
  if constexpr (Access::kMx) Access::initialize_scale_bank(s, t, warp, tm);
  if (warp < 4)
    run<BQ, false, Alignment, Access>(
        a, s, &map, &query_map, tm, t, first_split, jobs);
  else
    run<BQ, true, Alignment, Access>(
        a, s, &map, &query_map, tm, t, first_split, jobs);
  __syncthreads();
  if (warp == 0) cute::TMEM::Allocator1Sm().free(tm, Columns);
}

}  // namespace DOSA_NAMESPACE::refine_tasks
