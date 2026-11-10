#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace live_record5_completion {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr int PacketIds = 128;
template <int BQ>
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[2][128 * 128], q[BQ * kHeads * 128];
  alignas(128) __nv_bfloat162 weights[BQ * kHeads / 2];
  alignas(128) float scale[2][128];
  alignas(128) int ids[2][128];
  Barrier gathered[2], released[2], done;
  uint32_t tmem;
  int job, skip, power_safe;
  uint32_t emit_counts[BQ][4];
  uint64_t emit_starts[BQ][4];
};
struct Args {
  const float *w, *scale, *tau;
  const int *ids, *packs;
  const CandidateChunk* tasks;
  uint8_t* scores;
  int *outids, *outcounts, *cursor;
  const int *starts, *ends, *live_nk;
  int* error;
  int nk, origin, qrows, output_cap, candidate_seed, task_cap;
  int64_t id_capacity, score_stride, output_id_stride;
};
template <int BQ>
__device__ __forceinline__ void flag_group(const Args& a, int row, int bit) {
  for (int qb = 0; qb < BQ; ++qb)
    if (row * BQ + qb < a.qrows) atomicOr(a.error + row * BQ + qb, bit);
}
struct KeyRange {
  uint32_t begin, span;
};
template <int BQ>
__device__ __forceinline__ KeyRange group_key_range(const Args& a, int row) {
  int64_t begin = a.candidate_seed, end = a.nk;
#pragma unroll
  for (int qb = 0; qb < BQ; ++qb) {
    const int qr = row * BQ + qb;
    if (qr < a.qrows) {
      begin = max(begin, int64_t(a.starts[qr]) - a.origin);
      end = min(end, int64_t(a.ends[qr]) - a.origin);
    }
  }
  return {uint32_t(begin), end > begin ? uint32_t(end - begin) : 0u};
}
template <int BQ>
__device__ __forceinline__ void gather(Smem<BQ>& s,
                                       const cute::TmaDescriptor* map, int slot,
                                       int first_row) {
  // This warp's elected publisher reads only its own 32 published IDs.
#pragma unroll
  for (int packet_part = 0; packet_part < 8; ++packet_part) {
    const int r = first_row + 4 * packet_part;
    int4 coord = *reinterpret_cast<const int4*>(s.ids[slot] + r);
    deep_gemm::ptx::tma_gather4(
        map, s.gathered[slot], s.keys[slot] + r * 128, 0, coord,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
  }
}
// Copyright (c) 2025 DeepSeek
// MIT notice: see common.cuh.

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
#pragma unroll
    for (int query = 0; query < BQ; ++query)
      if (row * BQ + query < a_.qrows)
        group_tau_[query] = a_.tau[row * BQ + query];
    const float* weights = a_.w + row * BQ * kHeads;
    bool safe = true;
    for (int pair = thread_; pair < BQ * kHeads / 2; pair += kConsumerThreads) {
      const int head = 2 * pair;
      float lo = 0.0f, hi = 0.0f;
      if (row * BQ + head / kHeads < a_.qrows) {
        lo = weights[head];
        hi = weights[head + 1];
      }
      safe &= bounded_weight(lo) && bounded_weight(hi);
      s_.weights[pair] = __floats2bfloat162_rn(lo, hi);
    }
    if (__any_sync(0xffffffffu, !safe) && (thread_ & 31) == 0)
      atomicExch(&s_.power_safe, 0);
  }

  // The producer and consumer share two 128-key slots. Epochs continue across
  // tasks, including one-packet tails, so each refill waits for its last
  // reader.
  __device__ __forceinline__ void produce() {
    constexpr uint32_t Full = 0xffffffffu;
    const bool publisher = cute::elect_one_sync();
    const KeyRange range = group_key_range<BQ>(a_, work_.stream_row);
    for (int tile = 0; tile < packets_; ++tile) {
      const uint32_t epoch = first_packet_ + tile;
      const unsigned slot = epoch & 1, phase = (epoch >> 1) & 1;
      s_.released[slot].wait(phase ^ 1);
      const int at = work_.first_slot + tile * PacketIds + thread_;
      const bool active = at < work_.end_slot;
      const int id = active ? a_.ids[at] : 0;
      const bool valid = active && uint32_t(id) - range.begin < range.span;
      const float scale = valid ? a_.scale[id] : 0.f;
      s_.ids[slot][thread_] = valid ? id : 0;
      if (__any_sync(Full, active && !valid) && (thread_ & 31) == 0)
        flag_group<BQ>(a_, work_.stream_row, 1);
      // The K transfer needs the validated IDs, not the corresponding SF load.
      // Each elected sender registers its own bytes before issuing its copies.
      __syncwarp(Full);
      if (publisher) {
        const bool with_query = thread_ < 32 && tile == 0;
        s_.gathered[slot].expect_transaction(
            32 * 128 + (with_query ? BQ * kHeads * 128 : 0));
        if (with_query)
          deep_gemm::tma::copy<128, BQ * kHeads, 128>(
              query_map_, &s_.gathered[slot], s_.q, 0,
              work_.stream_row * BQ * kHeads);
        gather(s_, map_, slot, 32 * (thread_ / 32));
      }
      s_.scale[slot][thread_] = scale;
      __syncwarp(Full);
      if (publisher) s_.gathered[slot].arrive();
    }
  }

  __device__ __forceinline__ void before_tile(int tile) {
    const uint32_t epoch = first_packet_ + tile;
    slot_ = epoch & 1;
    active_keys_ =
        min(PacketIds, work_.end_slot - work_.first_slot - tile * PacketIds);
    s_.gathered[slot_].wait((epoch >> 1) & 1);
    asm volatile("" ::: "memory");
    if (thread_ == 0) {
      deep_gemm::ptx::tcgen05_after_thread_sync();
      issue_h32_mma<128, kMmaQueries * kHeads>(
          reinterpret_cast<cutlass::float_e4m3_t*>(s_.keys[slot_]),
          reinterpret_cast<cutlass::float_e4m3_t*>(s_.q), 0, tm_,
          cute::cast_smem_ptr_to_uint(&s_.done));
    }
    s_.done.wait(mma_sequence_ & 1u);
    deep_gemm::ptx::tcgen05_after_thread_sync();
  }

  // Drain each query half before reusing the128-column TMEM accumulator.
  __device__ __forceinline__ void advance_query_stage(int query) {
    if constexpr (kMmaQueries < BQ) {
      if (query + 1 < BQ && (query + 1) % kMmaQueries == 0) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        named_sync(kConsumerThreads, 3);
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

  __device__ __forceinline__ void run() {
    const bool bounded_weights = s_.power_safe != 0;
    for (int tile = 0; tile < packets_; ++tile) {
      before_tile(tile);
      const float scale = s_.scale[slot_][thread_];
      const bool rescale =
          bounded_weights && __all_sync(0xffffffffu, thread_ >= active_keys_ ||
                                                         bounded_power(scale));
      float scores[BQ]{};
      uint32_t votes[BQ]{};
#pragma unroll
      for (int query = 0; query < BQ; ++query) {
        const int row = work_.stream_row * BQ + query;
        if (row >= a_.qrows || (work_.failed_queries & (1u << query))) {
          advance_query_stage(query);
          continue;
        }
        float accum[kHeads];
#pragma unroll
        for (int part = 0; part < kHeads / 16; ++part)
          load_tmem<16>(tm_ + (query % kMmaQueries) * kHeads + part * 16,
                        accum + part * 16);
        cutlass::arch::fence_view_async_tmem_load();
        const auto* weights = s_.weights + query * kHeads / 2;
        const float value = native_bf16_reduce(accum, weights, scale, rescale);
        scores[query] = value;
        const bool active = thread_ < active_keys_;
        if (active && !isfinite(value)) atomicOr(a_.error + row, 8);
        const float tau = group_tau_[query];
        votes[query] = __ballot_sync(
            0xffffffffu, active && (!isfinite(tau) || value >= tau));
        advance_query_stage(query);
      }
      emit(scores, votes);
      deep_gemm::ptx::tcgen05_before_thread_sync();
      named_sync(kConsumerThreads, 3);
      if (thread_ == 0) s_.released[slot_].arrive();
      ++mma_sequence_;
    }
  }

  __device__ __forceinline__ void emit(const float* scores,
                                       const uint32_t (&votes)[BQ]) {
    const int lane = thread_ & 31, warp = thread_ / 32,
              row_base = work_.stream_row * BQ;
    uint32_t offsets[BQ];
    int base_solo = 0;
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      const uint32_t bits = votes[query], total = __popc(bits);
      offsets[query] = __popc(lane ? bits & ((1u << lane) - 1u) : 0u);
      if constexpr (BQ == 4) {
        if (lane == 0) s_.emit_counts[query][warp] = total;
      } else if (lane == 0 && row_base < a_.qrows && total) {
        base_solo = atomicAdd(a_.outcounts + row_base, int(total));
      }
    }
    if constexpr (BQ == 4) {
      named_sync(kConsumerThreads, 3);
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
      named_sync(kConsumerThreads, 3);
    }
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      int64_t base;
      if constexpr (BQ == 4)
        base = int64_t(s_.emit_starts[query][warp]);
      else
        base = int64_t(uint32_t(__shfl_sync(0xffffffffu, base_solo, 0)));
      if (!(votes[query] & (1u << lane))) continue;
      const float value = scores[query], tau = group_tau_[query];
      const int64_t key = uint32_t(int64_t(s_.ids[slot_][thread_]) + a_.origin);
      const uint32_t code = record5::encode(value);
      const bool valid = isfinite(value) && isfinite(tau) &&
                         __float_as_uint(value) != 0x80000000u && key >= 0 &&
                         key < 0x100000;
      const uint32_t stored = valid ? code : record5::Invalid;
      const int64_t column = base + offsets[query];
      const int row = row_base + query;
      if (column >= 0 && column < a_.output_cap) {
        a_.scores[int64_t(row) * a_.score_stride + column] = uint8_t(stored);
        a_.outids[int64_t(row) * a_.output_id_stride + column] =
            int32_t(record5::pack_word(stored, uint32_t(key)));
        if (!valid) atomicOr(a_.error + row, 32);
      } else
        atomicOr(a_.error + row, 16);
    }
  }

  Smem<BQ>& s_;
  const Args& a_;
  const cute::TmaDescriptor *map_, *query_map_;
  SparseWorkItem work_;
  uint32_t tm_;
  uint32_t first_packet_;
  uint32_t& mma_sequence_;
  int thread_;
  int packets_ =
      (work_.end_slot - work_.first_slot + PacketIds - 1) / PacketIds;
  int slot_ = 0, active_keys_ = 0;
  float group_tau_[BQ]{};
};

__device__ __forceinline__ bool valid_chunk(const Args& a, CandidateChunk task,
                                            int stream_rows) {
  return task.row >= 0 && task.row < stream_rows && task.begin >= 0 &&
         task.begin % 4096 == 0 && task.count > 0 && task.count <= 4096 &&
         task.count <= a.nk - a.candidate_seed &&
         int64_t(task.begin) + 4096 <= min(a.id_capacity, int64_t(INT_MAX));
}

// Keep each register assignment and its complete task loop in one
// warp-group branch. Named barrier 4 joins both roles at task boundaries.
template <int BQ, bool Producer>
__device__ __forceinline__ void work_loop(const Args& a, Smem<BQ>& s,
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
    if (!valid_chunk(a, task, streams)) {
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
      s.power_safe = 1;
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
    SpanScoreAccess<BQ> access{
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
      named_sync(SpanScoreAccess<BQ>::kConsumerThreads, 3);
      access.run();
    }
    first_packet += access.packets_;
    named_sync(256, 4);
  }
}

template <int BQ>
CUTLASS_GLOBAL void __launch_bounds__(256, (BQ * kHeads > 128 ? 3 : 4))
    exact(Args a, const __grid_constant__ cute::TmaDescriptor map,
          const __grid_constant__ cute::TmaDescriptor query_map) {
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
  const unsigned tail = min(unsigned(task_count), gridDim.x);
  const unsigned jobs = unsigned(task_count) + tail,
                 first_split = unsigned(task_count) - tail;
  if (blockIdx.x >= jobs) return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto& s = *reinterpret_cast<Smem<BQ>*>(buf);
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
    cute::TMEM::Allocator1Sm().allocate(TmemColumns, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();
  const int tm = deep_gemm::ptx::ld_shared(&s.tmem);
  if (warp < 4)
    work_loop<BQ, false>(a, s, &map, &query_map, tm, t, first_split, jobs);
  else
    work_loop<BQ, true>(a, s, &map, &query_map, tm, t, first_split, jobs);
  __syncthreads();
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
  const int cap = record5::checked_capacity(scores, outids);
  STD_TORCH_CHECK(std::min(scores.size(0), outids.size(0)) >= q.size(0),
                  "Record5 output planes have insufficient query rows");
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.mutable_data_ptr<int>() + 1, 0,
                                  sizeof(int), get_current_cuda_stream()));
  Args a{weights.const_data_ptr<float>(),
         scale.const_data_ptr<float>(),
         tau.const_data_ptr<float>(),
         ids.const_data_ptr<int>(),
         state.const_data_ptr<int>(),
         reinterpret_cast<const CandidateChunk*>(tasks.const_data_ptr<int>()),
         scores.mutable_data_ptr<uint8_t>(),
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
         cap,
         int(seed),
         int(tasks.size(0)),
         ids.numel(),
         scores.stride(0),
         outids.stride(0)};
  auto map = fp8_map(k, 1);
  // Empty Q admits no task; preserve its cursor/error flow without a zero map.
  auto query_map = q.size(0) ? fp8_map(q, BQ * kHeads) : map;
  launch<exact<BQ>, 256, sizeof(Smem<BQ>)>(4 * sms, a, map, query_map);
}
}  // namespace live_record5_host

namespace live_record5_seed {
constexpr int Top = 2048;
__global__ void validate5(const int* words, int64_t stride, int* counts,
                          const float* seed_tau, const float* safe_tau,
                          const int* starts, const int* ends, int* error,
                          const int* live_nk, int nk_capacity, int seed,
                          int64_t columns, int64_t column_stride) {
  const int nk = *live_nk;
  const int row = blockIdx.x, t = threadIdx.x;
  // The minimum selected seed score witnesses every seed record>=safe_tau.
  if (columns < Top || nk < seed || nk > nk_capacity || error[row] != 0 ||
      counts[row] != Top || !isfinite(seed_tau[row]) ||
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
    const uint32_t word =
        uint32_t(words[int64_t(row) * stride + j * column_stride]);
    const uint32_t high = word >> 20;
    const int id = int(word & 0xFFFFFu);
    // The finite20bit interval has complete low8 bins at both endpoints, so
    // checking its high12 bits is equivalent and does not read the byte plane.
    bad |= high < (record5::Min >> 8) || high > (record5::Max >> 8) ||
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
      live_nk.mutable_data_ptr<int>(), int(nk), int(seed), words.size(1),
      words.stride(1));
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
