#include "common.cuh"
#include "refine_tasks.cuh"
#include "paged_record5.cuh"

namespace DOSA_NAMESPACE {
namespace live_record5_completion {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using refine_tasks::PacketIds;
using refine_tasks::SparseWorkItem;
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
  paged_record5::WarpSpan page_spans[BQ];
};
struct Args {
  const float *w, *scale, *tau;
  const int *ids, *packs;
  const CandidateChunk* tasks;
  int *outcounts, *cursor;
  const int *starts, *ends, *live_nk;
  int* error;
  int nk, qrows, output_cap, candidate_seed, task_cap;
  int64_t id_capacity;
  paged_record5::Arena record_pages;
};
// Copyright (c) 2025 DeepSeek
// MIT notice: see common.cuh.


template <int BQ, bool StrictTies, bool Rank12>
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

  __device__ __forceinline__ float key_scale(int id, bool valid) {
    return valid ? a_.scale[id] : 0.f;
  }

  __device__ __forceinline__ void produce() {
    refine_tasks::produce<BQ, true>(*this);
  }

  static constexpr bool kMx = false, kScoreOnly = false, kStrict = StrictTies;

  __device__ __forceinline__ void issue(int query) {
    issue_h32_mma<128, kMmaQueries * kHeads>(
        reinterpret_cast<cutlass::float_e4m3_t*>(s_.keys[slot_]),
        reinterpret_cast<cutlass::float_e4m3_t*>(s_.q + query * kHeads * 128),
        0, tm_, cute::cast_smem_ptr_to_uint(&s_.done));
  }

  __device__ __forceinline__ void run() { refine_tasks::consume<BQ>(*this); }

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
        using namespace paged_record5;
        const int row = row_base + thread_;
        const uint32_t cap = uint32_t(a_.output_cap);
        const unsigned n = base < cap ? min(total, cap - base) : 0;
        WarpSpan span{int(base / PageCells), -1, -1};
        if (n) {
          const int last = int((base + n - 1) / PageCells);
          span.first = ensure_page(a_.record_pages, row, span.logical_page);
          if (span.first >= 0)
            span.last = last == span.logical_page
                            ? span.first
                            : ensure_page(a_.record_pages, row, last);
        }
        s_.page_spans[thread_] = span;
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
      const bool emit_record = (votes[query] & (1u << lane)) != 0;
      const int64_t column = base + offsets[query];
      unsigned active = 0;
      if constexpr (BQ == 1)
        active = __ballot_sync(
            0xffffffffu, emit_record && column >= 0 && column < a_.output_cap);
      if (!emit_record) continue;
      const float value = scores[query], tau = group_tau_[query];
      const int64_t key = uint32_t(s_.ids[slot_][thread_]);
      const uint32_t code = record5::encode(value);
      const bool valid = isfinite(value) && isfinite(tau) &&
                         __float_as_uint(value) != 0x80000000u && key >= 0 &&
                         key < 0x100000;
      const uint32_t stored = valid ? code : record5::Invalid;
      const int row = row_base + query;
      if (column >= 0 && column < a_.output_cap) {
        const int word =
            Rank12 ? int32_t(record12::pack_word(record5::encode(tau) >> 4,
                                               code >> 4, uint32_t(key)))
                   : int32_t(record5::pack_word(stored, uint32_t(key)));
        if constexpr (BQ == 4) {
          paged_record5::write_span<Rank12>(
              a_.record_pages, s_.page_spans[query], int(column),
              {uint8_t(stored), word});
        } else {
          const auto span = paged_record5::warp_ensure_span(
              a_.record_pages, row, int(base), __popc(active), active);
          paged_record5::write_span<Rank12>(
              a_.record_pages, span, int(column), {uint8_t(stored), word});
        }
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

template <int BQ, bool StrictTies, bool Rank12 = false>
CUTLASS_GLOBAL void __launch_bounds__(256, (BQ * kHeads > 128 ? 3 : 4))
    exact(Args a, const __grid_constant__ cute::TmaDescriptor map,
          const __grid_constant__ cute::TmaDescriptor query_map) {
  constexpr int TmemColumns = BQ * kHeads > 128 ? 128 : BQ * kHeads;
  extern __shared__ __align__(1024) uint8_t buf[];
  using Access = SpanScoreAccess<BQ, StrictTies, Rank12>;
  constexpr int Alignment = StrictTies ? PacketIds : 4096;
  refine_tasks::run_cta<BQ, Alignment, TmemColumns, Access>(
      a, *reinterpret_cast<Smem<BQ>*>(buf), map, query_map);
}
}  // namespace live_record5_completion

namespace live_record5_host {
void exact_span_completion(Tensor q, Tensor k, Tensor scale, Tensor weights,
                           Tensor tau, Tensor ids, Tensor tasks, Tensor state,
                           Tensor scores, Tensor outids, Tensor outcounts,
                           int64_t sms, Tensor starts,
                           Tensor ends, Tensor error, Tensor live_nk,
                           int64_t seed, Tensor page_table, Tensor page_head,
                           bool rank12, bool strict_ties, bool common) {
  using namespace live_record5_completion;
  // Read only published I32[capacity,3] tasks, each containing<=4096 IDs.
  DeviceGuard guard(q.get_device_index());
  const int width = rank12 ? int(outids.size(1))
                           : record5::checked_capacity(scores, outids);
  STD_TORCH_CHECK(ids.dim() == 2 && ids.size(1) == packed20::PageBytes &&
                      ids.is_contiguous(),
                  "BF16 refinement requires packed candidate pages");
  STD_TORCH_CHECK(
      page_head.numel() == 1 && width == paged_record5::PageCells &&
          (rank12 || (scores.sizes() == outids.sizes() &&
                      scores.is_contiguous())) &&
          outids.is_contiguous() &&
          outids.size(0) >= q.size(0) * paged_record5::PrefixPages &&
          page_table.dim() == 2 && page_table.is_contiguous() &&
          page_table.size(0) == q.size(0) &&
          page_table.size(1) == paged_record5::RowPages,
      "Paged Record5 requires aligned planes and one page table per row");
  const paged_record5::Arena records{
      scores.mutable_data_ptr<uint8_t>(), outids.mutable_data_ptr<int>(),
      page_table.mutable_data_ptr<int>(), error.mutable_data_ptr<int>(),
      page_head.mutable_data_ptr<int>(), int(outids.size(0)), int(q.size(0))};
  DOSA_CUDA_CHECK(cudaMemsetAsync(state.mutable_data_ptr<int>() + 1, 0,
                                  sizeof(int), get_current_cuda_stream()));
  Args a{weights.const_data_ptr<float>(),
         scale.const_data_ptr<float>(),
         tau.const_data_ptr<float>(),
         reinterpret_cast<const int*>(ids.const_data_ptr<uint8_t>()),
         state.const_data_ptr<int>(),
         reinterpret_cast<const CandidateChunk*>(tasks.const_data_ptr<int>()),
         outcounts.mutable_data_ptr<int>(),
         state.mutable_data_ptr<int>() + 1,
         starts.const_data_ptr<int>(),
         ends.const_data_ptr<int>(),
         live_nk.const_data_ptr<int>(),
         error.mutable_data_ptr<int>(),
         int(k.size(0)),
         int(q.size(0)),
         paged_record5::MaxCells,
         int(seed),
         int(tasks.size(0)),
         ids.size(0) * packed20::PageIds,
         records};
  auto map = fp8_map(k, 1);
  auto emit = [&]<int BQ>() {
    auto query_map = q.size(0) ? fp8_map(q, BQ * kHeads) : map;
    if (rank12)
      launch<exact<BQ, true, true>, 256, sizeof(Smem<BQ>)>(4 * sms, a, map, query_map);
    else if (strict_ties)
      launch<exact<BQ, true>, 256, sizeof(Smem<BQ>)>(4 * sms, a, map, query_map);
    else
      launch<exact<BQ, false>, 256, sizeof(Smem<BQ>)>(4 * sms, a, map, query_map);
  };
  if (common)
    emit.template operator()<4>();
  else
    emit.template operator()<1>();
}
}  // namespace live_record5_host

namespace live_record5_seed {
constexpr int Top = 2048;
template <bool Rank12>
__global__ void validate5(const int* words, int64_t stride, int* counts,
                          const float* seed_tau, const float* safe_tau,
                          const int* starts, const int* ends, int* error,
                          const int* live_nk, int nk_capacity, int seed,
                          int64_t columns, int64_t column_stride) {
  const int nk = *live_nk;
  const int row = blockIdx.x, t = threadIdx.x;
  const bool cutoff_matches =
      Rank12 ? safe_tau[row] == seed_tau[row] : safe_tau[row] <= seed_tau[row];
  // The minimum selected seed score witnesses every seed record>=safe_tau.
  if (columns < Top || nk < seed || nk > nk_capacity || error[row] != 0 ||
      counts[row] != Top || !isfinite(seed_tau[row]) ||
      !isfinite(safe_tau[row]) || !cutoff_matches ||
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
    if constexpr (!Rank12)
      bad |= high < (record5::Min >> 8) || high > (record5::Max >> 8);
    bad |= id < starts[row] || id >= ends[row] || id >= nk;
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
                            int64_t seed, bool rank12) {
  DeviceGuard guard(words.get_device_index());
  auto validate = [&]<bool Rank12>() {
    launch<live_record5_seed::validate5<Rank12>, 256>(
        words.size(0), words.mutable_data_ptr<int>(), words.stride(0),
        counts.mutable_data_ptr<int>(), seed_tau.mutable_data_ptr<float>(),
        safe_tau.mutable_data_ptr<float>(), starts.mutable_data_ptr<int>(),
        ends.mutable_data_ptr<int>(), error.mutable_data_ptr<int>(),
        live_nk.mutable_data_ptr<int>(), int(nk), int(seed), words.size(1),
        words.stride(1));
  };
  if (rank12)
    validate.template operator()<true>();
  else
    validate.template operator()<false>();
}

}  // namespace live_record5_host

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("validate_seed5_live_nk",
              "(Tensor words, Tensor! counts, Tensor seed_tau, Tensor "
              "safe_tau, Tensor starts, Tensor ends, "
              "Tensor! error, int nk, Tensor live_nk, int seed, "
              "bool rank12=False) -> ()",
              CUDA, live_record5_host::validate_seed5_live_nk)
DOSA_REGISTER(
    "exact_span_packed20_live_nk",
    "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor tau, Tensor "
    "ids, "
    "Tensor tasks, Tensor! state, Tensor! scores, Tensor! outids, "
    "Tensor! outcounts, int sms, Tensor starts, Tensor ends, "
    "Tensor! error, Tensor live_nk, int seed, Tensor(a!) page_table, "
    "Tensor(b!) page_head, bool rank12, bool strict_ties, bool common) -> ()",
    CUDA, live_record5_host::exact_span_completion)
