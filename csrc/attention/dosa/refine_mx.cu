#include "common.cuh"
#include "refine_mx_tile.cuh"
#include "refine_tasks.cuh"

namespace DOSA_NAMESPACE {
namespace live_record5_completion {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using refine_tasks::PacketIds;
using refine_tasks::SparseWorkItem;
template <int BQ>
struct alignas(1024) MxSmem {
  alignas(1024) uint8_t keys[2][128 * 128], q[BQ * kHeads * 128];
  alignas(128) float weights[BQ * kHeads];
  alignas(128) uint32_t scale[2][128], unit_q[128];
  alignas(128) int ids[2][128];
  Barrier gathered[2], released[2], done;
  uint32_t tmem;
  int job, skip;
};
struct Args {
  const float *w, *tau;
  const int *ids, *packs;
  const CandidateChunk* tasks;
  int* cursor;
  const int *starts, *ends, *live_nk;
  int* error;
  int nk, qrows, candidate_seed, task_cap;
  int64_t id_capacity;
  const uint32_t* packed_scale = nullptr;
  native_fp32_pages::Arena paged{};
  float* scores = nullptr;
};
// Copyright (c) 2025 DeepSeek
// MIT notice: see common.cuh.


template <int BQ, bool ScoreOnly = false, bool Packed20 = false>
struct SpanScoreAccess {
  static_assert(BQ == 1 || BQ == 4);
  // At H32, two queries and their MX scales fit in 128 TMEM columns.
  static constexpr int kMmaQueries = BQ == 4 ? 2 : BQ;
  static constexpr int kConsumerThreads = 128;

  __device__ __forceinline__ void initialize_work() {
    const int row = work_.stream_row;
#pragma unroll
    for (int query = 0; query < BQ; ++query)
      if constexpr (!ScoreOnly)
        if (row * BQ + query < a_.qrows)
          group_tau_[query] = a_.tau[row * BQ + query];
    const float* weights = a_.w + row * BQ * kHeads;
    for (int h = thread_; h < BQ * kHeads; h += kConsumerThreads)
      s_.weights[h] = weights[h];
  }

  __device__ __forceinline__ uint32_t key_scale(int id, bool valid) {
    return valid ? a_.packed_scale[id] : 0x7F7F7F7Fu;
  }

  __device__ __forceinline__ void produce() {
    refine_tasks::produce<BQ, Packed20>(*this);
  }

  static constexpr bool kMx = true, kScoreOnly = ScoreOnly, kStrict = false;

  __device__ __forceinline__ static void initialize_scale_bank(
      MxSmem<BQ>& s, int t, int warp, int tm) {
    if (t < 128) s.unit_q[t] = 0x7F7F7F7Fu;
    __syncthreads();
    if (warp == 0) mx_tile::upload128(s.unit_q, tm + kMmaQueries * kHeads);
    __syncthreads();
  }

  __device__ __forceinline__ void upload_scales() {
    if (thread_ < 32)
      mx_tile::upload128(s_.scale[slot_],
                         tm_ + kMmaQueries * kHeads + 4 + slot_ * 4);
    named_sync(kConsumerThreads, 3);
  }

  __device__ __forceinline__ void issue(int query) {
    mx_tile::issue<kMmaQueries * kHeads>(
        s_.keys[slot_], s_.q + query * kHeads * 128, tm_,
        tm_ + kMmaQueries * kHeads + 4 + slot_ * 4,
        tm_ + kMmaQueries * kHeads, cute::cast_smem_ptr_to_uint(&s_.done));
  }

  __device__ __forceinline__ void run() { refine_tasks::consume<BQ>(*this); }

  __device__ __forceinline__ void emit(const float* scores,
                                       const uint32_t (&votes)[BQ]) {
    static_assert((BQ == 1 || BQ == 4) && (kHeads == 32 || kHeads == 64),
                  "True-MX entry supports H32/H64 and BQ1/BQ4");
    const int lane = thread_ & 31;
#pragma unroll
    for (int query = 0; query < BQ; ++query) {
      const unsigned mask = votes[query];
      if (mask & (1u << lane)) {
        const int row = work_.stream_row * BQ + query;
        const int ordinal = refine_mx_pages::warp_reserve(a_.paged, row, mask);
        if (ordinal >= 0) {
          const uint32_t id = uint32_t(s_.ids[slot_][thread_]);
          const uint32_t code = record5::encode(scores[query]);
          native_fp32_pages::write_reserved(
              a_.paged, row, ordinal,
              {uint8_t(code), int32_t(record5::pack_word(code, id))});
        }
      }
    }
  }

  MxSmem<BQ>& s_;
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

template <int BQ, bool ScoreOnly = false, bool Packed20 = false>
CUTLASS_GLOBAL void __launch_bounds__(256, (BQ * kHeads > 128 ? 3 : 4))
    exact(Args a, const __grid_constant__ cute::TmaDescriptor map,
          const __grid_constant__ cute::TmaDescriptor query_map) {
  static_assert((BQ == 1 || BQ == 4) && (kHeads == 32 || kHeads == 64),
                "True-MX entry supports H32/H64 and BQ1/BQ4");
  constexpr int N = SpanScoreAccess<BQ, ScoreOnly, Packed20>::kMmaQueries * kHeads;
  constexpr int TmemColumns = N + 12 <= 64 ? 64 : N + 12 <= 128 ? 128 : 256;
  extern __shared__ __align__(1024) uint8_t buf[];
  using Access = SpanScoreAccess<BQ, ScoreOnly, Packed20>;
  constexpr int Alignment = Packed20 && kHeads == 32 ? PacketIds : 4096;
  refine_tasks::run_cta<BQ, Alignment, TmemColumns, Access>(
      a, *reinterpret_cast<MxSmem<BQ>*>(buf), map, query_map);
}
}  // namespace live_record5_completion

namespace {
using live_record5_completion::Args;

void check_operands(const Tensor& q, const Tensor& k, const Tensor& scales,
                    const Tensor& weights) {
  using Scalar = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(
      q.dim() == 3 && q.size(0) > 0 && q.size(1) == dosa_geometry::Heads &&
          q.size(2) == 128 && q.is_contiguous() && k.dim() == 2 &&
          k.size(1) == 128 && k.is_contiguous() &&
          weights.scalar_type() == Scalar::Float && weights.dim() == 2 &&
          weights.size(0) == q.size(0) && weights.size(1) == q.size(1) &&
          weights.is_contiguous() && scales.scalar_type() == Scalar::Int &&
          scales.dim() == 1 && scales.size(0) == k.size(0) &&
          scales.is_contiguous(),
      "MX refinement requires contiguous Q/K, F32 weights and packed scales");
}

Args bind_mx(Tensor q, Tensor k, Tensor packed_scale, Tensor weights,
             const int* ids, int64_t id_capacity, Tensor tasks, Tensor state,
             Tensor starts, Tensor ends, Tensor error, Tensor live_nk) {
  Args a{};
  a.w = weights.const_data_ptr<float>();
  a.ids = ids;
  a.id_capacity = id_capacity;
  a.packs = state.const_data_ptr<int>();
  a.tasks = reinterpret_cast<const CandidateChunk*>(tasks.const_data_ptr<int>());
  a.cursor = state.mutable_data_ptr<int>() + 1;
  a.starts = starts.const_data_ptr<int>();
  a.ends = ends.const_data_ptr<int>();
  a.live_nk = live_nk.const_data_ptr<int>();
  a.error = error.mutable_data_ptr<int>();
  a.nk = int(k.size(0));
  a.qrows = int(q.size(0));
  a.task_cap = int(tasks.size(0));
  a.packed_scale =
      reinterpret_cast<const uint32_t*>(packed_scale.const_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaMemsetAsync(a.cursor, 0, sizeof(int), get_current_cuda_stream()));
  return a;
}
}  // namespace

void exact_mx(Tensor q, Tensor k, Tensor packed_scale, Tensor weights,
              Tensor tau, Tensor pool, Tensor tasks, Tensor state, Tensor low,
              Tensor words, Tensor table, Tensor counts, Tensor error,
              Tensor next_page, Tensor starts, Tensor ends, Tensor live_nk,
              int64_t sms, int64_t seed, bool common) {
  using namespace live_record5_completion;
  const int bq = common ? 4 : 1;
  DeviceGuard guard(q.get_device_index());
  check_operands(q, k, packed_scale, weights);
  STD_TORCH_CHECK(
      q.size(0) % bq == 0 && table.size(0) == q.size(0) &&
          table.size(1) == paged_record5::RowPages &&
          low.size(0) >= q.size(0) * paged_record5::PrefixPages &&
          low.size(1) == paged_record5::PageCells && words.sizes() == low.sizes() &&
          counts.numel() == q.size(0) && error.numel() == q.size(0),
      "MX refinement requires paged Record5 storage for complete query groups");
  STD_TORCH_CHECK(
      pool.scalar_type() == torch::headeronly::ScalarType::Byte &&
          pool.dim() == 2 && pool.size(1) == packed20::PageBytes &&
          pool.stride(0) == packed20::PageBytes && pool.stride(1) == 1,
      "packed20 true-MX refinement expects contiguous packed candidate pages");
  auto a = bind_mx(q, k, packed_scale, weights,
                   reinterpret_cast<const int*>(pool.const_data_ptr<uint8_t>()),
                   pool.size(0) * packed20::PageIds, tasks, state, starts, ends,
                   error, live_nk);
  a.tau = tau.const_data_ptr<float>();
  a.candidate_seed = int(seed);
  a.paged = {
      {low.mutable_data_ptr<uint8_t>(), words.mutable_data_ptr<int>(),
       table.mutable_data_ptr<int>(), error.mutable_data_ptr<int>(),
       next_page.mutable_data_ptr<int>(), int(low.size(0)), int(q.size(0))},
      counts.mutable_data_ptr<int>(),
      int(q.size(0))};
  auto map = certicascade_streaming_exact::fp8_map(k, 1);
  auto query_map = certicascade_streaming_exact::fp8_map(q, bq * kHeads);
  if (common)
    launch<exact<4, false, true>, 256, sizeof(MxSmem<4>)>(4 * sms, a, map, query_map);
  else
    launch<exact<1, false, true>, 256, sizeof(MxSmem<1>)>(4 * sms, a, map, query_map);
}

void score_boundary(Tensor q, Tensor k, Tensor packed_scale, Tensor weights,
                    Tensor ids, Tensor tasks, Tensor state, Tensor scores,
                    Tensor starts, Tensor ends, Tensor error, Tensor live_nk,
                    int64_t sms) {
  using namespace live_record5_completion;
  DeviceGuard guard(q.get_device_index());
  check_operands(q, k, packed_scale, weights);
  STD_TORCH_CHECK(
      ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          ids.dim() == 2 && ids.size(0) == q.size(0) && ids.size(1) == 4096 &&
          ids.stride(0) == 4096 && ids.stride(1) == 1 &&
          scores.scalar_type() == torch::headeronly::ScalarType::Float &&
          scores.dim() == 2 && scores.size(0) == q.size(0) &&
          scores.size(1) == 4096 && scores.stride(0) == 4096 &&
          scores.stride(1) == 1,
      "direct E IDs and scores require contiguous Qx4096 planes");
  STD_TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == 3 &&
                      state.numel() == 2 && starts.numel() == q.size(0) &&
                      ends.numel() == q.size(0) && error.numel() == q.size(0) &&
                      live_nk.numel() == 1,
                  "direct E scoring requires CandidateChunk tasks and complete "
                  "physical-row metadata");
  const uintptr_t ib = reinterpret_cast<uintptr_t>(ids.data_ptr());
  const uintptr_t sb = reinterpret_cast<uintptr_t>(scores.data_ptr());
  STD_TORCH_CHECK(ib + ids.numel() * sizeof(int) <= sb ||
                      sb + scores.numel() * sizeof(float) <= ib,
                  "direct E score writes must not overlap E ID reads");
  auto a = bind_mx(q, k, packed_scale, weights, ids.const_data_ptr<int>(),
                   ids.numel(), tasks, state, starts, ends, error, live_nk);
  a.scores = scores.mutable_data_ptr<float>();
  auto map = certicascade_streaming_exact::fp8_map(k, 1);
  auto query_map =
      certicascade_streaming_exact::fp8_map(q, dosa_geometry::Heads);
  launch<exact<1, true, false>, 256, sizeof(MxSmem<1>)>(4 * sms, a, map,
                                                        query_map);
}


}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("score_boundary",
              "(Tensor q, Tensor k, Tensor packed_scale, Tensor weights, "
              "Tensor ids, Tensor tasks, "
              "Tensor! state, Tensor! scores, Tensor starts, Tensor ends, "
              "Tensor! error, Tensor live_nk, int sms) -> ()",
              CUDA, score_boundary)

DOSA_REGISTER("exact_mx_packed20",
              "(Tensor q, Tensor k, Tensor packed_scale, Tensor weights, "
              "Tensor tau, Tensor pool, "
              "Tensor tasks, Tensor! state, Tensor! low, Tensor! words, "
              "Tensor! table, Tensor! counts, "
              "Tensor! error, Tensor! next_page, Tensor starts, Tensor ends, "
              "Tensor live_nk, int sms, int seed, bool common) -> ()",
              CUDA, exact_mx)
