#include "common.cuh"
#include "../../fused_mqa_topk/deep_gemm_compat.cuh"

namespace DOSA_NAMESPACE {
namespace prefill_stream_ballots {
using namespace certicascade_streaming_exact;
using Storage =
    fused_mqa_topk::dg::MQALogitsSharedStorage<kHeads, 128, 4, 256, 2, 2, 2,
                                               __nv_fp8_e4m3, float>;
using dosa_geometry::HeadMask;
using dosa_geometry::UnsignedHeadMask;
struct PackedWeightSlots {
  __nv_bfloat162 pairs[2][4 * kHeads / 2];
  Storage::Barrier ready[2];
  int unit_safe[2];
};
// Keep the qualified reservation and use part of its tail for the score path.
constexpr size_t kSharedBytes =
    sizeof(Storage) + 8 * 4 * kHeads * (sizeof(float) + sizeof(HeadMask));
static_assert(sizeof(PackedWeightSlots) <= kSharedBytes - sizeof(Storage));

struct RecoveryOutput {
  const int *state, *rows, *starts, *ends, *logical_nk;
  int sms;
};
template <class Score>
struct SeedOutput {
  Score* scores;
  HeadMask* masks;
  int64_t stride;
  RecoveryOutput recovery;
};
template <bool EmitMasks, bool Recovery = false, class Score = float>
CUTLASS_GLOBAL void __launch_bounds__(384, 2)
    dense_seed_kernel(int nq, int nk, int splits,
                      const __grid_constant__ cute::TmaDescriptor mq,
                      const __grid_constant__ cute::TmaDescriptor mk,
                      const __grid_constant__ cute::TmaDescriptor ms,
                      const __grid_constant__ cute::TmaDescriptor mw,
                      SeedOutput<Score> out) {
  using S = Storage;
  constexpr bool Seed =
      !EmitMasks && !Recovery && std::is_same_v<Score, __nv_bfloat16>;
  constexpr int H = kHeads, BQ = 4, N = BQ * H, cols = 2 * N, FW = H,
                QStages = 2, TStages = 2;
  if constexpr (Recovery) {
    if (!out.recovery.state[1]) return;
    nq = (out.recovery.state[0] + BQ - 1) / BQ * BQ;
    nk = *out.recovery.logical_nk;
    splits = min((nk + 255) / 256,
                 max(4, (2 * out.recovery.sms + nq / BQ - 1) / (nq / BQ)));
    if (blockIdx.x >= nq / BQ * splits) return;
  }
  static_assert(std::extent_v<decltype(S::smem_weights), 1> >= FW * BQ);
  static_assert(sizeof(S::smem_q[0][0]) == 1 && sizeof(S::smem_kv[0][0]) == 1);
  static_assert(sizeof(S::smem_sf_kv[0][0]) == 4 &&
                sizeof(S::smem_weights[0][0]) == 4);
  extern __shared__ __align__(S::kSwizzleAlignment) uint8_t buf[];
  auto& s = *reinterpret_cast<S*>(buf);
  auto& packed = *reinterpret_cast<PackedWeightSlots*>(buf + sizeof(S));
  int warp = cutlass::canonical_warp_idx_sync(),
      lane = deep_gemm::ptx::get_lane_idx();
  int tid = warp * 32 + lane, works = nq / BQ * splits;
  if (warp == 8) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mk);
    if constexpr (!EmitMasks) {
      cute::prefetch_tma_descriptor(&ms);
      cute::prefetch_tma_descriptor(&mw);
    }
  }
  if (warp == 9 && cute::elect_one_sync()) {
    for (int i = 0; i < QStages; i++) {
      s.full_q_barriers[i].init(1);
      // The math and MMA readers are joined by the score-path packer.
      s.empty_q_barriers[i].init(288 + (EmitMasks ? 0 : 32));
      if constexpr (!EmitMasks) packed.ready[i].init(32);
    }
    for (int i = 0; i < TStages; i++) {
      s.full_tmem_barriers[i].init(1);
      s.empty_tmem_barriers[i].init(128);
    }
    for (int i = 0; i < 2; i++) {
      s.full_kv_barriers[i].init(1);
      s.empty_kv_barriers[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();
  if (warp == 10) {
    cute::TMEM::Allocator1Sm().allocate(cols, &s.tmem_ptr_in_smem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();
  const uint32_t tmem_base = deep_gemm::ptx::ld_shared(&s.tmem_ptr_in_smem);
  deep_gemm::RingPipeline<QStages> qp;
  deep_gemm::RingPipeline<TStages> tp;
  deep_gemm::RingPipeline<2> kp;
  if (warp >= 8) cutlass::arch::warpgroup_reg_dealloc<48>();
  if (warp == 8) {
    if (cute::elect_one_sync()) {
      for (int work = blockIdx.x; work < works; work += gridDim.x) {
        int query = work / splits;
        CUTE_TIE_DECL(qp.advance(), qs, phase);
        s.empty_q_barriers[qs].wait(phase ^ 1);
        asm volatile("" ::
                         : "memory");  // compiler acquire: the previous
                                       // row-slot readers are done

        asm volatile("" ::: "memory");
        deep_gemm::tma::copy<128, N, 0>(&mq, &s.full_q_barriers[qs],
                                        s.smem_q[qs], 0, query * N);
        if constexpr (!EmitMasks)
          deep_gemm::tma::copy<FW, BQ, 0>(&mw, &s.full_q_barriers[qs],
                                          s.smem_weights[qs], 0, query * BQ);
        s.full_q_barriers[qs].arrive_and_expect_tx(
            N * 128 + (EmitMasks ? 0 : FW * BQ * 4));
      }
    }
    __syncwarp();
  } else if (warp == 9) {
    if (cute::elect_one_sync())
      for (int work = blockIdx.x; work < works; work += gridDim.x) {
        int group = work % splits, tiles = (nk + 255) / 256;
        for (int tile = tiles * group / splits;
             tile < tiles * (group + 1) / splits; tile++) {
          CUTE_TIE_DECL(kp.advance(), ks, phase);
          s.empty_kv_barriers[ks].wait(phase ^ 1);
          deep_gemm::tma::copy<128, 256, 0>(&mk, &s.full_kv_barriers[ks],
                                            s.smem_kv[ks], 0, tile * 256);
          if constexpr (!EmitMasks)
            deep_gemm::tma::copy<256, 1, 0>(&ms, &s.full_kv_barriers[ks],
                                            s.smem_sf_kv[ks], tile * 256, 0);
          s.full_kv_barriers[ks].arrive_and_expect_tx(
              256 * (128 + (EmitMasks ? 0 : 4)));
        }
      }
    __syncwarp();
  } else if (warp == 10) {
    bool lead = cute::elect_one_sync();
    for (int work = blockIdx.x; work < works; work += gridDim.x) {
      int group = work % splits, tiles = (nk + 255) / 256;
      CUTE_TIE_DECL(qp.advance(), qs, qphase);
      s.full_q_barriers[qs].wait(qphase);
      asm volatile("" ::: "memory");
      for (int tile = tiles * group / splits;
           tile < tiles * (group + 1) / splits; tile++) {
        CUTE_TIE_DECL(kp.advance(), ks, kphase);
        if (lead) {
          // Only the issuing lane waits: non-issuing lanes are not counted in
          // empty_KV and could otherwise miss a phase when TMA refills this
          // slot.
          s.full_kv_barriers[ks].wait(kphase);
#pragma unroll
          for (int g = 0; g < 2; g++) {
            CUTE_TIE_DECL(tp.advance(), ts, tphase);
            s.empty_tmem_barriers[ts].wait(tphase ^ 1);
            deep_gemm::ptx::tcgen05_after_thread_sync();
            issue_h32_mma<128, N>(
                s.smem_kv[ks], s.smem_q[qs], g * 128, tmem_base + ts * N,
                cute::cast_smem_ptr_to_uint(&s.full_tmem_barriers[ts]));
          }
        }
        __syncwarp();
      }
      asm volatile("" ::
                       : "memory");  // compiler release: all row-slot reads
                                     // stay before reuse
      s.empty_q_barriers[qs].arrive();
    }
  } else if (warp == 11) {
    if constexpr (!EmitMasks) {
      for (int work = blockIdx.x; work < works; work += gridDim.x) {
        CUTE_TIE_DECL(qp.advance(), qs, qphase);
        s.full_q_barriers[qs].wait(qphase);
        asm volatile("" ::: "memory");
        bool safe = true;
        for (int i = lane; i < N / 2; i += 32) {
          const float2 pair = deep_gemm::ptx::ld_shared(
              reinterpret_cast<const float2*>(s.smem_weights[qs] + 2 * i));
          if constexpr (Seed)
            safe &= bounded_weight(pair.x) && bounded_weight(pair.y);
          packed.pairs[qs][i] = __floats2bfloat162_rn(pair.x, pair.y);
        }
        if constexpr (Seed) {
          const int all_safe = __all_sync(0xffffffffu, safe);
          if (lane == 0) packed.unit_safe[qs] = all_safe;
        }
        asm volatile("" ::: "memory");
        packed.ready[qs].arrive();
        s.empty_q_barriers[qs].arrive();
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<96>();
    tp.advance(warp / 4);
    for (int work = blockIdx.x; work < works; work += gridDim.x) {
      int query = work / splits, group = work % splits,
          tiles = (nk + 255) / 256;
      CUTE_TIE_DECL(qp.advance(), qs, qphase);
      if constexpr (EmitMasks)
        s.full_q_barriers[qs].wait(qphase);
      else
        packed.ready[qs].wait(qphase);
      asm volatile("" ::: "memory");

      bool bounded = false;
      if constexpr (Seed) bounded = packed.unit_safe[qs] != 0;
      const int tile_begin = tiles * group / splits,
                tile_end = tiles * (group + 1) / splits;

      for (int tile = tile_begin; tile < tile_end; tile++) {
        int key = tile * 256 + tid;

        CUTE_TIE_DECL(kp.advance(), ks, kphase);
        s.full_kv_barriers[ks].wait(kphase);
        float scale = 0;
        if constexpr (!EmitMasks)
          scale = deep_gemm::ptx::ld_shared(s.smem_sf_kv[ks] + tid);
        bool rescale = false;
        if constexpr (Seed)
          rescale = bounded &&
                    __all_sync(0xffffffffu, key >= nk || bounded_power(scale));
        CUTE_TIE_DECL(tp.advance(2), ts, tphase);
        s.full_tmem_barriers[ts].wait(tphase);
        deep_gemm::ptx::tcgen05_after_thread_sync();
        s.empty_kv_barriers[ks].arrive();

#pragma unroll
        for (int t = 0; t < BQ; ++t) {
          float accum[H];
          load_tmem<32>(tmem_base + ts * N + t * H, accum);
          cutlass::arch::fence_view_async_tmem_load();
          if constexpr (H == 64) {
            load_tmem<32>(tmem_base + ts * N + t * H + 32, accum + 32);
            cutlass::arch::fence_view_async_tmem_load();
          }
          if (t == BQ - 1) {
            deep_gemm::ptx::tcgen05_before_thread_sync();
            s.empty_tmem_barriers[ts].arrive();
          }
          int row = query * BQ + t, begin = 0, end = nk;
          if constexpr (Recovery) {
            if (row >= out.recovery.state[0]) continue;
            row = out.recovery.rows[row];
            begin = out.recovery.starts[row];
            end = out.recovery.ends[row];
          }
          if constexpr (EmitMasks) {
            UnsignedHeadMask mask = 0;
#pragma unroll
            for (int h = 0; h < H; ++h)
              if (accum[h] > 0) mask |= UnsignedHeadMask(1) << h;
            if (key < nk)
              out.masks[uint64_t(row) * out.stride + key] =
                  static_cast<HeadMask>(mask);
          } else {
            const float value = native_bf16_reduce(
                accum, packed.pairs[qs] + t * (H / 2), scale, rescale);
            if (key < nk && key < end) {
              const auto offset = uint64_t(row) * out.stride + key;
              if constexpr (std::is_same_v<Score, __nv_bfloat16>)
                out.scores[offset] = __ushort_as_bfloat16(
                    uint16_t(__float_as_uint(value) >> 16));
              else
                out.scores[offset] =
                    Recovery && key < begin ? -INFINITY : value;
            }
          }
        }
      }

      __syncwarp();
      asm volatile("" ::
                       : "memory");  // compiler release: all row-slot reads
                                     // stay before reuse
      s.empty_q_barriers[qs].arrive();
    }
    named_sync(256, 0);
    if (warp == 0) cute::TMEM::Allocator1Sm().free(tmem_base, cols);
  }
}

}  // namespace prefill_stream_ballots

template <bool EmitMasks, bool Recovery = false, class Score = float>
void launch_scores(Tensor q, Tensor k, Tensor scale, Tensor weights,
                   Tensor scores, Tensor masks, int64_t sms,
                   prefill_stream_ballots::RecoveryOutput recovery = {}) {
  using namespace certicascade_streaming_exact;
  using namespace prefill_stream_ballots;
  DeviceGuard guard(q.get_device_index());
  int nq = q.size(0), nk = k.size(0);
  // Borrow contiguous FP8 D128 Q4/K rows and F32 scale/weight tensors.
  if (!nq || !nk) return;
  int splits = std::min((nk + 255) / 256,
                        std::max<int>(4, (2 * sms + nq / 4 - 1) / (nq / 4)));
  auto mq = fp8_map(q, 4 * kHeads);
  auto mk = fp8_map(k, 256);
  cute::TmaDescriptor ms{}, mw{};
  if constexpr (!EmitMasks) {
    ms = make_map(scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                  {(cuuint64_t)nk, 1}, {scale_row_stride(nk)}, {256, 1},
                  CU_TENSOR_MAP_SWIZZLE_NONE);
    mw = make_map(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                  {kHeads, (cuuint64_t)nq}, {kHeads * sizeof(float)},
                  {kHeads, 4}, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  SeedOutput<Score> output{nullptr, nullptr, 0, recovery};
  if constexpr (EmitMasks) {
    output.masks = masks.mutable_data_ptr<HeadMask>();
    output.stride = masks.stride(0);
  } else {
    if constexpr (std::is_same_v<Score, float>)
      output.scores = scores.mutable_data_ptr<float>();
    else
      output.scores = static_cast<Score*>(scores.data_ptr());
    output.stride = scores.stride(0);
  }
  launch<dense_seed_kernel<EmitMasks, Recovery, Score>, 384, kSharedBytes>(
      std::min<int>(2 * sms, nq / 4 * splits), nq, nk, splits, mq, mk, ms, mw,
      output);
}

void sign_masks(Tensor q, Tensor k, Tensor masks, int64_t sms) {
  launch_scores<true>(q, k, Tensor(), Tensor(), Tensor(), masks, sms);
}
void dense_score(Tensor q, Tensor k, Tensor scale, Tensor weights,
                 Tensor scores, int64_t sms) {
  if (scores.scalar_type() == torch::headeronly::ScalarType::BFloat16)
    launch_scores<false, false, __nv_bfloat16>(q, k, scale, weights, scores,
                                               Tensor(), sms);
  else
    launch_scores<false>(q, k, scale, weights, scores, Tensor(), sms);
}
void recover_score(Tensor q, Tensor k, Tensor scale, Tensor weights,
                   Tensor scores, Tensor state, Tensor rows, Tensor starts,
                   Tensor ends, Tensor logical_nk, int64_t sms) {
  prefill_stream_ballots::RecoveryOutput recovery{
      state.mutable_data_ptr<int>(),      rows.mutable_data_ptr<int>(),
      starts.mutable_data_ptr<int>(),     ends.mutable_data_ptr<int>(),
      logical_nk.mutable_data_ptr<int>(), int(sms)};
  launch_scores<false, true>(q, k, scale, weights, scores, Tensor(), sms,
                             recovery);
}

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("sign_masks",
              "(Tensor q, Tensor k, Tensor! masks, int sms) -> ()", CUDA,
              sign_masks)
DOSA_REGISTER("dense_score",
              "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor! "
              "scores, int sms) -> ()",
              CUDA, dense_score)
DOSA_REGISTER("recover_score",
              "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor! "
              "scores, Tensor state, Tensor rows, "
              "Tensor starts, Tensor ends, Tensor logical_nk, int sms) -> ()",
              CUDA, recover_score)
