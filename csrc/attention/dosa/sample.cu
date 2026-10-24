#include "common.cuh"
#include "score.cuh"

namespace DOSA_NAMESPACE {
// One original H32 arithmetic/TMEM implementation for both modes.
namespace prefill_stream_ballots {
using namespace certicascade_streaming_exact;
// Original sgl-deep-gemm v0.2.2 non-MX layout, with F32 scales and weights.
// Keep the unused SFQ field and alignment: they are part of the qualified
// reservation.
struct Storage {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  static constexpr int kSwizzleAlignment = 1024;
  alignas(1024) __nv_fp8_e4m3 smem_q[2][4 * kHeads * 128];
  alignas(1024) __nv_fp8_e4m3 smem_kv[2][256 * 128];
  alignas(128) float smem_sf_q[2][1];
  alignas(128) float smem_sf_kv[2][256];
  alignas(128) float smem_weights[2][4 * kHeads];
  Barrier full_q_barriers[2], empty_q_barriers[2];
  Barrier full_kv_barriers[2], empty_kv_barriers[2];
  Barrier full_tmem_barriers[2], empty_tmem_barriers[2];
  uint32_t tmem_ptr_in_smem;
};
using dosa_geometry::HeadMask;
using dosa_geometry::UnsignedHeadMask;
struct RecoveryOutput {
  const int *state, *rows, *starts, *ends, *logical_nk;
  int sms;
};
struct SeedOutput {
  float* scores;
  HeadMask* masks;
  int64_t stride;
  RecoveryOutput recovery;
};
// Keep the validated dynamic-shared reservation. Reducing it is a separate
// occupancy experiment, not part of removing uninstantiated source paths.
constexpr size_t shared_bytes() {
  return sizeof(Storage) + 8 * 4 * kHeads * (sizeof(float) + sizeof(HeadMask));
}
__device__ __forceinline__ void tmem_load32(uint32_t address,
                                            float* destination) {
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    cute::SM100_TMEM_LOAD_32dp32b32x::copy(
        address, reinterpret_cast<uint32_t*>(destination)[Is]...);
  }(cute::make_index_sequence<32>{});
}

// Shared model-head-count/BQ4 pipeline; no certificate or append mode is
// instantiated.
template <bool EmitMasks, bool Recovery = false>
CUTLASS_GLOBAL void __launch_bounds__(384, 2)
    dense_seed_kernel(int nq, int nk, int splits,
                      const __grid_constant__ cute::TmaDescriptor mq,
                      const __grid_constant__ cute::TmaDescriptor mk,
                      const __grid_constant__ cute::TmaDescriptor ms,
                      const __grid_constant__ cute::TmaDescriptor mw,
                      SeedOutput out) {
  using S = Storage;
  constexpr int H = kHeads, BQ = 4, N = BQ * H, cols = 2 * N, FW = H,
                QStages = 2, TStages = 2;
  if constexpr (Recovery) {
    // All pointers are stable owner buffers, published before this launch.
    // An ineligible/empty call exits uniformly before barriers or Q4 division.
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
  int warp = cutlass::canonical_warp_idx_sync(),
      lane = deep_gemm::ptx::get_lane_idx();
  int tid = warp * 32 + lane, works = nq / BQ * splits;
  if (warp == 8) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mk);
    cute::prefetch_tma_descriptor(&ms);
    cute::prefetch_tma_descriptor(&mw);
  }
  if (warp == 9 && cute::elect_one_sync()) {
    for (int i = 0; i < QStages; i++) {
      s.full_q_barriers[i].init(1);
      s.empty_q_barriers[i].init(288);
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
        deep_gemm::tma::copy<FW, BQ, 0>(&mw, &s.full_q_barriers[qs],
                                        s.smem_weights[qs], 0, query * BQ);
        s.full_q_barriers[qs].arrive_and_expect_tx(N * 128 + FW * BQ * 4);
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
          deep_gemm::tma::copy<256, 1, 0>(&ms, &s.full_kv_barriers[ks],
                                          s.smem_sf_kv[ks], tile * 256, 0);
          s.full_kv_barriers[ks].arrive_and_expect_tx(256 * (128 + 4));
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
  } else {
    cutlass::arch::warpgroup_reg_alloc<96>();
    tp.advance(warp / 4);
    for (int work = blockIdx.x; work < works; work += gridDim.x) {
      int query = work / splits, group = work % splits,
          tiles = (nk + 255) / 256;
      CUTE_TIE_DECL(qp.advance(), qs, qphase);
      s.full_q_barriers[qs].wait(qphase);
      asm volatile("" ::: "memory");

      const int tile_begin = tiles * group / splits,
                tile_end = tiles * (group + 1) / splits;

      for (int tile = tile_begin; tile < tile_end; tile++) {
        int key = tile * 256 + tid;

        CUTE_TIE_DECL(kp.advance(), ks, kphase);
        s.full_kv_barriers[ks].wait(kphase);
        float scale = deep_gemm::ptx::ld_shared(s.smem_sf_kv[ks] + tid);
        CUTE_TIE_DECL(tp.advance(2), ts, tphase);
        s.full_tmem_barriers[ts].wait(tphase);
        deep_gemm::ptx::tcgen05_after_thread_sync();
        s.empty_kv_barriers[ks].arrive();

        constexpr int ReadQueries = 1;
#pragma unroll
        for (int tb = 0; tb < BQ; tb += ReadQueries) {
          // The dense path reads weights after the TMEM fragment, as before.

          float packed[ReadQueries * H];
          {
            tmem_load32(tmem_base + ts * N + tb * H, packed);
            cutlass::arch::fence_view_async_tmem_load();
            if constexpr (H == 64) {
              tmem_load32(tmem_base + ts * N + tb * H + 32, packed + 32);
              cutlass::arch::fence_view_async_tmem_load();
            }
          }
          if (tb + ReadQueries == BQ) {
            deep_gemm::ptx::tcgen05_before_thread_sync();
            s.empty_tmem_barriers[ts].arrive();
          }
#pragma unroll
          for (int sub = 0; sub < ReadQueries; sub++) {
            int t = tb + sub;
            const float* accum = packed + sub * H;
            int row = query * BQ + t;
            int begin = 0, end = nk;
            if constexpr (Recovery) {
              if (row >= out.recovery.state[0]) continue;
              row = out.recovery.rows[row];
              begin = out.recovery.starts[row];
              end = out.recovery.ends[row];
            }
            float late_weights[H];
            const float* weights;
            {
#pragma unroll
              for (int h = 0; h + 3 < H; h += 4) {
                float4 v =
                    deep_gemm::ptx::ld_shared(reinterpret_cast<const float4*>(
                        s.smem_weights[qs] + t * FW + h));
                late_weights[h] = v.x;
                late_weights[h + 1] = v.y;
                late_weights[h + 2] = v.z;
                late_weights[h + 3] = v.w;
              }
#pragma unroll
              for (int h = H / 4 * 4; h < H; h++)
                late_weights[h] =
                    deep_gemm::ptx::ld_shared(s.smem_weights[qs] + t * FW + h);
              weights = late_weights;
            }
            {
              float value = native_bf16_reduce(accum, weights, scale);
              {
                if (key < nk && key < end) {
                  UnsignedHeadMask mask = 0;
                  if constexpr (EmitMasks) {
#pragma unroll
                    for (int h = 0; h < H; h++)
                      if (accum[h] > 0) mask |= UnsignedHeadMask(1) << h;
                  }
                  uint64_t dst = uint64_t(row) * out.stride + key;
                  if constexpr (Recovery)
                    out.scores[dst] = key < begin ? -INFINITY : value;
                  else
                    out.scores[dst] = value;
                  if constexpr (EmitMasks)
                    out.masks[dst] = static_cast<HeadMask>(mask);
                }
              }
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

template <bool EmitMasks, bool Recovery = false>
void launch_scores(Tensor q, Tensor k, Tensor scale, Tensor weights,
                   Tensor scores, Tensor masks, int64_t sms,
                   prefill_stream_ballots::RecoveryOutput recovery = {}) {
  using namespace certicascade_streaming_exact;
  using namespace prefill_stream_ballots;
  DeviceGuard guard(q.get_device_index());
  int nq = q.size(0), nk = k.size(0);
  // Owner supplies Q4 FP8[Q,H,128], FP8[K,128], F32 scales/weights and
  // dense score/sign-mask outputs. Both sampler and four-anchor calls retain
  // these aligned, contiguous buffers until current-stream completion.
  if (!nq || !nk) return;
  int splits = std::min((nk + 255) / 256,
                        std::max<int>(4, (2 * sms + nq / 4 - 1) / (nq / 4)));
  auto mq = make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     {128, (cuuint64_t)nq * kHeads}, {128}, {128, 4 * kHeads},
                     CU_TENSOR_MAP_SWIZZLE_128B);
  auto mk = make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     {128, (cuuint64_t)nk}, {128}, {128, 256},
                     CU_TENSOR_MAP_SWIZZLE_128B);
  auto ms = make_map(scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     {(cuuint64_t)nk, 1}, {scale_row_stride(nk)}, {256, 1},
                     CU_TENSOR_MAP_SWIZZLE_NONE);
  auto mw = make_map(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     {kHeads, (cuuint64_t)nq}, {kHeads * sizeof(float)},
                     {kHeads, 4}, CU_TENSOR_MAP_SWIZZLE_NONE);
  constexpr size_t bytes = shared_bytes();
  HeadMask* mask_data = nullptr;
  if constexpr (EmitMasks) mask_data = masks.mutable_data_ptr<HeadMask>();
  SeedOutput output{scores.mutable_data_ptr<float>(), mask_data,
                    scores.stride(0), recovery};
  DOSA_CUDA_CHECK(
      cudaFuncSetAttribute(dense_seed_kernel<EmitMasks, Recovery>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  dense_seed_kernel<EmitMasks, Recovery>
      <<<std::min<int>(2 * sms, nq / 4 * splits), 384, bytes,
         get_current_cuda_stream()>>>(nq, nk, splits, mq, mk, ms, mw, output);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

// Q is a multiple of four. Scores are contiguous F32[Q,K]; scalar stores
// need no vector alignment. All inputs/outputs live until stream completion.
void seed_score(Tensor q, Tensor k, Tensor scale, Tensor weights, Tensor scores,
                Tensor masks, int64_t sms) {
  launch_scores<true>(q, k, scale, weights, scores, masks, sms);
}
// Score-only seed output uses the same MMA and four-chain reduction.
void dense_score(Tensor q, Tensor k, Tensor scale, Tensor weights,
                 Tensor scores, int64_t sms) {
  launch_scores<false>(q, k, scale, weights, scores, Tensor(), sms);
}
void recover_score(Tensor q, Tensor k, Tensor scale, Tensor weights,
                   Tensor scores, Tensor state, Tensor rows, Tensor starts,
                   Tensor ends, Tensor logical_nk, int64_t sms) {
  // Packed Q/W and local-row map/state belong to one serialized recovery owner.
  // K descriptors retain fixed capacity; live extent controls work and stores.
  prefill_stream_ballots::RecoveryOutput recovery{
      state.mutable_data_ptr<int>(),      rows.mutable_data_ptr<int>(),
      starts.mutable_data_ptr<int>(),     ends.mutable_data_ptr<int>(),
      logical_nk.mutable_data_ptr<int>(), int(sms)};
  launch_scores<false, true>(q, k, scale, weights, scores, Tensor(), sms,
                             recovery);
}

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("seed_score") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor! scores, Tensor! masks, int sms) -> ()");
  ops.def(DOSA_OP("dense_score") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor! scores, int sms) -> ()");
  ops.def(DOSA_OP("recover_score") "(Tensor q, Tensor k, Tensor scales, Tensor weights, Tensor! scores, Tensor state, Tensor rows, "
      "Tensor starts, Tensor ends, Tensor logical_nk, int sms) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("seed_score"), TORCH_BOX(&DOSA_NAMESPACE::seed_score));
  ops.impl(DOSA_OP("dense_score"), TORCH_BOX(&DOSA_NAMESPACE::dense_score));
  ops.impl(DOSA_OP("recover_score"), TORCH_BOX(&DOSA_NAMESPACE::recover_score));
}
