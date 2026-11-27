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
constexpr size_t kSharedBytes =
    sizeof(Storage) + 8 * 4 * kHeads * (sizeof(float) + sizeof(HeadMask));

CUTLASS_GLOBAL void __launch_bounds__(384, 2)
    sign_mask_scan(int nq, int nk, int splits,
                   const __grid_constant__ cute::TmaDescriptor mq,
                   const __grid_constant__ cute::TmaDescriptor mk,
                   HeadMask* masks, int64_t mask_stride) {
  using S = Storage;
  constexpr int H = kHeads, BQ = 4, N = BQ * H, cols = 2 * N,
                QStages = 2, TStages = 2;
  static_assert(sizeof(S::smem_q[0][0]) == 1 && sizeof(S::smem_kv[0][0]) == 1);
  extern __shared__ __align__(S::kSwizzleAlignment) uint8_t buf[];
  auto& s = *reinterpret_cast<S*>(buf);
  int warp = cutlass::canonical_warp_idx_sync(),
      lane = deep_gemm::ptx::get_lane_idx();
  int tid = warp * 32 + lane, works = nq / BQ * splits;
  if (warp == 8) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mk);
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
        s.full_q_barriers[qs].arrive_and_expect_tx(N * 128);
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
          s.full_kv_barriers[ks].arrive_and_expect_tx(256 * 128);
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
  } else if (warp < 8) {
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
          const int row = query * BQ + t;
          UnsignedHeadMask mask = 0;
#pragma unroll
          for (int h = 0; h < H; ++h)
            if (accum[h] > 0) mask |= UnsignedHeadMask(1) << h;
          if (key < nk)
            masks[uint64_t(row) * mask_stride + key] =
                static_cast<HeadMask>(mask);
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

void sign_masks(Tensor q, Tensor k, Tensor masks, int64_t sms) {
  using namespace certicascade_streaming_exact;
  using namespace prefill_stream_ballots;
  DeviceGuard guard(q.get_device_index());
  int nq = q.size(0), nk = k.size(0);
  if (!nq || !nk) return;
  int splits = std::min((nk + 255) / 256,
                        std::max<int>(4, (2 * sms + nq / 4 - 1) / (nq / 4)));
  auto mq = fp8_map(q, 4 * kHeads);
  auto mk = fp8_map(k, 256);
  launch<sign_mask_scan, 384, kSharedBytes>(
      std::min<int>(2 * sms, nq / 4 * splits), nq, nk, splits, mq, mk,
      masks.mutable_data_ptr<HeadMask>(), masks.stride(0));
}
}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("sign_masks",
              "(Tensor q, Tensor k, Tensor! masks, int sms) -> ()", CUDA,
              sign_masks)
