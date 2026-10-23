#include "common.cuh"
#include "emit.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
namespace dosa_screen {
struct IndexedFamilyWork {
  bool admitted;
  int group, logical_nk, active_word_count, first_block, blocks;
};

// Called by every CTA thread before either indexed family's math pipeline.
// The family supplies its budget predicate; all metadata guards stay here.
template <int Paid, int Tile, class ValidBudget>
__device__ __forceinline__ IndexedFamilyWork admit_indexed_family(
    const Args& a, const int32_t* live_nk, ValidBudget valid_budget) {
  int t = threadIdx.x, logical_nk = __ldg(live_nk);
  if (logical_nk < a.seed || logical_nk > a.nk) {
    if (blockIdx.x == 0)
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return {};
  }
  int active_word_count = (logical_nk - a.seed + 31) / 32;
  int total_blocks = (logical_nk - a.seed + Tile - 1) / Tile;
  int count = __ldg(a.counts), parts = __ldg(a.counts + 1), groups = a.nq / 8;
  bool malformed = count < 0 || count > groups || parts < 1 ||
                   parts > max(total_blocks, 1) ||
                   int64_t(count) * parts > gridDim.x;
  if (malformed) {
    if (blockIdx.x == 0)
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return {};
  }
  if (blockIdx.x == 0) {
    bool bad = false;
    for (int qg = t; qg < groups; qg += blockDim.x) {
      int b0 = __ldg(a.budget + qg * 2), b1 = __ldg(a.budget + qg * 2 + 1),
          pos = __ldg(a.members + qg);
      if (!valid_budget(b0))
        for (int j = 0; j < 4; ++j) atomicOr(a.error + qg * 8 + j, 1);
      if (!valid_budget(b1))
        for (int j = 4; j < 8; ++j) atomicOr(a.error + qg * 8 + j, 1);
      bool active = b0 == Paid || b1 == Paid;
      if (active) {
        if (pos < 0 || pos >= count)
          bad = true;
        else if (__ldg(a.ids + pos) != qg)
          bad = true;
      } else if (pos != -1)
        bad = true;
    }
    if (__syncthreads_or(bad))
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
  }
  int work = blockIdx.x;
  if (work >= count * parts) return {};
  int pos = work / parts, part = work % parts;
  int g = __ldg(a.ids + pos);
  if (g < 0 || g >= groups) {
    for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return {};
  }
  if (__ldg(a.members + g) != pos) {
    for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return {};
  }
  int first_block = total_blocks * part / parts;
  int blocks = total_blocks * (part + 1) / parts - first_block;
  bool eligible = false;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int b = a.budget[g * 2 + j];
    if (!valid_budget(b)) {
      if (t < 4) atomicOr(a.error + (g * 2 + j) * 4 + t, 1);
    }
    eligible |= b == Paid;
  }
  if (!eligible) return {};
  if (total_blocks == 0) return {};
  return {.admitted = true,
          .group = g,
          .logical_nk = logical_nk,
          .active_word_count = active_word_count,
          .first_block = first_block,
          .blocks = blocks};
}
}  // namespace dosa_screen

namespace f1_abs_qfma {
using namespace dosa_screen;
}
namespace f1_abs_qfma {
// All32 lanes reach both query ballots, including inactive/invisible keys.
// The certified keep predicate is consumed directly by the vote.
__device__ __forceinline__ uint32_t qualified_keep_ballot(float value,
                                                          float tau,
                                                          uint32_t eligible,
                                                          uint32_t qualified) {
  uint32_t ballot;
  asm volatile(
      "{.reg .pred finite,below,active,valid,keep; .reg .f32 magnitude; "
      "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
      "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; "
      "setp.ne.u32 valid,%4,0; and.pred keep,keep,valid; not.pred keep,keep; "
      "setp.ne.u32 active,%3,0; and.pred keep,keep,active; "
      "vote.ballot.sync.b32 %0,keep,0xffffffff;}"
      : "=r"(ballot)
      : "f"(value), "f"(tau), "r"(eligible), "r"(qualified));
  return ballot;
}
// Requires freshly prepared abs-shift-dd2-v1 fields; original H16 geometry.
// Four math WGs own2queries each. OriginalK256/raw4/twoUMA256;
// creditafterkh1read32.
struct alignas(1024) K256FourWGStorage {
  alignas(1024) uint8_t fine[8 * 16 * 128];
  alignas(16) float fw[8][16];
  alignas(1024) uint8_t raw_k[4][256 * 128];
  float raw_scale[4][256];
  uint32_t raw_rho[4][256];
  Param row[8];
  Barrier full_q, full_kv[4], empty_kv[4], full_umma[2], empty_umma[2];
  uint32_t tmem;
};
__device__ __forceinline__ void h16_load_tmem32(uint32_t address, float* out) {
  [&]<size_t... I>(cute::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_32dp32b32x::copy(
        address, reinterpret_cast<uint32_t*>(out)[I]...);
  }(cute::make_index_sequence<32>{});
}
// Same represented ABS/DD core, reduced in four RN chains. The retained
// direct planner SCREEN (80-round envelope) covers these six-round chains.
// Mixed weights are prepared once outside the K loop; no new query fields.
__device__ __forceinline__ void h16_abs_core(const float* acc,
                                             const float weights[2][16],
                                             const Param resident[2],
                                             float core[2],
                                             uint32_t& neg_valid) {
  float2 s0[2] = {make_float2(0, 0), make_float2(0, 0)},
         s1[2] = {make_float2(0, 0), make_float2(0, 0)};
#pragma unroll
  for (int h = 0; h < 16; h += 4) {
#pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
      float2 a0 = h == 0 ? make_float2(acc[qr * 16], acc[qr * 16 + 1])
                         : make_float2(fabsf(acc[qr * 16 + h]),
                                       fabsf(acc[qr * 16 + h + 1]));
      float2 a1 =
          make_float2(fabsf(acc[qr * 16 + h + 2]), fabsf(acc[qr * 16 + h + 3]));
      s0[qr] = __ffma2_rn(a0, make_float2(weights[qr][h], weights[qr][h + 1]),
                          s0[qr]);
      s1[qr] = __ffma2_rn(
          a1, make_float2(weights[qr][h + 2], weights[qr][h + 3]), s1[qr]);
    }
  }
#pragma unroll
  for (int qr = 0; qr < 2; ++qr) {
    float2 pair = __fadd2_rn(s0[qr], s1[qr]);
    core[qr] = __fadd_rn(pair.x, pair.y);
  }
  // Valid field scales and finite E4 raw dots imply finite DD. Bad/nonfinite
  // arithmetic still fails the unchanged final finite-value keep predicate.
  neg_valid = 3u;
}
__global__ __launch_bounds__(640, 1) void k256_four_math_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ PagedArgs page, const int32_t* live_nk) {
  constexpr int H = 16;
  int t = threadIdx.x, warp = t / 32, lane = t & 31;
  auto work =
      admit_indexed_family<14, 256>(a, live_nk, [] __device__(int budget) {
        return budget == 12 || dosa_geometry::valid_certificate_budget(budget);
      });
  if (!work.admitted) return;
  int g = work.group, active_word_count = work.active_word_count,
      first_block = work.first_block, blocks = work.blocks;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<K256FourWGStorage*>(memory);
  if (t < 8) {
    int r = g * 8 + t;
    Param p;
    p.guard = a.meta[r * 64 + 16];
    p.lo0 = a.fine_scale[r * 16];
    p.lo1 = a.fine_scale[r * 16 + 1];
    p.raw = a.raw[r];
    p.out = a.out[r];
    p.tau = a.tau[r];
    p.start = a.starts[r];
    p.end = a.ends[r];
    p.hint = a.hint[r];
    s.row[t] = p;
  }
  if (warp == 16) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mw);
    cute::prefetch_tma_descriptor(&mk);
    cute::prefetch_tma_descriptor(&ms);
    cute::prefetch_tma_descriptor(&mr);
  }
  if (warp == 16 && cute::elect_one_sync()) {
    s.full_q.init(1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s.full_kv[i].init(1);
      s.empty_kv[i].init(512);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 17) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        s.full_umma[i].init(1);
        s.empty_umma[i].init(512);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(512, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp >= 16) cutlass::arch::warpgroup_reg_dealloc<32>();
  if (warp == 16) {
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, 128, 128>(&mq, &s.full_q, s.fine, 0, g * 128);
      deep_gemm::tma::copy<16, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * 16 * 128 + 8 * 16 * 4);
      for (int block = 0; block < blocks; ++block) {
        int stage = block % 4, phase = (block / 4) & 1;
        s.empty_kv[stage].wait(phase ^ 1);
        deep_gemm::tma::copy<128, 256, 128>(
            &mk, &s.full_kv[stage], s.raw_k[stage], 0,
            a.seed + (first_block + block) * 256);
        deep_gemm::tma::copy<256, 1, 0>(
            &ms, &s.full_kv[stage], s.raw_scale[stage],
            a.seed + (first_block + block) * 256, 0);
        deep_gemm::tma::copy<256, 1, 0>(
            &mr, &s.full_kv[stage], s.raw_rho[stage],
            a.seed + (first_block + block) * 256, 0);
        s.full_kv[stage].arrive_and_expect_tx(256 * 128 + 2 * 256 * 4);
      }
    }
  } else if (warp == 17) {
    auto instruction = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, 128,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
    s.full_q.wait(0);
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 4, phase = (block / 4) & 1, u = block % 2,
          up = (block / 2) & 1;
      s.full_kv[stage].wait(phase);
      s.empty_umma[u].wait(up ^ 1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                          0, 128, 128>(
              reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),
              kh * 128, k * 32);
          auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                          0, 128, 128>(
              reinterpret_cast<cutlass::float_e4m3_t*>(s.fine), 0, k * 32);
          cute::SM100_MMA_F8F6F4_SS::fma(ad, bd, s.tmem + u * 256 + kh * 128, k,
                                         runtime);
        }
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[u]));
    }
  } else if (warp >= 18) {
  } else {
    cutlass::arch::warpgroup_reg_alloc<112>();
    // Same-epoch NK is immutable. Reload after register donation so it does not
    // spill across the startup / specialization boundary or inside the K loop.
    int math_nk = a.nk;
    asm volatile("ld.global.nc.u32 %0,[%1];"
                 : "=r"(math_nk)
                 : "l"(live_nk)
                 : "memory");
    const int math_word_count = (math_nk - a.seed + 31) / 32;
    int wg = t / 128, kt = t % 128, group = g * 2 + (wg / 2),
        channel = (wg & 1) * 2;
    float weights[2][16];
    Param resident[2];
    bool eligible_q[2];
    s.full_q.wait(0);
#pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
      resident[qr] = s.row[wg * 2 + qr];
      eligible_q[qr] = a.budget[group] == 14;
#pragma unroll
      for (int h = 0; h < 16; ++h)
        weights[qr][h] =
            h == 0 ? -resident[qr].lo0
                   : (h == 1 ? -resident[qr].lo1
                             : __fmul_rn(0.5f, deep_gemm::ptx::ld_shared(
                                                   &s.fw[wg * 2 + qr][h])));
    }
    // Original Q4/channel address, advanced eight physical words per K256.
    // Non-writing lanes form integers only; the original lane/bounds gate
    // dominates every conversion back to a dereferenced pointer.
    uint64_t word_address = reinterpret_cast<uint64_t>(page.words) +
                            (uint64_t(group) * uint64_t(page.word_count) +
                             uint64_t(first_block) * 8u + uint64_t(warp & 3)) *
                                16u +
                            uint64_t(channel + lane) * 4u;
    // Rebuild the immutable loop extent after register redistribution, like
    // math_nk above. Do not carry the producer's blocks value across
    // setmaxnreg.
    int math_blocks = (math_nk - a.seed + 255) / 256;
    {
      int math_parts = __ldg(a.counts + 1), math_part = blockIdx.x % math_parts;
      math_blocks = math_blocks * (math_part + 1) / math_parts -
                    math_blocks * math_part / math_parts;
    }
#pragma unroll 2
    for (int block = 0; block < math_blocks; ++block) {
      int stage = block % 4, phase = (block / 4) & 1, u = block % 2,
          up = (block / 2) & 1;
      s.full_kv[stage].wait(phase);
      float scales[2] = {
          deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]),
          deep_gemm::ptx::ld_shared(&s.raw_scale[stage][128 + kt])};
      uint32_t codes[2] = {
          deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]),
          deep_gemm::ptx::ld_shared(&s.raw_rho[stage][128 + kt])};
      s.full_umma[u].wait(up);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      s.empty_kv[stage].arrive();
      float first_core[2];
      uint32_t first_neg_valid;
      {
        float first_acc[32];
        h16_load_tmem32(s.tmem + u * 256 + wg * 32, first_acc);
        cutlass::arch::fence_view_async_tmem_load();
        h16_abs_core(first_acc, weights, resident, first_core, first_neg_valid);
      }
      // Only two scores plus the original finite-DD bits remain from kh0.
      // Both TMEM halves are retired before granting the original 512 credits.
      float second_acc[32];
      h16_load_tmem32(s.tmem + u * 256 + 128 + wg * 32, second_acc);
      cutlass::arch::fence_view_async_tmem_load();
      deep_gemm::ptx::tcgen05_before_thread_sync();
      s.empty_umma[u].arrive();
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        float scale = scales[kh];
        uint32_t code = codes[kh];
        float core[2];
        uint32_t neg_valid;
        if (kh == 0) {
          core[0] = first_core[0];
          core[1] = first_core[1];
          neg_valid = first_neg_valid;
        } else
          h16_abs_core(second_acc, weights, resident, core, neg_valid);
        __half_raw half = __nv_cvt_fp8_to_halfraw(
            static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
        float rho = __half2float(static_cast<__half>(half));
        int key = a.seed + (first_block + block) * 256 + kh * 128 + kt;
        bool keyvalid = key < math_nk && code < 127u;
        uint32_t ballots[2];
#pragma unroll
        for (int qr = 0; qr < 2; ++qr) {
          const Param& p = resident[qr];
          float signed_score = core[qr];
          bool finite_neg = (neg_valid >> qr) & 1u;
          float tail = __fmaf_ru(rho, p.guard, __fadd_ru(signed_score, p.raw));
          float value = __fmaf_ru(scale, tail, p.out);
          bool visible = key < math_nk && key >= p.start && key < p.end;
          uint32_t qualified = uint32_t(keyvalid && p.hint == 2 && finite_neg);
          ballots[qr] = qualified_keep_ballot(
              value, p.tau, uint32_t(eligible_q[qr]) & uint32_t(visible),
              qualified);
        }
        int word = (first_block + block) * 8 + kh * 4 + (warp & 3);
        unsigned value = lane == 0 ? ballots[0] : ballots[1];
        // Two WGs share a Q4 word: this WG owns only its two channels.
        if (lane < 2 && eligible_q[0] && word < math_word_count)
          *reinterpret_cast<uint32_t*>(word_address + uint64_t(kh) * 64u) =
              value;
      }
      word_address += 128u;
    }
  }
  __syncthreads();
  if (warp == 17) cute::TMEM::Allocator1Sm().free(s.tmem, 512);
}

}  // namespace f1_abs_qfma
void f1_abs_qfma_trusted_live_k(int64_t paid, std::vector<Tensor> x,
                                std::vector<Tensor> streams, Tensor pool,
                                Tensor table, Tensor head, Tensor status,
                                Tensor error, int64_t seed, Tensor words,
                                Tensor work_ids, Tensor work_inverse,
                                Tensor work_config, int64_t launch_grid,
                                Tensor live_nk) {
  // Private paid14 screen: owned x15/stream6, I32[Q/8] ids/inverse,
  // I32[2] schedule and immutable device I32[1] liveNK. Q is padded to8.
  // The serial owner retains these views and the page/word buffers for the
  // stream.
  using namespace f1_abs_qfma;
  auto [a, pg] = dosa_emit_host::bind<Args, PagedArgs>(
      paid, x, streams, pool, table, head, status, error, seed, words);
  int nq = a.nq, nk = a.nk;
  DeviceGuard guard(x[0].get_device_index());
  auto stream = get_current_cuda_stream();
  a.ids = work_ids.mutable_data_ptr<int>();
  a.members = work_inverse.mutable_data_ptr<int>();
  a.counts = work_config.mutable_data_ptr<int>();
  auto mq = make_map(x[5].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     {128, (cuuint64_t)nq * 16}, {128}, {128, 128},
                     CU_TENSOR_MAP_SWIZZLE_128B);
  auto mw = make_map(x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     {64, (cuuint64_t)nq}, {256}, {16, 8},
                     CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, 256);
  int grid = static_cast<int>(launch_grid);
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      k256_four_math_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(K256FourWGStorage)));
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      k256_four_math_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  k256_four_math_kernel<<<grid, 640, sizeof(K256FourWGStorage), stream>>>(
      a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
      live_nk.mutable_data_ptr<int32_t>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}

namespace f2_live_h18 {
using namespace dosa_screen;
}

namespace f2_live_h18 {
// Exact original Trusted keep predicate consumed directly by the warp vote.
__device__ __forceinline__ uint32_t h18_keep_ballot(float value, float tau,
                                                    uint32_t eligible) {
  uint32_t mask;
  asm volatile(
      "{.reg .pred finite,below,active,keep; .reg .f32 magnitude; "
      "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
      "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; not.pred "
      "keep,keep; "
      "setp.ne.u32 active,%3,0; and.pred keep,keep,active; "
      "vote.ballot.sync.b32 %0,keep,0xffffffff;}"
      : "=r"(mask)
      : "f"(value), "f"(tau), "r"(eligible));
  return mask;
}
// Two math WGs split eight queries across the SAME 128-key MMA tile.
// The full 64-column fragment is read before either WG starts scalar reduction.
struct alignas(1024) F2LiveH18Storage {
  static constexpr int H = 18;
  static constexpr int W = (H + 3) / 4 * 4;
  alignas(1024) uint8_t fine[8 * H * 128];
  alignas(16) float fw[8][W];
  alignas(1024) uint8_t raw_k[8][128 * 128];
  float raw_scale[8][128];
  uint32_t raw_rho[8][128];
  Param row[8];
  Barrier full_q, full_kv[8], empty_kv[8], full_umma[4], empty_umma[4];
  uint32_t tmem;
};
__global__ __launch_bounds__(384, 1) void f2_live_h18_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ PagedArgs page, const int32_t* live_nk) {
  constexpr int H = 18, N = 8 * H, W = (H + 3) / 4 * 4, Stages = 3;
  int t = threadIdx.x, warp = t / 32, lane = t & 31;
  auto work =
      admit_indexed_family<16, 128>(a, live_nk, [] __device__(int budget) {
        return dosa_geometry::valid_certificate_budget(budget);
      });
  if (!work.admitted) return;
  int g = work.group, logical_nk = work.logical_nk,
      active_word_count = work.active_word_count,
      first_block = work.first_block, blocks = work.blocks;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<F2LiveH18Storage*>(memory);
  if (t < 8) {
    int r = g * 8 + t;
    Param p;
    p.guard = a.meta[r * 64 + H];
    p.lo0 = a.fine_scale[r * H];
    p.lo1 = a.fine_scale[r * H + 1];
    p.raw = a.raw[r];
    p.out = a.out[r];
    p.tau = a.hint[r] == 2 ? a.tau[r] : __uint_as_float(0xff800000u);
    p.start = a.starts[r];
    p.end = a.ends[r];
    p.hint = 2;
    s.row[t] = p;
  }
  if (warp == 8) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mw);
    cute::prefetch_tma_descriptor(&mk);
    cute::prefetch_tma_descriptor(&ms);
    cute::prefetch_tma_descriptor(&mr);
  }
  if (warp == 8 && cute::elect_one_sync()) {
    s.full_q.init(1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s.full_kv[i].init(1);
      s.empty_kv[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 9) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (int i = 0; i < Stages; ++i) {
        s.full_umma[i].init(1);
        s.empty_umma[i].init(256);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(512, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp == 8) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, N, 128>(&mq, &s.full_q, s.fine, 0, g * N);
      deep_gemm::tma::copy<W, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * H * 128 + 8 * W * 4);
      for (int block = 0; block < blocks; ++block) {
        int stage = block % 8, phase = (block / 8) & 1;
        s.empty_kv[stage].wait(phase ^ 1);
        deep_gemm::tma::copy<128, 128, 128>(
            &mk, &s.full_kv[stage], s.raw_k[stage], 0,
            a.seed + (first_block + block) * 128);
        deep_gemm::tma::copy<128, 1, 0>(
            &ms, &s.full_kv[stage], s.raw_scale[stage],
            a.seed + (first_block + block) * 128, 0);
        deep_gemm::tma::copy<128, 1, 0>(
            &mr, &s.full_kv[stage], s.raw_rho[stage],
            a.seed + (first_block + block) * 128, 0);
        s.full_kv[stage].arrive_and_expect_tx(128 * 128 + 2 * 128 * 4);
      }
    }
  } else if (warp == 9) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    auto instruction = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
    s.full_q.wait(0);
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 8, phase = (block / 8) & 1, u = block % Stages,
          up = (block / Stages) & 1;
      s.full_kv[stage].wait(phase);
      s.empty_umma[u].wait(up ^ 1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                        128, 128>(
            reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]), 0,
            k * 32);
        auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                        128, 128>(
            reinterpret_cast<cutlass::float_e4m3_t*>(s.fine), 0, k * 32);
        cute::SM100_MMA_F8F6F4_SS::fma(ad, bd, s.tmem + u * N, k, runtime);
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[u]));
    }
  } else if (warp >= 10) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int wg = t / 128, kt = t % 128, group = g * 2 + wg;
    float weights[4][H];
    Param resident[4];
    bool eligible_q[4];
    s.full_q.wait(0);
#pragma unroll
    for (int qr = 0; qr < 4; ++qr) {
      resident[qr] = s.row[wg * 4 + qr];
      eligible_q[qr] = a.budget[group] == H - 2;
#pragma unroll
      for (int h = 0; h < H; ++h)
        weights[qr][h] = deep_gemm::ptx::ld_shared(&s.fw[wg * 4 + qr][h]);
    }
    const uint32_t query_tmem_base = s.tmem + wg * (4 * H);
    // Preserve Q4/lane ownership; form a byte address once and advance64B/K128.
    uint64_t word_address = reinterpret_cast<uint64_t>(page.words) +
                            (uint64_t(group) * uint64_t(page.word_count) +
                             uint64_t(first_block) * 4u + uint64_t(warp & 3)) *
                                16u +
                            uint64_t(lane) * 4u;
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 8, phase = (block / 8) & 1, u = block % Stages,
          up = (block / Stages) & 1;
      s.full_kv[stage].wait(phase);
      float scale = deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]);
      uint32_t code = deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]);
      s.full_umma[u].wait(up);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      s.empty_kv[stage].arrive();
      float acc[4 * H];
      uint32_t address = query_tmem_base + u * N;
      load_tmem<64>(address, acc);
      cutlass::arch::fence_view_async_tmem_load();
      load_tmem<8>(address + 64, acc + 64);
      cutlass::arch::fence_view_async_tmem_load();
      deep_gemm::ptx::tcgen05_before_thread_sync();
      s.empty_umma[u].arrive();
      __half_raw half = __nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
      float rho = __half2float(static_cast<__half>(half));
      int key = a.seed + (first_block + block) * 128 + kt;
      uint32_t word_value = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int qr = i;
        float2 s0 = make_float2(0, 0), s1 = make_float2(0, 0);
#pragma unroll
        for (int h = 0; h < H - 2; h += 4) {
          s0 = __ffma2_rn(
              make_float2(fabsf(acc[i * H + h + 2]), fabsf(acc[i * H + h + 3])),
              make_float2(weights[qr][h + 2], weights[qr][h + 3]), s0);
          if (h + 2 < H - 2)
            s1 = __ffma2_rn(make_float2(fabsf(acc[i * H + h + 4]),
                                        fabsf(acc[i * H + h + 5])),
                            make_float2(weights[qr][h + 4], weights[qr][h + 5]),
                            s1);
        }
        float2 pair = __fadd2_rn(s0, s1);
        float paid_sum = __fadd_rn(pair.x, pair.y);
        const Param& p = resident[qr];
        float lo0 = __fmul_rn(acc[i * H], p.lo0);
        float neg = __fadd_rn(0.0f, __fmaf_rn(acc[i * H + 1], p.lo1, lo0));
        float signed_score = __fmaf_rn(paid_sum, 0.5f, -neg);
        float tail = __fmaf_ru(rho, p.guard, __fadd_ru(signed_score, p.raw));
        float value = __fmaf_ru(scale, tail, p.out);
        if (!(code < 127u && isfinite(neg)))
          value = __uint_as_float(0x7f800000u);
        bool visible = key < logical_nk && key >= p.start && key < p.end;
        unsigned ballot = h18_keep_ballot(
            value, p.tau, uint32_t(eligible_q[qr]) & uint32_t(visible));
        if (lane == qr) word_value = ballot;
      }
      {
        int word = (first_block + block) * 4 + (warp & 3);

        if (lane < 4 && eligible_q[0] && word < active_word_count)
          *reinterpret_cast<uint32_t*>(word_address) = word_value;
        word_address += 64u;
      }
    }
  }
  __syncthreads();
  if (warp == 9) cute::TMEM::Allocator1Sm().free(s.tmem, 512);
}

}  // namespace f2_live_h18
void f2_live_h18_trusted_live_k(int64_t paid, std::vector<Tensor> x,
                                std::vector<Tensor> streams, Tensor pool,
                                Tensor table, Tensor head, Tensor status,
                                Tensor error, int64_t seed, Tensor words,
                                Tensor work_ids, Tensor work_inverse,
                                Tensor work_config, int64_t launch_grid,
                                Tensor live_nk) {
  // Private paid16 screen: owned x15/stream6, I32[Q/8] ids/inverse,
  // I32[2] schedule and immutable device I32[1] liveNK. Q is padded to8.
  // The serial owner retains these views and the page/word buffers for the
  // stream.
  using namespace f2_live_h18;
  auto [a, pg] = dosa_emit_host::bind<Args, PagedArgs>(
      paid, x, streams, pool, table, head, status, error, seed, words);
  int nq = a.nq, nk = a.nk;
  DeviceGuard guard(x[0].get_device_index());
  auto stream = get_current_cuda_stream();
  a.ids = work_ids.mutable_data_ptr<int>();
  a.members = work_inverse.mutable_data_ptr<int>();
  a.counts = work_config.mutable_data_ptr<int>();
  auto mq =
      dosa_screen::make_map(x[5].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                            {128, (cuuint64_t)nq * 18}, {128}, {128, 144},
                            CU_TENSOR_MAP_SWIZZLE_128B);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {64, (cuuint64_t)nq},
      {256}, {20, 8}, CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, 128);
  int grid = int(launch_grid);
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      f2_live_h18_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(F2LiveH18Storage)));
  f2_live_h18_kernel<<<grid, 384, sizeof(F2LiveH18Storage), stream>>>(
      a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
      live_nk.mutable_data_ptr<int32_t>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("emit_h16_indexed_trusted_live_k") "(int paid, Tensor[] x, Tensor(a!)[] streams, Tensor! pool, Tensor! table, Tensor! head, "
      "Tensor! status, Tensor! error, int seed, Tensor! words, Tensor work_ids, Tensor work_inverse, "
      "Tensor work_config, int grid, Tensor live_nk) -> ()");
  ops.def(DOSA_OP("emit_h18_indexed_trusted_live_k") "(int paid, Tensor[] x, Tensor(a!)[] streams, Tensor! pool, Tensor! table, Tensor! head, "
      "Tensor! status, Tensor! error, int seed, Tensor! words, Tensor work_ids, Tensor work_inverse, "
      "Tensor work_config, int grid, Tensor live_nk) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("emit_h16_indexed_trusted_live_k"),
           TORCH_BOX(&DOSA_NAMESPACE::f1_abs_qfma_trusted_live_k));
  ops.impl(DOSA_OP("emit_h18_indexed_trusted_live_k"),
           TORCH_BOX(&DOSA_NAMESPACE::f2_live_h18_trusted_live_k));
}
