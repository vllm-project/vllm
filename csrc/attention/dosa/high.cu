#include "common.cuh"
#include "emit.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
namespace f2_h22_late {
using namespace dosa_screen;
}
namespace f2_h22_late {
// Hardware-derived shared high family:4*(paid+2) stays within128 TMEM columns.
constexpr int kHighPaid = dosa_geometry::Heads == 64 ? 28 : 20;
constexpr int kScanHeads = kHighPaid + 2;
constexpr int kWeightPairs = kHighPaid / 2;
constexpr int kWeightWords = 4 * kWeightPairs;
constexpr int kRecordWords = kWeightWords + 24;
constexpr int kTileColumns = 4 * kScanHeads;
// Round high-family coefficients upward to BF16. A scale outside(0,2^64]
// retains every visible key; the original FP32 reduction and guards still run.
__global__ void prepare_kernel(const uint8_t* fine, const float* meta,
                               const float* dd_scale, const float* raw,
                               const float* out, const int* hint,
                               const int* starts, const int* ends,
                               const int* budget, const float* tau, int* error,
                               int nq, int nk, uint32_t* record,
                               int* fast_budget, int* slow_budget, int seed,
                               const int32_t* live_nk) {
  (void)fine;  // No fine scan: nonfinite raw-TC propagation is in the contract.
  const int group = blockIdx.x, t = threadIdx.x;
  const int qr = t >> 5, lane = t & 31, row = group * 4 + qr;
  if (group >= nq / 4) return;
  uint32_t* dst = record + uint64_t(group) * kRecordWords;
  const int old_budget = budget[group];
  if (t == 0) {
    dst[kWeightWords + 16] = 0;
    fast_budget[group] = 14;
    slow_budget[group] = old_budget;
  }
  const int logical_nk = __ldg(live_nk);
  if (logical_nk < seed || logical_nk > nk) {
    if (t < 4) atomicOr(error + group * 4 + t, 1);
    return;  // Initial flags/routes are already safe; no original plan read.
  }
  nk = logical_nk;  // The original ends<=NK domain, not the physical capacity.

  if (old_budget != kHighPaid)
    return;  // Uniform, before any inactive plan read.
  __shared__ int good_q[4], start_q[4], end_q[4];
  __shared__ uint32_t raw_q[4], out_q[4], dd_exp[8];
  if (t < 7) dst[kWeightWords + 17 + t] = 0;
  const float* mw = meta + uint64_t(row) * 96;
  bool good = true;
  if (lane < kWeightPairs) {
    const float a0 = mw[2 + 2 * lane], a1 = mw[3 + 2 * lane];
    const uint32_t b0 = __float_as_uint(a0), b1 = __float_as_uint(a1);
    // Positive low discarded bits need one upward BF16 ULP. Truncating a
    // negative finite payload already rounds toward +infinity. Preserve +/-0.
    const uint32_t h0 =
        (b0 >> 16) +
        (((b0 & 0x80000000u) == 0u && (b0 & 65535u) != 0u) ? 1u : 0u);
    const uint32_t h1 =
        (b1 >> 16) +
        (((b1 & 0x80000000u) == 0u && (b1 & 65535u) != 0u) ? 1u : 0u);
    const float f0 = __uint_as_float(h0 << 16), f1 = __uint_as_float(h1 << 16);
    good = isfinite(a0) && isfinite(a1) && fabsf(a0) <= 65504.f &&
           fabsf(a1) <= 65504.f && isfinite(f0) && isfinite(f1) && f0 >= a0 &&
           f1 >= a1;
    dst[qr * kWeightPairs + lane] = h0 | (h1 << 16);
  }
  const bool all_good = __all_sync(0xffffffffu, good);
  if (lane == 0) {
    const float c = mw[kScanHeads], rv = raw[row], ov = out[row], tv = tau[row];
    const int sv = starts[row], ev = ends[row];
    bool valid = all_good && hint[row] == 2 && error[row] == 0 && isfinite(c) &&
                 c >= 0.f && c <= 0x1p16f && isfinite(rv) && rv >= 0.f &&
                 rv <= 0x1p16f && isfinite(ov) && ov >= 0.f && ov <= 0x1p16f &&
                 isfinite(tv) && sv >= 0 && sv <= ev && ev <= nk;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float value = dd_scale[uint64_t(row) * kScanHeads + j];
      const uint32_t bits = __float_as_uint(value);
      const uint32_t exponent = (bits >> 23) & 255u;
      valid &= (bits & 0x807fffffu) == 0u && exponent > 0u && exponent < 255u &&
               value <= 0x1p16f;
      dd_exp[2 * qr + j] = exponent;
    }
    dst[kWeightWords + 8 + qr] = __float_as_uint(tv);
    dst[kWeightWords + 12 + qr] =
        __float_as_uint(c);  // Exact original Cold, no added guard.
    raw_q[qr] = __float_as_uint(rv);
    out_q[qr] = __float_as_uint(ov);
    start_q[qr] = sv;
    end_q[qr] = ev;
    good_q[qr] = valid ? 1 : 0;
  }
  __syncthreads();  // Invalid rows participate; flags publish only after all
                    // data.
  if (t == 0) {
    int base = end_q[0];
#pragma unroll
    for (int q = 1; q < 4; ++q) base = min(base, end_q[q]);
    bool valid = true;
    uint32_t offsets[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int64_t difference = int64_t(end_q[q]) - int64_t(base);
      valid &= good_q[q] != 0 && raw_q[q] == raw_q[0] && out_q[q] == out_q[0] &&
               start_q[q] == start_q[0] && difference >= 0 &&
               difference <= 65535;
      offsets[q] = uint32_t(difference) & 65535u;
    }
    dst[kWeightWords + 0] =
        dd_exp[0] | (dd_exp[1] << 8) | (dd_exp[2] << 16) | (dd_exp[3] << 24);
    dst[kWeightWords + 1] =
        dd_exp[4] | (dd_exp[5] << 8) | (dd_exp[6] << 16) | (dd_exp[7] << 24);
    dst[kWeightWords + 2] = raw_q[0];
    dst[kWeightWords + 3] = out_q[0];
    dst[kWeightWords + 4] = uint32_t(start_q[0]);
    dst[kWeightWords + 5] = uint32_t(base);
    dst[kWeightWords + 6] = offsets[0] | (offsets[1] << 16);
    dst[kWeightWords + 7] = offsets[2] | (offsets[3] << 16);
    fast_budget[group] = valid ? kHighPaid : 14;
    slow_budget[group] = valid ? 14 : old_budget;
    dst[kWeightWords + 16] = valid ? 1u : 0u;
  }
}

// Fixed V1: Q4 K amortization, two raw stages, one UMMA slot perK128mathWG.
// record: 40half2 /2DDexp /raw,out,start,endbase /2endoffs /4tau /4guard
// /flags.
struct alignas(1024) PackedStorage {
  alignas(1024) uint8_t fine[4 * kScanHeads * 128];
  alignas(16) uint32_t record[kRecordWords];
  alignas(16) uint2 decoded_weights[kWeightWords];
  alignas(1024) uint8_t raw_k[2][256 * 128];
  float raw_scale[2][256];
  uint32_t raw_rho[2][256];
  Barrier full_q, full_coeff, full_kv[2], empty_kv[2], full_umma[2],
      empty_umma[2];
  uint32_t tmem;
};
__device__ __forceinline__ float2 decode_bf16_pair(uint32_t bits) {
  float2 result;
  // Exact BF16 payload reconstruction, including BF16 subnormals. These
  // volatile bit operations retain40 packed words across the K loop; no
  // F16 conversion/FMA is needed to restore the two independent F32 values.
  uint32_t lo, hi;
  asm volatile(
      "{.reg .b32 packed; mov.b32 packed,%2; shl.b32 %0,packed,16; and.b32 "
      "%1,packed,0xffff0000;}"
      : "=r"(lo), "=r"(hi)
      : "r"(bits));
  result.x = __uint_as_float(lo);
  result.y = __uint_as_float(hi);
  return result;
}
// Two exact F32 bit patterns loaded together from the CTA coefficient cache.
__device__ __forceinline__ float2 load_decoded_pair(const uint2* value) {
  uint32_t lo, hi;
  uint32_t address = uint32_t(__cvta_generic_to_shared(value));
  asm volatile("ld.shared.v2.b32 {%0,%1},[%2];"
               : "=r"(lo), "=r"(hi)
               : "r"(address)
               : "memory");
  return make_float2(__uint_as_float(lo), __uint_as_float(hi));
}
template <int Q>
__device__ __forceinline__ uint32_t consume_packed_query(
    const uint32_t start, int buf, PackedStorage& s, const uint32_t (&dd)[2],
    const uint32_t (&end_delta)[2], const float (&tau)[4],
    const float (&guard)[4], float raw, float out, int visible_start,
    int end_base, float scale, float rho, int key, int nk) {
  // Current query only: issue coefficient loads before the three TMEM waits.
  float2 current_weights[kWeightPairs];
#pragma unroll
  for (int h = 0; h < kWeightPairs; ++h)
    current_weights[h] =
        load_decoded_pair(s.decoded_weights + Q * kWeightPairs + h);
  float acc[kScanHeads];
  uint32_t address = start + Q * kScanHeads;
  load_tmem<16>(address, acc);
  cutlass::arch::fence_view_async_tmem_load();
  if constexpr (kScanHeads == 30) {
    load_tmem<8>(address + 16, acc + 16);
    cutlass::arch::fence_view_async_tmem_load();
    load_tmem<4>(address + 24, acc + 24);
    cutlass::arch::fence_view_async_tmem_load();
    load_tmem<2>(address + 28, acc + 28);
    cutlass::arch::fence_view_async_tmem_load();
  } else {
    load_tmem<4>(address + 16, acc + 16);
    cutlass::arch::fence_view_async_tmem_load();
    load_tmem<2>(address + 20, acc + 20);
    cutlass::arch::fence_view_async_tmem_load();
  }
  if constexpr (Q == 3) {
    deep_gemm::ptx::tcgen05_before_thread_sync();
    s.empty_umma[buf].arrive();
  }
  float2 s0 = make_float2(0, 0), s1 = make_float2(0, 0);
#pragma unroll
  for (int h = 0; h < kHighPaid; h += 4) {
    float2 w0 = current_weights[h / 2];
    s0 = __ffma2_rn(
        __fadd2_rn(make_float2(acc[h + 2], acc[h + 3]),
                   make_float2(fabsf(acc[h + 2]), fabsf(acc[h + 3]))),
        w0, s0);
    float2 w1 = current_weights[h / 2 + 1];
    s1 = __ffma2_rn(
        __fadd2_rn(make_float2(acc[h + 4], acc[h + 5]),
                   make_float2(fabsf(acc[h + 4]), fabsf(acc[h + 5]))),
        w1, s1);
  }
  float2 pair = __fadd2_rn(s0, s1);
  float paid = __fmul_rn(__fadd_rn(pair.x, pair.y), 0.5f);
  uint32_t ex = dd[Q / 2] >> ((Q % 2) * 16);
  float lo_scale0 = __uint_as_float((ex & 255u) << 23),
        lo_scale1 = __uint_as_float(((ex >> 8) & 255u) << 23);
  float lo0 = __fmul_rn(acc[0], lo_scale0);
  float neg = __fadd_rn(0.0f, __fmaf_rn(acc[1], lo_scale1, lo0));
  float signed_score = __fadd_rn(paid, -fmaxf(neg, 0.0f));
  float guard_term = __fmul_rn(rho, guard[Q]);
  float u = __fadd_rn(
      __fmul_rn(scale, __fadd_rn(__fadd_rn(signed_score, raw), guard_term)),
      out);
  int visible_end =
      end_base + int((end_delta[Q / 2] >> ((Q % 2) * 16)) & 65535u);
  bool visible = key < nk && key >= visible_start && key < visible_end;
  // RU-BF16 monotonicity uses this finite keyscale domain. Domain failure
  // keeps the visible pair; all TMEM/slot lifecycle remains unconditional.
  bool scale_admitted = isfinite(scale) && scale > 0.f && scale <= 0x1p64f;
  return !scale_admitted
             ? (visible ? (1u << Q) : 0u)
             : trusted_keep_mask(u, tau[Q], uint32_t(visible), 1u << Q);
}
__global__ __launch_bounds__(384, 2) void packed_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mp,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ PagedArgs page, const uint32_t* records,
    const int32_t* live_nk) {
  int t = threadIdx.x, g = blockIdx.x, warp = t / 32, lane = t & 31;
  int logical_nk = a.nk, active_word_count = page.word_count;
  logical_nk = __ldg(live_nk);
  if (logical_nk < a.seed || logical_nk > a.nk) {
    if (blockIdx.x == 0 && blockIdx.y == 0)
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return;
  }
  active_word_count = (logical_nk - a.seed + 31) / 32;

  if (a.budget[g] != kHighPaid) return;
  if (records[uint64_t(g) * kRecordWords + kWeightWords + 16] != 1u) {
    if (t < 4) atomicOr(a.error + g * 4 + t, 1);
    return;
  }
  if (logical_nk == a.seed) return;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<PackedStorage*>(memory);
  if (warp == 8) {
    cute::prefetch_tma_descriptor(&mq);
    cute::prefetch_tma_descriptor(&mp);
    cute::prefetch_tma_descriptor(&mk);
    cute::prefetch_tma_descriptor(&ms);
    cute::prefetch_tma_descriptor(&mr);
  }
  if (warp == 8 && cute::elect_one_sync()) {
    s.full_q.init(1);
    s.full_coeff.init(1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      s.full_kv[i].init(1);
      s.empty_kv[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 9) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        s.full_umma[i].init(1);
        s.empty_umma[i].init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(256, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  int total = (logical_nk - a.seed + 255) / 256,
      first = total * int(blockIdx.y) / 8,
      stop = total * int(blockIdx.y + 1) / 8, blocks = stop - first;
  if (warp >= 8) cutlass::arch::warpgroup_reg_dealloc<40>();
  if (warp == 8) {
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, kTileColumns, 128>(&mq, &s.full_q, s.fine, 0,
                                                   g * kTileColumns);
      deep_gemm::tma::copy<kRecordWords, 1, 0>(&mp, &s.full_q, s.record, 0, g);
      s.full_q.arrive_and_expect_tx(kTileColumns * 128 + kRecordWords * 4);
      for (int block = 0; block < blocks; ++block) {
        int stage = block % 2, phase = (block / 2) & 1;
        s.empty_kv[stage].wait(phase ^ 1);
        int offset = a.seed + (first + block) * 256;
        deep_gemm::tma::copy<128, 256, 128>(&mk, &s.full_kv[stage],
                                            s.raw_k[stage], 0, offset);
        deep_gemm::tma::copy<256, 1, 0>(&ms, &s.full_kv[stage],
                                        s.raw_scale[stage], offset, 0);
        deep_gemm::tma::copy<256, 1, 0>(&mr, &s.full_kv[stage],
                                        s.raw_rho[stage], offset, 0);
        s.full_kv[stage].arrive_and_expect_tx(34816);
      }
    }
  } else if (warp == 9 || warp == 10) {
    int wg = warp - 9;
    auto instruction = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, kTileColumns,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
    s.full_q.wait(0);
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 2, phase = (block / 2) & 1, up = block & 1;
      s.full_kv[stage].wait(phase);
      s.empty_umma[wg].wait(up ^ 1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                        128, 128>(
            reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]), wg * 128,
            k * 32);
        auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                        128, 128>(
            reinterpret_cast<cutlass::float_e4m3_t*>(s.fine), 0, k * 32);
        cute::SM100_MMA_F8F6F4_SS::fma(ad, bd, s.tmem + wg * kTileColumns, k,
                                       runtime);
      }
      cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&s.full_umma[wg]));
    }
  } else if (warp == 11) {
    s.full_q.wait(0);
    for (int i = lane; i < kWeightWords; i += 32) {
      uint32_t bits = deep_gemm::ptx::ld_shared(s.record + i);
      s.decoded_weights[i] = make_uint2(bits << 16, bits & 0xffff0000u);
    }
    __syncwarp();
    if (cute::elect_one_sync()) s.full_coeff.arrive();
  } else if (warp < 8) {
    cutlass::arch::warpgroup_reg_alloc<96>();
    // Same stable scalar epoch, reloaded once AFTER the dynamic register
    // allocation. Do not carry startup NK across the donation boundary.
    int math_nk = a.nk;
    asm volatile("ld.global.nc.u32 %0,[%1];"
                 : "=r"(math_nk)
                 : "l"(live_nk)
                 : "memory");

    const int math_word_count = (math_nk - a.seed + 31) / 32;

    int wg = t / 128;
    s.full_coeff.wait(0);
    const uint32_t math_tmem_base = s.tmem + wg * kTileColumns;
    uint32_t dd[2], end_delta[2];
    float tau[4], guard[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      dd[i] = deep_gemm::ptx::ld_shared(s.record + kWeightWords + 0 + i);
      end_delta[i] = deep_gemm::ptx::ld_shared(s.record + kWeightWords + 6 + i);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      tau[i] = __uint_as_float(
          deep_gemm::ptx::ld_shared(s.record + kWeightWords + 8 + i));
      guard[i] = __uint_as_float(
          deep_gemm::ptx::ld_shared(s.record + kWeightWords + 12 + i));
    }
    float raw = __uint_as_float(
              deep_gemm::ptx::ld_shared(s.record + kWeightWords + 2)),
          out = __uint_as_float(
              deep_gemm::ptx::ld_shared(s.record + kWeightWords + 3));
    int start = int(deep_gemm::ptx::ld_shared(s.record + kWeightWords + 4)),
        end_base = int(deep_gemm::ptx::ld_shared(s.record + kWeightWords + 5));
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 2, phase = (block / 2) & 1, up = block & 1;
      s.full_kv[stage].wait(phase);
      float scale = deep_gemm::ptx::ld_shared(&s.raw_scale[stage][t]);
      uint32_t code = deep_gemm::ptx::ld_shared(&s.raw_rho[stage][t]) & 255u;
      s.full_umma[wg].wait(up);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      s.empty_kv[stage].arrive();
      __half_raw half = __nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
      float rho = __half2float(static_cast<__half>(half));
      int key = a.seed + (first + block) * 256 + t;
      uint32_t address = math_tmem_base, m = 0;
      m |= consume_packed_query<0>(address, wg, s, dd, end_delta, tau, guard,
                                   raw, out, start, end_base, scale, rho, key,
                                   math_nk);
      asm volatile("" : "+r"(m)::"memory");
      m |= consume_packed_query<1>(address, wg, s, dd, end_delta, tau, guard,
                                   raw, out, start, end_base, scale, rho, key,
                                   math_nk);
      asm volatile("" : "+r"(m)::"memory");
      m |= consume_packed_query<2>(address, wg, s, dd, end_delta, tau, guard,
                                   raw, out, start, end_base, scale, rho, key,
                                   math_nk);
      asm volatile("" : "+r"(m)::"memory");
      m |= consume_packed_query<3>(address, wg, s, dd, end_delta, tau, guard,
                                   raw, out, start, end_base, scale, rho, key,
                                   math_nk);
      int word = (first + block) * 8 + warp;
      unsigned value = 0;
#pragma unroll
      for (int qr = 0; qr < 4; ++qr) {
        unsigned mask = __ballot_sync(0xffffffffu, (m & (1u << qr)) != 0);
        if (lane == qr) value = mask;
      }
      if (lane < 4 && word < math_word_count)
        page.words[(int64_t(g) * page.word_count + word) * 4 + lane] = value;
    }
  }
  __syncthreads();
  if (warp == 9) cute::TMEM::Allocator1Sm().free(s.tmem, 256);
}

}  // namespace f2_h22_late
namespace f2_h22_late_original {
using namespace f2_h22_late;
template <int H, int BQ = 4>
struct alignas(1024) Storage {
  static constexpr int W = (H + 3) / 4 * 4;
  alignas(1024) uint8_t fine[BQ * H * 128];
  alignas(16) float fw[BQ][W];
  alignas(1024) uint8_t raw_k[4][256 * 128];
  float raw_scale[4][256];
  uint32_t raw_rho[4][256];
  Param row[BQ];
  Barrier full_q, full_kv[4], empty_kv[4], full_umma[4], empty_umma[4];
  uint32_t tmem;
};
template <int H, int BQ = 4>
__global__ __launch_bounds__(384, 1) void kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ PagedArgs page, const int32_t* live_nk) {
  constexpr int N = BQ * H, W = (H + 3) / 4 * 4, TM = N * 4 <= 256 ? 256 : 512;
  int t = threadIdx.x, g = blockIdx.x, warp = t / 32, lane = t & 31;
  int logical_nk = a.nk, active_word_count = page.word_count;
  logical_nk = __ldg(live_nk);
  if (logical_nk < a.seed || logical_nk > a.nk) {
    if (blockIdx.x == 0 && blockIdx.y == 0)
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return;
  }
  active_word_count = (logical_nk - a.seed + 31) / 32;

  int basegroup = g * (BQ / 4);
  bool eligible = false;
#pragma unroll
  for (int j = 0; j < BQ / 4; ++j) {
    int budget = a.budget[basegroup + j];
    if (!dosa_geometry::valid_certificate_budget(budget)) {
      if (t < 4) atomicOr(a.error + (basegroup + j) * 4 + t, 1);
    }
    eligible |= budget == H - 2;
  }
  if (!eligible) return;
  if (logical_nk == a.seed) return;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<Storage<H, BQ>*>(memory);
  constexpr int Words = H >= 22 ? 96 : 64;
  if (t < BQ) {
    int r = g * BQ + t;
    Param p;
    p.guard = a.meta[r * Words + H];
    p.lo0 = a.fine_scale[r * H];
    p.lo1 = a.fine_scale[r * H + 1];
    p.raw = a.raw[r];
    p.out = a.out[r];
    p.tau = a.tau[r];
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
    for (int i = 0; i < 4; ++i) {
      s.full_kv[i].init(1);
      s.empty_kv[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp == 9) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        s.full_umma[i].init(1);
        s.empty_umma[i].init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(TM, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const int all_blocks = (logical_nk - a.seed + 255) / 256;
  const int first_block = H >= 22 ? all_blocks * int(blockIdx.y) / 8 : 0;
  const int stop_block =
      H >= 22 ? all_blocks * int(blockIdx.y + 1) / 8 : all_blocks;
  const int blocks = stop_block - first_block;
  if (warp == 8) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, BQ * H, 128>(&mq, &s.full_q, s.fine, 0,
                                             g * BQ * H);
      deep_gemm::tma::copy<W, BQ, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * BQ);
      s.full_q.arrive_and_expect_tx(BQ * H * 128 + BQ * W * 4);
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
  } else if (warp == 9 || warp == 10) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    const int wg = warp - 9;
    auto instruction = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
    s.full_q.wait(0);
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 4, phase = (block / 4) & 1;
      s.full_kv[stage].wait(phase);
      int us = block % 2, up = (block / 2) & 1;
      {
        int buf = wg * 2 + us;
        s.empty_umma[buf].wait(up ^ 1);
        deep_gemm::ptx::tcgen05_after_thread_sync();
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                          0, 128, 128>(
              reinterpret_cast<cutlass::float_e4m3_t*>(s.raw_k[stage]),
              wg * 128, k * 32);
          auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                          0, 128, 128>(
              reinterpret_cast<cutlass::float_e4m3_t*>(s.fine), 0, k * 32);
          cute::SM100_MMA_F8F6F4_SS::fma(ad, bd, s.tmem + buf * N, k, runtime);
        }
        cutlass::arch::umma_arrive(
            reinterpret_cast<uint64_t*>(&s.full_umma[buf]));
      }
    }
  } else if (warp >= 11) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int wg = t / 128;
    float weights[BQ][H];
    Param resident[BQ];
    bool eligible_q[BQ];
    s.full_q.wait(0);
#pragma unroll
    for (int qr = 0; qr < BQ; ++qr) {
      resident[qr] = s.row[qr];
      eligible_q[qr] = a.budget[basegroup + qr / 4] == H - 2;
#pragma unroll
      for (int h = 0; h < H; ++h)
        weights[qr][h] = deep_gemm::ptx::ld_shared(&s.fw[qr][h]);
    }
    for (int block = 0; block < blocks; ++block) {
      int stage = block % 4, phase = (block / 4) & 1;
      int buf = wg * 2 + (block % 2), up = (block / 2) & 1;
      s.full_kv[stage].wait(phase);
      float scale = deep_gemm::ptx::ld_shared(&s.raw_scale[stage][t]);
      uint32_t code = deep_gemm::ptx::ld_shared(&s.raw_rho[stage][t]) & 255u;
      s.full_umma[buf].wait(up);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      s.empty_kv[stage].arrive();
      uint32_t start = s.tmem + buf * N;
      __half_raw half = __nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
      float rho = __half2float(static_cast<__half>(half));
      int key = a.seed + (first_block + block) * 256 + t;
      uint32_t m = 0;
#pragma unroll
      for (int chunk = 0; chunk < BQ / 4; ++chunk) {
        constexpr int Part = 4 * H;
        float acc[Part];
        uint32_t address = start + chunk * Part;
        load_tmem<64>(address, acc);
        cutlass::arch::fence_view_async_tmem_load();
        if constexpr (Part == 88) {
          load_tmem<16>(address + 64, acc + 64);
          cutlass::arch::fence_view_async_tmem_load();
          load_tmem<8>(address + 80, acc + 80);
          cutlass::arch::fence_view_async_tmem_load();
        }
        if constexpr (Part == 120) {
          load_tmem<32>(address + 64, acc + 64);
          cutlass::arch::fence_view_async_tmem_load();
          load_tmem<16>(address + 96, acc + 96);
          cutlass::arch::fence_view_async_tmem_load();
          load_tmem<8>(address + 112, acc + 112);
          cutlass::arch::fence_view_async_tmem_load();
        }
        if (chunk + 1 == BQ / 4) {
          deep_gemm::ptx::tcgen05_before_thread_sync();
          s.empty_umma[buf].arrive();
        }
#pragma unroll
        for (int pair_query = 0; pair_query < 4; pair_query += 2) {
          // Two independent queries share scheduling opportunities. Each
          // query's original four RN paid chains retain exactly the same update
          // order.
          constexpr int Paid = H - 2;
          float2 s0[2] = {make_float2(0, 0), make_float2(0, 0)};
          float2 s1[2] = {make_float2(0, 0), make_float2(0, 0)};
#pragma unroll
          for (int h = 0; h < Paid; h += 4) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              int i = pair_query + j, qr = chunk * 4 + i;
              s0[j] = __ffma2_rn(
                  __fadd2_rn(
                      make_float2(acc[i * H + h + 2], acc[i * H + h + 3]),
                      make_float2(fabsf(acc[i * H + h + 2]),
                                  fabsf(acc[i * H + h + 3]))),
                  make_float2(weights[qr][h + 2], weights[qr][h + 3]), s0[j]);
              if (h + 2 < Paid)
                s1[j] = __ffma2_rn(
                    __fadd2_rn(
                        make_float2(acc[i * H + h + 4], acc[i * H + h + 5]),
                        make_float2(fabsf(acc[i * H + h + 4]),
                                    fabsf(acc[i * H + h + 5]))),
                    make_float2(weights[qr][h + 4], weights[qr][h + 5]), s1[j]);
            }
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            int i = pair_query + j, qr = chunk * 4 + i;
            float2 pair = __fadd2_rn(s0[j], s1[j]);
            float paid = __fmul_rn(__fadd_rn(pair.x, pair.y), 0.5f);
            const Param& p = resident[qr];
            float lo0 = __fmul_rn(acc[i * H], p.lo0);
            float neg = __fadd_rn(0.0f, __fmaf_rn(acc[i * H + 1], p.lo1, lo0));
            float signed_score = __fadd_rn(paid, -fmaxf(neg, 0.0f));
            float guard = __fmul_rn(rho, p.guard);
            float u = __fadd_rn(
                __fmul_rn(scale,
                          __fadd_rn(__fadd_rn(signed_score, p.raw), guard)),
                p.out);
            bool visible = key < logical_nk && key >= p.start && key < p.end;

            const uint32_t bit = trusted_keep_mask(
                u, p.tau, uint32_t(eligible_q[qr]) & uint32_t(visible),
                1u << qr);
            m |= bit;
          }
        }
      }
      {
        int word = (first_block + block) * 8 + warp;
        unsigned value = 0;
        bool owner = false;
#pragma unroll
        for (int qr = 0; qr < BQ; ++qr) {
          unsigned mask = __ballot_sync(0xffffffffu, ((m >> qr) & 1u) != 0);
          if (lane == qr) {
            value = mask;
            owner = eligible_q[qr];
          }
        }
        // One coalesced warp store for each Q4's four ballot words, including
        // zero. Public transport layout is [Q4group, raw_word,
        // query_within_group].
        if (lane < BQ && owner && word < active_word_count) {
          int row = g * BQ + lane;
          page.words[(int64_t(row / 4) * page.word_count + word) * 4 +
                     row % 4] = value;
        }
      }
    }
  }
  __syncthreads();
  if (warp == 9) cute::TMEM::Allocator1Sm().free(s.tmem, TM);
}

}  // namespace f2_h22_late_original
namespace f2_h22_late_host {
std::pair<f2_h22_late::Args, f2_h22_late::PagedArgs> bind(
    int paid, const std::vector<Tensor>& x, const std::vector<Tensor>& streams,
    Tensor pool, Tensor table, Tensor head, Tensor status, Tensor error,
    int64_t seed, Tensor words) {
  return dosa_emit_host::bind<f2_h22_late::Args, f2_h22_late::PagedArgs>(
      paid, x, streams, pool, table, head, status, error, seed, words);
}

void prepare_live(std::vector<Tensor> x, Tensor error, Tensor record,
                  Tensor fast_budget, Tensor slow_budget, int64_t seed,
                  Tensor live_nk) {
  // Packed record and fast/slow Q4 budgets are disjoint owned I32 tensors.
  // Their x15 inputs follow emit.cuh; record width is kRecordWords and NK is
  // the immutable device scalar for this stream epoch. No borrowed storage
  // aliases them.
  const int nq = x[0].size(0), nk = x[1].size(0);
  DeviceGuard guard(x[0].get_device_index());
  auto stream = get_current_cuda_stream();
  f2_h22_late::prepare_kernel<<<nq / 4, 128, 0, stream>>>(
      x[5].mutable_data_ptr<uint8_t>(), x[7].mutable_data_ptr<float>(),
      x[6].mutable_data_ptr<float>(), x[8].mutable_data_ptr<float>(),
      x[9].mutable_data_ptr<float>(), x[10].mutable_data_ptr<int>(),
      x[11].mutable_data_ptr<int>(), x[12].mutable_data_ptr<int>(),
      x[13].mutable_data_ptr<int>(), x[14].mutable_data_ptr<float>(),
      error.mutable_data_ptr<int>(), nq, nk,
      reinterpret_cast<uint32_t*>(record.mutable_data_ptr<int>()),
      fast_budget.mutable_data_ptr<int>(), slow_budget.mutable_data_ptr<int>(),
      int(seed), live_nk.mutable_data_ptr<int32_t>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
template <int H>
void emit_slow_family(int paid, std::vector<Tensor> x,
                      std::vector<Tensor> streams, Tensor pool, Tensor table,
                      Tensor head, Tensor status, Tensor error, int64_t seed,
                      Tensor words, Tensor live_nk) {
  // Slow high-family scan consumes owned x15/stream6 and liveNK on the current
  // stream.
  auto pair =
      bind(paid, x, streams, pool, table, head, status, error, seed, words);
  auto a = pair.first;
  auto pg = pair.second;
  DeviceGuard guard(x[0].get_device_index());
  auto stream = get_current_cuda_stream();
  auto maps = dosa_emit_host::key_maps(x, 256);
  auto mq =
      dosa_screen::make_map(x[5].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                            {128, (cuuint64_t)a.nq * H}, {128}, {128, 4 * H},
                            CU_TENSOR_MAP_SWIZZLE_128B);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {96, (cuuint64_t)a.nq},
      {384}, {(H + 3) / 4 * 4, 4}, CU_TENSOR_MAP_SWIZZLE_NONE);
  using namespace f2_h22_late_original;
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      kernel<H, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(Storage<H, 4>)));
  kernel<H, 4><<<dim3(a.nq / 4, 8), 384, sizeof(Storage<H, 4>), stream>>>(
      a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
      live_nk.mutable_data_ptr<int32_t>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
void emit_slow_live(int64_t paid, std::vector<Tensor> x,
                    std::vector<Tensor> streams, Tensor pool, Tensor table,
                    Tensor head, Tensor status, Tensor error, int64_t seed,
                    Tensor words, Tensor live_nk) {
  if constexpr (dosa_geometry::Heads == 64) {
    if (paid == 28) {
      return emit_slow_family<30>(paid, std::move(x), std::move(streams), pool,
                                  table, head, status, error, seed, words,
                                  live_nk);
    }
  }
  emit_slow_family<22>(paid, std::move(x), std::move(streams), pool, table,
                       head, status, error, seed, words, live_nk);
}
void emit_fast_live(int64_t paid, std::vector<Tensor> x,
                    std::vector<Tensor> streams, Tensor pool, Tensor table,
                    Tensor head, Tensor status, Tensor error, int64_t seed,
                    Tensor words, Tensor record, Tensor live_nk) {
  // Packed high-family scan also consumes the owned Q4 record from
  // prepare_live.
  auto pair =
      bind(paid, x, streams, pool, table, head, status, error, seed, words);
  auto a = pair.first;
  auto pg = pair.second;
  DeviceGuard guard(x[0].get_device_index());
  auto stream = get_current_cuda_stream();
  auto maps = dosa_emit_host::key_maps(x, 256);
  auto mq = dosa_screen::make_map(
      x[5].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
      {128, (cuuint64_t)a.nq * f2_h22_late::kScanHeads}, {128},
      {128, f2_h22_late::kTileColumns}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mp = dosa_screen::make_map(
      record.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT32,
      {f2_h22_late::kRecordWords, (cuuint64_t)a.nq / 4},
      {f2_h22_late::kRecordWords * 4}, {f2_h22_late::kRecordWords, 1},
      CU_TENSOR_MAP_SWIZZLE_NONE);
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(
      f2_h22_late::packed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(f2_h22_late::PackedStorage)));
  DOSA_CUDA_CHECK(
      cudaFuncSetAttribute(f2_h22_late::packed_kernel,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxShared));
  f2_h22_late::packed_kernel<<<dim3(a.nq / 4, 8), 384,
                               sizeof(f2_h22_late::PackedStorage), stream>>>(
      a, mq, mp, maps.keys, maps.scales, maps.rho, pg,
      reinterpret_cast<uint32_t*>(record.mutable_data_ptr<int>()),
      live_nk.mutable_data_ptr<int32_t>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace f2_h22_late_host

int64_t packed_record_words() { return f2_h22_late::kRecordWords; }

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("packed_record_words") "() -> int");
  ops.def(DOSA_OP("prepare_fp16_h22_live_k") "(Tensor[] x, Tensor! error, Tensor! record, Tensor! fast_budget, Tensor! slow_budget, int seed, "
      "Tensor live_nk) -> ()");
  ops.def(DOSA_OP("emit_trusted_candidates_live_k") "(int paid, Tensor[] x, Tensor(a!)[] streams, Tensor! pool, Tensor! table, Tensor! head, "
      "Tensor! status, Tensor! error, int seed, Tensor! words, Tensor live_nk) -> ()");
  ops.def(DOSA_OP("emit_fp16_candidates_live_k") "(int paid, Tensor[] x, Tensor(a!)[] streams, Tensor! pool, Tensor! table, Tensor! head, "
      "Tensor! status, Tensor! error, int seed, Tensor! words, Tensor record, Tensor live_nk) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
  ops.impl(DOSA_OP("packed_record_words"),
           TORCH_BOX(&DOSA_NAMESPACE::packed_record_words));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("prepare_fp16_h22_live_k"),
           TORCH_BOX(&DOSA_NAMESPACE::f2_h22_late_host::prepare_live));
  ops.impl(DOSA_OP("emit_trusted_candidates_live_k"),
           TORCH_BOX(&DOSA_NAMESPACE::f2_h22_late_host::emit_slow_live));
  ops.impl(DOSA_OP("emit_fp16_candidates_live_k"),
           TORCH_BOX(&DOSA_NAMESPACE::f2_h22_late_host::emit_fast_live));
}
