#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace f2_h22_late {
using namespace dosa_screen;
constexpr int kHighPaid = dosa_geometry::Heads == 64 ? 28 : 20;
constexpr int kScanHeads = kHighPaid + 2;
constexpr int kWeightPairs = kHighPaid / 2;
constexpr int kWeightWords = 4 * kWeightPairs;
constexpr int kRecordWords = kWeightWords + 24;
constexpr int kTileColumns = 4 * kScanHeads;
// Packed-domain failures use raw recovery only with independent SAFE admission;
// all other failures use exhaustive native-MX recovery.
constexpr int kPackedDomainError = 2048;
__global__ void prepare_kernel(const __grid_constant__ Args a, uint32_t* record,
                               int* admitted, BitmapArgs page,
                               const int32_t* live_nk) {
  int nk = a.nk;
  const int group = blockIdx.x, t = threadIdx.x;
  const int qr = t >> 5, lane = t & 31, row = group * 4 + qr;
  if (group >= a.nq / 4) return;
  uint32_t* dst = record + uint64_t(group) * kRecordWords;
  const int old_budget = a.budget[group];
  if (t == 0) {
    dst[kWeightWords + 16] = 0;
    admitted[group] = 14;
  }
  const int logical_nk = __ldg(live_nk);
  if (logical_nk < a.seed || logical_nk > nk) {
    if (t < 4) atomicOr(a.error + group * 4 + t, 1);
    return;  // Initial flags/routes are already safe; no original plan read.
  }
  nk =
      logical_nk;  // The original a.ends<=NK domain, not the physical capacity.

  if (old_budget != kHighPaid) {
    if (t < 4 && !dosa_geometry::valid_certificate_budget(old_budget))
      atomicOr(a.error + group * 4 + t, 1);
    return;  // Uniform, before any inactive plan read.
  }
  __shared__ int good_q[4], start_q[4], end_q[4], packable;
  __shared__ uint32_t raw_q[4], out_q[4], dd_exp[8];
  if (t < 7) dst[kWeightWords + 17 + t] = 0;
  const float* mw = a.meta + uint64_t(row) * 96;
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
    const float c = mw[kScanHeads], rv = a.raw[row], ov = a.out[row],
                tv = a.tau[row];
    const int sv = a.starts[row], ev = a.ends[row];
    bool valid = all_good && a.hint[row] == 2 && a.error[row] == 0 &&
                 isfinite(c) && c >= 0.f && c <= 0x1p16f && isfinite(rv) &&
                 rv >= 0.f && rv <= 0x1p16f && isfinite(ov) && ov >= 0.f &&
                 ov <= 0x1p16f && isfinite(tv) && sv >= 0 && sv <= ev &&
                 ev <= nk;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float value = a.fine_scale[uint64_t(row) * kScanHeads + j];
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
  __syncthreads();  // Join row metadata before group flags.
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
    if (!valid)
      for (int q = 0; q < 4; ++q)
        atomicOr(a.error + group * 4 + q, kPackedDomainError);
    admitted[group] = valid ? kHighPaid : 14;
    packable = valid;
    dst[kWeightWords + 16] = valid ? 1u : 0u;
  }
  __syncthreads();
  if (!packable)
    for (int word = t; word < (logical_nk - a.seed + 31) / 32;
         word += blockDim.x)
      reinterpret_cast<uint4*>(
          page.words)[int64_t(group) * page.word_count + word] =
          make_uint4(0, 0, 0, 0);
}

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
    const __grid_constant__ BitmapArgs page, const uint32_t* records,
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
  initialize_screen<8, 256, 256, 128>(s, warp, mq, mp, mk, ms, mr);
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
      produce_keys<256>(s, mk, ms, mr, a.seed, first, blocks);
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
      issue_screen_mma(s.raw_k[stage], s.fine, wg * 128,
                       s.tmem + wg * kTileColumns, runtime);
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
    // Reload immutable NK after register donation.
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
namespace f2_h22_late_host {

void prepare_live(std::vector<Tensor> x, Tensor error, Tensor record,
                  Tensor admitted, Tensor words, int64_t seed, Tensor live_nk) {
  DeviceGuard guard(x[0].get_device_index());
  auto [a, page] = dosa_emit_host::bind(x, error, seed, words);
  launch<f2_h22_late::prepare_kernel, 128>(
      a.nq / 4, a, reinterpret_cast<uint32_t*>(record.mutable_data_ptr<int>()),
      admitted.mutable_data_ptr<int>(), page,
      live_nk.mutable_data_ptr<int32_t>());
}
void emit_fast_live(std::vector<Tensor> x, Tensor error, int64_t seed,
                    Tensor words, Tensor record, Tensor live_nk) {
  auto [a, pg] = dosa_emit_host::bind(x, error, seed, words);
  DeviceGuard guard(x[0].get_device_index());
  auto maps = dosa_emit_host::key_maps(x, 256);
  auto mq = dosa_screen::fp8_map(x[5], f2_h22_late::kTileColumns);
  auto mp = dosa_screen::make_map(
      record.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT32,
      {f2_h22_late::kRecordWords, (cuuint64_t)a.nq / 4},
      {f2_h22_late::kRecordWords * 4}, {f2_h22_late::kRecordWords, 1},
      CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<f2_h22_late::packed_kernel, 384, sizeof(f2_h22_late::PackedStorage),
         true>(dim3(a.nq / 4, 8), a, mq, mp, maps.keys, maps.scales, maps.rho,
               pg, reinterpret_cast<uint32_t*>(record.mutable_data_ptr<int>()),
               live_nk.mutable_data_ptr<int32_t>());
}
}  // namespace f2_h22_late_host

int64_t packed_record_words() { return f2_h22_late::kRecordWords; }

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("packed_record_words", "() -> int", CompositeExplicitAutograd,
              packed_record_words)
DOSA_REGISTER("prepare_fp16_h22_live_k",
              "(Tensor[] x, Tensor! error, Tensor! record, Tensor! admitted, "
              "Tensor! words, int seed, "
              "Tensor live_nk) -> ()",
              CUDA, f2_h22_late_host::prepare_live)
DOSA_REGISTER("emit_fp16_candidates_live_k",
              "(Tensor[] x, Tensor! error, int seed, Tensor! words, Tensor "
              "record, Tensor live_nk) -> ()",
              CUDA, f2_h22_late_host::emit_fast_live)
