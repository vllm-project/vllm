#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace dosa_screen {
struct IndexedFamilyWork {
  bool admitted;
  int group, logical_nk, active_word_count, first_block, blocks;
};

// Full resident waves have one CTA per active Q8 group. Only the residual
// groups are split over K. width=0 denotes the ordinary rectangular schedule.
__device__ __forceinline__ int whole_groups(int count, int64_t width) {
  return width > 0 && width <= count ? count - count % int(width) : 0;
}
__device__ __forceinline__ int3 locate_work(int index, int bulk, int count,
                                            int parts) {
  if (index < bulk) return make_int3(index, 0, 1);
  const int i = index - bulk, tail = count - bulk;
  return make_int3(bulk + i % tail, i / tail, parts);
}

template <int Paid, int Tile, class ValidBudget>
__device__ __forceinline__ IndexedFamilyWork
admit_indexed_family(const Args& a, const int32_t* live_nk, int64_t width,
                     ValidBudget valid_budget) {
  int t = threadIdx.x, logical_nk = __ldg(live_nk);
  if (logical_nk < a.seed || logical_nk > a.nk) {
    if (blockIdx.x == 0)
      for (int r = t; r < a.nq; r += blockDim.x) atomicOr(a.error + r, 1);
    return {};
  }
  int active_word_count = (logical_nk - a.seed + 31) / 32;
  int total_blocks = (logical_nk - a.seed + Tile - 1) / Tile;
  int count = __ldg(a.counts), parts = __ldg(a.counts + 1), groups = a.nq / 8;
  const int bulk = whole_groups(count, width);
  const int64_t tasks = bulk + (int64_t(count) - bulk) * parts;
  bool malformed = count < 0 || count > groups || parts < 1 ||
                   parts > max(total_blocks, 1) || tasks > gridDim.x;
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
  if (work >= tasks) return {};
  // Adjacent CTAs stream the same key partition for distinct query groups.
  const int3 loc = locate_work(work, bulk, count, parts);
  const int pos = loc.x, part = loc.y;
  parts = loc.z;
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
  if (!eligible || !total_blocks) return {};
  return {.admitted = true,
          .group = g,
          .logical_nk = logical_nk,
          .active_word_count = active_word_count,
          .first_block = first_block,
          .blocks = blocks};
}
// Both indexed families advance the same RAW/MMA credit pipeline. An F1 tile
// contains two K128 halves; H18 contains one.
template <int Columns, int Epochs, class Storage>
__device__ __forceinline__ void issue_indexed_blocks(Storage& s, int blocks) {
  constexpr int RawSlots = std::extent_v<decltype(Storage::raw_k)>;
  constexpr int Halves = sizeof(s.raw_k[0]) / (128 * 128);
  auto instruction =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, 128, Columns, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
  const bool leader = cute::elect_one_sync();
  s.full_q.wait(0);
  for (int block = 0; block < blocks; ++block) {
    const int stage = block % RawSlots, u = block % Epochs;
    s.full_kv[stage].wait((block / RawSlots) & 1);
    s.empty_umma[u].wait(((block / Epochs) & 1) ^ 1);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    if (leader) {
#pragma unroll
      for (int h = 0; h < Halves; ++h)
        issue_four_d32(s.raw_k[stage], s.fine, h * 128,
                       s.tmem + (u * Halves + h) * Columns, runtime);
      deep_gemm::ptx::umma_arrive_no_elect(s.full_umma[u]);
    }
    __syncwarp();
  }
}
// All lanes vote. A finite owner-published SAFE implies the fixed plan and
// live keys have been validated. A failed owner row has SAFE=-infinity.
template <bool Prepared>
__device__ __forceinline__ uint32_t keep_ballot(float value, float tau,
                                                bool visible, bool valid) {
  if constexpr (Prepared) {
    return __ballot_sync(0xffffffffu, visible && !(value < tau));
  } else {
    uint32_t ballot;
    asm volatile(
        "{.reg .pred finite,below,active,valid,keep; .reg .f32 magnitude; "
        "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
        "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; "
        "setp.ne.u32 valid,%4,0; and.pred keep,keep,valid; not.pred keep,keep; "
        "setp.ne.u32 active,%3,0; and.pred keep,keep,active; "
        "vote.ballot.sync.b32 %0,keep,0xffffffff;}"
        : "=r"(ballot)
        : "f"(value), "f"(tau), "r"(uint32_t(visible)), "r"(uint32_t(valid)));
    return ballot;
  }
}
}  // namespace dosa_screen

namespace f1_abs_qfma {
using namespace dosa_screen;
struct alignas(1024) K256FourWGStorage {
  alignas(1024) uint8_t fine[8 * 16 * 128];
  alignas(16) float fw[8][16];
  alignas(1024) uint8_t raw_k[4][256 * 128];
  float raw_scale[4][256];
  uint32_t raw_rho[4][256];
  Param row[8];
  Barrier full_q, full_kv[4], empty_kv[4], full_umma[2], empty_umma[2];
  uint32_t tmem;
  uint32_t span_blocks;
};

// Four RN chains are covered by the direct planner's 80-round envelope.
__device__ __forceinline__ void h16_abs_core(const float* acc,
                                             const float weights[2][16],
                                             float core[2]) {
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
}
template <bool Prepared>
__global__ __launch_bounds__(640, 1) void k256_four_math_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ BitmapArgs page, const int32_t* live_nk,
    int64_t width) {
  constexpr int H = 16;
  int t = threadIdx.x, warp = t / 32, lane = t & 31;
  auto work = admit_indexed_family<14, 256>(
      a, live_nk, width, [] __device__(int budget) {
        return budget == 12 || dosa_geometry::valid_certificate_budget(budget);
      });
  if (!work.admitted) return;
  int g = work.group, active_word_count = work.active_word_count,
      first_block = work.first_block, blocks = work.blocks;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<K256FourWGStorage*>(memory);
  if (t == 0) s.span_blocks = blocks;
  if (t < 8) s.row[t] = row_parameters<16, false>(a, g * 8 + t);
  initialize_screen<16, 512, 512, 512>(s, warp, mq, mw, mk, ms, mr);
  if (warp >= 16) cutlass::arch::warpgroup_reg_dealloc<32>();
  if (warp == 16) {
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, 128, 128>(&mq, &s.full_q, s.fine, 0, g * 128);
      deep_gemm::tma::copy<16, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * 16 * 128 + 8 * 16 * 4);
      produce_keys<256>(s, mk, ms, mr, a.seed, first_block, blocks);
    }
  } else if (warp == 17) {
    issue_indexed_blocks<128, 2>(s, blocks);
  } else if (warp >= 18) {
  } else {
    cutlass::arch::warpgroup_reg_alloc<112>();
    // Reload immutable NK after register donation to avoid spills.
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
    uint32_t visible_start[2], visible_span[2];
    s.full_q.wait(0);
    const bool eligible = a.budget[group] == 14;
#pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
      Param& p = resident[qr];
      p = s.row[wg * 2 + qr];
      const int end = min(p.end, math_nk);
      visible_start[qr] = uint32_t(p.start);
      // Unsigned subtraction represents even the full signed endpoint span.
      visible_span[qr] =
          eligible && end > p.start ? uint32_t(end) - visible_start[qr] : 0u;
      p.tau = p.hint == 2 ? p.tau : -INFINITY;
#pragma unroll
      for (int h = 0; h < 16; ++h)
        weights[qr][h] =
            h == 0 ? -resident[qr].lo0
                   : (h == 1 ? -resident[qr].lo1
                             : __fmul_rn(0.5f, deep_gemm::ptx::ld_shared(
                                                   &s.fw[wg * 2 + qr][h])));
    }
    uint64_t word_address = reinterpret_cast<uint64_t>(page.words) +
                            (uint64_t(group) * uint64_t(page.word_count) +
                             uint64_t(first_block) * 8u + uint64_t(warp & 3)) *
                                16u +
                            uint64_t(channel + lane) * 4u;
    // Read the admitted span after register donation.
    const int math_blocks = deep_gemm::ptx::ld_shared(&s.span_blocks);
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
      {
        float first_acc[32];
        load_tmem<32>(s.tmem + u * 256 + wg * 32, first_acc);
        cutlass::arch::fence_view_async_tmem_load();
        h16_abs_core(first_acc, weights, first_core);
      }
      // Retire both TMEM halves before granting512credits.
      float second_acc[32];
      load_tmem<32>(s.tmem + u * 256 + 128 + wg * 32, second_acc);
      cutlass::arch::fence_view_async_tmem_load();
      deep_gemm::ptx::tcgen05_before_thread_sync();
      s.empty_umma[u].arrive();
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        float scale = scales[kh];
        uint32_t code = codes[kh];
        float core[2];
        if (kh == 0) {
          core[0] = first_core[0];
          core[1] = first_core[1];
        } else
          h16_abs_core(second_acc, weights, core);
        __half_raw half = __nv_cvt_fp8_to_halfraw(
            static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
        float rho = __half2float(static_cast<__half>(half));
        int key = a.seed + (first_block + block) * 256 + kh * 128 + kt;
        const uint32_t qualified = code < 127u;
        uint32_t ballots[2];
#pragma unroll
        for (int qr = 0; qr < 2; ++qr) {
          const Param& p = resident[qr];
          float signed_score = core[qr];
          float tail = __fmaf_ru(rho, p.guard, __fadd_ru(signed_score, p.raw));
          float value = __fmaf_ru(scale, tail, p.out);
          const bool visible =
              uint32_t(key) - visible_start[qr] < visible_span[qr];
          ballots[qr] = keep_ballot<Prepared>(value, p.tau, visible, qualified);
        }
        int word = (first_block + block) * 8 + kh * 4 + (warp & 3);
        unsigned value = lane == 0 ? ballots[0] : ballots[1];
        if (lane < 2 && eligible && word < math_word_count)
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
template <int Heads, int Tile, int Threads, class Storage, auto Raw,
          auto Checked>
void emit_indexed(std::vector<Tensor> x, Tensor error, int64_t seed,
                  Tensor words, Tensor work_ids, Tensor work_inverse,
                  Tensor work_config, int64_t launch_grid, Tensor live_nk,
                  int64_t width, bool prepared) {
  auto [a, pg] = dosa_emit_host::bind(x, error, seed, words);
  const int nq = a.nq;
  DeviceGuard guard(x[0].get_device_index());
  a.ids = work_ids.mutable_data_ptr<int>();
  a.members = work_inverse.mutable_data_ptr<int>();
  a.counts = work_config.mutable_data_ptr<int>();
  auto mq = dosa_screen::fp8_map(x[5], 8 * Heads);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {64, (cuuint64_t)nq},
      {256}, {(Heads + 3) / 4 * 4, 8}, CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, Tile);
  auto invoke = [&]<auto Kernel>() {
    launch<Kernel, Threads, sizeof(Storage), Threads == 640>(
        int(launch_grid), a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
        live_nk.mutable_data_ptr<int32_t>(), width);
  };
  if (prepared)
    invoke.template operator()<Checked>();
  else
    invoke.template operator()<Raw>();
}

namespace f2_live_h18 {
using namespace dosa_screen;
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
template <bool Prepared>
__global__ __launch_bounds__(384, 1) void f2_live_h18_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ BitmapArgs page, const int32_t* live_nk,
    int64_t width) {
  constexpr int H = 18, N = 8 * H, W = (H + 3) / 4 * 4, Stages = 3;
  int t = threadIdx.x, warp = t / 32, lane = t & 31;
  auto work = admit_indexed_family<16, 128>(
      a, live_nk, width, [] __device__(int budget) {
        return dosa_geometry::valid_certificate_budget(budget);
      });
  if (!work.admitted) return;
  int g = work.group, logical_nk = work.logical_nk,
      active_word_count = work.active_word_count,
      first_block = work.first_block, blocks = work.blocks;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<F2LiveH18Storage*>(memory);
  if (t < 8) s.row[t] = row_parameters<H, true>(a, g * 8 + t);
  initialize_screen<8, 512, 256, 256, Stages>(s, warp, mq, mw, mk, ms, mr);
  if (warp == 8) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (cute::elect_one_sync()) {
      deep_gemm::tma::copy<128, N, 128>(&mq, &s.full_q, s.fine, 0, g * N);
      deep_gemm::tma::copy<W, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * H * 128 + 8 * W * 4);
      produce_keys<128>(s, mk, ms, mr, a.seed, first_block, blocks);
    }
  } else if (warp == 9) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    issue_indexed_blocks<N, Stages>(s, blocks);
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
        const bool visible =
            eligible_q[qr] && key < logical_nk && key >= p.start && key < p.end;
        const unsigned ballot = keep_ballot<Prepared>(
            value, p.tau, visible, code < 127u && isfinite(neg));
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

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER("emit_h16_indexed_trusted_live_k",
              "(Tensor[] x, Tensor! error, int seed, Tensor! words, Tensor "
              "work_ids, Tensor work_inverse, "
              "Tensor work_config, int grid, Tensor live_nk, int "
              "resident_width=0, bool validated_plan=False) -> ()",
              CUDA,
              emit_indexed<16, 256, 640, f1_abs_qfma::K256FourWGStorage,
                           f1_abs_qfma::k256_four_math_kernel<false>,
                           f1_abs_qfma::k256_four_math_kernel<true>>)
DOSA_REGISTER("emit_h18_indexed_trusted_live_k",
              "(Tensor[] x, Tensor! error, int seed, Tensor! words, Tensor "
              "work_ids, Tensor work_inverse, "
              "Tensor work_config, int grid, Tensor live_nk, int "
              "resident_width=0, bool validated_plan=False) -> ()",
              CUDA,
              emit_indexed<18, 128, 384, f2_live_h18::F2LiveH18Storage,
                           f2_live_h18::f2_live_h18_kernel<false>,
                           f2_live_h18::f2_live_h18_kernel<true>>)
