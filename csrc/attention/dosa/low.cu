#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace dosa_screen {
template <int Width>
CUtensorMap plan_map(const Tensor& q, bool pitched) {
  if (!pitched) return fp8_map(q, 8 * Width);
  return make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  {128, Width, cuuint64_t(q.size(0))},
                  {128, cuuint64_t(q.stride(0))}, {128, Width, 8},
                  CU_TENSOR_MAP_SWIZZLE_128B);
}

template <int Width>
__device__ __forceinline__ void load_plan_q(
    const cute::TmaDescriptor& map, Barrier* ready, uint8_t* fine, int group,
    bool pitched) {
  if (pitched)
    deep_gemm::tma::copy<128, 8 * Width, 128, uint8_t, true>(
        &map, ready, fine, 0, 0, 1, group * 8);
  else
    deep_gemm::tma::copy<128, 8 * Width, 128>(
        &map, ready, fine, 0, group * 8 * Width);
}

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

template <int Paid, int Tile, bool Cold, class ValidBudget>
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
  const int full_blocks = (logical_nk - a.seed + Tile - 1) / Tile;
  int first_tile = 0;
  if constexpr (Cold) first_tile = __ldg(a.counts + 2);
  const bool bad_window = first_tile < 0 || first_tile > full_blocks;
  int total_blocks = bad_window ? 0 : full_blocks - first_tile;
  int count = __ldg(a.counts), parts = __ldg(a.counts + 1), groups = a.nq / 8;
  int bulk = whole_groups(count, width), work = blockIdx.x;
  int64_t tasks = bulk + (int64_t(count) - bulk) * parts;
  bool malformed = bad_window || count < 0 || count > groups || parts < 1 ||
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
      if (!valid_budget(b0) && !dosa_geometry::native_dense_route(b0, a.error + qg * 8))
        for (int j = 0; j < 4; ++j) atomicOr(a.error + qg * 8 + j, 1);
      if (!valid_budget(b1) && !dosa_geometry::native_dense_route(b1, a.error + qg * 8 + 4))
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
  if (int64_t(blockIdx.x) >= tasks) return {};
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
    if (!valid_budget(b) &&
        !dosa_geometry::native_dense_route(b, a.error + (g * 2 + j) * 4)) {
      if (t < 4) atomicOr(a.error + (g * 2 + j) * 4 + t, 1);
    }
    eligible |= b == Paid;
  }
  if (!eligible || !total_blocks) return {};
  return {.admitted = true,
          .group = g,
          .logical_nk = logical_nk,
          .active_word_count = active_word_count,
          .first_block = first_tile + first_block,
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
constexpr int ProbeTiles = 4;
constexpr int ProbeWords = ProbeTiles * 8;
constexpr size_t ProbeBytes = 2 * ProbeWords * sizeof(uint4) + 2 * sizeof(int);

__device__ __forceinline__ int probe_tile(int index, int total) {
  return (2 * index + 1) * total / (2 * ProbeTiles);
}

template <bool Probe = false, bool Estimate = false,
          bool Cold = false, bool Uniform = false>
__global__ __launch_bounds__(640, 1) void k256_four_math_kernel(
    const __grid_constant__ Args a,
    const __grid_constant__ cute::TmaDescriptor mq,
    const __grid_constant__ cute::TmaDescriptor mw,
    const __grid_constant__ cute::TmaDescriptor mk,
    const __grid_constant__ cute::TmaDescriptor ms,
    const __grid_constant__ cute::TmaDescriptor mr,
    const __grid_constant__ BitmapArgs page, const int32_t* live_nk,
    int64_t width, int32_t* out_sample_ids) {
  static_assert(!Estimate || Probe);
  static_assert(!Cold || !Probe);
  static_assert(!Uniform || !Probe);
  constexpr int H = 16;
  int t = threadIdx.x, warp = t / 32, lane = t & 31;
  int g, first_block, blocks;
  int probe_full = 0;
  if constexpr (Probe) {
    g = blockIdx.x * (Estimate ? 8 : 1);
    const int live = __ldg(live_nk);
    probe_full = (live - a.seed) / 256;
    if constexpr (Estimate) {
      if (g >= a.nq / 8 || live < 262144 || live < a.seed || live > a.nk)
        return;
      if (a.budget[2 * g] != 14 && a.budget[2 * g + 1] != 14) return;
    } else {
      if (live < 262144 || a.budget[2 * g] != 14 || a.budget[2 * g + 1] != 14)
        return;
    }
    first_block = 0;
    blocks = ProbeTiles;
  } else {
    auto work = admit_indexed_family<14, 256, Cold>(
        a, live_nk, width, [] __device__(int budget) {
          return budget == 12 || dosa_geometry::valid_certificate_budget(budget);
        });
    if (!work.admitted) return;
    g = work.group;
    first_block = work.first_block;
    blocks = work.blocks;
  }
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<K256FourWGStorage*>(memory);
  auto* sampled = reinterpret_cast<uint4*>(memory + sizeof(K256FourWGStorage));
  if constexpr (Estimate)
    if (t < 2 * ProbeWords) sampled[t] = make_uint4(0, 0, 0, 0);
  if (t == 0) s.span_blocks = blocks;
  if (t < 8) {
    s.row[t] = row_parameters<16, false>(a, g * 8 + t);
  }
  if constexpr (Uniform)
    initialize_screen<16, 512, 512, 512>(s, warp, mq, mw, mk);
  else
    initialize_screen<16, 512, 512, 512>(s, warp, mq, mw, mk, ms, mr);
  if (warp >= 16) cutlass::arch::warpgroup_reg_dealloc<32>();
  if (warp == 16) {
    if (cute::elect_one_sync()) {
      load_plan_q<16>(mq, &s.full_q, s.fine, g, a.pitched_plan);
      deep_gemm::tma::copy<16, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * 16 * 128 + 8 * 16 * 4);
      if constexpr (Probe) {
#pragma unroll
        for (int i = 0; i < ProbeTiles; ++i) {
          s.empty_kv[i].wait(1);
          const int offset = a.seed + 256 * probe_tile(i, probe_full);
          deep_gemm::tma::copy<128, 256, 128>(&mk, &s.full_kv[i], s.raw_k[i], 0,
                                             offset);
          deep_gemm::tma::copy<256, 1, 0>(&ms, &s.full_kv[i], s.raw_scale[i], offset, 0);
          deep_gemm::tma::copy<256, 1, 0>(&mr, &s.full_kv[i], s.raw_rho[i], offset, 0);
          s.full_kv[i].arrive_and_expect_tx(256 * 128 + 2 * 256 * 4);
        }
      } else {
        produce_keys<256, Uniform>(s, mk, ms, mr, a.seed, first_block, blocks);
      }
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
      p.tau = p.hint == 2 ? p.tau
                         : (Uniform ? __uint_as_float(0x7fc00000u) : -INFINITY);
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
      float scales[2];
      uint32_t codes[2];
      // With keys only, the issuer has waited for K before signaling full_umma.
      if constexpr (!Uniform) {
        s.full_kv[stage].wait(phase);
        scales[0] = deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]);
        scales[1] = deep_gemm::ptx::ld_shared(&s.raw_scale[stage][128 + kt]);
        codes[0] = deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]);
        codes[1] = deep_gemm::ptx::ld_shared(&s.raw_rho[stage][128 + kt]);
      }
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
        float scale = 0.f;
        uint32_t code = 0;
        if constexpr (!Uniform) {
          scale = scales[kh];
          code = codes[kh];
        }
        float core[2];
        if (kh == 0) {
          core[0] = first_core[0];
          core[1] = first_core[1];
        } else
          h16_abs_core(second_acc, weights, core);
        float rho = 0.f;
        if constexpr (!Uniform) rho = decode_rho(code);
        const int physical_block = Probe ? probe_tile(block, probe_full) : first_block + block;
        int key = a.seed + physical_block * 256 + kh * 128 + kt;
        uint32_t ballots[2];
#pragma unroll
        for (int qr = 0; qr < 2; ++qr) {
          const Param& p = resident[qr];
          float signed_score = core[qr];
          if constexpr (Uniform) {
            // The owner moves only keys inside every row's common prefix.
            ballots[qr] = __ballot_sync(0xffffffffu, !(signed_score <= p.tau));
          } else {
            const bool visible =
                uint32_t(key) - visible_start[qr] < visible_span[qr];
            float tail = __fmaf_ru(rho, p.guard, __fadd_ru(signed_score, p.raw));
            float value = __fmaf_ru(scale, tail, p.out);
            ballots[qr] = __ballot_sync(0xffffffffu, visible && !(value < p.tau));
          }
        }
        int word = physical_block * 8 + kh * 4 + (warp & 3);
        unsigned value = lane == 0 ? ballots[0] : ballots[1];
        if (lane < 2 && eligible && word < math_word_count) {
          if constexpr (Probe) {
            const int sample_word = (group & 1) * ProbeWords + block * 8 +
                                    kh * 4 + (warp & 3);
            reinterpret_cast<unsigned*>(sampled)[sample_word * 4 + channel + lane] =
                value;
          } else {
            *reinterpret_cast<uint32_t*>(word_address + uint64_t(kh) * 64u) =
                value;
          }
        }
      }
      word_address += 128u;
    }
  }
  __syncthreads();
  if constexpr (Probe) {
    auto* totals = reinterpret_cast<int*>(sampled + 2 * ProbeWords);
    if (t < 2 * ProbeWords) {
      const uint4 p = sampled[t];
      const unsigned common = (p.x & p.y) | (p.x & p.z) | (p.x & p.w) |
                              (p.y & p.z) | (p.y & p.w) | (p.z & p.w);
      const int work = (Estimate ? 1 : 4) * __popc(common) + __popc(p.x & ~common) +
                       __popc(p.y & ~common) + __popc(p.z & ~common) +
                       __popc(p.w & ~common);
      const int total = __reduce_add_sync(0xffffffffu, work);
      if (lane == 0) totals[warp] = total;
    }
    __syncthreads();
    if constexpr (Estimate) {
      if (t == 0) atomicAdd(out_sample_ids, totals[0] + totals[1]);
    } else {
      constexpr int DenseWork = 3 * (4 * ProbeTiles * 256) / 4;
      if (totals[0] >= DenseWork && totals[1] >= DenseWork) {
        if (t < 2) a.budget[g * 2 + t] = dosa_geometry::NativeDenseBudget;
        if (t < 8) atomicOr(a.error + g * 8 + t, dosa_geometry::NativeDenseStatus);
      }
    }
  }
  if (warp == 17) cute::TMEM::Allocator1Sm().free(s.tmem, 512);
}

#if DOSA_SCORE_HEADS == 32
void probe_native_dense(std::vector<Tensor> x, Tensor error, int64_t seed,
                        Tensor words, Tensor live_nk, Tensor budget) {
  auto [a, page] = dosa_emit_host::bind(x, error, seed, words);
  a.budget = budget.mutable_data_ptr<int>();
  DeviceGuard guard(x[0].get_device_index());
  auto mq = dosa_screen::plan_map<16>(x[5], a.pitched_plan);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      {64, cuuint64_t(a.nq)}, {256}, {16, 8}, CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, 256);
  launch<k256_four_math_kernel<true>, 640,
         sizeof(K256FourWGStorage) + ProbeBytes, true>(
      int(a.nq / 8), a, mq, mw, maps.keys, maps.scales, maps.rho, page,
      live_nk.const_data_ptr<int32_t>(), 0, nullptr);
}

// Sample candidate IDs from every eighth physical Q8. The caller clears the
// scalar; unlike probe_native_dense this never changes routes or error state.
void estimate_f1_pressure(std::vector<Tensor> x, Tensor error, int64_t seed,
                          Tensor words, Tensor live_nk, Tensor out_sample_ids) {
  auto [a, page] = dosa_emit_host::bind(x, error, seed, words);
  DeviceGuard guard(x[0].get_device_index());
  auto mq = dosa_screen::plan_map<16>(x[5], a.pitched_plan);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      {64, cuuint64_t(a.nq)}, {256}, {16, 8}, CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, 256);
  launch<k256_four_math_kernel<true, true>, 640,
         sizeof(K256FourWGStorage) + ProbeBytes, true>(
      (a.nq / 8 + 7) / 8, a, mq, mw, maps.keys, maps.scales, maps.rho, page,
      live_nk.const_data_ptr<int32_t>(), 0,
      out_sample_ids.mutable_data_ptr<int32_t>());
}
#endif
}  // namespace f1_abs_qfma

template <int Heads, int Tile, int Threads, class Storage, auto Kernel>
void emit_indexed(std::vector<Tensor> x, Tensor error, int64_t seed,
                  Tensor words, Tensor work_ids, Tensor work_inverse,
                  Tensor work_config, int64_t launch_grid, Tensor live_nk,
                  int64_t width) {
  auto [a, pg] = dosa_emit_host::bind(x, error, seed, words);
  const int nq = a.nq;
  DeviceGuard guard(x[0].get_device_index());
  a.ids = work_ids.mutable_data_ptr<int>();
  a.members = work_inverse.mutable_data_ptr<int>();
  a.counts = work_config.mutable_data_ptr<int>();
  auto mq = dosa_screen::plan_map<Heads>(x[5], a.pitched_plan);
  auto mw = dosa_screen::make_map(
      x[7].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {64, (cuuint64_t)nq},
      {256}, {(Heads + 3) / 4 * 4, 8}, CU_TENSOR_MAP_SWIZZLE_NONE);
  auto maps = dosa_emit_host::key_maps(x, Tile);
  if constexpr (Heads == 16)
    launch<Kernel, Threads, sizeof(Storage), Threads == 640>(
        int(launch_grid), a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
        live_nk.mutable_data_ptr<int32_t>(), width, nullptr);
  else
    launch<Kernel, Threads, sizeof(Storage), Threads == 640>(
        int(launch_grid), a, mq, mw, maps.keys, maps.scales, maps.rho, pg,
        live_nk.mutable_data_ptr<int32_t>(), width);
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
template <bool Cold = false, bool Uniform = false>
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
  auto work = admit_indexed_family<16, 128, Cold>(
      a, live_nk, width, [] __device__(int budget) {
        return dosa_geometry::valid_certificate_budget(budget);
      });
  if (!work.admitted) return;
  int g = work.group, logical_nk = work.logical_nk,
      active_word_count = work.active_word_count,
      first_block = work.first_block, blocks = work.blocks;
  extern __shared__ __align__(1024) uint8_t memory[];
  auto& s = *reinterpret_cast<F2LiveH18Storage*>(memory);
  if (t < 8) {
    s.row[t] = row_parameters<H, true>(a, g * 8 + t);
  }
  if constexpr (Uniform)
    initialize_screen<8, 512, 256, 256, Stages>(s, warp, mq, mw, mk);
  else
    initialize_screen<8, 512, 256, 256, Stages>(s, warp, mq, mw, mk, ms, mr);
  if (warp == 8) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (cute::elect_one_sync()) {
      load_plan_q<H>(mq, &s.full_q, s.fine, g, a.pitched_plan);
      deep_gemm::tma::copy<W, 8, 0>(&mw, &s.full_q, &s.fw[0][0], 0, g * 8);
      s.full_q.arrive_and_expect_tx(8 * H * 128 + 8 * W * 4);
      produce_keys<128, Uniform>(s, mk, ms, mr, a.seed, first_block, blocks);
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
      float scale = Uniform ? 0.0f : deep_gemm::ptx::ld_shared(&s.raw_scale[stage][kt]);
      uint32_t code = Uniform ? 0u : deep_gemm::ptx::ld_shared(&s.raw_rho[stage][kt]);
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
      float rho = Uniform ? 0.0f : decode_rho(code);
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
        unsigned ballot;
        if constexpr (Uniform) {
          ballot = __ballot_sync(0xffffffffu, eligible_q[qr] && !(signed_score <= p.tau));
        } else {
          float tail = __fmaf_ru(rho, p.guard, __fadd_ru(signed_score, p.raw));
          float value = __fmaf_ru(scale, tail, p.out);
          const bool visible =
              eligible_q[qr] && key < logical_nk && key >= p.start && key < p.end;
          ballot = __ballot_sync(0xffffffffu, visible && !(value < p.tau));
        }
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


void emit_low(std::vector<Tensor> x, Tensor error, int64_t seed,
              Tensor words, Tensor work_ids, Tensor work_inverse,
              Tensor work_config, int64_t launch_grid, Tensor live_nk,
              int64_t width, int64_t paid, bool offset, bool uniform) {
  auto invoke = [&]<bool Offset, bool Uniform>() {
    if (paid == 14)
      emit_indexed<16, 256, 640, f1_abs_qfma::K256FourWGStorage,
                   f1_abs_qfma::k256_four_math_kernel<false, false, Offset, Uniform>>(
          x, error, seed, words, work_ids, work_inverse, work_config,
          launch_grid, live_nk, width);
    else
      emit_indexed<18, 128, 384, f2_live_h18::F2LiveH18Storage,
                   f2_live_h18::f2_live_h18_kernel<Offset, Uniform>>(
          x, error, seed, words, work_ids, work_inverse, work_config,
          launch_grid, live_nk, width);
  };
  if constexpr (dosa_geometry::Heads == 32) {
    if (offset) {
      if (uniform) invoke.template operator()<true, true>();
      else invoke.template operator()<true, false>();
      return;
    }
  }
  invoke.template operator()<false, false>();
}
}  // namespace DOSA_NAMESPACE

DOSA_REGISTER(
    "emit_indexed",
    "(Tensor[] x, Tensor! error, int seed, Tensor! words, Tensor ids, "
    "Tensor inverse, Tensor config, int grid, Tensor live_nk, "
    "int width, int paid, bool offset, bool uniform) -> ()",
    CUDA, emit_low)

#if DOSA_SCORE_HEADS == 32
DOSA_REGISTER(
    "probe_native_dense",
    "(Tensor[] x, Tensor! error, int seed, Tensor words, Tensor live_nk, "
    "Tensor! budget) -> ()",
    CUDA, f1_abs_qfma::probe_native_dense)
DOSA_REGISTER(
    "estimate_f1_pressure",
    "(Tensor[] x, Tensor error, int seed, Tensor words, Tensor live_nk, "
    "Tensor! out_sample_ids) -> ()",
    CUDA, f1_abs_qfma::estimate_f1_pressure)
#endif
