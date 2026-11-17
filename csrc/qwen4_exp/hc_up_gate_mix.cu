// Fused HyperConnection up projection + gate mix for Qwen4Exp on SM100.
//
// out[m, c] = sum_s sigmoid(lowrank[m] . w_up[s * H + c]) *
//             xn[m, s * H + c] / HC
//
// with HC = 4, H = 2560 and rank R = 320. The [M, HC * H] gate is never
// written to global memory and stays in fp32 (the unfused F.linear +
// hc_gate_mix chain rounds it to bf16); the sigmoid uses tanh.approx.
//
// Persistent kernel, one CTA per SM. An output tile is 128 (or, with a CTA
// pair, 256) rows x BC hidden columns; its GEMM tile has N = HC * BC columns,
// gathered from the HC streams of w_up by TMA loads per stage, so all streams
// of an output column sit in one TMEM accumulator row.
//   warp 0: TMA producer (lowrank 128 x 64 and w_up HC x (BC x 64) per stage)
//   warp 1: tcgen05.mma issuer, fp32 accumulators in TMEM (the leader CTA's
//           warp 1 for a CTA pair)
//   warp 2: TMEM allocator
//   warp 3: TMA producer for the xn tile (HC x 128 x BC)
//   warps 4..: epilogue warpgroups of one 16-column chunk each, storing the
//              output through shared memory with one TMA store per
//              warpgroup.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../core/registration.h"
#include "../torch_utils.h"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>

#include <algorithm>
#include <type_traits>
#include <limits>

namespace {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint32_t HC = 4, H = 2560, R = 320;
constexpr uint32_t BLOCK_M = 128, BLOCK_K = 64, UMMA_K = 16;
constexpr uint32_t NUM_K_BLOCKS = R / BLOCK_K;
constexpr uint32_t kChunk = 16;  // output columns per epilogue thread
constexpr uint32_t kEpiWarp0 = 4;

// Tile configurations, picked by M on the host. Small M has few tiles per SM,
// so it favors narrow tiles on single CTAs; large M uses wide tiles on CTA
// pairs (tcgen05.mma.cta_group::2), which halve the w_up tile each CTA holds
// and leave room for deeper pipelines.
template <uint32_t BC_, uint32_t kStages_, uint32_t kAccStages_,
          uint32_t kXStages_, uint32_t kCtas_>
struct Config {
  static constexpr uint32_t BC = BC_, kStages = kStages_,
                            kAccStages = kAccStages_, kXStages = kXStages_;
  // kCtas = 2: a CTA pair runs a 256-row tile; each CTA holds its 128 rows of
  // lowrank and HC / 2 streams of w_up.
  static constexpr uint32_t kCtas = kCtas_;
  static constexpr uint32_t BLOCK_N = HC * BC, NUM_N_BLOCKS = H / BC;
  static constexpr uint32_t kBRows = BLOCK_N / kCtas;
  static constexpr uint32_t kEpiWarpGroups = BC / kChunk;
  static constexpr uint32_t kThreads = 128 + 128 * kEpiWarpGroups;
  static constexpr uint32_t kTmemCols = kAccStages * BLOCK_N;
  static_assert(kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0);
  // xn rows are BC bf16 in shared memory, swizzled by TMA in 16-byte chunks.
  static constexpr uint32_t kXRowBytes = BC * 2;
  static constexpr CUtensorMapSwizzle kXSwizzle =
      kXRowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                        : CU_TENSOR_MAP_SWIZZLE_64B;
  // 16-byte chunk c of row r lands at chunk c ^ ((r >> kXRowShift) & mask).
  static constexpr uint32_t kXSwizzleMask = kXRowBytes / 16 - 1;
  static constexpr uint32_t kXRowShift = kXRowBytes == 128 ? 0 : 1;
  static_assert(kXRowBytes == 64 || kXRowBytes == 128);

  struct SharedStorage {
    alignas(1024) cutlass::bfloat16_t a[kStages][BLOCK_M * BLOCK_K];
    alignas(1024) cutlass::bfloat16_t b[kStages][kBRows * BLOCK_K];
    alignas(1024) cutlass::bfloat16_t x[kXStages][HC * BLOCK_M * BC];
    // Output staging, double-buffered: one BLOCK_M x kChunk block per
    // epilogue warpgroup.
    alignas(1024) cutlass::bfloat16_t o[2][kEpiWarpGroups][BLOCK_M * kChunk];
    Barrier full[kStages], empty[kStages];
    Barrier x_full[kXStages], x_empty[kXStages];
    Barrier tmem_full[kAccStages], tmem_empty[kAccStages];
    uint32_t tmem_base;
  };
};
using SmallM = Config<32, 4, 4, 2, 1>;
using LargeM = Config<64, 4, 2, 1, 2>;
// Measured crossover between the two configurations on B300.
constexpr int64_t kLargeM = 1024;

// K-major operand with 128B swizzle: 8-row atoms of 128 bytes (BLOCK_K bf16),
// stacked 1024 bytes apart along M/N, advanced by k elements along K.
__device__ __forceinline__ uint64_t make_sw128_desc(const void* smem,
                                                    uint32_t k) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ =
      static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  desc.start_address_ = static_cast<uint16_t>(
      (cute::cast_smem_ptr_to_uint(smem) + k * sizeof(cutlass::bfloat16_t)) >>
      4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = (8 * BLOCK_K * sizeof(cutlass::bfloat16_t)) >> 4;
  desc.leading_byte_offset_ = 0;
  return desc.desc_;
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

// Orders this thread's shared memory accesses before later TMA accesses.
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Barrier over the 128 threads of one warpgroup; id 0 is __syncthreads.
__device__ __forceinline__ void warpgroup_bar_sync(uint32_t id) {
  asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

__device__ __forceinline__ uint4 ld_shared_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void st_shared_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5.
__device__ __forceinline__ float sigmoid(float x) {
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}

template <typename Cfg>
__global__ void __launch_bounds__(Cfg::kThreads, 1)
    hc_up_gate_mix_kernel(const __grid_constant__ CUtensorMap tm_lowrank,
                          const __grid_constant__ CUtensorMap tm_w_up,
                          const __grid_constant__ CUtensorMap tm_xn,
                          const __grid_constant__ CUtensorMap tm_out, int M) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  constexpr uint32_t BC = Cfg::BC, BLOCK_N = Cfg::BLOCK_N;
  constexpr uint32_t kStages = Cfg::kStages, kAccStages = Cfg::kAccStages,
                     kXStages = Cfg::kXStages;
  constexpr uint32_t kEpiThreads = 128 * Cfg::kEpiWarpGroups;
  constexpr uint32_t kCtas = Cfg::kCtas;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<typename Cfg::SharedStorage*>(smem_raw);
  // Warp-uniform for the compiler, so role branches compile to uniform code.
  const uint32_t warp = cutlass::canonical_warp_idx_sync(),
                 lane = threadIdx.x % 32;
  // A tile spans kCtas * 128 rows; CTA `rank` of the pair owns 128 of them.
  const uint32_t rank = kCtas == 2 ? cute::block_rank_in_cluster() : 0;
  const uint32_t first_tile = blockIdx.x / kCtas, tile_step = gridDim.x / kCtas;
  const uint32_t num_m_blocks = (M + BLOCK_M * kCtas - 1) / (BLOCK_M * kCtas);
  const uint32_t num_tiles = num_m_blocks * Cfg::NUM_N_BLOCKS;

  if (warp == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tm_lowrank);
    cute::prefetch_tma_descriptor(&tm_w_up);
    cute::prefetch_tma_descriptor(&tm_xn);
    cute::prefetch_tma_descriptor(&tm_out);
  } else if (warp == 1 && cute::elect_one_sync()) {
    for (uint32_t i = 0; i < kStages; ++i) {
      sm.full[i].init(1);
      sm.empty[i].init(1);
    }
    for (uint32_t i = 0; i < kXStages; ++i) {
      sm.x_full[i].init(1);
      sm.x_empty[i].init(kEpiThreads);
    }
    for (uint32_t i = 0; i < kAccStages; ++i) {
      sm.tmem_full[i].init(1);
      // With a CTA pair, the leader's barrier gets one arrival per epilogue
      // warp of both CTAs.
      sm.tmem_empty[i].init(kCtas == 1 ? kEpiThreads
                                       : kCtas * kEpiThreads / 32);
    }
    cutlass::arch::fence_barrier_init();
  }
  if constexpr (kCtas == 2) {
    // The peer arrives on the leader's barriers: both must be initialized.
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  }
  // lowrank and xn are produced by the preceding kernels. Allocate TMEM only
  // after they finish: a preceding kernel that triggers its dependents before
  // allocating TMEM itself would otherwise wait on us while we wait on it.
  cudaGridDependencySynchronize();
  using TmemAllocator = std::conditional_t<kCtas == 2, cute::TMEM::Allocator2Sm,
                                           cute::TMEM::Allocator1Sm>;
  if (warp == 2) {
    TmemAllocator().allocate(Cfg::kTmemCols, &sm.tmem_base);
    TmemAllocator().release_allocation_lock();
  }
  tcgen05_before_thread_sync();
  __syncthreads();
  tcgen05_after_thread_sync();
  const uint32_t tmem_base = sm.tmem_base;

  // Tiles are M-fastest so that concurrent CTAs share a w_up tile. lowrank and
  // w_up are re-read by other tiles; xn is read once.
  if (warp == 0) {
    if (cute::elect_one_sync()) {
      uint32_t stage = 0, phase = 0;
      for (uint32_t t = first_tile; t < num_tiles; t += tile_step) {
        const int32_t m0 =
            (t % num_m_blocks) * BLOCK_M * kCtas + rank * BLOCK_M;
        const int32_t c0 = (t / num_m_blocks) * BC;
        for (uint32_t kb = 0; kb < NUM_K_BLOCKS; ++kb) {
          sm.empty[stage].wait(phase ^ 1);
          auto* full = reinterpret_cast<uint64_t*>(&sm.full[stage]);
          const int32_t k0 = kb * BLOCK_K;
          // With a CTA pair, both CTAs' loads complete on the leader's barrier.
          using TmaLoad =
              std::conditional_t<kCtas == 2, cute::SM100_TMA_2SM_LOAD_2D,
                                 cute::SM90_TMA_LOAD_2D>;
          if (rank == 0) {
            sm.full[stage].arrive_and_expect_tx(
                kCtas * (sizeof(sm.a[0]) + sizeof(sm.b[0])));
          }
          TmaLoad::copy(
              &tm_lowrank, full,
              static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
              sm.a[stage], k0, m0);
  #pragma unroll
          for (uint32_t i = 0; i < HC / kCtas; ++i) {
            const uint32_t s = rank * (HC / kCtas) + i;
            TmaLoad::copy(
                &tm_w_up, full,
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sm.b[stage] + i * BC * BLOCK_K, k0, int32_t(s * H) + c0);
          }
          stage = (stage + 1) % kStages;
          phase ^= stage == 0;
        }
      }
    }
    __syncwarp();
  } else if (warp == 1 && rank == 0) {
    // With a CTA pair, the leader issues the MMAs for both CTAs and its
    // commits arrive on the barriers of both.
    constexpr uint32_t MMA_M = BLOCK_M * kCtas;
    using Mma = std::conditional_t<
        kCtas == 2,
        cute::SM100_MMA_F16BF16_2x1SM_SS<
            cutlass::bfloat16_t, cutlass::bfloat16_t, float, MMA_M, BLOCK_N,
            cute::UMMA::Major::K, cute::UMMA::Major::K>,
        cute::SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                   float, MMA_M, BLOCK_N, cute::UMMA::Major::K,
                                   cute::UMMA::Major::K>>;
    auto commit = [](Barrier& bar) {
      if constexpr (kCtas == 2) {
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&bar), 0x3);
      } else {
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&bar));
      }
    };
    constexpr auto instr =
        cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                    float, MMA_M, BLOCK_N, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>();
    const uint64_t idesc = cute::UMMA::make_runtime_instr_desc<>(instr);
    uint32_t stage = 0, phase = 0, iter = 0;
    for (uint32_t t = first_tile; t < num_tiles; t += tile_step, ++iter) {
      const uint32_t acc = iter % kAccStages;
      sm.tmem_empty[acc].wait(((iter / kAccStages) & 1) ^ 1);
      tcgen05_after_thread_sync();
      for (uint32_t kb = 0; kb < NUM_K_BLOCKS; ++kb) {
        sm.full[stage].wait(phase);
        tcgen05_after_thread_sync();
        // fma and umma_arrive are called by the whole warp and both elect
        // lane 0, so each commit tracks the MMAs issued before it.
  #pragma unroll
        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
          Mma::fma(make_sw128_desc(sm.a[stage], k * UMMA_K),
                   make_sw128_desc(sm.b[stage], k * UMMA_K),
                   tmem_base + acc * BLOCK_N, (kb | k) > 0, idesc);
        }
        commit(sm.empty[stage]);
        __syncwarp();
        stage = (stage + 1) % kStages;
        phase ^= stage == 0;
      }
      commit(sm.tmem_full[acc]);
      __syncwarp();
    }
  } else if (warp == 3) {
    if (cute::elect_one_sync()) {
      uint32_t iter = 0;
      for (uint32_t t = first_tile; t < num_tiles; t += tile_step, ++iter) {
        const uint32_t xs = iter % kXStages;
        sm.x_empty[xs].wait(((iter / kXStages) & 1) ^ 1);
        const int32_t m0 =
            (t % num_m_blocks) * BLOCK_M * kCtas + rank * BLOCK_M;
        const int32_t c0 = (t / num_m_blocks) * BC;
        auto* full = reinterpret_cast<uint64_t*>(&sm.x_full[xs]);
  #pragma unroll
        for (uint32_t s = 0; s < HC; ++s) {
          cute::SM90_TMA_LOAD_2D::copy(
              &tm_xn, full,
              static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST),
              sm.x[xs] + s * BLOCK_M * BC, int32_t(s * H) + c0, m0);
        }
        sm.x_full[xs].arrive_and_expect_tx(sizeof(sm.x[0]));
      }
    }
    __syncwarp();
  } else if (warp >= kEpiWarp0) {
    // Each warpgroup mixes one kChunk-column chunk of every tile, one row per
    // thread. A warp may only access its own 32-lane quarter of TMEM.
    const uint32_t quarter = warp % 4;
    const uint32_t wg = (warp - kEpiWarp0) / 4, j = wg * kChunk;
    const uint32_t row = quarter * 32 + lane;
    const bool store_thread = quarter == 0 && lane == 0;
    uint32_t iter = 0;
    for (uint32_t t = first_tile; t < num_tiles; t += tile_step, ++iter) {
      const uint32_t acc = iter % kAccStages, xs = iter % kXStages;
      const int32_t m0 = (t % num_m_blocks) * BLOCK_M * kCtas + rank * BLOCK_M;
      const int32_t c0 = (t / num_m_blocks) * BC;

      // xn: kChunk columns of this row for every stream, undoing the swizzle.
      sm.x_full[xs].wait((iter / kXStages) & 1);
      uint4 x[HC][kChunk / 8];
  #pragma unroll
      for (uint32_t s = 0; s < HC; ++s) {
  #pragma unroll
        for (uint32_t k = 0; k < kChunk / 8; ++k) {
          const uint32_t chunk =
              (j / 8 + k) ^ ((row >> Cfg::kXRowShift) & Cfg::kXSwizzleMask);
          x[s][k] =
              ld_shared_v4(cute::cast_smem_ptr_to_uint(sm.x[xs]) +
                           (s * BLOCK_M + row) * Cfg::kXRowBytes + chunk * 16);
        }
      }
      // The next TMA load into this stage must not overtake the reads above.
      fence_proxy_async_smem();
      sm.x_empty[xs].arrive();

      sm.tmem_full[acc].wait((iter / kAccStages) & 1);
      tcgen05_after_thread_sync();
      uint32_t g[HC][kChunk];
  #pragma unroll
      for (uint32_t s = 0; s < HC; ++s) {
        uint32_t* v = g[s];
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
            tmem_base + ((quarter * 32) << 16) + acc * BLOCK_N + s * BC + j,
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
            v[11], v[12], v[13], v[14], v[15]);
      }
      cutlass::arch::fence_view_async_tmem_load();

      float o[kChunk] = {};
  #pragma unroll
      for (uint32_t s = 0; s < HC; ++s) {
        const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x[s]);
  #pragma unroll
        for (uint32_t i = 0; i < kChunk / 2; ++i) {
          const float2 xf = __bfloat1622float2(x2[i]);
          o[2 * i] =
              fmaf(sigmoid(__uint_as_float(g[s][2 * i])), xf.x, o[2 * i]);
          o[2 * i + 1] = fmaf(sigmoid(__uint_as_float(g[s][2 * i + 1])), xf.y,
                              o[2 * i + 1]);
        }
      }
      uint4 y[kChunk / 8];
      auto* y2 = reinterpret_cast<__nv_bfloat162*>(y);
  #pragma unroll
      for (uint32_t i = 0; i < kChunk / 2; ++i) {
        y2[i] = __floats2bfloat162_rn(o[2 * i] * (1.0f / HC),
                                      o[2 * i + 1] * (1.0f / HC));
      }

      // Stage the warpgroup's BLOCK_M x kChunk output block and store it with
      // one TMA store, which clips rows past M; one store per warpgroup
      // instead of per warp measured up to 1% faster. Rows are 32 bytes,
      // swizzled like CU_TENSOR_MAP_SWIZZLE_32B to avoid bank conflicts.
      static_assert(kChunk * 2 == 32);
      auto* staged = sm.o[iter & 1][wg];
      const uint32_t o_smem =
          cute::cast_smem_ptr_to_uint(staged + row * kChunk);
  #pragma unroll
      for (uint32_t k = 0; k < kChunk / 8; ++k) {
        st_shared_v4(o_smem + (k ^ ((lane >> 2) & 1)) * 16, y[k]);
      }
      fence_proxy_async_smem();
      // The store thread waits for its earlier stores to finish reading before
      // the barrier, so the buffer written on the next tile is free.
      if (store_thread) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
      warpgroup_bar_sync(1 + wg);
      if (store_thread) {
        cute::SM90_TMA_STORE_2D::copy(&tm_out, staged, c0 + int32_t(j), m0);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
      // Releasing the accumulator before the math above measured 2-4% slower.
      tcgen05_before_thread_sync();
      if constexpr (kCtas == 2) {
        __syncwarp();
        if (lane == 0) sm.tmem_empty[acc].arrive(0u, 1u);  // leader CTA
      } else {
        sm.tmem_empty[acc].arrive();
      }
    }
    if (store_thread) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  // Neither CTA of a pair frees TMEM while the pair may still use it.
  tcgen05_before_thread_sync();
  if constexpr (kCtas == 2) {
    cute::cluster_arrive();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }
  tcgen05_after_thread_sync();
  if (warp == 2) TmemAllocator().free(tmem_base, Cfg::kTmemCols);
#endif
}

PFN_cuTensorMapEncodeTiled_v12000 get_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    STD_TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                    "cuTensorMapEncodeTiled is unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

// Row-major [rows, cols] bf16 tensor with row stride `ld` elements.
CUtensorMap make_tensor_map(const void* ptr, uint64_t rows, uint64_t cols,
                            uint64_t ld, uint32_t box_cols, uint32_t box_rows,
                            CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {ld * sizeof(__nv_bfloat16)};
  const cuuint32_t box[2] = {box_cols, box_rows};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult res = get_encode_fn()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
      strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(res == CUDA_SUCCESS,
                  "cuTensorMapEncodeTiled failed: ", static_cast<int>(res));
  return map;
}

template <typename Cfg>
void launch(const torch::stable::Tensor& out,
            const torch::stable::Tensor& lowrank,
            const torch::stable::Tensor& xn, const torch::stable::Tensor& w_up,
            int64_t M, const cudaDeviceProp* prop) {
  const CUtensorMap tm_lowrank =
      make_tensor_map(lowrank.data_ptr(), M, R, lowrank.stride(0), BLOCK_K,
                      BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tm_w_up =
      make_tensor_map(w_up.data_ptr(), HC * H, R, R, BLOCK_K, Cfg::BC,
                      CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tm_xn = make_tensor_map(
      xn.data_ptr(), M, HC * H, xn.stride(0), Cfg::BC, BLOCK_M, Cfg::kXSwizzle);
  const CUtensorMap tm_out =
      make_tensor_map(out.data_ptr(), M, H, out.stride(0), kChunk, BLOCK_M,
                      CU_TENSOR_MAP_SWIZZLE_32B);

  auto* kernel = hc_up_gate_mix_kernel<Cfg>;
  constexpr size_t smem = sizeof(typename Cfg::SharedStorage) + 1024;
  STD_TORCH_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) == cudaSuccess);
  constexpr uint32_t kCtas = Cfg::kCtas;
  const int64_t num_tiles =
      (M + BLOCK_M * kCtas - 1) / (BLOCK_M * kCtas) * Cfg::NUM_N_BLOCKS;

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = kCtas;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(
      kCtas * std::min<int64_t>(num_tiles, prop->multiProcessorCount / kCtas));
  config.blockDim = dim3(Cfg::kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = get_current_cuda_stream(lowrank.get_device_index());
  config.attrs = attrs;
  config.numAttrs = 2;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, kernel, tm_lowrank, tm_w_up, tm_xn, tm_out, static_cast<int>(M));
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hc_up_gate_mix launch failed: ", cudaGetErrorString(err));
}

}  // namespace

void qwen4_exp_hc_up_gate_mix(torch::stable::Tensor& out,
                              torch::stable::Tensor const& lowrank,
                              torch::stable::Tensor const& xn,
                              torch::stable::Tensor const& w_up) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(lowrank.dim() == 2 && xn.dim() == 2 && w_up.dim() == 2 &&
                  out.dim() == 2);
  const int64_t M = lowrank.size(0);
  STD_TORCH_CHECK(lowrank.size(1) == R && w_up.size(0) == HC * H &&
                      w_up.size(1) == R && xn.size(0) == M &&
                      xn.size(1) == HC * H && out.size(0) == M &&
                      out.size(1) == H,
                  "expected hc_count 4, hidden size 2560 and rank 320");
  STD_TORCH_CHECK(lowrank.scalar_type() == ScalarType::BFloat16 &&
                  xn.scalar_type() == ScalarType::BFloat16 &&
                  w_up.scalar_type() == ScalarType::BFloat16 &&
                  out.scalar_type() == ScalarType::BFloat16);
  STD_TORCH_CHECK(w_up.stride(0) == R && w_up.stride(1) == 1,
                  "w_up must be contiguous");
  // TMA needs unit-stride, 16-byte aligned rows.
  auto check_rows = [](const torch::stable::Tensor& t) {
    STD_TORCH_CHECK(t.stride(1) == 1 && t.stride(0) % 8 == 0 &&
                        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "rows must be unit-stride and 16-byte aligned");
  };
  check_rows(lowrank);
  check_rows(xn);
  check_rows(out);
  STD_TORCH_CHECK(M <= std::numeric_limits<int>::max());
  if (M == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      lowrank.get_device_index());
  const cudaDeviceProp* prop = get_device_prop();
  STD_TORCH_CHECK(prop->major == 10, "requires an SM100-family GPU");
  if (M >= kLargeM) {
    launch<LargeM>(out, lowrank, xn, w_up, M, prop);
  } else {
    launch<SmallM>(out, lowrank, xn, w_up, M, prop);
  }
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen4_exp_hc_up_gate_mix", TORCH_BOX(&qwen4_exp_hc_up_gate_mix));
}
