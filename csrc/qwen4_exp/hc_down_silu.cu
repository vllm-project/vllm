// Fused HyperConnection down+inject GEMM + SiLU for Qwen4Exp on SM100:
//   acc = xn @ w_down^T  (fp32; K = 10240, N = 324)
//   lowrank = bf16(silu(acc[:, :320] / HC)), inj_out = bf16(acc[:, 320:324])
//
// One tcgen05 kernel template for M > 48:
// - SPLITK > 1 (decode, mid-M): 1-wave cluster split-K, reduced over DSMEM.
// - SPLITK == 1 (prefill): persistent 2-SM with L2-aware xn cache hints.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../core/registration.h"
#include "../torch_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int UMMA_K = 16;

constexpr long kEvictNormal = 0x1000000000000000L;
constexpr long kEvictFirst = 0x12F0000000000000L;

// ---- PTX wrappers; kCtas fills cta_group::%N through an "n" operand ----

__device__ __forceinline__ int smem_addr(const void* p) {
  return static_cast<int>(__cvta_generic_to_shared(p));
}

template <int kCtas>
__device__ __forceinline__ int leader_cta_addr(const void* p) {
  return kCtas == 2 ? (smem_addr(p) & ~(1 << 24)) : smem_addr(p);
}

// Not volatile: PTX elects the same lane for the same membermask every time,
// so the compiler may CSE or hoist it.
__device__ __forceinline__ int elect_sync() {
  int pred;
  asm("{\n"
      ".reg .pred p;\n"
      "elect.sync _|p, 0xffffffff;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(pred));
  return pred;
}

__device__ __forceinline__ void bar_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : : "memory");
}

__device__ __forceinline__ void barrier_cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" : : : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_addr(bar)), "r"(count)
               : "memory");
}

// Every arrive here is relaxed: none publishes data written by the arriving
// thread. TMA and st.async publish theirs through complete_tx (a cluster-scope
// release), and TMEM loads are ordered by tcgen05.wait::ld +
// tcgen05.fence::before_thread_sync. (A .release.cluster arrive would also
// cost a MEMBAR.ALL.GPU.)
template <int kCtas>
__device__ __forceinline__ void mbarrier_arrive_relaxed(uint64_t* bar) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
               :
               : "r"(leader_cta_addr<kCtas>(bar))
               : "memory");
}

// `issue` (here and below) is a PTX predicate: the whole warp runs the loop
// and only the elected lane issues, with no branch per iteration.
template <int kCtas>
__device__ __forceinline__ void mbarrier_arrive_expect_tx_relaxed(uint64_t* bar,
                                                                  int tx_bytes,
                                                                  int issue) {
  asm volatile(
      "{\n"
      ".reg .pred q;\n"
      "setp.ne.b32 q, %2, 0;\n"
      "@q mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, "
      "[%0], %1;\n"
      "}\n"
      :
      : "r"(leader_cta_addr<kCtas>(bar)), "r"(tx_bytes), "r"(issue)
      : "memory");
}

// Every wait acquires. The TMEM handoffs would be correct with a relaxed wait
// (the tcgen05 fences order them), but acquire costs nothing here (no MEMBAR,
// same speed) and a relaxed wait on an smem handoff would be a silent race.
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, "
      "0x989680;\n"
      "@!P1 bra LAB_WAIT;\n"
      "}\n"
      :
      : "r"(smem_addr(bar)), "r"(phase)
      : "memory");
}

template <int kCtas>
__device__ __forceinline__ void tmem_alloc(int* dst, int cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
      :
      : "r"(smem_addr(dst)), "r"(cols), "n"(kCtas)
      : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::%0.sync.aligned;"
               :
               : "n"(kCtas)
               : "memory");
}

template <int kCtas>
__device__ __forceinline__ void tmem_free(int tmem_addr, int cols) {
  asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
               :
               : "r"(tmem_addr), "r"(cols), "n"(kCtas)
               : "memory");
}

template <int kCtas>
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* tm,
                                            int c0, int c1, uint64_t* bar,
                                            long hint, int issue) {
  asm volatile(
      "{\n"
      ".reg .pred q;\n"
      "setp.ne.b32 q, %7, 0;\n"
      "@q cp.async.bulk.tensor.2d.cta_group::%6.shared::cluster.global"
      ".mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;\n"
      "}\n"
      :
      : "r"(smem_addr(dst)), "l"(tm), "r"(c0), "r"(c1),
        "r"(leader_cta_addr<kCtas>(bar)), "l"(hint), "n"(kCtas), "r"(issue)
      : "memory");
}

// K-major 128B-swizzle SMEM descriptor (SBO = 8*128B, LBO = 0, version = 1).
// Only the low word depends on the address, so the math stays 32-bit.
__device__ __forceinline__ long make_sw128_desc(int addr) {
  constexpr long kHi = ((8 * 128) >> 4) | (1 << 14) | (2 << 29);
  return (kHi << 32) | uint32_t(addr >> 4);  // zero-extend into the high word
}

constexpr int make_bf16_idesc(int mma_m, int mma_n) {
  return (1 << 4) | (1 << 7) | (1 << 10) | ((mma_n >> 3) << 17) |
         ((mma_m >> 4) << 24);
}

template <int kCtas>
__device__ __forceinline__ void tcgen05_mma(int tmem, long ad, long bd,
                                            int idesc, int accum, int issue) {
  asm volatile(
      "{\n"
      ".reg .pred p, q;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "setp.ne.b32 q, %6, 0;\n"
      "@q tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n"
      "}\n"
      :
      : "r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(accum), "n"(kCtas),
        "r"(issue));
}

template <int kCtas>
__device__ __forceinline__ void tcgen05_commit(uint64_t* bar,
                                               uint16_t mcast_mask, int issue) {
  if constexpr (kCtas == 2) {
    asm volatile(
        "{\n"
        ".reg .pred q;\n"
        "setp.ne.b32 q, %2, 0;\n"
        "@q tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
        ".multicast::cluster.b64 [%0], %1;\n"
        "}\n"
        :
        : "r"(smem_addr(bar)), "h"(mcast_mask), "r"(issue)
        : "memory");
  } else {
    asm volatile(
        "{\n"
        ".reg .pred q;\n"
        "setp.ne.b32 q, %1, 0;\n"
        "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
        ".b64 [%0];\n"
        "}\n"
        :
        : "r"(smem_addr(bar)), "r"(issue)
        : "memory");
  }
}

template <int N>
__device__ __forceinline__ void tmem_load(int addr, float (&v)[N]) {
  if constexpr (N == 16) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
          "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]),
          "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]),
          "=f"(v[15])
        : "r"(addr));
  } else {
    static_assert(N == 8);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
        "[%8];"
        : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
          "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
        : "r"(addr));
  }
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ int mapa(const void* p, int cta_rank) {
  int addr;
  asm("mapa.shared::cluster.u32 %0, %1, %2;"
      : "=r"(addr)
      : "r"(smem_addr(p)), "r"(cta_rank));
  return addr;
}

// 16 B remote store; on arrival, complete_tx(16) on `bar` in the same CTA.
// `addr` and `bar` are shared::cluster addresses (mapa).
__device__ __forceinline__ void st_async_f32x4(int addr, float4 v, int bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32"
      " [%0], {%1, %2, %3, %4}, [%5];"
      :
      : "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void stg_l1na_b32x4(void* ptr, uint4 v) {
  asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void stg_l1na_b32x8(void* ptr, uint4 v0, uint4 v1) {
  asm volatile(
      "st.global.L1::no_allocate.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, "
      "%8};"
      :
      : "l"(ptr), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w), "r"(v1.x),
        "r"(v1.y), "r"(v1.z), "r"(v1.w)
      : "memory");
}

// Columns n < R go through SiLU into lowrank; R <= n < DN go to inj_out.
template <int N, int HC, int R>
__device__ __forceinline__ void store_silu_chunk(nv_bfloat16* lowrank,
                                                 nv_bfloat16* inj_out, int m,
                                                 int col0,
                                                 const float (&vals)[N]) {
  constexpr int DN = R + HC;
  const int64_t lr_row = int64_t(m) * R;
  // silu(x) = x * sigmoid(x) = h + h * tanh(h), with x = acc / HC, h = x / 2.
  auto silu_hc = [](float acc) {
    const float h = acc * (0.5f / HC);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
    return fmaf(h, t, h);
  };

  if (col0 + N <= R) {
    if constexpr (N == 16) {
      uint4 o0, o1;
      nv_bfloat16* p0 = reinterpret_cast<nv_bfloat16*>(&o0);
      nv_bfloat16* p1 = reinterpret_cast<nv_bfloat16*>(&o1);
#pragma unroll
      for (int j = 0; j < 8; ++j) p0[j] = __float2bfloat16(silu_hc(vals[j]));
#pragma unroll
      for (int j = 0; j < 8; ++j)
        p1[j] = __float2bfloat16(silu_hc(vals[8 + j]));
      // 32 B store needs 32 B alignment; BM = 64 warps 2-3 start at BN / 2,
      // which is 8 mod 16 for BN = 176. Two 16 B stores everywhere: 0.3-0.5%
      // slower at M = 8192-16384.
      if ((col0 & 15) == 0) {
        stg_l1na_b32x8(lowrank + lr_row + col0, o0, o1);
      } else {
        stg_l1na_b32x4(lowrank + lr_row + col0, o0);
        stg_l1na_b32x4(lowrank + lr_row + col0 + 8, o1);
      }
    } else if constexpr (N == 8) {
      uint4 o0;
      nv_bfloat16* p0 = reinterpret_cast<nv_bfloat16*>(&o0);
#pragma unroll
      for (int j = 0; j < 8; ++j) p0[j] = __float2bfloat16(silu_hc(vals[j]));
      stg_l1na_b32x4(lowrank + lr_row + col0, o0);
    } else {
      static_assert(N == 4);
      uint2 o0;
      nv_bfloat16* p0 = reinterpret_cast<nv_bfloat16*>(&o0);
#pragma unroll
      for (int j = 0; j < 4; ++j) p0[j] = __float2bfloat16(silu_hc(vals[j]));
      *reinterpret_cast<uint2*>(lowrank + lr_row + col0) = o0;
    }
    return;
  }

#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int n = col0 + j;
    if (n < R) {
      lowrank[lr_row + n] = __float2bfloat16(silu_hc(vals[j]));
    } else if (n < DN) {
      inj_out[int64_t(m) * HC + (n - R)] = __float2bfloat16(vals[j]);
    }
  }
}

// ---- Unified kernel config & implementation ----

template <int HC_, int H_, int R_>
struct HcShape {
  static constexpr int HC = HC_, H = H_, HD = HC * H, R = R_, DN = R + HC;
};

template <typename Shape_, int BM_, int BN_, int BK_, int SPLITK_, int STAGES_,
          int kCtas_, int kAccStages_>
struct DownCfg {
  using Shape = Shape_;
  static constexpr int HC = Shape::HC, HD = Shape::HD;
  static constexpr int R = Shape::R, DN = Shape::DN;
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int SPLITK = SPLITK_, STAGES = STAGES_;
  static constexpr int kCtas = kCtas_, kAccStages = kAccStages_;
  static constexpr bool kPersistent = (SPLITK == 1);

  // Warps 0-3: epilogue (warp 0 also allocates TMEM); 4: MMA; 5: TMA w_down;
  // 6: TMA xn.
  static constexpr int kThreads = 7 * 32;

  static constexpr int kBRows = BN / kCtas;  // w_down rows loaded per CTA
  static constexpr int TOTAL_K = HD / BK;
  static constexpr int SUB_K = BK / 64;  // 64-wide K sub-tiles (128B swizzle)
  static constexpr int NUM_N_TILES = (DN + BN - 1) / BN;
  // BM = 64: the accumulator is split across two TMEM column halves.
  static constexpr int TMEM_STRIDE = (BM == 64) ? (BN / 2) : BN;
  static constexpr int TMEM_TOTAL = kAccStages * TMEM_STRIDE;
  static constexpr int TMEM_COLS =
      TMEM_TOTAL <= 64
          ? 64
          : (TMEM_TOTAL <= 128 ? 128 : (TMEM_TOTAL <= 256 ? 256 : 512));
  static constexpr int ROWS_PER_CTA = (BM + SPLITK - 1) / SPLITK;
  // Split-K partials: epi[SPLITK][BN / 4][kEpiRows][4]. Rows of a 4-column
  // group are contiguous, so the rows one store instruction sends to an owner
  // form one run (remote stores need >= 64 B runs for full DSMEM bandwidth).
  // An odd row count keeps the reduction's column-fastest LDS.128
  // conflict-free.
  static constexpr int kEpiRows = ROWS_PER_CTA | 1;
  static constexpr int kEpiFloats =
      SPLITK > 1 ? SPLITK * (BN / 4) * kEpiRows * 4 : 1;

  static_assert((BM == 64 || BM == 128) && (BK == 64 || BK == 128));
  static_assert(BN % 16 == 0 && BN % kCtas == 0 && TMEM_TOTAL <= 512);

  struct SharedStorage {
    alignas(1024) nv_bfloat16 a[STAGES][BM * BK];
    alignas(1024) nv_bfloat16 b[STAGES][kBRows * BK];
    alignas(1024) float epi[kEpiFloats];  // split-K partials
    alignas(8) uint64_t tma_full[STAGES];
    alignas(8) uint64_t tma_empty[STAGES];
    alignas(8) uint64_t tmem_full[kAccStages];
    alignas(8) uint64_t tmem_empty[kPersistent ? kAccStages : 1];
    alignas(8) uint64_t epi_full;  // split-K: all partials of our rows landed
    int tmem_base;
  };
};

template <typename Cfg>
__global__ void __launch_bounds__(Cfg::kThreads, 1)
    hc_down_silu_kernel(const __grid_constant__ CUtensorMap tm_xn,
                        const __grid_constant__ CUtensorMap tm_w_down,
                        nv_bfloat16* __restrict__ lowrank,
                        nv_bfloat16* __restrict__ inj_out, int M,
                        bool xn_fits_l2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK;
  constexpr int SPLITK = Cfg::SPLITK, STAGES = Cfg::STAGES;
  constexpr int kCtas = Cfg::kCtas, kAccStages = Cfg::kAccStages;
  constexpr int kBRows = Cfg::kBRows, TMEM_STRIDE = Cfg::TMEM_STRIDE;
  constexpr int ROWS_PER_CTA = Cfg::ROWS_PER_CTA;
  constexpr bool kPersistent = Cfg::kPersistent;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<typename Cfg::SharedStorage*>(smem_raw);

  const int warp = __shfl_sync(0xffffffffu, int(threadIdx.x) >> 5, 0);
  const int lane = int(threadIdx.x) & 31;
  // 1-D grid and cluster. rank = K slice * kCtas + pair rank: the pair rank
  // varies fastest because a 2-SM MMA pair needs ranks differing in bit 0.
  constexpr int kClusterSize = kCtas * SPLITK;
  const int rank = int(blockIdx.x) % kClusterSize;
  const int cta_m_rank = rank % kCtas;
  const int kt = rank / kCtas;  // split-K slice (0 when persistent)
  const int cluster_id = int(blockIdx.x) / kClusterSize;

  // Persistent: N-fastest tile order.
  const int num_m_pairs = (M + BM * kCtas - 1) / (BM * kCtas);
  const int num_tiles = kPersistent ? (num_m_pairs * Cfg::NUM_N_TILES) : 1;
  const int first_tile = kPersistent ? cluster_id : 0;
  const int tile_step = kPersistent ? (int(gridDim.x) / kClusterSize) : 1;

  const int kb_begin = kPersistent ? 0 : (kt * Cfg::TOTAL_K) / SPLITK;
  const int kb_end =
      kPersistent ? Cfg::TOTAL_K : ((kt + 1) * Cfg::TOTAL_K) / SPLITK;
  const int k_blocks = kb_end - kb_begin;

  // Split-K: this CTA reduces tile rows row_base .. row_base + my_rows.
  const int row_base = kt * ROWS_PER_CTA;
  const int my_rows = max(0, min(ROWS_PER_CTA, BM - row_base));

  auto advance = [](int& stage, int& phase) {
    if (++stage == STAGES) {
      stage = 0;
      phase ^= 1;
    }
  };
  auto tile_coords = [&](int t, int& m0, int& nt) {
    if constexpr (kPersistent) {
      const int mp = t / Cfg::NUM_N_TILES;
      m0 = mp * (BM * kCtas) + cta_m_rank * BM;
      nt = t % Cfg::NUM_N_TILES;
    } else {
      // One tile per cluster, N tile fastest: clusters launched back to back
      // read the same xn rows.
      const int mg = cluster_id / Cfg::NUM_N_TILES;
      nt = cluster_id % Cfg::NUM_N_TILES;
      m0 = mg * (BM * kCtas) + cta_m_rank * BM;
    }
  };

  // ---- Setup: barriers and tensor-map prefetch ----
  // Named barrier ids (0 is __syncthreads), each for warps 0-4.
  constexpr int BAR_TMEM_ALLOC = 1;  // warp 0 publishes the TMEM base
  constexpr int BAR_TMEM_FREE = 2;   // 1-SM split-K: TMEM loads done
  if (warp == 4 && elect_sync()) {
    for (int i = 0; i < STAGES; ++i) {
      // Each CTA of the pair arms its own bytes on the leader CTA's barrier.
      mbarrier_init(&sm.tma_full[i], 2 * kCtas);  // 2 TMA warps per CTA
      mbarrier_init(&sm.tma_empty[i], 1);
    }
    for (int i = 0; i < kAccStages; ++i) {
      mbarrier_init(&sm.tmem_full[i], 1);
      if constexpr (kPersistent) mbarrier_init(&sm.tmem_empty[i], kCtas * 128);
    }
    if constexpr (!kPersistent) {
      // Every split-K CTA st.asyncs its partial of our rows (all BN columns).
      mbarrier_init(&sm.epi_full, 1);
      mbarrier_arrive_expect_tx_relaxed<1>(&sm.epi_full,
                                           SPLITK * my_rows * BN * 4, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
  } else if (warp == 5 && elect_sync()) {
    asm volatile("prefetch.tensormap [%0];" : : "l"(&tm_xn));
    asm volatile("prefetch.tensormap [%0];" : : "l"(&tm_w_down));
  }
  if constexpr (kCtas == 2) {
    // The pair signals each other's barriers from here on.
    barrier_cluster_arrive_relaxed();
    barrier_cluster_wait_acquire();
  } else {
    __syncthreads();
    if constexpr (!kPersistent) {
      // Split-K peers store into our epi later: announce we are running, and
      // wait for the peers only right before our first remote store.
      barrier_cluster_arrive_relaxed();
    }
  }

  if (warp >= 5) {
    // ---- TMA (warp 5: w_down, warp 6: xn). Only xn is written by the
    // previous kernel, so only warp 6 waits on PDL.
    const bool is_x = warp == 6;
    if (is_x) cudaGridDependencySynchronize();
    const CUtensorMap* tm = is_x ? &tm_xn : &tm_w_down;
    nv_bfloat16* dst = is_x ? sm.a[0] : sm.b[0];
    const int stage_elems = is_x ? BM * BK : kBRows * BK;
    const int box_elems = is_x ? BM * 64 : kBRows * 64;

    const int leader = elect_sync();
    int stage = 0;
    int phase = 0;
    for (int t = first_tile; t < num_tiles; t += tile_step) {
      int m0, nt;
      tile_coords(t, m0, nt);
      const int row0 = is_x ? m0 : nt * BN + cta_m_rank * kBRows;
      // xn stays in L2 (EVICT_NORMAL) when it fits, for the other N tiles.
      const bool keep_x = xn_fits_l2 && nt != 0;
      const long hint = (is_x && !keep_x) ? kEvictFirst : kEvictNormal;
      for (int kb = 0; kb < k_blocks; ++kb) {
        mbarrier_wait(&sm.tma_empty[stage], phase ^ 1);
        __syncwarp();  // lanes leave the wait together (the elect is hoisted)
        mbarrier_arrive_expect_tx_relaxed<kCtas>(&sm.tma_full[stage],
                                                 stage_elems * 2, leader);
  #pragma unroll
        for (int sub = 0; sub < Cfg::SUB_K; ++sub) {
          tma_load_2d<kCtas>(dst + stage * stage_elems + sub * box_elems, tm,
                             (kb_begin + kb) * BK + sub * 64, row0,
                             &sm.tma_full[stage], hint, leader);
        }
        advance(stage, phase);
      }
    }
    // Drain: wait for the last MMA commit to each stage, so none is still in
    // flight to this CTA's barriers when it exits.
    for (int i = 0; i < STAGES; ++i) {
      mbarrier_wait(&sm.tma_empty[stage], phase ^ 1);
      advance(stage, phase);
    }
    if constexpr (kCtas == 1 && !kPersistent) barrier_cluster_wait_acquire();

  } else if (warp == 4) {
    // ---- MMA (warp 4): the leader CTA issues for the pair ----
    bar_sync(BAR_TMEM_ALLOC, 5 * 32);
    tcgen05_fence_after_thread_sync();
    const int tmem_base = sm.tmem_base;
    if (cta_m_rank == 0) {
      constexpr int idesc = make_bf16_idesc(BM * kCtas, BN);
      const uint16_t mcast_mask = static_cast<uint16_t>(0x3 << (2 * kt));

      const int leader = elect_sync();
      int stage = 0, iter = 0;
      int phase = 0;
      for (int t = first_tile; t < num_tiles; t += tile_step, ++iter) {
        const int acc = iter % kAccStages;
        const int tmem_acc = tmem_base + acc * TMEM_STRIDE;
        if constexpr (kPersistent) {
          mbarrier_wait(&sm.tmem_empty[acc], ((iter / kAccStages) & 1) ^ 1);
          __syncwarp();
          tcgen05_fence_after_thread_sync();
        }

        for (int kb = 0; kb < k_blocks; ++kb) {
          mbarrier_wait(&sm.tma_full[stage], phase);
          __syncwarp();
          tcgen05_fence_after_thread_sync();
  #pragma unroll
          for (int sub = 0; sub < Cfg::SUB_K; ++sub) {
            const int a = smem_addr(sm.a[stage] + sub * BM * 64);
            const int b = smem_addr(sm.b[stage] + sub * kBRows * 64);
  #pragma unroll
            for (int k = 0; k < 64 / UMMA_K; ++k) {
              tcgen05_mma<kCtas>(tmem_acc, make_sw128_desc(a + 32 * k),
                                 make_sw128_desc(b + 32 * k), idesc,
                                 (kb | sub | k) > 0, leader);
            }
          }
          tcgen05_commit<kCtas>(&sm.tma_empty[stage], mcast_mask, leader);
          advance(stage, phase);
        }
        tcgen05_commit<kCtas>(&sm.tmem_full[acc], mcast_mask, leader);
      }
      if constexpr (kPersistent) {
        // Drain: wait for the epilogue's last release of each accumulator, so
        // no peer arrive is still in flight when this CTA exits.
        for (int i = 0; i < kAccStages; ++i, ++iter) {
          mbarrier_wait(&sm.tmem_empty[iter % kAccStages],
                        ((iter / kAccStages) & 1) ^ 1);
        }
      }
    }
    if constexpr (kCtas == 1 && !kPersistent) barrier_cluster_wait_acquire();

  } else {
    // ---- Epilogue (warps 0-3): TMEM -> SiLU -> global (split-K: -> DSMEM).
    // Warp 0 allocates TMEM after gdc_wait: the previous kernel may still
    // hold it.
    if (warp == 0) {
      cudaGridDependencySynchronize();
      tmem_alloc<kCtas>(&sm.tmem_base, Cfg::TMEM_COLS);
    }
    tcgen05_fence_before_thread_sync();
    bar_sync(BAR_TMEM_ALLOC, 5 * 32);
    tcgen05_fence_after_thread_sync();
    const int tmem_base = sm.tmem_base;
    // Warp w can only access TMEM lanes 32w..32w+31.
    const int row = (BM == 64) ? ((warp & 1) * 32 + lane) : (warp * 32 + lane);
    const int col_off = (BM == 64) ? ((warp >> 1) * (BN / 2)) : 0;
    constexpr int COLS_PER_WARP = (BM == 64) ? (BN / 2) : BN;
    constexpr int N16 = COLS_PER_WARP / 16;

    // Split-K: this row's partial goes to the cluster rank that reduces it,
    // into the slot for column group col_off / 4 of K slice kt.
    int epi_dst = 0, epi_bar = 0;
    if constexpr (!kPersistent) {
      const int owner_kt = row / ROWS_PER_CTA;
      const int r_local = row % ROWS_PER_CTA;
      const int peer_rank = cta_m_rank + kCtas * owner_kt;
      float* epi_row = sm.epi + (kt * (BN / 4) * Cfg::kEpiRows + r_local) * 4;
      epi_dst = mapa(epi_row + col_off * Cfg::kEpiRows, peer_rank);
      epi_bar = mapa(&sm.epi_full, peer_rank);
    }
    if constexpr (kCtas == 1 && !kPersistent) barrier_cluster_wait_acquire();

    int iter = 0;
    for (int t = first_tile; t < num_tiles; t += tile_step, ++iter) {
      const int acc = iter % kAccStages;
      int m0, nt;
      tile_coords(t, m0, nt);
      const int n0 = nt * BN;
      const int m = m0 + row;

      mbarrier_wait(&sm.tmem_full[acc], (iter / kAccStages) & 1);
      tcgen05_fence_after_thread_sync();
      const int tmem_row = tmem_base + ((warp * 32) << 16) + acc * TMEM_STRIDE;

      // tcgen05.ld is warp-collective: load all lanes, predicate the stores.
      auto handle_chunk = [&]<int N>(int c_off) {
        float v[N];
        tmem_load<N>(tmem_row + c_off, v);
        if constexpr (!kPersistent) {
          // Column group (c_off + j) / 4; groups are kEpiRows * 16 B apart.
  #pragma unroll
          for (int j = 0; j < N; j += 4) {
            st_async_f32x4(epi_dst + (c_off + j) * Cfg::kEpiRows * 4,
                           make_float4(v[j], v[j + 1], v[j + 2], v[j + 3]),
                           epi_bar);
          }
        } else if (m < M) {
          store_silu_chunk<N, Cfg::HC, Cfg::R>(lowrank, inj_out, m,
                                               n0 + col_off + c_off, v);
        }
      };
  #pragma unroll
      for (int c = 0; c < N16; ++c) {
        handle_chunk.template operator()<16>(c * 16);
      }
      if constexpr (COLS_PER_WARP % 16 == 8) {
        handle_chunk.template operator()<8>(N16 * 16);
      }

      if constexpr (kPersistent) {
        // Every epilogue thread of the pair arrives after its TMEM loads.
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive_relaxed<kCtas>(&sm.tmem_empty[acc]);
      }
    }
  }

  if constexpr (kPersistent) {
    // ---- Persistent: the cluster barrier covers the peer's MMAs into our
    // TMEM.
    tcgen05_fence_before_thread_sync();
    barrier_cluster_arrive_relaxed();
    barrier_cluster_wait_acquire();
    tcgen05_fence_after_thread_sync();
    if (warp == 0) tmem_free<kCtas>(sm.tmem_base, Cfg::TMEM_COLS);
  } else {
    // ---- Split-K: TMEM free (warp 4) and the reduction (warps 0-3) proceed
    // independently; the other warps are done.
    tcgen05_fence_before_thread_sync();
    if constexpr (kCtas == 2) {
      barrier_cluster_arrive_relaxed();  // every thread: TMEM loads done
    } else if (warp <= 4) {
      bar_sync(BAR_TMEM_FREE, 5 * 32);  // warp 4 arrives first
    }
    if (warp == 4) {
      if constexpr (kCtas == 2) barrier_cluster_wait_acquire();  // + the peer's
      tcgen05_fence_after_thread_sync();
      tmem_free<kCtas>(sm.tmem_base, Cfg::TMEM_COLS);
    } else if (warp < 4) {
      // ---- Split-K reduction (epilogue warps) of this CTA's ROWS_PER_CTA
      // rows: one float4 of columns per item, a 128-bit smem load per K slice.
      constexpr int V4_PER_ROW = BN / 4;
      int m0, nt;
      tile_coords(0, m0, nt);
      const int n0 = nt * BN;
      mbarrier_wait(&sm.epi_full, 0);

      for (int idx = int(threadIdx.x); idx < my_rows * V4_PER_ROW;
           idx += 4 * 32) {
        const int r_local = idx / V4_PER_ROW;
        const int c4 = (idx % V4_PER_ROW) * 4;
        const int m = m0 + row_base + r_local;
        const int n = n0 + c4;
        if (m < M && n < Cfg::DN) {
          float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  #pragma unroll
          for (int p = 0; p < SPLITK; ++p) {
            const float4 v = *reinterpret_cast<const float4*>(
                sm.epi +
                ((p * V4_PER_ROW + c4 / 4) * Cfg::kEpiRows + r_local) * 4);
            acc[0] += v.x;
            acc[1] += v.y;
            acc[2] += v.z;
            acc[3] += v.w;
          }
          store_silu_chunk<4, Cfg::HC, Cfg::R>(lowrank, inj_out, m, n, acc);
        }
      }
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---- Host launch & analytical roofline + wave-quantization dispatch ----

CUtensorMap make_tensor_map(const void* ptr, uint64_t rows, uint64_t cols,
                            uint64_t ld, int box_rows) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {ld * sizeof(nv_bfloat16)};
  const cuuint32_t box[2] = {64, cuuint32_t(box_rows)};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
      strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(res == CUDA_SUCCESS,
                  "cuTensorMapEncodeTiled failed: ", static_cast<int>(res));
  return map;
}

template <typename Cfg>
double estimate_cost(int64_t M, int64_t sm_count) {
  const int64_t m_groups =
      (M + Cfg::BM * Cfg::kCtas - 1) / (Cfg::BM * Cfg::kCtas);
  const int64_t clusters = m_groups * Cfg::NUM_N_TILES;
  const int64_t csize = Cfg::kCtas * Cfg::SPLITK;

  int64_t waves;
  if constexpr (Cfg::SPLITK > 1) {
    // 1-wave Push-DSMEM: leave 20% headroom for 8-GPC fragmentation when
    // cluster_size >= 6; reject configs that spill past 1 wave.
    const int64_t budget = (csize >= 6) ? (sm_count * 4 / 5) : sm_count;
    if (clusters * csize > budget) return 1e30;
    waves = 1;
  } else {
    const int64_t slots = std::max<int64_t>(1, sm_count / Cfg::kCtas);
    waves = (clusters + slots - 1) / slots;
  }

  // Per-CTA TMA load (xn rows + w_down rows, per K element) bounds every tile
  // here; the MMA side never does. The 80% budget and the split-K penalties
  // are empirical (GB300).
  const double k_work = double((Cfg::HD + Cfg::SPLITK - 1) / Cfg::SPLITK);
  const double tma_work =
      double(Cfg::BM) + double(Cfg::BN) / double(Cfg::kCtas);
  static_assert(Cfg::BM * Cfg::BN / 128 <= Cfg::BM + Cfg::BN / Cfg::kCtas,
                "MMA-bound tile: add an MMA term (BM*BN/128) to the cost");
  double cost = double(waves) * k_work * tma_work;
  if constexpr (Cfg::SPLITK > 1) cost += Cfg::kCtas == 1 ? 1.5e5 : 2.5e5;
  return cost;
}

template <typename Cfg>
void launch(torch::stable::Tensor& lowrank, torch::stable::Tensor& inj_out,
            torch::stable::Tensor const& xn,
            torch::stable::Tensor const& w_down, int64_t M,
            const cudaDeviceProp& prop) {
  const CUtensorMap tm_xn =
      make_tensor_map(xn.data_ptr(), M, Cfg::HD, xn.stride(0), Cfg::BM);
  const CUtensorMap tm_w_down =
      make_tensor_map(w_down.data_ptr(), w_down.size(0), Cfg::HD,
                      w_down.stride(0), Cfg::kBRows);

  auto* kernel = hc_down_silu_kernel<Cfg>;
  constexpr size_t smem = sizeof(typename Cfg::SharedStorage);
  STD_TORCH_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) == cudaSuccess);
  if constexpr (Cfg::kCtas * Cfg::SPLITK > 8) {
    STD_TORCH_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                        1) == cudaSuccess);
  }

  const int64_t m_groups =
      (M + Cfg::BM * Cfg::kCtas - 1) / (Cfg::BM * Cfg::kCtas);
  dim3 grid;
  if constexpr (Cfg::kPersistent) {
    const int64_t num_tiles = m_groups * Cfg::NUM_N_TILES;
    const int64_t grid_pairs = std::max<int64_t>(
        1, std::min<int64_t>(num_tiles, prop.multiProcessorCount / Cfg::kCtas));
    grid = dim3(static_cast<uint32_t>(grid_pairs * Cfg::kCtas));
  } else {
    grid = dim3(static_cast<uint32_t>(m_groups * Cfg::NUM_N_TILES * Cfg::kCtas *
                                      Cfg::SPLITK));
  }

  // xn cache policy: see the TMA warp.
  const int64_t l2_bytes =
      prop.l2CacheSize > 0 ? int64_t(prop.l2CacheSize) : int64_t(64 << 20);
  const bool xn_fits_l2 =
      M * Cfg::HD * int64_t(sizeof(nv_bfloat16)) <= l2_bytes;

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {Cfg::kCtas * Cfg::SPLITK, 1, 1};

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(Cfg::kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = get_current_cuda_stream(xn.get_device_index());
  config.attrs = attrs;
  config.numAttrs = 2;

  STD_TORCH_CHECK(
      cudaLaunchKernelEx(&config, kernel, tm_xn, tm_w_down,
                         reinterpret_cast<nv_bfloat16*>(lowrank.data_ptr()),
                         reinterpret_cast<nv_bfloat16*>(inj_out.data_ptr()),
                         static_cast<int>(M), xn_fits_l2) == cudaSuccess);
}

template <typename... Cfgs>
void dispatch_best(torch::stable::Tensor& lowrank,
                   torch::stable::Tensor& inj_out,
                   torch::stable::Tensor const& xn,
                   torch::stable::Tensor const& w_down, int64_t M,
                   const cudaDeviceProp& prop) {
  using LaunchFn =
      void (*)(torch::stable::Tensor&, torch::stable::Tensor&,
               torch::stable::Tensor const&, torch::stable::Tensor const&,
               int64_t, const cudaDeviceProp&);
  const double costs[] = {estimate_cost<Cfgs>(M, prop.multiProcessorCount)...};
  const LaunchFn fns[] = {&launch<Cfgs>...};
  const int best = static_cast<int>(
      std::min_element(costs, costs + sizeof...(Cfgs)) - costs);
  fns[best](lowrank, inj_out, xn, w_down, M, prop);
}

void qwen4_exp_hc_down_silu(torch::stable::Tensor& lowrank,
                            torch::stable::Tensor& inj_out,
                            torch::stable::Tensor const& xn,
                            torch::stable::Tensor const& w_down) {
  using Shape = HcShape<4, 2560, 320>;
  const auto bf16 = torch::headeronly::ScalarType::BFloat16;
  STD_TORCH_CHECK(lowrank.scalar_type() == bf16 &&
                  inj_out.scalar_type() == bf16 && xn.scalar_type() == bf16 &&
                  w_down.scalar_type() == bf16);
  STD_TORCH_CHECK(lowrank.dim() == 2 && inj_out.dim() == 2 && xn.dim() == 2 &&
                  w_down.dim() == 2);
  const int64_t M = xn.size(0);
  STD_TORCH_CHECK(M <= std::numeric_limits<int>::max());
  STD_TORCH_CHECK(xn.size(1) == Shape::HD && w_down.size(1) == Shape::HD &&
                  w_down.size(0) >= Shape::DN);
  STD_TORCH_CHECK(lowrank.size(0) == M && lowrank.size(1) == Shape::R);
  STD_TORCH_CHECK(inj_out.size(0) == M && inj_out.size(1) == Shape::HC);
  STD_TORCH_CHECK(xn.stride(1) == 1 && (xn.stride(0) % 8 == 0));
  STD_TORCH_CHECK(w_down.is_contiguous() && lowrank.is_contiguous() &&
                  inj_out.is_contiguous());
  const int32_t dev = xn.get_device_index();
  STD_TORCH_CHECK(w_down.get_device_index() == dev &&
                  lowrank.get_device_index() == dev &&
                  inj_out.get_device_index() == dev);
  // TMA needs 16 B aligned bases; lowrank takes up to 32 B vector stores.
  auto aligned = [](const torch::stable::Tensor& t, uintptr_t a) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % a == 0;
  };
  STD_TORCH_CHECK(aligned(xn, 16) && aligned(w_down, 16) &&
                  aligned(lowrank, 32));
  if (M == 0) return;

  // get_device_prop() and cudaFuncSetAttribute act on the current device.
  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  const auto* prop = get_device_prop();
  STD_TORCH_CHECK(prop->major == 10,
                  "qwen4_exp_hc_down_silu requires SM100 (got sm_", prop->major,
                  prop->minor, ")");

  //                      BM   BN   BK  SPLITK STAGES kCtas kAccStages
  dispatch_best<DownCfg<Shape, 128, 48, 128, 16, 4, 1, 1>,
                DownCfg<Shape, 128, 48, 128, 8, 4, 1, 1>,
                DownCfg<Shape, 64, 112, 64, 5, 12, 2, 1>,
                DownCfg<Shape, 64, 112, 64, 2, 12, 2, 1>,
                DownCfg<Shape, 64, 176, 64, 2, 8, 2, 1>,
                DownCfg<Shape, 64, 176, 64, 1, 11, 2, 1>,
                DownCfg<Shape, 128, 112, 64, 1, 9, 2, 2>,
                DownCfg<Shape, 128, 176, 64, 1, 8, 2, 2>>(lowrank, inj_out, xn,
                                                          w_down, M, *prop);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen4_exp_hc_down_silu", TORCH_BOX(&qwen4_exp_hc_down_silu));
}

}  // namespace
