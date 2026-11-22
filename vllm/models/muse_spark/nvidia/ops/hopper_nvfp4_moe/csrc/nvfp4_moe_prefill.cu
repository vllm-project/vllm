// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Hopper (SM90a) NVFP4 MoE prefill engine (T >= 1537) for Muse Spark:
//   - route_k: token-to-expert bucketing, tile size-class partitioning,
//     per-tile L2 weight eviction tagging, and cost-balanced split-K scheduling
//     across SMs.
//   - gemm_k<0>: W13 NVFP4 GEMM with dual-stage K-loop producer, epilogue ALU
//     overlap, and fused SwiGLU (supports split gate/up global scales).
//   - gemm_k<1>: W2 NVFP4 GEMM with 2D TMA activation loads, epilogue ALU
//     overlap, and PDL overlap.
//   - combine_p: fused weight-first post-expert RMSNorm (dynamic post_eps) and
//     FP32 weighted sum across routed experts with PDL overlap.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "nvfp4_moe_common.cuh"

namespace vllm::muse_spark_hopper_nvfp4_moe::prefill {

using bf16 = __nv_bfloat16;
using torch::Tensor;

// ---------------------------------------------------------------------------
// Architecture constants and per-mode shared-memory layout
// ---------------------------------------------------------------------------

constexpr int kHiddenDim = 4096;
constexpr int kNumKBlocks = 64;     // kHiddenDim / 64 (4 x k16 per K-block)
constexpr int kWeightChunk = 9216;  // 4 sub-tiles x (2048 B FP4 + 256 B E5M3)
constexpr int kSubTileBytes = kWeightChunk / 4;  // 2304 B per 64x64 sub-tile
constexpr int kNumThreads = 384;  // 1 producer WG (128) + 2 consumer WGs
constexpr int kTileNarrow = 128;  // Max narrow-tile N rows
constexpr int kTileWide = 256;    // Max wide-tile N rows
constexpr int kPartSlot = kTileNarrow * 256;  // 32768 FP32 partial elements
constexpr int kSplitSnap = 4;         // Minimum split-K slice in K-blocks
constexpr int kTinyWeightOff = 2048;  // 16 x 128 B activation prefix for N=16
constexpr int kMaxWorkItems = 960;
constexpr int kNumClasses = 8;      // {256, 224, 192, 144, 128, 96, 64, 16}
constexpr int kNumWideClasses = 4;  // Classes 0..3 split M across 2 halves
constexpr int kNumYBoxMaps = 9;     // {16, 64, 96, 128, 144, 192, 224, 256, 32}
constexpr int kNumHBoxMaps = 4;     // {16, 64, 96, 128}
constexpr int kMaxLocalExperts = 64;
constexpr int kExpertMask = 0xFF;
constexpr int kEvictFirstBit = 1 << 8;  // Bit 8 of pinfo.x: last tile of expert
constexpr int kRouteTokensPerCta = 1024;
constexpr int kMaxRouteCtas = 256;
constexpr int kMaxCostTable = 512;

static_assert(kMaxLocalExperts <= kExpertMask + 1,
              "kMaxLocalExperts exceeds kExpertMask range");
static_assert((kEvictFirstBit & kExpertMask) == 0,
              "kEvictFirstBit must not overlap kExpertMask");

template <int MODE>
struct StageCfg {
  static_assert(
      MODE == 0 || MODE == 1,
      "Hopper NVFP4 MoE prefill supports MODE 0 (W13) and MODE 1 (W2)");
  static constexpr int kActNarrowBytes = kTileNarrow * 128;
  static constexpr int kActWideBytes = kTileWide * 128;
  static constexpr int kStageBytes =
      (kActWideBytes + 2 * kSubTileBytes + 1023) / 1024 * 1024;
  static constexpr int kNumStages = (MODE == 1) ? 4 : 5;
  static constexpr int kStagingBytes =
      (MODE == 0) ? (kTileWide * 72 * 2) : (256 * 128 * 2);
  static constexpr int kSmemBytes =
      kNumStages * kStageBytes + kStagingBytes + 2 * kNumStages * 8 + 64 + 1024;
};

static_assert(StageCfg<0>::kSmemBytes + (kMaxWorkItems + 1) * sizeof(int) <=
                  232448,
              "MODE 0 shared memory exceeds Hopper 227 KiB limit");
static_assert(StageCfg<1>::kSmemBytes + (kMaxWorkItems + 1) * sizeof(int) <=
                  232448,
              "MODE 1 shared memory exceeds Hopper 227 KiB limit");

struct GemmParams {
  const uint8_t* w;
  const float* gs;     // Gate scale (MODE=0) or down scale (MODE=1)
  const float* gs_up;  // Optional up scale (MODE=0); nullptr falls back to gs
  const bf16* x;
  const int* rows;
  const int4* pinfo;
  const int2* seg;
  float* part;
  int* flags;
  const int* kdp;
  bf16* out;
  int num_m_blocks;  // 32 for W13, 16 for W2
  size_t expert_stride;

  __device__ __forceinline__ const uint8_t* expert_weights(int e) const {
    return w + static_cast<size_t>(e) * expert_stride;
  }
};

struct XMaps {
  CUtensorMap m[kNumHBoxMaps];
};

struct YMaps {
  CUtensorMap m[kNumYBoxMaps];
};

struct TensorMaps {
  CUtensorMap x256;
  XMaps h;
  YMaps y;
};

// ---------------------------------------------------------------------------
// Small device helpers specific to the prefill engine
// ---------------------------------------------------------------------------

__device__ __forceinline__ void bar_consumers() { named_bar(1, 256); }

__device__ __forceinline__ int cta_id() {
  int id;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(id));
  return id;
}

__device__ __forceinline__ float ex2_approx_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float silu_mul_ex2ftz(float gate, float up) {
  constexpr float kNegLog2E =
      -1.44269504f;  // IEEE-754 f32 0xBFB8AA3B (-log2(e))
  return __fdividef(gate, 1.0f + ex2_approx_ftz(gate * kNegLog2E)) * up;
}

__device__ __forceinline__ void dequant_subtile(uint4 wv, uint2 sv, int j,
                                                uint32_t* out) {
  const uint32_t w = (j == 0) ? wv.x : (j == 1) ? wv.y : (j == 2) ? wv.z : wv.w;
  const uint32_t sw = (j < 2) ? sv.x : sv.y;
  const uint32_t pv = decode_scale_pair(sw, j & 1);
  dequant_fp4x8(w, pv, out);
}

constexpr __device__ __forceinline__ int size_class(int n) {
  return n >= 256   ? 0
         : n >= 224 ? 1
         : n >= 192 ? 2
         : n >= 144 ? 3
         : n >= 128 ? 4
         : n >= 96  ? 5
         : n >= 64  ? 6
                    : 7;
}

__device__ __forceinline__ int menu_size(int count) {
  return count <= 16    ? 16
         : count <= 64  ? 64
         : count <= 96  ? 96
         : count <= 128 ? 128
         : count <= 144 ? 144
         : count <= 192 ? 192
         : count <= 224 ? 224
                        : 256;
}

__device__ __forceinline__ int local_slot(int id, int local_start, int nloc) {
  return static_cast<uint32_t>(id - local_start) < static_cast<uint32_t>(nloc)
             ? (id - local_start)
             : 255;
}

// ---------------------------------------------------------------------------
// Consumer tile execution: WGMMA / MMA mainloop, split-K reduction, epilogue
// ---------------------------------------------------------------------------

template <int MODE, int N>
__device__ __forceinline__ void consume_tile(const GemmParams& p,
                                             uint32_t sbase, uint32_t bar0,
                                             int cw, int tid, int pos0, int mb,
                                             int h, int e, int ks, int ke,
                                             int& s, uint32_t& ph, int na,
                                             const YMaps& ymap) {
  using C = StageCfg<MODE>;
  constexpr bool kTinyTile = (N == 16);
  constexpr int Q = kTinyTile ? 4 : (N <= kTileNarrow ? 2 : 1);

  const int ct = tid - 128;
  const int lt = tid & 127;
  const int wq = lt >> 5;
  const int lane = lt & 31;
  const int g = lane >> 2;

  float acc[Q][N / 2] = {};
  const float sc = __ldg(p.gs + e);
  const float sc_up = (MODE == 0 && p.gs_up) ? __ldg(p.gs_up + e) : sc;

  if constexpr (kTinyTile) {
    const int rr = (lane & 7) + ((lane >> 4) << 3);
    const int ch = (lane >> 3) & 1;
    const bool half = (na <= 8);

#pragma unroll 1
    for (int kb = ks; kb < ke; ++kb) {
      mbar_wait(bar0 + 8 * s, ph);
      const uint32_t xs = sbase + s * C::kStageBytes;
      const uint32_t wb = xs + kTinyWeightOff + cw * kWeightChunk;

      uint4 wv[Q];
      uint2 sv[Q];
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        wv[q] = lds128(wb + q * kSubTileBytes + lt * 16);
        sv[q] = lds64(wb + q * kSubTileBytes + 2048 + (wq * 8 + g) * 8);
      }

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t af[Q][4];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          dequant_subtile(wv[q], sv[q], j, af[q]);
        }

        uint32_t bfr[N / 16][4];
#pragma unroll
        for (int g16 = 0; g16 < N / 16; ++g16) {
          const int r = 16 * g16 + rr;
          ldmatrix_x4(xs + static_cast<uint32_t>(
                               r * 128 + (((2 * j + ch) ^ (r & 7)) << 4)),
                      bfr[g16]);
        }

#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
          for (int nb = 0; nb < N / 8; ++nb) {
            if (nb == 0 || !half) {
              mma16816(&acc[q][4 * nb], af[q], bfr[nb >> 1][2 * (nb & 1)],
                       bfr[nb >> 1][2 * (nb & 1) + 1]);
            }
          }
        }
      }

      __threadfence_block();
      if (lane == 0) {
        mbar_arrive(bar0 + 8 * (C::kNumStages + s));
      }
      ring_next<C::kNumStages>(s, ph);
    }
  } else {
    uint32_t a[2][Q][4];
    int prev = -1;

#pragma unroll 1
    for (int kb = ks; kb < ke; ++kb) {
      mbar_wait(bar0 + 8 * s, ph);
      const uint32_t xs = sbase + s * C::kStageBytes;
      const uint32_t wb =
          (Q == 1) ? (xs + C::kActWideBytes + cw * kSubTileBytes)
                   : (xs + C::kActNarrowBytes + (2 * cw) * kSubTileBytes);

      uint4 wv[Q];
      uint2 sv[Q];
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        wv[q] = lds128(wb + q * kSubTileBytes + lt * 16);
        sv[q] = lds64(wb + q * kSubTileBytes + 2048 + (wq * 8 + g) * 8);
      }

      const uint64_t d0 = desc_swizzle<1024, 1>(xs);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          dequant_subtile(wv[q], sv[q], j, a[j & 1][q]);
        }
        wgmma_fence();
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          wgmma_bf16_rs<N>(acc[q], a[j & 1][q], d0 + 2 * j);
        }
        wgmma_commit();
        wgmma_wait<1>();
        if (j == 0 && prev >= 0 && lane == 0) {
          mbar_arrive(bar0 + 8 * (C::kNumStages + prev));
        }
      }

      prev = s;
      ring_next<C::kNumStages>(s, ph);
    }

    wgmma_wait<0>();
    if (lane == 0) {
      mbar_arrive(bar0 + 8 * (C::kNumStages + prev));
    }
  }

  // Split-K partial publish (for trailing slice CTAs with ks > 0).
  const int b = cta_id();
  if (ks > 0) {
    float4* dst =
        reinterpret_cast<float4*>(p.part + static_cast<size_t>(b) * kPartSlot);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
      for (int m = 0; m < N / 8; ++m) {
        __stcg(dst + (q * (N / 8) + m) * 256 + ct,
               make_float4(acc[q][4 * m], acc[q][4 * m + 1], acc[q][4 * m + 2],
                           acc[q][4 * m + 3]));
      }
    }
    bar_consumers();
    if (ct == 0) {
      st_release(p.flags + b, 1);
    }
    return;
  }

  // Split-K partial reduction (for leading CTA with ks == 0 and ke < 64).
  if (ke < kNumKBlocks) {
    const int t = p.seg[b + 1].x;
#pragma unroll 1
    for (int sp = b + 1;; ++sp) {
      const int2 st0 = p.seg[sp];
      const int2 st1 = p.seg[sp + 1];
      if (st0.x != t) {
        break;
      }
      if (!(st0.x == st1.x && st0.y == st1.y)) {
        if (ct == 0) {
          while (ld_acquire(p.flags + sp) == 0) {
          }
        }
        bar_consumers();
        const float4* src = reinterpret_cast<const float4*>(
            p.part + static_cast<size_t>(sp) * kPartSlot);
        constexpr int kGroupM = 16 / Q;
#pragma unroll
        for (int m0 = 0; m0 < N / 8; m0 += kGroupM) {
          float4 v[Q][kGroupM];
#pragma unroll
          for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int mm = 0; mm < kGroupM; ++mm) {
              if (m0 + mm < N / 8) {
                v[q][mm] = __ldcg(src + (q * (N / 8) + m0 + mm) * 256 + ct);
              }
            }
          }
#pragma unroll
          for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int mm = 0; mm < kGroupM; ++mm) {
              if (m0 + mm < N / 8) {
                const float vals[4] = {v[q][mm].x, v[q][mm].y, v[q][mm].z,
                                       v[q][mm].w};
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                  acc[q][4 * (m0 + mm) + c] += vals[c];
                }
              }
            }
          }
          asm volatile("" ::: "memory");
        }
      }
      if (st1.x != t) {
        break;
      }
    }
  }

  // Epilogue: hoist FP32 scaling, SwiGLU, and BF16 pack ALU ahead of
  // tma_store_wait_read() so prior TMA store completion overlaps with epilogue
  // compute while shrinking the live register set (acc -> packed BF16x2 ra).
  const uint32_t stg_s = sbase + C::kNumStages * C::kStageBytes;
  constexpr int kMapIdx = (N == 32) ? 8 : (7 - size_class(N));
  constexpr int kNI = (N + 31) / 32;

  if constexpr (MODE == 0) {
    uint32_t ra[Q][kNI][4];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
      for (int i = 0; i < kNI; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int jj = 4 * i + k;
          ra[q][i][k] =
              (jj < N / 8)
                  ? pack_bf16x2(silu_mul_ex2ftz(acc[q][4 * jj] * sc,
                                                acc[q][4 * jj + 2] * sc_up),
                                silu_mul_ex2ftz(acc[q][4 * jj + 1] * sc,
                                                acc[q][4 * jj + 3] * sc_up))
                  : 0u;
        }
      }
    }

    if (ct == 0) {
      tma_store_wait_read();
    }
    bar_consumers();

#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int fb = 32 * (Q * cw + q) + 8 * wq;
#pragma unroll
      for (int i = 0; i < kNI; ++i) {
        const int n = 8 * (4 * i + (lane >> 3)) + (lane & 7);
        if (4 * i + 4 <= N / 8) {
          stmatrix_x4_trans(stg_s + static_cast<uint32_t>(
                                        ((fb >> 6) * N + n) * 128 +
                                        ((((fb & 63) >> 3) ^ (n & 7)) << 4)),
                            ra[q][i][0], ra[q][i][1], ra[q][i][2], ra[q][i][3]);
        } else {
          const int n2 = (n & 15) + 32 * i;
          stmatrix_x2_trans(stg_s + static_cast<uint32_t>(
                                        ((fb >> 6) * N + n2) * 128 +
                                        ((((fb & 63) >> 3) ^ (n2 & 7)) << 4)),
                            ra[q][i][0], ra[q][i][1]);
        }
      }
    }
  } else {
    uint32_t ra[Q][2][kNI][4];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
#pragma unroll
        for (int i = 0; i < kNI; ++i) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int jj = 4 * i + kk;
            ra[q][hi][i][kk] =
                (jj < N / 8) ? pack_bf16x2(acc[q][4 * jj + 2 * hi] * sc,
                                           acc[q][4 * jj + 2 * hi + 1] * sc)
                             : 0u;
          }
        }
      }
    }

    if (ct == 0) {
      tma_store_wait_read();
    }
    bar_consumers();

#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int ob = 64 * (Q * cw + q) + 16 * wq + 8 * hi;
#pragma unroll
        for (int i = 0; i < kNI; ++i) {
          const int n = 8 * (4 * i + (lane >> 3)) + (lane & 7);
          const int row = (4 * i + 4 <= N / 8) ? n : ((n & 15) + 32 * i);
          const uint32_t addr =
              stg_s +
              static_cast<uint32_t>((ob >> 6) * N * 128 + row * 128 +
                                    (((ob & 63) ^ ((row & 7) * 8)) * 2));
          if (4 * i + 4 <= N / 8) {
            stmatrix_x4_trans(addr, ra[q][hi][i][0], ra[q][hi][i][1],
                              ra[q][hi][i][2], ra[q][hi][i][3]);
          } else {
            stmatrix_x2_trans(addr, ra[q][hi][i][0], ra[q][hi][i][1]);
          }
        }
      }
    }
  }

  fence_proxy_async();
  bar_consumers();

  if (ct == 0) {
    constexpr int kOutColsPerQ = (MODE == 0) ? 64 : 128;
#pragma unroll
    for (int col = 0; col < Q * kOutColsPerQ; col += 64) {
      tma_store_2d(
          &ymap.m[kMapIdx], stg_s + static_cast<uint32_t>((col / 64) * N * 128),
          2 * kOutColsPerQ * mb + (Q == 1 ? kOutColsPerQ * h : 0) + col, pos0);
    }
    tma_store_commit();
  }
}

// ---------------------------------------------------------------------------
// Persistent warp-specialized GEMM kernel (MODE 0: W13+SwiGLU, MODE 1: W2)
// ---------------------------------------------------------------------------

template <int MODE>
__global__ void __launch_bounds__(kNumThreads, 1)
    gemm_k(const GemmParams p, const __grid_constant__ CUtensorMap xmap256,
           const __grid_constant__ XMaps hmap,
           const __grid_constant__ YMaps ymap) {
  constexpr bool kUseTmaAct = (MODE == 1);
  using C = StageCfg<MODE>;

  extern __shared__ uint8_t smraw[];
  uint8_t* sm =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smraw) + 1023u) &
                                 ~static_cast<uintptr_t>(1023u));
  const uint32_t sbase = smem_u32(sm);
  const uint32_t bar0 =
      sbase + C::kNumStages * C::kStageBytes + C::kStagingBytes;

  const int tid = threadIdx.x;
  __shared__ int wl[kMaxWorkItems];
  __shared__ int nwl;

  if constexpr (MODE == 0) {
    pdl_launch();
  }

  // Warp 0 initializes mbarriers and builds this CTA's tile worklist.
  if (tid < 32) {
    if (tid == 0) {
      for (int s = 0; s < C::kNumStages; ++s) {
        mbar_init(bar0 + 8 * s, kUseTmaAct ? 1u : 129u);
        mbar_init(bar0 + 8 * (C::kNumStages + s), 8u);
      }
      mbar_init_fence();
    }

    const int G = static_cast<int>(gridDim.x);
    const int b = static_cast<int>(blockIdx.x);
    const int2 a0 = p.seg[b];
    const int2 a1 = p.seg[b + 1];
    const int K = p.kdp[0];
    const uint32_t n2 = static_cast<uint32_t>(p.kdp[1]);
    const uint32_t NT = static_cast<uint32_t>(p.kdp[2]);
    const uint32_t MB = static_cast<uint32_t>(p.num_m_blocks);

    auto pack_item = [&](uint32_t pos, int ks, int ke) -> int {
      const uint32_t c2 = pos * n2 / NT;
      const uint32_t c2n = (pos + 1) * n2 / NT;
      int k;
      uint32_t j;
      if (c2n > c2) {
        k = kNumClasses - 1;
        j = c2;
      } else {
        j = pos - c2;
        k = 0;
        while (k < kNumClasses - 2 &&
               j >= static_cast<uint32_t>(p.kdp[3 + k])) {
          j -= static_cast<uint32_t>(p.kdp[3 + k]);
          ++k;
        }
      }
      const uint32_t tpp = (k < kNumWideClasses)
                               ? (2 * MB)
                               : ((k == kNumClasses - 1) ? (MB / 2) : MB);
      const uint32_t pi =
          static_cast<uint32_t>(p.kdp[3 + kNumClasses + k]) + j / tpp;
      const uint32_t sub = j % tpp;
      const uint32_t mb = (k < kNumWideClasses)
                              ? (sub >> 1)
                              : ((k == kNumClasses - 1) ? (2 * sub) : sub);
      const uint32_t h = (k < kNumWideClasses) ? (sub & 1u) : 0u;
      return static_cast<int>((h << 30) | (pi << 18) | (mb << 13) |
                              (static_cast<uint32_t>(ks) << 7) |
                              static_cast<uint32_t>(ke));
    };

    const uint32_t lmask = (1u << tid) - 1u;
    int n = 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
      const int k = k0 + tid;
      const uint32_t pos =
          static_cast<uint32_t>(k * G + ((k & 1) ? (G - 1 - b) : b));
      const bool valid = (k < K) && (pos < NT);
      const uint32_t mask = __ballot_sync(kFullWarpMask, valid);
      const int idx = n + __popc(mask & lmask);
      if (valid && idx < kMaxWorkItems) {
        wl[idx] = pack_item(pos, 0, kNumKBlocks);
      }
      n += __popc(mask);
    }

    for (int t0 = a0.x; t0 <= a1.x; t0 += 32) {
      const int t = t0 + tid;
      const int ks = (t == a0.x) ? a0.y : 0;
      const int ke = (t == a1.x) ? a1.y : kNumKBlocks;
      const bool valid = (t <= a1.x) && (ks < ke);
      const uint32_t mask = __ballot_sync(kFullWarpMask, valid);
      const int idx = n + __popc(mask & lmask);
      if (valid && idx < kMaxWorkItems) {
        wl[idx] = pack_item(static_cast<uint32_t>(t), ks, ke);
      }
      n += __popc(mask);
    }

    if (n >= kMaxWorkItems) {
      __trap();
    }
    if (tid == 0) {
      nwl = n;
    }
  }

  __syncthreads();
  if constexpr (MODE == 1) {
    pdl_wait();
  }

  const int wg = tid >> 7;
  if (wg == 0) {
    // Producer warpgroup (40 registers).
    setmaxnreg_dec<40>();

    const uint32_t sb = (smem_u32(smraw) + 1023u) & ~1023u;
    const uint32_t b0 = sb + C::kNumStages * C::kStageBytes + C::kStagingBytes;
    const int lt = tid;
    const int c = lt & 7;
    const int r0 = lt >> 3;
    const uint32_t dsw = static_cast<uint32_t>((c ^ (r0 & 7)) << 4);
    int s = 0;
    uint32_t ph = 0;

#pragma unroll 1
    for (int it = 0; it < nwl; ++it) {
      const int w = wl[it];
      const int h = (w >> 30) & 1;
      const int mb = (w >> 13) & 31;
      const int ks = (w >> 7) & 63;
      const int ke = w & 127;
      const int4 pf = p.pinfo[(w >> 18) & 4095];
      const int e = pf.x & kExpertMask;
      const bool evict_first = (pf.x & kEvictFirstBit) != 0;
      const int pos0 = pf.y;
      const int N = pf.z;
      const bool wide = (N > kTileNarrow);
      const uint8_t* wsrc =
          p.expert_weights(e) +
          static_cast<size_t>(mb) * (kNumKBlocks * kWeightChunk) +
          (wide ? h * 2 * kSubTileBytes : 0);

      if constexpr (kUseTmaAct) {
        // MODE 1 (W2): warp 0 issues bulk weight + 2D TMA activation loads.
        if (lt < 32) {
          const int ku = __shfl_sync(kFullWarpMask, ks, 0);
          const int keu = __shfl_sync(kFullWarpMask, ke, 0);
          const int p0u = __shfl_sync(kFullWarpMask, pos0, 0);
          const int Nu = __shfl_sync(kFullWarpMask, N, 0);
          const int efu = __shfl_sync(kFullWarpMask, evict_first ? 1 : 0, 0);
          const bool wdu = (Nu > kTileNarrow);
          const uint8_t* wsu = reinterpret_cast<const uint8_t*>(__shfl_sync(
              kFullWarpMask, reinterpret_cast<unsigned long long>(wsrc), 0));
          const uint64_t wpol =
              (efu || Nu == 16) ? policy_evict_first() : policy_evict_normal();

#pragma unroll 1
          for (int kb = ku; kb < keu; ++kb) {
            const uint32_t xs = sb + s * C::kStageBytes;
            const uint32_t fb = b0 + 8 * s;
            const uint8_t* wk = wsu + static_cast<size_t>(kb) * kWeightChunk;
            const int kx = kb * 64;
            asm volatile("" ::"r"(xs), "r"(fb), "l"(wk), "r"(kx));

            mbar_wait(b0 + 8 * (C::kNumStages + s), ph ^ 1u);
            if (elect_one()) {
              if (Nu == 16) {
                mbar_expect_tx(fb, 2 * kWeightChunk + 16 * 128);
                bulk_g2s_hint(xs + kTinyWeightOff, wk, kWeightChunk, fb, wpol);
                bulk_g2s_hint(
                    xs + kTinyWeightOff + kWeightChunk,
                    wk + static_cast<size_t>(kNumKBlocks) * kWeightChunk,
                    kWeightChunk, fb, wpol);
                tma_load_2d(xs, &hmap.m[0], kx, p0u, fb);
              } else if (wdu) {
                mbar_expect_tx(
                    fb, 2 * kSubTileBytes + static_cast<uint32_t>(Nu) * 128u);
                bulk_g2s_hint(xs + C::kActWideBytes, wk, 2 * kSubTileBytes, fb,
                              wpol);
                if (Nu == 256) {
                  tma_load_2d(xs, &xmap256, kx, p0u, fb);
                } else {
                  // hmap.m: [0]=16, [1]=64, [2]=96, [3]=128
                  const int tail_idx = (Nu == 144) ? 0 : ((Nu == 192) ? 1 : 2);
                  tma_load_2d(xs, &hmap.m[3], kx, p0u, fb);
                  tma_load_2d(xs + 128u * 128u, &hmap.m[tail_idx], kx,
                              p0u + 128, fb);
                }
              } else {
                mbar_expect_tx(
                    fb, 4 * kSubTileBytes + static_cast<uint32_t>(Nu) * 128u);
                bulk_g2s_hint(xs + C::kActNarrowBytes, wk, 4 * kSubTileBytes,
                              fb, wpol);
                const int map_idx = (Nu >= 128) ? 3 : ((Nu == 96) ? 2 : 1);
                tma_load_2d(xs, &hmap.m[map_idx], kx, p0u, fb);
              }
            }
            __syncwarp();
            ring_next<C::kNumStages>(s, ph);
          }
        }
        continue;
      }

      // MODE 0 (W13): dual-stage K producer loop with cp.async token gather +
      // TMA bulk weight transfer.
      const char* xb = reinterpret_cast<const char*>(p.x) + c * 16;
      const uint32_t dst0 = static_cast<uint32_t>(r0 * 128) + dsw;

      if (!wide) {
        uint32_t ro[8];
        uint32_t padm = 0;
        uint32_t usem = 0;
        const int nr = (N == 64) ? (pf.w <= 32 ? 32 : 64) : N;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = r0 + 16 * i;
          int row = 0;
          if (r < nr) {
            usem |= 1u << i;
            row = p.rows[pos0 + r];
            if (row < 0) {
              padm |= 1u << i;
              row = 0;
            }
          }
          ro[i] = static_cast<uint32_t>(row) *
                  static_cast<uint32_t>(kHiddenDim * 2);
        }

        int fk = ks;
        for (; fk + 1 < ke; fk += 2) {
          int nx = s;
          uint32_t np = ph;
          ring_next<C::kNumStages>(nx, np);
          const uint32_t s0 = sb + s * C::kStageBytes;
          const uint32_t s1 = sb + nx * C::kStageBytes;
          const uint32_t r0b = b0 + 8 * s;
          const uint32_t r1b = b0 + 8 * nx;
          mbar_wait(b0 + 8 * (C::kNumStages + s), ph ^ 1u);
          mbar_wait(b0 + 8 * (C::kNumStages + nx), np ^ 1u);

          const uint8_t* w0 = wsrc + static_cast<size_t>(fk) * kWeightChunk;
          const char* sc_ptr = xb + static_cast<uint32_t>(fk) * 128u;
          if (lt == 0) {
            if (N == 16) {
              const uint64_t pf1 = policy_evict_first();
              mbar_expect_tx(r0b, 2 * kWeightChunk);
              bulk_g2s_hint(s0 + kTinyWeightOff, w0, kWeightChunk, r0b, pf1);
              bulk_g2s_hint(
                  s0 + kTinyWeightOff + kWeightChunk,
                  w0 + static_cast<size_t>(kNumKBlocks) * kWeightChunk,
                  kWeightChunk, r0b, pf1);
              mbar_expect_tx(r1b, 2 * kWeightChunk);
              bulk_g2s_hint(s1 + kTinyWeightOff, w0 + kWeightChunk,
                            kWeightChunk, r1b, pf1);
              bulk_g2s_hint(
                  s1 + kTinyWeightOff + kWeightChunk,
                  w0 + static_cast<size_t>(kNumKBlocks + 1) * kWeightChunk,
                  kWeightChunk, r1b, pf1);
            } else {
              const uint64_t wpol =
                  evict_first ? policy_evict_first() : policy_evict_normal();
              mbar_expect_tx(r0b, 4 * kSubTileBytes);
              bulk_g2s_hint(s0 + C::kActNarrowBytes, w0, 4 * kSubTileBytes, r0b,
                            wpol);
              mbar_expect_tx(r1b, 4 * kSubTileBytes);
              bulk_g2s_hint(s1 + C::kActNarrowBytes, w0 + kWeightChunk,
                            4 * kSubTileBytes, r1b, wpol);
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if ((usem >> i) & 1u) {
              const uint32_t sz = ((padm >> i) & 1u) ? 0u : 16u;
              cp_async_cg_16_zfill(s0 + dst0 + i * 2048u, sc_ptr + ro[i], sz);
              cp_async_cg_16_zfill(s1 + dst0 + i * 2048u, sc_ptr + 128u + ro[i],
                                   sz);
            }
          }
          cp_async_mbar_arrive(r0b);
          cp_async_mbar_arrive(r1b);
          s = nx;
          ph = np;
          ring_next<C::kNumStages>(s, ph);
        }

#pragma unroll 1
        for (int kb = fk; kb < ke; ++kb) {
          const uint32_t xs = sb + s * C::kStageBytes;
          const uint32_t fb = b0 + 8 * s;
          const uint8_t* wk = wsrc + static_cast<size_t>(kb) * kWeightChunk;
          const char* xk = xb + kb * 128;
          const uint32_t d = xs + dst0;
          asm volatile("" ::"r"(xs), "r"(fb), "l"(wk), "l"(xk), "r"(d));

          mbar_wait(b0 + 8 * (C::kNumStages + s), ph ^ 1u);
          if (lt == 0) {
            if (N == 16) {
              const uint64_t pf1 = policy_evict_first();
              mbar_expect_tx(fb, 2 * kWeightChunk);
              bulk_g2s_hint(xs + kTinyWeightOff, wk, kWeightChunk, fb, pf1);
              bulk_g2s_hint(
                  xs + kTinyWeightOff + kWeightChunk,
                  wk + static_cast<size_t>(kNumKBlocks) * kWeightChunk,
                  kWeightChunk, fb, pf1);
            } else {
              const uint64_t wpol =
                  evict_first ? policy_evict_first() : policy_evict_normal();
              mbar_expect_tx(fb, 4 * kSubTileBytes);
              bulk_g2s_hint(xs + C::kActNarrowBytes, wk, 4 * kSubTileBytes, fb,
                            wpol);
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if ((usem >> i) & 1u) {
              cp_async_cg_16_zfill(d + i * 2048u, xk + ro[i],
                                   ((padm >> i) & 1u) ? 0u : 16u);
            }
          }
          cp_async_mbar_arrive(fb);
          ring_next<C::kNumStages>(s, ph);
        }
        continue;
      }

      // Wide tile (N > 128): unguarded 16-byte cp.async loads (padded tail rows
      // safely alias an earlier valid token row and are ignored downstream).
      uint32_t ro[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int r = r0 + 16 * i;
        const int row = (r < N) ? p.rows[pos0 + r] : 0;
        ro[i] = (i >= 8 && row < 0) ? ro[i >= 8 ? i - 8 : 0]
                                    : static_cast<uint32_t>(max(row, 0)) *
                                          static_cast<uint32_t>(kHiddenDim * 2);
      }
      const int nrt = N >> 4;  // 9 (144), 12 (192), 14 (224), or 16 (256)

      int fk = ks;
      for (; fk + 1 < ke; fk += 2) {
        int nx = s;
        uint32_t np = ph;
        ring_next<C::kNumStages>(nx, np);
        const uint32_t s0 = sb + s * C::kStageBytes;
        const uint32_t s1 = sb + nx * C::kStageBytes;
        const uint32_t r0b = b0 + 8 * s;
        const uint32_t r1b = b0 + 8 * nx;
        mbar_wait(b0 + 8 * (C::kNumStages + s), ph ^ 1u);
        mbar_wait(b0 + 8 * (C::kNumStages + nx), np ^ 1u);

        const uint8_t* w0 = wsrc + static_cast<size_t>(fk) * kWeightChunk;
        const char* sc_ptr = xb + static_cast<uint32_t>(fk) * 128u;
        if (lt == 0) {
          const uint64_t wpol =
              evict_first ? policy_evict_first() : policy_evict_normal();
          mbar_expect_tx(r0b, 2 * kSubTileBytes);
          bulk_g2s_hint(s0 + C::kActWideBytes, w0, 2 * kSubTileBytes, r0b,
                        wpol);
          mbar_expect_tx(r1b, 2 * kSubTileBytes);
          bulk_g2s_hint(s1 + C::kActWideBytes, w0 + kWeightChunk,
                        2 * kSubTileBytes, r1b, wpol);
        }
        auto copy_pair = [&](int i) {
          cp_async_cg_16(s0 + dst0 + i * 2048u, sc_ptr + ro[i]);
          cp_async_cg_16(s1 + dst0 + i * 2048u, sc_ptr + 128u + ro[i]);
        };
        if (nrt == 16) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            copy_pair(i);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 9; ++i) {
            copy_pair(i);
          }
          if (nrt > 9) {
#pragma unroll
            for (int i = 9; i < 12; ++i) {
              copy_pair(i);
            }
          }
          if (nrt > 12) {
#pragma unroll
            for (int i = 12; i < 14; ++i) {
              copy_pair(i);
            }
          }
        }
        cp_async_mbar_arrive(r0b);
        cp_async_mbar_arrive(r1b);
        s = nx;
        ph = np;
        ring_next<C::kNumStages>(s, ph);
      }

#pragma unroll 1
      for (int kb = fk; kb < ke; ++kb) {
        const uint32_t xs = sb + s * C::kStageBytes;
        const uint32_t fb = b0 + 8 * s;
        const uint8_t* wk = wsrc + static_cast<size_t>(kb) * kWeightChunk;
        const char* xk = xb + static_cast<uint32_t>(kb) * 128u;
        const uint32_t d = xs + dst0;
        asm volatile("" ::"r"(fb), "l"(wk), "l"(xk), "r"(d));

        mbar_wait(b0 + 8 * (C::kNumStages + s), ph ^ 1u);
        if (lt == 0) {
          const uint64_t wpol =
              evict_first ? policy_evict_first() : policy_evict_normal();
          mbar_expect_tx(fb, 2 * kSubTileBytes);
          bulk_g2s_hint(xs + C::kActWideBytes, wk, 2 * kSubTileBytes, fb, wpol);
        }
        if (nrt == 16) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            cp_async_cg_16(d + i * 2048u, xk + ro[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 9; ++i) {
            cp_async_cg_16(d + i * 2048u, xk + ro[i]);
          }
          if (nrt > 9) {
#pragma unroll
            for (int i = 9; i < 12; ++i) {
              cp_async_cg_16(d + i * 2048u, xk + ro[i]);
            }
          }
          if (nrt > 12) {
#pragma unroll
            for (int i = 12; i < 14; ++i) {
              cp_async_cg_16(d + i * 2048u, xk + ro[i]);
            }
          }
        }
        cp_async_mbar_arrive(fb);
        ring_next<C::kNumStages>(s, ph);
      }
    }
  } else {
    // Consumer warpgroups (232 registers each).
    setmaxnreg_inc<232>();

    const int cw = wg - 1;
    int s = 0;
    uint32_t ph = 0;
    int4 pfn = make_int4(0, 0, 0, 0);
    if (nwl > 0) {
      pfn = ld_global_nc_i4(p.pinfo + ((wl[0] >> 18) & 4095));
    }

#pragma unroll 1
    for (int it = 0; it < nwl; ++it) {
      const int w = wl[it];
      const int h = (w >> 30) & 1;
      const int mb = (w >> 13) & 31;
      const int ks = (w >> 7) & 63;
      const int ke = w & 127;
      const int4 pf = pfn;
      if (it + 1 < nwl) {
        pfn = ld_global_nc_i4(p.pinfo + ((wl[it + 1] >> 18) & 4095));
      }
      const int e = pf.x & kExpertMask;
      const int pos0 = pf.y;
      const int N = pf.z;
      const int na = pf.w;

      switch (((N == 64) ? (na <= 32 ? 32 : 64) : N) >> 4) {
        case 1:
          consume_tile<MODE, 16>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                 ke, s, ph, na, ymap);
          break;
        case 2:
          consume_tile<MODE, 32>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                 ke, s, ph, na, ymap);
          break;
        case 4:
          consume_tile<MODE, 64>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                 ke, s, ph, na, ymap);
          break;
        case 6:
          consume_tile<MODE, 96>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                 ke, s, ph, na, ymap);
          break;
        case 8:
          consume_tile<MODE, 128>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                  ke, s, ph, na, ymap);
          break;
        case 9:
          consume_tile<MODE, 144>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                  ke, s, ph, na, ymap);
          break;
        case 12:
          consume_tile<MODE, 192>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                  ke, s, ph, na, ymap);
          break;
        case 14:
          consume_tile<MODE, 224>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                  ke, s, ph, na, ymap);
          break;
        default:
          consume_tile<MODE, 256>(p, sbase, bar0, cw, tid, pos0, mb, h, e, ks,
                                  ke, s, ph, na, ymap);
          break;
      }
    }
  }

  if (tid == 128) {
    tma_store_wait_all();
    if constexpr (MODE == 1) {
      pdl_launch();
    }
  }
}

// ---------------------------------------------------------------------------
// Routing, tile-menu partitioning, and split-K scheduling kernel
// ---------------------------------------------------------------------------

struct RouteArgs {
  const int* ids;
  const int* remap;  // Optional 256-entry map: global id -> local or -1
  int remap_len;
  int local_start;
  int n;  // Total routing entries (T * 8)
  int* rows;
  int* rpos;
  int4* pinfo;
  int2* seg13;
  int2* seg2;
  int* flags;
  int G;
  int sc[2][kNumClasses];
  int ot[2];
  int skw;
  int nloc;
};

__device__ int g_route_part[kMaxRouteCtas][kMaxLocalExperts];

// Exclusive prefix sum over each 512-thread half of a 1024-thread CTA.
__device__ __forceinline__ long long half_excl(long long v, long long* red,
                                               long long& total) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = (tid >> 5) & 15;
  const int h = tid >> 9;

  long long x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const long long y = __shfl_up_sync(kFullWarpMask, x, o);
    if (lane >= o) {
      x += y;
    }
  }
  __syncthreads();
  if (lane == 31) {
    red[h * 16 + w] = x;
  }
  __syncthreads();

  const long long wt = (lane < 16) ? red[h * 16 + lane] : 0ll;
  long long base = (lane < w) ? wt : 0ll;
  long long tt = wt;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    base += __shfl_xor_sync(kFullWarpMask, base, o);
    tt += __shfl_xor_sync(kFullWarpMask, tt, o);
  }
  total = tt;
  return base + x - v;
}

__global__ void __launch_bounds__(1024) route_k(const RouteArgs a) {
  __shared__ int cnt[kMaxLocalExperts];
  __shared__ int rb[kMaxLocalExperts];
  __shared__ int mbase[kMaxLocalExperts + 1];
  __shared__ int pbase[kMaxLocalExperts + 1];
  __shared__ int gn[3][kMaxLocalExperts];
  __shared__ int gs[3][kMaxLocalExperts];
  __shared__ int coff[kNumClasses][kMaxLocalExperts];
  __shared__ int ccnt[kNumClasses];
  __shared__ int wtot[2][10];
  __shared__ int sl[256];
  __shared__ long long red[32];
  __shared__ int wsum[kMaxLocalExperts];

  union RouteScratch {
    struct {
      int wc[32][kMaxLocalExperts + 1];
      int wpre[32][kMaxLocalExperts + 1];
    } rank;
    struct {
      long long stc[2][kNumClasses];
      uint32_t scn[2][kNumClasses];
      uint32_t scc[2][kNumClasses];
      long long sa[2][kMaxCostTable / 2];
      long long ct[2][kMaxCostTable];
    } sched;
  };
  __shared__ RouteScratch scratch;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int ls = a.local_start;
  const int NL = a.nloc;
  const int cb = blockIdx.x;
  const int nb = gridDim.x;

  const int T = a.n >> 3;
  const int tpc = (nb > 1) ? ((T + nb - 2) / (nb - 1)) : T;
  const int t0 = min(T, cb * tpc);
  const int t1 = min(T, t0 + tpc);
  const bool one = (t1 - t0 <= 1024);
  const bool has_remap = (a.remap != nullptr);

  int tk = 0;
  int xi[8];
  uint64_t upk = 0;
  uint64_t wpk = 0;

  auto load_tok = [&](int r0) {
    tk = r0 + tid;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      *reinterpret_cast<int4*>(xi + 4 * h) =
          (tk < t1) ? *reinterpret_cast<const int4*>(
                          a.ids + static_cast<size_t>(tk) * 8 + 4 * h)
                    : make_int4(-1, -1, -1, -1);
    }
  };

  if (one) {
    load_tok(t0);
  }
  if (tid < kMaxLocalExperts) {
    cnt[tid] = 0;
    rb[tid] = 0;
  }
  if (has_remap && tid < 256) {
    const int mapped = (tid < a.remap_len) ? a.remap[tid] : -1;
    sl[tid] = local_slot(mapped, ls, NL);
  }
  __syncthreads();

  auto loc = [&](int id) -> int {
    if (!has_remap) {
      return local_slot(id, ls, NL);
    }
    if (static_cast<uint32_t>(id) >= 256u) {
      return 255;
    }
    return sl[id];
  };

  // Deterministic warp + CTA ranking of routed tokens per local expert.
  auto rank_chunk = [&](int r0) {
    if (!one) {
      load_tok(r0);
    }
    uint64_t m = 0;
    upk = 0;
    wpk = 0;
    int nr = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const uint32_t s = static_cast<uint32_t>(loc(xi[q]));
      upk |= static_cast<uint64_t>(s) << (8 * q);
      if (s < static_cast<uint32_t>(NL)) {
        m |= 1ull << s;
        ++nr;
      }
    }

    if (!__ballot_sync(kFullWarpMask, __popcll(m) != nr)) {
      uint32_t b0 = 0;
      uint32_t b1 = 0;
      for (int s = 0; s < NL; ++s) {
        const uint32_t bm =
            __ballot_sync(kFullWarpMask, static_cast<uint32_t>(m >> s) & 1u);
        if (lane == s) {
          b0 = bm;
        }
        if (lane == s - 32) {
          b1 = bm;
        }
      }
      scratch.rank.wc[w][lane] = __popc(b0);
      scratch.rank.wc[w][32 + lane] = __popc(b1);
      const uint32_t lm = (1u << lane) - 1u;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t s = static_cast<uint32_t>(upk >> (8 * q)) & 255u;
        const uint32_t c0 = __shfl_sync(kFullWarpMask, b0, s & 31u);
        const uint32_t c1 = __shfl_sync(kFullWarpMask, b1, s & 31u);
        wpk |= static_cast<uint64_t>(
                   static_cast<uint32_t>(__popc((s < 32u ? c0 : c1) & lm)))
               << (8 * q);
      }
    } else {
      scratch.rank.wc[w][lane] = 0;
      scratch.rank.wc[w][32 + lane] = 0;
      __syncwarp();
      for (int l = 0; l < 32; ++l) {
        if (lane == l) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint32_t s = static_cast<uint32_t>(upk >> (8 * q)) & 255u;
            if (s < static_cast<uint32_t>(NL)) {
              wpk |= static_cast<uint64_t>(
                         static_cast<uint32_t>(scratch.rank.wc[w][s]++))
                     << (8 * q);
            }
          }
        }
        __syncwarp();
      }
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int s = w + 32 * h;
      const int c = scratch.rank.wc[lane][s];
      int y = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int z = __shfl_up_sync(kFullWarpMask, y, o);
        if (lane >= o) {
          y += z;
        }
      }
      scratch.rank.wpre[lane][s] = y - c;
      if (lane == 31) {
        wsum[s] = y;
      }
    }
    __syncthreads();
  };

  if (nb == 1) {
    rank_chunk(0);
    if (tid < kMaxLocalExperts) {
      cnt[tid] = wsum[tid];
    }
    __syncthreads();
  } else {
#pragma unroll 1
    for (int r0 = t0; r0 < t1; r0 += 1024) {
      rank_chunk(r0);
      if (tid < kMaxLocalExperts) {
        cnt[tid] += wsum[tid];
      }
    }
    if (tid < kMaxLocalExperts) {
      g_route_part[cb][tid] = cnt[tid];
    }
    cooperative_groups::this_grid().sync();

    const int s = tid >> 4;
    int tot = 0;
    int pre = 0;
    for (int b = tid & 15; b < nb; b += 16) {
      const int c = __ldcg(&g_route_part[b][s]);
      tot += c;
      pre += (b < cb) ? c : 0;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
      tot += __shfl_xor_sync(kFullWarpMask, tot, o);
      pre += __shfl_xor_sync(kFullWarpMask, pre, o);
    }
    if ((tid & 15) == 0) {
      cnt[s] = tot;
      rb[s] = pre;
    }
    __syncthreads();
  }

  // Partition each local expert's token count into tile size classes and
  // compute aligned row bases (mbase), partition bases (pbase), and per-class
  // offsets (coff).
  int vq[kNumClasses];
  int xq[kNumClasses];
  int vm = 0;
  int vp = 0;

  if (tid < kMaxLocalExperts) {
    const int c = cnt[tid];
    int nA = 0, sA = 0, nB = 0, sB = 0, nC = 0, sC = 0;
    if (c > 0) {
      int k = 0;
      int cr = c;
      int r1 = 0;
      int r2 = 0;
      while (cr > 256) {
        ++k;
        cr -= 256;
      }
      if (k >= 1 && cr <= 32) {
        --k;
        r1 = 192;
        r2 = 96;
      } else {
        r1 = menu_size(cr);
      }
      nA = k;
      sA = 256;
      nB = 1;
      sB = r1;
      nC = r2 ? 1 : 0;
      sC = r2;
    }
    gn[0][tid] = nA;
    gs[0][tid] = sA;
    gn[1][tid] = nB;
    gs[1][tid] = sB;
    gn[2][tid] = nC;
    gs[2][tid] = sC;

    vm = nA * sA + nB * sB + nC * sC;
    vp = nA + nB + nC;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int um = __shfl_up_sync(kFullWarpMask, vm, o);
      const int up = __shfl_up_sync(kFullWarpMask, vp, o);
      if (lane >= o) {
        vm += um;
        vp += up;
      }
    }

    const int cA = nA ? size_class(sA) : -1;
    const int cB = nB ? size_class(sB) : -1;
    const int cC = nC ? size_class(sC) : -1;
#pragma unroll
    for (int q = 0; q < kNumClasses; ++q) {
      vq[q] = (cA == q ? nA : 0) + (cB == q ? nB : 0) + (cC == q ? nC : 0);
      int x = vq[q];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(kFullWarpMask, x, o);
        if (lane >= o) {
          x += y;
        }
      }
      xq[q] = x;
    }

    if (lane == 31) {
      wtot[tid >> 5][0] = vm;
      wtot[tid >> 5][1] = vp;
#pragma unroll
      for (int q = 0; q < kNumClasses; ++q) {
        wtot[tid >> 5][2 + q] = xq[q];
      }
    }
  }
  __syncthreads();

  if (tid < kMaxLocalExperts) {
    const int w1 = tid >> 5;
    mbase[tid + 1] = vm + (w1 ? wtot[0][0] : 0);
    pbase[tid + 1] = vp + (w1 ? wtot[0][1] : 0);
    if (tid == 0) {
      mbase[0] = 0;
      pbase[0] = 0;
    }
    int run = 0;
#pragma unroll
    for (int q = 0; q < kNumClasses; ++q) {
      const int tot = wtot[0][2 + q] + wtot[1][2 + q];
      coff[q][tid] = run + xq[q] + (w1 ? wtot[0][2 + q] : 0) - vq[q];
      if (tid == 0) {
        ccnt[q] = tot;
      }
      run += tot;
    }
  }
  __syncthreads();

  // Scatter sorted token indices (rows) and inverse routing positions (rpos).
#pragma unroll 1
  for (int r0 = t0; r0 < t1; r0 += 1024) {
    if (!one) {
      rank_chunk(r0);
    }
    if (r0 + tid < t1) {
      int4 p4[2];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t s = static_cast<uint32_t>(upk >> (8 * q)) & 255u;
        int pos = -1;
        if (s < static_cast<uint32_t>(NL)) {
          pos = mbase[s] + rb[s] + scratch.rank.wpre[w][s] +
                static_cast<int>((wpk >> (8 * q)) & 255u);
          a.rows[pos] = tk;
        }
        (&p4[q >> 2].x)[q & 3] = pos;
      }
      *reinterpret_cast<int4*>(a.rpos + static_cast<size_t>(tk) * 8) = p4[0];
      *reinterpret_cast<int4*>(a.rpos + static_cast<size_t>(tk) * 8 + 4) =
          p4[1];
    }
    if (!one) {
      __syncthreads();
      if (tid < kMaxLocalExperts) {
        rb[tid] += wsum[tid];
      }
    }
  }

  if (cb < nb - 1) {
    return;
  }

  // Final CTA writes partition descriptors, computes cost-balanced split-K
  // schedules for W13 (mode 0) and W2 (mode 1), zeroes split-K flags, and
  // pads unused tail slots with -1.
  const int NP = pbase[kMaxLocalExperts];

  auto partition_info = [&](int i, int& e, int& j, int& N) -> int {
    e = 0;
#pragma unroll
    for (int st = kMaxLocalExperts / 2; st > 0; st >>= 1) {
      if (pbase[e + st] <= i) {
        e += st;
      }
    }
    j = i - pbase[e];
    const int nA = gn[0][e];
    const int sA = gs[0][e];
    const int nB = gn[1][e];
    const int sB = gs[1][e];
    const int sC = gs[2][e];
    const int grp = (j < nA) ? 0 : ((j < nA + nB) ? 1 : 2);
    N = (grp == 0) ? sA : ((grp == 1) ? sB : sC);
    return mbase[e] + ((grp == 0) ? (sA * j)
                                  : ((grp == 1) ? (nA * sA + sB * (j - nA))
                                                : (nA * sA + nB * sB)));
  };

  // Tag each expert's sole or final token tile (j == vp_e - 1) with
  // kEvictFirstBit so the GEMM producer issues L2::evict_first when no later
  // token tile will reuse that expert's weights.
  for (int i = tid; i < NP; i += 1024) {
    int e, j, N;
    const int pos0 = partition_info(i, e, j, N);
    const int vp_e = pbase[e + 1] - pbase[e];
    const int evict_flag = (j == vp_e - 1) ? kEvictFirstBit : 0;
    const int cls = size_class(N);
    const int nA = gn[0][e];
    const int nB = gn[1][e];
    int ic = 0;
    if (nA && size_class(gs[0][e]) == cls) {
      ic += min(j, nA);
    }
    if (nB && size_class(gs[1][e]) == cls && j >= nA) {
      ic += min(j - nA, nB);
    }
    if (j >= nA + nB && size_class(gs[2][e]) == cls) {
      ic += j - nA - nB;
    }
    a.pinfo[coff[cls][e] + ic] = make_int4(
        e | evict_flag, pos0, N, max(0, min(N, cnt[e] - (pos0 - mbase[e]))));
  }
  __syncthreads();

  // Threads 0..511 schedule W13 (mode 0); threads 512..1023 schedule W2 (mode
  // 1).
  {
    const int mode = tid >> 9;
    const int lt = tid & 511;
    const int MB = mode ? 16 : 32;
    const int ot = a.ot[mode];
    const int G = a.G;

    long long* stc = scratch.sched.stc[mode];
    uint32_t* scn = scratch.sched.scn[mode];
    uint32_t* scc = scratch.sched.scc[mode];
    long long* sa = scratch.sched.sa[mode];
    long long* ct = scratch.sched.ct[mode];

    if (lt == 0) {
      uint32_t run = 0;
#pragma unroll
      for (int k = 0; k < kNumClasses; ++k) {
        const uint32_t x = static_cast<uint32_t>(
            (k < kNumWideClasses ? 2 * MB
                                 : (k == kNumClasses - 1 ? MB / 2 : MB)) *
            ccnt[k]);
        stc[k] = 64ll * a.sc[mode][k] + ot;
        scn[k] = x;
        run += x;
        scc[k] = run;
      }
    }
    if (lt < G) {
      sa[lt] = 0;
    }
    __syncthreads();

    const uint32_t n2 = scn[kNumClasses - 1];
    const uint32_t NT = scc[kNumClasses - 1];
    const float fr =
        NT ? (static_cast<float>(n2) / static_cast<float>(NT)) : 0.f;

    auto c2of = [&](uint32_t i) -> uint32_t {
      uint32_t q = static_cast<uint32_t>(static_cast<float>(i) * fr);
      const uint32_t p = i * n2;
      q += ((q + 1) * NT <= p);
      return q - (q * NT > p);
    };

    auto cpre = [&](uint32_t i) -> long long {
      const uint32_t q2 = min(n2, c2of(i));
      uint32_t r1 = i - q2;
      const uint32_t r2 = q2;
      long long c = 0;
#pragma unroll
      for (int k = 0; k < kNumClasses - 1; ++k) {
        const uint32_t x = min(r1, scn[k]);
        c += static_cast<long long>(x) * stc[k];
        r1 -= x;
      }
      return c + static_cast<long long>(min(r2, n2)) * stc[kNumClasses - 1];
    };

    auto ccls = [&](uint32_t i) -> long long {
      const uint32_t q2 = min(n2, c2of(i));
      const uint32_t q2n = min(n2, c2of(i + 1));
      const uint32_t j = i - q2;
      long long c = stc[kNumClasses - 2];
#pragma unroll
      for (int k = kNumClasses - 3; k >= 0; --k) {
        if (j < scc[k]) {
          c = stc[k];
        }
      }
      return (q2n > q2) ? stc[kNumClasses - 1] : c;
    };

    const int waves = static_cast<int>((NT + G - 1) / G);
    const bool sk = (waves <= a.skw);
    const int K = !sk ? waves
                      : (NT >= 2u * static_cast<uint32_t>(G)
                             ? static_cast<int>(NT / G) - 1
                             : 0);
    const uint32_t KG = sk ? (static_cast<uint32_t>(K) * G) : NT;

    if (static_cast<uint32_t>(lt) <= NT - KG) {
      ct[lt] = cpre(KG + static_cast<uint32_t>(lt));
    }

    const int nj = max(1, 512 / G);
    const int l0 = lt % G;
    long long A = 0;
    if (K > 0 && lt < nj * G) {
#pragma unroll 4
      for (int k = lt / G; k < K; k += nj) {
        const uint32_t pos =
            static_cast<uint32_t>(k * G + ((k & 1) ? (G - 1 - l0) : l0));
        if (pos < NT) {
          A += ccls(pos);
        }
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(sa + l0),
                static_cast<unsigned long long>(A));
    }
    __syncthreads();

    A = (lt < G) ? sa[lt] : 0;
    const long long pre = ct[0];
    const long long tot = ct[NT - KG];
    const long long S = tot - pre;
    const long long X = (tot + G - 1) / G;
    const long long d = (lt < G) ? max(0ll, X - A) : 0ll;
    long long D = 0;
    const long long P = half_excl(d, red, D);

    if (lt <= G) {
      int2 res = make_int2(static_cast<int>(NT), 0);
      if (lt < G && S > 0 && D > 0) {
        const long long target =
            pre + static_cast<long long>(static_cast<double>(S) *
                                         static_cast<double>(P) /
                                         static_cast<double>(D));
        uint32_t lo = KG;
        uint32_t hi = NT;
        while (hi - lo > 1) {
          const uint32_t mid = (lo + hi) >> 1;
          if (ct[mid - KG] <= target) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        const long long tcl = ccls(lo);
        int kb = static_cast<int>(static_cast<uint32_t>(target - ct[lo - KG]) /
                                  static_cast<uint32_t>((tcl - ot) / 64));
        uint32_t tile = lo;
        if (kb < kSplitSnap) {
          kb = 0;
        }
        if (kb > 64 - kSplitSnap) {
          ++tile;
          kb = 0;
        }
        res = make_int2(static_cast<int>(tile), kb);
      }
      (mode ? a.seg2 : a.seg13)[lt] = res;
    }

    if (lt == 0) {
      int* kd = a.flags + 2 * G + 32 * mode;
      kd[0] = K;
      kd[1] = static_cast<int>(n2);
      kd[2] = static_cast<int>(NT);
      int pc = 0;
#pragma unroll
      for (int k = 0; k < kNumClasses; ++k) {
        kd[3 + k] = static_cast<int>(scn[k]);
        kd[3 + kNumClasses + k] = pc;
        pc += ccnt[k];
      }
    }
  }

  for (int q = tid; q < 2 * a.G; q += 1024) {
    a.flags[q] = 0;
  }

  for (int e = tid >> 5; e < NL; e += 32) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int q = mbase[e] + cnt[e] + lane + 32 * k;
      if (q < mbase[e + 1]) {
        a.rows[q] = -1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fused weight-first post-expert RMSNorm + FP32 weighted sum combine kernel
// ---------------------------------------------------------------------------

template <int OCC, int B>
__global__ void __launch_bounds__(256, OCC)
    combine_p(const bf16* __restrict__ y, const int* __restrict__ rpos,
              const float* __restrict__ topk_weights,
              const float* __restrict__ post_weight, float post_eps,
              bf16* __restrict__ out, int T) {
  __shared__ float red[2][B][8];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wp = tid >> 5;

  // Preload invariant post_weight before pdl_wait() to overlap with gemm_k<1>
  // epilogue tail.
  float pw[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *reinterpret_cast<float4*>(pw + 4 * i) = reinterpret_cast<const float4*>(
        post_weight + (i >> 1) * 2048 + tid * 8)[i & 1];
  }
  pdl_wait();

  int mp = (static_cast<int>(blockIdx.x) < T && lane < 8)
               ? __ldg(rpos + static_cast<size_t>(blockIdx.x) * 8 + lane)
               : -1;
  int buf = 0;

#pragma unroll 1
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const uint32_t m = __ballot_sync(kFullWarpMask, mp >= 0);
    const int tn = t + gridDim.x;
    const int np = (tn < T && lane < 8)
                       ? __ldg(rpos + static_cast<size_t>(tn) * 8 + lane)
                       : -1;
    bf16* orow = out + static_cast<size_t>(t) * kHiddenDim;

    if (m == 0u) {
      mp = np;
      *reinterpret_cast<uint4*>(orow + tid * 8) = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(orow + 2048 + tid * 8) =
          make_uint4(0u, 0u, 0u, 0u);
      continue;
    }

    float mw = (lane < 8)
                   ? __ldg(topk_weights + static_cast<size_t>(t) * 8 + lane)
                   : 0.f;
    const int nv = __popc(m);
    if (m != 0xffu) {
      const uint32_t src = __fns(m, 0, lane + 1) & 7u;
      mp = __shfl_sync(kFullWarpMask, mp, src);
      mw = __shfl_sync(kFullWarpMask, mw, src);
    }

    float acc[16] = {};
#pragma unroll 1
    for (int j0 = 0; j0 < (B == 8 ? 1 : nv); j0 += B) {
      uint4 v[B][2] = {};
#pragma unroll
      for (int j = 0; j < B; ++j) {
        if (j0 + j < nv) {
          const bf16* yr =
              y + static_cast<size_t>(__shfl_sync(kFullWarpMask, mp, j0 + j)) *
                      kHiddenDim;
          v[j][0] = __ldg(reinterpret_cast<const uint4*>(yr + tid * 8));
          v[j][1] = __ldg(reinterpret_cast<const uint4*>(yr + 2048 + tid * 8));
        }
      }
#pragma unroll
      for (int j = 0; j < B; ++j) {
        if (j0 + j < nv) {
          float s = 0.f;
          const uint32_t w[8] = {v[j][0].x, v[j][0].y, v[j][0].z, v[j][0].w,
                                 v[j][1].x, v[j][1].y, v[j][1].z, v[j][1].w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float a = bf16_lo_to_f32(w[i]) * pw[2 * i];
            const float b = bf16_hi_to_f32(w[i]) * pw[2 * i + 1];
            s = fmaf(a, a, s);
            s = fmaf(b, b, s);
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(kFullWarpMask, s, o);
          }
          if (lane == 0) {
            red[buf][j][wp] = s;
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int j = 0; j < B; ++j) {
        if (j0 + j < nv) {
          float ss = 0.f;
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            ss += red[buf][j][w];
          }
          const float c = __shfl_sync(kFullWarpMask, mw, j0 + j) *
                          rsqrtf(ss * (1.f / 4096.f) + post_eps);
          const uint32_t u[8] = {v[j][0].x, v[j][0].y, v[j][0].z, v[j][0].w,
                                 v[j][1].x, v[j][1].y, v[j][1].z, v[j][1].w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            acc[2 * i] += c * bf16_lo_to_f32(u[i]);
            acc[2 * i + 1] += c * bf16_hi_to_f32(u[i]);
          }
        }
      }
      buf ^= 1;
    }

    mp = np;
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o[i] =
          pack_bf16x2(acc[2 * i] * pw[2 * i], acc[2 * i + 1] * pw[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(orow + tid * 8) =
        make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(orow + 2048 + tid * 8) =
        make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// ---------------------------------------------------------------------------
// Per-device LRU cache of 2D TMA CUtensorMap descriptors: 16 buffer entries,
// each holding one descriptor per box size (9 per entry)
// ---------------------------------------------------------------------------

static PFN_cuTensorMapEncodeTiled_v12000 init_tma_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 encode_fn = nullptr;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status{};
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status);
    if (err == cudaSuccess && status == cudaDriverEntryPointSuccess) {
      encode_fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
    } else {
      (void)cudaGetLastError();
    }
  });
  TORCH_CHECK(encode_fn != nullptr, "cuTensorMapEncodeTiled unavailable");
  return encode_fn;
}

struct BufferKey {
  int device_id;
  const void* data_ptr;
  uint64_t num_rows;

  bool operator==(const BufferKey& other) const {
    return device_id == other.device_id && data_ptr == other.data_ptr &&
           num_rows == other.num_rows;
  }
};

class TensorMapCache {
 public:
  static constexpr size_t kCapacity = 16;
  static constexpr int kNumUnifiedBoxSizes = 9;

  static constexpr int box_rows_to_slot(uint32_t box_rows) {
    switch (box_rows) {
      case 16:
        return 0;
      case 32:
        return 1;
      case 64:
        return 2;
      case 96:
        return 3;
      case 128:
        return 4;
      case 144:
        return 5;
      case 192:
        return 6;
      case 224:
        return 7;
      case 256:
        return 8;
      default:
        return -1;
    }
  }

  void get_or_encode_batch(int device_id, const void* data_ptr,
                           uint64_t num_rows, const uint32_t* box_rows,
                           size_t count, CUtensorMap* out_maps) {
    TORCH_CHECK(count <= static_cast<size_t>(kNumUnifiedBoxSizes),
                "TensorMapCache: batch count exceeds canonical box table");
    int slots[kNumUnifiedBoxSizes];
    bool hit[kNumUnifiedBoxSizes] = {};
    bool all_hit = true;
    const BufferKey key{device_id, data_ptr, num_rows};

    for (size_t k = 0; k < count; ++k) {
      slots[k] = box_rows_to_slot(box_rows[k]);
      TORCH_CHECK(slots[k] >= 0,
                  "TensorMapCache: unsupported box_rows=", box_rows[k]);
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      const int idx = find_locked(key);
      if (idx >= 0) {
        Entry& e = entries_[idx];
        e.last_used = ++clock_;
        for (size_t k = 0; k < count; ++k) {
          if ((e.valid_mask >> slots[k]) & 1u) {
            out_maps[k] = e.maps[slots[k]];
            hit[k] = true;
          } else {
            all_hit = false;
          }
        }
      } else {
        all_hit = false;
      }
    }

    if (all_hit) {
      return;
    }

    for (size_t k = 0; k < count; ++k) {
      if (!hit[k]) {
        out_maps[k] = encode_tma_2d(data_ptr, num_rows, box_rows[k]);
      }
    }

    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_locked(key);
    if (idx < 0) {
      idx = alloc_lru_slot_locked();
      entries_[idx].key = key;
      entries_[idx].valid_mask = 0;
      for (size_t k = 0; k < count; ++k) {
        entries_[idx].maps[slots[k]] = out_maps[k];
        entries_[idx].valid_mask |= static_cast<uint16_t>(1u << slots[k]);
      }
    } else {
      for (size_t k = 0; k < count; ++k) {
        if (!hit[k]) {
          entries_[idx].maps[slots[k]] = out_maps[k];
          entries_[idx].valid_mask |= static_cast<uint16_t>(1u << slots[k]);
        }
      }
    }
    entries_[idx].last_used = ++clock_;
  }

 private:
  struct Entry {
    BufferKey key{};
    uint64_t last_used = 0;
    uint16_t valid_mask = 0;
    CUtensorMap maps[kNumUnifiedBoxSizes]{};
  };

  int find_locked(const BufferKey& key) const {
    for (size_t i = 0; i < size_; ++i) {
      if (entries_[i].key == key) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  int alloc_lru_slot_locked() {
    if (size_ < kCapacity) {
      return static_cast<int>(size_++);
    }
    size_t victim = 0;
    uint64_t oldest = entries_[0].last_used;
    for (size_t i = 1; i < kCapacity; ++i) {
      if (entries_[i].last_used < oldest) {
        oldest = entries_[i].last_used;
        victim = i;
      }
    }
    return static_cast<int>(victim);
  }

  static CUtensorMap encode_tma_2d(const void* data_ptr, uint64_t num_rows,
                                   uint32_t box_rows) {
    const auto encode_fn = init_tma_encode_fn();

    CUtensorMap map{};
    cuuint64_t dims[2] = {static_cast<cuuint64_t>(kHiddenDim),
                          static_cast<cuuint64_t>(num_rows)};
    cuuint64_t strides[1] = {static_cast<cuuint64_t>(kHiddenDim) * 2};
    cuuint32_t box[2] = {64u, static_cast<cuuint32_t>(box_rows)};
    cuuint32_t elem_strides[2] = {1u, 1u};
    const CUresult res = encode_fn(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(data_ptr),
        dims, strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(
        res == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed with CUresult=", static_cast<int>(res));
    return map;
  }

  std::mutex mu_;
  std::array<Entry, kCapacity> entries_{};
  size_t size_ = 0;
  uint64_t clock_ = 0;
};

static TensorMapCache& get_tmap_cache(int device_id) {
  static constexpr int kMaxCachedDevices = 16;
  static std::array<TensorMapCache, kMaxCachedDevices> caches;
  static TensorMapCache overflow_cache;
  if (device_id >= 0 && device_id < kMaxCachedDevices) {
    return caches[device_id];
  }
  return overflow_cache;
}

template <int MODE>
static void ensure_gemm_smem_configured(int device_id) {
  static constexpr int kMaxTrackedDevices = 64;
  static std::array<std::atomic<bool>, kMaxTrackedDevices> smem_configured{};
  if (device_id >= 0 && device_id < kMaxTrackedDevices) {
    if (!smem_configured[device_id].load(std::memory_order_acquire)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          gemm_k<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          StageCfg<MODE>::kSmemBytes));
      smem_configured[device_id].store(true, std::memory_order_release);
    }
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemm_k<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        StageCfg<MODE>::kSmemBytes));
  }
}

template <int MODE>
static void launch_gemm(const GemmParams& p, const TensorMaps& maps, int grid,
                        int device_id) {
  ensure_gemm_smem_configured<MODE>(device_id);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if constexpr (MODE == 1) {
    C10_CUDA_CHECK(launch_pdl(gemm_k<MODE>, dim3(grid), dim3(kNumThreads),
                              StageCfg<MODE>::kSmemBytes, stream, p, maps.x256,
                              maps.h, maps.y));
  } else {
    gemm_k<MODE><<<grid, kNumThreads, StageCfg<MODE>::kSmemBytes, stream>>>(
        p, maps.x256, maps.h, maps.y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

static void check_cuda_tensor(const Tensor& t, torch::Device dev,
                              torch::ScalarType dtype, const char* op,
                              const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == dev, "nvfp4_moe_prefill ", op, ": ",
              name, " must be on the same CUDA device");
  TORCH_CHECK(t.is_contiguous(), "nvfp4_moe_prefill ", op, ": ", name,
              " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, "nvfp4_moe_prefill ", op, ": ", name,
              " has unexpected dtype");
}

// ---------------------------------------------------------------------------
// Host entry points exposed via PyBind11
// ---------------------------------------------------------------------------

void configure_device(int64_t dev_idx) {
  TORCH_CHECK(dev_idx >= 0 && dev_idx < 128,
              "nvfp4_moe_prefill configure_device: invalid dev_idx=", dev_idx);
  const int dev = static_cast<int>(dev_idx);
  c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(dev));
  ensure_gemm_smem_configured<0>(dev);
  ensure_gemm_smem_configured<1>(dev);
  (void)init_tma_encode_fn();
}

void route(Tensor topk_ids, int64_t local_start, Tensor rows, Tensor rpos,
           Tensor pinfo, Tensor seg, Tensor flags, int64_t grid,
           std::vector<int64_t> cost, int64_t nloc,
           c10::optional<Tensor> remap) {
  TORCH_CHECK(topk_ids.is_cuda(),
              "nvfp4_moe_prefill route: topk_ids must be a CUDA tensor");
  const auto dev = topk_ids.device();
  check_cuda_tensor(topk_ids, dev, torch::kInt32, "route", "topk_ids");
  check_cuda_tensor(rows, dev, torch::kInt32, "route", "rows");
  check_cuda_tensor(rpos, dev, torch::kInt32, "route", "rpos");
  check_cuda_tensor(pinfo, dev, torch::kInt32, "route", "pinfo");
  check_cuda_tensor(seg, dev, torch::kInt32, "route", "seg");
  check_cuda_tensor(flags, dev, torch::kInt32, "route", "flags");
  TORCH_CHECK(cost.size() == 19,
              "nvfp4_moe_prefill route: cost must contain 19 entries");
  TORCH_CHECK(
      std::all_of(cost.begin(), cost.end(),
                  [](int64_t c) { return c > 0 && c <= INT32_MAX; }),
      "nvfp4_moe_prefill route: cost entries must be positive 32-bit ints");
  TORCH_CHECK(nloc > 0 && nloc <= kMaxLocalExperts,
              "nvfp4_moe_prefill route: nloc must be in [1, 64]");
  TORCH_CHECK(grid > 0 && 2 * grid < kMaxCostTable,
              "nvfp4_moe_prefill route: invalid grid size");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(1) == 8,
              "nvfp4_moe_prefill route: topk_ids must have shape [T, 8]");
  const int64_t n_routes = topk_ids.numel();
  const int64_t max_rows = n_routes + 64 * nloc;
  const int64_t max_parts = (max_rows + 255) / 256 + 2 * nloc;
  TORCH_CHECK(rows.numel() >= max_rows && rpos.numel() >= n_routes &&
                  pinfo.numel() >= 4 * max_parts &&
                  seg.numel() >= 4 * (grid + 1) &&
                  flags.numel() >= 2 * grid + 64,
              "nvfp4_moe_prefill route: workspace tensors are too small");

  const int* remap_ptr = nullptr;
  int remap_len = 0;
  if (remap.has_value() && remap->numel() > 0) {
    check_cuda_tensor(*remap, dev, torch::kInt32, "route", "remap");
    remap_ptr = remap->data_ptr<int>();
    remap_len = static_cast<int>(std::min<int64_t>(remap->numel(), 256));
  }

  c10::cuda::CUDAGuard device_guard(dev);

  RouteArgs a{};
  a.ids = topk_ids.data_ptr<int>();
  a.remap = remap_ptr;
  a.remap_len = remap_len;
  a.local_start = static_cast<int>(local_start);
  a.n = static_cast<int>(topk_ids.numel());
  a.rows = rows.data_ptr<int>();
  a.rpos = rpos.data_ptr<int>();
  a.pinfo = reinterpret_cast<int4*>(pinfo.data_ptr<int>());
  a.seg13 = reinterpret_cast<int2*>(seg.data_ptr<int>());
  a.seg2 = reinterpret_cast<int2*>(seg.data_ptr<int>()) + (grid + 1);
  a.flags = flags.data_ptr<int>();
  a.G = static_cast<int>(grid);
  a.skw = static_cast<int>(cost[18]);
  a.nloc = static_cast<int>(nloc);

  TORCH_CHECK(((reinterpret_cast<uintptr_t>(a.ids) |
                reinterpret_cast<uintptr_t>(a.rpos) |
                reinterpret_cast<uintptr_t>(a.pinfo)) &
               15u) == 0u &&
                  (reinterpret_cast<uintptr_t>(a.seg13) & 7u) == 0u &&
                  (a.n % 8 == 0),
              "nvfp4_moe_prefill route: ids/rpos/pinfo must be 16-B aligned, "
              "seg 8-B aligned, and n % 8 == 0");

  for (int m = 0; m < 2; ++m) {
    for (int k = 0; k < kNumClasses; ++k) {
      a.sc[m][k] = static_cast<int>(cost[9 * m + k]);
    }
    a.ot[m] = static_cast<int>(cost[9 * m + 8]);
  }

  const int T = a.n / 8;
  const int nb =
      (T <= kRouteTokensPerCta)
          ? 1
          : std::min({static_cast<int>(grid), kMaxRouteCtas,
                      (T + kRouteTokensPerCta - 1) / kRouteTokensPerCta + 1});
  TORCH_CHECK(T <= kRouteTokensPerCta || nb > 1,
              "nvfp4_moe_prefill route: T > ", kRouteTokensPerCta,
              " requires grid >= 2");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (nb == 1) {
    route_k<<<1, 1024, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    void* kernel_args[] = {&a};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(route_k),
                                               dim3(nb), dim3(1024),
                                               kernel_args, 0, stream));
  }
}

void gemm(int64_t mode, Tensor w, Tensor gs, c10::optional<Tensor> gs_up,
          Tensor x, Tensor rows, Tensor pinfo, Tensor seg, Tensor part,
          Tensor flags, Tensor out, int64_t grid) {
  TORCH_CHECK(mode == 0 || mode == 1,
              "nvfp4_moe_prefill gemm: mode must be 0 (W13) or 1 (W2)");
  TORCH_CHECK(grid > 0 && 2 * grid < kMaxCostTable,
              "nvfp4_moe_prefill gemm: invalid grid size");
  TORCH_CHECK(out.is_cuda(),
              "nvfp4_moe_prefill gemm: out must be a CUDA tensor");
  const auto dev = out.device();
  check_cuda_tensor(w, dev, torch::kUInt8, "gemm", "w");
  check_cuda_tensor(gs, dev, torch::kFloat32, "gemm", "gs");
  check_cuda_tensor(x, dev, torch::kBFloat16, "gemm", "x");
  check_cuda_tensor(out, dev, torch::kBFloat16, "gemm", "out");
  check_cuda_tensor(rows, dev, torch::kInt32, "gemm", "rows");
  check_cuda_tensor(pinfo, dev, torch::kInt32, "gemm", "pinfo");
  check_cuda_tensor(seg, dev, torch::kInt32, "gemm", "seg");
  check_cuda_tensor(part, dev, torch::kFloat32, "gemm", "part");
  check_cuda_tensor(flags, dev, torch::kInt32, "gemm", "flags");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kHiddenDim,
              "nvfp4_moe_prefill gemm: x must have shape [*, 4096]");
  TORCH_CHECK(out.dim() == 2 && out.size(1) == kHiddenDim,
              "nvfp4_moe_prefill gemm: out must have shape [*, 4096]");
  const int64_t num_m_blocks = (mode == 1) ? 16 : 32;
  TORCH_CHECK(w.dim() == 4 && w.size(1) == num_m_blocks &&
                  w.size(2) == kNumKBlocks && w.size(3) == kWeightChunk,
              "nvfp4_moe_prefill gemm: w must have shape [E, ", num_m_blocks,
              ", 64, 9216]");
  TORCH_CHECK(gs.numel() >= w.size(0),
              "nvfp4_moe_prefill gemm: gs must cover all local experts");
  TORCH_CHECK(part.numel() >= grid * kPartSlot &&
                  seg.numel() >= 4 * (grid + 1) &&
                  flags.numel() >= 2 * grid + 64,
              "nvfp4_moe_prefill gemm: part/seg/flags too small for grid");

  const float* gs_up_ptr = nullptr;
  if (mode == 0 && gs_up.has_value() && gs_up->numel() > 0) {
    check_cuda_tensor(*gs_up, dev, torch::kFloat32, "gemm", "gs_up");
    TORCH_CHECK(gs_up->numel() >= w.size(0),
                "nvfp4_moe_prefill gemm: gs_up must cover all local experts");
    gs_up_ptr = gs_up->data_ptr<float>();
  }

  if (out.size(0) == 0 || x.size(0) == 0) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(dev);
  const int dev_id = out.get_device();
  const int num_sms = get_device_sm_count(dev_id);
  TORCH_CHECK(grid <= num_sms,
              "nvfp4_moe_prefill gemm: split-K requires grid <= SM count (",
              num_sms, "), got ", grid);

  const bool is_w2 = (mode == 1);
  const int2* seg_base = reinterpret_cast<const int2*>(seg.data_ptr<int>());
  int* flags_base = flags.data_ptr<int>();

  GemmParams p{};
  p.w = w.data_ptr<uint8_t>();
  p.gs = gs.data_ptr<float>();
  p.gs_up = gs_up_ptr;
  p.x = reinterpret_cast<const bf16*>(x.data_ptr());
  p.rows = rows.data_ptr<int>();
  p.pinfo = reinterpret_cast<const int4*>(pinfo.data_ptr<int>());
  p.seg = seg_base + (is_w2 ? (grid + 1) : 0);
  p.part = part.data_ptr<float>();
  p.flags = flags_base + (is_w2 ? grid : 0);
  p.kdp = flags_base + 2 * grid + (is_w2 ? 32 : 0);
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.num_m_blocks = is_w2 ? 16 : 32;
  p.expert_stride = static_cast<size_t>(w.stride(0));

  TensorMaps maps{};
  TensorMapCache& cache = get_tmap_cache(dev_id);

  static constexpr uint32_t kYBoxRows[kNumYBoxMaps] = {16,  64,  96,  128, 144,
                                                       192, 224, 256, 32};
  cache.get_or_encode_batch(dev_id, out.data_ptr(),
                            static_cast<uint64_t>(out.size(0)), kYBoxRows,
                            kNumYBoxMaps, maps.y.m);

  if (is_w2) {
    static constexpr uint32_t kXBoxRows[kNumHBoxMaps + 1] = {16, 64, 96, 128,
                                                             kTileWide};
    CUtensorMap xmaps[kNumHBoxMaps + 1];
    cache.get_or_encode_batch(dev_id, x.data_ptr(),
                              static_cast<uint64_t>(x.size(0)), kXBoxRows,
                              kNumHBoxMaps + 1, xmaps);
    for (int i = 0; i < kNumHBoxMaps; ++i) {
      maps.h.m[i] = xmaps[i];
    }
    maps.x256 = xmaps[kNumHBoxMaps];
    launch_gemm<1>(p, maps, static_cast<int>(grid), dev_id);
  } else {
    launch_gemm<0>(p, maps, static_cast<int>(grid), dev_id);
  }
}

void combine(Tensor y, Tensor rpos, Tensor topk_weights, Tensor post_weight,
             double post_eps, Tensor out) {
  TORCH_CHECK(out.is_cuda(),
              "nvfp4_moe_prefill combine: out must be a CUDA tensor");
  const auto dev = out.device();
  check_cuda_tensor(y, dev, torch::kBFloat16, "combine", "y");
  check_cuda_tensor(rpos, dev, torch::kInt32, "combine", "rpos");
  check_cuda_tensor(topk_weights, dev, torch::kFloat32, "combine",
                    "topk_weights");
  check_cuda_tensor(post_weight, dev, torch::kFloat32, "combine",
                    "post_weight");
  check_cuda_tensor(out, dev, torch::kBFloat16, "combine", "out");
  TORCH_CHECK(y.dim() == 2 && y.size(1) == kHiddenDim,
              "nvfp4_moe_prefill combine: y must have shape [*, 4096]");
  TORCH_CHECK(post_weight.numel() == kHiddenDim,
              "nvfp4_moe_prefill combine: post_weight must have 4096 elements");
  TORCH_CHECK(out.dim() == 2 && out.size(1) == kHiddenDim,
              "nvfp4_moe_prefill combine: out must have shape [T, 4096]");

  const int T = static_cast<int>(out.size(0));
  if (T == 0) {
    return;
  }
  TORCH_CHECK(rpos.numel() >= static_cast<int64_t>(T) * 8 &&
                  topk_weights.numel() >= static_cast<int64_t>(T) * 8,
              "nvfp4_moe_prefill combine: rpos/topk_weights must cover [T, 8]");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(post_weight.data_ptr()) & 15u) == 0u,
              "nvfp4_moe_prefill combine: post_weight must be 16-B aligned");

  c10::cuda::CUDAGuard device_guard(dev);
  const int nsm = get_device_sm_count(out.get_device());
  const bool dense_8 =
      (y.size(0) == static_cast<int64_t>(T) * 8) && (T >= 1024);
  const float eps = static_cast<float>(post_eps);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (dense_8) {
    C10_CUDA_CHECK(launch_pdl(
        combine_p<2, 8>, dim3(std::min(T, 4 * nsm)), dim3(256), 0, stream,
        reinterpret_cast<const bf16*>(y.data_ptr()), rpos.data_ptr<int>(),
        topk_weights.data_ptr<float>(), post_weight.data_ptr<float>(), eps,
        reinterpret_cast<bf16*>(out.data_ptr()), T));
  } else {
    C10_CUDA_CHECK(launch_pdl(
        combine_p<3, 2>, dim3(std::min(T, 6 * nsm)), dim3(256), 0, stream,
        reinterpret_cast<const bf16*>(y.data_ptr()), rpos.data_ptr<int>(),
        topk_weights.data_ptr<float>(), post_weight.data_ptr<float>(), eps,
        reinterpret_cast<bf16*>(out.data_ptr()), T));
  }
}

}  // namespace vllm::muse_spark_hopper_nvfp4_moe::prefill

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace p = vllm::muse_spark_hopper_nvfp4_moe::prefill;
  m.def("configure_device", &p::configure_device, py::arg("dev_idx"));
  m.def("route", &p::route, py::arg("topk_ids"), py::arg("local_start"),
        py::arg("rows"), py::arg("rpos"), py::arg("pinfo"), py::arg("seg"),
        py::arg("flags"), py::arg("grid"), py::arg("cost"), py::arg("nloc"),
        py::arg("remap") = py::none());
  m.def("gemm", &p::gemm, py::arg("mode"), py::arg("w"), py::arg("gs"),
        py::arg("gs_up"), py::arg("x"), py::arg("rows"), py::arg("pinfo"),
        py::arg("seg"), py::arg("part"), py::arg("flags"), py::arg("out"),
        py::arg("grid"));
  m.def("combine", &p::combine, py::arg("y"), py::arg("rpos"),
        py::arg("topk_weights"), py::arg("post_weight"), py::arg("post_eps"),
        py::arg("out"));
}
