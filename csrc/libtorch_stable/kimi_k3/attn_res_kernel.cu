/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

// Production AttnRes forward for Blackwell (SM100).
//
// Warp-specialized online softmax + residual + RMSNorm:
//   - 1 producer warp issues cp.async.bulk row loads into shared memory.
//   - 8 consumer warps compute reductions and output.
//   - Q=res_weight*rms_weight remains in registers across persistent tokens.
//   - V rows are converted once and cached as FP32 in TMEM between passes.
//
// Integration contract: Kimi K3 H=7168, 1<=num_blocks<=8, and token-major
// block residual storage.

#include "../torch_utils.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>

using bf16_t = __nv_bfloat16;

namespace sm100 {
namespace fwd_prod_v2 {

constexpr int K_TILE = 1024;
constexpr int N_CHUNK_DEFAULT = 4;
constexpr int CHUNK_DEPTH = 2;
constexpr int BLK = 288;  // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS = BLK - 32;  // 256
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;  // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int FIRST_USER_NAMED_BARRIER = 8;

__device__ __forceinline__ const bf16_t* residual_addr(
    const bf16_t* block_res, const bf16_t* layer_res, int source, int N,
    int token, int block_stride_m, int block_stride_r, int H) {
  if (source < N - 1) {
    return block_res + static_cast<long long>(token) * block_stride_m +
           source * block_stride_r;
  }
  return layer_res + static_cast<long long>(token) * H;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xffffffff));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(uint64_t& barrier,
                                              int thread_count) {
  uint32_t const barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier_addr),
               "r"(thread_count));
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t& barrier,
                                                   uint32_t bytes) {
  uint32_t const barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   barrier_addr),
               "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t& barrier, int phase) {
  uint32_t const barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra DONE;\n"
      "bra WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(barrier_addr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t& barrier) {
  uint32_t const barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(barrier_addr)
      : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync(uint32_t num_threads,
                                                   uint32_t user_barrier_id) {
  asm volatile(
      "bar.sync %0, %1;" ::"r"(user_barrier_id + FIRST_USER_NAMED_BARRIER),
      "r"(num_threads)
      : "memory");
}

__device__ __forceinline__ void tmem_allocate(int num_columns, uint32_t* dst) {
  uint32_t const dst_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          dst_addr),
      "r"(num_columns));
}

__device__ __forceinline__ void tmem_free(uint32_t tmem_ptr, int num_columns) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_ptr),
      "r"(num_columns));
}

__device__ __forceinline__ void tmem_release_allocation_lock() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ __forceinline__ void tmem_store_wait() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

template <int N, typename T>
__device__ __forceinline__ void tmem_load(uint32_t src_addr, T* dst) {
  uint32_t* values = reinterpret_cast<uint32_t*>(dst);
  if constexpr (N == 8) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
          "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
        : "r"(src_addr));
  } else {
    static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
        : "r"(src_addr));
  }
}

template <int N, typename T>
__device__ __forceinline__ void tmem_store(uint32_t dst_addr, T* src) {
  uint32_t* values = reinterpret_cast<uint32_t*>(src);
  if constexpr (N == 8) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n" ::"r"(values[0]),
        "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(values[4]),
        "r"(values[5]), "r"(values[6]), "r"(values[7]), "r"(dst_addr));
  } else {
    static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        "[%4], {%0, %1, %2, %3};\n" ::"r"(values[0]),
        "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(dst_addr));
  }
}

__device__ __forceinline__ float2 float2_add(const float2& a, const float2& b) {
  float2 result;
  asm volatile("add.rn.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(result))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)));
  return result;
}

__device__ __forceinline__ float2 float2_mul(const float2& a, const float2& b) {
  float2 result;
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(result))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)));
  return result;
}

__device__ __forceinline__ float2 float2_fma(const float2& a, const float2& b,
                                             const float2& c) {
  float2 result;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(result))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
  return result;
}

template <int NC>
struct FwdSmemPlan {
  alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
  alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
  alignas(16) uint64_t bar_output_norm_ready;
  alignas(16) float2 ws_stats[CONSUMER_WARPS][NC];
  uint32_t tmem_base;
};

__device__ __forceinline__ void cp_async_bulk(void* smem_dst,
                                              const void* gmem_src, int bytes,
                                              uint64_t& mbar) {
  uint32_t const s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t const m = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];\n" ::"r"(s),
      "l"(gmem_src), "r"(bytes), "r"(m)
      : "memory");
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, int B = 1,
          bool RELEASE_TMEM = false, bool HAS_DELTA = false,
          bool HAS_OUTPUT_NORM = false, bool OUTPUT_NORM_IN_SMEM = false>
__global__ void __launch_bounds__(BLK, 1) attn_res_fwd_online_v2_kernel(
    const bf16_t* __restrict__ block_res, bf16_t* __restrict__ layer_res,
    const bf16_t* __restrict__ delta, const bf16_t* __restrict__ res_w,
    const bf16_t* __restrict__ rms_w, bf16_t* __restrict__ output, int T,
    int block_stride_m, int block_stride_r, float rms_eps,
    const bf16_t* __restrict__ output_norm_weight, float output_norm_eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  constexpr float LOG2_E = 1.4426950408889634f;
  constexpr int N_CHUNK = NC;
  // The two-source specialization only consumes half of the TMEM columns.
  constexpr int TMEM_COLS_ALLOC = NC == 2 ? 128 : 256;
  constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
  constexpr int NHT = H / K_TILE;
  constexpr int SLICES_PER_GROUP =
      (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
  constexpr int VEC = 8;
  constexpr int ACC_PER_THREAD = H == 7168 ? 28 : SLICES_PER_GROUP * VEC;
  constexpr int TMEM_V_COLS_PER_GROUP = SLICES_PER_GROUP * N_CHUNK * VEC;
  constexpr int TMEM_V_COLS_TOTAL = CONSUMER_GROUPS * TMEM_V_COLS_PER_GROUP;
  static_assert(TMEM_V_COLS_TOTAL <= TMEM_COLS_ALLOC);
  static_assert(H >= 4096 && H <= 8192);
  static_assert(H % K_TILE == 0);

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int TB = T * B;
  const int num_ctas = gridDim.x;
  constexpr int num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

  const int comp_wid = wid - 1;
  const int comp_tid = tid - 32;
  const int group = (comp_wid >= 4) ? 1 : 0;
  const int ct_in_group =
      (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
  const int k_local = ct_in_group * VEC;

  constexpr size_t V_BYTES = (size_t)NUM_BUFS * H * sizeof(bf16_t);
  constexpr size_t DELTA_BYTES =
      HAS_DELTA ? (size_t)CHUNK_DEPTH * H * sizeof(bf16_t) : 0;
  constexpr size_t OUTPUT_NORM_BYTES =
      OUTPUT_NORM_IN_SMEM ? (size_t)H * sizeof(bf16_t) : 0;
  extern __shared__ __align__(16) char smem_raw[];
  bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw);  // [NUM_BUFS][H]
  bf16_t* delta_bufs = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES);
  bf16_t* output_norm_buf =
      reinterpret_cast<bf16_t*>(smem_raw + V_BYTES + DELTA_BYTES);
  FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(
      smem_raw + V_BYTES + DELTA_BYTES + OUTPUT_NORM_BYTES);

  auto slot_of = [](long long gci, int n) {
    return (int)(gci % CHUNK_DEPTH) * N_CHUNK + n;
  };
  auto phase_of = [](long long gci) { return (int)((gci / CHUNK_DEPTH) & 1); };
  auto buf_ptr = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };
  auto delta_buf_ptr = [&](int chunk_slot) -> bf16_t* {
    return delta_bufs + chunk_slot * H;
  };

  if (wid == 0 && elect_one_sync()) {
  #pragma unroll
    for (int i = 0; i < CHUNK_DEPTH; i++) {
      mbarrier_init(plan.bar_ready[i], 1);
      mbarrier_init(plan.bar_consumed[i], CONSUMER_WARPS);
    }
    if constexpr (OUTPUT_NORM_IN_SMEM) {
      mbarrier_init(plan.bar_output_norm_ready, 1);
    }
    fence_mbarrier_init();
  }

  // gdc wait BEFORE tmem alloc
  cudaGridDependencySynchronize();

  if (wid == 1) {
    tmem_allocate(TMEM_COLS_ALLOC, &plan.tmem_base);
    if constexpr (RELEASE_TMEM) {
      tmem_release_allocation_lock();
    }
  }
  __syncthreads();

  if constexpr (OUTPUT_NORM_IN_SMEM) {
    if (wid == 0 && elect_one_sync()) {
      mbarrier_expect_tx(plan.bar_output_norm_ready, H * (int)sizeof(bf16_t));
      cp_async_bulk(output_norm_buf, output_norm_weight, H * sizeof(bf16_t),
                    plan.bar_output_norm_ready);
    }
  }

  const uint32_t my_v_tmem =
      comp_tid >= 0 ? plan.tmem_base + group * TMEM_V_COLS_PER_GROUP : 0;
  float q_cache[ACC_PER_THREAD];
  if (comp_tid >= 0) {
  #pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      if constexpr (H == 7168) {
        if (si == SLICES_PER_GROUP - 1) {
          int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
          int2 rms_v = *reinterpret_cast<const int2*>(rms_w + h_base);
          int2 res_v = *reinterpret_cast<const int2*>(res_w + h_base);
          auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
          auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
  #pragma unroll
          for (int k = 0; k < 2; k++) {
            float2 rf = __bfloat1622float2(rms2[k]);
            float2 sf = __bfloat1622float2(res2[k]);
            q_cache[si * VEC + 2 * k] = rf.x * sf.x;
            q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
          }
          continue;
        }
      }
      int dt = si * CONSUMER_GROUPS + group;
      if (dt >= NHT) continue;
      int h_base = dt * K_TILE + k_local;
      int4 rms_v = *reinterpret_cast<const int4*>(rms_w + h_base);
      int4 res_v = *reinterpret_cast<const int4*>(res_w + h_base);
      auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
      auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
  #pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 rf = __bfloat1622float2(rms2[k]);
        float2 sf = __bfloat1622float2(res2[k]);
        q_cache[si * VEC + 2 * k] = rf.x * sf.x;
        q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
      }
    }
  }

  if (wid == 0) {
    if (elect_one_sync()) {
      long long gci = 0;
      for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
        const int t = tb / B;
        for (int ci = 0; ci < num_chunks; ci++, gci++) {
          int ns = ci * N_CHUNK;
          int an = min(N_CHUNK, N - ns);
          int chunk_slot = (int)(gci % CHUNK_DEPTH);
          int pc = phase_of(gci);
          mbarrier_wait(plan.bar_consumed[chunk_slot], pc ^ 1);
          int transaction_bytes = an * H * (int)sizeof(bf16_t);
          if constexpr (HAS_DELTA) {
            int prefix_n = N - 1 - ns;
            if (prefix_n >= 0 && prefix_n < an) {
              transaction_bytes += H * (int)sizeof(bf16_t);
            }
          }
          mbarrier_expect_tx(plan.bar_ready[chunk_slot], transaction_bytes);
  #pragma unroll
          for (int n = 0; n < N_CHUNK; n++) {
            if (n >= an) continue;
            int slot = slot_of(gci, n);
            const bf16_t* src =
                residual_addr(block_res, layer_res, ns + n, N, t,
                              block_stride_m, block_stride_r, H);
            cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t),
                          plan.bar_ready[chunk_slot]);
          }
          if constexpr (HAS_DELTA) {
            int prefix_n = N - 1 - ns;
            if (prefix_n >= 0 && prefix_n < an) {
              cp_async_bulk(delta_buf_ptr(chunk_slot),
                            delta + (long long)tb * H, H * sizeof(bf16_t),
                            plan.bar_ready[chunk_slot]);
            }
          }
        }
      }
    }
  } else {
    float acc32[ACC_PER_THREAD] = {};
    float eps_cache;
    asm volatile("mov.b32 %0, %1;" : "=f"(eps_cache) : "f"(rms_eps));

    long long gci = 0;
    for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
      float m_running = -FLT_MAX;
      float s_running = 0.f;
  #pragma unroll
      for (int i = 0; i < ACC_PER_THREAD; i++) {
        acc32[i] = 0.f;
      }

  #pragma unroll
      for (int ci = 0; ci < num_chunks; ci++, gci++) {
        int ns = ci * N_CHUNK;
        int an = min(N_CHUNK, N - ns);
        int chunk_slot = (int)(gci % CHUNK_DEPTH);
        int pr = phase_of(gci);
        mbarrier_wait(plan.bar_ready[chunk_slot], pr);

        float2 sq_local[N_CHUNK] = {};
        float2 dot_local[N_CHUNK] = {};

        auto pass_A_body = [&](auto AN_TOK) {
          constexpr int AN = decltype(AN_TOK)::value;
  #pragma unroll
          for (int si = 0; si < SLICES_PER_GROUP; si++) {
            if constexpr (H == 7168) {
              if (si == SLICES_PER_GROUP - 1) {
                int h_base =
                    6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                const float* qv = &q_cache[si * VEC];
  #pragma unroll
                for (int n = 0; n < AN; n++) {
                  int slot = slot_of(gci, n);
                  int2 vp =
                      *reinterpret_cast<const int2*>(buf_ptr(slot) + h_base);
                  auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                  if constexpr (HAS_DELTA) {
                    int prefix_n = N - 1 - ns;
                    if (n == prefix_n) {
                      const bf16_t* delta_ptr =
                          delta_buf_ptr(chunk_slot) + h_base;
  #pragma unroll
                      for (int j = 0; j < 2; j++) {
                        auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(
                            delta_ptr + 2 * j);
                        v2[j] = __hadd2(v2[j], delta2);
                      }
                      *reinterpret_cast<int2*>(layer_res + (long long)tb * H +
                                               h_base) = vp;
                    }
                  }
                  float2 f[2] = {__bfloat1622float2(v2[0]),
                                 __bfloat1622float2(v2[1])};
                  tmem_store<4>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
                  sq_local[n] = float2_fma(f[0], f[0], sq_local[n]);
                  sq_local[n] = float2_fma(f[1], f[1], sq_local[n]);
                  dot_local[n] =
                      float2_fma(f[0], make_float2(qv[0], qv[1]), dot_local[n]);
                  dot_local[n] =
                      float2_fma(f[1], make_float2(qv[2], qv[3]), dot_local[n]);
                }
                continue;
              }
            }
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT) continue;
            int h_base = dt * K_TILE + k_local;
            const float* qv = &q_cache[si * VEC];

  #pragma unroll
            for (int n = 0; n < AN; n++) {
              int slot = slot_of(gci, n);
              int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + h_base);
              auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
              if constexpr (HAS_DELTA) {
                int prefix_n = N - 1 - ns;
                if (n == prefix_n) {
                  const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
  #pragma unroll
                  for (int j = 0; j < VEC / 2; j++) {
                    auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(
                        delta_ptr + 2 * j);
                    v2[j] = __hadd2(v2[j], delta2);
                  }
                  *reinterpret_cast<int4*>(layer_res + (long long)tb * H +
                                           h_base) = vp;
                }
              }
              float2 f[4] = {
                  __bfloat1622float2(v2[0]), __bfloat1622float2(v2[1]),
                  __bfloat1622float2(v2[2]), __bfloat1622float2(v2[3])};
              tmem_store<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
  #pragma unroll
              for (int j = 0; j < VEC / 2; j++) {
                sq_local[n] = float2_fma(f[j], f[j], sq_local[n]);
                dot_local[n] = float2_fma(
                    f[j], make_float2(qv[2 * j], qv[2 * j + 1]), dot_local[n]);
              }
            }
          }
        };
        if constexpr (NC == 4) {
          switch (an) {
            case 4:
              pass_A_body(std::integral_constant<int, 4>{});
              break;
            case 3:
              pass_A_body(std::integral_constant<int, 3>{});
              break;
            case 2:
              pass_A_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_A_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        } else if constexpr (NC == 3) {
          switch (an) {
            case 3:
              pass_A_body(std::integral_constant<int, 3>{});
              break;
            case 2:
              pass_A_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_A_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        } else {
          static_assert(NC == 2);
          switch (an) {
            case 2:
              pass_A_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_A_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        }
        if (lane == 0) {
          mbarrier_arrive(plan.bar_consumed[chunk_slot]);
        }
        tmem_store_wait();

        float2 reduce_pair[N_CHUNK];
  #pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          reduce_pair[n] = make_float2(sq_local[n].x + sq_local[n].y,
                                       dot_local[n].x + dot_local[n].y);
        }
  #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
  #pragma unroll
          for (int n = 0; n < N_CHUNK; n++) {
            uint64_t packed = reinterpret_cast<uint64_t&>(reduce_pair[n]);
            packed = __shfl_xor_sync(0xffffffff, packed, offset);
            float2 other = reinterpret_cast<float2&>(packed);
            reduce_pair[n] = float2_add(reduce_pair[n], other);
          }
        }
        if (lane == 0) {
  #pragma unroll
          for (int n = 0; n < N_CHUNK; n++) {
            plan.ws_stats[comp_wid][n] = reduce_pair[n];
          }
        }
        named_barrier_sync(CONSUMER_THREADS, 0);

        float local_rsig = 0.f;
        float local_logit = 0.f;
        int stat_n = lane / CONSUMER_WARPS;
        int stat_w = lane % CONSUMER_WARPS;
        float2 totals = {};
        if (stat_n < N_CHUNK) {
          totals = plan.ws_stats[stat_w][stat_n];
        }
  #pragma unroll
        for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
          totals.x +=
              __shfl_down_sync(0xffffffff, totals.x, offset, CONSUMER_WARPS);
          totals.y +=
              __shfl_down_sync(0xffffffff, totals.y, offset, CONSUMER_WARPS);
        }
        if (stat_n < N_CHUNK && stat_w == 0) {
          local_rsig = rsqrtf(totals.x / H + eps_cache);
          local_logit = totals.y * local_rsig;
        }
        float logit_n[N_CHUNK];
  #pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          logit_n[n] = __shfl_sync(0xffffffff, local_logit, n * CONSUMER_WARPS);
        }

        float m_chunk = -FLT_MAX;
  #pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          if (n < an) m_chunk = fmaxf(m_chunk, logit_n[n]);
        }
        float m_new = fmaxf(m_running, m_chunk);
        float corr = exp2f((m_running - m_new) * LOG2_E);
        float w_n[N_CHUNK] = {};
        float w_sum = 0.f;
  #pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          if (n < an) {
            w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
            w_sum += w_n[n];
          }
        }

        auto pass_B_body = [&](auto AN_TOK) {
          constexpr int AN = decltype(AN_TOK)::value;
  #pragma unroll
          for (int si = 0; si < SLICES_PER_GROUP; si++) {
            if constexpr (H == 7168) {
              if (si == SLICES_PER_GROUP - 1) {
                float2 corr2 = make_float2(corr, corr);
                float2 a[2];
  #pragma unroll
                for (int j = 0; j < 2; j++) {
                  float2 old = make_float2(acc32[si * VEC + 2 * j],
                                           acc32[si * VEC + 2 * j + 1]);
                  a[j] = float2_mul(old, corr2);
                }
                float2 f_cache[AN][2];
  #pragma unroll
                for (int n = 0; n < AN; n++) {
                  tmem_load<4>(my_v_tmem + (si * N_CHUNK + n) * VEC,
                               f_cache[n]);
                }
  #pragma unroll
                for (int n = 0; n < AN; n++) {
                  float2 wn = make_float2(w_n[n], w_n[n]);
  #pragma unroll
                  for (int j = 0; j < 2; j++) {
                    a[j] = float2_fma(wn, f_cache[n][j], a[j]);
                  }
                }
  #pragma unroll
                for (int j = 0; j < 2; j++) {
                  acc32[si * VEC + 2 * j] = a[j].x;
                  acc32[si * VEC + 2 * j + 1] = a[j].y;
                }
                continue;
              }
            }
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT) continue;
            float2 corr2 = make_float2(corr, corr);
            float2 a[VEC / 2];
  #pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
              float2 old = make_float2(acc32[si * VEC + 2 * j],
                                       acc32[si * VEC + 2 * j + 1]);
              a[j] = float2_mul(old, corr2);
            }
            float2 f_cache[AN][VEC / 2];
  #pragma unroll
            for (int n = 0; n < AN; n++) {
              tmem_load<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f_cache[n]);
            }
  #pragma unroll
            for (int n = 0; n < AN; n++) {
              float2 wn = make_float2(w_n[n], w_n[n]);
  #pragma unroll
              for (int j = 0; j < VEC / 2; j++) {
                a[j] = float2_fma(wn, f_cache[n][j], a[j]);
              }
            }
  #pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
              acc32[si * VEC + 2 * j] = a[j].x;
              acc32[si * VEC + 2 * j + 1] = a[j].y;
            }
          }
        };
        if constexpr (NC == 4) {
          switch (an) {
            case 4:
              pass_B_body(std::integral_constant<int, 4>{});
              break;
            case 3:
              pass_B_body(std::integral_constant<int, 3>{});
              break;
            case 2:
              pass_B_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_B_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        } else if constexpr (NC == 3) {
          switch (an) {
            case 3:
              pass_B_body(std::integral_constant<int, 3>{});
              break;
            case 2:
              pass_B_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_B_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        } else {
          static_assert(NC == 2);
          switch (an) {
            case 2:
              pass_B_body(std::integral_constant<int, 2>{});
              break;
            case 1:
              pass_B_body(std::integral_constant<int, 1>{});
              break;
            default:
              __builtin_unreachable();
          }
        }

        s_running = s_running * corr + w_sum;
        m_running = m_new;
      }

      float inv_s = 1.f / s_running;
      bf16_t* out_ptr = output + (long long)tb * H;
      float2 output_sq_pair = {};
      // When output RMSNorm is fused, the softmax denominator cancels:
      // (acc / s) * rsqrt(mean((acc / s)^2) + eps)
      //   = acc * rsqrt(mean(acc^2) + eps * s^2).
  #pragma unroll
      for (int si = 0; si < SLICES_PER_GROUP; si++) {
        if constexpr (H == 7168) {
          if (si == SLICES_PER_GROUP - 1) {
            int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
            uint2 packed;
            auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
            float2 inv2 = make_float2(inv_s, inv_s);
  #pragma unroll
            for (int j = 0; j < 2; j++) {
              float2 old = make_float2(acc32[si * VEC + 2 * j],
                                       acc32[si * VEC + 2 * j + 1]);
              if constexpr (HAS_OUTPUT_NORM) {
                output_sq_pair = float2_fma(old, old, output_sq_pair);
              } else {
                float2 mixed = float2_mul(old, inv2);
                ov2[j] = __float22bfloat162_rn(mixed);
              }
            }
            if constexpr (!HAS_OUTPUT_NORM) {
              *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
            }
            continue;
          }
        }
        int dt = si * CONSUMER_GROUPS + group;
        if (dt >= NHT) continue;
        int h_base = dt * K_TILE + k_local;
        uint4 packed;
        auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
        float2 inv2 = make_float2(inv_s, inv_s);
  #pragma unroll
        for (int j = 0; j < VEC / 2; j++) {
          float2 old =
              make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
          if constexpr (HAS_OUTPUT_NORM) {
            output_sq_pair = float2_fma(old, old, output_sq_pair);
          } else {
            float2 mixed = float2_mul(old, inv2);
            ov2[j] = __float22bfloat162_rn(mixed);
          }
        }
        if constexpr (!HAS_OUTPUT_NORM) {
          *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
        }
      }

      if constexpr (HAS_OUTPUT_NORM) {
        if constexpr (OUTPUT_NORM_IN_SMEM) {
          // The immutable weight copy is acquired once, at its first use.
          if (tb == blockIdx.x) {
            mbarrier_wait(plan.bar_output_norm_ready, 0);
          }
        }
        float output_sq = output_sq_pair.x + output_sq_pair.y;
  #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          output_sq += __shfl_xor_sync(0xffffffff, output_sq, offset);
        }
        if (lane == 0) {
          plan.ws_stats[comp_wid][0] = make_float2(output_sq, 0.f);
        }
        named_barrier_sync(CONSUMER_THREADS, 0);
        float total_sq = lane < CONSUMER_WARPS ? plan.ws_stats[lane][0].x : 0.f;
  #pragma unroll
        for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
          total_sq +=
              __shfl_down_sync(0xffffffff, total_sq, offset, CONSUMER_WARPS);
        }
        if (lane == 0) {
          total_sq =
              rsqrtf(total_sq / H + output_norm_eps * s_running * s_running);
        }
        float output_rsigma = __shfl_sync(0xffffffff, total_sq, 0);
  #pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
          if constexpr (H == 7168) {
            if (si == SLICES_PER_GROUP - 1) {
              int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
              uint2 packed;
              auto* values = reinterpret_cast<bf16_t*>(&packed);
  #pragma unroll
              for (int j = 0; j < 4; j++) {
                const bf16_t* weight_ptr =
                    OUTPUT_NORM_IN_SMEM ? output_norm_buf : output_norm_weight;
                float weight = __bfloat162float(weight_ptr[h_base + j]);
                values[j] = __float2bfloat16(acc32[si * VEC + j] *
                                             output_rsigma * weight);
              }
              *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
              continue;
            }
          }
          int dt = si * CONSUMER_GROUPS + group;
          if (dt >= NHT) continue;
          int h_base = dt * K_TILE + k_local;
          uint4 packed;
          auto* values = reinterpret_cast<bf16_t*>(&packed);
  #pragma unroll
          for (int j = 0; j < VEC; j++) {
            const bf16_t* weight_ptr =
                OUTPUT_NORM_IN_SMEM ? output_norm_buf : output_norm_weight;
            float weight = __bfloat162float(weight_ptr[h_base + j]);
            values[j] =
                __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
          }
          *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
        }
      }
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
  __syncthreads();
  if (wid == 1) {
    tmem_free(plan.tmem_base, TMEM_COLS_ALLOC);
  }
#else
  if (threadIdx.x == 0) {
    printf("attn_res_fwd_online_v2_kernel requires sm_10x\n");
  }
#endif
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false,
          bool HAS_DELTA = false, bool HAS_OUTPUT_NORM = false,
          bool OUTPUT_NORM_IN_SMEM = false>
static void launch_fwd(const bf16_t* block_residual, bf16_t* layer_residual,
                       const bf16_t* delta, const bf16_t* res_weight,
                       const bf16_t* rms_weight, bf16_t* output, int T,
                       float rms_eps, int num_sm, cudaStream_t stream,
                       const bf16_t* output_norm_weight = nullptr,
                       float output_norm_eps = 0.f, int block_stride_m = 0,
                       int block_stride_r = 0) {
  constexpr size_t smem_size =
      ((size_t)CHUNK_DEPTH * (NC + (HAS_DELTA ? 1 : 0)) * H * sizeof(bf16_t) +
       (OUTPUT_NORM_IN_SMEM ? (size_t)H * sizeof(bf16_t) : 0) +
       sizeof(FwdSmemPlan<NC>) + 15) &
      ~size_t(15);
  auto kernel =
      &attn_res_fwd_online_v2_kernel<H, N, NC, 1, RELEASE_TMEM, HAS_DELTA,
                                     HAS_OUTPUT_NORM, OUTPUT_NORM_IN_SMEM>;
  static bool attrs_set = false;
  if (!attrs_set) {
    if (smem_size > 48 * 1024) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    attrs_set = true;
  }
  int grid = RELEASE_TMEM ? num_sm * 2 : num_sm;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = BLK;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, block_residual, layer_residual, delta,
                     res_weight, rms_weight, output, T, block_stride_m,
                     block_stride_r, rms_eps, output_norm_weight,
                     output_norm_eps);
}

}  // namespace fwd_prod_v2
}  // namespace sm100

void kimi_k3_attn_res(torch::stable::Tensor& prefix,
                      torch::stable::Tensor const& delta,
                      torch::stable::Tensor const& blocks,
                      torch::stable::Tensor const& norm_weight,
                      torch::stable::Tensor const& qk_weight,
                      torch::stable::Tensor const& output_norm_weight,
                      torch::stable::Tensor& output, int64_t num_blocks,
                      double eps, double output_norm_eps) {
  int const num_tokens = static_cast<int>(prefix.size(0));
  int const device = prefix.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 10,
                  "Kimi K3 AttnRes requires the SM100 family");
  int64_t const hidden_size = prefix.size(1);
  STD_TORCH_CHECK(prefix.stride(0) == hidden_size &&
                      delta.stride(0) == hidden_size &&
                      output.stride(0) == hidden_size,
                  "Kimi K3 AttnRes requires densely packed rows; got strides ",
                  prefix.stride(0), ", ", delta.stride(0), ", ",
                  output.stride(0), " for hidden_size ", hidden_size);

  using namespace sm100::fwd_prod_v2;
  // Two-source chunks and two resident CTAs are beneficial once setup is
  // amortized by the long, full eight-block prefill workload.
  if (num_blocks == 8 && num_tokens >= 4096) {
    launch_fwd<7168, 9, 2, true, true, true, true>(
        static_cast<bf16_t const*>(blocks.data_ptr()),
        static_cast<bf16_t*>(prefix.data_ptr()),
        static_cast<bf16_t const*>(delta.data_ptr()),
        static_cast<bf16_t const*>(qk_weight.data_ptr()),
        static_cast<bf16_t const*>(norm_weight.data_ptr()),
        static_cast<bf16_t*>(output.data_ptr()), num_tokens,
        static_cast<float>(eps), properties->multiProcessorCount,
        get_current_cuda_stream(device),
        static_cast<bf16_t const*>(output_norm_weight.data_ptr()),
        static_cast<float>(output_norm_eps), static_cast<int>(blocks.stride(0)),
        static_cast<int>(blocks.stride(1)));
  } else {
    auto dispatch = [&](auto nsrc_tok) {
      constexpr int NSRC = decltype(nsrc_tok)::value;
      launch_fwd<7168, NSRC, 3, false, true, true, true>(
          static_cast<bf16_t const*>(blocks.data_ptr()),
          static_cast<bf16_t*>(prefix.data_ptr()),
          static_cast<bf16_t const*>(delta.data_ptr()),
          static_cast<bf16_t const*>(qk_weight.data_ptr()),
          static_cast<bf16_t const*>(norm_weight.data_ptr()),
          static_cast<bf16_t*>(output.data_ptr()), num_tokens,
          static_cast<float>(eps), properties->multiProcessorCount,
          get_current_cuda_stream(device),
          static_cast<bf16_t const*>(output_norm_weight.data_ptr()),
          static_cast<float>(output_norm_eps),
          static_cast<int>(blocks.stride(0)),
          static_cast<int>(blocks.stride(1)));
    };
    // The source count is num_blocks + 1; specialising on it makes the chunk
    // count, per-chunk source count and prefix-chunk test compile-time.
    switch (num_blocks) {
      case 1:
        dispatch(std::integral_constant<int, 2>{});
        break;
      case 2:
        dispatch(std::integral_constant<int, 3>{});
        break;
      case 3:
        dispatch(std::integral_constant<int, 4>{});
        break;
      case 4:
        dispatch(std::integral_constant<int, 5>{});
        break;
      case 5:
        dispatch(std::integral_constant<int, 6>{});
        break;
      case 6:
        dispatch(std::integral_constant<int, 7>{});
        break;
      case 7:
        dispatch(std::integral_constant<int, 8>{});
        break;
      case 8:
        dispatch(std::integral_constant<int, 9>{});
        break;
      default:
        STD_TORCH_CHECK(false, "Kimi K3 AttnRes: num_blocks must be 1..8");
    }
  }
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "Kimi K3 AttnRes kernel launch failed: ", cudaGetErrorString(error));
}
