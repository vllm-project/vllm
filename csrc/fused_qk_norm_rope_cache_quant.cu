// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused QK RMSNorm + RoPE + KV Cache Write + FP8 Per-Tensor Quantisation
//
// v4 – Optimized warp-per-head with unified scheduling:
//   Grid: 2D (num_tokens, num_heads_kv) — one block per (token, KV group)
//   Block: dynamic warp count = min(2 + gqa_ratio, 5) * 32
//   Scheduling: V/K/Q ops distributed round-robin across warps
//
//   NeoX path (!interleave): interleaved thread mapping
//     Lane j holds elements [j, j+32, j+64, j+96] for head_dim=128.
//     RoPE pairs (j, j+HALF) in same thread → zero shuffle, zero syncwarp.
//
//   GPT-J path (interleave): contiguous thread mapping
//     Lane j holds elements [j*EPT .. j*EPT+EPT-1].
//     RoPE pairs (2i, 2i+1) in same lane → zero shuffle.
//
//   Key optimizations over v3:
//     1. Interleaved thread mapping for NeoX RoPE (zero cross-thread comm)
//     2. Unified warp scheduling (V/K/Q round-robin, no phase split)
//     3. 2D grid (token × kv_group parallelism)
//     4. Dynamic warp count (5 warps optimal for GQA ratio=4)
//     5. Lazy K weight loading (only in the warp that processes K)

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

// FP8 header – available since CUDA 11.8.
// HW-accelerated conversion requires SM >= 89 (Ada / Hopper).
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#define HAS_CUDA_FP8 1
#else
#define HAS_CUDA_FP8 0
#endif

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL
  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
  #endif
#else
  #define FINAL_MASK 0xffffffff
#endif

namespace fused_qknrc {

// ── Vectorized type helpers (from TRT-LLM) ──────────────────────────
template <typename T, int num>
struct packed_as;
template <>
struct packed_as<uint, 1> {
  using type = uint;
};
template <>
struct packed_as<uint, 2> {
  using type = uint2;
};
template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

// ── Warp-level reduction ─────────────────────────────────────────────
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// ── FP8 E4M3 conversion ─────────────────────────────────────────────
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
#if HAS_CUDA_FP8
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
#else
  #error "FP8 support requires CUDA 11.8+"
#endif
}

// ── FP8 vectorized store type (for GPT-J contiguous path) ───────────
template <int N>
struct fp8_store_type;
template <>
struct fp8_store_type<2> {
  using type = uint16_t;
};
template <>
struct fp8_store_type<4> {
  using type = uint32_t;
};
template <>
struct fp8_store_type<8> {
  using type = uint2;
};

// ── Main fused kernel (v4 – unified warp scheduling) ────────────────
//
// Template params:
//   scalar_t_in    – model dtype (c10::BFloat16 or c10::Half)
//   scalar_t_cache – cos/sin cache dtype (may differ from model dtype)
//   head_dim       – compile-time head dimension (64, 128, 256)
//   interleave     – true = GPT-J style RoPE, false = GPT-NeoX style
//   IS_FP8         – true = write KV cache as FP8 E4M3
//
// Grid: dim3(num_tokens, num_heads_kv)
// Block: min(2 + gqa_ratio, 5) * 32 threads
// Shared memory: 0

template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave, bool IS_FP8>
__global__ void fused_kernel(
    void* __restrict__ qkv_void,  // [T, (nq+2*nkv)*hd] – Q in-place
    void* __restrict__ k_cache_void,  // paged [B, blk_sz, nkv, hd]
    void* __restrict__ v_cache_void,  // paged [B, blk_sz, nkv, hd]
    void const* __restrict__ q_weight_void,  // [head_dim]
    void const* __restrict__ k_weight_void,  // [head_dim]
    void const* __restrict__ cos_sin_cache_void,  // [max_pos, rotary_dim]
    int64_t const* __restrict__ positions,  // [T]
    int64_t const* __restrict__ slot_mapping,  // [T]
    float const k_scale,
    float const v_scale,
    float const epsilon,
    int const num_heads_q,
    int const num_heads_kv,
    int const block_size,
    int const num_tokens,
    int const rotary_dim) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    // ── Type aliases ──
    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input data type is not supported for this CUDA architecture");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache data type is not supported for this CUDA architecture");
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    // ── 2D grid indexing ──
    int const token_idx = blockIdx.x;
    int const kv_head = blockIdx.y;
    if (token_idx >= num_tokens) return;

    // ── Warp / lane ──
    int const warp_id = threadIdx.x / 32;
    int const lane = threadIdx.x & 31;
    int const num_warps = blockDim.x / 32;

    // ── GQA scheduling ──
    int const gqa_ratio = num_heads_q / num_heads_kv;
    int const total_ops = 2 + gqa_ratio;  // V + K + Q heads
    int const q_start = kv_head * gqa_ratio;

    // ── QKV layout: [T, (nq + 2*nkv) * hd] ──
    int const q_size = num_heads_q * head_dim;
    int const kv_size = num_heads_kv * head_dim;
    int const total_qkv = q_size + 2 * kv_size;

    // ── Token pointers ──
    T_in* tok_base = qkv + (int64_t)token_idx * total_qkv;
    T_in const* q_in = tok_base;
    T_in const* k_in = tok_base + q_size;
    T_in const* v_in = tok_base + q_size + kv_size;

    // ── Cos/sin cache ──
    int64_t const pos = positions[token_idx];
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cos_sin_cache + pos * rotary_dim;
    T_cache const* sin_ptr = cos_ptr + embed_dim;

    // ── Cache address setup ──
    int64_t const slot_idx = slot_mapping[token_idx];
    int64_t cache_head_offset = 0;
    bool const valid_slot = (slot_idx >= 0);
    if (valid_slot) {
      int64_t const blk_idx = slot_idx / block_size;
      int64_t const blk_off = slot_idx % block_size;
      cache_head_offset =
          blk_idx * (int64_t)block_size * num_heads_kv * head_dim +
          blk_off * (int64_t)num_heads_kv * head_dim +
          (int64_t)kv_head * head_dim;
    }

    // ==============================================================
    //  Compile-time branch: NeoX vs GPT-J mapping
    // ==============================================================

    if constexpr (!interleave) {
      // ============================================================
      //  NeoX PATH: Interleaved thread mapping
      //  Lane j holds: r_lo[p] = elem[j + p*32]
      //                r_hi[p] = elem[j + p*32 + HALF]
      //  RoPE pair (r_lo[p], r_hi[p]) → same thread, zero shuffle
      // ============================================================

      static_assert(head_dim % 64 == 0,
                    "head_dim must be divisible by 64 for interleaved mapping");
      constexpr int HALF = head_dim / 2;
      constexpr int PAIRS_PER_THREAD = HALF / 32;

      // Pre-load cos/sin into registers
      float r_cos[PAIRS_PER_THREAD], r_sin[PAIRS_PER_THREAD];
#pragma unroll
      for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
        int const dim = lane + p * 32;
        if (dim < embed_dim) {
          r_cos[p] = CacheConverter::convert(VLLM_LDG(cos_ptr + dim));
          r_sin[p] = CacheConverter::convert(VLLM_LDG(sin_ptr + dim));
        }
      }

      // Pre-load Q weights (most warps process Q heads)
      float w_q_lo[PAIRS_PER_THREAD], w_q_hi[PAIRS_PER_THREAD];
#pragma unroll
      for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
        int const lo_idx = lane + p * 32;
        w_q_lo[p] = Converter::convert(q_weight[lo_idx]);
        w_q_hi[p] = Converter::convert(q_weight[lo_idx + HALF]);
      }

      // ── Unified round-robin scheduling ──
      for (int op = warp_id; op < total_ops; op += num_warps) {
        if (op == 0) {
          // ── V: fire-and-forget cache write ──
          if (valid_slot) {
            T_in const* v_head = v_in + kv_head * head_dim;
            if constexpr (IS_FP8) {
              uint8_t* v_cache =
                  reinterpret_cast<uint8_t*>(v_cache_void);
#pragma unroll
              for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
                int const lo_idx = lane + p * 32;
                float lo = Converter::convert(v_head[lo_idx]);
                float hi = Converter::convert(v_head[lo_idx + HALF]);
                v_cache[cache_head_offset + lo_idx] =
                    float_to_fp8_e4m3(lo / v_scale);
                v_cache[cache_head_offset + lo_idx + HALF] =
                    float_to_fp8_e4m3(hi / v_scale);
              }
            } else {
              T_in* v_cache = reinterpret_cast<T_in*>(v_cache_void);
#pragma unroll
              for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
                int const lo_idx = lane + p * 32;
                v_cache[cache_head_offset + lo_idx] = v_head[lo_idx];
                v_cache[cache_head_offset + lo_idx + HALF] =
                    v_head[lo_idx + HALF];
              }
            }
          }

        } else if (op == 1) {
          // ── K: RMSNorm + RoPE + cache write ──
          if (valid_slot) {
            // Lazy load K weights (only this warp needs them)
            float w_k_lo[PAIRS_PER_THREAD], w_k_hi[PAIRS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
              int const lo_idx = lane + p * 32;
              w_k_lo[p] = Converter::convert(k_weight[lo_idx]);
              w_k_hi[p] = Converter::convert(k_weight[lo_idx + HALF]);
            }

            // Load K
            T_in const* k_head = k_in + kv_head * head_dim;
            float k_lo[PAIRS_PER_THREAD], k_hi[PAIRS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
              int const lo_idx = lane + p * 32;
              k_lo[p] = Converter::convert(k_head[lo_idx]);
              k_hi[p] = Converter::convert(k_head[lo_idx + HALF]);
            }

            // RMSNorm
            float var = 0.f;
#pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p)
              var += k_lo[p] * k_lo[p] + k_hi[p] * k_hi[p];
            float const inv_rms = rsqrtf(
                warpReduceSum(var) / static_cast<float>(head_dim) + epsilon);
#pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
              k_lo[p] *= inv_rms * w_k_lo[p];
              k_hi[p] *= inv_rms * w_k_hi[p];
            }

            // RoPE (NeoX: purely local, zero shuffle)
#pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
              int const dim = lane + p * 32;
              if (dim < embed_dim) {
                float const new_lo =
                    k_lo[p] * r_cos[p] - k_hi[p] * r_sin[p];
                float const new_hi =
                    k_hi[p] * r_cos[p] + k_lo[p] * r_sin[p];
                k_lo[p] = new_lo;
                k_hi[p] = new_hi;
              }
            }

            // Store K to cache
            if constexpr (IS_FP8) {
              uint8_t* k_cache =
                  reinterpret_cast<uint8_t*>(k_cache_void);
#pragma unroll
              for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
                int const lo_idx = lane + p * 32;
                k_cache[cache_head_offset + lo_idx] =
                    float_to_fp8_e4m3(k_lo[p] / k_scale);
                k_cache[cache_head_offset + lo_idx + HALF] =
                    float_to_fp8_e4m3(k_hi[p] / k_scale);
              }
            } else {
              T_in* k_cache = reinterpret_cast<T_in*>(k_cache_void);
#pragma unroll
              for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
                int const lo_idx = lane + p * 32;
                k_cache[cache_head_offset + lo_idx] =
                    Converter::convert(k_lo[p]);
                k_cache[cache_head_offset + lo_idx + HALF] =
                    Converter::convert(k_hi[p]);
              }
            }
          }

        } else {
          // ── Q: RMSNorm + RoPE + in-place writeback ──
          int const q_head = q_start + (op - 2);
          T_in const* q_head_ptr = q_in + q_head * head_dim;

          // Load Q
          float q_lo[PAIRS_PER_THREAD], q_hi[PAIRS_PER_THREAD];
#pragma unroll
          for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
            int const lo_idx = lane + p * 32;
            q_lo[p] = Converter::convert(q_head_ptr[lo_idx]);
            q_hi[p] = Converter::convert(q_head_ptr[lo_idx + HALF]);
          }

          // RMSNorm
          float var = 0.f;
#pragma unroll
          for (int p = 0; p < PAIRS_PER_THREAD; ++p)
            var += q_lo[p] * q_lo[p] + q_hi[p] * q_hi[p];
          float const inv_rms = rsqrtf(
              warpReduceSum(var) / static_cast<float>(head_dim) + epsilon);
#pragma unroll
          for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
            q_lo[p] *= inv_rms * w_q_lo[p];
            q_hi[p] *= inv_rms * w_q_hi[p];
          }

          // RoPE (NeoX: purely local, zero shuffle)
#pragma unroll
          for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
            int const dim = lane + p * 32;
            if (dim < embed_dim) {
              float const new_lo =
                  q_lo[p] * r_cos[p] - q_hi[p] * r_sin[p];
              float const new_hi =
                  q_hi[p] * r_cos[p] + q_lo[p] * r_sin[p];
              q_lo[p] = new_lo;
              q_hi[p] = new_hi;
            }
          }

          // Write Q back in-place to qkv
          T_in* q_dst = tok_base + q_head * head_dim;
#pragma unroll
          for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
            int const lo_idx = lane + p * 32;
            q_dst[lo_idx] = Converter::convert(q_lo[p]);
            q_dst[lo_idx + HALF] = Converter::convert(q_hi[p]);
          }
        }
      }

    } else {
      // ============================================================
      //  GPT-J PATH: Contiguous thread mapping
      //  Lane j holds elements [j*EPT .. j*EPT+EPT-1]
      //  RoPE pairs (2i, 2i+1) already in same lane
      // ============================================================

      static_assert(head_dim % (32 * 2) == 0,
                    "head_dim must be divisible by 64");
      constexpr int numElemsPerThread = head_dim / 32;
      constexpr int elemSizeBytes = numElemsPerThread * sizeof(T_in);
      static_assert(elemSizeBytes % 4 == 0,
                    "elemSizeBytes must be a multiple of 4");
      constexpr int vecSize = elemSizeBytes / 4;
      using vec_T = typename packed_as<uint, vecSize>::type;
      constexpr int num_packed = elemSizeBytes / sizeof(T2_in);
      int const rotary_lanes = rotary_dim / numElemsPerThread;

      // Pre-load Q weights (contiguous layout)
      float w_q[numElemsPerThread];
#pragma unroll
      for (int i = 0; i < num_packed; i++) {
        T2_in packed =
            *(reinterpret_cast<T2_in const*>(
                  &q_weight[lane * numElemsPerThread]) +
              i);
        float2 vals = Converter::convert(packed);
        w_q[2 * i] = vals.x;
        w_q[2 * i + 1] = vals.y;
      }

      // ── Unified round-robin scheduling ──
      for (int op = warp_id; op < total_ops; op += num_warps) {
        if (op == 0) {
          // ── V: vectorized load + cache write ──
          if (valid_slot) {
            T_in const* v_head = v_in + kv_head * head_dim;
            int const thr_off = lane * numElemsPerThread;
            vec_T v_vec =
                *reinterpret_cast<vec_T const*>(&v_head[thr_off]);

            if constexpr (IS_FP8) {
              uint8_t* v_cache =
                  reinterpret_cast<uint8_t*>(v_cache_void);
              uint8_t fp8_vals[numElemsPerThread];
#pragma unroll
              for (int i = 0; i < num_packed; i++) {
                T2_in packed =
                    *(reinterpret_cast<T2_in const*>(&v_vec) + i);
                float2 vals = Converter::convert(packed);
                fp8_vals[2 * i] =
                    float_to_fp8_e4m3(vals.x / v_scale);
                fp8_vals[2 * i + 1] =
                    float_to_fp8_e4m3(vals.y / v_scale);
              }
              using fp8_vec_t =
                  typename fp8_store_type<numElemsPerThread>::type;
              *reinterpret_cast<fp8_vec_t*>(
                  &v_cache[cache_head_offset + thr_off]) =
                  *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
            } else {
              T_in* v_cache =
                  reinterpret_cast<T_in*>(v_cache_void);
              *reinterpret_cast<vec_T*>(
                  &v_cache[cache_head_offset + thr_off]) = v_vec;
            }
          }

        } else if (op == 1) {
          // ── K: RMSNorm + GPT-J RoPE + cache write ──
          if (valid_slot) {
            // Lazy load K weights
            float w_k[numElemsPerThread];
#pragma unroll
            for (int i = 0; i < num_packed; i++) {
              T2_in packed =
                  *(reinterpret_cast<T2_in const*>(
                        &k_weight[lane * numElemsPerThread]) +
                    i);
              float2 vals = Converter::convert(packed);
              w_k[2 * i] = vals.x;
              w_k[2 * i + 1] = vals.y;
            }

            // Load K
            T_in const* k_head = k_in + kv_head * head_dim;
            vec_T k_vec = *reinterpret_cast<vec_T const*>(
                &k_head[lane * numElemsPerThread]);
            float elements[numElemsPerThread];
            float sumSq = 0.f;
#pragma unroll
            for (int i = 0; i < num_packed; i++) {
              T2_in packed =
                  *(reinterpret_cast<T2_in const*>(&k_vec) + i);
              float2 vals = Converter::convert(packed);
              sumSq += vals.x * vals.x + vals.y * vals.y;
              elements[2 * i] = vals.x;
              elements[2 * i + 1] = vals.y;
            }

            // RMSNorm
            float const inv_rms = rsqrtf(
                warpReduceSum(sumSq) / static_cast<float>(head_dim) +
                epsilon);
#pragma unroll
            for (int i = 0; i < numElemsPerThread; i++)
              elements[i] *= inv_rms * w_k[i];

            // GPT-J RoPE: pairs (2i, 2i+1) in same lane
            if (lane < rotary_lanes) {
#pragma unroll
              for (int i = 0; i < numElemsPerThread / 2; ++i) {
                int const idx0 = 2 * i;
                int const idx1 = 2 * i + 1;
                int const dim_idx = lane * numElemsPerThread + idx0;
                int const half_dim = dim_idx / 2;
                float const cos_val = CacheConverter::convert(
                    VLLM_LDG(cos_ptr + half_dim));
                float const sin_val = CacheConverter::convert(
                    VLLM_LDG(sin_ptr + half_dim));
                float const v0 = elements[idx0];
                float const v1 = elements[idx1];
                elements[idx0] = v0 * cos_val - v1 * sin_val;
                elements[idx1] = v0 * sin_val + v1 * cos_val;
              }
            }

            // Store K to cache
            int const thr_off = lane * numElemsPerThread;
            if constexpr (IS_FP8) {
              uint8_t* k_cache =
                  reinterpret_cast<uint8_t*>(k_cache_void);
              uint8_t fp8_vals[numElemsPerThread];
#pragma unroll
              for (int i = 0; i < numElemsPerThread; i++)
                fp8_vals[i] =
                    float_to_fp8_e4m3(elements[i] / k_scale);
              using fp8_vec_t =
                  typename fp8_store_type<numElemsPerThread>::type;
              *reinterpret_cast<fp8_vec_t*>(
                  &k_cache[cache_head_offset + thr_off]) =
                  *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
            } else {
              T_in* k_cache =
                  reinterpret_cast<T_in*>(k_cache_void);
              vec_T vec;
#pragma unroll
              for (int i = 0; i < num_packed; i++) {
                T2_in packed = Converter::convert(
                    make_float2(elements[2 * i], elements[2 * i + 1]));
                *(reinterpret_cast<T2_in*>(&vec) + i) = packed;
              }
              *reinterpret_cast<vec_T*>(
                  &k_cache[cache_head_offset + thr_off]) = vec;
            }
          }

        } else {
          // ── Q: RMSNorm + GPT-J RoPE + in-place writeback ──
          int const q_head = q_start + (op - 2);
          T_in const* q_head_ptr = q_in + q_head * head_dim;
          int const thr_off = lane * numElemsPerThread;

          // Load Q
          vec_T q_vec = *reinterpret_cast<vec_T const*>(
              &q_head_ptr[thr_off]);
          float elements[numElemsPerThread];
          float sumSq = 0.f;
#pragma unroll
          for (int i = 0; i < num_packed; i++) {
            T2_in packed =
                *(reinterpret_cast<T2_in const*>(&q_vec) + i);
            float2 vals = Converter::convert(packed);
            sumSq += vals.x * vals.x + vals.y * vals.y;
            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
          }

          // RMSNorm
          float const inv_rms = rsqrtf(
              warpReduceSum(sumSq) / static_cast<float>(head_dim) +
              epsilon);
#pragma unroll
          for (int i = 0; i < numElemsPerThread; i++)
            elements[i] *= inv_rms * w_q[i];

          // GPT-J RoPE: pairs (2i, 2i+1) in same lane
          if (lane < rotary_lanes) {
#pragma unroll
            for (int i = 0; i < numElemsPerThread / 2; ++i) {
              int const idx0 = 2 * i;
              int const idx1 = 2 * i + 1;
              int const dim_idx = lane * numElemsPerThread + idx0;
              int const half_dim = dim_idx / 2;
              float const cos_val = CacheConverter::convert(
                  VLLM_LDG(cos_ptr + half_dim));
              float const sin_val = CacheConverter::convert(
                  VLLM_LDG(sin_ptr + half_dim));
              float const v0 = elements[idx0];
              float const v1 = elements[idx1];
              elements[idx0] = v0 * cos_val - v1 * sin_val;
              elements[idx1] = v0 * sin_val + v1 * cos_val;
            }
          }

          // Write Q back in-place to qkv
          T_in* q_dst = tok_base + q_head * head_dim;
          vec_T vec;
#pragma unroll
          for (int i = 0; i < num_packed; i++) {
            T2_in packed = Converter::convert(
                make_float2(elements[2 * i], elements[2 * i + 1]));
            *(reinterpret_cast<T2_in*>(&vec) + i) = packed;
          }
          *reinterpret_cast<vec_T*>(&q_dst[thr_off]) = vec;
        }
      }
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ── Dispatch helpers ─────────────────────────────────────────────────

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

#define DISPATCH_FP8(is_fp8, IS_FP8, ...) \
  if (is_fp8) {                            \
    const bool IS_FP8 = true;              \
    __VA_ARGS__                            \
  } else {                                 \
    const bool IS_FP8 = false;             \
    __VA_ARGS__                            \
  }

// ── Launcher ─────────────────────────────────────────────────────────

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedKernel(void* qkv, void* k_cache, void* v_cache,
                       void const* q_weight, void const* k_weight,
                       void const* cos_sin_cache,
                       int64_t const* positions,
                       int64_t const* slot_mapping, float k_scale,
                       float v_scale, float epsilon, int num_heads_q,
                       int num_heads_kv, int head_dim, int block_size,
                       int num_tokens, int rotary_dim, bool interleave,
                       bool is_fp8, cudaStream_t stream) {
  // 2D grid: one block per (token, KV group)
  dim3 const grid(num_tokens, num_heads_kv);

  // Dynamic warp count: enough to cover all ops in ~1 round
  int const gqa_ratio = num_heads_q / num_heads_kv;
  int const total_ops = 2 + gqa_ratio;
  constexpr int MAX_WARPS = 5;
  int const warps_per_block = total_ops < MAX_WARPS ? total_ops : MAX_WARPS;
  int const blockSize = warps_per_block * 32;

#define LAUNCH_KERNEL(HD, INTERLEAVE, IS_FP8)                               \
  fused_kernel<scalar_t_in, scalar_t_cache, HD, INTERLEAVE, IS_FP8>        \
      <<<grid, blockSize, 0, stream>>>(                                     \
          qkv, k_cache, v_cache, q_weight, k_weight, cos_sin_cache,        \
          positions, slot_mapping, k_scale, v_scale, epsilon, num_heads_q,  \
          num_heads_kv, block_size, num_tokens, rotary_dim)

  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        DISPATCH_FP8(is_fp8, IS_FP8, { LAUNCH_KERNEL(64, INTERLEAVE, IS_FP8); });
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        DISPATCH_FP8(is_fp8, IS_FP8, {
          LAUNCH_KERNEL(128, INTERLEAVE, IS_FP8);
        });
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        DISPATCH_FP8(is_fp8, IS_FP8, {
          LAUNCH_KERNEL(256, INTERLEAVE, IS_FP8);
        });
      });
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head_dim for fused_qk_norm_rope_cache_quant: ",
                  head_dim);
  }

#undef LAUNCH_KERNEL
}

}  // namespace fused_qknrc

// ── Torch wrapper ────────────────────────────────────────────────────

void fused_qk_norm_rope_cache_quant(
    torch::Tensor qkv,  // [T, (nq + 2*nkv) * hd] – Q segment mutated in-place
    torch::Tensor k_cache,  // [num_blocks, block_size, nkv, hd]
    torch::Tensor v_cache,  // same shape
    torch::Tensor q_weight,  // [head_dim]
    torch::Tensor k_weight,  // [head_dim]
    torch::Tensor cos_sin_cache,  // [max_pos, rotary_dim]
    torch::Tensor positions,  // [T]
    torch::Tensor slot_mapping,  // [T]
    double k_scale,
    double v_scale,
    double epsilon,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t block_size,
    bool is_neox,
    bool is_fp8) {
  int const num_tokens = positions.numel();
  if (num_tokens == 0) return;

  int const rotary_dim = cos_sin_cache.size(1);
  at::cuda::OptionalCUDAGuard const guard(qkv.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qk_norm_rope_cache_quant", [&] {
        using qkv_scalar_t = scalar_t;
        VLLM_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "fused_qk_norm_rope_cache_quant",
            [&] {
              using cache_scalar_t = scalar_t;
              fused_qknrc::launchFusedKernel<qkv_scalar_t, cache_scalar_t>(
                  qkv.data_ptr(),
                  is_fp8
                      ? reinterpret_cast<void*>(k_cache.data_ptr<uint8_t>())
                      : k_cache.data_ptr(),
                  is_fp8
                      ? reinterpret_cast<void*>(v_cache.data_ptr<uint8_t>())
                      : v_cache.data_ptr(),
                  q_weight.data_ptr(), k_weight.data_ptr(),
                  cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(),
                  slot_mapping.data_ptr<int64_t>(),
                  static_cast<float>(k_scale), static_cast<float>(v_scale),
                  static_cast<float>(epsilon),
                  static_cast<int>(num_heads_q),
                  static_cast<int>(num_heads_kv),
                  static_cast<int>(head_dim), static_cast<int>(block_size),
                  num_tokens, rotary_dim,
                  /*interleave=*/!is_neox, is_fp8, stream);
            });
      });
}
