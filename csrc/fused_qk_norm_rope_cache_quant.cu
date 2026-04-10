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
//   NeoX path (!interleave): contiguous-within-half thread mapping
//     Lane j holds elements [PPT*j .. PPT*j+PPT-1] in each half,
//     e.g. for head_dim=128 (PPT=2): lo=[2j,2j+1], hi=[64+2j,64+2j+1].
//     RoPE pairs (d, d+HALF) in same thread → zero shuffle, zero syncwarp.
//     Vectorized loads: vec2 for hd=128, vec4 for hd=256.
//
//   GPT-J path (interleave): contiguous thread mapping
//     Lane j holds elements [j*EPT .. j*EPT+EPT-1].
//     RoPE pairs (2i, 2i+1) in same lane → zero shuffle.
//
//   Key optimizations over v3:
//     1. Contiguous-within-half mapping for NeoX (vec2/vec4 loads, zero shuffle)
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

// ── NeoX RMSNorm + RoPE (warp-level, contiguous-within-half layout) ──
//
// r_lo/r_hi hold PPT elements each.  Lane j owns contiguous elements
// [PPT*j .. PPT*j+PPT-1] in each half of the head.
template <int head_dim>
__device__ __forceinline__ void rmsnorm_rope_neox(
    float* r_lo, float* r_hi,
    float const* w_lo, float const* w_hi,
    float const* r_cos, float const* r_sin,
    int lane, int embed_dim, float epsilon) {
  constexpr int PPT = head_dim / 2 / 32;

  float var = 0.f;
#pragma unroll
  for (int p = 0; p < PPT; ++p)
    var += r_lo[p] * r_lo[p] + r_hi[p] * r_hi[p];
  float const inv_rms =
      rsqrtf(warpReduceSum(var) / static_cast<float>(head_dim) + epsilon);

#pragma unroll
  for (int p = 0; p < PPT; ++p) {
    r_lo[p] *= inv_rms * w_lo[p];
    r_hi[p] *= inv_rms * w_hi[p];
  }

#pragma unroll
  for (int p = 0; p < PPT; ++p) {
    int const dim = PPT * lane + p;
    if (dim < embed_dim) {
      float const new_lo = r_lo[p] * r_cos[p] - r_hi[p] * r_sin[p];
      float const new_hi = r_hi[p] * r_cos[p] + r_lo[p] * r_sin[p];
      r_lo[p] = new_lo;
      r_hi[p] = new_hi;
    }
  }
}

// ── GPT-J RMSNorm + RoPE (warp-level, contiguous layout) ────────────
//
// elements[] holds numEPT contiguous elements per lane.
// sumSq is pre-accumulated from the vectorized load phase.
template <typename CacheConverter, typename T_cache, int numEPT>
__device__ __forceinline__ void rmsnorm_rope_gptj(
    float* elements, float sumSq,
    float const* weights,
    T_cache const* cos_ptr, T_cache const* sin_ptr,
    int lane, int rotary_lanes, float epsilon, int head_dim) {
  float const inv_rms =
      rsqrtf(warpReduceSum(sumSq) / static_cast<float>(head_dim) + epsilon);

#pragma unroll
  for (int i = 0; i < numEPT; i++)
    elements[i] *= inv_rms * weights[i];

  if (lane < rotary_lanes) {
#pragma unroll
    for (int i = 0; i < numEPT / 2; ++i) {
      int const idx0 = 2 * i;
      int const idx1 = 2 * i + 1;
      int const dim_idx = lane * numEPT + idx0;
      int const half_dim = dim_idx / 2;
      float const cos_val =
          CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
      float const sin_val =
          CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));
      float const v0 = elements[idx0];
      float const v1 = elements[idx1];
      elements[idx0] = v0 * cos_val - v1 * sin_val;
      elements[idx1] = v0 * sin_val + v1 * cos_val;
    }
  }
}

// ── NeoX load/store helpers (vectorized for PPT >= 2) ────────────────

template <typename Converter, int head_dim>
__device__ __forceinline__ void load_head_neox(
    typename Converter::hip_type const* src,
    float* lo, float* hi, int lane) {
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;
  constexpr int HALF = head_dim / 2;
  constexpr int PPT = HALF / 32;

  if constexpr (PPT == 1) {
    lo[0] = Converter::convert(src[lane]);
    hi[0] = Converter::convert(src[lane + HALF]);
  } else {
    constexpr int halfBytes = PPT * sizeof(T_in);
    constexpr int vecSize = halfBytes / 4;
    using vec_T = typename packed_as<uint, vecSize>::type;
    constexpr int num_packed = halfBytes / sizeof(T2_in);
    int const thr_off = PPT * lane;

    vec_T v_lo = *reinterpret_cast<vec_T const*>(&src[thr_off]);
#pragma unroll
    for (int i = 0; i < num_packed; i++) {
      float2 vals = Converter::convert(
          *(reinterpret_cast<T2_in const*>(&v_lo) + i));
      lo[2 * i] = vals.x;
      lo[2 * i + 1] = vals.y;
    }

    vec_T v_hi = *reinterpret_cast<vec_T const*>(&src[HALF + thr_off]);
#pragma unroll
    for (int i = 0; i < num_packed; i++) {
      float2 vals = Converter::convert(
          *(reinterpret_cast<T2_in const*>(&v_hi) + i));
      hi[2 * i] = vals.x;
      hi[2 * i + 1] = vals.y;
    }
  }
}

template <typename Converter, int head_dim>
__device__ __forceinline__ void store_head_neox(
    typename Converter::hip_type* dst,
    float const* lo, float const* hi, int lane) {
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;
  constexpr int HALF = head_dim / 2;
  constexpr int PPT = HALF / 32;

  if constexpr (PPT == 1) {
    dst[lane] = Converter::convert(lo[0]);
    dst[lane + HALF] = Converter::convert(hi[0]);
  } else {
    constexpr int halfBytes = PPT * sizeof(T_in);
    constexpr int vecSize = halfBytes / 4;
    using vec_T = typename packed_as<uint, vecSize>::type;
    constexpr int num_packed = halfBytes / sizeof(T2_in);
    int const thr_off = PPT * lane;

    vec_T vec_lo;
#pragma unroll
    for (int i = 0; i < num_packed; i++) {
      T2_in packed = Converter::convert(
          make_float2(lo[2 * i], lo[2 * i + 1]));
      *(reinterpret_cast<T2_in*>(&vec_lo) + i) = packed;
    }
    *reinterpret_cast<vec_T*>(&dst[thr_off]) = vec_lo;

    vec_T vec_hi;
#pragma unroll
    for (int i = 0; i < num_packed; i++) {
      T2_in packed = Converter::convert(
          make_float2(hi[2 * i], hi[2 * i + 1]));
      *(reinterpret_cast<T2_in*>(&vec_hi) + i) = packed;
    }
    *reinterpret_cast<vec_T*>(&dst[HALF + thr_off]) = vec_hi;
  }
}

template <typename Converter, int head_dim, bool IS_FP8>
__device__ __forceinline__ void write_cache_neox(
    void* cache_void, int64_t offset,
    float const* lo, float const* hi,
    float scale, int lane) {
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;
  constexpr int HALF = head_dim / 2;
  constexpr int PPT = HALF / 32;

  if constexpr (IS_FP8) {
    uint8_t* cache = reinterpret_cast<uint8_t*>(cache_void);
    if constexpr (PPT == 1) {
      cache[offset + lane] = float_to_fp8_e4m3(lo[0] / scale);
      cache[offset + lane + HALF] = float_to_fp8_e4m3(hi[0] / scale);
    } else {
      int const thr_off = PPT * lane;
      uint8_t fp8_lo[PPT];
#pragma unroll
      for (int i = 0; i < PPT; i++)
        fp8_lo[i] = float_to_fp8_e4m3(lo[i] / scale);
      using fp8_vec_t = typename fp8_store_type<PPT>::type;
      *reinterpret_cast<fp8_vec_t*>(&cache[offset + thr_off]) =
          *reinterpret_cast<fp8_vec_t const*>(fp8_lo);

      uint8_t fp8_hi[PPT];
#pragma unroll
      for (int i = 0; i < PPT; i++)
        fp8_hi[i] = float_to_fp8_e4m3(hi[i] / scale);
      *reinterpret_cast<fp8_vec_t*>(&cache[offset + HALF + thr_off]) =
          *reinterpret_cast<fp8_vec_t const*>(fp8_hi);
    }
  } else {
    T_in* cache = reinterpret_cast<T_in*>(cache_void);
    if constexpr (PPT == 1) {
      cache[offset + lane] = Converter::convert(lo[0]);
      cache[offset + lane + HALF] = Converter::convert(hi[0]);
    } else {
      constexpr int halfBytes = PPT * sizeof(T_in);
      constexpr int vecSize = halfBytes / 4;
      using vec_T = typename packed_as<uint, vecSize>::type;
      constexpr int num_packed = halfBytes / sizeof(T2_in);
      int const thr_off = PPT * lane;

      vec_T vec_lo;
#pragma unroll
      for (int i = 0; i < num_packed; i++) {
        T2_in packed = Converter::convert(
            make_float2(lo[2 * i], lo[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec_lo) + i) = packed;
      }
      *reinterpret_cast<vec_T*>(&cache[offset + thr_off]) = vec_lo;

      vec_T vec_hi;
#pragma unroll
      for (int i = 0; i < num_packed; i++) {
        T2_in packed = Converter::convert(
            make_float2(hi[2 * i], hi[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec_hi) + i) = packed;
      }
      *reinterpret_cast<vec_T*>(&cache[offset + HALF + thr_off]) = vec_hi;
    }
  }
}

// ── GPT-J load/store helpers ────────────────────────────────────────

template <typename Converter, int head_dim>
__device__ __forceinline__ float load_head_gptj(
    typename Converter::hip_type const* src, int thr_off,
    float* elements) {
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;
  constexpr int numEPT = head_dim / 32;
  constexpr int elemSizeBytes = numEPT * sizeof(T_in);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as<uint, vecSize>::type;
  constexpr int num_packed = elemSizeBytes / sizeof(T2_in);

  vec_T v = *reinterpret_cast<vec_T const*>(&src[thr_off]);
  float sumSq = 0.f;
#pragma unroll
  for (int i = 0; i < num_packed; i++) {
    T2_in packed = *(reinterpret_cast<T2_in const*>(&v) + i);
    float2 vals = Converter::convert(packed);
    sumSq += vals.x * vals.x + vals.y * vals.y;
    elements[2 * i] = vals.x;
    elements[2 * i + 1] = vals.y;
  }
  return sumSq;
}

template <typename Converter, int head_dim>
__device__ __forceinline__ void store_head_gptj(
    typename Converter::hip_type* dst, int thr_off,
    float const* elements) {
  using T2_in = typename Converter::packed_hip_type;
  using T_in = typename Converter::hip_type;
  constexpr int numEPT = head_dim / 32;
  constexpr int elemSizeBytes = numEPT * sizeof(T_in);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as<uint, vecSize>::type;
  constexpr int num_packed = elemSizeBytes / sizeof(T2_in);

  vec_T vec;
#pragma unroll
  for (int i = 0; i < num_packed; i++) {
    T2_in packed = Converter::convert(
        make_float2(elements[2 * i], elements[2 * i + 1]));
    *(reinterpret_cast<T2_in*>(&vec) + i) = packed;
  }
  *reinterpret_cast<vec_T*>(&dst[thr_off]) = vec;
}

template <typename Converter, int head_dim, bool IS_FP8>
__device__ __forceinline__ void write_cache_gptj(
    void* cache_void, int64_t offset, int thr_off,
    float const* elements, float scale) {
  using T_in = typename Converter::hip_type;
  using T2_in = typename Converter::packed_hip_type;
  constexpr int numEPT = head_dim / 32;
  constexpr int elemSizeBytes = numEPT * sizeof(T_in);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as<uint, vecSize>::type;
  constexpr int num_packed = elemSizeBytes / sizeof(T2_in);

  if constexpr (IS_FP8) {
    uint8_t* cache = reinterpret_cast<uint8_t*>(cache_void);
    uint8_t fp8_vals[numEPT];
#pragma unroll
    for (int i = 0; i < numEPT; i++)
      fp8_vals[i] = float_to_fp8_e4m3(elements[i] / scale);
    using fp8_vec_t = typename fp8_store_type<numEPT>::type;
    *reinterpret_cast<fp8_vec_t*>(&cache[offset + thr_off]) =
        *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
  } else {
    T_in* cache = reinterpret_cast<T_in*>(cache_void);
    vec_T vec;
#pragma unroll
    for (int i = 0; i < num_packed; i++) {
      T2_in packed = Converter::convert(
          make_float2(elements[2 * i], elements[2 * i + 1]));
      *(reinterpret_cast<T2_in*>(&vec) + i) = packed;
    }
    *reinterpret_cast<vec_T*>(&cache[offset + thr_off]) = vec;
  }
}

// ── Main fused kernel (unified warp scheduling) ────────────────
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
    void* __restrict__ query_void,       // [T, nq, hd] – mutated in-place
    void* __restrict__ key_void,         // [T, nkv, hd] – mutated in-place
    void const* __restrict__ value_void, // [T, nkv, hd] – read only
    void* __restrict__ k_cache_void,     // paged [B, blk_sz, nkv, hd]
    void* __restrict__ v_cache_void,     // paged [B, blk_sz, nkv, hd]
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

    T_in* query = reinterpret_cast<T_in*>(query_void);
    T_in* key = reinterpret_cast<T_in*>(key_void);
    T_in const* value = reinterpret_cast<T_in const*>(value_void);
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

    // ── Per-token strides ──
    int const q_stride = num_heads_q * head_dim;
    int const kv_stride = num_heads_kv * head_dim;

    // ── Token pointers (separate Q/K/V buffers) ──
    T_in* q_in = query + (int64_t)token_idx * q_stride;
    T_in* k_in = key + (int64_t)token_idx * kv_stride;
    T_in const* v_in = value + (int64_t)token_idx * kv_stride;

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
      //  NeoX PATH: Contiguous-within-half thread mapping
      //  Lane j holds: lo[p] = elem[PPT*j + p]
      //                hi[p] = elem[PPT*j + p + HALF]
      //  RoPE pair (lo[p], hi[p]) → same thread, zero shuffle
      //  Vec load: PPT contiguous elements per half per thread
      // ============================================================

      static_assert(head_dim % 64 == 0,
                    "head_dim must be divisible by 64 for NeoX mapping");
      constexpr int HALF = head_dim / 2;
      constexpr int PAIRS_PER_THREAD = HALF / 32;
      int const thr_off_neox = PAIRS_PER_THREAD * lane;

      float r_cos[PAIRS_PER_THREAD], r_sin[PAIRS_PER_THREAD];
#pragma unroll
      for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
        int const dim = thr_off_neox + p;
        if (dim < embed_dim) {
          r_cos[p] = CacheConverter::convert(VLLM_LDG(cos_ptr + dim));
          r_sin[p] = CacheConverter::convert(VLLM_LDG(sin_ptr + dim));
        }
      }

      float w_q_lo[PAIRS_PER_THREAD], w_q_hi[PAIRS_PER_THREAD];
      load_head_neox<Converter, head_dim>(q_weight, w_q_lo, w_q_hi, lane);

      for (int op = warp_id; op < total_ops; op += num_warps) {
        if (op == 0) {
          // ── V: cache write ──
          if (valid_slot) {
            float v_lo[PAIRS_PER_THREAD], v_hi[PAIRS_PER_THREAD];
            load_head_neox<Converter, head_dim>(
                v_in + kv_head * head_dim, v_lo, v_hi, lane);
            write_cache_neox<Converter, head_dim, IS_FP8>(
                v_cache_void, cache_head_offset, v_lo, v_hi, v_scale, lane);
          }

        } else if (op == 1) {
          // ── K: RMSNorm + RoPE + cache write + writeback ──
          if (valid_slot) {
            float w_k_lo[PAIRS_PER_THREAD], w_k_hi[PAIRS_PER_THREAD];
            load_head_neox<Converter, head_dim>(k_weight, w_k_lo, w_k_hi, lane);

            float k_lo[PAIRS_PER_THREAD], k_hi[PAIRS_PER_THREAD];
            load_head_neox<Converter, head_dim>(
                k_in + kv_head * head_dim, k_lo, k_hi, lane);

            rmsnorm_rope_neox<head_dim>(
                k_lo, k_hi, w_k_lo, w_k_hi,
                r_cos, r_sin, lane, embed_dim, epsilon);

            write_cache_neox<Converter, head_dim, IS_FP8>(
                k_cache_void, cache_head_offset, k_lo, k_hi, k_scale, lane);
            store_head_neox<Converter, head_dim>(
                k_in + kv_head * head_dim, k_lo, k_hi, lane);
          }

        } else {
          // ── Q: RMSNorm + RoPE + writeback ──
          int const q_head = q_start + (op - 2);
          T_in* q_ptr = q_in + q_head * head_dim;

          float q_lo[PAIRS_PER_THREAD], q_hi[PAIRS_PER_THREAD];
          load_head_neox<Converter, head_dim>(q_ptr, q_lo, q_hi, lane);

          rmsnorm_rope_neox<head_dim>(
              q_lo, q_hi, w_q_lo, w_q_hi,
              r_cos, r_sin, lane, embed_dim, epsilon);

          store_head_neox<Converter, head_dim>(q_ptr, q_lo, q_hi, lane);
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
      constexpr int numEPT = head_dim / 32;
      int const rotary_lanes = rotary_dim / numEPT;
      int const thr_off = lane * numEPT;

      float w_q[numEPT];
      load_head_gptj<Converter, head_dim>(q_weight, thr_off, w_q);

      for (int op = warp_id; op < total_ops; op += num_warps) {
        if (op == 0) {
          // ── V: cache write ──
          if (valid_slot) {
            float v_elems[numEPT];
            load_head_gptj<Converter, head_dim>(
                v_in + kv_head * head_dim, thr_off, v_elems);
            write_cache_gptj<Converter, head_dim, IS_FP8>(
                v_cache_void, cache_head_offset, thr_off, v_elems, v_scale);
          }

        } else if (op == 1) {
          // ── K: RMSNorm + RoPE + cache write + writeback ──
          if (valid_slot) {
            float w_k[numEPT];
            load_head_gptj<Converter, head_dim>(k_weight, thr_off, w_k);

            float elements[numEPT];
            float sumSq = load_head_gptj<Converter, head_dim>(
                k_in + kv_head * head_dim, thr_off, elements);

            rmsnorm_rope_gptj<CacheConverter, T_cache, numEPT>(
                elements, sumSq, w_k,
                cos_ptr, sin_ptr, lane, rotary_lanes, epsilon, head_dim);

            write_cache_gptj<Converter, head_dim, IS_FP8>(
                k_cache_void, cache_head_offset, thr_off, elements, k_scale);
            store_head_gptj<Converter, head_dim>(
                k_in + kv_head * head_dim, thr_off, elements);
          }

        } else {
          // ── Q: RMSNorm + RoPE + writeback ──
          int const q_head = q_start + (op - 2);
          T_in* q_ptr = q_in + q_head * head_dim;

          float elements[numEPT];
          float sumSq = load_head_gptj<Converter, head_dim>(
              q_ptr, thr_off, elements);

          rmsnorm_rope_gptj<CacheConverter, T_cache, numEPT>(
              elements, sumSq, w_q,
              cos_ptr, sin_ptr, lane, rotary_lanes, epsilon, head_dim);

          store_head_gptj<Converter, head_dim>(q_ptr, thr_off, elements);
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
void launchFusedKernel(void* query, void* key, void const* value,
                       void* k_cache, void* v_cache,
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
          query, key, value, k_cache, v_cache, q_weight, k_weight,         \
          cos_sin_cache, positions, slot_mapping, k_scale, v_scale,         \
          epsilon, num_heads_q, num_heads_kv, block_size, num_tokens,       \
          rotary_dim)

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
    torch::Tensor query,   // [T, nq, hd] – mutated in-place
    torch::Tensor key,     // [T, nkv, hd] – mutated in-place
    torch::Tensor value,   // [T, nkv, hd] – read only
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
  at::cuda::OptionalCUDAGuard const guard(query.device());
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_HALF_TYPES(
      query.scalar_type(), "fused_qk_norm_rope_cache_quant", [&] {
        using qkv_scalar_t = scalar_t;
        VLLM_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "fused_qk_norm_rope_cache_quant",
            [&] {
              using cache_scalar_t = scalar_t;
              fused_qknrc::launchFusedKernel<qkv_scalar_t, cache_scalar_t>(
                  query.data_ptr(), key.data_ptr(), value.data_ptr(),
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
