// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused QK RMSNorm + RoPE + KV Cache Write + FP8 Per-Tensor Quantisation
//
// v3 – Warp-per-head design (based on fused_qknorm_rope_kernel.cu skeleton):
//   Each warp independently processes one (token, head) pair.
//   Q warp:  RMSNorm → RoPE → write back to qkv in-place
//   KV warp: V fire-and-forget store → K RMSNorm → RoPE → store k_cache
//
//   V stores overlap with K's warp-shuffle RMSNorm reduction,
//   hiding V's store latency.  No shared memory needed.
//
// Build (standalone, for rapid testing):
//   torch.utils.cpp_extension.load(
//       name="fused_qknrc",
//       sources=["csrc/fused_qk_norm_rope_cache_quant.cu"],
//       extra_cuda_cflags=["-O3", "--use_fast_math"])

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

// ── FP8 vectorized store type ────────────────────────────────────────
// Maps numElemsPerThread (= byte count for FP8) to a vector store type.
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

// ── Main fused kernel (v3 – warp-per-head) ──────────────────────────
//
// Template params:
//   scalar_t_in    – model dtype (c10::BFloat16 or c10::Half)
//   scalar_t_cache – cos/sin cache dtype (may differ from model dtype)
//   head_dim       – compile-time head dimension (64, 128, 256)
//   interleave     – true = GPT-J style RoPE, false = GPT-NeoX style
//   IS_FP8         – true = write KV cache as FP8 E4M3
//
// Grid: divUp(num_tokens * (num_heads_q + num_heads_kv), warpsPerBlock)
// Block: 256 threads = 8 warps
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

    // ── Warp / lane indices ──
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // ── Warp → (token, head) mapping ──
    int const total_qk_heads = num_heads_q + num_heads_kv;
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;
    if (tokenIdx >= num_tokens) return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    // ── Per-thread element count & vectorized types ──
    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64 (warp of 32, even elems)");
    constexpr int numElemsPerThread = head_dim / 32;
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(T_in);
    static_assert(elemSizeBytes % 4 == 0,
                  "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename packed_as<uint, vecSize>::type;
    constexpr int num_packed = elemSizeBytes / sizeof(T2_in);

    // ── QKV layout: [T, (nq + 2*nkv) * hd] ──
    int const q_size = num_heads_q * head_dim;
    int const kv_size = num_heads_kv * head_dim;
    int const total_qkv = q_size + 2 * kv_size;

    // ==============================================================
    //  Issue loads: KV warp issues V and K back-to-back;
    //  Q warp issues Q only.
    //  Both LDGs enter the memory pipeline simultaneously so
    //  K's load latency is hidden behind V processing.
    // ==============================================================

    // Q/K source offset
    int const qk_offset =
        isQ ? (tokenIdx * total_qkv + headIdx * head_dim)
            : (tokenIdx * total_qkv + q_size + headIdx * head_dim);
    int const qk_thread_offset = qk_offset + laneId * numElemsPerThread;

    // KV warp: issue V load first (LDG #1)
    vec_T v_vec;
    if (!isQ) {
      int const v_offset = tokenIdx * total_qkv + q_size + kv_size +
                           headIdx * head_dim + laneId * numElemsPerThread;
      v_vec = *reinterpret_cast<vec_T const*>(&qkv[v_offset]);
    }

    // All warps: issue Q or K load (LDG #2 for KV warps)
    // For KV warps, both V and K loads are now in-flight.
    vec_T qk_vec = *reinterpret_cast<vec_T const*>(&qkv[qk_thread_offset]);

    // ==============================================================
    //  KV warp: process V (stalls on v_vec; K load continues in bg)
    // ==============================================================
    if (!isQ) {
      int64_t const slot_idx = slot_mapping[tokenIdx];
      if (slot_idx >= 0) {
        int64_t const blk_idx = slot_idx / block_size;
        int64_t const blk_off = slot_idx % block_size;
        int64_t const v_cache_offset =
            (blk_idx * block_size + blk_off) *
                (int64_t)num_heads_kv * head_dim +
            headIdx * head_dim + laneId * numElemsPerThread;

        if constexpr (IS_FP8) {
          uint8_t* v_cache = reinterpret_cast<uint8_t*>(v_cache_void);
          uint8_t fp8_vals[numElemsPerThread];
#pragma unroll
          for (int i = 0; i < num_packed; i++) {
            T2_in packed_val =
                *(reinterpret_cast<T2_in const*>(&v_vec) + i);
            float2 vals = Converter::convert(packed_val);
            fp8_vals[2 * i] = float_to_fp8_e4m3(vals.x / v_scale);
            fp8_vals[2 * i + 1] = float_to_fp8_e4m3(vals.y / v_scale);
          }
          using fp8_vec_t =
              typename fp8_store_type<numElemsPerThread>::type;
          *reinterpret_cast<fp8_vec_t*>(&v_cache[v_cache_offset]) =
              *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
        } else {
          T_in* v_cache = reinterpret_cast<T_in*>(v_cache_void);
          *reinterpret_cast<vec_T*>(&v_cache[v_cache_offset]) = v_vec;
        }
        // V stores are fire-and-forget.
      }
    }

    // ==============================================================
    //  Unpack Q/K from qk_vec & compute RMSNorm
    //  For KV warps, K load likely completed during V processing.
    // ==============================================================
    float elements[numElemsPerThread];
    float sumOfSquares = 0.0f;

    {
#pragma unroll
      for (int i = 0; i < num_packed; i++) {
        T2_in packed_val =
            *(reinterpret_cast<T2_in const*>(&qk_vec) + i);
        float2 vals = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x + vals.y * vals.y;
        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    // Warp-level RMSNorm reduction (no shared memory)
    sumOfSquares = warpReduceSum(sumOfSquares);
    float const rms_rcp =
        rsqrtf(sumOfSquares / static_cast<float>(head_dim) + epsilon);

    // Normalise & apply weight
    T_in const* weight = isQ ? q_weight : k_weight;
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim = laneId * numElemsPerThread + i;
      float w = Converter::convert(weight[dim]);
      elements[i] *= rms_rcp * w;
    }

    // ==============================================================
    //  RoPE  (identical to fused_qknorm_rope_kernel.cu skeleton)
    // ==============================================================
    int64_t const pos_id = positions[tokenIdx];
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / numElemsPerThread;

    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // GPT-J style: pairs are (2i, 2i+1) within each lane's elements
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          int const dim_idx = laneId * numElemsPerThread + idx0;

          float const val0 = elements[idx0];
          float const val1 = elements[idx1];

          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // GPT-NeoX style: pairs across warp halves via shuffle
        __syncwarp();
        int const pairOffset = (rotary_dim / 2) / numElemsPerThread;
        float elements2[numElemsPerThread];
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] =
              __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);
          if (laneId < pairOffset) elements2[i] = -elements2[i];

          int dim_idx = laneId * numElemsPerThread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        __syncwarp();
      }
    }

    // ==============================================================
    //  Store results
    // ==============================================================
    if (isQ) {
      // Q: write back in-place to qkv
      int const q_offset = tokenIdx * total_qkv + headIdx * head_dim +
                           laneId * numElemsPerThread;
      vec_T vec;
#pragma unroll
      for (int i = 0; i < num_packed; i++) {
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[q_offset]) = vec;
    } else {
      // K: write to paged k_cache
      int64_t const slot_idx = slot_mapping[tokenIdx];
      if (slot_idx >= 0) {
        int64_t const blk_idx = slot_idx / block_size;
        int64_t const blk_off = slot_idx % block_size;
        int64_t const k_cache_offset =
            (blk_idx * block_size + blk_off) *
                (int64_t)num_heads_kv * head_dim +
            headIdx * head_dim + laneId * numElemsPerThread;

        if constexpr (IS_FP8) {
          uint8_t* k_cache = reinterpret_cast<uint8_t*>(k_cache_void);
          uint8_t fp8_vals[numElemsPerThread];
#pragma unroll
          for (int i = 0; i < numElemsPerThread; i++) {
            fp8_vals[i] = float_to_fp8_e4m3(elements[i] / k_scale);
          }
          using fp8_vec_t =
              typename fp8_store_type<numElemsPerThread>::type;
          *reinterpret_cast<fp8_vec_t*>(&k_cache[k_cache_offset]) =
              *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
        } else {
          T_in* k_cache = reinterpret_cast<T_in*>(k_cache_void);
          vec_T vec;
#pragma unroll
          for (int i = 0; i < num_packed; i++) {
            T2_in packed_val = Converter::convert(
                make_float2(elements[2 * i], elements[2 * i + 1]));
            *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
          }
          *reinterpret_cast<vec_T*>(&k_cache[k_cache_offset]) = vec;
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
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalWarps = num_tokens * (num_heads_q + num_heads_kv);
  int const gridSize = divUp(totalWarps, warpsPerBlock);

#define LAUNCH_KERNEL(HD, INTERLEAVE, IS_FP8)                               \
  fused_kernel<scalar_t_in, scalar_t_cache, HD, INTERLEAVE, IS_FP8>        \
      <<<gridSize, blockSize, 0, stream>>>(                                 \
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
