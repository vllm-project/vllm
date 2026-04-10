// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused QK RMSNorm + RoPE + KV Cache Write + FP8 Per-Tensor Quantisation
//
// v2 – Interleaved V-store / K-compute design:
//   Phase 1 (per KV head):
//     Load V[h] → FP8 convert → store v_cache  (fire-and-forget)
//     Load K[h] → smem → RMSNorm → RoPE → store k_cache
//   Phase 2 (per Q head):
//     Load Q[h] → smem → RMSNorm → RoPE → store query in-place
//
// V stores are non-blocking and overlap with K's RMSNorm computation,
// effectively hiding V's store latency for free.
//
// Build (standalone, for rapid testing):
//   torch.utils.cpp_extension.load(
//       name="fused_qknrc",
//       sources=["csrc/fused_qk_norm_rope_cache_quant.cu"],
//       extra_cuda_cflags=["-O3", "--use_fast_math"])

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "dispatch_utils.h"

// FP8 header – available since CUDA 11.8.
// HW-accelerated conversion requires SM >= 89 (Ada / Hopper).
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#define HAS_CUDA_FP8 1
#else
#define HAS_CUDA_FP8 0
#endif

namespace fused_qknrc_v2 {

// ── Warp / block reduction ────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask);
  return val;
}

// Returns the sum across the full block.  `reduce_smem` must have room for
// at least (blockDim.x / 32) floats.
__device__ float block_reduce_sum(float val, float* reduce_smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) reduce_smem[wid] = val;
  __syncthreads();

  const int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? reduce_smem[threadIdx.x] : 0.f;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;
}

// ── FP8 E4M3 conversion helpers ──────────────────────────────────────

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
#if HAS_CUDA_FP8 && __CUDA_ARCH__ >= 890
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
#else
  // Software fallback – saturate to E4M3 range [-448, 448], then cast
  // through half for a rough approximation.  Good enough for correctness
  // testing on Ampere / older GPUs.
  val = fminf(fmaxf(val, -448.f), 448.f);
  __half h = __float2half_rn(val);
  unsigned short bits = *reinterpret_cast<unsigned short*>(&h);
  uint8_t sign = (bits >> 15) & 1;
  int exp_h = ((bits >> 10) & 0x1F) - 15;
  int man_h = (bits >> 7) & 0x7;
  int exp_fp8 = exp_h + 7;
  if (exp_fp8 <= 0) {
    return (sign << 7);
  }
  if (exp_fp8 >= 15) {
    return (sign << 7) | 0x7E;
  }
  return (sign << 7) | ((exp_fp8 & 0xF) << 3) | (man_h & 0x7);
#endif
}

// ── Cache write helpers ──────────────────────────────────────────────

// Write a single element to the paged cache (FP8 or model dtype).
template <typename scalar_t, bool IS_FP8>
__device__ __forceinline__ void write_cache_elem(
    void* cache_ptr, int idx, float val, float scale) {
  if constexpr (IS_FP8) {
    reinterpret_cast<uint8_t*>(cache_ptr)[idx] =
        float_to_fp8_e4m3(val / scale);
  } else {
    reinterpret_cast<scalar_t*>(cache_ptr)[idx] =
        static_cast<scalar_t>(val);
  }
}

// ── Main fused kernel (v2 – interleaved V/K) ────────────────────────
//
// Template params:
//   scalar_t  – BFloat16 or Half (model dtype)
//   IS_NEOX   – true = GPT-NeoX style RoPE, false = GPT-J style
//   IS_FP8    – true = write KV cache as FP8 E4M3, false = same as scalar_t

template <typename scalar_t, bool IS_NEOX, bool IS_FP8>
__global__ void fused_kernel(
    // ── outputs / mutated inputs ──
    scalar_t* __restrict__ query,          // [T, num_heads_q, head_dim]
    scalar_t* __restrict__ key,            // [T, num_heads_kv, head_dim]
    void* __restrict__ k_cache,            // NHD [B, blk_sz, nkv, hd]
    void* __restrict__ v_cache,            // NHD [B, blk_sz, nkv, hd_v]
    // ── inputs ──
    const scalar_t* __restrict__ value,    // [T, num_heads_kv, head_dim_v]
    const scalar_t* __restrict__ q_weight, // [head_dim]
    const scalar_t* __restrict__ k_weight, // [head_dim]
    const scalar_t* __restrict__ cos_sin_cache, // [max_pos, rot_dim]
    const int64_t* __restrict__ positions,      // [T]
    const int64_t* __restrict__ slot_mapping,   // [T]
    // ── scalars ──
    const float k_scale, const float v_scale, const float epsilon,
    const int num_heads_q, const int num_heads_kv, const int head_dim,
    const int head_dim_v, const int block_size) {

  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const int q_size = num_heads_q * head_dim;
  const int kv_size = num_heads_kv * head_dim;
  const int v_size = num_heads_kv * head_dim_v;
  const int embed_dim = head_dim / 2;  // rotary half

  // ── Token pointers ──
  scalar_t* q_in = query + (int64_t)token_idx * q_size;
  scalar_t* k_in = key + (int64_t)token_idx * kv_size;
  const scalar_t* v_in = value + (int64_t)token_idx * v_size;

  // ── Cos / Sin for this position ──
  const int64_t pos = positions[token_idx];
  const scalar_t* cos_ptr = cos_sin_cache + pos * head_dim;
  const scalar_t* sin_ptr = cos_ptr + embed_dim;

  // ── Shared memory layout ──
  //    [0 .. head_dim-1]     : per-head data buffer (for RMSNorm)
  //    [head_dim .. head_dim + 8] : reduction scratch (≤ 8 warps)
  extern __shared__ float smem[];
  float* head_buf = smem;
  float* reduce_buf = smem + head_dim;
  __shared__ float s_inv_rms;

  // =================================================================
  //  Helper: RMSNorm – load one head from `src` into smem, normalise
  // =================================================================
  auto do_rmsnorm = [&](const scalar_t* src, const scalar_t* weight) {
    float variance = 0.f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
      float x = static_cast<float>(src[i]);
      head_buf[i] = x;
      variance += x * x;
    }
    float total = block_reduce_sum(variance, reduce_buf);
    if (tid == 0) s_inv_rms = rsqrtf(total / head_dim + epsilon);
    __syncthreads();
    for (int i = tid; i < head_dim; i += blockDim.x) {
      head_buf[i] *= s_inv_rms * static_cast<float>(weight[i]);
    }
    __syncthreads();
  };

  // =================================================================
  //  Helper: RoPE from smem → write to model-dtype output
  // =================================================================
  auto do_rope_write_model = [&](scalar_t* dst) {
    for (int i = tid; i < head_dim; i += blockDim.x) {
      float result;
      if constexpr (IS_NEOX) {
        if (i < embed_dim) {
          result = head_buf[i] * static_cast<float>(cos_ptr[i]) -
                   head_buf[i + embed_dim] * static_cast<float>(sin_ptr[i]);
        } else {
          int r = i - embed_dim;
          result = head_buf[i] * static_cast<float>(cos_ptr[r]) +
                   head_buf[r] * static_cast<float>(sin_ptr[r]);
        }
      } else {
        int pair = i / 2;
        float c = static_cast<float>(cos_ptr[pair]);
        float s = static_cast<float>(sin_ptr[pair]);
        float x = head_buf[2 * pair];
        float y = head_buf[2 * pair + 1];
        result = (i & 1) == 0 ? (x * c - y * s) : (y * c + x * s);
      }
      dst[i] = static_cast<scalar_t>(result);
    }
    __syncthreads();
  };

  // =================================================================
  //  Helper: RoPE from smem → write to model-dtype output and KV cache
  // =================================================================
  auto do_rope_write_model_and_cache =
      [&](scalar_t* model_dst, void* cache_ptr, float scale) {
    for (int i = tid; i < head_dim; i += blockDim.x) {
      float result;
      if constexpr (IS_NEOX) {
        if (i < embed_dim) {
          result = head_buf[i] * static_cast<float>(cos_ptr[i]) -
                   head_buf[i + embed_dim] * static_cast<float>(sin_ptr[i]);
        } else {
          int r = i - embed_dim;
          result = head_buf[i] * static_cast<float>(cos_ptr[r]) +
                   head_buf[r] * static_cast<float>(sin_ptr[r]);
        }
      } else {
        int pair = i / 2;
        float c = static_cast<float>(cos_ptr[pair]);
        float s = static_cast<float>(sin_ptr[pair]);
        float x = head_buf[2 * pair];
        float y = head_buf[2 * pair + 1];
        result = (i & 1) == 0 ? (x * c - y * s) : (y * c + x * s);
      }
      model_dst[i] = static_cast<scalar_t>(result);
      write_cache_elem<scalar_t, IS_FP8>(cache_ptr, i, result, scale);
    }
    __syncthreads();
  };

  // =================================================================
  //  Phase 1:  Interleaved V-store + K-compute  (per KV head)
  //
  //  For each KV head h:
  //    a) Load V[h] from global → FP8 convert → store v_cache
  //       (fire-and-forget, non-blocking)
  //    b) Load K[h] → smem → RMSNorm reduction → RoPE → store k_cache
  //       (V stores complete in background during K's compute)
  // =================================================================
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx >= 0) {
    const int64_t blk_idx = slot_idx / block_size;
    const int64_t blk_off = slot_idx % block_size;
    const int64_t cache_elem_size = IS_FP8 ? 1 : sizeof(scalar_t);
    const int64_t k_page_stride =
        (int64_t)num_heads_kv * head_dim * cache_elem_size;
    const int64_t v_page_stride =
        (int64_t)num_heads_kv * head_dim_v * cache_elem_size;
    const int64_t k_blk_stride = (int64_t)block_size * k_page_stride;
    const int64_t v_blk_stride = (int64_t)block_size * v_page_stride;
    const int64_t k_base_offset =
        blk_idx * k_blk_stride + blk_off * k_page_stride;
    const int64_t v_base_offset =
        blk_idx * v_blk_stride + blk_off * v_page_stride;

    char* k_base = reinterpret_cast<char*>(k_cache) + k_base_offset;
    char* v_base = reinterpret_cast<char*>(v_cache) + v_base_offset;

    for (int h = 0; h < num_heads_kv; ++h) {
      const int64_t k_head_cache_off =
          (int64_t)h * head_dim * cache_elem_size;
      const int64_t v_head_cache_off =
          (int64_t)h * head_dim_v * cache_elem_size;

      // ── (a) V store: load → convert → fire-and-forget store ──
      const scalar_t* v_src = v_in + h * head_dim_v;
      void* v_dst = v_base + v_head_cache_off;
      for (int i = tid; i < head_dim_v; i += blockDim.x) {
        float val = static_cast<float>(v_src[i]);
        write_cache_elem<scalar_t, IS_FP8>(v_dst, i, val, v_scale);
      }
      // No __syncthreads here – V stores are in flight, keep going.

      // ── (b) K: RMSNorm + RoPE → k_cache ──
      // While the memory subsystem drains V stores in the background,
      // threads load K into shared memory and start the RMSNorm reduction.
      do_rmsnorm(k_in + h * head_dim, k_weight);
      do_rope_write_model_and_cache(
          k_in + h * head_dim, k_base + k_head_cache_off, k_scale);
    }
  }

  // =================================================================
  //  Phase 2:  Q heads – RMSNorm + RoPE → query (model dtype)
  //
  //  By the time we reach here, all KV cache stores are complete or
  //  nearly so.  Query output is written back in place.
  // =================================================================
  for (int h = 0; h < num_heads_q; ++h) {
    do_rmsnorm(q_in + h * head_dim, q_weight);
    do_rope_write_model(q_in + h * head_dim);
  }
}

}  // namespace fused_qknrc_v2

// ── Torch wrapper ────────────────────────────────────────────────────

torch::Tensor fused_qk_norm_rope_cache_quant_v2(
    torch::Tensor query,          // [T, num_heads_q, head_dim]  – mutated
    torch::Tensor key,            // [T, num_heads_kv, head_dim] – mutated
    torch::Tensor value,          // [T, num_heads_kv, head_dim_v]
    torch::Tensor k_cache,        // [num_blocks, block_size, nkv, hd]
    torch::Tensor v_cache,        // [num_blocks, block_size, nkv, hd_v]
    torch::Tensor q_weight,       // [head_dim]
    torch::Tensor k_weight,       // [head_dim]
    torch::Tensor cos_sin_cache,  // [max_pos, rot_dim]
    torch::Tensor positions,      // [T]
    torch::Tensor slot_mapping,   // [T]
    double k_scale,
    double v_scale,
    double epsilon,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t block_size,
    bool is_neox,
    bool is_fp8) {

  const int num_tokens = positions.numel();
  if (num_tokens == 0) return torch::empty(0, query.options());

  const int head_dim_v = value.size(2);
  const int work_dim = std::max<int>(head_dim, head_dim_v);
  const int threads = ((std::max<int>(work_dim, 256) + 31) / 32) * 32;
  const int smem_bytes = (head_dim + 8) * sizeof(float);
  const at::cuda::OptionalCUDAGuard guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Macro for dispatching across (dtype × neox × fp8) combinations.
#define LAUNCH(scalar_t, NEOX, FP8)                                           \
  fused_qknrc_v2::fused_kernel<scalar_t, NEOX, FP8><<<num_tokens, threads,      \
                                                     smem_bytes, stream>>>(   \
      query.data_ptr<scalar_t>(),                                             \
      key.data_ptr<scalar_t>(),                                               \
      FP8 ? reinterpret_cast<void*>(k_cache.data_ptr<uint8_t>())             \
          : reinterpret_cast<void*>(k_cache.data_ptr<scalar_t>()),            \
      FP8 ? reinterpret_cast<void*>(v_cache.data_ptr<uint8_t>())             \
          : reinterpret_cast<void*>(v_cache.data_ptr<scalar_t>()),            \
      value.data_ptr<scalar_t>(), q_weight.data_ptr<scalar_t>(),             \
      k_weight.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),     \
      positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),        \
      static_cast<float>(k_scale), static_cast<float>(v_scale),              \
      static_cast<float>(epsilon), static_cast<int>(num_heads_q),            \
      static_cast<int>(num_heads_kv), static_cast<int>(head_dim),            \
      static_cast<int>(head_dim_v), static_cast<int>(block_size))

  VLLM_DISPATCH_HALF_TYPES(
      query.scalar_type(), "fused_qk_norm_rope_cache_quant_v2", [&] {
        if (is_neox) {
          if (is_fp8) {
            LAUNCH(scalar_t, true, true);
          } else {
            LAUNCH(scalar_t, true, false);
          }
        } else {
          if (is_fp8) {
            LAUNCH(scalar_t, false, true);
          } else {
            LAUNCH(scalar_t, false, false);
          }
        }
      });

#undef LAUNCH
  return torch::empty(0, query.options());
}

// PyBind registration is in torch_bindings.cpp (CMake build).
// For standalone JIT testing, uncomment the PYBIND11_MODULE block below.
//
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("fused_qk_norm_rope_cache_quant",
//         &fused_qk_norm_rope_cache_quant, ...);
// }
