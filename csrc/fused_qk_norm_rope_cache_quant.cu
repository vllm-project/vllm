// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused QK RMSNorm + RoPE + KV Cache Write + FP8 Per-Tensor Quantisation
//
// One CUDA thread-block processes one token.
//   Q heads:  RMSNorm → RoPE → write BF16/FP16 to q_out
//   K heads:  RMSNorm → RoPE → FP8 quantise → write to paged k_cache
//   V heads:  FP8 quantise → write to paged v_cache
//
// Build (standalone, for rapid testing):
//   torch.utils.cpp_extension.load(
//       name="fused_qknrc",
//       sources=["csrc/fused_qk_norm_rope_cache_quant.cu"],
//       extra_cuda_cflags=["-O3", "--use_fast_math"])

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP8 header – available since CUDA 11.8.
// HW-accelerated conversion requires SM >= 89 (Ada / Hopper).
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#define HAS_CUDA_FP8 1
#else
#define HAS_CUDA_FP8 0
#endif

namespace fused_qknrc {

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
  // Pack into unsigned byte via half roundtrip (lossy but deterministic).
  __half h = __float2half_rn(val);
  // Re-interpret the 16-bit pattern; truncate mantissa.
  unsigned short bits = *reinterpret_cast<unsigned short*>(&h);
  // Simplified E4M3: sign(1) | exp(4) | man(3)
  uint8_t sign = (bits >> 15) & 1;
  int exp_h = ((bits >> 10) & 0x1F) - 15;   // de-bias half exponent
  int man_h = (bits >> 7) & 0x7;             // top 3 mantissa bits
  int exp_fp8 = exp_h + 7;                   // re-bias for E4M3 (bias=7)
  if (exp_fp8 <= 0) {
    return (sign << 7);                       // ±0 / underflow
  }
  if (exp_fp8 >= 15) {
    return (sign << 7) | 0x7E;               // max normal (±448)
  }
  return (sign << 7) | ((exp_fp8 & 0xF) << 3) | (man_h & 0x7);
#endif
}

// ── Main fused kernel ────────────────────────────────────────────────
//
// Template params:
//   scalar_t  – BFloat16 or Half (model dtype)
//   IS_NEOX   – true = GPT-NeoX style RoPE, false = GPT-J style
//   IS_FP8    – true = write KV cache as FP8 E4M3, false = same as scalar_t

template <typename scalar_t, bool IS_NEOX, bool IS_FP8>
__global__ void fused_kernel(
    // ── outputs ──
    scalar_t* __restrict__ q_out,          // [T, num_heads_q * head_dim]
    void* __restrict__ k_cache,            // NHD [B, blk_sz, nkv, hd]
    void* __restrict__ v_cache,            // NHD [B, blk_sz, nkv, hd]
    // ── inputs ──
    const scalar_t* __restrict__ qkv,      // [T, (nq+2*nkv)*hd]
    const scalar_t* __restrict__ q_weight, // [head_dim]
    const scalar_t* __restrict__ k_weight, // [head_dim]
    const scalar_t* __restrict__ cos_sin_cache, // [max_pos, rot_dim]
    const int64_t* __restrict__ positions,      // [T]
    const int64_t* __restrict__ slot_mapping,   // [T]
    // ── scalars ──
    const float k_scale, const float v_scale, const float epsilon,
    const int num_heads_q, const int num_heads_kv, const int head_dim,
    const int block_size) {

  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const int q_size = num_heads_q * head_dim;
  const int kv_size = num_heads_kv * head_dim;
  const int embed_dim = head_dim / 2;  // rotary half

  // ── Token pointers ──
  const scalar_t* q_in = qkv + (int64_t)token_idx * (q_size + 2 * kv_size);
  const scalar_t* k_in = q_in + q_size;
  const scalar_t* v_in = k_in + kv_size;

  // ── Cos / Sin for this position ──
  const int64_t pos = positions[token_idx];
  const scalar_t* cos_ptr = cos_sin_cache + pos * head_dim;
  const scalar_t* sin_ptr = cos_ptr + embed_dim;

  // ── Shared memory layout ──
  //    [0 .. head_dim-1]     : per-head data buffer
  //    [head_dim .. head_dim + 8] : reduction scratch (≤ 8 warps)
  extern __shared__ float smem[];
  float* head_buf = smem;
  float* reduce_buf = smem + head_dim;
  __shared__ float s_inv_rms;

  // =================================================================
  //  Helper lambdas (capture everything by reference)
  // =================================================================

  // RMSNorm: load one head → normalise in smem
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

  // RoPE: apply in-place in smem.  Reads from smem, writes to `dst`.
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

  // RoPE + write to KV cache (FP8 or model dtype).
  auto do_rope_write_cache = [&](void* cache_ptr, float scale) {
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
      if constexpr (IS_FP8) {
        reinterpret_cast<uint8_t*>(cache_ptr)[i] =
            float_to_fp8_e4m3(result / scale);
      } else {
        reinterpret_cast<scalar_t*>(cache_ptr)[i] =
            static_cast<scalar_t>(result);
      }
    }
    __syncthreads();
  };

  // =================================================================
  //  1.  Q heads – RMSNorm + RoPE → q_out (model dtype)
  // =================================================================
  scalar_t* q_dst = q_out + (int64_t)token_idx * q_size;
  for (int h = 0; h < num_heads_q; ++h) {
    do_rmsnorm(q_in + h * head_dim, q_weight);
    do_rope_write_model(q_dst + h * head_dim);
  }

  // =================================================================
  //  2.  K heads – RMSNorm + RoPE → k_cache (FP8 or model dtype)
  // =================================================================
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx >= 0) {
    const int64_t blk_idx = slot_idx / block_size;
    const int64_t blk_off = slot_idx % block_size;
    // NHD layout: [num_blocks, block_size, num_kv_heads, head_dim]
    const int64_t cache_elem_size = IS_FP8 ? 1 : sizeof(scalar_t);
    const int64_t page_stride = (int64_t)num_heads_kv * head_dim * cache_elem_size;
    const int64_t blk_stride = (int64_t)block_size * page_stride;
    char* k_base =
        reinterpret_cast<char*>(k_cache) + blk_idx * blk_stride + blk_off * page_stride;

    for (int h = 0; h < num_heads_kv; ++h) {
      do_rmsnorm(k_in + h * head_dim, k_weight);
      do_rope_write_cache(k_base + (int64_t)h * head_dim * cache_elem_size,
                          k_scale);
    }

    // =================================================================
    //  3.  V heads – just copy (optionally FP8-quantise) to v_cache
    // =================================================================
    char* v_base =
        reinterpret_cast<char*>(v_cache) + blk_idx * blk_stride + blk_off * page_stride;

    for (int h = 0; h < num_heads_kv; ++h) {
      const scalar_t* v_src = v_in + h * head_dim;
      for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(v_src[i]);
        if constexpr (IS_FP8) {
          reinterpret_cast<uint8_t*>(
              v_base + (int64_t)h * head_dim * cache_elem_size)[i] =
              float_to_fp8_e4m3(val / v_scale);
        } else {
          reinterpret_cast<scalar_t*>(
              v_base + (int64_t)h * head_dim * cache_elem_size)[i] =
              static_cast<scalar_t>(val);
        }
      }
    }
  }
}

}  // namespace fused_qknrc

// ── Torch wrapper ────────────────────────────────────────────────────

void fused_qk_norm_rope_cache_quant(
    torch::Tensor q_out,           // [T, num_heads_q * head_dim]  – output
    torch::Tensor k_cache,         // [num_blocks, block_size, nkv, hd]
    torch::Tensor v_cache,         // same shape
    torch::Tensor qkv,            // [T, (nq + 2*nkv) * hd]
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
  if (num_tokens == 0) return;

  const int threads = std::min<int>(head_dim, 256);
  const int smem_bytes = (head_dim + 8) * sizeof(float);
  const at::cuda::OptionalCUDAGuard guard(qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Macro for dispatching across (dtype × neox × fp8) combinations.
#define LAUNCH(scalar_t, NEOX, FP8)                                           \
  fused_qknrc::fused_kernel<scalar_t, NEOX, FP8><<<num_tokens, threads,      \
                                                     smem_bytes, stream>>>(   \
      q_out.data_ptr<scalar_t>(),                                             \
      FP8 ? reinterpret_cast<void*>(k_cache.data_ptr<uint8_t>())             \
          : reinterpret_cast<void*>(k_cache.data_ptr<scalar_t>()),            \
      FP8 ? reinterpret_cast<void*>(v_cache.data_ptr<uint8_t>())             \
          : reinterpret_cast<void*>(v_cache.data_ptr<scalar_t>()),            \
      qkv.data_ptr<scalar_t>(), q_weight.data_ptr<scalar_t>(),               \
      k_weight.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),     \
      positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),        \
      static_cast<float>(k_scale), static_cast<float>(v_scale),              \
      static_cast<float>(epsilon), static_cast<int>(num_heads_q),            \
      static_cast<int>(num_heads_kv), static_cast<int>(head_dim),            \
      static_cast<int>(block_size))

  AT_DISPATCH_SWITCH(
      qkv.scalar_type(), "fused_qk_norm_rope_cache_quant",
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        if (is_neox) {
          if (is_fp8) { LAUNCH(scalar_t, true, true); }
          else        { LAUNCH(scalar_t, true, false); }
        } else {
          if (is_fp8) { LAUNCH(scalar_t, false, true); }
          else        { LAUNCH(scalar_t, false, false); }
        }
      })
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        if (is_neox) {
          if (is_fp8) { LAUNCH(scalar_t, true, true); }
          else        { LAUNCH(scalar_t, true, false); }
        } else {
          if (is_fp8) { LAUNCH(scalar_t, false, true); }
          else        { LAUNCH(scalar_t, false, false); }
        }
      }));

#undef LAUNCH
}

// ── PyBind ──

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_qk_norm_rope_cache_quant",
        &fused_qk_norm_rope_cache_quant,
        "Fused QK RMSNorm + RoPE + KV cache write + FP8 quant (CUDA)",
        py::arg("q_out"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("qkv"),
        py::arg("q_weight"),
        py::arg("k_weight"),
        py::arg("cos_sin_cache"),
        py::arg("positions"),
        py::arg("slot_mapping"),
        py::arg("k_scale"),
        py::arg("v_scale"),
        py::arg("epsilon"),
        py::arg("num_heads_q"),
        py::arg("num_heads_kv"),
        py::arg("head_dim"),
        py::arg("block_size"),
        py::arg("is_neox"),
        py::arg("is_fp8"));
}
