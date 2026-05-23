/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused per-head QK RMSNorm + multimodal RoPE (mRoPE) kernel.
 *
 * mRoPE uses three separate position streams (time/height/width) stored as
 *   cos: [3, num_tokens, rotary_dim/2]
 *   sin: [3, num_tokens, rotary_dim/2]
 * instead of a flat cos_sin_cache indexed by scalar position IDs.
 *
 * Section boundaries (mrope_section_t, mrope_section_h) determine which
 * stream each half-dimension index maps to:
 *   [0, t_end)           → time stream
 *   [t_end, h_end)       → height stream
 *   [h_end, rotary_dim/2) → width stream
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define MROPE_CHECK_TYPE(x, st)                                        \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define MROPE_CHECK_TH_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define MROPE_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define MROPE_CHECK_INPUT(x) \
  MROPE_CHECK_TH_CUDA(x);    \
  MROPE_CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define MROPE_FINAL_MASK 0xffffffffffffffffULL
#else
  #define MROPE_FINAL_MASK 0xffffffff
#endif

namespace vllm::fused_qknorm_mrope {

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(MROPE_FINAL_MASK, val, mask, 32);
  return val;
}

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

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// One warp processes one (token, Q/K-head) pair.
// cos_void / sin_void each have shape [3, num_tokens, half_rd]
//   stream 0 (time):   offset [0, num_tokens * half_rd)
//   stream 1 (height): offset [num_tokens * half_rd, 2 * num_tokens * half_rd)
//   stream 2 (width):  offset [2 * num_tokens * half_rd, 3 * num_tokens *
//   half_rd)
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void fusedQKNormMRopeKernel(
    void* qkv_void, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, float const eps, void const* q_weight_void,
    void const* k_weight_void,
    void const* cos_void,  // [3, num_tokens, rotary_dim/2]
    void const* sin_void,  // [3, num_tokens, rotary_dim/2]
    int const num_tokens, int const rotary_dim, int const mrope_section_t,
    int const mrope_section_h) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input QKV dtype not supported for this CUDA architecture.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache dtype not supported for this CUDA architecture.");
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_base = reinterpret_cast<T_cache const*>(cos_void);
    T_cache const* sin_base = reinterpret_cast<T_cache const*>(sin_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
    int const total_qk_heads = num_heads_q + num_heads_k;
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    if (tokenIdx >= num_tokens) return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
    constexpr int numElemsPerThread = head_dim / 32;
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0,
                  "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename packed_as<uint, vecSize>::type;

    int offsetWarp;
    if (isQ) {
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                   headIdx * head_dim;
    }
    int const offsetThread = offsetWarp + laneId * numElemsPerThread;

    float sumOfSquares = 0.0f;
    float elements[numElemsPerThread];

    // Load elements and accumulate sum of squares for RMSNorm.
    {
      vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        float2 vals = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x + vals.y * vals.y;
        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    sumOfSquares = warpReduceSum(sumOfSquares);
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int const dim = laneId * numElemsPerThread + i;
      float weight = isQ ? Converter::convert(q_weight[dim])
                         : Converter::convert(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

    // mRoPE: select cos/sin stream per half-dim based on section boundaries.
    int const half_rd = rotary_dim / 2;
    int const t_end = mrope_section_t;
    int const h_end = mrope_section_t + mrope_section_h;

    T_cache const* t_cos = cos_base + tokenIdx * half_rd;
    T_cache const* h_cos = cos_base + (num_tokens + tokenIdx) * half_rd;
    T_cache const* w_cos = cos_base + (2 * num_tokens + tokenIdx) * half_rd;
    T_cache const* t_sin = sin_base + tokenIdx * half_rd;
    T_cache const* h_sin = sin_base + (num_tokens + tokenIdx) * half_rd;
    T_cache const* w_sin = sin_base + (2 * num_tokens + tokenIdx) * half_rd;

    float elements2[numElemsPerThread];
    int const rotary_lanes = rotary_dim / numElemsPerThread;

    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // Interleaved style: consecutive pairs (x0, x1) rotate together.
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          int const dim_idx = laneId * numElemsPerThread + idx0;
          float const val0 = elements[idx0];
          float const val1 = elements[idx1];
          int const half_dim = dim_idx / 2;

          T_cache cos_raw = (half_dim < t_end)   ? VLLM_LDG(t_cos + half_dim)
                            : (half_dim < h_end) ? VLLM_LDG(h_cos + half_dim)
                                                 : VLLM_LDG(w_cos + half_dim);
          T_cache sin_raw = (half_dim < t_end)   ? VLLM_LDG(t_sin + half_dim)
                            : (half_dim < h_end) ? VLLM_LDG(h_sin + half_dim)
                                                 : VLLM_LDG(w_sin + half_dim);
          float const cos_val = CacheConverter::convert(cos_raw);
          float const sin_val = CacheConverter::convert(sin_raw);

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // NeoX style: left half [0, half_rd) and right half [half_rd, rd).
        __syncwarp();
        int const pairOffset = half_rd / numElemsPerThread;
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] =
              __shfl_xor_sync(MROPE_FINAL_MASK, elements[i], pairOffset);
          if (laneId < pairOffset) elements2[i] = -elements2[i];

          int dim_idx = laneId * numElemsPerThread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int const half_dim = dim_idx / 2;

          T_cache cos_raw = (half_dim < t_end)   ? VLLM_LDG(t_cos + half_dim)
                            : (half_dim < h_end) ? VLLM_LDG(h_cos + half_dim)
                                                 : VLLM_LDG(w_cos + half_dim);
          T_cache sin_raw = (half_dim < t_end)   ? VLLM_LDG(t_sin + half_dim)
                            : (half_dim < h_end) ? VLLM_LDG(h_sin + half_dim)
                                                 : VLLM_LDG(w_sin + half_dim);
          float const cos_val = CacheConverter::convert(cos_raw);
          float const sin_val = CacheConverter::convert(sin_raw);

          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        __syncwarp();
      }
    }

    // Store.
    {
      vec_T vec;
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

#define DISPATCH_MROPE_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                            \
    const bool INTERLEAVE = true;                              \
    __VA_ARGS__                                                \
  } else {                                                     \
    const bool INTERLEAVE = false;                             \
    __VA_ARGS__                                                \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormMRope(void* qkv, int const num_tokens,
                            int const num_heads_q, int const num_heads_k,
                            int const num_heads_v, int const head_dim,
                            int const rotary_dim, float const eps,
                            void const* q_weight, void const* k_weight,
                            void const* cos, void const* sin,
                            bool const interleave, int const mrope_section_t,
                            int const mrope_section_h, cudaStream_t stream) {
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;
  int const gridSize = divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  switch (head_dim) {
    case 64:
      DISPATCH_MROPE_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos, sin, num_tokens, rotary_dim, mrope_section_t,
                mrope_section_h);
      });
      break;
    case 128:
      DISPATCH_MROPE_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos, sin, num_tokens, rotary_dim, mrope_section_t,
                mrope_section_h);
      });
      break;
    case 256:
      DISPATCH_MROPE_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos, sin, num_tokens, rotary_dim, mrope_section_t,
                mrope_section_h);
      });
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fused_qk_norm_mrope: ",
                  head_dim);
  }
}

}  // namespace vllm::fused_qknorm_mrope

// Host entry point registered as torch op.
void fused_qk_norm_mrope(
    torch::Tensor& qkv,       // [num_tokens, (nq+nk+nv)*head_dim], in-place
    int64_t num_heads_q,      // Q heads (per TP rank)
    int64_t num_heads_k,      // K heads (per TP rank)
    int64_t num_heads_v,      // V heads (per TP rank)
    int64_t head_dim,         // dimension per head
    double eps,               // RMSNorm epsilon
    torch::Tensor& q_weight,  // [head_dim]
    torch::Tensor& k_weight,  // [head_dim]
    torch::Tensor& cos,       // [3, num_tokens, rotary_dim/2]
    torch::Tensor& sin,       // [3, num_tokens, rotary_dim/2]
    bool is_neox,             // true → NeoX (non-interleaved) style
    int64_t mrope_section_t,  // half-dims mapped to time stream
    int64_t mrope_section_h   // half-dims mapped to height stream
) {
  MROPE_CHECK_INPUT(qkv);
  MROPE_CHECK_INPUT(q_weight);
  MROPE_CHECK_INPUT(k_weight);
  MROPE_CHECK_INPUT(cos);
  MROPE_CHECK_INPUT(sin);

  TORCH_CHECK(qkv.dim() == 2,
              "qkv must be 2D: [num_tokens, total_heads * head_dim]");
  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D: [head_dim]");
  TORCH_CHECK(cos.dim() == 3, "cos must be 3D: [3, num_tokens, rotary_dim/2]");
  TORCH_CHECK(sin.dim() == 3, "sin must be 3D: [3, num_tokens, rotary_dim/2]");
  TORCH_CHECK(cos.size(0) == 3 && sin.size(0) == 3,
              "cos/sin first dimension must be 3 (t/h/w streams)");

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(cos.size(1) == num_tokens, "cos.size(1) must match num_tokens");
  TORCH_CHECK(sin.size(1) == num_tokens, "sin.size(1) must match num_tokens");

  int64_t half_rd = cos.size(2);
  int64_t rotary_dim = half_rd * 2;
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  // The kernel processes all rotary lanes inside a single warp; requiring
  // rotary_dim == head_dim guarantees all 32 threads participate in the NeoX
  // __shfl_xor_sync / __syncwarp calls and avoids warp divergence issues.
  TORCH_CHECK(rotary_dim == head_dim,
              "fused_qk_norm_mrope requires rotary_dim == head_dim; "
              "partial rotary is not supported by this kernel");
  TORCH_CHECK(mrope_section_t + mrope_section_h <= half_rd,
              "mrope_section_t + mrope_section_h must be <= rotary_dim/2");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim,
              "qkv.size(1) must equal (nq+nk+nv)*head_dim");
  TORCH_CHECK(q_weight.size(0) == head_dim,
              "q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim,
              "k_weight size must match head_dim");
  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type(),
              "qkv, q_weight and k_weight must share dtype");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
              "cos and sin must share dtype");

  auto device_id = qkv.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // BFloat16 requires SM80+. Check here to give a clear error rather than
  // silently returning early inside the kernel (same constraint as
  // fused_qk_norm_rope).
  if (qkv.scalar_type() == at::kBFloat16 ||
      cos.scalar_type() == at::kBFloat16) {
    auto* dev_prop = at::cuda::getDeviceProperties(device_id);
    TORCH_CHECK(dev_prop->major >= 8,
                "fused_qk_norm_mrope with BFloat16 requires SM80 (Ampere) "
                "or newer; device SM is ",
                dev_prop->major, ".", dev_prop->minor);
  }

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "fused_qk_norm_mrope", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(cos.scalar_type(), "fused_qk_norm_mrope", [&] {
      using cache_scalar_t = scalar_t;
      vllm::fused_qknorm_mrope::launchFusedQKNormMRope<qkv_scalar_t,
                                                       cache_scalar_t>(
          qkv.data_ptr(), static_cast<int>(num_tokens),
          static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
          static_cast<int>(num_heads_v), static_cast<int>(head_dim),
          static_cast<int>(rotary_dim), static_cast<float>(eps),
          q_weight.data_ptr(), k_weight.data_ptr(), cos.data_ptr(),
          sin.data_ptr(), !is_neox, static_cast<int>(mrope_section_t),
          static_cast<int>(mrope_section_h), stream);
    });
  });
}
