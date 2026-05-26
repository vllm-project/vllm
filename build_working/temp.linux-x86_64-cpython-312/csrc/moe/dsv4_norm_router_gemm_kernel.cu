/*
 * Fused RMSNorm + router GEMV for DeepSeek V4 (logits are fp32; bf16
 * output is unsupported because DSV4 hard-codes fp32 logits).  See
 * dsv4_norm_router_gemm.h for the math.
 *
 * The GEMV body mirrors csrc/moe/dsv3_router_gemm_float_out.cu (warp
 * butterfly reduction + smem cross-warp reduction, fp32 accumulation,
 * 128-thread block, PDL on SM90+).  RMSNorm is folded into the same
 * pass via the identity
 *   logits[m,n] = rsqrt[m] * sum_k(x[m,k] * nw[k] * gw[n,k])
 * so x is read exactly once per block during the GEMV phase.  Blocks
 * 0..kNumTokens-1 each materialize one row of normed_x for downstream
 * experts / shared_experts to consume.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsv4_norm_router_gemm.h"

namespace {

// Convert 8 bf16 values packed in uint4 into 8 floats. Mirrors the helper
// in dsv3_router_gemm_float_out.cu (kept local so the dsv3 file stays
// untouched).
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec,
                                                     float* dst) {
  __nv_bfloat16* bf16_ptr =
      reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void norm_router_gemm_kernel(
    float* __restrict__ logits, __nv_bfloat16* __restrict__ normed_x,
    T const* __restrict__ x, T const* __restrict__ norm_weight,
    T const* __restrict__ gate_weight, float eps) {
  static_assert(kBlockSize == 128, "kernel assumes blockDim.x == 128");
  static_assert(kHiddenDim % (VPT * kBlockSize) == 0,
                "kHiddenDim must be a multiple of VPT * kBlockSize");

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_iter = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_iter;

  T const* gw_col = gate_weight + n_idx * kHiddenDim;

  // Per-thread accumulators — fp32 throughout, matching dsv3 / layernorm.
  float partial[kNumTokens] = {};
  float ss[kNumTokens] = {};

  // Cross-warp reduction scratch.
  __shared__ float sm_partial[kNumTokens][kNumWarps];
  __shared__ float sm_ss[kNumTokens][kNumWarps];
  __shared__ float s_rsqrt[kNumTokens];

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_iter + tid * VPT;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // ---- Phase 1: single pass over x, accumulate partial GEMV and ss. ----
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    uint4 nw_vec = *reinterpret_cast<uint4 const*>(norm_weight + k_base);
    float nw_f[VPT];
    bf16_uint4_to_float8<VPT>(nw_vec, nw_f);

    uint4 b_vec = *reinterpret_cast<uint4 const*>(gw_col + k_base);
    float b_f[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_f);

#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      uint4 a_vec =
          *reinterpret_cast<uint4 const*>(x + m * kHiddenDim + k_base);
      float a_f[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_f);

#pragma unroll
      for (int k = 0; k < VPT; k++) {
        float a = a_f[k];
        ss[m] += a * a;
        partial[m] += a * nw_f[k] * b_f[k];
      }
    }
  }

  // ---- Phase 2: warp butterfly reduction for both ss[] and partial[]. ----
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float p = partial[m];
    float s = ss[m];

    p += __shfl_xor_sync(0xffffffff, p, 16);
    s += __shfl_xor_sync(0xffffffff, s, 16);
    p += __shfl_xor_sync(0xffffffff, p, 8);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    p += __shfl_xor_sync(0xffffffff, p, 4);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    p += __shfl_xor_sync(0xffffffff, p, 2);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    p += __shfl_xor_sync(0xffffffff, p, 1);
    s += __shfl_xor_sync(0xffffffff, s, 1);

    if (laneId == 0) {
      sm_partial[m][warpId] = p;
      sm_ss[m][warpId] = s;
    }
  }

  __syncthreads();

  // ---- Phase 3: tid 0 finalises the reduction, writes logits, stashes
  //               rsqrt[m] in smem for phase 4. ----
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float p_sum = 0.0f;
      float s_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        p_sum += sm_partial[m][w];
        s_sum += sm_ss[m][w];
      }
      // Order matches layernorm_kernels.cu: rsqrtf(variance / H + eps).
      // Use division (not multiply-by-reciprocal) to avoid an extra ULP
      // mismatch with the reference RMSNorm.
      float rs = rsqrtf(s_sum / static_cast<float>(kHiddenDim) + eps);
      s_rsqrt[m] = rs;
      logits[m * kNumExperts + n_idx] = p_sum * rs;
    }
  }

  __syncthreads();

  // ---- Phase 4: spread normed_x writes across blocks 0..kNumTokens-1.
  //              Each writer block handles exactly one token row,
  //              avoiding the long tail of block 0 doing all M rows.
  //              Every block has every token's rsqrt[] in s_rsqrt
  //              already (computed independently in phase 3), so no
  //              cross-block synchronization is required. ----
  if (n_idx < kNumTokens) {
    int const m_writer = n_idx;
    float const rs = s_rsqrt[m_writer];
    __nv_bfloat16 const* x_row = x + m_writer * kHiddenDim;
    __nv_bfloat16* normed_row = normed_x + m_writer * kHiddenDim;

#pragma unroll
    for (int ki = 0; ki < k_iterations; ki++) {
      int const k_base = k_bases[ki];

      uint4 nw_vec = *reinterpret_cast<uint4 const*>(norm_weight + k_base);
      float nw_f[VPT];
      bf16_uint4_to_float8<VPT>(nw_vec, nw_f);

      uint4 a_vec = *reinterpret_cast<uint4 const*>(x_row + k_base);
      float a_f[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_f);

      uint4 normed_vec;
      __nv_bfloat16* np = reinterpret_cast<__nv_bfloat16*>(&normed_vec);
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        np[k] = __float2bfloat16(a_f[k] * rs * nw_f[k]);
      }
      *reinterpret_cast<uint4*>(normed_row + k_base) = normed_vec;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeNormRouterGemm(float* logits, __nv_bfloat16* normed_x, T const* x,
                          T const* norm_weight, T const* gate_weight, float eps,
                          cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;

  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config,
                     norm_router_gemm_kernel<T, kBlockSize, VPT, kNumTokens,
                                             kNumExperts, kHiddenDim>,
                     logits, normed_x, x, norm_weight, gate_weight, eps);
}

// Template instantiations — DSV4-Pro is the only supported configuration:
// num_experts=384, hidden_dim=7168.  Other shapes (e.g. DSV4-Flash with
// hidden_dim=4096) fall back to the unfused path on the Python side.
#define INSTANTIATE(M)                                                    \
  template void invokeNormRouterGemm<__nv_bfloat16, M, 384, 7168>(        \
      float*, __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, \
      __nv_bfloat16 const*, float, cudaStream_t);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)
INSTANTIATE(5)
INSTANTIATE(6)
INSTANTIATE(7)
INSTANTIATE(8)
INSTANTIATE(9)
INSTANTIATE(10)
INSTANTIATE(11)
INSTANTIATE(12)
INSTANTIATE(13)
INSTANTIATE(14)
INSTANTIATE(15)
INSTANTIATE(16)

#undef INSTANTIATE
