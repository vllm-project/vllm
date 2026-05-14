/*
 * Fused RMSNorm + router GEMV for DeepSeek V4.
 *
 * Computes in a single kernel:
 *   normed_x[m,k]   = x[m,k] * rsqrt(mean(x[m]^2) + eps) * norm_weight[k]
 *   router_logits[m,n] = sum_k(normed_x[m,k] * gate_weight[n,k])
 *
 * The GEMV body mirrors the algorithm in csrc/moe/dsv3_router_gemm_*.cu
 * (warp butterfly + smem cross-warp reduction, fp32 accumulation, PDL on
 * SM90+).  Blocks 0..kNumTokens-1 each materialize one token's normed_x
 * row to global memory using the algebraic identity
 *      logits[m,n] = rsqrt[m] * sum_k(x[m,k] * nw[k] * gw[n,k])
 * which lets every block produce its column of logits before normed_x
 * exists in gmem.
 *
 * Logits output is fp32 only — DeepSeek V4 router gate is hard-coded to
 * fp32 (vllm/model_executor/models/deepseek_v4.py:749).
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsv3_router_gemm_utils.h"

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeNormRouterGemm(float* logits, __nv_bfloat16* normed_x, T const* x,
                          T const* norm_weight, T const* gate_weight, float eps,
                          cudaStream_t stream);
