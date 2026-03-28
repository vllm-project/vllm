// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "core/registration.h"
#include "dsv3_router_gemm_utils.h"

static constexpr int FP32_NUM_EXPERTS = 256;
static constexpr int FP32_HIDDEN_DIM  = 3072;
static constexpr int FP32_MAX_TOKENS  = 32;

template <int kNumTokens>
void invokeFp32RouterGemm(float* output, float const* mat_a,
                          float const* mat_b, cudaStream_t stream);

template <int kBegin, int kEnd>
struct Fp32LoopUnroller {
  static void unroll(int num_tokens, float* output, float const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeFp32RouterGemm<kBegin>(output, mat_a, mat_b, stream);
    } else {
      Fp32LoopUnroller<kBegin + 1, kEnd>::unroll(
          num_tokens, output, mat_a, mat_b, stream);
    }
  }
};

template <int kEnd>
struct Fp32LoopUnroller<kEnd, kEnd> {
  static void unroll(int num_tokens, float* output, float const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeFp32RouterGemm<kEnd>(output, mat_a, mat_b, stream);
    } else {
      throw std::invalid_argument(
          "fp32_router_gemm: num_tokens must be in [1, 32]");
    }
  }
};

void fp32_router_gemm(at::Tensor& output,       // [num_tokens, num_experts]
                      const at::Tensor& mat_a,  // [num_tokens, hidden_dim]
                      const at::Tensor& mat_b   // [num_experts, hidden_dim]
) {
  TORCH_CHECK(output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2);

  const int num_tokens  = mat_a.size(0);
  const int num_experts = mat_b.size(0);
  const int hidden_dim  = mat_a.size(1);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(1),
              "fp32_router_gemm: mat_a and mat_b must have the same hidden_dim");
  TORCH_CHECK(hidden_dim == FP32_HIDDEN_DIM,
              "fp32_router_gemm: expected hidden_dim=", FP32_HIDDEN_DIM,
              ", got ", hidden_dim);
  TORCH_CHECK(num_experts == FP32_NUM_EXPERTS,
              "fp32_router_gemm: expected num_experts=", FP32_NUM_EXPERTS,
              ", got ", num_experts);
  TORCH_CHECK(num_tokens >= 1 && num_tokens <= FP32_MAX_TOKENS,
              "fp32_router_gemm: num_tokens must be in [1, ", FP32_MAX_TOKENS,
              "], got ", num_tokens);
  TORCH_CHECK(mat_a.dtype() == at::kFloat, "fp32_router_gemm: mat_a must be float32");
  TORCH_CHECK(mat_b.dtype() == at::kFloat, "fp32_router_gemm: mat_b must be float32");
  TORCH_CHECK(output.dtype() == at::kFloat, "fp32_router_gemm: output must be float32");

  const int sm = getSMVersion();
  TORCH_CHECK(sm >= 90, "fp32_router_gemm: requires SM90+, got SM", sm);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  Fp32LoopUnroller<1, FP32_MAX_TOKENS>::unroll(
      num_tokens,
      reinterpret_cast<float*>(output.mutable_data_ptr()),
      reinterpret_cast<float const*>(mat_a.data_ptr()),
      reinterpret_cast<float const*>(mat_b.data_ptr()),
      stream);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("fp32_router_gemm", &fp32_router_gemm);
}
