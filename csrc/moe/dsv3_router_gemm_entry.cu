/*
 * Adapted from SGLang's sgl-kernel implementation, which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsv3_router_gemm_utils.h"

static constexpr int DEFAULT_NUM_EXPERTS = 256;
static constexpr int KIMI_K2_NUM_EXPERTS = 384;
static constexpr int DEFAULT_HIDDEN_DIM = 7168;

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,
                                 cudaStream_t stream);

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a,
                                T const* mat_b, cudaStream_t stream);

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller {
  static void unroll_float_output(int num_tokens, float* output,
                                  __nv_bfloat16 const* input,
                                  __nv_bfloat16 const* weights,
                                  cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeRouterGemmFloatOutput<__nv_bfloat16, kBegin, kNumExperts,
                                  kHiddenDim>(output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts,
                   kHiddenDim>::unroll_float_output(num_tokens, output, input,
                                                    weights, stream);
    }
  }

  static void unroll_bf16_output(int num_tokens, __nv_bfloat16* output,
                                 __nv_bfloat16 const* input,
                                 __nv_bfloat16 const* weights,
                                 cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeRouterGemmBf16Output<__nv_bfloat16, kBegin, kNumExperts,
                                 kHiddenDim>(output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts,
                   kHiddenDim>::unroll_bf16_output(num_tokens, output, input,
                                                   weights, stream);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim> {
  static void unroll_float_output(int num_tokens, float* output,
                                  __nv_bfloat16 const* input,
                                  __nv_bfloat16 const* weights,
                                  cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeRouterGemmFloatOutput<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(
          output, input, weights, stream);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }

  static void unroll_bf16_output(int num_tokens, __nv_bfloat16* output,
                                 __nv_bfloat16 const* input,
                                 __nv_bfloat16 const* weights,
                                 cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeRouterGemmBf16Output<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(
          output, input, weights, stream);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }
};

void dsv3_router_gemm(at::Tensor& output,       // [num_tokens, num_experts]
                      const at::Tensor& mat_a,  // [num_tokens, hidden_dim]
                      const at::Tensor& mat_b   // [num_experts, hidden_dim]
) {
  TORCH_CHECK(output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2);

  const int num_tokens = mat_a.size(0);
  const int num_experts = mat_b.size(0);
  const int hidden_dim = mat_a.size(1);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(1),
              "mat_a and mat_b must have the same hidden_dim");
  TORCH_CHECK(hidden_dim == DEFAULT_HIDDEN_DIM,
              "Expected hidden_dim=", DEFAULT_HIDDEN_DIM,
              ", but got hidden_dim=", hidden_dim);
  TORCH_CHECK(
      num_experts == DEFAULT_NUM_EXPERTS || num_experts == KIMI_K2_NUM_EXPERTS,
      "Expected num_experts=", DEFAULT_NUM_EXPERTS,
      " or num_experts=", KIMI_K2_NUM_EXPERTS,
      ", but got num_experts=", num_experts);
  TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
              "currently num_tokens must be less than or equal to 16 for "
              "router_gemm");
  TORCH_CHECK(mat_a.dtype() == at::kBFloat16, "mat_a must be bf16");
  TORCH_CHECK(mat_b.dtype() == at::kBFloat16, "mat_b must be bf16");
  TORCH_CHECK(output.dtype() == at::kFloat || output.dtype() == at::kBFloat16,
              "output must be float32 or bf16");

  auto const sm = getSMVersion();
  TORCH_CHECK(sm >= 90, "required CUDA ARCH >= SM_90");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (output.dtype() == at::kFloat) {
    if (num_experts == DEFAULT_NUM_EXPERTS) {
      LoopUnroller<1, 16, DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM>::
          unroll_float_output(
              num_tokens, reinterpret_cast<float*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    } else if (num_experts == KIMI_K2_NUM_EXPERTS) {
      LoopUnroller<1, 16, KIMI_K2_NUM_EXPERTS, DEFAULT_HIDDEN_DIM>::
          unroll_float_output(
              num_tokens, reinterpret_cast<float*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
  } else if (output.dtype() == at::kBFloat16) {
    if (num_experts == DEFAULT_NUM_EXPERTS) {
      LoopUnroller<1, 16, DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM>::
          unroll_bf16_output(
              num_tokens,
              reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    } else if (num_experts == KIMI_K2_NUM_EXPERTS) {
      LoopUnroller<1, 16, KIMI_K2_NUM_EXPERTS, DEFAULT_HIDDEN_DIM>::
          unroll_bf16_output(
              num_tokens,
              reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
  }
}
