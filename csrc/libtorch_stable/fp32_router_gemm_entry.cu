// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {

inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}

}  // namespace

static constexpr int FP32_MAX_TOKENS = 32;

// Supported (hidden_dim, num_experts) pairs (must match the instantiations in
// fp32_router_gemm.cu): (3072, 256) for MiniMax-M2/M2.5, (6144, 128) for M3,
// (6144, 256) for GLM-5.
static inline bool fp32_router_gemm_supported(int hidden_dim, int num_experts) {
  return (hidden_dim == 3072 && num_experts == 256) ||
         (hidden_dim == 6144 && num_experts == 128) ||
         (hidden_dim == 6144 && num_experts == 256);
}

// Forward declarations — 4 template params must match fp32_router_gemm.cu
template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream);

// LoopUnroller templated on InputT, kNumExperts and kHiddenDim
template <typename InputT, int kNumExperts, int kHiddenDim, int kBegin,
          int kEnd>
struct Fp32LoopUnroller {
  static void unroll(int num_tokens, float* output, InputT const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeFp32RouterGemm<InputT, kBegin, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    } else {
      Fp32LoopUnroller<InputT, kNumExperts, kHiddenDim, kBegin + 1,
                       kEnd>::unroll(num_tokens, output, mat_a, mat_b, stream);
    }
  }
};

template <typename InputT, int kNumExperts, int kHiddenDim, int kEnd>
struct Fp32LoopUnroller<InputT, kNumExperts, kHiddenDim, kEnd, kEnd> {
  static void unroll(int num_tokens, float* output, InputT const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeFp32RouterGemm<InputT, kEnd, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    } else {
      throw std::invalid_argument(
          "fp32_router_gemm: num_tokens must be in [1, 32]");
    }
  }
};

// Dispatch over the supported (num_experts, hidden_dim) pairs.
template <typename InputT>
void dispatchFp32RouterGemm(int num_experts, int hidden_dim, int num_tokens,
                            float* output, InputT const* mat_a,
                            float const* mat_b, cudaStream_t stream) {
  if (num_experts == 256 && hidden_dim == 3072) {
    Fp32LoopUnroller<InputT, 256, 3072, 1, FP32_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, stream);
  } else if (num_experts == 128 && hidden_dim == 6144) {
    Fp32LoopUnroller<InputT, 128, 6144, 1, FP32_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, stream);
  } else if (num_experts == 256 && hidden_dim == 6144) {
    Fp32LoopUnroller<InputT, 256, 6144, 1, FP32_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, stream);
  } else {
    throw std::invalid_argument(
        "fp32_router_gemm: unsupported (hidden_dim, num_experts) pair");
  }
}

void fp32_router_gemm(
    torch::stable::Tensor& output,       // [num_tokens, num_experts]
    torch::stable::Tensor const& mat_a,  // [num_tokens, hidden_dim]
    torch::stable::Tensor const& mat_b   // [num_experts, hidden_dim]
) {
  STD_TORCH_CHECK(output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2);
  STD_TORCH_CHECK(output.is_cuda() && mat_a.is_cuda() && mat_b.is_cuda(),
                  "fp32_router_gemm: all tensors must be CUDA tensors");
  STD_TORCH_CHECK(output.get_device_index() == mat_a.get_device_index() &&
                      output.get_device_index() == mat_b.get_device_index(),
                  "fp32_router_gemm: all tensors must be on the same device");
  STD_TORCH_CHECK(
      output.is_contiguous() && mat_a.is_contiguous() && mat_b.is_contiguous(),
      "fp32_router_gemm: all tensors must be contiguous");

  const int num_tokens = mat_a.size(0);
  const int num_experts = mat_b.size(0);
  const int hidden_dim = mat_a.size(1);

  STD_TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == num_experts,
                  "fp32_router_gemm: output must have shape [num_tokens, "
                  "num_experts]");
  STD_TORCH_CHECK(
      mat_a.size(1) == mat_b.size(1),
      "fp32_router_gemm: mat_a and mat_b must have the same hidden_dim");
  STD_TORCH_CHECK(
      fp32_router_gemm_supported(hidden_dim, num_experts),
      "fp32_router_gemm: supported (hidden_dim, num_experts) pairs are "
      "(3072, 256), (6144, 128) and (6144, 256)");
  STD_TORCH_CHECK(num_tokens <= FP32_MAX_TOKENS,
                  "fp32_router_gemm: num_tokens must be in [0, 32]");
  STD_TORCH_CHECK(
      mat_a.scalar_type() == torch::headeronly::ScalarType::Float ||
          mat_a.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "fp32_router_gemm: mat_a must be float32 or bfloat16");
  STD_TORCH_CHECK(mat_b.scalar_type() == torch::headeronly::ScalarType::Float,
                  "fp32_router_gemm: mat_b (weight) must be float32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Float,
                  "fp32_router_gemm: output must be float32");

  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      mat_a.get_device_index());
  STD_TORCH_CHECK(getSMVersion() >= 90, "fp32_router_gemm: requires SM90+");

  auto stream = get_current_cuda_stream(mat_a.get_device_index());
  float* out_ptr = reinterpret_cast<float*>(output.mutable_data_ptr());
  float const* mat_b_ptr = reinterpret_cast<float const*>(mat_b.data_ptr());

  if (mat_a.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    auto const* mat_a_ptr =
        reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr());
    dispatchFp32RouterGemm<__nv_bfloat16>(num_experts, hidden_dim, num_tokens,
                                          out_ptr, mat_a_ptr, mat_b_ptr,
                                          stream);
  } else {
    auto const* mat_a_ptr = reinterpret_cast<float const*>(mat_a.data_ptr());
    dispatchFp32RouterGemm<float>(num_experts, hidden_dim, num_tokens, out_ptr,
                                  mat_a_ptr, mat_b_ptr, stream);
  }
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("fp32_router_gemm", TORCH_BOX(&fp32_router_gemm));
}
