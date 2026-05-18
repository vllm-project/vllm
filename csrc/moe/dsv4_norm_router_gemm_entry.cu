/*
 * TORCH op entry for the fused RMSNorm + router GEMV kernel
 * (DeepSeek V4 Pro).  This op is DSV4-Pro-specific: the kernel is
 * instantiated only for ``num_experts == 384`` and ``hidden_dim ==
 * 7168``.  Other configurations (e.g. DSV4-Flash with H=4096) must
 * fall back to the unfused ``rms_norm`` + ``dsv3_router_gemm`` path.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "core/registration.h"
#include "dsv4_norm_router_gemm.h"

namespace {

// DSV4-Pro hard-coded shape constants.  Renamed from the earlier
// ``kKimiK2NumExperts`` to avoid the misleading impression that this
// kernel targets Kimi K2 — 384 happens to match Kimi K2's gate but the
// intent here is DSV4-Pro.
constexpr int kDsv4NumExperts = 384;
constexpr int kDsv4HiddenDim = 7168;

template <int kBegin, int kEnd>
struct LoopUnroller {
  static void unroll(int num_tokens, float* logits, __nv_bfloat16* normed_x,
                     __nv_bfloat16 const* x, __nv_bfloat16 const* norm_weight,
                     __nv_bfloat16 const* gate_weight, float eps,
                     cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeNormRouterGemm<__nv_bfloat16, kBegin, kDsv4NumExperts,
                           kDsv4HiddenDim>(logits, normed_x, x, norm_weight,
                                           gate_weight, eps, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd>::unroll(num_tokens, logits, normed_x, x,
                                             norm_weight, gate_weight, eps,
                                             stream);
    }
  }
};

template <int kEnd>
struct LoopUnroller<kEnd, kEnd> {
  static void unroll(int num_tokens, float* logits, __nv_bfloat16* normed_x,
                     __nv_bfloat16 const* x, __nv_bfloat16 const* norm_weight,
                     __nv_bfloat16 const* gate_weight, float eps,
                     cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeNormRouterGemm<__nv_bfloat16, kEnd, kDsv4NumExperts,
                           kDsv4HiddenDim>(logits, normed_x, x, norm_weight,
                                           gate_weight, eps, stream);
    } else {
      throw std::invalid_argument(
          "Invalid num_tokens, only supports 1 to 16 for "
          "dsv4_norm_router_gemm");
    }
  }
};

}  // namespace

void dsv4_norm_router_gemm(at::Tensor& logits,    // [num_tokens, E] fp32
                           at::Tensor& normed_x,  // [num_tokens, H] bf16
                           at::Tensor const& x,   // [num_tokens, H] bf16
                           at::Tensor const& norm_weight,  // [H] bf16
                           at::Tensor const& gate_weight,  // [E, H] bf16
                           double eps) {
  TORCH_CHECK(x.dim() == 2 && norm_weight.dim() == 1 && gate_weight.dim() == 2,
              "x must be 2D, norm_weight 1D, gate_weight 2D");
  TORCH_CHECK(logits.dim() == 2 && normed_x.dim() == 2,
              "logits and normed_x must be 2D");

  int const num_tokens = x.size(0);
  int const hidden_dim = x.size(1);
  int const num_experts = gate_weight.size(0);

  TORCH_CHECK(hidden_dim == kDsv4HiddenDim,
              "Expected hidden_dim=", kDsv4HiddenDim,
              " (DSV4-Pro), but got hidden_dim=", hidden_dim);
  TORCH_CHECK(gate_weight.size(1) == hidden_dim,
              "gate_weight.shape[1] must equal x.shape[1]");
  TORCH_CHECK(norm_weight.size(0) == hidden_dim,
              "norm_weight.shape[0] must equal x.shape[1]");
  TORCH_CHECK(num_experts == kDsv4NumExperts,
              "Expected num_experts=", kDsv4NumExperts,
              " (DSV4-Pro), but got num_experts=", num_experts);
  TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
              "num_tokens must be in [1, 16] for dsv4_norm_router_gemm");

  TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(norm_weight.dtype() == at::kBFloat16, "norm_weight must be bf16");
  TORCH_CHECK(gate_weight.dtype() == at::kBFloat16, "gate_weight must be bf16");
  TORCH_CHECK(normed_x.dtype() == at::kBFloat16, "normed_x must be bf16");
  TORCH_CHECK(logits.dtype() == at::kFloat,
              "logits must be float32 (DSV4 router output is hard-coded fp32)");

  TORCH_CHECK(normed_x.size(0) == num_tokens && normed_x.size(1) == hidden_dim,
              "normed_x must be [num_tokens, hidden_dim]");
  TORCH_CHECK(logits.size(0) == num_tokens && logits.size(1) == num_experts,
              "logits must be [num_tokens, num_experts]");

  TORCH_CHECK(x.is_contiguous() && norm_weight.is_contiguous() &&
                  gate_weight.is_contiguous() && normed_x.is_contiguous() &&
                  logits.is_contiguous(),
              "all tensors must be contiguous");

  auto const sm = getSMVersion();
  TORCH_CHECK(sm >= 90 && sm <= 103,
              "dsv4_norm_router_gemm requires SM_90 <= CUDA ARCH <= SM_103");

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();

  auto* logits_ptr = reinterpret_cast<float*>(logits.mutable_data_ptr());
  auto* nx_ptr = reinterpret_cast<__nv_bfloat16*>(normed_x.mutable_data_ptr());
  auto* x_ptr = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
  auto* nw_ptr = reinterpret_cast<__nv_bfloat16 const*>(norm_weight.data_ptr());
  auto* gw_ptr = reinterpret_cast<__nv_bfloat16 const*>(gate_weight.data_ptr());
  float const eps_f = static_cast<float>(eps);

  LoopUnroller<1, 16>::unroll(num_tokens, logits_ptr, nx_ptr, x_ptr, nw_ptr,
                              gw_ptr, eps_f, stream);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("dsv4_norm_router_gemm", &dsv4_norm_router_gemm);
}
