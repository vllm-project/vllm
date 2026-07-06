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

static constexpr int SKINNY_MAX_TOKENS = 32;

// Supported (N, K) pairs (must match the instantiations in
// bf16_skinny_gemm.cu): eh_proj shard TP8 / TP4 / unsharded.
static inline bool bf16_skinny_gemm_supported(int n, int k) {
  if (k == 12288 && (n == 768 || n == 1536 || n == 6144)) return true;
  // LL-mode backbone shapes (wire callers with an M <= 8 guard; the GEMV
  // family loses to cuBLAS at larger M).
  if (k == 2048 && n == 2048) return true;  // q_b_proj (TP8)
  if (k == 6144 && n == 2624) return true;  // fused_qkv_a (wire M <= 2 only)
  if (k == 7168 && n == 2112) return true;  // DSv3.2 fused_qkv_a (M <= 2)
  if (k == 14336 && n == 7168) return true;  // DSv3.2 eh_proj (M <= 2)
  if (k == 6144 && n == 512) return true;   // shared-expert gate_up (TP8)
  return false;
}

// Forward declarations - template params must match bf16_skinny_gemm.cu
template <int kBlockSize, int kNPB, int kNumTokens, int kN, int kK>
void invokeBf16SkinnyGemm(__nv_bfloat16* output, __nv_bfloat16 const* mat_a,
                          __nv_bfloat16 const* mat_b, int64_t out_stride,
                          cudaStream_t stream);

template <int kNPB, int kN, int kK, int kBegin, int kEnd>
struct SkinnyLoopUnroller {
  static void unroll(int num_tokens, __nv_bfloat16* output,
                     __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b,
                     int64_t out_stride, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeBf16SkinnyGemm<128, kNPB, kBegin, kN, kK>(output, mat_a, mat_b,
                                                      out_stride, stream);
    } else {
      SkinnyLoopUnroller<kNPB, kN, kK, kBegin + 1, kEnd>::unroll(
          num_tokens, output, mat_a, mat_b, out_stride, stream);
    }
  }
};

template <int kNPB, int kN, int kK, int kEnd>
struct SkinnyLoopUnroller<kNPB, kN, kK, kEnd, kEnd> {
  static void unroll(int num_tokens, __nv_bfloat16* output,
                     __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b,
                     int64_t out_stride, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeBf16SkinnyGemm<128, kNPB, kEnd, kN, kK>(output, mat_a, mat_b,
                                                    out_stride, stream);
    } else {
      throw std::invalid_argument(
          "bf16_skinny_gemm: num_tokens must be in [1, 32]");
    }
  }
};

static void dispatchBf16SkinnyGemm(int n, int k, int num_tokens,
                                   __nv_bfloat16* output,
                                   __nv_bfloat16 const* mat_a,
                                   __nv_bfloat16 const* mat_b,
                                   int64_t out_stride, cudaStream_t stream) {
  if (n == 768 && k == 12288) {
    SkinnyLoopUnroller<4, 768, 12288, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 1536 && k == 12288) {
    SkinnyLoopUnroller<4, 1536, 12288, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 6144 && k == 12288) {
    SkinnyLoopUnroller<2, 6144, 12288, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 2048 && k == 2048) {
    SkinnyLoopUnroller<4, 2048, 2048, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 2624 && k == 6144) {
    SkinnyLoopUnroller<4, 2624, 6144, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 2112 && k == 7168) {
    SkinnyLoopUnroller<4, 2112, 7168, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 7168 && k == 14336) {
    SkinnyLoopUnroller<2, 7168, 14336, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else if (n == 512 && k == 6144) {
    SkinnyLoopUnroller<4, 512, 6144, 1, SKINNY_MAX_TOKENS>::unroll(
        num_tokens, output, mat_a, mat_b, out_stride, stream);
  } else {
    throw std::invalid_argument(
        "bf16_skinny_gemm: unsupported (N, K) pair");
  }
}

void bf16_skinny_gemm(
    torch::stable::Tensor& output,       // [num_tokens, N] bf16
    torch::stable::Tensor const& mat_a,  // [num_tokens, K] bf16
    torch::stable::Tensor const& mat_b   // [N, K] bf16
) {
  STD_TORCH_CHECK(output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2);
  STD_TORCH_CHECK(output.is_cuda() && mat_a.is_cuda() && mat_b.is_cuda(),
                  "bf16_skinny_gemm: all tensors must be CUDA tensors");
  STD_TORCH_CHECK(mat_a.is_contiguous() && mat_b.is_contiguous(),
                  "bf16_skinny_gemm: inputs must be contiguous");
  // Output may be a column-slice view of a wider padded buffer: unit column
  // stride, row stride >= N (rows must not overlap).
  STD_TORCH_CHECK(output.stride(1) == 1,
                  "bf16_skinny_gemm: output columns must be contiguous");

  const int num_tokens = mat_a.size(0);
  const int n = mat_b.size(0);
  const int k = mat_a.size(1);

  STD_TORCH_CHECK(output.size(0) == num_tokens && output.size(1) == n,
                  "bf16_skinny_gemm: output must be [num_tokens, N]");
  STD_TORCH_CHECK(mat_b.size(1) == k,
                  "bf16_skinny_gemm: mat_a and mat_b must share K");
  STD_TORCH_CHECK(bf16_skinny_gemm_supported(n, k),
                  "bf16_skinny_gemm: unsupported (N, K) pair");
  const int64_t out_stride = output.stride(0);
  STD_TORCH_CHECK(num_tokens == 1 || out_stride >= n,
                  "bf16_skinny_gemm: output rows overlap");
  STD_TORCH_CHECK(num_tokens >= 1 && num_tokens <= SKINNY_MAX_TOKENS,
                  "bf16_skinny_gemm: num_tokens must be in [1, 32]");
  STD_TORCH_CHECK(
      mat_a.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          mat_b.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "bf16_skinny_gemm: all tensors must be bfloat16");

  auto stream = get_current_cuda_stream(mat_a.get_device_index());
  dispatchBf16SkinnyGemm(
      n, k, num_tokens,
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), out_stride,
      stream);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("bf16_skinny_gemm", TORCH_BOX(&bf16_skinny_gemm));
}
