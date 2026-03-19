#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/types.h>

namespace vllm {
namespace tensorrt_llm {

template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<nv_bfloat16> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
  using vec_type = float4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams {
  int nranks{};
  int rank{};
  at::ScalarType dtype{at::ScalarType::Undefined};
  int size_q{};
  int hidden_dim{};
  int size_k{};
  int hidden_dim_k{};
  int stride_q{};  // row stride for q (elements); when > hidden_dim, q is
                   // part of a wider qkv tensor
  int stride_k{};  // row stride for k (elements); when > hidden_dim_k, k is
                   // part of a wider qkv tensor
  void** workspace{};
  void* allreduce_in{};
  void* rms_norm_out{};
  void* rms_gamma{};
  void* allreduce_in_k{};
  void* rms_norm_out_k{};
  void* rms_gamma_k{};
  float rms_eps{};
  cudaStream_t stream{};
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

}  // namespace tensorrt_llm
}  // namespace vllm
