#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/types.h>

namespace vllm {
namespace minimax_ar {

struct alignas(16) bf16x8 {
  __nv_bfloat16 array[8];
};

struct alignas(8) bf16x4 {
  __nv_bfloat16 array[4];
};

template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
  using norm_weight_type = bf16x8;
};

template <>
struct ElemsPerAccess<nv_bfloat16> {
  static constexpr int value = 8;
  using norm_weight_type = bf16x8;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
  using norm_weight_type = bf16x4;
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
  void** workspace{};
  void* allreduce_in{};
  void* rms_norm_out{};
  void* rms_gamma{};
  void* allreduce_in_k{};
  void* rms_norm_out_k{};
  void* rms_gamma_k{};
  float rms_eps{};
  cudaStream_t stream{};
  bool trigger_completion_at_end = true;
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

}  // namespace minimax_ar
}  // namespace vllm
