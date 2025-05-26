#pragma once

#include <cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

#include "core/scalar_type.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <typename ElementAB, typename ElementC, typename ElementAccumulator>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets, ElementAB** a_offsets, ElementAB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int, int64_t n, int64_t k,
    bool per_act_token, bool per_out_ch, bool is_a_padded,
    int64_t padded_a_offset) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  if (is_a_padded) {
    a_offsets[expert_id] = a_base_as_int + padded_a_offset * expert_id * k;
  } else {
    a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  }
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] =
      a_scales_base_as_int + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] =
      b_scales_base_as_int + (per_out_ch ? n * expert_id : expert_id);

  // printf("K: %ld, N: %ld\n", k, n);
  // printf("expert offsets: %d -> %ld, %ld, %ld, %ld, %ld\n", expert_id,
  //      (is_a_padded ? padded_a_offset * expert_id * k : expert_offset * k),
  //      expert_id * k * n, expert_offset * n,
  //      (per_act_token ? expert_offset : 0),
  //      (per_out_ch ? n * expert_id : expert_id));
  // printf("computed offsets: %d -> %ld, %ld, %ld, %ld, %ld\n", expert_id,
  //      a_offsets[expert_id], b_offsets[expert_id], out_offsets[expert_id],
  //      a_scales_offsets[expert_id], b_scales_offsets[expert_id]);
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                    \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                         \
    get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float>            \
        <<<1, num_experts, 0, stream>>>(                                   \
            static_cast<int32_t*>(expert_offsets.data_ptr()),              \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),       \
            static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),       \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                    \
            static_cast<float**>(a_scales_ptrs.data_ptr()),                \
            static_cast<float**>(b_scales_ptrs.data_ptr()),                \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),     \
            static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),     \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                  \
            static_cast<float*>(a_scales.data_ptr()),                      \
            static_cast<float*>(b_scales.data_ptr()), out_tensors.size(1), \
            a_tensors.size(1), per_act_token, per_out_ch, is_a_padded,     \
            padded_a_offset);                                              \
  }

namespace {

void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets, torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs,
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;
  // printf("per_act_token: %d, per_out_ch: %d\n", per_act_token, per_out_ch);

  bool is_a_padded = a_tensors.size(0) != out_tensors.size(0);
  // printf("is_a_padded: %d\n", is_a_padded);
  // std::stringstream ss;
  // ss << "a_tensors shape: [";
  // for (int i = 0; i < a_tensors.dim(); i++) {
  //   ss << a_tensors.size(i);
  //   if (i < a_tensors.dim() - 1) {
  //     ss << ", ";
  //   }
  // }
  // ss << "]" << std::endl;
  // ss << "out_tensors shape: [";
  // for (int i = 0; i < out_tensors.dim(); i++) {
  //   ss << out_tensors.size(i);
  //   if (i < out_tensors.dim() - 1) {
  //     ss << ", ";
  //   }
  // }
  // ss << "]" << std::endl;
  // std::cout << ss.str();
  int64_t padded_a_offset = is_a_padded ? a_tensors.size(0) / num_experts : 0;
  // printf("padded_a_offset: %ld/%d -> %ld\n", a_tensors.size(0), num_experts,
  //       padded_a_offset);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

}  // namespace