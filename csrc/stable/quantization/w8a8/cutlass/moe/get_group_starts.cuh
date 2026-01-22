#pragma once

#include <cuda.h>

#include "stable/torch_utils.h"

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <typename ElementAB, typename ElementC, typename ElementAccumulator>
__global__ void get_group_gemm_starts(
    int64_t* expert_offsets, ElementAB** a_offsets, ElementAB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int, int64_t n, int64_t k,
    bool per_act_token, bool per_out_ch) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] =
      a_scales_base_as_int + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] =
      b_scales_base_as_int + (per_out_ch ? n * expert_id : expert_id);
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                        \
  else if (out_tensors.scalar_type() == TENSOR_C_TYPE) {                       \
    get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float>                \
        <<<1, num_experts, 0, stream>>>(                                       \
            static_cast<int64_t*>(expert_offsets.mutable_data_ptr()),          \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.mutable_data_ptr()),   \
            static_cast<cutlass::float_e4m3_t**>(b_ptrs.mutable_data_ptr()),   \
            static_cast<C_TYPE**>(out_ptrs.mutable_data_ptr()),                \
            static_cast<float**>(a_scales_ptrs.mutable_data_ptr()),            \
            static_cast<float**>(b_scales_ptrs.mutable_data_ptr()),            \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.mutable_data_ptr()), \
            static_cast<cutlass::float_e4m3_t*>(b_tensors.mutable_data_ptr()), \
            static_cast<C_TYPE*>(out_tensors.mutable_data_ptr()),              \
            static_cast<float*>(a_scales.mutable_data_ptr()),                  \
            static_cast<float*>(b_scales.mutable_data_ptr()),                  \
            out_tensors.size(1), a_tensors.size(1), per_act_token,             \
            per_out_ch);                                                       \
  }

namespace {

void run_get_group_gemm_starts(
    torch::stable::Tensor const& expert_offsets, torch::stable::Tensor& a_ptrs,
    torch::stable::Tensor& b_ptrs, torch::stable::Tensor& out_ptrs,
    torch::stable::Tensor& a_scales_ptrs, torch::stable::Tensor& b_scales_ptrs,
    torch::stable::Tensor const& a_tensors,
    torch::stable::Tensor const& b_tensors, torch::stable::Tensor& out_tensors,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  STD_TORCH_CHECK(a_tensors.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b_tensors.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  // expect int64_t to avoid overflow during offset calculations
  STD_TORCH_CHECK(expert_offsets.scalar_type() ==
                  torch::headeronly::ScalarType::Long);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  auto stream = get_current_cuda_stream(a_tensors.get_device_index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::headeronly::ScalarType::BFloat16,
                           cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL(torch::headeronly::ScalarType::Half, half)
  else {
    STD_TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

}  // namespace
