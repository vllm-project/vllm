/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from SGLang's W4A8 grouped-MoE implementation introduced in
 * https://github.com/sgl-project/sglang/pull/7772. Modified by the vLLM
 * project for stable libtorch and vLLM CUDA helpers.
 */

#pragma once

#include <cuda.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
__global__ void int4_fp8_get_group_gemm_starts(
    int32_t* expert_offsets, ElementA** a_offsets, ElementB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets, ElementA* a_base_as_int,
    ElementB* b_base_as_int, ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int, int64_t n, int64_t k) {
  int expert_id = threadIdx.x;
  int32_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n / 2;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int;
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * k / 128;
}

namespace {

void run_int4_fp8_get_group_gemm_starts(
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
                  torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(out_tensors.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto stream = get_current_cuda_stream(expert_offsets.get_device_index());
  int4_fp8_get_group_gemm_starts<cutlass::float_e4m3_t, cutlass::int8_t,
                                 cutlass::bfloat16_t, float>
      <<<1, num_experts, 0, stream>>>(
          static_cast<int32_t*>(expert_offsets.data_ptr()),
          static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),
          static_cast<cutlass::int8_t**>(b_ptrs.data_ptr()),
          static_cast<cutlass::bfloat16_t**>(out_ptrs.data_ptr()),
          static_cast<float**>(a_scales_ptrs.data_ptr()),
          static_cast<cutlass::bfloat16_t**>(b_scales_ptrs.data_ptr()),
          static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),
          static_cast<cutlass::int8_t*>(b_tensors.data_ptr()),
          static_cast<cutlass::bfloat16_t*>(out_tensors.data_ptr()),
          static_cast<float*>(a_scales.data_ptr()),
          static_cast<cutlass::bfloat16_t*>(b_scales.data_ptr()),
          out_tensors.size(1), a_tensors.size(1));
}

}  // namespace
