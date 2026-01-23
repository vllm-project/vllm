// see csrc/quantization/w8a8/cutlass/moe/get_group_starts.cuh
#pragma once

#include <cuda.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>

#include "../../torch_utils.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

// ElementB is int32 (packed int4)
// ElementGroupScale is cutlass::Array<cutlass::float_e4m3_t, 8> (packed fp8)
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator, typename ElementGroupScale>
__global__ void get_group_gemm_starts(
    int64_t* expert_offsets, ElementA** a_offsets, ElementB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementGroupScale** b_group_scales_offsets, ElementA* a_base_as_int,
    ElementB* b_base_as_int, ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int,
    ElementGroupScale* b_group_scales_base_as_int, int64_t n, int64_t k,
    int64_t scale_k) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  // same as w8a8
  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + expert_offset;
  b_scales_offsets[expert_id] = b_scales_base_as_int + (n * expert_id);

  // w4a8 specific
  constexpr int pack_factor = 8;  // pack 8 int4 into int32
  b_offsets[expert_id] = b_base_as_int + (expert_id * k * n / pack_factor);
  b_group_scales_offsets[expert_id] =
      b_group_scales_base_as_int + (expert_id * scale_k * n);
}

#define __CALL_GET_STARTS_KERNEL(SCALAR_TYPE, C_TYPE)                    \
  else if (out_tensors.scalar_type() == SCALAR_TYPE) {                   \
    get_group_gemm_starts<cutlass::float_e4m3_t, int32_t, C_TYPE, float, \
                          cutlass::Array<cutlass::float_e4m3_t, 8>>      \
        <<<1, num_experts, 0, stream>>>(                                 \
            static_cast<int64_t*>(expert_offsets.data_ptr()),            \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),     \
            static_cast<int32_t**>(b_ptrs.data_ptr()),                   \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                  \
            static_cast<float**>(a_scales_ptrs.data_ptr()),              \
            static_cast<float**>(b_scales_ptrs.data_ptr()),              \
            static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>**>(     \
                b_group_scales_ptrs.data_ptr()),                         \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),   \
            static_cast<int32_t*>(b_tensors.data_ptr()),                 \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                \
            static_cast<float*>(a_scales.data_ptr()),                    \
            static_cast<float*>(b_scales.data_ptr()),                    \
            static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>*>(      \
                b_group_scales.data_ptr()),                              \
            n, k, scale_k);                                              \
  }

namespace {

void run_get_group_gemm_starts(
    torch::stable::Tensor const& expert_offsets, torch::stable::Tensor& a_ptrs,
    torch::stable::Tensor& b_ptrs, torch::stable::Tensor& out_ptrs,
    torch::stable::Tensor& a_scales_ptrs, torch::stable::Tensor& b_scales_ptrs,
    torch::stable::Tensor& b_group_scales_ptrs,
    torch::stable::Tensor const& a_tensors,
    torch::stable::Tensor const& b_tensors, torch::stable::Tensor& out_tensors,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& b_group_scales, const int64_t b_group_size) {
  STD_TORCH_CHECK(a_tensors.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(
      b_tensors.scalar_type() ==
      torch::headeronly::ScalarType::Int);  // int4 8x packed into int32
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(
      b_group_scales.scalar_type() ==
      torch::headeronly::ScalarType::Float8_e4m3fn);  // the underlying torch
                                                      // type is e4m3
  STD_TORCH_CHECK(
      out_tensors.scalar_type() ==
      torch::headeronly::ScalarType::BFloat16);  // only support bf16 for now
  // expect int64_t to avoid overflow during offset calculations
  STD_TORCH_CHECK(expert_offsets.scalar_type() ==
                  torch::headeronly::ScalarType::Long);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  // logical k, n
  int64_t n = out_tensors.size(1);
  int64_t k = a_tensors.size(1);
  int64_t scale_k = cutlass::ceil_div(k, b_group_size);

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
