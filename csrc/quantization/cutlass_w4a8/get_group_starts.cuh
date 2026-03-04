// see csrc/quantization/w8a8/cutlass/moe/get_group_starts.cuh
#pragma once

#include <cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

#include "core/scalar_type.hpp"
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

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                  \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                       \
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
    torch::Tensor const& expert_offsets, torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs,
    torch::Tensor& b_group_scales_ptrs, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor& out_tensors,
    torch::Tensor const& a_scales, torch::Tensor const& b_scales,
    torch::Tensor const& b_group_scales, const int64_t b_group_size) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kInt32);  // int4 8x packed into int32
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_group_scales.dtype() ==
              torch::kFloat8_e4m3fn);  // the underlying torch type is e4m3
  TORCH_CHECK(out_tensors.dtype() ==
              torch::kBFloat16);  // only support bf16 for now
  // expect int64_t to avoid overflow during offset calculations
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt64);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  // logical k, n
  int64_t n = out_tensors.size(1);
  int64_t k = a_tensors.size(1);
  int64_t scale_k = cutlass::ceil_div(k, b_group_size);

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