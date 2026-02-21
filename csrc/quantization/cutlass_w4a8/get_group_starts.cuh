// see csrc/quantization/w8a8/cutlass/moe/get_group_starts.cuh
#pragma once

#include <cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

#include "core/scalar_type.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

// ElementB is int32 (packed int4)
// ElementGroupScale is cutlass::Array<cutlass::float_e4m3_t, 8> (packed fp8) or
// cutlass::bfloat16_t
// a_scales_offsets and b_scales_offsets are optional for W4A16
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator, typename ElementGroupScale>
__global__ void get_group_gemm_starts(
    int64_t* expert_offsets, ElementA** a_offsets, ElementB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementGroupScale** b_group_scales_offsets, ElementA* a_base,
    ElementB* b_base_as_int, ElementC* out_base,
    ElementAccumulator* a_scales_base, ElementAccumulator* b_scales_base,
    ElementGroupScale* b_group_scales_base, int64_t n, int64_t k,
    int64_t scale_k) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  // same as w8a8
  a_offsets[expert_id] = a_base + expert_offset * k;
  out_offsets[expert_id] = out_base + expert_offset * n;
  if (a_scales_offsets)
    a_scales_offsets[expert_id] = a_scales_base + expert_offset;
  if (b_scales_offsets)
    b_scales_offsets[expert_id] = b_scales_base + (n * expert_id);

  // int4 specific
  constexpr int pack_factor = 8;  // pack 8 int4 into int32
  b_offsets[expert_id] = b_base_as_int + (expert_id * k * n / pack_factor);
  b_group_scales_offsets[expert_id] =
      b_group_scales_base + (expert_id * scale_k * n);
}

#define __CALL_GET_STARTS_KERNEL_FP8(TENSOR_C_TYPE, C_TYPE)              \
  else if (a_tensors.dtype() == torch::kFloat8_e4m3fn &&                 \
           out_tensors.dtype() == TENSOR_C_TYPE) {                       \
    get_group_gemm_starts<cutlass::float_e4m3_t, int32_t, C_TYPE, float, \
                          cutlass::Array<cutlass::float_e4m3_t, 8>>      \
        <<<1, num_experts, 0, stream>>>(                                 \
            static_cast<int64_t*>(expert_offsets.data_ptr()),            \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),     \
            static_cast<int32_t**>(b_ptrs.data_ptr()),                   \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                  \
            a_scales.has_value()                                         \
                ? static_cast<float**>(a_scales_ptrs.value().data_ptr()) \
                : nullptr,                                               \
            b_scales.has_value()                                         \
                ? static_cast<float**>(b_scales_ptrs.value().data_ptr()) \
                : nullptr,                                               \
            static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>**>(     \
                b_group_scales_ptrs.data_ptr()),                         \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),   \
            static_cast<int32_t*>(b_tensors.data_ptr()),                 \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                \
            a_scales.has_value()                                         \
                ? static_cast<float*>(a_scales.value().data_ptr())       \
                : nullptr,                                               \
            b_scales.has_value()                                         \
                ? static_cast<float*>(b_scales.value().data_ptr())       \
                : nullptr,                                               \
            static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>*>(      \
                b_group_scales.data_ptr()),                              \
            n, k, scale_k);                                              \
  }

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                      \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                           \
    get_group_gemm_starts<C_TYPE, int32_t, C_TYPE, float, C_TYPE>            \
        <<<1, num_experts, 0, stream>>>(                                     \
            static_cast<int64_t*>(expert_offsets.data_ptr()),                \
            static_cast<C_TYPE**>(a_ptrs.data_ptr()),                        \
            static_cast<int32_t**>(b_ptrs.data_ptr()),                       \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                      \
            a_scales.has_value()                                             \
                ? static_cast<float**>(a_scales_ptrs.value().data_ptr())     \
                : nullptr,                                                   \
            b_scales.has_value()                                             \
                ? static_cast<float**>(b_scales_ptrs.value().data_ptr())     \
                : nullptr,                                                   \
            static_cast<C_TYPE**>(b_group_scales_ptrs.data_ptr()),           \
            static_cast<C_TYPE*>(a_tensors.data_ptr()),                      \
            static_cast<int32_t*>(b_tensors.data_ptr()),                     \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                    \
            a_scales.has_value()                                             \
                ? static_cast<float*>(a_scales.value().data_ptr())           \
                : nullptr,                                                   \
            b_scales.has_value()                                             \
                ? static_cast<float*>(b_scales.value().data_ptr())           \
                : nullptr,                                                   \
            static_cast<C_TYPE*>(b_group_scales.data_ptr()), n, k, scale_k); \
  }

namespace {

void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets, torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    std::optional<torch::Tensor> a_scales_ptrs,
    std::optional<torch::Tensor> b_scales_ptrs,
    torch::Tensor& b_group_scales_ptrs, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor& out_tensors,
    const std::optional<torch::Tensor>& a_scales,
    const std::optional<torch::Tensor>& b_scales,
    torch::Tensor const& b_group_scales, const int64_t b_group_size) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn ||
              a_tensors.dtype() == torch::kBFloat16);
  TORCH_CHECK(b_tensors.dtype() == torch::kInt32);  // int4 8x packed into int32
  if (a_scales.has_value())
    TORCH_CHECK(a_scales.value().dtype() == torch::kFloat32);
  if (b_scales.has_value())
    TORCH_CHECK(b_scales.value().dtype() == torch::kFloat32);
  TORCH_CHECK(b_group_scales.dtype() == torch::kFloat8_e4m3fn ||
              b_group_scales.dtype() ==
                  torch::kBFloat16);  // the underlying torch type is e4m3
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
  __CALL_GET_STARTS_KERNEL_FP8(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL_FP8(torch::kFloat16, half)
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

}  // namespace