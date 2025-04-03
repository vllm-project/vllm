#pragma once

#include <cuda.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

#include "core/scalar_type.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

// get tensors with pointers pointing to the start index of each group's data

template <typename ElementAB, typename ElementC, typename ElementAccumulator>
__global__ void get_moe_gemm_starts(
    int32_t* expert_offsets, ElementAB** a_offsets, ElementAB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets, ElementAB* a_base, ElementAB* b_base,
    ElementC* out_base, ElementAccumulator* a_scales_base,
    ElementAccumulator* b_scales_base, int64_t n, int64_t k, bool per_act_token,
    bool per_out_ch) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base + expert_offset * k;
  b_offsets[expert_id] = b_base + expert_id * k * n;
  out_offsets[expert_id] = out_base + expert_offset * n;
  if (a_scales_offsets != nullptr && a_scales_base != nullptr)
    a_scales_offsets[expert_id] =
        a_scales_base + (per_act_token ? expert_offset : 0);
  if (b_scales_offsets != nullptr && b_scales_base != nullptr)
    b_scales_offsets[expert_id] =
        b_scales_base + (per_out_ch ? n * expert_id : expert_id);
}

#define __CALL_GET_STARTS_KERNEL_8_BIT(TENSOR_C_TYPE, C_TYPE)              \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                         \
    get_moe_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float>              \
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
            a_tensors.size(1), per_act_token, per_out_ch);                 \
  }

#define __CALL_GET_STARTS_KERNEL_16_BIT(TENSOR_ABC_TYPE, ABC_TYPE)            \
  else if (out_tensors.dtype() == TENSOR_ABC_TYPE) {                          \
    get_moe_gemm_starts<ABC_TYPE, ABC_TYPE, float>                            \
        <<<1, num_experts, 0, stream>>>(                                      \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                 \
            static_cast<ABC_TYPE**>(a_ptrs.data_ptr()),                       \
            static_cast<ABC_TYPE**>(b_ptrs.data_ptr()),                       \
            static_cast<ABC_TYPE**>(out_ptrs.data_ptr()), nullptr, nullptr,   \
            static_cast<ABC_TYPE*>(a_tensors.data_ptr()),                     \
            static_cast<ABC_TYPE*>(b_tensors.data_ptr()),                     \
            static_cast<ABC_TYPE*>(out_tensors.data_ptr()), nullptr, nullptr, \
            out_tensors.size(1), a_tensors.size(1), false, false);            \
  }

namespace {

void run_get_moe_gemm_starts_8_bit(
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

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL_8_BIT(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL_8_BIT(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

void run_get_moe_gemm_starts_16_bit(torch::Tensor const& expert_offsets,
                                    torch::Tensor& a_ptrs,
                                    torch::Tensor& b_ptrs,
                                    torch::Tensor& out_ptrs,
                                    torch::Tensor const& a_tensors,
                                    torch::Tensor const& b_tensors,
                                    torch::Tensor& out_tensors) {
  TORCH_CHECK(a_tensors.dtype() == torch::kBFloat16 ||
              a_tensors.dtype() == torch::kFloat16);
  TORCH_CHECK(a_tensors.dtype() == b_tensors.dtype());
  TORCH_CHECK(a_tensors.dtype() == out_tensors.dtype());

  int num_experts = (int)expert_offsets.size(0);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL_16_BIT(torch::kBFloat16, cutlass::bfloat16_t)
  __CALL_GET_STARTS_KERNEL_16_BIT(torch::kFloat16, half)
  else {
    TORCH_CHECK(false, "Invalid i/o type (must be float16 or bfloat16)");
  }
}

// common structs and types used by moe gemm

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <typename ElementAB_, typename ElementC_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_moe_gemm {
  using ElementAB = ElementAB_;
  using ElementC = void;
  using ElementD = ElementC_;
  using ElementAccumulator = float;

  using Epilogue = Epilogue_<ElementAccumulator, ElementD, TileShape>;

  using StrideC =
      cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutC*, AlignmentC, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA*, AlignmentAB, ElementAB,
          LayoutB*, AlignmentAB, ElementAccumulator, TileShape, ClusterShape,
          Stages, KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_only<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

}  // namespace