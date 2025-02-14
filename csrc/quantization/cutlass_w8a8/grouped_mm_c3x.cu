#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"

#include "grouped_mm_c3x.cuh"

using namespace cute;

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
  #define ENABLE_SM90_KERNEL_LEVEL 1
#endif

// for debugging
// __global__ void print_elements(int64_t* tensor, int64_t elements) {
//   if (threadIdx.x == 0) {
//     for (int64_t i = 0; i < elements; ++i) {
//       printf("%ld/%ld ", i, tensor[i]);
//     }
//     printf("\n---\n");
//   }
// }

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (16, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M16 {
  // M in [1, 16]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_4, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

}  // namespace

// TODO
void cutlass_grouped_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  using Cutlass3xGemmM16 = typename sm90_fp8_config_M16<
      ElementAB_Type, ElementC_Type,
      vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      ElementAB_Type, ElementC_Type,
      vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);

  if (m <= 16) {
    cutlass_group_gemm_caller<Cutlass3xGemmM16>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else {
    cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  }
}

__global__ void get_a_expert_offsets(const int* __restrict__ topk_ids,
                                     int32_t* expert_offsets,
                                     int32_t* problem_sizes1,
                                     int32_t* problem_sizes2, int topk_length,
                                     int n, int k) {
  int expert_id = threadIdx.x;
  int num_experts = blockDim.x;

  int occurrences = 0;
  for (int i = 0; i < topk_length; ++i) {
    occurrences += (topk_ids[i] == expert_id);
  }
  problem_sizes1[expert_id * 3] = occurrences;
  problem_sizes1[expert_id * 3 + 1] = 2 * n;
  problem_sizes1[expert_id * 3 + 2] = k;
  problem_sizes2[expert_id * 3] = occurrences;
  problem_sizes2[expert_id * 3 + 1] = k;
  problem_sizes2[expert_id * 3 + 2] = n;
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      tot_offset += problem_sizes1[i * 3];
      expert_offsets[i + 1] = tot_offset;
    }
  }
}

void compute_expert_offsets_caller(const torch::Tensor& topk_ids,
                                   torch::Tensor& expert_offsets,
                                   torch::Tensor& problem_sizes1,
                                   torch::Tensor& problem_sizes2,
                                   const int64_t num_experts, const int64_t n,
                                   const int64_t k) {
  get_a_expert_offsets<<<1, num_experts>>>(
      (const int32_t*)topk_ids.data_ptr(), (int32_t*)expert_offsets.data_ptr(),
      (int32_t*)problem_sizes1.data_ptr(), (int32_t*)problem_sizes2.data_ptr(),
      topk_ids.numel(), n, k);
}
