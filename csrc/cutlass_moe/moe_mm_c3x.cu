#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "moe_mm_c3x_8_bit.cuh"
#include "moe_mm_c3x_16_bit.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_8_bit_config_default {
  // M in (16, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_8_bit_config_M16 {
  // M in [1, 16]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_4, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_8_bit_config_K8192 {
  // K in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_8_bit_config_N8192 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_16_bit_config_M512 {
  // M in [1, 512]
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_16_bit_config_default {
  // M in (1024, inf]
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_256, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_moe_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm90_8_bit(
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

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmN8192 = typename sm90_8_bit_config_N8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmK8192 = typename sm90_8_bit_config_K8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM16 = typename sm90_8_bit_config_M16<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmDefault = typename sm90_8_bit_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (n >= 8192) {
    cutlass_moe_gemm_caller_8_bit<Cutlass3xGemmN8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else if (k >= 8192) {
    cutlass_moe_gemm_caller_8_bit<Cutlass3xGemmK8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else if (m <= 16) {
    cutlass_moe_gemm_caller_8_bit<Cutlass3xGemmM16>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else {
    cutlass_moe_gemm_caller_8_bit<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  }
}

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm90_16_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  using Cutlass3xGemmM512 = typename sm90_16_bit_config_M512<
      InType, OutType, vllm::c3x::TrivialEpilogue>::Cutlass3xGemm;
  using Cutlass3xGemmDefault = typename sm90_16_bit_config_default<
      InType, OutType, vllm::c3x::TrivialEpilogue>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (m <= 512) {
    cutlass_moe_gemm_caller_16_bit<Cutlass3xGemmM512>(
        out_tensors, a_tensors, b_tensors, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides);
  } else {
    cutlass_moe_gemm_caller_16_bit<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides);
  }
}

void dispatch_moe_mm_sm90_8_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  if (out_tensors.dtype() == torch::kBFloat16) {
    run_cutlass_moe_mm_sm90_8_bit<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else {
    run_cutlass_moe_mm_sm90_8_bit<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  }
}

void dispatch_moe_mm_sm90_16_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  if (out_tensors.dtype() == torch::kBFloat16) {
    run_cutlass_moe_mm_sm90_16_bit<cutlass::bfloat16_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides);
  } else {
    run_cutlass_moe_mm_sm90_16_bit<cutlass::half_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides);
  }
}

}  // namespace

void cutlass_moe_mm_sm90_8_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  dispatch_moe_mm_sm90_8_bit(out_tensors, a_tensors, b_tensors, a_scales,
                             b_scales, expert_offsets, problem_sizes, a_strides,
                             b_strides, c_strides);
}

void cutlass_moe_mm_sm90_16_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  dispatch_moe_mm_sm90_16_bit(out_tensors, a_tensors, b_tensors, expert_offsets,
                              problem_sizes, a_strides, b_strides, c_strides);
}
