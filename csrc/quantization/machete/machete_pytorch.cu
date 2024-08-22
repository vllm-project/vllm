#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

namespace machete {

using namespace vllm;

//
//  Utils (type dispatching)
//

template <typename Fn>
static auto scalar_type_dispatch(ScalarType const& type, Fn fn) {
  if (type == vllm::kU4) {
    return fn(cutlass::uint4b_t{});
  } else if (type == vllm::kU8) {
    return fn(cutlass::uint8_t{});
  } else if (type == vllm::kU4B8) {
    return fn(cutlass::vllm_uint4b8_t{});
  } else if (type == vllm::kU8B128) {
    return fn(cutlass::vllm_uint8b128_t{});
  } else {
    TORCH_CHECK(false, "Unsupported type ", type.str());
  }
}

#define AT_DISPATCH_CASE_SUPPORTED_COMPUTE_TYPES(...) \
  AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__)

#define AT_DISPATCH_SUPPORTED_COMPUTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                             \
                     AT_DISPATCH_CASE_SUPPORTED_COMPUTE_TYPES(__VA_ARGS__))

//
//  Interface
//

std::vector<std::string> supported_schedules(ScalarTypeTorchPtr const& btype) {
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12
  return scalar_type_dispatch(*btype, [&](auto BType) {
    return GemmDispatcher<half_t, decltype(BType)>::supported_schedules();
  });
#else
  TORCH_CHECK(false, "Machete requires CUDA 12.0 or later");
#endif
}

torch::Tensor gemm(torch::Tensor const& A, torch::Tensor const& B,
                   ScalarTypeTorchPtr const& btype,
                   c10::optional<torch::Tensor> const& scales,
                   c10::optional<torch::Tensor> const& zeros,
                   c10::optional<int64_t> group_size,
                   c10::optional<torch::Tensor> const& C,
                   c10::optional<double> alpha, c10::optional<double> beta,
                   c10::optional<std::string> schedule) {
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12
  auto args = PyTorchArguments{.A = A,
                               .B = B,
                               .scales = scales,
                               .zeros = zeros,
                               .group_size = group_size,
                               .C = C,
                               .alpha = alpha,
                               .beta = beta,
                               .schedule = schedule};

  return scalar_type_dispatch(*btype, [&](auto BType) {
    return AT_DISPATCH_SUPPORTED_COMPUTE_TYPES(
        A.scalar_type(), "machete_gemm", [&] {
          using ComputeType = equivalent_cutlass_type_t<scalar_t>;
          return GemmDispatcher<ComputeType, decltype(BType)>::dispatch(args);
        });
  });
#else
  TORCH_CHECK(false, "Machete requires CUDA 12.0 or later");
#endif
}

torch::Tensor prepack_B(torch::Tensor const& B,
                        ScalarTypeTorchPtr const& btype) {
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12
  return scalar_type_dispatch(*btype, [&](auto BType) {
    return PrepackBDispatcher<half_t, decltype(BType), half_t>::dispatch(B);
  });
#else
  TORCH_CHECK(false, "Machete requires CUDA 12.0 or later");
#endif
}

};  // namespace machete
