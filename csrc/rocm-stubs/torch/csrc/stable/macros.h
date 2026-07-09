#pragma once
#include <hip/hip_runtime.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace detail {
inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    STD_TORCH_CHECK(false, "HIP error: ", hipGetErrorString(err));
  }
}
} // namespace detail

#define STD_CUDA_CHECK(EXPR)           \
  do {                                 \
    hipError_t _err = (EXPR);          \
    ::detail::hip_check(_err);         \
  } while (0)
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(hipGetLastError())

#define STABLE_TORCH_ERROR_CODE_CHECK(call) TORCH_ERROR_CODE_CHECK(call)





#define TORCH_UTILS_CHECK(cond, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)

#define TORCH_DYNAMIC_VERSION_CALL(VER, SHIM, FALLBACK, ...) FALLBACK(__VA_ARGS__)
#define TORCH_DYNAMIC_VERSION_CALL_2_13_0(SHIM, FALLBACK, ...) FALLBACK(__VA_ARGS__)
