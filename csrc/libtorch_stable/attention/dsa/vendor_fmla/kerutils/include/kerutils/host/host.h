#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cutlass/cuda_host_adapter.hpp>

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
  std::string message = {};

 public:
  template <typename... Args>
  explicit KUException(const char* name, const char* file, const int line,
                       Args&&... args) {
    std::ostringstream oss;

    oss << name << " error (" << file << ":" << line << "): ";
    (oss << ... << args);
    message = oss.str();
  }

  const char* what() const noexcept override { return message.c_str(); }
};

#define THROW_KU_EXCEPTION(name, ...) \
  throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

#define KU_CUDA_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
      THROW_KU_EXCEPTION("CUDA", "CUDA error: ", cudaGetErrorString(status_)); \
    }                                                                          \
  } while (0)

#define KU_CUTLASS_CHECK(call)                                           \
  do {                                                                   \
    cutlass::Status status_ = call;                                      \
    if (status_ != cutlass::Status::kSuccess) {                          \
      fprintf(stderr, "CUTLASS error (%s:%d): %d\n", __FILE__, __LINE__, \
              static_cast<int>(status_));                                \
      THROW_KU_EXCEPTION("CUTLASS",                                      \
                         "CUTLASS error: ", static_cast<int>(status_));  \
    }                                                                    \
  } while (0)

// This `KU_ASSERT` is triggered no matter if the code is compiled with
// `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                              \
  do {                                                                    \
    if (not(cond)) {                                                      \
      fprintf(stderr, "Assertion `%s` failed (%s:%d): ", #cond, __FILE__, \
              __LINE__);                                                  \
      if constexpr (sizeof(#__VA_ARGS__) > 1) {                           \
        fprintf(stderr, ", " __VA_ARGS__);                                \
      }                                                                   \
      fprintf(stderr, "\n");                                              \
      THROW_KU_EXCEPTION("Assertion", "Assertion `", #cond, "` failed."); \
    }                                                                     \
  } while (0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template <typename T>
inline __host__ __device__ constexpr T ceil_div(const T& a, const T& b) {
  return (a + b - 1) / b;
}

template <typename T>
inline __host__ __device__ constexpr T ceil(const T& a, const T& b) {
  return (a + b - 1) / b * b;
}

// A wrapper for make_tensor_map
static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t>& size,
    const std::vector<uint64_t>& strides,  // PAY ATTENTION: In BYTES
    const std::vector<uint32_t>& box_size, void* global_ptr,
    CUtensorMapDataType data_type, CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode =
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t>& element_strides_ = {}) {
  int dim = size.size();
  KU_ASSERT(dim >= 1);

  std::vector<uint32_t> element_strides;
  if (element_strides_.empty()) {
    for (int i = 0; i < dim; ++i) element_strides.push_back(1);
  } else {
    element_strides = element_strides_;
  }
  KU_ASSERT(strides.size() == (uint32_t)dim - 1 &&
            box_size.size() == (uint32_t)dim &&
            element_strides.size() == (uint32_t)dim);

  CUtensorMap result;
  CUresult ret_code = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &result, data_type, dim, global_ptr, size.data(), strides.data(),
      box_size.data(), element_strides.data(), interleave_mode, swizzle_mode,
      l2_promotion, oob_fill);
  if (ret_code != CUresult::CUDA_SUCCESS) {
    auto print_vector = [&](auto t, const char* fmt, const char end = '\n') {
      for (auto elem : t) {
        printf(fmt, elem);
      }
      printf("%c", end);
    };
    fprintf(stderr, "Failed to create tensormap\n");
    fprintf(stderr, "Dim: %d\n", dim);
    printf("size: ");
    print_vector(size, "%lu ");
    printf("strides: ");
    print_vector(strides, "%lu ");
    printf("box_size: ");
    print_vector(box_size, "%u ");
    printf("element_strides: ");
    print_vector(element_strides, "%u ");
    printf("global ptr: 0x%lx\n", (int64_t)global_ptr);
    printf("data_type: %d\n", (int)data_type);
    printf("swizzle_mode: %d\n", (int)swizzle_mode);
    printf("l2_promotion: %d\n", (int)l2_promotion);
    printf("interleave_mode: %d\n", (int)interleave_mode);
    printf("oob_fill: %d\n", (int)oob_fill);
    KU_ASSERT(false);
  }
  return result;
}

// Given strides (in number of elements), this function converts their datatype
// in uint64_t and then multiplies by elem_size
template <typename T>
static inline std::vector<uint64_t> make_stride_helper(
    const std::vector<T>& strides_in_elems, size_t elem_size) {
  std::vector<uint64_t> res;
  for (auto stride : strides_in_elems) {
    res.push_back(((uint64_t)stride) * elem_size);
  }
  return res;
}

}  // namespace kerutils