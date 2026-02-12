#ifndef FLASHINFER_UTILS_MINIMAL_CUH_
#define FLASHINFER_UTILS_MINIMAL_CUH_

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>

namespace vllm {

// Error class
class Error : public std::exception {
 private:
  std::string msg_;

 public:
  explicit Error(const std::string& func, const std::string& file, int lineno,
                 const std::string& msg)
      : msg_(func + " at " + file + ":" + std::to_string(lineno) + ": " + msg) {}
  const char* what() const noexcept override { return msg_.c_str(); }
};

}  // namespace vllm

// Error macro
#define FLASHINFER_ERROR(message) \
  throw vllm::Error(__FUNCTION__, __FILE__, __LINE__, message)

// CUDA error checking macro
#define FLASHINFER_CUDA_CALL(func, ...)                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      return e;                                                                \
    }                                                                          \
  }

// Dispatch macro for vectorization size
#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      FLASHINFER_ERROR(err_msg.str());                                     \
    }                                                                      \
  }

namespace vllm {

namespace math {

// PTX reciprocal (approximate)
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace math

// Math utilities
template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_down(const T1 x, const T2 y) noexcept {
  return (x / y) * y;
}

}  // namespace vllm

#endif  // FLASHINFER_UTILS_MINIMAL_CUH_
