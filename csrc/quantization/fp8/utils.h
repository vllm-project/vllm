/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <sstream>

#ifndef USE_ROCM
// Adapt from FlashInfer

  #define _DISPATCH_CASE_F16(c_type, ...) \
    case at::ScalarType::Half: {          \
      using c_type = nv_half;             \
      return __VA_ARGS__();               \
    }

  #define _DISPATCH_CASE_BF16(c_type, ...) \
    case at::ScalarType::BFloat16: {       \
      using c_type = nv_bfloat16;          \
      return __VA_ARGS__();                \
    }

  #define _DISPATCH_CASE_FP8_E4M3(c_type, ...) \
    case at::ScalarType::Float8_e4m3fn: {      \
      using c_type = __nv_fp8_e4m3;            \
      return __VA_ARGS__();                    \
    }

  #define _DISPATCH_CASE_FP8_E5M2(c_type, ...) \
    case at::ScalarType::Float8_e5m2: {        \
      using c_type = __nv_fp8_e5m2;            \
      return __VA_ARGS__();                    \
    }

  #define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...) \
    [&]() -> bool {                                                        \
      switch (pytorch_dtype) {                                             \
        _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                            \
        _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                           \
        default:                                                           \
          std::ostringstream oss;                                          \
          oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "   \
              << pytorch_dtype;                                            \
          TORCH_CHECK(false, oss.str());                                   \
          return false;                                                    \
      }                                                                    \
    }()

  #define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(pytorch_dtype, c_type, ...)    \
    [&]() -> bool {                                                          \
      switch (pytorch_dtype) {                                               \
        _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                         \
        _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                         \
        default:                                                             \
          std::ostringstream oss;                                            \
          oss << __PRETTY_FUNCTION__ << " failed to dispatch fp8 data type " \
              << pytorch_dtype;                                              \
          TORCH_CHECK(false, oss.str());                                     \
          return false;                                                      \
      }                                                                      \
    }()

  #define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)    \
    [&]() -> bool {                                                      \
      switch (pytorch_dtype) {                                           \
        _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                          \
        _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                         \
        _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                     \
        _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                     \
        default:                                                         \
          std::ostringstream oss;                                        \
          oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " \
              << pytorch_dtype;                                          \
          TORCH_CHECK(false, oss.str());                                 \
          return false;                                                  \
      }                                                                  \
    }()

  #define _DISPATCH_SWITCH(var_name, cond, ...)                             \
    [&]() -> bool {                                                         \
      switch (cond) {                                                       \
        __VA_ARGS__                                                         \
        default:                                                            \
          std::ostringstream oss;                                           \
          oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " \
              << int(cond);                                                 \
          TORCH_CHECK(false, oss.str());                                    \
          return false;                                                     \
      }                                                                     \
    }()

  #define _DISPATCH_SWITCH_U16x2(var1_name, var2_name, cond1, cond2, ...) \
    [&]() -> bool {                                                       \
      switch (pack_u16(cond1, cond2)) {                                   \
        __VA_ARGS__                                                       \
        default:                                                          \
          std::ostringstream oss;                                         \
          oss << __PRETTY_FUNCTION__                                      \
              << " failed to dispatch (" var1_name ", " var2_name "): ("  \
              << int(cond1) << ", " << int(cond2) << ")";                 \
          TORCH_CHECK(false, oss.str());                                  \
          return false;                                                   \
      }                                                                   \
    }()

  #define _DISPATCH_CASE(case_expr, case_var, ...) \
    case case_expr: {                              \
      constexpr auto case_var = case_expr;         \
      return __VA_ARGS__();                        \
    }

  #define _DISPATCH_CASE_U16x2(case_expr1, case_expr2, case_var1, case_var2, \
                               ...)                                          \
    case pack_u16(case_expr1, case_expr2): {                                 \
      constexpr auto case_var1 = case_expr1;                                 \
      constexpr auto case_var2 = case_expr2;                                 \
      return __VA_ARGS__();                                                  \
    }

  #define DISPATCH_BOOL(expr, const_expr, ...) \
    [&]() -> bool {                            \
      if (expr) {                              \
        constexpr bool const_expr = true;      \
        return __VA_ARGS__();                  \
      } else {                                 \
        constexpr bool const_expr = false;     \
        return __VA_ARGS__();                  \
      }                                        \
    }()

inline void check_shape(const at::Tensor& a, const at::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

  #define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads)        \
    TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads(",    \
                num_qo_heads, ") must be divisible by num_kv_heads(", \
                num_kv_heads, ")")

  #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

  #define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
  #define CHECK_LAST_DIM_CONTIGUOUS(x)                    \
    TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, \
                #x "must be contiguous at last dimension")

  #define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)
  #define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
    CHECK_CUDA(x);                           \
    CHECK_LAST_DIM_CONTIGUOUS(x)

  #define CHECK_DIM(d, x) \
    TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

  #define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

  #define CHECK_EQ(a, b) \
    TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

  #define CHECK_GE(a, b) \
    TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
         tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}
#endif

struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(const char* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw cuda_error(_message.str());                                 \
    }                                                                   \
  } while (0)

#define CHECK_IS_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_IS_CUDA(x);         \
  CHECK_IS_CONTIGUOUS(x)

inline int getSMVersion() {
  int device{-1};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline bool getBoolEnv(char const* name) {
  char const* env = std::getenv(name);
  return env && env[0] == '1' && env[1] == '\0';
}

inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to
      // `1`
      enablePDL = getBoolEnv("TRTLLM_ENABLE_PDL");
    }
  });
  return enablePDL;
}

// SGLANG_SHFL_XOR_* adapted from
// https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/cuda_compat.h#L19-L28
#ifndef USE_ROCM
  #define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) \
    __shfl_xor_sync((mask), (var), (lane_mask))
  #define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
    __shfl_xor_sync((mask), (var), (lane_mask), (width))
#else
  #define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) \
    __shfl_xor((var), (lane_mask))
  #define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
    __shfl_xor((var), (lane_mask), (width))
#endif

#ifndef USE_ROCM
  #define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, \
                                                     ...)                   \
    [&]() -> bool {                                                         \
      switch (pytorch_dtype) {                                              \
        case at::ScalarType::Float: {                                       \
          using c_type = float;                                             \
          return __VA_ARGS__();                                             \
        }                                                                   \
          _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                           \
          _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                          \
        default:                                                            \
          std::ostringstream oss;                                           \
          oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "    \
              << pytorch_dtype;                                             \
          TORCH_CHECK(false, oss.str());                                    \
          return false;                                                     \
      }                                                                     \
    }()
#endif

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

#ifndef USE_ROCM
  #include <c10/util/Float8_e4m3fn.h>
using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX =
    std::numeric_limits<FP8_TYPE>::max();
#else
  #include <c10/util/Float8_e4m3fnuz.h>

using FP8_TYPE = c10::Float8_e4m3fnuz;
constexpr auto FP8_E4M3_MAX = 224.0f;
#endif

#ifndef USE_ROCM
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

__device__ __forceinline__ float warpReduceMax(float max_value) {
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 16));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 8));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 4));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 2));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 1));
  return max_value;
}

__device__ __forceinline__ float blockReduceMax(float max_value) {
  static __shared__ float warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  max_value = warpReduceMax(max_value);

  if (laneId == 0) warpLevelMaxs[warpId] = max_value;
  __syncthreads();

  max_value =
      (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) max_value = warpReduceMax(max_value);

  return max_value;
}
#endif

// Pads to a multiple of `alignment` rows.
inline torch::Tensor pad_tensor(const torch::Tensor& tensor,
                                int64_t alignment = 4,
                                bool is_column_major = false) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t pad_rows =
      (alignment - (rows % alignment)) % alignment;  // Compute padding size

  if (pad_rows == 0) {
    return tensor;  // Already aligned
  }

  torch::Tensor padding = torch::zeros({pad_rows, cols}, tensor.options());
  torch::Tensor tensor_padded =
      torch::cat({tensor, padding}, 0);  // Pad along rows

  // Ensure column-major layout
  if (is_column_major) {
    return tensor_padded.t().contiguous().t();
  }
  return tensor_padded;
}

// Get the next power of 2 of a number
inline uint32_t next_pow2(uint32_t x) noexcept {
  if (x <= 1) return 1;
  return 1u << (32 - __builtin_clz(x - 1));
}