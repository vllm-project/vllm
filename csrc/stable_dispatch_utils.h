/*
 * Stable ABI compatible dispatch utilities for vLLM.
 * Adapted from dispatch_utils.h to use PyTorch's header-only (THO_*) macros
 * instead of the ATen (AT_*) macros.
 *
 * These macros use:
 * - THO_DISPATCH_SWITCH instead of AT_DISPATCH_SWITCH
 * - THO_DISPATCH_CASE instead of AT_DISPATCH_CASE
 * - torch::headeronly::ScalarType instead of at::ScalarType
 *
 * Add more macros here as needed when migrating additional kernels.
 */
#pragma once

#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

// Need a special dispatch case macro since we will nest the FP8 dispatch.
// Instead of the usual 'scalar_t', this names the dispatched type 'fp8_t'.
#define VLLM_STABLE_DISPATCH_FP8_CASE(enum_type, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define VLLM_STABLE_DISPATCH_CASE_FLOATING_TYPES(...)                  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__) \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                            \
                      VLLM_STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

// FP8 type dispatch - ROCm uses FNUZ format, CUDA uses OCP format
#ifdef USE_ROCM
  #define VLLM_STABLE_DISPATCH_CASE_FP8_TYPES(...)                 \
    VLLM_STABLE_DISPATCH_FP8_CASE(                                 \
        torch::headeronly::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
    VLLM_STABLE_DISPATCH_FP8_CASE(                                 \
        torch::headeronly::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

  #define VLLM_STABLE_DISPATCH_CASE_QUANT_TYPES(...)                  \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e4m3fn,   \
                      __VA_ARGS__)                                    \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e4m3fnuz, \
                      __VA_ARGS__)                                    \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Char, __VA_ARGS__)
#else
  #define VLLM_STABLE_DISPATCH_CASE_FP8_TYPES(...) \
    VLLM_STABLE_DISPATCH_FP8_CASE(                 \
        torch::headeronly::ScalarType::Float8_e4m3fn, __VA_ARGS__)

  #define VLLM_STABLE_DISPATCH_CASE_QUANT_TYPES(...)                \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e4m3fn, \
                      __VA_ARGS__)                                  \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Char, __VA_ARGS__)
#endif

// When using this dispatch macro, the type is 'fp8_t' not 'scalar_t'.
// See VLLM_STABLE_DISPATCH_FP8_CASE above.
#define VLLM_STABLE_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                       \
                      VLLM_STABLE_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

#define VLLM_STABLE_DISPATCH_QUANT_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                         \
                      VLLM_STABLE_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__))

// Vector size dispatch
#define VLLM_STABLE_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
  switch (VEC_SIZE) {                                \
    case 16: {                                       \
      constexpr int vec_size = 16;                   \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 8: {                                        \
      constexpr int vec_size = 8;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 4: {                                        \
      constexpr int vec_size = 4;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    case 2: {                                        \
      constexpr int vec_size = 2;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
    default: {                                       \
      constexpr int vec_size = 1;                    \
      __VA_ARGS__();                                 \
      break;                                         \
    }                                                \
  }

// Boolean dispatch
#define VLLM_STABLE_DISPATCH_BOOL(expr, const_expr, ...) \
  if (expr) {                                            \
    constexpr bool const_expr = true;                    \
    __VA_ARGS__();                                       \
  } else {                                               \
    constexpr bool const_expr = false;                   \
    __VA_ARGS__();                                       \
  }

// Group size dispatch (for quantization)
#define VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, const_group_size, ...) \
  if (group_size == 128) {                                                 \
    constexpr int const_group_size = 128;                                  \
    __VA_ARGS__();                                                         \
  } else if (group_size == 64) {                                           \
    constexpr int const_group_size = 64;                                   \
    __VA_ARGS__();                                                         \
  }

// Tensor rank dispatch (2D, 3D, 4D)
#define VLLM_STABLE_DISPATCH_RANK234(NUM_DIMS, ...)                     \
  switch (NUM_DIMS) {                                                   \
    case 2: {                                                           \
      constexpr int tensor_rank = 2;                                    \
      __VA_ARGS__();                                                    \
      break;                                                            \
    }                                                                   \
    case 3: {                                                           \
      constexpr int tensor_rank = 3;                                    \
      __VA_ARGS__();                                                    \
      break;                                                            \
    }                                                                   \
    case 4: {                                                           \
      constexpr int tensor_rank = 4;                                    \
      __VA_ARGS__();                                                    \
      break;                                                            \
    }                                                                   \
    default:                                                            \
      STD_TORCH_CHECK(false, "Expects rank 2, 3 or 4 tensors but got ", \
                      NUM_DIMS);                                        \
  }
