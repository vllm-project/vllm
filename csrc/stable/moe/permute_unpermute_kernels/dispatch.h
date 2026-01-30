#pragma once
#include <cuda_fp8.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define MOE_SWITCH(TYPE, ...)                                         \
  torch::headeronly::ScalarType _st = TYPE;                           \
  switch (_st) {                                                      \
    __VA_ARGS__                                                       \
    default:                                                          \
      STD_TORCH_CHECK(false, "[moe permute]data type dispatch fail!") \
  }

#define MOE_DISPATCH_CASE(enum_type, ...)                  \
  case enum_type: {                                        \
    using scalar_t = ScalarType2CudaType<enum_type>::type; \
    __VA_ARGS__();                                         \
    break;                                                 \
  }

#define MOE_DISPATCH_FLOAT_CASE(...)                                           \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)         \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)          \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)      \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e5m2, __VA_ARGS__)   \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
  MOE_DISPATCH_CASE(torch::headeronly::ScalarType::Byte, __VA_ARGS__)

#define MOE_DISPATCH(TYPE, ...) \
  MOE_SWITCH(TYPE, MOE_DISPATCH_FLOAT_CASE(__VA_ARGS__))

template <torch::headeronly::ScalarType type>
struct ScalarType2CudaType;

template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::Float> {
  using type = float;
};
template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::Half> {
  using type = half;
};
template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::BFloat16> {
  using type = __nv_bfloat16;
};
// uint8 for packed fp4
template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::Byte> {
  using type = uint8_t;
};

// fp8
template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::Float8_e5m2> {
  using type = __nv_fp8_e5m2;
};
template <>
struct ScalarType2CudaType<torch::headeronly::ScalarType::Float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
};
