#pragma once

#define MOE_SWITCH(TYPE, ...)                       \
  at::ScalarType _st = ::detail::scalar_type(TYPE); \
  switch (_st) {                                    \
    __VA_ARGS__                                     \
    default:                                        \
      TORCH_CHECK(false, "dispatch fail!")          \
  }

#define MOE_DISPATCH_CASE(enum_type, ...)                  \
  case enum_type: {                                        \
    using scalar_t = ScalarType2CudaType<enum_type>::type; \
    return __VA_ARGS__();                                  \
  }
#define MOE_DISPATCH_FLOAT_CASE(...)                          \
  MOE_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)       \
  MOE_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)        \
  MOE_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)    \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__) \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)

#define MOE_DISPATCH(TYPE, ...) \
  MOE_SWITCH(TYPE, MOE_DISPATCH_FLOAT_CASE(__VA_ARGS__))