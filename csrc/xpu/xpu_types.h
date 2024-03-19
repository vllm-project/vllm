
#ifndef XPU_TYPES_H
#define XPU_TYPES_H

#include <torch/extension.h>

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_XPU_DISPATCH_CASE_FLOATING_TYPES(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_XPU_DISPATCH_CASE_FLOATING_TYPES_FLOAT_ONLY(...) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)        \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define VLLM_XPU_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE, NAME, VLLM_XPU_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_XPU_DISPATCH_FLOATING_TYPES_FLOAT_ONLY(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE, NAME, VLLM_XPU_DISPATCH_CASE_FLOATING_TYPES_FLOAT_ONLY(__VA_ARGS__))

#endif