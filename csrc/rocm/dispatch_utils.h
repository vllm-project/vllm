/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#pragma once

#include <torch/all.h>

// Need a special dispatch case macro since we will nest the FP8 dispatch.
// Instead of the usual 'scalar_t', this names the dispatched type 'fp8_t'.
#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

// ROCm devices might use either fn or fnuz, so set up dispatch table for both.
// A host-based check at runtime will create a preferred FP8 type for ROCm
// such that the correct kernel is dispatched.
#define VLLM_DISPATCH_CASE_FP8_TYPES(...)                          \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

// When using this dispatch macro, the type is 'fp8_t' not 'scalar_t'.
// See AT_DISPATCH_FP8_CASE above.
#define VLLM_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))
