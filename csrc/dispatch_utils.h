/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#pragma once

#include <torch/extension.h>

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(TYPE, NAME, ...)           \
  AT_DISPATCH_SWITCH(                                                    \
    TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(__VA_ARGS__))
    
#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES(...)             \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))
