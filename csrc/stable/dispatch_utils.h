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

#define VLLM_STABLE_DISPATCH_CASE_FLOATING_TYPES(...)                  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__) \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                            \
                      VLLM_STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
