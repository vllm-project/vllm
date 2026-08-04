// SPDX-License-Identifier: Apache-2.0

/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/cuda_compat.h
 */

#pragma once
#ifndef USE_ROCM
  #define LMCACHE_LDG(arg) __ldg(arg)
#else
  #define LMCACHE_LDG(arg) *(arg)
#endif