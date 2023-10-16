#pragma once

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif
