#pragma once

#include "attention_generic.cuh"

#include <stdint.h>
#ifdef ENABLE_FP8
  #ifndef USE_ROCM
    #include <cuda_fp8.h>
  #endif  // USE_ROCM
#endif    // ENABLE_FP8

namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  kNvFp4 = 3,
};

// fp8 vector types for quantization of kv cache
template <>
struct Vec<uint8_t, 1> {
  using Type = uint8_t;
};

template <>
struct Vec<uint8_t, 2> {
  using Type = uint16_t;
};

template <>
struct Vec<uint8_t, 4> {
  using Type = uint32_t;
};

template <>
struct Vec<uint8_t, 8> {
  using Type = uint2;
};

}  // namespace vllm
