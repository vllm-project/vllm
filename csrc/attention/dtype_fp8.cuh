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
};

inline Fp8KVCacheDataType get_fp8_kv_cache_data_type(
    const std::string& dtype_str) {
  // dtype_str refers to CacheDType at vllm.config.cache.CacheDType
  if (dtype_str == "auto" || dtype_str == "float16" ||
      dtype_str == "bfloat16") {
    // unquantized kv cache
    return Fp8KVCacheDataType::kAuto;
  } else if (dtype_str == "fp8" || dtype_str == "fp8_ds_mla" ||
             dtype_str == "fp8_e4m3") {
    return Fp8KVCacheDataType::kFp8E4M3;
  } else if (dtype_str == "fp8_e5m2") {
    return Fp8KVCacheDataType::kFp8E5M2;
  }
  TORCH_CHECK(false, "Unsupported fp8 kv cache data type: ", dtype_str);
}

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
