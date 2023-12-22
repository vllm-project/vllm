#pragma once

#include "attention_generic.cuh"

#include <stdint.h>
#include <cuda_fp8.h>

namespace vllm {
// fp8 vector types for quantization of kv cache

template<>
struct Vec<uint8_t, 1> {
    using Type = uint8_t;
};

template<>
struct Vec<uint8_t, 2> {
    using Type = uint16_t;
};

template<>
struct Vec<uint8_t, 4> {
    using Type = uint32_t;
};

template<>
struct Vec<uint8_t, 8> {
    using Type = uint2;
};

} // namespace vllm
