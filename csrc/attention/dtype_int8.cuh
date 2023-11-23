#pragma once

#include <stdint.h>
#include "attention_generic.cuh"
#include "dtype_float32.cuh"

namespace vllm {
// define int8  vector types for quantization of kv cache

template<>
struct Vec<int8_t, 1> {
    using Type = int8_t;
};

template<>
struct Vec<int8_t, 2> {
    using Type = int16_t;
};

template<>
struct Vec<int8_t, 4> {
    using Type = int32_t;
};

template<>
struct Vec<int8_t, 8> {
    using Type = int64_t;
};

template<>
struct FloatVec<int8_t> {
    using Type = float;
};

template<>
struct FloatVec<int16_t> {
    using Type = float2;
};

template<>
struct FloatVec<int32_t> {
    using Type = Float4_;
};

template<>
struct FloatVec<int64_t> {
    using Type = Float8_;
};
}
