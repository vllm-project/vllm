#pragma once

/**
 * __device__ algorithms that perform vectorized loads/stores of input/output.
 */

#include "vectorization.cuh"

namespace vllm {

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

typedef struct __align__(4) {
  c10::Float8_e4m3fn x;
  c10::Float8_e4m3fn y;
  c10::Float8_e4m3fn z;
  c10::Float8_e4m3fn w;
}
float8x4_t;

typedef struct __align__(4) {
  int8_t x;
  int8_t y;
  int8_t z;
  int8_t w;
}
int8x4_t;

} // vllm
