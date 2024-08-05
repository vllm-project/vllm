#pragma once
/**
 * __device__ algorithms that perform vectorized loads/stores of input/output.
 */

namespace vllm {

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

}  // namespace vllm
