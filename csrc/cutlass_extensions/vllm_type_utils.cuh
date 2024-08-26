#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cuda_bf16.h"

#include "cutlass_extensions/vllm_custom_types.cuh"

namespace cutlass {

template <typename T>
struct nameof {
  static constexpr char const* value = "unknown";
};

template <typename T>
inline constexpr auto nameof_v = nameof<T>::value;

#define NAMEOF_TYPE(T)                       \
  template <>                                \
  struct nameof<T> {                         \
    static constexpr char const* value = #T; \
  };

NAMEOF_TYPE(float_e4m3_t)
NAMEOF_TYPE(float_e5m2_t)
NAMEOF_TYPE(half_t)
NAMEOF_TYPE(nv_bfloat16)
NAMEOF_TYPE(bfloat16_t)
NAMEOF_TYPE(float)

NAMEOF_TYPE(int4b_t)
NAMEOF_TYPE(int8_t)
NAMEOF_TYPE(int32_t)
NAMEOF_TYPE(int64_t)

NAMEOF_TYPE(vllm_uint4b8_t)
NAMEOF_TYPE(uint4b_t)
NAMEOF_TYPE(uint8_t)
NAMEOF_TYPE(vllm_uint8b128_t)
NAMEOF_TYPE(uint32_t)
NAMEOF_TYPE(uint64_t)

};  // namespace cutlass