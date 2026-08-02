#pragma once

#include "kerutils/device/common.h"
#include "kerutils/device/sm80/intrinsics.cuh"

namespace kerutils {

// Retrieve the value of `%smid` and check its range
CUTE_DEVICE
uint32_t get_sm_id_with_range_check(uint32_t num_physical_sms) {
  uint32_t sm_id = get_sm_id();
  if (!(sm_id < num_physical_sms)) {
    trap();
  }
  return sm_id;
}

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
  #define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
    do {                                   \
      if (not(cond)) asm("trap;");         \
    } while (0)
#endif

// Construct a `float2` from a single `float` by duplicating the value
CUTE_DEVICE
float2 float2float2(const float& x) { return float2{x, x}; }

CUTE_DEVICE
void st_shared(void* ptr, __int128_t val) {
  asm volatile("st.shared.b128 [%0], %1;" ::"l"(__cvta_generic_to_shared(ptr)),
               "q"(val));
}

CUTE_DEVICE
void st_shared(void* ptr, float4 val) { st_shared(ptr, *(__int128_t*)&val); }

CUTE_DEVICE
__int128_t ld_shared(void* ptr) {
  __int128_t val;
  asm volatile("ld.shared.b128 %0, [%1];"
               : "=q"(val)
               : "l"(__cvta_generic_to_shared(ptr)));
  return val;
}

CUTE_DEVICE
float4 ld_shared_float4(void* ptr) {
  __int128_t temp = ld_shared(ptr);
  return *(float4*)&temp;
}

}  // namespace kerutils
