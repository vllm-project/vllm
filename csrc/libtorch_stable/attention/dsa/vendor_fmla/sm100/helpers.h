#pragma once

#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "defines.h"

namespace sm100 {

using namespace cute;

CUTE_DEVICE
int int4_max(int4 t) { return max(max(t.x, t.y), max(t.z, t.w)); }

CUTE_DEVICE
int int4_min(int4 t) { return min(min(t.x, t.y), min(t.z, t.w)); }

// Convert 2x fp8_e4m3 to 2x bf16 with scaling
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data,
                                        nv_bfloat16 scale) {
  // TODO Use native conversion for CUDA >= 13.1
  float2 data_float2 = (float2)data;
  nv_bfloat162 data_bf16x2 = __float22bfloat162_rn(data_float2);
  return nv_bfloat162{data_bf16x2.x * scale, data_bf16x2.y * scale};
}

}  // namespace sm100
