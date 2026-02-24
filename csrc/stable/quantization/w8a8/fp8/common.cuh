#pragma once

#include "../../vectorization.cuh"
#include "../../utils.cuh"
#include "../../../torch_utils.h"

#include <cmath>
#include <string>

#ifndef USE_ROCM
  #include "../../../../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include "../../../../quantization/w8a8/fp8/amd/quant_utils.cuh"
#endif

// Determines the preferred FP8 type for the current platform.
// Returns true for OCP format (Float8_e4m3fn), false for FNUZ format
// (Float8_e4m3fnuz). On CUDA this always returns true. On ROCm it checks
// device properties to determine the format.
inline bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto* dprops = get_device_prop();
  std::string device_arch = dprops->gcnArchName;
  // gfx94x devices use FNUZ format, others use OCP format
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
#ifndef USE_ROCM
  // Use hardware cvt instruction for fp8 on nvidia
  // Currently only support fp8_type = c10::Float8_e4m3fn
  return fp8::vec_conversion<fp8_type, float>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

}  // namespace vllm
