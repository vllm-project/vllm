#pragma once

#include "libtorch_stable/quantization/vectorization.cuh"
#include "quantization/utils.cuh"

#include <cmath>

// This header is shared between _C and _C_stable_libtorch targets.
// torch_utils.h provides get_device_prop(). We need to pass USE_CUDA
// to the .so to expose some of the shims used by torch_utils.h. For now
// this is only done for _C_stable_libtorch and not for _C, so we use the
// non stable at::cuda::getCurrentDeviceProperties for _C for now.
#ifdef TORCH_TARGET_VERSION
  #include "../../../libtorch_stable/torch_utils.h"
#else
  #ifdef USE_ROCM
    #include <ATen/hip/HIPContext.h>
  #endif
#endif

#ifndef USE_ROCM
  #include "nvidia/quant_utils.cuh"
#else
  #include "amd/quant_utils.cuh"
#endif

// Determines the preferred FP8 type for the current platform.
// Note that for CUDA this just returns true,
// but on ROCm it will check device props.
static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  #ifdef TORCH_TARGET_VERSION
  auto* dprops = get_device_prop();
  #else
  auto* dprops = at::cuda::getCurrentDeviceProperties();
  #endif
  std::string device_arch = dprops->gcnArchName;
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
