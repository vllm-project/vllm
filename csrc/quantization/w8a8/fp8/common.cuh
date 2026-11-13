#pragma once

#include "core/cuda_compat.h"
#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"

#include <cmath>

// This header is shared between _rocm_C and _C_stable_libtorch targets.
// torch_utils.h provides get_device_prop(). We need to pass USE_CUDA
// to the .so to expose some of the shims used by torch_utils.h. For now
// this is only done for _C_stable_libtorch and not for _rocm_C, so we use
// the non stable at::cuda::getCurrentDeviceProperties for _rocm_C for now.
#ifdef TORCH_TARGET_VERSION
  #include "torch_utils.h"
#else
  #ifdef USE_ROCM
    #include <ATen/hip/HIPContext.h>
  #endif
#endif

#ifndef USE_ROCM
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
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

// Reciprocal scale for scaled_fp8_conversion_vec. A zero scale only comes
// with all-zero data, which must quantize to zero: 0 * inf is NaN, and the
// saturating cvt, unlike a clamp, keeps NaN.
__device__ __forceinline__ float fp8_reciprocal_scale(float const scale) {
  return scale == 0.0f ? 0.0f : 1.0f / scale;
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

// Quantizes N values, matching N scaled_fp8_conversion calls for non-NaN
// input. With VLLM_FP8_CVT_X2 each cvt converts two values and saturates, so
// the clamp is skipped; an inverted scale must come from fp8_reciprocal_scale.
template <bool is_scale_inverted, typename fp8_type, int N>
__device__ __forceinline__ void scaled_fp8_conversion_vec(fp8_type* dst,
                                                          const float* val,
                                                          float const scale) {
  int i = 0;
#if VLLM_FP8_CVT_X2
  static_assert(std::is_same_v<fp8_type, c10::Float8_e4m3fn>);
  #pragma unroll
  for (; i + 1 < N; i += 2) {
    const float2 x = is_scale_inverted
                         ? make_float2(val[i] * scale, val[i + 1] * scale)
                         : make_float2(val[i] / scale, val[i + 1] / scale);
    const __nv_fp8x2_storage_t q =
        __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    dst[i] = fp8_type(static_cast<uint8_t>(q), fp8_type::from_bits());
    dst[i + 1] = fp8_type(static_cast<uint8_t>(q >> 8), fp8_type::from_bits());
  }
#endif
#pragma unroll
  for (; i < N; ++i) {
    dst[i] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(val[i], scale);
  }
}

}  // namespace vllm
