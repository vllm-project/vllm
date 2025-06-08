#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cmath>

#include "../../dispatch_utils.h"
#include "../vectorization_utils.cuh"

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #include <cub/util_type.cuh>
#else
  #include <hipcub/hipcub.hpp>
  #include <hipcub/util_type.hpp>
#endif

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  // int32_max is not exactly representable as float.
  // Therefore, we need to be careful and manually return int32_max on overflow.
  // For symmetry, we also do the same for int32_min, even though it is exactly
  // representable as float and the conversion should be exact.
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate on the higher end.
  if (dst >= i32_max_f) {
    return i32_max;
  }
  // saturate on the lower end.
  if (dst <= i32_min_f) {
    return i32_min;
  }

  return static_cast<int32_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int32_t&>(dst);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // int32_t dst = std::clamp(x, i8_min, i8_max);
  int32_t dst = (x < i8_min) ? i8_min : (x > i8_max) ? i8_max : x;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_t>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const int hidden_size) {
  int tid = threadIdx.x;
  int stride = blockDim.x;
  float scale = *scale_ptr;

  const scalar_t* row_in = input + blockIdx.x * hidden_size;
  int8_t* row_out = output + blockIdx.x * hidden_size;

  vectorize_with_alignment(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(vec_n_t<int8_t, 16> & dst,
                     const vec_n_t<scalar_t, 16>& src) {
#pragma unroll
        for (int k = 0; k < 16; ++k)
          dst.val[k] = float_to_int8_rn(float(src.val[k]) / scale);
      },
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(float(src) / scale);
      });
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void static_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const azp_t* azp_ptr, const int hidden_size) {
  int tid = threadIdx.x;
  int stride = blockDim.x;
  float scale = *scale_ptr;
  azp_t azp = *azp_ptr;
  float inv_s = 1.0f / scale;

  const scalar_t* row_in = input + blockIdx.x * hidden_size;
  int8_t* row_out = output + blockIdx.x * hidden_size;

  vectorize_with_alignment(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(vec_n_t<int8_t, 16> & dst,
                     const vec_n_t<scalar_t, 16>& src) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          float v = float(src.val[k]) * inv_s;
          dst.val[k] = int32_to_int8(float_to_int32_rn(v) + azp);
        }
      },
      [=] __device__(int8_t& dst, const scalar_t& src) {
        float v = float(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

template <typename scalar_t, typename scale_t>
__global__ void dynamic_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, const int hidden_size) {
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* row_in = input + blockIdx.x * hidden_size;
  int8_t* row_out = output + blockIdx.x * hidden_size;

  // calculate for absmax
  float thread_max = 0.f;
  for (int i = tid; i < hidden_size; i += stride) {
    float v = fabsf(float(row_in[i]));
    thread_max = fmaxf(thread_max, v);
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max = BlockReduce(tmp).Reduce(thread_max, cub::Max{}, blockDim.x);
  __shared__ float absmax;
  if (tid == 0) {
    absmax = block_max;
    scale_out[blockIdx.x] = absmax / 127.f;
  }
  __syncthreads();

  float inv_s = (absmax == 0.f) ? 0.f : 127.f / absmax;

  // 2. quantize
  vectorize_with_alignment(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(vec_n_t<int8_t, 16> & dst,
                     const vec_n_t<scalar_t, 16>& src) {
#pragma unroll
        for (int k = 0; k < 16; ++k)
          dst.val[k] = float_to_int8_rn(float(src.val[k]) * inv_s);
      },
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(float(src) * inv_s);
      });
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, azp_t* azp_out, const int hidden_size) {
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* row_in = input + blockIdx.x * hidden_size;
  int8_t* row_out = output + blockIdx.x * hidden_size;

  // 1. calculate min & max
  float tmin = FLT_MAX;
  float tmax = -FLT_MAX;
  for (int i = tid; i < hidden_size; i += stride) {
    float v = float(row_in[i]);
    tmin = fminf(tmin, v);
    tmax = fmaxf(tmax, v);
  }

  // This is used for BlockReduce
  struct MinMax {
    float mn, mx;
  };
  struct ReduceMinMax {
    __device__ MinMax operator()(const MinMax& a, const MinMax& b) const {
      return {fminf(a.mn, b.mn), fmaxf(a.mx, b.mx)};
    }
  };

  using BlockReduce = cub::BlockReduce<MinMax, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;

  MinMax thread_mm{tmin, tmax};
  MinMax mm = BlockReduce(tmp).Reduce(thread_mm, ReduceMinMax(), blockDim.x);
  __shared__ float scale_sh;
  __shared__ azp_t azp_sh;
  if (tid == 0) {
    float s = (mm.mx - mm.mn) / 255.f;
    float zp = nearbyintf(-128.f - mm.mn / s);  // round-to-even
    scale_sh = s;
    azp_sh = azp_t(zp);
    scale_out[blockIdx.x] = s;
    azp_out[blockIdx.x] = azp_sh;
  }
  __syncthreads();

  float inv_s = 1.f / scale_sh;
  azp_t azp = azp_sh;

  // 2. quantize
  vectorize_with_alignment(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(vec_n_t<int8_t, 16> & dst,
                     const vec_n_t<scalar_t, 16>& src) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          float v = float(src.val[k]) * inv_s;
          dst.val[k] = int32_to_int8(float_to_int32_rn(v) + azp);
        }
      },
      [=] __device__(int8_t& dst, const scalar_t& src) {
        float v = float(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              torch::Tensor const& input,  // [..., hidden_size]
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp || azp->numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 256));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::static_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), hidden_size);
        } else {
          vllm::static_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}

void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales, std::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 256));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), hidden_size);
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}
