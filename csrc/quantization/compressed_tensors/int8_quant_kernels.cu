#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
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
  dst = std::clamp(dst, i8_min, i8_max);
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
  int32_t dst = std::clamp(x, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) / scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void static_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, azp_type const* azp_ptr,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;
  azp_type const azp = *azp_ptr;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[i]);
    auto const quant_val = int32_to_int8(float_to_int32_rn(val / scale) + azp);
    out[i] = quant_val;
  }
}

template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) * tmp_scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, const int hidden_size) {
  int64_t const token_idx = blockIdx.x;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  // Scan for the min and max value for this token
  float max_val = std::numeric_limits<float>::min();
  float min_val = std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[i]);
    max_val = std::max(max_val, val);
    min_val = std::min(min_val, val);
  }

  // Reduce the max and min values across the block
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  max_val = BlockReduce(reduceStorage).Reduce(max_val, cub::Max{}, blockDim.x);
  __syncthreads();  // Make sure min doesn't mess with max shared memory
  min_val = BlockReduce(reduceStorage).Reduce(min_val, cub::Min{}, blockDim.x);

  __shared__ scale_type scale_sh;
  __shared__ azp_type azp_sh;

  // Compute the scale and zero point and store them, only on the first thread
  if (threadIdx.x == 0) {
    float const scale_val = (max_val - min_val) / 255.0f;
    // Use rounding to even (same as torch.round)
    auto const azp_float = std::nearbyint(-128.0f - min_val / scale_val);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Store the scale and azp into shared and global
    scale[token_idx] = scale_sh = scale_val;
    azp[token_idx] = azp_sh = azp_val;
  }

  // Wait for the scale and azp to be computed
  __syncthreads();

  float const scale_val = scale_sh;
  azp_type const azp_val = azp_sh;

  // Quantize the values
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[i]);
    auto const quant_val =
        int32_to_int8(float_to_int32_rn(val / scale_val) + azp_val);
    out[i] = quant_val;
  }
}

}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              torch::Tensor const& input,  // [..., hidden_size]
                              torch::Tensor const& scale,
                              c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp || azp->numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
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
    torch::Tensor& scales, c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
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
