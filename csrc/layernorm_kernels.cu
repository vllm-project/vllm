#include "type_convert.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
rms_norm_kernel(scalar_t* __restrict__ out,           // [..., hidden_size]
                const scalar_t* __restrict__ input,   // [..., hidden_size]
                const scalar_t* __restrict__ weight,  // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size, const int vec_hidden_size) {
  __shared__ float s_variance;
  float v8_variance_sum = 0.0f;

  const int64_t tx = threadIdx.x;
  const int64_t bx = blockIdx.x;
  const int64_t num_threads = blockDim.x;

  auto* __restrict__ out_v = reinterpret_cast<_f16Vec<scalar_t, width>*>(out);
  auto* __restrict__ input_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(
          input + bx * static_cast<int64_t>(hidden_size));
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  // Compute variance. Be careful, hidden_size should multiple of 4.
  for (int idx = tx; idx < vec_hidden_size; idx += num_threads) {
    _f16Vec<scalar_t, width> temp = input_v[idx];
    v8_variance_sum += temp.sum_squares();
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;

  float variance =
      BlockReduce(reduceStore).Reduce(v8_variance_sum, cub::Sum{}, num_threads);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  variance = s_variance;

  for (int idx = tx; idx < vec_hidden_size; idx += num_threads) {
    _f16Vec<scalar_t, width> temp = input_v[idx];
    temp *= variance;
    temp *= weight_v[idx];
    out_v[bx * static_cast<int64_t>(vec_hidden_size) + idx] = temp;
  }
}

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
rms_norm_kernel(scalar_t* __restrict__ out,           // [..., hidden_size]
                const scalar_t* __restrict__ input,   // [..., hidden_size]
                const scalar_t* __restrict__ weight,  // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size, const int vec_hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x =
        (float)input[blockIdx.x * static_cast<int64_t>(hidden_size) + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x =
        (float)input[blockIdx.x * static_cast<int64_t>(hidden_size) + idx];
    out[blockIdx.x * static_cast<int64_t>(hidden_size) + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */

template <>
struct Vec<c10::Float8_e4m3fnuz, 8> {
  using Type = uint2;
};

template <>
struct Vec<c10::Half, 8> {
  using Type = uint4;
};

template <>
struct Vec<c10::BFloat16, 8> {
  using Type = bf16_8_t;
};

}  // namespace vllm

#define LAUNCH_RMS_NORM(width)                                               \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] { \
    vllm::rms_norm_kernel<scalar_t, width><<<grid, block, 0, stream>>>(      \
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                \
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size,       \
        vec_hidden_size);                                                    \
  });

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int vec_size = 16 / input.element_size();
  int vec_hidden_size = hidden_size / vec_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(vec_hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#ifdef __HIP__MI300_MI250__
  if (vec_size % 8 == 0) {
    LAUNCH_RMS_NORM(8);
  } else {
    LAUNCH_RMS_NORM(0);
  }
#else
    LAUNCH_RMS_NORM(0);
#endif
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                  \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                       \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                         residual.data_ptr<scalar_t>(),        \
                                         weight.data_ptr<scalar_t>(), epsilon, \
                                         num_tokens, hidden_size);             \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
